import cv2
import lpips
import numpy as np
import os
import os.path as osp
import sys
import torch
import torchvision

from collections import OrderedDict
from copy import deepcopy
from math import ceil
from ptflops import get_model_complexity_info
from torch.nn.functional import pad
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import lr_scheduler
from utils.common import mkdir_and_rename, TensorboardLogger, TextLogger, ManualNormalize, MultiStepRestartLR, CosineAnnealingRestartLR


def make_model(opt):
    return BaseModel(opt)


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.task = opt['task']
        self.scale = opt.get('scale', 1)
        self.schedulers = []
        self.optimizers = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # prepare output directories and loggers
        self.is_train = opt.get('train', False)
        self.exp_dir = osp.join('experiments', self.task, opt['name'])
        filename = 'train_log.txt' if self.is_train else 'test_log.txt'
        
        if self.is_train:
            mkdir_and_rename(osp.join(self.exp_dir))
            mkdir_and_rename(osp.join('tb_loggers', self.task, opt['name']))
            self.tb_logger = TensorboardLogger(log_dir=osp.join('tb_loggers', self.task, opt['name']))
        else:
            # ensure that output directory exists in evaluation only mode
            os.makedirs(self.exp_dir, exist_ok=True)
        self.text_logger = TextLogger(save=self.exp_dir, filename=filename)
        
        # save environment settings
        if opt['dist']:
            self.dist = True
            self.text_logger.write('torchrun --nproc_per_node {} '.format(opt['world_size']) + ' '.join(sys.argv), print_log=False)
        else:
            self.dist = False
            self.text_logger.write('python ' + ' '.join(sys.argv), print_log=False)
        self.text_logger.write('Random seed : {}'.format(opt['manual_seed']))
            
        # transform functions
        self.normalize = ManualNormalize()
        
        # for lpips calculation
        if opt['test'].get('calculate_lpips', False):
            self.net_lpips = lpips.LPIPS(net='vgg').to(self.device)

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def model_to_device(self, net, is_trainable=True):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if not is_trainable:
            # if the net does not requires grad, then not wrap it with DDP
            return net
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            broadcast_buffers = self.opt.get('broadcast_buffers', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported.')
        return optimizer

    def setup_schedulers(self, data_loader_len, name=None, optimizer=None):
        """Set up schedulers."""
        if (name is not None):
            scheduler_opt = self.opt['train'][f'scheduler_{name}'].copy()
        else:
            scheduler_opt = self.opt['train']['scheduler'].copy()
        scheduler_type = scheduler_opt.pop('type').lower()
        
        # compensate updating at each iteration
        if scheduler_type  == 'cosineannealinglr':
            _scheduler = lr_scheduler.CosineAnnealingLR
            scheduler_opt['T_max'] *= data_loader_len            
        elif scheduler_type == 'cosineannealingrestartlr':
            _scheduler = CosineAnnealingRestartLR
            for idx in range(len(scheduler_opt['periods'])):
                scheduler_opt['periods'][idx] *= data_loader_len
        elif scheduler_type  == 'steplr':
            _scheduler = lr_scheduler.StepLR
            scheduler_opt['step_size'] *= data_loader_len
        elif scheduler_type == 'multisteplr':
            _scheduler = lr_scheduler.MultiStepLR
            for idx in range(len(scheduler_opt['milestones'])):
                scheduler_opt['milestones'][idx] *= data_loader_len
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
        
        if (name is not None) and (optimizer is not None):
            self.schedulers.append(_scheduler(optimizer, **scheduler_opt))
        else:
            # use same scheduler for all optimizers
            for optimizer in self.optimizers:
                self.schedulers.append(_scheduler(optimizer, **scheduler_opt))

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def print_network(self, net, tag='None'):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        
        if not hasattr(self, 'arch_logger'):
            self.arch_logger = TextLogger(save=self.exp_dir, filename="arch.txt")

        self.arch_logger.write(f'Network [{tag}]: {net_cls_str}, with parameters: {net_params:,d}', print_log=False)
        self.arch_logger.write(net_str, print_log=False)
        self.arch_logger.write('\n\n', print_log=False)
        
    def add_metric(self, metric_summary, name, value, epoch):
        """Add metric on loggers.

        Args
            metric_summary (str):
            name (str)
            value (float)
            epoch (int)
        """
        metric_summary += f" {name} {value:.3f}"
        if self.is_train:
            self.tb_logger.add_scalar('metrics/{}'.format(name.lower()), value, epoch)
        return metric_summary

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            self.text_logger.write('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                self.text_logger.write(f'  {v}')
            self.text_logger.write('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                self.text_logger.write(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    self.text_logger.write(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, name, tag='net_sr'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        load_path = self.opt['path'].get(name, None)
        strict = self.opt['path'].get('strict_load', True)
        param_key = self.opt['path'].get(f'{name}_key', None)
        
        if self.opt.get('test_only'):
            if tag == 'net_sr':
                load_path = osp.join(self.exp_dir, 'models/net_sr_latest.pth')
            if tag == 'net_det':
                load_path = osp.join(self.exp_dir, 'models/net_det_latest.pth')
        
        if load_path is None:
            return
        
        if os.path.exists(load_path):
            net = self.get_bare_model(net)
            load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
            if param_key is not None:
                if param_key not in load_net and 'params' in load_net:
                    param_key = 'params'
                load_net = load_net[param_key]
            self.text_logger.write(f'--> Load {tag} model from {load_path}, with param key: [{param_key}].')
            # remove unnecessary 'module.'
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            self._print_different_keys_loading(net, load_net, strict)
            net.load_state_dict(load_net, strict=strict)
        else:
            try:
                load_net = torchvision.models.get_model(self.opt[name]['name'],
                                                        weights=load_path,
                                                        num_classes=self.opt[name]['num_classes']).state_dict()
                self._print_different_keys_loading(net, load_net, strict)
                net.load_state_dict(load_net, strict=strict)
                self.text_logger.write(f'--> Load {tag} model from {load_path}.')
            except:
                raise NotImplementedError(f'{load_path} is not valid model path!')

    def resume_training(self, resume_path):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_state = torch.load(resume_path, map_location="cpu")
        
        resume_schedulers = resume_state['schedulers']
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
        
        if hasattr(self, 'optimizer_sr') and 'optimizer_sr' in resume_state.keys():
            self.optimizer_sr.load_state_dict(resume_state['optimizer_sr'])
        if hasattr(self, 'optimizer_cls') and 'optimizer_cls' in resume_state.keys():
            self.optimizer_cls.load_state_dict(resume_state['optimizer_cls'])
          
        if hasattr(self, 'net_sr') and 'net_sr' in resume_state.keys():
            self.get_bare_model(self.net_sr).load_state_dict(resume_state['net_sr'], strict=True)
        if hasattr(self, 'net_cls') and 'net_cls' in resume_state.keys():
            self.get_bare_model(self.net_cls).load_state_dict(resume_state['net_cls'], strict=True)
            
        self.text_logger.write(f'Resume training from {resume_path}')
        
        return resume_state['epoch']

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def list_to_batch(self, img_list):
        # img_list : List of images (C H W)
        max_h, max_w = 0, 0
        for img in img_list:
            max_h, max_w = max(img.size(1), max_h), max(img.size(2), max_w)
        max_h, max_w = ceil(max_h/self.scale) * self.scale, ceil(max_w/self.scale) * self.scale
        
        img_batch = torch.Tensor().to(self.device)
        for img in img_list:
            ph, pw = max_h - img.size(1), max_w - img.size(2)
            if ph < img.size(1) and pw < img.size(2):
                mode = 'reflect'
            else:
                mode = 'replicate'
            img_padded = pad(img.unsqueeze(0), pad=(0, pw, 0, ph), mode=mode)  # 4D Tensor
            img_batch = torch.cat((img_batch, img_padded), dim=0)
            
        return img_batch

    def batch_to_list(self, img_batch, img_list):
        new_img_list = list()
        for idx, img in enumerate(img_list):
            new_img_list.append(img_batch[idx][:, :img.size(1), :img.size(2)])
            
        return new_img_list
    
    
    def draw_box(self, img, target, score_threshold=0.9, imwrite=True):
        target = target.copy()
        img_tmp = np.array(img.permute(1,2,0).detach().cpu()*255).copy()
        if 'scores' in target:
            mask = (target['scores'] > score_threshold)
            target['boxes'] = target['boxes']*mask.unsqueeze(1)
            target['labels'] = target['labels']*mask
        if len(target['boxes']) > 0:
            for idx in range(len(target['boxes'])):
                x1, y1, x2, y2 = target['boxes'][idx]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)                
                img_label = int(target['labels'][idx])
                obj_label = self.convert2label(img_label-1)
                
                if (x1 < 0 or x2 > img.size(2) or y1 < 0 or y2 > img.size(1)) or (x1 < 10 and y1 < 10 and obj_label == 'tvmonitor'):
                    # box location exceed image, remove for better visualizaton
                    continue
                img_tmp = cv2.rectangle(img_tmp, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
                cv2.putText(img_tmp, obj_label, (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,108,34), 2, 2)
                
        if imwrite:
            cv2.imwrite('img.png', cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR))
        
        return img_tmp
    
    def visualize(self, img, target, filename, imwrite=True, score_threshold=0.9):
        img = self.draw_box(img, target, score_threshold, imwrite=False)
        os.makedirs(osp.join(self.exp_dir, 'visualize'), exist_ok=True)
        return cv2.imwrite(osp.join(self.exp_dir, 'visualize', filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def convert2label(self, label):
        table = ['aeroplane', 'bicycle', 'bird', 'boat' ,'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        return table[label]
        
    @torch.inference_mode()
    def calculate_cost(self):
        
        inp_shape = (3, 480 // self.scale, 480 // self.scale)
        macs_sr, params_sr = get_model_complexity_info(self.net_sr, inp_shape, verbose=False, print_per_layer_stat=False)
        print('sr network :')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_sr))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_sr))
        
        
        inp_shape = (3, 480, 480)
        macs_det, params_det = get_model_complexity_info(self.net_det, inp_shape, verbose=False, print_per_layer_stat=False)
        print('det network :')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_det))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_det))
        return

    def imwrite(self, img, name='img.png'):
        import cv2
        return cv2.imwrite(name, img.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]*255)
    