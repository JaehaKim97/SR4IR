import os
import os.path as osp
import torch
import warnings

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.cls import MetricLogger, SmoothedValue, calculate_accuracy
from utils.common import save_on_master, quantize, reduce_across_processes, calculate_psnr_batch, calculate_lpips_batch, visualize_image_from_batch

from .base_model import BaseModel


def make_model(opt):
    return LRClassificationModel(opt)


class LRClassificationModel(BaseModel):
    """Super-Resolution model for Image Classification."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # define network cls
        self.net_cls = build_network(opt['network_cls'], self.text_logger, task=self.task, tag='net_cls')
        self.load_network(self.net_cls, name='network_cls', tag='net_cls')
        self.net_cls = self.model_to_device(self.net_cls, is_trainable=True)
        self.print_network(self.net_cls, tag='net_cls')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_cls.train()
        elif mode == 'eval':
            self.net_cls.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('ce_opt'):
            self.cri_ce = build_loss(train_opt['ce_opt'], self.text_logger).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train))
        
        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer cls
        optim_type = train_opt['optim_cls'].pop('type')        
        self.optimizer_cls = self.get_optimizer(optim_type, self.net_cls.parameters(), **train_opt['optim_cls'])
        self.optimizers.append(self.optimizer_cls)
        
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_cls", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr, label) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr, label = img_hr.to(self.device), label.to(self.device)
            
            # make on-the-fly LR image
            img_lr = self.net_up(quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic')))
            
            # image classification
            pred_lr = self.net_cls(self.normalize(img_lr))
            
            # loss calculation and backwarding
            self.optimizer_cls.zero_grad()
            
            l_total = 0
            current_iter = iter + len(data_loader_train)*(epoch-1)
            if hasattr(self, 'cri_ce'):
                l_ce = self.cri_ce(pred_lr, label)
                metric_logger.meters["l_ce"].update(l_ce.item())
                self.tb_logger.add_scalar('losses/l_ce', l_ce.item(), current_iter)
                l_total += l_ce
            
            l_total.backward()
            self.optimizer_cls.step()

            # logging training state
            batch_size = img_lr.shape[0]
            acc1_lr, _ = calculate_accuracy(pred_lr, label, topk=(1, 5))
            metric_logger.meters["acc1_lr"].update(acc1_lr.item(), n=batch_size)
            metric_logger.update(lr_cls=round(self.optimizers[0].param_groups[0]["lr"], 8))
            
            # update learning rate
            self.update_learning_rate()
        return

    @torch.inference_mode()            
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
            
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test: "

        num_processed_samples = 0
        for (img_hr, label), filename in metric_logger.log_every(data_loader_test, 2000, self.text_logger, header, return_filename=True):
            img_hr, label = img_hr.to(self.device), label.to(self.device)

            # make on-the-fly LR image
            img_lr = self.net_up(quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic')))
            
            # image classification
            pred_lr = self.net_cls(self.normalize(img_lr))
            
            # visualizing tool
            if self.opt['test'].get('visualize', False):
                visualize_image_from_batch(img_lr, osp.join(self.exp_dir, 'visualize'), filename)

            # evaluation on validation batch
            batch_size = img_lr.shape[0]
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_lr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_lr), img_hr, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            acc1_lr, _ = calculate_accuracy(pred_lr, label, topk=(1, 5))
            metric_logger.meters["acc1_lr"].update(acc1_lr.item(), n=batch_size)
            num_processed_samples += batch_size            
    
        # gather the stats from all processes
        num_processed_samples = reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader_test.dataset, "__len__")
            and len(data_loader_test.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader_test.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        # metirc logger
        metric_logger.synchronize_between_processes()
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        metric_summary = self.add_metric(metric_summary, 'ACC_LR@1', metric_logger.acc1_lr.global_avg, epoch)
        self.text_logger.write(metric_summary)
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_cls": self.get_bare_model(self.net_cls).state_dict(),
                      "optimizer_cls": self.optimizer_cls.state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_cls).state_dict(), osp.join(self.exp_dir, 'models', "net_cls_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_cls).state_dict(), osp.join(self.exp_dir, 'models', "net_cls_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
