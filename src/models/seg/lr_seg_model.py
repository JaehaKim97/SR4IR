import os
import os.path as osp
import torch
import warnings
warnings.filterwarnings('ignore')
# to ignore below warning message from COCO evaluator:
# UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or
# sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).

from archs import build_network
from losses import build_loss
from math import ceil
from torch.nn.functional import pad, interpolate
from utils.common import save_on_master, reduce_across_processes, quantize, calculate_psnr_batch, calculate_lpips_batch, visualize_image_from_batch
from utils.seg import MetricLogger, SmoothedValue, calculate_mat, compute_iou

from .base_model import BaseModel


def make_model(opt):
    return LRSegmentationModel(opt)


class LRSegmentationModel(BaseModel):
    """Semantic Segmentation model using LR images."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # define network segmentation
        self.net_seg = build_network(opt['network_seg'], self.text_logger, task=self.task, tag='net_seg')
        self.load_network(self.net_seg, name='network_seg', tag='net_seg')
        self.net_seg = self.model_to_device(self.net_seg)
        self.print_network(self.net_seg, tag='net_seg')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_seg.train()
        elif mode == 'eval':
            self.net_seg.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode('train')
        train_opt = self.opt['train']
        
        # define losses
        if train_opt.get('auxce_opt'):
            self.cri_auxce = build_loss(train_opt['auxce_opt'], self.text_logger).to(self.device)
        else:
            raise NotImplementedError("AUXCELoss is required to train Segmentation Model")

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
        
        # optimizer seg
        optim_type = train_opt['optim_seg'].pop('type')
        net_seg_parameters = [
            {"params": [p for p in self.get_bare_model(self.net_seg).backbone.parameters() if p.requires_grad]},
            {"params": [p for p in self.get_bare_model(self.net_seg).classifier.parameters() if p.requires_grad]},
        ]
        if self.opt['network_seg'].get('aux_loss', True):
            params = [p for p in self.get_bare_model(self.net_seg).aux_classifier.parameters() if p.requires_grad]
            net_seg_parameters.append({"params": params, "lr": train_opt['optim_seg']['lr'] * 10})
        self.optimizer_seg = self.get_optimizer(optim_type, net_seg_parameters, **train_opt['optim_seg'])
        self.optimizers.append(self.optimizer_seg)
    
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode('train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_seg", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr, target) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr, target = img_hr.to(self.device), target.to(self.device)
            
            # make on-the-fly LR image
            img_lr = self.net_up(quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic')))
            
            # semantic segmentation
            pred_lr = self.net_seg(self.normalize(img_lr))
            
            # loss calculation and backwarding
            self.optimizer_seg.zero_grad()

            l_total = 0
            current_iter = iter + len(data_loader_train)*(epoch-1)
            if hasattr(self, 'cri_auxce'):
                l_auxce = self.cri_auxce(pred_lr, target)
                metric_logger.meters["l_auxce"].update(l_auxce.item())
                self.tb_logger.add_scalar('losses/l_auxce', l_auxce.item(), current_iter)
                l_total += l_auxce
            
            l_total.backward()
            self.optimizer_seg.step()
            
            # logging training state
            metric_logger.update(l_auxce=l_auxce.item(), lr_seg=self.optimizer_seg.param_groups[0]["lr"])

            # update learning rate
            self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0, num_classes=21):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode('eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)
        
        num_processed_samples = 0
        for (img_hr, target), filename in metric_logger.log_every(data_loader_test, 500, self.text_logger, header, return_filename=True):
            img_hr, target = img_hr.to(self.device), target.to(self.device)
            
            # make on-the-fly LR image
            h, w = img_hr.shape[2:]
            ph, pw = ceil(h/self.scale) * self.scale - h, ceil(w/self.scale) * self.scale - w
            img_hr_padded = pad(img_hr, pad=(0, pw, 0, ph), mode='replicate')
            
            # make on-the-fly LR image
            img_lr = self.net_up(quantize(interpolate(img_hr_padded, scale_factor=(1/self.scale), mode='bicubic')))[:,:,:h,:w]
            
            # semantic segmentation
            pred_lr = self.net_seg(self.normalize(img_lr))
            
            # visualizing tool
            if self.opt['test'].get('visualize', False): # and (num_processed_samples < 20):
                seg_map = self.convert2color(pred_lr['out'].argmax(1))
                visualize_image_from_batch(img_lr, osp.join(self.exp_dir, 'visualize'), [filename])
                visualize_image_from_batch(seg_map.unsqueeze(0), osp.join(self.exp_dir, 'visualize'), [filename.replace('.jpg', '_seg.jpg')])
            
            # evaluation on validation batch
            confmat += calculate_mat(target.flatten(), pred_lr['out'].argmax(1).flatten(), n=num_classes)
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_lr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_lr), img_hr, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            num_processed_samples += img_hr.shape[0]

        num_processed_samples = reduce_across_processes(num_processed_samples)
        confmat = reduce_across_processes(confmat)
        if (
            hasattr(data_loader_test.dataset, "__len__")
            and len(data_loader_test.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            warnings.warn(
                f"It looks like the dataset has {len(data_loader_test.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )
        
        # metirc logger
        iu = compute_iou(confmat)
        metric_logger.synchronize_between_processes()
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        metric_summary = self.add_metric(metric_summary, 'mean IoU-LR', iu.mean().item() * 100, epoch)
        self.text_logger.write(metric_summary)
        return
        
    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_seg": self.get_bare_model(self.net_seg).state_dict(),
                      "optimizer_seg": self.optimizer_seg.state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_seg).state_dict(), osp.join(self.exp_dir, 'models', "net_seg_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_seg).state_dict(), osp.join(self.exp_dir, 'models', "net_seg_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
