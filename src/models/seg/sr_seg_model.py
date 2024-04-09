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
from utils.common import save_on_master, quantize, calculate_psnr_batch, reduce_across_processes, visualize_image_from_batch, calculate_lpips_batch, visualize_image
from utils.seg import SmoothedValue, calculate_mat, compute_iou, MetricLogger

from .base_model import BaseModel


def make_model(opt):
    return SRSegmentationModel(opt)


class SRSegmentationModel(BaseModel):
    """Semantic Segmentation model using Super-Resolution images."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network sr
        self.sr_is_trainable = self.is_train and opt['train'].get('sr_is_trainable', True)
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=self.sr_is_trainable)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network segmentation for hr (not trainable)
        self.seg_is_trainable = self.is_train and opt['train'].get('seg_is_trainable', True)
        self.net_seg = build_network(opt['network_seg'], self.text_logger, task=self.task, tag='net_seg')
        self.load_network(self.net_seg, name='network_seg', tag='net_seg')
        self.net_seg = self.model_to_device(self.net_seg, is_trainable=self.seg_is_trainable)
        self.print_network(self.net_seg, tag='net_seg')
        
    def set_mode(self, mode):
        if mode == 'train':
            if self.sr_is_trainable:
                self.net_sr.train()
            else:
                self.net_sr.eval()
            if self.seg_is_trainable:
                self.net_seg.train()
            else:
                self.net_seg.eval()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_seg.eval()    
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)

        if train_opt.get('auxce_opt'):
            self.cri_auxce = build_loss(train_opt['auxce_opt'], self.text_logger).to(self.device)
            
        # set up optimizers and schedulers
        self.setup_optimizers()
        if self.sr_is_trainable:
            self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        if self.seg_is_trainable:
            self.setup_schedulers(len(data_loader_train), name='seg', optimizer=self.optimizer_seg)
        
        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        if self.sr_is_trainable:
            self.text_logger.write('NOTICE: net_sr is trainable')
            optim_type = train_opt['optim_sr'].pop('type')
            self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
            self.optimizers.append(self.optimizer_sr)
        else:
            self.text_logger.write('NOTICE: net_sr is NOT trainable')
        
        # optimizer seg
        if self.seg_is_trainable:
            self.text_logger.write('NOTICE: net_seg is trainable')
            optim_type = train_opt['optim_seg'].pop('type')
            net_seg_parameters = [{"params": [p for p in self.get_bare_model(self.net_seg).backbone.parameters() if p.requires_grad]},
                                {"params": [p for p in self.get_bare_model(self.net_seg).classifier.parameters() if p.requires_grad]},]
            if self.opt['network_seg'].get('aux_loss', True):
                params = [p for p in self.get_bare_model(self.net_seg).aux_classifier.parameters() if p.requires_grad]
                net_seg_parameters.append({"params": params, "lr": train_opt['optim_seg']['lr'] * 10})
            self.optimizer_seg = self.get_optimizer(optim_type, net_seg_parameters, **train_opt['optim_seg'])
            self.optimizers.append(self.optimizer_seg)
        else:
            self.text_logger.write('NOTICE: net_seg is NOT trainable')
        
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        if self.sr_is_trainable:
            metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        if self.seg_is_trainable:
            metric_logger.add_meter("lr_seg", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr, target) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr, target = img_hr.to(self.device), target.to(self.device)
            
            # make on-the-fly LR image
            img_lr = quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic'))
            
            # super resolution
            img_sr = self.net_sr(img_lr)
            
            # loss calculation and backwarding
            if self.sr_is_trainable:
                self.optimizer_sr.zero_grad()
            if self.seg_is_trainable:
                self.optimizer_seg.zero_grad()
            
            l_total = 0
            current_iter = iter + len(data_loader_train)*(epoch-1)
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr, img_hr)
                metric_logger.meters["l_pix"].update(l_pix.item()) 
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total += l_pix
            if hasattr(self, 'cri_auxce'):
                # semantic segmentation
                pred_sr = self.net_seg(self.normalize(img_sr))
                l_auxce = self.cri_auxce(pred_sr, target)
                metric_logger.meters["l_auxce"].update(l_auxce.item()) 
                self.tb_logger.add_scalar('losses/l_auxce', l_auxce.item(), current_iter)
                l_total += l_auxce
            
            l_total.backward()
            if self.sr_is_trainable:
                self.optimizer_sr.step()
            if self.seg_is_trainable:
                self.optimizer_seg.step()
            
            # logging training state
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)                
            if self.sr_is_trainable:
                metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
            if self.seg_is_trainable:
                metric_logger.update(lr_seg=round(self.optimizer_seg.param_groups[0]["lr"], 8))
            
            # update learning rate
            self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0, num_classes=21):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
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
            img_lr = quantize(interpolate(img_hr_padded, scale_factor=(1/self.scale), mode='bicubic'))
            
            # perform SR
            img_sr = self.net_sr(img_lr)[:,:,:h,:w]
            
            # semantic segmentation
            pred_sr = self.net_seg(self.normalize(img_sr))
            
            # visualizing tool
            if self.opt['test'].get('visualize', False): # and (num_processed_samples < 20):
                seg_map = self.convert2color(pred_sr['out'].argmax(1))
                visualize_image_from_batch(img_sr, osp.join(self.exp_dir, 'visualize'), [filename])
                visualize_image_from_batch(seg_map.unsqueeze(0), osp.join(self.exp_dir, 'visualize'), [filename.replace('.jpg', '_seg.jpg')])
            
            # evaluation on validation batch
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr), img_hr, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            confmat += calculate_mat(target.flatten(), pred_sr["out"].argmax(1).flatten(), n=num_classes)
            num_processed_samples += img_hr.shape[0]

        num_processed_samples = reduce_across_processes(num_processed_samples)
        confmat = reduce_across_processes(confmat)
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
        iu_sr = compute_iou(confmat)
        metric_logger.synchronize_between_processes()
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        metric_summary = self.add_metric(metric_summary, 'mIoU-SR', iu_sr.mean().item() * 100, epoch)
        
        self.text_logger.write(metric_summary)
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_seg": self.get_bare_model(self.net_seg).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_seg).state_dict(), osp.join(self.exp_dir, 'models', "net_seg_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_seg).state_dict(), osp.join(self.exp_dir, 'models', "net_seg_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
