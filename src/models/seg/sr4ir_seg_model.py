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
from utils.common import save_on_master, quantize, calculate_psnr_batch, reduce_across_processes, visualize_image_from_batch, calculate_lpips_batch
from utils.seg import SmoothedValue, calculate_mat, compute_iou, MetricLogger

from .base_model import BaseModel


def make_model(opt):
    return SR4IRSegmentationModel(opt)


class SR4IRSegmentationModel(BaseModel):
    """Semantic Segmentation model using Super-Resolution for Image Recognition."""

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
            self.net_sr.train()
            self.net_seg.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_seg.eval()    
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']
        
        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
        
        if train_opt.get('tdp_opt'):
            # task driven perceptual loss
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
            
        # phase 2
        if train_opt.get('auxce_sr_opt'):
            self.cri_auxce_sr = build_loss(train_opt['auxce_sr_opt'], self.text_logger).to(self.device)
        
        if train_opt.get('auxce_hr_opt'):
            self.cri_auxce_hr = build_loss(train_opt['auxce_hr_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('auxce_cqmix_opt'):
            self.cri_auxce_cqmix = build_loss(train_opt['auxce_cqmix_opt'], self.text_logger).to(self.device)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        self.setup_schedulers(len(data_loader_train), name='seg', optimizer=self.optimizer_seg)
        
        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
        # warmup epoch
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # optimizer seg
        self.text_logger.write('NOTICE: net_seg is trainable')
        optim_type = train_opt['optim_seg'].pop('type')
        net_seg_parameters = [{"params": [p for p in self.get_bare_model(self.net_seg).backbone.parameters() if p.requires_grad]},
                            {"params": [p for p in self.get_bare_model(self.net_seg).classifier.parameters() if p.requires_grad]},]
        if self.opt['network_seg'].get('aux_loss', True):
            params = [p for p in self.get_bare_model(self.net_seg).aux_classifier.parameters() if p.requires_grad]
            net_seg_parameters.append({"params": params, "lr": train_opt['optim_seg']['lr'] * 10})
        self.optimizer_seg = self.get_optimizer(optim_type, net_seg_parameters, **train_opt['optim_seg'])
        self.optimizers.append(self.optimizer_seg)
        
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr_seg", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")

        torch.cuda.empty_cache()
        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr, target) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr, target = img_hr.to(self.device), target.to(self.device)
            current_iter = iter + len(data_loader_train)*(epoch-1)
            
            # make on-the-fly LR image
            img_lr = quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic'))
            
            # phase 1;
            # update net_sr, freeze net_seg
            img_sr = self.net_sr(img_lr)
            for p in self.net_seg.parameters(): p.requires_grad = False
            self.optimizer_sr.zero_grad()
            l_total_sr = 0
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr, img_hr)
                metric_logger.meters["l_pix"].update(l_pix.item()) 
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total_sr += l_pix
            if epoch > self.warmup_epoch:
                if hasattr(self, 'cri_tdp'):
                    self.net_seg.eval()
                    _, feat_sr = self.net_seg(self.normalize(img_sr), return_feats=True)
                    _, feat_hr = self.net_seg(self.normalize(img_hr), return_feats=True)
                    self.net_seg.train()
                    
                    l_tdp = self.cri_tdp(feat_sr, feat_hr)
                    metric_logger.meters["l_tdp"].update(l_tdp.item()) 
                    self.tb_logger.add_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                    l_total_sr += l_tdp
            l_total_sr.backward()
            self.optimizer_sr.step()
            
            # phase 2;
            # update net_seg, freeze net_sr
            img_sr = self.net_sr(img_lr).detach()
            for p in self.net_seg.parameters(): p.requires_grad = True
            self.optimizer_seg.zero_grad()
            l_total_seg = 0
            if hasattr(self, 'cri_auxce_sr'):
                pred_sr = self.net_seg(self.normalize(img_sr))
                l_auxce_sr = self.cri_auxce_sr(pred_sr, target)
                metric_logger.meters["l_auxce_sr"].update(l_auxce_sr.item())
                self.tb_logger.add_scalar('losses/l_auxce_sr', l_auxce_sr.item(), current_iter)
                l_total_seg += l_auxce_sr
            if hasattr(self, 'cri_auxce_hr'):
                pred_hr = self.net_seg(self.normalize(img_hr))
                l_auxce_hr = self.cri_auxce_hr(pred_hr, target)
                metric_logger.meters["l_auxce_hr"].update(l_auxce_hr.item())
                self.tb_logger.add_scalar('losses/l_auxce_hr', l_auxce_hr.item(), current_iter)
                l_total_seg += l_auxce_hr
            if hasattr(self, 'cri_auxce_cqmix'):
                batch_size = img_hr.shape[0]
                mask = interpolate((torch.randn(batch_size,1,8,8)).bernoulli_(p=0.5), scale_factor=60, mode='nearest').to(self.device)
                img_cqmix = img_sr*mask + img_hr*(1-mask)
                pred_cqmix = self.net_seg(self.normalize(img_cqmix))
                l_auxce_cqmix = self.cri_auxce_cqmix(pred_cqmix, target)
                metric_logger.meters["l_auxce_cqmix"].update(l_auxce_cqmix.item())
                self.tb_logger.add_scalar('losses/l_auxce_cqmix', l_auxce_cqmix.item(), current_iter)
                l_total_seg += l_auxce_cqmix
            l_total_seg.backward()
            self.optimizer_seg.step()
            
            # logging training state
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr), img_hr)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)                
            metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
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
            
            # super resolution
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
