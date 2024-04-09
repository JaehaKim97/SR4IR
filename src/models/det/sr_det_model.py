import os
import os.path as osp
import torch

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.common import save_on_master, quantize, calculate_psnr_batch, calculate_lpips_batch, visualize_image
from utils.det import MetricLogger, SmoothedValue, get_coco_api_from_dataset, _get_iou_types, CocoEvaluator

from .base_model import BaseModel


def make_model(opt):
    return SRDetectionModel(opt)


class SRDetectionModel(BaseModel):
    """Base Super-Resolution model for Object Detection."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network sr
        self.sr_is_trainable = self.is_train and opt['train'].get('sr_is_trainable', True)
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=self.sr_is_trainable)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network detction
        self.det_is_trainable = self.is_train and opt['train'].get('det_is_trainable', True)
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        self.net_det = self.model_to_device(self.net_det, is_trainable=self.det_is_trainable)
        self.print_network(self.net_det, tag='net_det')
        
    def set_mode(self, mode):
        if mode == 'train':
            if self.sr_is_trainable:
                self.net_sr.train()
            else:
                self.net_sr.eval()
            if self.det_is_trainable:
                self.net_det.train()
            else:
                self.net_det.eval()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_det.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('det_opt'):
            # Note: detection losses are defined in the model forwarding process
            self.cri_det = build_loss(train_opt['det_opt'], self.text_logger).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        if self.sr_is_trainable:
            self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        if self.det_is_trainable:
            self.setup_schedulers(len(data_loader_train), name='det', optimizer=self.optimizer_det)
        
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
        
        # optimizer det
        if self.det_is_trainable:
            self.text_logger.write('NOTICE: net_det is trainable')
            optim_type = train_opt['optim_det'].pop('type')
            net_det_parameters = [p for p in self.net_det.parameters() if p.requires_grad]
            self.optimizer_det = self.get_optimizer(optim_type, net_det_parameters, **train_opt['optim_det'])
            self.optimizers.append(self.optimizer_det)
        else:
            self.text_logger.write('NOTICE: net_det is NOT trainable')
            
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        if self.sr_is_trainable:
            metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        if self.det_is_trainable:
            metric_logger.add_meter("lr_det", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        # NOTE: without warmup, training explodes!!
        lr_scheduler_s = None
        lr_scheduler_d = None
        if epoch == 1:
            warmup_factor = 1.0 / len(data_loader_train)
            warmup_iters = len(data_loader_train)
            if self.sr_is_trainable:
                lr_scheduler_s = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_sr, start_factor=warmup_factor, total_iters=warmup_iters)
            if self.det_is_trainable:
                lr_scheduler_d = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_det, start_factor=warmup_factor, total_iters=warmup_iters)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr_list, target_list) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # super resolution
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
                        
            # loss calculation and backwarding
            if self.sr_is_trainable:
                self.optimizer_sr.zero_grad()
            if self.det_is_trainable:
                self.optimizer_det.zero_grad()
            
            l_total = 0
            current_iter = iter + len(data_loader_train)*(epoch-1)
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr_batch, img_hr_batch)
                metric_logger.meters["l_pix"].update(l_pix.item()) 
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total += l_pix
            if hasattr(self, 'cri_det'):
                # object detection
                _, loss_dict_sr = self.net_det(img_sr_list, target_list)
                l_det = self.cri_det(loss_dict_sr)
                metric_logger.meters["l_det"].update(l_det.item())
                self.tb_logger.add_scalar('losses/l_det', l_det.item(), current_iter)
                l_total += l_det
            
            l_total.backward()
            if self.sr_is_trainable:
                self.optimizer_sr.step()
            if self.det_is_trainable:
                self.optimizer_det.step()
            
            # logging training state
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.sr_is_trainable:
                metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
            if self.det_is_trainable:
                metric_logger.update(lr_det=round(self.optimizer_det.param_groups[0]["lr"], 8))
            
            # update learning rate
            if epoch == 1:
                if self.sr_is_trainable:
                    lr_scheduler_s.step()
                if self.det_is_trainable:
                    lr_scheduler_d.step()
            else:
                self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        iou_types = _get_iou_types(self.net_det)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        num_processed_samples = 0
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header, return_filename=True):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # super resolution
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            
            # object detection
            if torch.cuda.is_available(): torch.cuda.synchronize()
            outputs_sr, _ = self.net_det(img_sr_list)
            outputs_sr = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs_sr]

            # visualizing tool
            if self.opt['test'].get('visualize', False): # and num_processed_samples < 20:
                self.visualize(img_sr_list[0], outputs_sr[0], filename)

            # evaluation on validation batch
            batch_size = len(img_sr_list)
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr_batch), img_hr_batch, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            res = {target["image_id"]: output for target, output in zip(target_list, outputs_sr)}
            coco_evaluator.update(res)
            num_processed_samples += batch_size
    
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()
        
        # metirc logger
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        self.text_logger.write(metric_summary)

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize(self.text_logger, tag='SR')
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_det": self.get_bare_model(self.net_det).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
