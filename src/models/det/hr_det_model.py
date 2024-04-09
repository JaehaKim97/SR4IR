import os
import os.path as osp
import torch

from archs import build_network
from losses import build_loss
from utils.common import save_on_master
from utils.det import MetricLogger, SmoothedValue, get_coco_api_from_dataset, _get_iou_types, CocoEvaluator

from .base_model import BaseModel


def make_model(opt):
    return HRDetectionModel(opt)


class HRDetectionModel(BaseModel):
    """Object Detection model."""

    def __init__(self, opt):
        super(HRDetectionModel, self).__init__(opt)
        
        # define network detction
        self.net_det = build_network(opt['network_det'], self.text_logger, task=self.task, tag='net_det')
        self.load_network(self.net_det, name='network_det', tag='net_det')
        self.net_det = self.model_to_device(self.net_det, is_trainable=True)
        self.print_network(self.net_det, tag='net_det')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_det.train()
        elif mode == 'eval':
            self.net_det.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode('train')
        train_opt = self.opt['train']
        
        # define losses
        if train_opt.get('det_opt'):
            # Note: detection losses are defined in the model forwarding process
            self.cri_det = build_loss(train_opt['det_opt'], self.text_logger).to(self.device)
        else:
            raise NotImplementedError("DETLoss is required to train Detection Model")

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
        
        # optimizer det
        optim_type = train_opt['optim_det'].pop('type')
        net_det_parameters = [p for p in self.net_det.parameters() if p.requires_grad]
        self.optimizer_det = self.get_optimizer(optim_type, net_det_parameters, **train_opt['optim_det'])
        self.optimizers.append(self.optimizer_det)

    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode('train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_det", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        # NOTE: without warmup, training explodes!!
        lr_scheduler = None
        if epoch == 1:
            warmup_factor = 1.0 / len(data_loader_train)
            warmup_iters = len(data_loader_train)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_det, start_factor=warmup_factor, total_iters=warmup_iters)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr_list, target_list) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]
            
            # object detection
            _, loss_dict_hr = self.net_det(img_hr_list, target_list)
            
            # loss calculation and backwarding
            self.optimizer_det.zero_grad()
            
            l_total = 0
            current_iter = iter + len(data_loader_train)*(epoch-1)
            if hasattr(self, 'cri_det'):
                l_det = self.cri_det(loss_dict_hr)
                metric_logger.meters["l_det"].update(l_det.item())
                self.tb_logger.add_scalar('losses/l_det', l_det.item(), current_iter)
                l_total += l_det
                
            l_total.backward()
            self.optimizer_det.step()
            
            # logging training state
            metric_logger.update(lr_det=round(self.optimizer_det.param_groups[0]["lr"], 8))
            
            # update learning rate
            if (lr_scheduler is not None) and (iter < warmup_iters):
                lr_scheduler.step()
            else:
                self.update_learning_rate()
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode('eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        iou_types = _get_iou_types(self.net_det)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        num_processed_samples = 0
        for (img_hr_list, target_list), filename in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header, return_filename=True):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            
            # object detection
            batch_size = len(img_hr_list)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            outputs_hr, _ = self.net_det(img_hr_list)
            outputs_hr = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs_hr]
            
            # visualizing tool
            if self.opt['test'].get('visualize', False): # and num_processed_samples < 20:
                self.visualize(img_hr_list[0], target_list[0], filename.replace('.jpg', '_gt.jpg'))
                self.visualize(img_hr_list[0], outputs_hr[0], filename)

            # evaluation on validation batch
            res = {target["image_id"]: output for target, output in zip(target_list, outputs_hr)}
            coco_evaluator.update(res)
            num_processed_samples += batch_size
            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize(self.text_logger)
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_det": self.get_bare_model(self.net_det).state_dict(),
                      "optimizer_det": self.optimizer_det.state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_det).state_dict(), osp.join(self.exp_dir, 'models', "net_det_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return
