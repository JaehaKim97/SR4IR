import os
import torch
import torchvision

from utils.cls import ClassificationPresetTrain, ClassificationPresetEval


def load_cls_data(opt):
    use_trainset = opt.get('train', False)
    
    # data path
    traindir = os.path.join(opt['data']['path'], "train")
    valdir = os.path.join(opt['data']['path'], "val")
    if opt['data'].get('val_part', False):
        valdir = os.path.join(opt['data']['path'], "val_part")
        
    # transforms
    transform_train = ClassificationPresetTrain(crop_size=opt['data']['train_crop_size'])
    transform_test = ClassificationPresetEval()
    
    # datasets
    dataset_train = None
    if use_trainset:
        dataset_train = torchvision.datasets.ImageFolder(traindir, transform_train)
    dataset_test = torchvision.datasets.ImageFolder(valdir, transform_test)
    
    # distributed training
    train_sampler = None
    if opt['dist']:
        if use_trainset:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if use_trainset:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    # data loader
    data_loader_train = None
    if opt.get('train', False):
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=opt['train']['batch_size'], sampler=train_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=None)
        
    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=opt['test']['batch_size'], sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=None)

    return data_loader_train, data_loader_test, train_sampler, test_sampler
