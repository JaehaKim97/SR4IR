import torch

from torchvision.datasets import VOCDetection
from utils.det import DetectionPresetTrain, DetectionPresetEval, get_coco, create_aspect_ratio_groups, GroupedBatchSampler, collate_fn


def load_det_data(opt):
    use_trainset = opt.get('train', False)
    data_format = opt['data']['format']
    is_voc = opt['data'].get('is_voc', False)
    
    # transform
    transform_train = DetectionPresetTrain(crop_size=opt['data'].get('crop_size', 0))
    transform_test = DetectionPresetEval()
    
    # datasets
    dataset_train = None
    if use_trainset:
        if data_format == 'coco':
            dataset_train = get_coco(root=opt['data']['path'], image_set='train', transforms=transform_train, mode="instances", is_voc=is_voc)
        elif data_format == 'voc':
            dataset_train =  VOCDetection(root=opt['data']['path'], year='2012', image_set='train', transforms=transform_train)
            
    if data_format == 'coco':
        dataset_test = get_coco(root=opt['data']['path'], image_set='val', transforms=transform_test, mode="instances", is_voc=is_voc)
    elif data_format == 'voc':
        dataset_test =  VOCDetection(root=opt['data']['path'], year='2012', image_set='val', transforms=transform_test)

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
    
    if use_trainset:
        # aspect ratio batch sampler
        if opt['data'].get('aspect_ratio_group_factor', 3) >= 0:
            group_ids = create_aspect_ratio_groups(dataset_train, k=opt['data']['aspect_ratio_group_factor'])
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, opt['train']['batch_size'])
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opt['train']['batch_size'], drop_last=True)  
    
    # data loader    
    data_loader_train = None
    if use_trainset:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_batch_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=collate_fn)

    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=collate_fn)


    return data_loader_train, data_loader_test, train_sampler, test_sampler
