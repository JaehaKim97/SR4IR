import torch

from torchvision.datasets import VOCSegmentation
from utils.seg import SegmentationPresetTrain, SegmentationPresetEval, get_coco, collate_fn


def load_seg_data(opt):
    use_trainset = opt.get('train', False)
    data_name = opt['data']['name']
    
    # transform
    transform_train = SegmentationPresetTrain(base_size=opt['data'].get('base_size', 520), crop_size=opt['data'].get('crop_size', 480))
    transform_test = SegmentationPresetEval(base_size=opt['data'].get('base_size', 520))
    
    # datasets
    dataset_train = None
    if use_trainset:
        if data_name == 'coco':
            dataset_train = get_coco(opt['data']['path'], image_set='train', transforms=transform_train)
        elif data_name == 'voc':
            dataset_train =  VOCSegmentation(root=opt['data']['path'], year='2012', image_set='train', transforms=transform_train)
            
    if data_name == 'coco':
        dataset_test = get_coco(root=opt['data']['path'], image_set='val', transforms=transform_test)
    elif data_name == 'voc':
        dataset_test =  VOCSegmentation(root=opt['data']['path'], year='2012', image_set='val', transforms=transform_test)

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
    if use_trainset:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=opt['train']['batch_size'], sampler=train_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=None)

    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True, collate_fn=collate_fn)


    return data_loader_train, data_loader_test, train_sampler, test_sampler
