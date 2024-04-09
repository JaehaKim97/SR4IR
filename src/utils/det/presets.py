import torch
import utils.det.transforms as T


class DetectionPresetTrain:
    def __init__(
        self,
        *,
        crop_size=0,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
    ):

        transforms = []
        
        if crop_size > 0:
            transforms += [T.FixedSizeCrop(size=(crop_size, crop_size), fill=mean),]
            
        transforms += [T.RandomHorizontalFlip(p=hflip_prob),
                       T.PILToTensor(),
                       T.ToDtype(torch.float, scale=True),]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        
        transforms = [T.PILToTensor(),
                      T.ToDtype(torch.float, scale=True),]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
