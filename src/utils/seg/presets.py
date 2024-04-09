import torch
import utils.seg.transforms as T


class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size=0,
        hflip_prob=0.5,
    ):
        
        transforms = [T.RandomResize(min_size=base_size, max_size=base_size)]  # make short axis to base_size

        if crop_size > 0:
            transforms += [T.RandomCrop(crop_size)]
        
        transforms += [T.RandomHorizontalFlip(hflip_prob),
                       T.PILToTensor(),
                       T.ToDtype(torch.float, scale=True),]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target=None):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
            self,
            *,
            base_size,
        ):
        
        transforms = [T.RandomResize(min_size=base_size, max_size=base_size),
                      T.PILToTensor(),
                      T.ToDtype(torch.float, scale=True),]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
