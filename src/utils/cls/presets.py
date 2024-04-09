import torch
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size=224,
        resize_size=256,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
    ):

        transforms = [T.Resize(resize_size, interpolation=interpolation)]
        
        if crop_size > 0:
            transforms += [T.RandomCrop(crop_size),]
            
        transforms += [T.RandomHorizontalFlip(hflip_prob),
                       T.PILToTensor(),
                       T.ConvertImageDtype(torch.float),]
        
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
            self,
            *,
            crop_size=224,
            resize_size=256,
            interpolation=InterpolationMode.BILINEAR,
        ):
        
        transforms = [T.Resize(resize_size, interpolation=interpolation),
                      T.CenterCrop(crop_size),
                      T.PILToTensor(),
                      T.ConvertImageDtype(torch.float),]

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
