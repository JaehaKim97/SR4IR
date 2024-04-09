from torchvision.transforms import transforms


class ManualNormalize:
    def __init__(self):
        self.transforms = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.inv_transforms = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def __call__(self, img, inv=False):
        if inv:
            return self.inv_transforms(img)
        else:
            return self.transforms(img)
