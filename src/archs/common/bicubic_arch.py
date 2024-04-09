import torch.nn as nn

def build_network(**kwargs):
    return BICUBIC(**kwargs)

class BICUBIC(nn.Module):
    def __init__(self, scale=4):
        super(BICUBIC, self).__init__()

        self.layer = nn.Upsample(scale_factor=scale, mode='bicubic')

    def forward(self, x):
        return self.layer(x)
