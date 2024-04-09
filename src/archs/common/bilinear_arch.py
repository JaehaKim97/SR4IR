import torch.nn as nn

def build_network(**kwargs):
    return BILINEAR(**kwargs)

class BILINEAR(nn.Module):
    def __init__(self, scale=4):
        super(BILINEAR, self).__init__()

        self.layer = nn.UpsamplingBilinear2d(scale_factor=scale)

    def forward(self, x):
        return self.layer(x)
