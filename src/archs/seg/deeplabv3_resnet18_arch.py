from .deeplabv3 import deeplabv3_resnet18

def build_network(**kwargs):
    return deeplabv3_resnet18(**kwargs)
