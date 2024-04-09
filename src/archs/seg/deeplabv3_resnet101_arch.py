from .deeplabv3 import deeplabv3_resnet101

def build_network(**kwargs):
    return deeplabv3_resnet101(**kwargs)
