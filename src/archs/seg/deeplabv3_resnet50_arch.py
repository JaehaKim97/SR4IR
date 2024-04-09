from .deeplabv3 import deeplabv3_resnet50

def build_network(**kwargs):
    return deeplabv3_resnet50(**kwargs)
