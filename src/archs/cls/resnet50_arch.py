from .resnet import resnet50

def build_network(**kwargs):
    return resnet50(**kwargs)
