from .resnet import resnet18

def build_network(**kwargs):
    return resnet18(**kwargs)
