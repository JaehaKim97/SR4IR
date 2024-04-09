from .deeplabv3 import deeplabv3_mobilenet_v3_large

def build_network(**kwargs):
    return deeplabv3_mobilenet_v3_large(**kwargs)
