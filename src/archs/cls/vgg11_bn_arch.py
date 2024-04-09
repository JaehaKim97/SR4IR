from .vgg import vgg11_bn

def build_network(**kwargs):
    return vgg11_bn(**kwargs)
