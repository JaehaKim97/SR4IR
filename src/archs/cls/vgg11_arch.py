from .vgg import vgg11

def build_network(**kwargs):
    return vgg11(**kwargs)
