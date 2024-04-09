from .faster_rcnn import fasterrcnn_mobilenet_v3_large_fpn

def build_network(**kwargs):
    return fasterrcnn_mobilenet_v3_large_fpn(**kwargs)
