from importlib import import_module

def build_network(opt_network, logger, task='common', tag=None):
    opt_network = opt_network.copy()
    arch_name = opt_network.pop('name')
    is_trainable = opt_network.pop('trainable', True)
    
    module = import_module(f'archs.{task}.{arch_name.lower()}_arch')
    arch = module.build_network(**opt_network)
    # import torchvision
    # arch = torchvision.models.get_model(arch_name, **opt_network)
    
    if not is_trainable:
        arch.eval()
        for p in arch.parameters():
            p.requires_grad = False
    
    log = f'Arch [{arch_name}] is created'
    if tag is not None:
        log += f' ({tag})'
    logger.write(log)
    return arch
