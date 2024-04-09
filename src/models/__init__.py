from importlib import import_module

def make_model(opt):
    task = (opt['task']).lower()
    model_type = (opt['model_type']).lower()
    
    module = import_module(f'models.{task}.{model_type}_model')
    print(f'Model [{model_type}_model] is created')
    return module.make_model(opt)
