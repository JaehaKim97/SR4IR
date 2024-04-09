"""BasicSR Projects
Code Reference: https://github.com/XPixelGroup/BasicSR
"""
import argparse
import os
import random
import yaml

from collections import OrderedDict
from os import path as osp
from .misc import set_random_seed
from .dist import is_main_process


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)
    
    # random seed
    seed = opt.get('manual_seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
        # print('random seed: {:05d}'.format(seed))
    set_random_seed(seed)
    
    # set task
    task = opt.get('task', None)
    if task is None:
        task = args.opt.split('/')[1]
        if not task in ['cls', 'det', 'seg']:
            raise NotImplementedError("task should be specided in [cls, det, seg]")
        opt['task'] = task
        
    # test_only
    if args.test_only:
        opt['test_only'] = True
    if args.visualize:
        opt['test']['visualize'] = True
    
    if opt.get('test_only', False):
        opt['test']['calculate_lpips'] = True
        if opt.get('train', False):
            opt.pop('train')
    
    return opt, args


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    if is_main_process():
        cmd = ' '.join(sys.argv)
        filename = osp.join(experiments_root, opt_file)
        os.makedirs(osp.join(experiments_root, osp.dirname(opt_file)), exist_ok=True)
        copyfile(opt_file, filename)

        with open(filename, 'r+') as f:
            lines = f.readlines()
            lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
            f.seek(0)
            f.writelines(lines)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value
