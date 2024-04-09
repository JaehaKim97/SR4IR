'''
BasicSR Project
Code Reference: https://github.com/XPixelGroup/BasicSR
'''
import numpy as np
import os
import random
import time
import torch

from .dist import is_main_process


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def quantize(image, img_range=1.0):
    return torch.clamp((image * (255.0 / img_range)).round(), 0, 255.0) / (255.0 / img_range)


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if is_main_process():
        if os.path.exists(path):
            new_name = path + '_archived_' + get_time_str()
            print(f'Path already exists. Rename it to {new_name}', flush=True)
            os.rename(path, new_name)
        os.makedirs(path, exist_ok=True)
        

def rename_and_mkdir(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if is_main_process():
        if os.path.exists(path):
            new_name = path + '_tmp_' + get_time_str()
            print(f'Path already exists. Rename experiment name to {new_name}', flush=True)
            path = new_name
        os.makedirs(new_name, exist_ok=True)
    return new_name
        

def check_then_rename(path):
    """
    """
    if is_main_process():
        if os.path.exists(path):
            new_name = path + '_tmp_' + get_time_str()
            print(f'Path already exists. Rename experiment name to {new_name}', flush=True)
    return new_name
