import os
import torch
import numpy as np
import math

from scipy.ndimage import convolve
from torchvision.transforms.functional import normalize
from .color_util import bgr2ycbcr
from .niqe import compute_feature, niqe
from .matlab_functions import imresize


def calculate_psnr(img, img2, crop_border=8, img_range=1.0, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (torch.Tensor): Images with range [0, 255].
        img2 (torch.Tensor): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range

    if crop_border != 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return (10. * torch.log10(255. * 255. / mse)).item()


def calculate_psnr_batch(img, img2, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    img = img * 255.0 / img_range
    img2 = img2 * 255.0 / img_range

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img - img2)**2, dim=(1,2,3))  # batch-wise mse
    valid_mask = (mse != 0.)  # do not count if mse is 0; it causes infinity
    # NOTE : there exist zero mse case where the patch is from 
    # val/n03623198/ILSVRC2012_val_00017853.JPEG (all-zero patch)
    mse = mse[valid_mask]
    
    return (10. * torch.log10(255. * 255. / mse)).mean(), valid_mask.sum()  # batch-wise mean


def calculate_lpips_batch(img, img2, net_lpips, crop_border=8, img_range=1.0, **kwargs):
    """Computes the PSNR (Peak-Signal-Noise-Ratio) in batch"""
        
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # norm to [-1, 1]
    img = normalize(img, mean, std)
    img2 = normalize(img2, mean, std)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
    lpips = net_lpips(img, img2).squeeze(1,2,3)  # batch-wise lpips
    valid_mask = (lpips != 0.)  # do not count if mse is 0; it causes infinity
    # NOTE : there exist zero mse case where the patch is from 
    # val/n03623198/ILSVRC2012_val_00017853.JPEG (all-zero patch)
    lpips = lpips[valid_mask]
    
    return lpips.mean(), valid_mask.sum()  # batch-wise mean


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    ``Paper: Making a "Completely Blind" Image Quality Analyzer``

    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, ('Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                      idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam)))

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality


def calculate_niqe_batch(img_batch, crop_border=8, input_order='HWC', convert_to='y', **kwargs):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    ``Paper: Making a "Completely Blind" Image Quality Analyzer``

    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    > MATLAB R2021a result for tests/data/baboon.png: 5.72957338 (5.7296)
    > Our re-implementation result for tests/data/baboon.png: 5.7295763 (5.7296)

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load(os.path.join(ROOT_DIR, 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']
    
    img_batch = img_batch * 255.0
    img_batch = np.array(img_batch.detach().cpu()).astype(np.float32)
    
    niqe_result = 0
    for img in img_batch:
        img = img.transpose(1, 2, 0)  # HWC
        
        img = img.astype(np.float32) / 255.
        if img.ndim == 3 and img.shape[2] == 3:
            img = bgr2ycbcr(img, y_only=True)
            img = img[..., None]
            
        img = np.squeeze(img)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border]

        # round is necessary for being consistent with MATLAB's result
        img = img.round()
        
        print(img.mean())
        niqe_result += niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
    niqe_result /= img_batch.shape[0]

    return niqe_result