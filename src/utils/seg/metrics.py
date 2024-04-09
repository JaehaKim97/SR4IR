'''
Torchvision
Code Reference: https://github.com/pytorch/vision/tree/main/references
'''
import torch


def calculate_mat(pred, target, n):
    k = (pred >= 0) & (pred < n)
    inds = n * pred[k].to(torch.int64) + target[k]
    return torch.bincount(inds, minlength=n**2).reshape(n, n)


def compute_iou(mat):
    h = mat.float()
    iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    return iu