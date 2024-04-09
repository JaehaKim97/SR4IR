import os.path as osp
from shutil import copyfile
from tqdm import tqdm

with open(osp.join('datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/train.txt')) as f:
    lines = f.readlines()

for name in tqdm(lines):
    name = name[:-1]
    copyfile(osp.join('datasets/VOC/VOCdevkit/VOC2012/JPEGImages', f"{name}.jpg"), osp.join('datasets/VOC_COCO/train', f"{name}.jpg"))

with open(osp.join('datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/val.txt')) as f:
    lines = f.readlines()

for name in tqdm(lines):
    name = name[:-1]
    copyfile(osp.join('datasets/VOC/VOCdevkit/VOC2012/JPEGImages', f"{name}.jpg"), osp.join('datasets/VOC_COCO/val', f"{name}.jpg"))
