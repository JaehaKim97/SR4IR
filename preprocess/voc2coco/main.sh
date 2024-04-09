# make directories
mkdir -p datasets/VOC_COCO/train
mkdir -p datasets/VOC_COCO/val
mkdir -p datasets/VOC_COCO/annotations

# copy image files belong to train/val set
echo Copying images..
python preprocess/voc2coco/copy_imgs.py

# make COCO-style annotation
echo Converting to COCO style..
python preprocess/voc2coco/voc2coco.py --ann_dir datasets/VOC/VOCdevkit/VOC2012/Annotations/ --ann_ids datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/train.txt --labels preprocess/voc2coco/voc_labels.txt --output datasets/VOC_COCO/annotations/instances_train.json --ext xml
python preprocess/voc2coco/voc2coco.py --ann_dir datasets/VOC/VOCdevkit/VOC2012/Annotations/ --ann_ids datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main/val.txt --labels preprocess/voc2coco/voc_labels.txt --output datasets/VOC_COCO/annotations/instances_val.json --ext xml

echo Done!