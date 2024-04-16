## Installation

1. Make a clone of this repo.

```
git clone https://github.com/JaehaKim97/SR4IR.git
cd SR4IR
```

2. Set the environment. We recommend using [**Anaconda**](https://www.anaconda.com/products/distribution).

```
 conda env create -f environment.yaml
 conda activate SR4IR
 python src/setup.py
```

If you encounter an error, please try another installation command as noted [**here**](./Alternative.md).


## Dataset preparation

### 1. Semantic Segmentation

We use the Pascal-VOC dataset. You can download it from the [**official homepage**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

Locate the dataset file ```VOCtrainval_11-May-2012.tar``` at ```datasets/``` then run the below command:

```
 python preprocess/voc/main.py
```

The processed dataset will located at ```datasets/VOC```.

### 2. Object Detection

We used the Pascal-VOC dataset again, but with some pre-processing steps that convert the dataset's annotation into the COCO style.

Make sure that you follow the above dataset preparation for semantic segmentation, i.e., you have the VOC dataset within the path: ```datasets/VOC```.

Then execute the below command:

```
 bash preprocess/voc2coco/main.sh
```

It will generate COCO style annotated VOC dataset within the path ```datasets/VOC_COCO```

### 3. Image Classification

We use two datasets; StanfordCars and CUB-200-2011 datasets.

#### Stanford Cars

Download the dataset via Kaggle in [**here**](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) (Download button in top-right).

Locate the dataset file ```archive.zip``` at ```datasets/``` then run the below command:

```
 python preprocess/stanfordcars/main.py
```

The processed dataset will located at ```datasets/StanfordCars```.

#### CUB-200-2011

Download the dataset via Kaggle in [**here**](https://www.kaggle.com/datasets/wenewone/cub2002011?select=CUB_200_2011) (Download button in top-right).

Locate the dataset file ```archive.zip``` at ```datasets/``` then run the below command:

```
 python preprocess/cub-200-2011/main.py
```

The processed dataset will located at ```datasets/CUB200```.

<br />
<br />

<div align="right">
 Link to <a href="./Training.md" style="float: right;">Training</a> and <a href="./Testing.md" style="float: right;">Testing</a>.
 
 Return to main: https://github.com/JaehaKim97/SR4IR
</div>
