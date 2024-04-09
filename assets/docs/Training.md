## Pretrained SR models

We initialize the SR network from the network weights trained on the DIV2K dataset.

You can download those pre-trained SR network weights from [**here**](https://drive.google.com/drive/folders/1Ly8yS8SF5V5itOQj3b9DE2k_IO1NvLXE?usp=sharing), then please locate them in `experiments/`.

To be precise, the weights should be located as `datasets/pretrained_models/*`.


## Training command

### Single GPU
```
CUDA_VISIBLE_DEVICES=0 python src/main.py -opt path/to/config
# e.g., CUDA_VISIBLE_DEVICES=0 python src/main.py -opt options/seg/000_H2T.yml
```

### Multi GPU (DDP)
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 src/main.py -opt path/to/config      # 2 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 src/main.py -opt path/to/config  # 4 GPU
```

## Configuration files

We offer all training configuration files used in our experimental section (Table 1, 2, and 3 of our main manuscript).

You can check detailed training configurations in ```options/``` at the following directories:

```
 ${ROOT}
 ├──options
    ├──seg
    ├──det
    └──cls
       ├── StanfordCars
       └── CUB200
```

For each training configuration, the required number of GPU differs, and you can find it on the commented note beside training `batch_size`.

By default, we assume that the memory size of the single GPU is 48GB.

However, if your GPU memory size is smaller, so the OOM issue occurs, then you can consider reducing the training `batch_size` and using multi-GPU.


<br />
<br />

<div align="right">
 Return to main: https://github.com/JaehaKim97/SR4IR
</div>

