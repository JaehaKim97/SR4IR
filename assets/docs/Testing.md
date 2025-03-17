## Pre-trained model

You can download the pre-trained model in [**here**](https://drive.google.com/drive/folders/1ChS_olbEhA7o4JRyqHGZD1YrSwx9-Pqg?usp=sharing).

Note that we only offer models for {Segmentation, Detection, Classification-StanfordCars} with X8 SR scale combined with the EDSR-baseline SR model.

(*Update 2025.03.17* - We additionally upload SR4IR with SwinIR models in both x4 and x8 scales)

Locate them on `experiments/`, e.g., `experiments/seg/000_H2T/*`.

## Testing command

The testing command is below:

```
CUDA_VISIBLE_DEVICES=0 python src/main.py -opt path/to/config --test_only
```

The ` --test_only` flag will automatically load the latest SR and Task network weights in the corresponding experiment folder.

The results of the pre-trained models are following:

![alt text](/assets/images/SR4IR_results.png)

If you want to visualize the results in images, add the `--visualize` flag.

Note that you can test your own trained model in same way.

<br />
<br />

<div align="right">
 Return to main: https://github.com/JaehaKim97/SR4IR
</div>
