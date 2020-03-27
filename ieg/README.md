# Tensorflow source code for [Distilling Effective Supervision from Severe Label Noise](https://arxiv.org/pdf/1910.00701.pdf), CVPR2020
We present a holistic framework to train deep neural networks in a way that is highly invulnerable to label
noise. Our method sets the new state of the art on various types of label noise.

This code mainly conducts experiments on the CIFAR datasets.

This is not an officially supported Google product.


## Install
Make sure your python version is 3.5-3.7 (required by Tensorflow 1).

- Use a virtual environment

```bash
virtualenv -p python3 --system-site-packages env
source env/bin/activate
```

- Install packages

```bash
pip3 install -r requirements.txt
```

## Run Experiments

### Train
Train the model with 20%, 40, and 80% uniform noise ratios on CIFAR100.
You can go to higher noise ratio, e.g. 95%, with the following command.

```bash
SAVEPATH=./ieg/checkpoints

GPU=0
for ratio in 0.2 0.4 0.8; do
  CUDA_VISIBLE_DEVICES=${GPU} python -m ieg.main --dataset=cifar100_uniform_${ratio}\
  --network_name=wrn28-10 --checkpoint_path=$SAVEPATH/ieg &
  GPU=$((GPU + 1))
done
```

#### Key training setting choices
- CIFAR10 with uniform noise: ```--dataset=cifar10_uniform_${ratio}```.

- CIFAR10 with asymmetric noise: ```--dataset=cifar10_asymmetric_${ratio}```.

- ResNet29 architecture, set ```--network_name=resnet29```.

- Data augmentation. The default strong augmentation is AutoAugment. You can change to RandAugment by
```--aug_type=randaug```. The performance should be on par. See discussion in the paper.

- MultiGPU is supported by simply ```CUDA_VISIBLE_DEVICES=<id-1>,<id-2>,...```.

- The two key hyperparameters for unsupervsied losses is controled by ```--ce_factor``` and ```--consistency_factor``` (see options.py).

- ```--max_iteration``` controls the total epochs. 200000 means ~400 epochs given batch size 100. 200 epochs can obtain reported results. Training longer is better most time.

- ```--probe_dataset_hold_ratio``` determines the size of clean probe data. 0.02 indicates 1000 clean labeled data (10 image/class) for CIFAR100. For CIFAR10 experiments, set ```--probe_dataset_hold_ratio=0.002``` for the 10 image/class setting.

#### Comparison experiments
```bash
# L2R
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --method=l2r --checkpoint_path=${SAVEPATH}/l2r

# Supervised training with label noise
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --method=supervised --checkpoint_path=${SAVEPATH}/supervised

# IEG with cifar10 asymmetric noise
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar10_asymmetric_0.4 --network_name=resnet29 --probe_dataset_hold_ratio=0.002 --checkpoint_path=${SAVEPATH}/ieg
```

### Evaluation
```bash
ratio=0.8
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_${ratio} --network_name=wrn28-10 --mode=evaluation --checkpoint_path=${SAVEPATH}/ieg
```


## Citation

```
@inproceedings{zhang2019distill,
  title={Distilling Effective Supervision from Severe Label Noise},
  author={Zhang, Zizhao and Zhang, Han and Arik, Sercan O and Lee, Honglak and Pfister, Tomas},
  booktitle={CVPR},
  year={2020}
}
```
