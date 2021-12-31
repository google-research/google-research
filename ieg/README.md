# Source Code of Meta Sample Re-weighting for Robust Training

This repository contains tensorflow code for the following works

-   IEG:
    [Distilling Effective Supervision from Severe Label Noise](https://arxiv.org/pdf/1910.00701.pdf),
    CVPR2020

-   FSR:
    [Learning Fast Sample Re-weighting Without Reward Data](https://arxiv.org/pdf/2109.03216.pdf),
    ICCV2021

This is not an officially supported Google product.

## Install

-   Use a virtual environment

    ```
    virtualenv -p python3 --system-site-packages iegenv
    source iegenv/bin/activate
    ```

-   Install packages

    ```
    pip3 install -r requirements.txt
    ```

## Run Experiments of IEG

### Train on CIFAR

Train the model with 20%, 40, and 80% uniform noise ratios on CIFAR100. You can
go to higher noise ratio, e.g. 95%, with the following command.

```bash
SAVEPATH=./ieg/checkpoints

GPU=0
for ratio in 0.2 0.4 0.8; do
  CUDA_VISIBLE_DEVICES=${GPU} python -m ieg.main --dataset=cifar100_uniform_${ratio}\
  --network_name=wrn28-10 --checkpoint_path=$SAVEPATH &
  GPU=$((GPU + 1))
done
```

#### Useful training setting options

-   CIFAR10 with uniform noise: `--dataset=cifar10_uniform_${ratio}`.

-   CIFAR10 with asymmetric noise: `--dataset=cifar10_asymmetric_${ratio}`.

-   ResNet29 architecture, set `--network_name=resnet29`.

-   Data augmentation. The default strong augmentation is AutoAugment. You can
    change to RandAugment by `--aug_type=randaug`. The performance should be on
    par. See discussion in the paper.

-   MultiGPU is supported by simply `CUDA_VISIBLE_DEVICES=<id-1>,<id-2>,...`.

-   The two key hyperparameters for unsupervsied losses is controled by
    `--ce_factor` and `--consistency_factor` (see options.py).

-   `--max_iteration` controls the total epochs. 200000 means ~400 epochs given
    batch size 100. 200 epochs can obtain reported results. Training longer is
    better most time.

-   `--probe_dataset_hold_ratio` determines the size of clean probe data. 0.02
    indicates 1000 clean labeled data (10 image/class) for CIFAR100. For CIFAR10
    experiments, set `--probe_dataset_hold_ratio=0.002` for the 10 image/class
    setting.

#### Comparison experiments

```bash
# L2R
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --method=l2r --checkpoint_path=${SAVEPATH}

# Supervised training with label noise
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_0.8 --method=supervised --checkpoint_path=${SAVEPATH}

# IEG with cifar10 asymmetric noise
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar10_asymmetric_0.4 --network_name=resnet29 --probe_dataset_hold_ratio=0.002 --checkpoint_path=${SAVEPATH}
```

### Evaluation

```bash
ratio=0.8
CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar100_uniform_${ratio} --network_name=wrn28-10 --mode=evaluation --checkpoint_path=${SAVEPATH}
```

### Train on Webvision (mini)

The following scripts will train on 8 GPUs and evaluate on ImageNet val.

#### Download dataset

```bash
mkdir -p data/tensorflow_datasets
cd data/tensorflow_datasets
wget https://storage.googleapis.com/gresearch/ieg/webvisionmini2.zip
unzip webvisionmini.zip
```

Since we evaluate on ImageNet val, at the first time, follow the prompted
guideline in the terminal to download imagenet `ILSVRC2012_img_[train/val].par`
to the folder (`./data/tensorflow_datasets/downloads/manual/imagenet2012`) for
tensorflow_datasets to prepare automatically.

#### Train

```bash
SAVEPATH=./ieg/checkpoints

# Train on 8 GPUs.
python -m ieg.main --dataset=webvisionmini \
--network_name=resnet50 --checkpoint_path=$SAVEPATH \
--ce_factor=4 --consistency_factor=8 --batch_size=8 --use_imagenet_as_eval=true \
--val_batch_size=50 --eval_freq=10000 --max_iteration=1000000
```

## Run Experiments of FSR

### Train on CIFAR

Train the model with 20%, 40, and 80% uniform noise ratios on CIFAR100. You can
go to higher noise ratio, e.g. 95%, with the following command.

```bash
SAVEPATH=./ieg/checkpoints

GPU=0
for ratio in 0.2 0.4 0.8; do
  CUDA_VISIBLE_DEVICES=${GPU} python -m ieg.main --dataset=cifar100_uniform_${ratio}\
  --network_name=wrn28-10 --method='fsr' --checkpoint_path=$SAVEPATH \
  --max_iteration=63500 --cos_t_mul=2 --probe_dataset_hold_ratio=0 \
  --meta_partial_level=0 --ds_include_metadata=True &

  GPU=$((GPU + 1))
done
```

### Train on long-tailed CIFAR.

#### Download dataset

```bash
mkdir -p data/imbalance
cd data/imbalance
```

Follow the
[instruction](https://github.com/richardaecn/class-balanced-loss#datasets) to
download long-tailed CIFAR (data.zip) and

```bash
unzip data.zip
mv data imbalance # Chang efolder name to imbalance.
```

```bash
bash ./ieg/scripts/run_fsr_cifar_longtail.sh
```

#### Train on complex of long-tailed and label noise CIFAR.

```bash
bash ./ieg/scripts/run_fsr_cifar_longtail_noise.sh
```

### Train on Webvision (mini)

Follow the instruction above to download the webvision data first.

```bash
bash ./ieg/scripts/run_fsr_webvision.sh
```

## Citation

Please cite our work if you find it is useful

```
@inproceedings{zhang2019distill,
    title={Distilling Effective Supervision from Severe Label Noise},
    author={Zhang, Zizhao and Zhang, Han and Arik, Sercan O and Lee, Honglak and Pfister, Tomas},
    booktitle={CVPR},
    year={2020}
}

@inproceedings{zhang2021learning,
    title={Learning Fast Sample Re-weighting Without Reward Data},
    author={Zhang, Zizhao and Pfister, Tomas},
    booktitle={ICCV},
    year={2021} }

```
