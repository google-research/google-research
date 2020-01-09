# Out-of-Distribution Detection for Images

This directory contains scripts for training pixel_cnn models and computing likelihood ratios for detecting out-of-distribution images that are used in the paper of [Ren J, Liu PJ, Fertig E, Snoek J, Poplin R, DePristo MA, Dillon JV, Lakshminarayanan B. Likelihood Ratios for Out-of-Distribution Detection. arXiv preprint arXiv:1906.02845. 2019 Jun 7.](https://arxiv.org/abs/1906.02845)

## Installation

```bash
pip install -r genomics_ood/images_ood/requirements.txt
```

## Image Dataset Preparation

```bash
DATADIR=./genomics_ood/images_ood/image_data
```

### Download datasets of FashionMNIST, MNIST, CIFAR10, and SVHN.
```bash
for DNAME in fashion_mnist mnist cifar10 svhn_cropped;
do python -m genomics_ood.images_ood.tfds_to_np \
--out_dir=$DATADIR \
--name=$DNAME; 
done
```

### Download dataset of NotMNIST
```bash
wget -O $DATADIR/notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
tar xzf $DATADIR/notMNIST_small.tar.gz -C $DATADIR

python -m genomics_ood.images_ood.notmnist_to_np \
--raw_data_dir=$DATADIR/notMNIST_small \
--out_dir=$DATADIR
```

## Train Pixel CNN models

```bash
OUTDIR=./genomics_ood/images_ood/output
```

### train a foreground model for FashionMNIST. (For testing purpose, we only train for 10 steps.)
```bash
EXPNAME=fashion
TOTAL_STEPS=10
EVAL_EVERY=10

python -m genomics_ood.images_ood.train \
--exp=$EXPNAME \
--data_dir=$DATADIR \
--out_dir=$OUTDIR \
--total_steps=$TOTAL_STEPS \
--eval_every=$EVAL_EVERY \
--mutation_rate=0.0 \
--reg_weight=0.0
```

## train backgrond models for FashionMNIST
```bash
for MR in 0.1 0.2 0.3; 
do for REG in 0 10 100;
do python -m genomics_ood.images_ood.train \
--exp=$EXPNAME \
--data_dir=$DATADIR \
--out_dir=$OUTDIR \
--total_steps=$TOTAL_STEPS \
--eval_every=$EVAL_EVERY \
--mutation_rate=$MR \
--reg_weight=$REG;
done;
done
```





