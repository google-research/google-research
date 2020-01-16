# Out-of-Distribution Detection for Images

This directory contains scripts for training Pixel CNN models and computing likelihood ratios for detecting out-of-distribution images that are used in the paper of [Ren J, Liu PJ, Fertig E, Snoek J, Poplin R, DePristo MA, Dillon JV, Lakshminarayanan B. Likelihood Ratios for Out-of-Distribution Detection. arXiv preprint arXiv:1906.02845.](https://arxiv.org/abs/1906.02845)

## Installation

```bash
virtualenv -p python3 .
source ./bin/activate

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

### Train a foreground model for FashionMNIST. (For testing purpose, we only train for 10 steps.)
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

### Train background models for FashionMNIST
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

### Model parameters for training Pixel CNN model for FashionMNIST experiment and CIFAR experiment

For training Pixel CNN models for FashionMNIST, we set num_filters=32, num_logistic_mix=1, rescale_pixel_value=False, total_steps=50000. 
For training Pixel CNN models for CIFAR10, we set num_filters=32, num_logistic_mix=1, rescale_pixel_value=True, total_steps=600000.
The rest parameters are set as the default values in train.py. Both foreground and background models are trained using the same model parameters, except that the background models are trained with additional input perturbations and L2 regularizations.


## Evaluate models (find the best background model using validation OOD data, compute likelihood ratios, evaluate AUROC, and plot heatmaps)
```bash
python -m genomics_ood.images_ood.eval \
--exp=$EXPNAME \
--data_dir=$DATADIR \
--model_dir=$OUTDIR/exp$EXPNAME/rescaleFalse \
--ckpt_step=$TOTAL_STEPS \
--repeat_id=-1
```

## Evaluate the models we trained for FashionMNIST-MNIST for producing results in the paper
```bash
python -m genomics_ood.images_ood.eval \
--exp=fashion \
--data_dir=$DATADIR \
--model_dir=./genomics_ood/images_ood/testmodels/expfashion/rescaleFalse \
--ckpt_step=50000 \
--repeat_id=0
```

