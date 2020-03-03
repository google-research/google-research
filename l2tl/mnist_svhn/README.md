# Codebase for "Learning to Transfer Learn"  (WORK IN PROGRESS)

Authors: Linchao Zhu, Sercan O. Arik, Yi Yang, and Tomas Pfister

Paper: https://arxiv.org/abs/1908.11406

Learning to transfer learn (L2TL), to improve transfer learning on a target dataset by judicious extraction of information from a source dataset.

This repository contains an example implementation of L2TL framework.

This codebase has not been finalized yet. Once it is finalized, usage examples will be added here.

In this repo, we provide a baseline of transfer from MNIST from SVHN. For SVHN, we sampled 600 images from the original dataset.


## Installation

pip install tensorflow-gpu==0.14.0

pip install tensorflow-datasets

pip install tfp-nightly==0.7.0.dev20190529

### Setup dataset
1. We add the small svhn dataset. Copy `scripts/__init__.py scripts/svhn_small.py` to `xxx/site-packages/tensorflow_datasets/image`

`cp scripts/__init__.py scripts/svhn_small.py xxx/site-packages/tensorflow_datasets/image`

2. Download SVHN dataset.

```
cd scripts
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```

3. Generate new SVHN split.

```
cd scripts
python gen_svhn_mat.py
```

## Evaluation
1. Train the SVHN baseline.

```
python finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=6000 \
    --model_dir=/tmp/l2tl/svhn_small_train \
    --train_batch_size=64
```

2. Evaluate the SVHN baseline model.

```
python evaluate.py \
    --ckpt_path=/tmp/l2tl/svhn_small_train/model.ckpt-6000 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst
```

The results are:

`global_step = 6000, loss = 2.269133, top_1_accuracy = 0.5996094, top_5_accuracy = 0.93359375`

3. Pretrain on the MNIST dataset.

```
python finetuning.py \
    --target_dataset=mnist \
    --train_steps=10000 \
    --model_dir=/tmp/l2tl/mnist_train \
    --train_batch_size=64
```

Evaluate the model:

```
python evaluate.py \
    --ckpt_path=/tmp/l2tl/mnist_train/model.ckpt-10000 \
    --target_dataset=mnist \
    --cls_dense_name=final_dense_dst
```

The results are:

`global_step = 10000, loss = 0.034860205, top_1_accuracy = 0.9780459, top_5_accuracy = 0.988924`

4. Train L2TL.

```
python train_l2tl.py \
    --train_batch_size=64 \
    --train_steps=6000 \
    --model_dir=/tmp/l2tl/l2tl_train \
    --warm_start_ckpt_path=/tmp/l2tl/mnist_train/model.ckpt-10000
```

```
python evaluate.py \
    --ckpt_path=/tmp/l2tl/l2tl_train/model.ckpt-6000 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_target_dense
```

The results are:

`global_step = 6000, loss = 2.2735336, top_1_accuracy = 0.7109375, top_5_accuracy = 0.953125`
