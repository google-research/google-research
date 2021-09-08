#!/bin/bash

SAVEPATH=./ieg/checkpoints
mkdir -p $SAVEPATH

RATIO=0.02  # MODIFY BY USERS
NOISERATIO=0.2  # MODIFY BY USERS
ARCH='resnet32'
DATASET=cifar10_imbal_${RATIO}_${NOISERATIO}
DIR=./ieg/data

GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python3 -m ieg.main \
--dataset=${DATASET} \
--network_name=${ARCH} \
--dataset_dir=${DIR} \
--checkpoint_path=$SAVEPATH/fsr/cifar_longtail_noise \
--method='fsr' \
--ds_include_metadata=True \
--probe_dataset_hold_ratio=0 \
--batch_size=128 \
--lr_schedule='custom_step' \
--max_epoch=200 \
--decay_rate=0.01 \
--label_smoothing=0.1 \
-decay_epochs='160,180'\
--meta_start_epoch=160 \
--queue_capacity=2000 \
--queue_bs=200 \
--clip_meta_weight=True
