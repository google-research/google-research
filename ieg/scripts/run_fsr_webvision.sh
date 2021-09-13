#!/bin/bash

SAVEPATH=./ieg/checkpoints
mkdir -p $SAVEPATH

ARCH='resnet50'
DATASET='webvisionmini'

# Train on 8 GPUs.
python3 -m ieg.main \
--dataset=${DATASET} \
--network_name=${ARCH} \
--checkpoint_path=$SAVEPATH/fsr_webvisionmini \
--method='fsr' \
--ds_include_metadata=True \
--probe_dataset_hold_ratio=0 \
--batch_size=16 \
--max_iteration=123300 \
--meta_start_epoch=20 \
--queue_capacity=5000 \
--use_pseudo_loss='all_1_0.1_1' \
--val_batch_size=50 \
--eval_freq=10000 \
--use_imagenet_as_eval=True \
--l2_weight_decay=0.0001 \
--warmup_epochs=3 \
--cos_t_mul=2 \
--cos_m_mul=0.9


