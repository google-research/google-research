# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3FCN model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# ./finetune_on_flowers.sh \
#     /{path to model_checkpoints}/cervical_10x \
#     model.ckpt-2000200 \
#     911

set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=$1  # e.g. /tmp/model_checkpoints/some_model_10x
PRETRAINED_CHECKPOINT_PREFIX=$2  # e.g. model.ckpt-2000200
MODEL_RECEPTIVE_FIELD=$3  # 911 for inception_v3_fcn or 129 for inception_v3_fcn_small

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/flowers-models/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/tmp/flowers

# Download the dataset
echo "Downloading data"
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only network head.
echo "Fine-tuning network head"
python train_inception_v3_fcn.py \
  --train_dir=${TRAIN_DIR}/head_only \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --receptive_field_size=911 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${PRETRAINED_CHECKPOINT_PREFIX} \
  --checkpoint_exclude_scopes=Logits,Softmax \
  --trainable_scopes=Logits,Softmax \
  --max_number_of_steps=1000 \
  --batch_size=16 \
  --save_interval_secs=60 \
  --log_every_n_steps=50

# Run evaluation.
echo "Running evaluation for network head fine-tuning"
python eval_inception_v3_fcn.py \
  --checkpoint_path=${TRAIN_DIR}/head_only \
  --eval_dir=${TRAIN_DIR}/head_only \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --receptive_field_size=911

# Fine-tune all layers.
echo "Fine-tuning all layers"
python train_inception_v3_fcn.py \
  --train_dir=${TRAIN_DIR}/all_layers \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --receptive_field_size=911 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${PRETRAINED_CHECKPOINT_PREFIX} \
  --checkpoint_exclude_scopes=Logits,Softmax \
  --max_number_of_steps=100 \
  --batch_size=16 \
  --save_interval_secs=60 \
  --log_every_n_steps=5

# Run evaluation.
echo "Running evaluation for all layers fine-tuning"
python eval_inception_v3_fcn.py \
  --checkpoint_path=${TRAIN_DIR}/all_layers \
  --eval_dir=${TRAIN_DIR}/all_layers \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --receptive_field_size=911
