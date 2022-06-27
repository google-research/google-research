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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

DEBUG=false

EXPERIMENT_NAME='delete_me'
DATASET_NAME='CIFAR10'  # CIFAR10, CIFAR100, TinyImageNet, ImageNet2012
MODEL='densenet_cifar'  # resnet18, resnet34, resnet50, densenet_cifar
CASCADED=false

TDL_MODE='OSD'  # OSD, EWS, noise
TDL_ALPHA=0.0
NOISE_VAR=0.0

EPOCHS=5
BATCH_SIZE=128
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
NESTEROV=true

python -m cascaded_networks.train.py -- \
  --config=configs/train_base_flags.py \
    --config.debug=$DEBUG \
    --config.on_gcp=false \
    --config.experiment_name=$EXPERIMENT_NAME \
    --config.epochs=$EPOCHS \
    --config.model_key=$MODEL \
    --config.cascaded=$CASCADED \
    --config.tdl_mode=$TDL_MODE \
    --config.tdl_alpha=$TDL_ALPHA \
    --config.noise_var=$NOISE_VAR \
    --config.dataset_name=$DATASET_NAME \
    --config.batch_size=$BATCH_SIZE \
    --config.learning_rate=$LR \
    --config.momentum=$MOMENTUM \
    --config.weight_decay=$WEIGHT_DECAY \
    --config.nesterov=$NESTEROV
