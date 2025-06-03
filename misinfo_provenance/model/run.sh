# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Run Training Script. 

set -eu
TRAIN_DATASET=ImageNet
TRAIN_INPUT=""# Insert Path to the TF.Records of the training set.
EVAL_INPUT=""# Insert Path to the TF.Records of the evaluation set.

BATCH_SIZE=256
EPOCHS=20
IMAGE_SIZE=256
LOSS_TYPE="local-global" # use both loss during training.
# LOSS_TYPE="global" # use global loss only during training.
INFONCE_TEMPERATURE=0.03
LOSS_WEIGHT=0.5 # weight each loss during training.
LOCAL_TEMPERATURE=0.1
ENTROPY_WEIGHT=5

# Chosse the ResNet layer to extract the local features.
# Conv5 8dims.
# CONV_OUT_LAYER='conv5_block3_out'
# BLOCK_LOCAL_DIMS=8

# Conv4 16dims.
# CONV_OUT_LAYER='conv4_block3_out'
# BLOCK_LOCAL_DIMS=16

# Conv3 32dims.
CONV_OUT_LAYER='conv3_block3_out'
BLOCK_LOCAL_DIMS=32

# Conv2 32dims.
# CONV_OUT_LAYER='conv2_block3_out'
# BLOCK_LOCAL_DIMS=32

TIMESTAMP=$(date "+%m%d-%H%M")
EXP_NAME="SSL-GLOBAL-LOSS_${LOSS_TYPE}-${TIMESTAMP}"
LOGDIR="./"

python -m model/train.py \
  --exp_name="${EXP_NAME}" \
  --seed="42"\
  --logdir="${LOGDIR}" \
  --train_file_pattern="${TRAIN_INPUT}" \
  --validation_file_pattern="${EVAL_INPUT}" \
  --batch_size="${BATCH_SIZE}" \
  --epochs="${EPOCHS}"\
  --image_size="${IMAGE_SIZE}"\
  --block_local_dims="${BLOCK_LOCAL_DIMS}"\
  --conv_output_layer="${CONV_OUT_LAYER}"\
  --loss_type="${LOSS_TYPE}"\
  --local_temperature="${LOCAL_TEMPERATURE}"\
  --infonce_temperature="${INFONCE_TEMPERATURE}"\
  --entropy_weight="${ENTROPY_WEIGHT}"