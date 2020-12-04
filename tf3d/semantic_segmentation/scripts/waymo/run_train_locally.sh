# Copyright 2020 The Google Research Authors.
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

VERSION=001
JOB_NAME="seg_waymo_${VERSION}"
TRAIN_DIR="/home/${USER}/temp/tf3d/${JOB_NAME}"

LEARNING_RATES=0.3
BATCH_SIZE=4
NUM_WORKERS=1
NUM_GPUS=1

NUM_STEPS_PER_EPOCH=100
NUM_EPOCHS=80
LOG_FREQ=100

TRAIN_GIN_CONFIG="tf3d/semantic_segmentation/configs/waymo_train.gin"
IMPORT_MODULE='tf3d.gin_imports'
DATASET_NAME='waymo_object_per_frame'
TRAIN_SPLIT='train'

PARAMS="get_tf_data_dataset.dataset_name = ${DATASET_NAME}
get_tf_data_dataset.split_name = ${TRAIN_SPLIT}
step_decay.initial_learning_rate = ${LEARNING_RATES}
"

echo "Deleting TRAIN_DIR at ${TRAIN_DIR}..."
rm -r "${TRAIN_DIR}"

python -m tf3d.train \
  --params="${PARAMS}" \
  --import_module="${IMPORT_MODULE}" \
  --config_file="${TRAIN_GIN_CONFIG}" \
  --train_dir="${TRAIN_DIR}" \
  --num_workers="${NUM_WORKERS}" \
  --num_gpus="${NUM_GPUS}" \
  --run_functions_eagerly=false \
  --num_steps_per_epoch="${NUM_STEPS_PER_EPOCH}" \
  --log_freq="${LOG_FREQ}" \
  --distribution_strategy="mirrored" \
  --batch_size="${BATCH_SIZE}" \
  --alsologtostderr
