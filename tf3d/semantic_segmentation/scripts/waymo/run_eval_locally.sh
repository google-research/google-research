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
EVAL_DIR="/home/${USER}/temp/tf3d/${JOB_NAME}"
CKPT_DIR="PATH_TO_CKPT"

NUM_STEPS_PER_EPOCH=100
LOG_FREQ=100

EVAL_GIN_CONFIG="tf3d/semantic_segmentation/configs/waymo_eval.gin"
IMPORT_MODULE='tf3d.gin_imports'
DATASET_NAME='waymo_object_per_frame'
EVAL_SPLIT='val'

PARAMS="get_tf_data_dataset.dataset_name = ${DATASET_NAME}
get_tf_data_dataset.split_name = ${EVAL_SPLIT}
"

echo "Deleting EVAL_DIR at ${EVAL_DIR}..."
rm -r "${EVAL_DIR}"

python -m tf3d.eval \
  --params="${PARAMS}" \
  --import_module="${IMPORT_MODULE}" \
  --config_file="${EVAL_GIN_CONFIG}" \
  --eval_dir="${EVAL_DIR}" \
  --ckpt_dir="${CKPT_DIR}" \
  --run_functions_eagerly=false \
  --num_steps_per_epoch="${NUM_STEPS_PER_EPOCH}" \
  --num_steps_per_log="${LOG_FREQ}" \
  --alsologtostderr
