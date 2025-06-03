#!/bin/bash
# Copyright 2025 The Google Research Authors.
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



# Train JAX-BARF on all Blender scenes sequentially.
# Note, update path to the nerf_synthetic/Blender specified by DATA_DIR below.
for SCENE in hotdog chair materials ship ficus mic drums lego
do
  DATA_DIR=./nerf_synthetic/${SCENE}
  TRAIN_DIR=./models/${SCENE}
  mkdir -p ${TRAIN_DIR}

  python -m src.train \
    --data_dir=${DATA_DIR} \
    --train_dir=${TRAIN_DIR} \
    --config=blender \
    --init_poses_from_gt=True \
    > ./models/${SCENE}/train_log.txt
done
