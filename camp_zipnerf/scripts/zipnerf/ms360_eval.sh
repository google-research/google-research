# Copyright 2024 The Google Research Authors.
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
# Script for evaluating on the multiscale 360 dataset.

export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/usr/local/google/home/barron/data/nerf_real_360
CHECKPOINT_DIR=~/tmp/zipnerf/ms360

# Outdoor scenes.
for SCENE in bicycle flowerbed gardenvase stump treehill
do
  python -m eval \
    --gin_configs=configs/zipnerf/360.gin \
    --gin_configs=configs/zipnerf/use_multiscale.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${SCENE}'"
done

# Indoor scenes.
for SCENE in fulllivingroom kitchencounter kitchenlego officebonsai
do
  python -m eval \
    --gin_configs=configs/zipnerf/360.gin \
    --gin_configs=configs/zipnerf/use_multiscale.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
    --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}/${SCENE}'" \
    --gin_bindings="Config.factor = 2" # Important change from outdoor data
done
