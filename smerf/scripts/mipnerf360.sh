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
#
# Runs the SMERF pipeline on all mip-NeRF 360 scenes. Requires 16x A100s.
#
# This will take a very long time. We recommend training one model per host.
#

set -eux

TIMESTAMP="$(date +'%Y%m%d_%H%M')"
SCENES=(
  bicycle
  flowerbed
  fulllivingroom
  gardenvase
  kitchencounter
  kitchenlego
  officebonsai
  stump
  treehill
)

for SCENE in "${SCENES[@]}"; do
  CHECKPOINT_DIR="$(pwd)/checkpoints/${TIMESTAMP}-${SCENE}"
  python3 -m smerf.train \
    --gin_configs=configs/models/smerf.gin \
    --gin_configs=configs/mipnerf360/${SCENE}.gin \
    --gin_configs=configs/mipnerf360/extras.gin \
    --gin_bindings="smerf.internal.configs.Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
    --alsologtostderr
done
