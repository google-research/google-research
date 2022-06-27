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

# Example of training just the classification head, using a pretrained encoder
# and projection head (the projection head is only relevant for measuring
# contrastive loss). This script expects an argument which is a path to the
# train directory of a previously trained model, which was launched using the
# simclr_imagenet_resnet50.sh script. If your previously trained model used a
# different configuration, then flags in this file will likely have to be
# updated so that it will restore correctly.

CKPT=$1
shift

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/configs.sh" || exit 1


set -x

"${LAUNCH}" "${LAUNCHER}" \
  "${TPU_FLAGS[@]}" \
  "${RESNET50x1_FLAGS[@]}" \
  "${IMAGENET_FLAGS[@]}" \
  "${SIMCLR_FLAGS[@]}" \
  "${ONLY_CLASSIFIER_FLAGS[@]}" \
  --batch_size=6144 \
  --augmentation_type=SIMCLR --augmentation_magnitude=1.0 \
  --stage_1_weight_decay=1e-6 \
  --stage_1_epochs=90 \
  --stage_1_base_learning_rate=0.1 \
  --stage_1_optimizer=NESTEROV \
  --reference_ckpt="${CKPT}"
  "$@"
