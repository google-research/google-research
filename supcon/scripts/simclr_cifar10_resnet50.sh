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

# Intended to be run on a 32-core TPU.

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/configs.sh" || exit 1


set -x

"${LAUNCH}" "${LAUNCHER[@]}" \
  "${TPU_FLAGS[@]}" \
  "${RESNET50x1_FLAGS[@]}" \
  "${CIFAR10_FLAGS[@]}" \
  "${SIMCLR_FLAGS[@]}" \
  --batch_size=1024 \
  --augmentation_type=SIMCLR --augmentation_magnitude=0.5 \
  --temperature=0.5 \
  --stage_1_weight_decay=1e-4 --stage_2_weight_decay=0 \
  --label_smoothing=0 \
  --stage_1_epochs=400 --stage_2_epochs=90 \
  --stage_1_warmup_epochs=10 --stage_2_warmup_epochs=0 \
  --stage_1_base_learning_rate=0.5 --stage_2_base_learning_rate=0.25 \
  --stage_1_learning_rate_decay=COSINE --stage_2_learning_rate_decay=COSINE \
  --stage_1_optimizer=MOMENTUM  --stage_2_optimizer=MOMENTUM \
  "$@"
