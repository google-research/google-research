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

# Intended to be run on a 256-core TPU.

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/configs.sh" || exit 1


set -x

"${LAUNCH}" "${LAUNCHER[@]}" \
  "${TPU_FLAGS[@]}" \
  "${RESNET200x1_FLAGS[@]}" \
  "${IMAGENET_FLAGS[@]}" \
  "${XENT_FLAGS[@]}" \
  --batch_size=4096 \
  --augmentation_type=STACKED_RANDAUGMENT --augmentation_magnitude=1.0 \
  --stage_1_weight_decay=1e-4 \
  --stage_1_epochs=700 \
  --stage_1_base_learning_rate=0.1 \
  --stage_1_optimizer=NESTEROV \
  "$@"

