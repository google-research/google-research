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



for s in {0..5}; do
python dice_rl/scripts/create_dataset.py \
  --save_dir=/tmp/procedure_cloning \
  --env_name=maze:16-tunnel \
  --num_trajectory=4 \
  --max_trajectory_length=100 \
  --tabular_obs=0 \
  --alpha=1.0 \
  --seed=$s;
done
