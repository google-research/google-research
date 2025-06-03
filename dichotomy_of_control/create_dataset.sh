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



python dice_rl/scripts/create_dataset.py \
  --load_dir=./tests/testdata \
  --save_dir=./tests/testdata \
  --env_name=bernoulli_bandit \
  --num_trajectory=1000 \
  --max_trajectory_length=1 \
  --tabular_obs=1 \
  --alpha=0.1 \
  --force \

python dice_rl/scripts/create_dataset.py \
  --load_dir=./tests/testdata \
  --save_dir=./tests/testdata \
  --env_name=FrozenLake-v1 \
  --num_trajectory=100 \
  --max_trajectory_length=100 \
  --tabular_obs=1 \
  --alpha=-1.0 \
  --force \

