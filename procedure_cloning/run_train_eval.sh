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



python -m procedure_cloning.scripts.train_eval \
  --env_name='maze:16-tunnel' \
  --load_dir='./tests/testdata' \
  --train_seeds=5 \
  --test_seeds=1 \
  --num_trajectory=4 \
  --algo_name='pc' \
  --num_steps=100_000 \
  --eval_interval=1_000 \
  --num_eval_episodes=3 \
  --max_eval_episode_length=50 \
