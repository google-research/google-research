# Copyright 2021 The Google Research Authors.
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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r experience_replay/requirements.txt

python -m experience_replay.train \
  --gin_files=experience_replay/configs/dqn.gin \
  --schedule=continuous_train_and_eval \
  --base_dir=/tmp/experience_replay \
  --gin_bindings=experience_replay.replay_memory.prioritized_replay_buffer.WrappedPrioritizedReplayBuffer.replay_capacity=1000000 \
  --gin_bindings=ElephantDQNAgent.oldest_policy_in_buffer=250000 \
  --gin_bindings="ElephantDQNAgent.replay_scheme='uniform'" \
  --gin_bindings="atari_lib.create_atari_environment.game_name='Breakout'"
