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

pip install -r behavior_regularized_offline_rl/requirements.txt
python -m behavior_regularized_offline_rl.brac.train_online --logtostderr --sub_dir=0 --env_name=Pendulum-v0 --eval_target=-150 --agent_name=sac --total_train_steps=100 --n_eval_episodes=1
