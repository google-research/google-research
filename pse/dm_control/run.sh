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

virtualenv -p python3 .
source ./bin/activate

# Install tf-agents
pip install --user tf-agents[reverb]

git clone https://github.com/google_research/distracting_control.git
pip install -e distracting_control
pip install -r pse.dm_control.requirements.txt

python -m pse.dm_control.run_train_eval --trial_id=1 --seed 0\
 --env_name=cartpole-swingup --root_dir=/tmp/drq --num_train_steps=10000 \
 --eval_interval=25 --policy_save_interval=50 --checkpoint_interval=100 \
 --gin_bindings="load_dm_env_for_eval.action_repeat=8" \
 --contrastive_loss_weight=0 \
 --gin_bindings="drq_agent.train_eval.initial_collect_steps=300" \
 --gin_bindings="drq_agent.train_eval.eval_episodes_per_run=1" \
 --alsologtostderr
