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
set -e
set -x

virtualenv -p python2.7 "$HOME/merl_env"
source "$HOME/merl_env/bin/activate"

DATA_LOCATION="$HOME/projects/data/"
mkdir -p $DATA_LOCATION

pushd $DATA_LOCATION
wget -N https://storage.googleapis.com/merl/wikitable.zip; unzip -o wikitable.zip
popd

pip install -r meta_reward_learning/semantic_parsing/requirements.txt
bash meta_reward_learning/semantic_parsing/run_single.sh  --config mapo --batch_size 50 --n_replay 1 \
--n_steps 20 --n_explore 1 --name debug --lr 1e-3 --test unittest \
--save_every_n 10 --init_buffer full_mapo_iml_buffer --dry_run nodry_run
