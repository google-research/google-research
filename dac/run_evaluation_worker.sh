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

root_dir='/tmp/'
seed=20
env='HalfCheetah-v2'
num_trajs=4
use_gpu=false
learn_absorbing=true

name="lfd_state_action_traj_${num_trajs}_${env}_${seed}"
python3 evaluation_worker.py \
  --log_dir="${root_dir}/${name}/eval" \
  --load_dir="${root_dir}/${name}/eval_save" \
  --use_gpu=${use_gpu} \
  --env=${env} \
  --wrap_for_absorbing=${learn_absorbing}

