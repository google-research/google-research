# Copyright 2019 The Google Research Authors.
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

# NIS

#python -m eim.small_problems \
#  --target=nine_gaussians \
#  --algo=nis \
#  --K=1024 \
#  --batch_size=128 \
#  --learning_rate=3e-4 \
#  --energy_fn_sizes=20,20 \
#  --proposal_variance=0.01

# HIS

#python -m eim.small_problems \
#  --target=nine_gaussians \
#  --algo=his \
#  --his_t=5 \
#  --his_learn_stepsize \
#  --his_learn_alpha \
#  --batch_size=128 \
#  --learning_rate=3e-4 \
#  --energy_fn_sizes=20,20 \
#  --proposal_variance=0.01 

# TRS

python -m eim.small_problems \
  --target=nine_gaussians \
  --algo=rejection_sampling \
  --K=1024 \
  --batch_size=128 \
  --learning_rate=3e-4 \
  --energy_fn_sizes=20,20 \
  --proposal_variance=0.01

# LARS

#python -m eim.small_problems \
#  --target=nine_gaussians \
#  --algo=lars \
#  --K=1024 \
#  --batch_size=128 \
#  --learning_rate=3e-4 \
#  --energy_fn_sizes=20,20 \
#  --proposal_variance=0.01
