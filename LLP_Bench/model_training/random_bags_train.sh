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


declare -a method_list=(
  'dllp_bce' 'dllp_mse' 'genbags' 'easy_llp' 'ot_llp' 'soft_erot_llp' 'hard_erot_llp' 'sim_llp' 'mean_map'
)

for split_no in $(seq 0 4)
do
  for bag_size in 64 128 256 512
  do
    for method in "${method_list[@]}"
    do
      python3 model_training/train.py --method=$method --split=$split_no --bag_size=$bag_size --random_bags=True --which_dataset=criteo_ctr
    done
  done
done

declare -a method_list_sscl=(
  'dllp_mse' 'dllp_mae' 'genbags' 'sim_llp'
)

for split_no in $(seq 0 4)
do
  for bag_size in 64 128 256 512
  do
    for method in "${method_list_sscl[@]}"
    do
      python3 model_training/train.py --method=$method --split=$split_no --bag_size=$bag_size --random_bags=True --which_dataset=criteo_sscl
    done
  done
done
