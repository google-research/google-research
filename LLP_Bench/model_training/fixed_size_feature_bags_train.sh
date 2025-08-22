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


declare -a c1_c2_list=(
  '1 7' '1 10' '2 7' '2 10' '2 11' '2 13' '3 7' '3 10'
  '3 11' '3 13' '4 7' '4 10' '4 11' '4 13' '4 15' '6 7'
  '6 10' '7 8' '7 10' '7 12' '7 14' '7 15' '7 16' '7 18'
  '7 20' '7 21' '7 24' '7 26' '10 12' '10 14' '10 15'
  '10 16' '10 17' '10 18' '10 20' '10 21' '10 24' '10 26'
  '11 12' '11 15' '11 16' '11 18' '11 21' '11 24' '11 26'
  '12 13' '13 15' '13 16' '13 18' '13 21' '13 24' '13 26'
)

declare -a method_list=(
  'dllp_bce' 'dllp_mse' 'genbags' 'easy_llp' 'ot_llp' 'soft_erot_llp' 'hard_erot_llp' 'sim_llp' 'mean_map'
)

for c1_c2 in "${c1_c2_list[@]}"; do
  read -a strarr <<< "$c1_c2"  # uses default whitespace IFS
  # python feature_bag_ds_creation.py --c1=${strarr[0]} --c2=${strarr[1]}
  for method in "${method_list[@]}"
  do
    for split_no in $(seq 0 4)
    do
      for bag_size in 64 128 256 512
      do
        python3 model_training/train.py --c1=${strarr[0]} --c2=${strarr[1]} --split=$split_no --method=$method --bag_size=$bag_size --feature_random_bags=True
      done
    done
  done
done