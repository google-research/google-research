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

wget -P value_dice/datasets/ https://storage.googleapis.com/gresearch/value_dice/datasets/Ant-v2.npz
wget -P value_dice/datasets/ https://storage.googleapis.com/gresearch/value_dice/datasets/HalfCheetah-v2.npz
wget -P value_dice/datasets/ https://storage.googleapis.com/gresearch/value_dice/datasets/Hopper-v2.npz
wget -P value_dice/datasets/ https://storage.googleapis.com/gresearch/value_dice/datasets/Walker2d-v2.npz

declare -a env_names=("HalfCheetah-v2"  "Hopper-v2"  "Walker2d-v2" "Ant-v2")
declare -a algos=("bc" "dac" "value_dice")

expert_dir="./datasets/"
save_dir="./save/"

for algo in "${algos[@]}"
do
  for env_name in "${env_names[@]}"
  do
    for ((seed=0;seed<10;seed+=1))
    do
      python -m value_dice.train_eval \
        --expert_dir $expert_dir \
        --save_dir $save_dir \
        --algo $algo \
        --env_name $env_name \
        --seed $seed \
        --num_trajectories 1 \
        --alsologtostderr
    done
  done
done
