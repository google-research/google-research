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

while getopts e:p:i: flag
do
    case "${flag}" in
        e) embodiment=${OPTARG};;
        i) iters=${OPTARG};;
    esac
done

ENV_NAME="SweepToTop-"${embodiment^}"-State-Allo-TestLayout-v0"
NAME="env_reward_$embodiment"

for seed in {0..2}
do
    python train_policy.py \
        --experiment_name $NAME \
        --env_name $ENV_NAME \
        --seed=$seed \
        --device="cuda:0" \
        --config.num_train_steps=$iters \
        &
done

for seed in {3..4}
do
    python train_policy.py \
        --experiment_name $NAME \
        --env_name $ENV_NAME \
        --seed=$seed \
        --device="cuda:1" \
        --config.num_train_steps=$iters \
        &
done
