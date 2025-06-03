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

 -eu

# ========= Input parameters ==========
if [[ $# -ge 1 ]]; then
  dataset=$1
else
  echo "Usage: $0 dataset [loss_weight_method] [class_imbalance] [label]"
  echo "Arguments:"
  echo "  dataset              The name of dataset"
  echo "  loss_weight_method   Weight on the BCE loss (linear, sqrt) (default: linear)"
  echo "  class_imbalance      The ratio of examples (neg/pos) (default: 1.0)"
  echo "  label                (Optional) label for experiment"
  exit 1;
fi

if [[ $# -ge 2 ]]; then
  loss_weight_method=$2
else
  loss_weight_method="linear"
fi

if [[ $# -ge 3 ]]; then
  class_imbalance=$3
else
  class_imbalance=1.0
fi

model="lstm"
data_dir="data/${dataset}"
output_dir="exp/${model}_${dataset}_${loss_weight_method}_${class_imbalance}"
if [[ $# -ge 4 ]]; then
    output_dir="${output_dir}_$4"
fi
train_steps=20000

# Training
python run_cls.py \
    --model ${model} \
    --data_dir ${data_dir} \
    --do_train=true \
    --output_dir ${output_dir} \
    --class_imbalance ${class_imbalance} \
    --loss_weight_method ${loss_weight_method} \
    --train_steps ${train_steps} \
    --checkpoint_iter 1000 \
    --eval_iter 1000 \
    --display_iter 100

# Testing
python run_cls.py \
    --model ${model} \
    --data_dir ${data_dir} \
    --do_train=false \
    --dataset "test" \
    --output_dir ${output_dir} \
    --class_imbalance ${class_imbalance} \
    --loss_weight_method ${loss_weight_method}
