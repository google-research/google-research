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
if [[ $# -ge 2 && $# -le 3 ]]; then
  dataset=$1
  num_encoder_layers=$2
else
  echo "Usage: $0 dataset num_encoder_layer [label]"
  echo "Arguments:"
  echo "  dataset              The name of dataset"
  echo "  num_encoder_layers   Number of the encoder layers in Relative Transformer"
  echo "  label                (Optional) label for experiment"
  exit 1;
fi

model="relative_transformer"
data_dir="data/${dataset}"

output_dir="exp/${model}_${dataset}_${num_encoder_layers}layers"
if [[ $# -ge 3 ]]; then
    output_dir="${output_dir}_$3"
fi

# Default parameters
use_attention_mask="false"
use_relative_attention="false"
parse_tree_attention="false"
block_attention="false"
entity_cross_link="false"
block_attention_sep="false"
restart_query_pos="false"
unique_structure_token_pos="false"
train_pos_embed="true"
share_pos_embed="false"
loss_weight_method="linear"
class_imbalance=1.0
cross_link_exact="true"

batch_size=112
if [[ $1 == "random_random" ]]; then
  train_steps=10000
elif [[ $1 == "mcd1_random" ]]; then
  train_steps=10000
elif [[ $1 == "mcd1_model" ]]; then
  train_steps=100000
elif [[ $1 == "mcd1_symmetric_random" ]]; then
  train_steps=10000
elif [[ $1 == "mcd1_symmetric_model" ]]; then
  train_steps=200000
else
  train_steps=200000
fi
learning_rate=0.001
checkpoint_iter=2000
eval_iter=2000

# Training
python run_cls.py \
    --model ${model} \
    --data_dir ${data_dir} \
    --parse_tree_input=False \
    --num_encoder_layers ${num_encoder_layers} \
    --use_attention_mask=${use_attention_mask} \
    --use_relative_attention=${use_relative_attention} \
    --parse_tree_attention=${parse_tree_attention} \
    --block_attention=${block_attention} \
    --block_attention_sep=${block_attention_sep} \
    --entity_cross_link=${entity_cross_link} \
    --cross_link_exact=${cross_link_exact} \
    --restart_query_pos=${restart_query_pos} \
    --unique_structure_token_pos=${unique_structure_token_pos} \
    --learned_position_encoding=${train_pos_embed} \
    --share_pos_embed=${share_pos_embed} \
    --do_train=true \
    --output_dir ${output_dir} \
    --class_imbalance ${class_imbalance} \
    --loss_weight_method ${loss_weight_method} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --train_steps ${train_steps} \
    --checkpoint_iter ${checkpoint_iter} \
    --eval_iter ${eval_iter} \
    --display_iter 100

# Testing
python run_cls.py \
    --model ${model} \
    --data_dir ${data_dir} \
    --parse_tree_input=False \
    --num_encoder_layers ${num_encoder_layers} \
    --use_attention_mask=${use_attention_mask} \
    --use_relative_attention=${use_relative_attention} \
    --parse_tree_attention=${parse_tree_attention} \
    --block_attention=${block_attention} \
    --block_attention_sep=${block_attention_sep} \
    --entity_cross_link=${entity_cross_link} \
    --cross_link_exact=${cross_link_exact} \
    --restart_query_pos=${restart_query_pos} \
    --unique_structure_token_pos=${unique_structure_token_pos} \
    --learned_position_encoding=${train_pos_embed} \
    --share_pos_embed=${share_pos_embed} \
    --do_train=false \
    --dataset "test" \
    --output_dir ${output_dir} \
    --class_imbalance ${class_imbalance} \
    --loss_weight_method ${loss_weight_method} \
    --batch_size ${batch_size}
