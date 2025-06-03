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

if [[ "$#" != 4 ]]; then
  echo "Usage: $0 cfq_split neg_method train_hold_out output_tree"
  echo "Possible arguments are as following:"
  echo "  cfq_split       random or mcd1"
  echo "  neg_method      random or model"
  echo "  train_hold_out  true or false"
  echo "  output_tree     tree or notree"
  exit 1;
fi

# ========= Input parameters ==========
# Path to the CFQ dataset extracted
cfq_root="./cfq"

# URL to the CFQ dataset.
dataset_url="https://storage.cloud.google.com/cfq_dataset/cfq.tar.gz"

# Name of the dataset split file. It will be located in ${cfq_root}/splits.
if [[ "$1" == "random" ]]; then
  split_file="random_split.json"
elif [[ "$1" == "mcd1" ]]; then
  split_file="mcd1.json"
elif [[ "$1" == "mcd1_symmetric" ]]; then
  split_file="mcd1_symmetric.json"
else
  echo "Wrong cfq_split: $1"
  exit 1;
fi

# Negative data generation method and options
if [[ "$2" == "random" ]]; then
  negative_example="random"
elif [[ "$2" == "model" ]]; then
  negative_example="model_output"
  model_output_dir="cfq_model_outputs/$1"
  max_neg=3
  sort_by_score=false
else
  echo "Wrong neg_method: $2"
  exit 1;
fi

# Hold out train options
train_hold_out="$3"
if [[ "$train_hold_out" != "true" && "$train_hold_out" != "false" ]]; then
  echo "Wrong train_hold_out: $train_hold_out"
  exit 1;
fi

# Tree output options
output_tree="$4"
if [[ "$output_tree" == "tree" ]]; then
  script_fname="create_cls_tree_dataset.py"
  output_dir="../data/$1_$2_tree"
elif [[ "$output_tree" == "notree" ]]; then
  script_fname="create_cls_dataset.py"
  output_dir="../data/$1_$2"
else
  echo "Wrong output_tree: $output_tree"
  exit 1;
fi


# Check the CFQ dataset
if [[ ! -d "${cfq_root}" || ! -f "${cfq_root}/dataset.json" ]]; then
  # TODO: Automatic download script
  echo "CFQ dataset not found."
  echo "Please download the dataset from (${dataset_url})"
  echo "and extract in this directory. The dataset file should be located at (${cfq_root}/dataset.json)."
  exit 1
fi

# Generate dataset!
if [[ "$2" == "random" ]]; then
  python $script_fname --cfq_root $cfq_root \
    --split_file $split_file \
    --output_dir $output_dir \
    --negative_example $negative_example \
    --train_hold_out=$train_hold_out
elif [[ "$2" == "model" ]]; then
  python $script_fname --cfq_root $cfq_root \
    --split_file $split_file \
    --output_dir $output_dir \
    --negative_example $negative_example \
    --model_output_dir $model_output_dir \
    --max_neg $max_neg \
    --sort_by_score=$sort_by_score \
    --train_hold_out=$train_hold_out
fi
