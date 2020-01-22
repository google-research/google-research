# Copyright 2020 The Google Research Authors.
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

#!/bin/bash -eu
# ================= Input parameters ================

# Models from our paper: lstm_seq2seq_attention, transformer,
# universal_transformer. Other models (see subdirectories as well):
# https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models
model="lstm_seq2seq_attention"

# Custom hyperparameters are defined in cfq/cfq.py. You can select tensor2tensor
# default parameters as well.
hparams_set="cfq_lstm_attention_multi"

# We report experiments with 35,000 steps in our paper.
train_steps="35000"

# URL to the CFQ dataset.
dataset_url="https://storage.cloud.google.com/cfq_dataset/cfq.tar.gz"

# Local path to the dataset (after it has been downloaded).
dataset_local_path="dataset.json"

# Location of the dataset split to run the experiment for.
split_path="splits/random_split.json"

# Evaluation results will be written to this path.
eval_results_path="evaluation.txt"

# Tensor2tensor results will be written to this path. This includes encode/
# decode files, the vocabulary, and the trained models.
save_path="t2t_data"

# The tensor2tensor problem to use. The cfq problem is defined in cfq/cfq.py.
problem="cfq"

# Other path-related variables.
tmp_path="/tmp/cfq_tmp"
work_dir="$(pwd)"
output_dir="${save_path}/output"
checkpoint_path="${save_path}/output/model.ckpt-${train_steps}"
# We evaluate the trained model on the dev split of the dataset.
encode_path="${save_path}/dev/dev_encode.txt"
decode_path="${save_path}/dev/dev_decode.txt"
decode_inferred_path="${save_path}/dev/dev_decode_inferred.txt"

# ================= Pipeline ================
# Download dataset if it doesn't exist yet.
if [[ ! -f "${dataset_local_path}" || ! -f "${split_path}" ]]; then
  echo "ERROR: Dataset not found."
  echo "Please download the dataset first from ${dataset_url}!"
  echo "See further instructions in the README."
  exit 1
fi

python3 -m preprocess_dataset --dataset_path="${dataset_local_path}" \
  --split_path="${split_path}" --save_path="${save_path}"

t2t-datagen --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --tmp_dir="${tmp_path}"

t2t-trainer --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --model="${model}" --hparams_set="${hparams_set}" \
  --output_dir="${output_dir}" --train_steps="${train_steps}"

t2t-decoder --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --model="${model}" --hparams_set="${hparams_set}" \
  --checkpoint_path="${checkpoint_path}" \
  --decode_from_file="${encode_path}" \
  --decode_to_file="${decode_inferred_path}"

python3 -m evaluate --questions_path="${encode_path}" \
  --golden_answers_path="${decode_path}" \
  --inferred_answers_path="${decode_inferred_path}" \
  --output_path="${eval_results_path}"
