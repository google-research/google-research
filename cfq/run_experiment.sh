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

# The dataset to use (cfq or scan)
dataset="cfq"

# The split of the dataset (random, mcd1, mcd2, mcd3).
split="mcd1"

# Evaluation results will be written to this path.
eval_results_path="evaluation-${dataset}-${split}-${model}.txt"

# Tensor2tensor results will be written to this path. This includes encode/
# decode files, the vocabulary, and the trained models.
save_path="t2t_data/${dataset}/${split}/${model}"

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

python3 -m preprocess_main --dataset="${dataset}" \
  --split="${split}" --save_path="${save_path}"

t2t-datagen --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --tmp_dir="${tmp_path}"

t2t-trainer --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --model="${model}" --hparams_set="${hparams_set}" \
  --output_dir="${output_dir}" --train_steps="${train_steps}"

t2t-decoder --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="${problem}" --model="${model}" --hparams_set="${hparams_set}" \
  --checkpoint_path="${checkpoint_path}" \
  --decode_from_file="${encode_path}" \
  --decode_to_file="${decode_inferred_path}" \
  --output_dir="${output_dir}"

python3 -m evaluate_main --questions_path="${encode_path}" \
  --golden_answers_path="${decode_path}" \
  --inferred_answers_path="${decode_inferred_path}" \
  --output_path="${eval_results_path}"
