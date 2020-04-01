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

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

save_path="test_output"
output_dir="${save_path}/output"
work_dir=$(pwd)
train_step=800
checkpoint_path="${save_path}/output/model.ckpt-${train_steps}"

encode_path="${save_path}/dev/dev_encode.txt"
decode_path="${save_path}/dev/dev_decode.txt"
decode_inferred_path="${save_path}/dev/dev_decode_inferred.txt"

pip3 install -r requirements.txt
python3 -m preprocess_main --dataset_path="google/example_data/dataset.json" \
  --split_path="google/example_data/test_split.json" --save_path="${save_path}"

t2t-datagen --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="cfq" --tmp_dir="tmp_dir"

t2t-trainer --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="cfq" --model="lstm_seq2seq_attention" \
  --hparams_set="cfq_lstm_attention_multi" --output_dir="${output_dir}" \
  --train_steps=800

t2t-decoder --t2t_usr_dir="${work_dir}/cfq/" --data_dir="${save_path}" \
  --problem="cfq" --model="lstm_seq2seq_attention" \
  --hparams_set="cfq_lstm_attention_multi" \
  --checkpoint_path="${checkpoint_path}" \
  --decode_from_file="${encode_path}" \
  --decode_to_file="${decode_inferred_path}"

python3 -m evaluate_main --questions_path="${encode_path}" \
  --golden_answers_path="${decode_path}" \
  --inferred_answers_path="${decode_inferred_path}" \
  --output_path="evaluation.txt"
