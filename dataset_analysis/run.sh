# Copyright 2019 The Google Research Authors.
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
# Finetune & predict the targets using BERT.
# Save output into easily readable csv format.#

source gbash.sh || exit

DEFINE_bool multilabel true "Whether to perform multilabel classification."
DEFINE_string data_dir "data/model_input" "Directory containing the input data."
DEFINE_string train_fname "train.tsv" "Train filename."
DEFINE_string dev_fname "dev.tsv" "Dev filename."
DEFINE_string test_fname "test.tsv" "Test filename."
DEFINE_string output_dir "checkpoints" "Directory for storing the output (model & predictions)."
DEFINE_string bert_dir "pretrained_BERT/cased_L-12_H-768_A-12" "BERT model directory."
DEFINE_string target_file "data/targets.txt" "File containing list of target categories."
DEFINE_bool do_train true "Whether to perform training & evaluation on dev set."
DEFINE_bool do_predict true "Whether to perform prediction."
DEFINE_bool calculate_metrics true "Whether to calculate performance metrics (valid only if the prediction file has labels)."
DEFINE_int max_length 50 "Maximum sequence length."

gbash::init_google "$@"

if (( FLAGS_do_train ))
then
  echo "Performing training & evaluation..."
  python -m dataset_analysis.bert_classifier \
  --do_train=true \
  --train_fname="${FLAGS_train_fname}" \
  --dev_fname="${FLAGS_dev_fname}" \
  --multilabel="${FLAGS_multilabel}" \
  --data_dir="${FLAGS_data_dir}" \
  --target_file="${FLAGS_target_file}" \
  --vocab_file="${FLAGS_bert_dir}"/vocab.txt \
  --bert_config_file="${FLAGS_bert_dir}"/bert_config.json \
  --init_checkpoint="${FLAGS_bert_dir}"/bert_model.ckpt \
  --max_seq_length="${FLAGS_max_length}" \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=4 \
  --sentiment=1 \
  --entailment=1e-3 \
  --correlation=10 \
  --save_checkpoints_steps=1361 \
  --output_dir="${FLAGS_output_dir}" \
  --do_lower_case=false \
  --alsologtostderr \
  --minloglevel=0
fi

if (( FLAGS_do_predict ))
then
  echo "Performing prediction..."
  python -m dataset_analysis.bert_classifier \
  --do_predict=true \
  --calculate_metrics="${FLAGS_calculate_metrics}" \
  --multilabel="${FLAGS_multilabel}" \
  --data_dir="${FLAGS_dat_dir}" \
  --test_fname="dev.tsv" \
  --target_file="${FLAGS_target_file}" \
  --vocab_file="${FLAGS_bert_dir}"/vocab.txt \
  --bert_config_file="${FLAGS_bert_dir}"/bert_config.json \
  --init_checkpoint="${FLAGS_output_dir}"/ \
  --max_seq_length="${FLAGS_max_length}" \
  --output_dir="${FLAGS_output_dir}" \
  --do_lower_case=false \
  --alsologtostderr \
  --minloglevel=0



