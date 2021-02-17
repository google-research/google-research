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
# Please update these paths.
export OUTPUT_DIR=/path/to/output
# BERT can be found at https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3
export BERT_BASE_DIR=/path/to/cased_L-12_H-768_A-12
# DiscoFuse can be found at https://github.com/google-research-datasets/discofuse
export DISCOFUSE_DIR=/path/to/discofuse
export PREDICTION_FILE=${OUTPUT_DIR}/pred.tsv

# If False FelixInsert is used.
export USE_POINTING='True'


# Label map construction
echo "Constructing vocabulary"
python phrase_vocabulary_constructor_main \
--output="${OUTPUT_DIR}/label_map.json" \
--use_pointing="${USE_POINTING}" \
--do_lower_case="True"

# Preprocess
echo "Preprocessing data"
python preprocess_main \
  --input_file="${DISCOFUSE_DIR}/train.tsv" \
  --input_format="wikisplit" \
  --output_file="${OUTPUT_DIR}/train.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${BERT_BASE_DIR}/assets/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --max_seq_length="128" \
  --use_pointing="${USE_POINTING}" \
  --split_on_punc="True"

python preprocess_main.py \
  --input_file="${DISCOFUSE_DIR}/tune.tsv" \
  --input_format="wikisplit" \
  --output_file="${OUTPUT_DIR}/tune.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${BERT_BASE_DIR}/assets/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --max_seq_length="128" \
  --use_pointing="${USE_POINTING}" \
  --split_on_punc="True"

# Train
echo "Training tagging model"
rm -rf "${OUTPUT_DIR}/model_tagging"
mkdir -p "${OUTPUT_DIR}/model_tagging"

python run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord" \
    --eval_file="${OUTPUT_DIR}/tune.tfrecord" \
    --model_dir_tagging="${OUTPUT_DIR}/model_tagging" \
    --bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="32" \
    --eval_batch_size="32" \
    --log_steps="100" \
    --steps_per_loop="100" \
    --train_insertion="False" \
    --use_pointing="${USE_POINTING}" \
    --learning_rate="0.0005" \
    --pointing_weight="1" \
    --use_weighted_labels="True"

echo "Training insertion model"
rm -rf "${DATA_DIRECTORY}/model_insertion"
mkdir "${DATA_DIRECTORY}/model_insertion"
python run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --eval_file="${OUTPUT_DIR}/tune.tfrecord.ins" \
    --model_dir_insertion="${OUTPUT_DIR}/model_insertion" \
    --bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="32" \
    --eval_batch_size="32" \
    --log_steps="100" \
    --steps_per_loop="100" \
    --train_insertion="False" \
    --use_pointing="${USE_POINTING}" \
    --learning_rate="0.0005" \
    --pointing_weight="1" \
    --train_insertion="True"

# Predict
echo "Generating predictions"

python predict_main \
--input_format="wikisplit" \
--predict_input_file="${DISCOFUSE_DIR}/test.tsv" \
--predict_output_file="${PREDICTION_FILE}"\
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${BERT_BASE_DIR}/assets/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=32 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
--bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="${USE_POINTING}"
