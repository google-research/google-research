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
set -x


virtualenv -p python3 .
source ./bin/activate
pip install -r felix/requirements.txt

echo "Running tests"

python -m felix.beam_search_test
python -m felix.bert_example_test
python -m felix.felix_models_test
python -m felix.felix_tagger_test
python -m felix.insertion_converter_test
python -m felix.pointing_converter_test
python -m felix.predict_test
python -m felix.utils_test
python -m felix.converter_for_felix_insert_test
python -m felix.example_builder_for_felix_insert_test


export OUTPUT_DIR=felix/test
mkdir -p "${OUTPUT_DIR}"
export DISCOFUSE_DIR=felix/felix_code_test
export PREDICTION_FILE="${DISCOFUSE_DIR}/train.tsv.pred"

# Label map construction
echo "Constructing vocabulary"
python -m felix.phrase_vocabulary_constructor_main \
--output="${OUTPUT_DIR}/label_map.json" \
--do_lower_case="True"

# Preprocess
echo "Preprocessing data"
python -m felix.preprocess_main \
  --input_file="${DISCOFUSE_DIR}/train.tsv" \
  --input_format="wikisplit" \
  --output_file="${OUTPUT_DIR}/train.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${DISCOFUSE_DIR}/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --max_seq_length="128" \
  --split_on_punc="True"

# Train
echo "Training tagging model"
rm -rf "${OUTPUT_DIR}/model_tagging"
mkdir -p "${OUTPUT_DIR}/model_tagging"
python -m felix.run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord" \
    --eval_file="${OUTPUT_DIR}/train.tfrecord" \
    --model_dir_tagging="${OUTPUT_DIR}/model_tagging" \
    --bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="8" \
    --eval_batch_size="8" \
    --log_steps="10" \
    --steps_per_loop="5" \
    --train_insertion="False" \
    --use_pointing="True" \
    --learning_rate="0.005" \
    --pointing_weight="1" \
    --use_weighted_labels="True"

echo "Training insertion model"
rm -rf "${OUTPUT_DIR}/model_insertion"
mkdir "${OUTPUT_DIR}/model_insertion"
python -m felix.run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --eval_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --model_dir_insertion="${OUTPUT_DIR}/model_insertion" \
    --bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="8" \
    --eval_batch_size="8" \
    --log_steps="10" \
    --steps_per_loop="5" \
    --train_insertion="False" \
    --use_pointing="True" \
    --learning_rate="0.005" \
    --pointing_weight="1" \
    --train_insertion="True"

# Predict
echo "Generating predictions"
python -m felix.predict_main \
--input_format="wikisplit" \
--predict_input_file="${DISCOFUSE_DIR}/train.tsv" \
--predict_output_file="${PREDICTION_FILE}" \
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${DISCOFUSE_DIR}/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=4 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
--bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="True"


# FelixInsert
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
# Label map construction
echo "Constructing vocabulary"
python -m felix.phrase_vocabulary_constructor_main \
  --output="${OUTPUT_DIR}/label_map.json" \
  --use_pointing="False" \
  --do_lower_case="True"

# Preprocess
echo "Preprocessing data"
python -m felix.preprocess_main \
  --input_file="${DISCOFUSE_DIR}/train.tsv" \
  --input_format="wikisplit" \
  --output_file="${OUTPUT_DIR}/train.tfrecord" \
  --label_map_file="${OUTPUT_DIR}/label_map.json" \
  --vocab_file="${DISCOFUSE_DIR}/vocab.txt" \
  --do_lower_case="True" \
  --use_open_vocab="True" \
  --use_pointing="False" \
  --max_seq_length="128" \
  --split_on_punc="True"

# Train
echo "Training tagging model"
rm -rf "${OUTPUT_DIR}/model_tagging"
mkdir -p "${OUTPUT_DIR}/model_tagging"

python -m felix.run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord" \
    --eval_file="${OUTPUT_DIR}/train.tfrecord" \
    --model_dir_tagging="${OUTPUT_DIR}/model_tagging" \
    --bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="8" \
    --eval_batch_size="8" \
    --log_steps="10" \
    --steps_per_loop="5" \
    --train_insertion="False" \
    --use_pointing="False" \
    --learning_rate="0.005" \
    --pointing_weight="1" \
    --use_weighted_labels="True"

echo "Training insertion model"
rm -rf "${OUTPUT_DIR}/model_insertion"
mkdir "${OUTPUT_DIR}/model_insertion"
python -m felix.run_felix \
    --train_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --eval_file="${OUTPUT_DIR}/train.tfrecord.ins" \
    --model_dir_insertion="${OUTPUT_DIR}/model_insertion" \
    --bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
    --max_seq_length=128 \
    --num_train_epochs=500 \
    --num_train_examples=8 \
    --num_eval_examples=8 \
    --train_batch_size="8" \
    --eval_batch_size="8" \
    --log_steps="10" \
    --steps_per_loop="5" \
    --train_insertion="False" \
    --use_pointing="False" \
    --learning_rate="0.005" \
    --pointing_weight="1" \
    --train_insertion="True"

# Predict
echo "Generating predictions"
python -m felix.predict_main \
--input_format="wikisplit" \
--predict_input_file="${DISCOFUSE_DIR}/train.tsv" \
--predict_output_file="${PREDICTION_FILE}" \
--label_map_file="${OUTPUT_DIR}/label_map.json" \
--vocab_file="${DISCOFUSE_DIR}/vocab.txt" \
--max_seq_length=128 \
--predict_batch_size=4 \
--do_lower_case="True" \
--use_open_vocab="True" \
--bert_config_tagging="${DISCOFUSE_DIR}/felix_config.json" \
--bert_config_insertion="${DISCOFUSE_DIR}/felix_config.json" \
--model_tagging_filepath="${OUTPUT_DIR}/model_tagging" \
--model_insertion_filepath="${OUTPUT_DIR}/model_insertion" \
--use_pointing="False"






