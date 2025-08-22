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


set -e
set -x

virtualenv -p python3 .
source ./bin/activate
pip install -r requirements.txt

export OUTPUT_DIR=redace/output
mkdir -p "${OUTPUT_DIR}"

export TRAIN_FILE=redace/train_other.json
export DEV_FILE=redace/dev_other.json
export TEST_FILE=redace/test_other.json

# See https://github.com/google-research/bert
export CHECKPOINT=redace/bert_model.ckpt
export VOCAB_FILE=redace/vocab.txt

echo "Preprocessing training data"
python -m preprocess_main \
  --input_file="${TRAIN_FILE}" \
  --output_file="${OUTPUT_DIR}/train.tfrecord" \
  --vocab_file="${VOCAB_FILE}"

echo "Preprocessing dev data"
python -m preprocess_main \
  --input_file="${DEV_FILE}" \
  --output_file="${OUTPUT_DIR}/dev.tfrecord" \
  --vocab_file="${VOCAB_FILE}"
  
echo "Train"
python -m run_redace \
  --train_file="${OUTPUT_DIR}/train.tfrecord" \
  --eval_file="${OUTPUT_DIR}/dev.tfrecord" \
  --model_dir="${OUTPUT_DIR}/redace_model" \
  --init_checkpoint="${CHECKPOINT}"

echo "Predict"
python -m predict_main \
   --predict_input_file="${TEST_FILE}" \
   --predict_output_file="${OUTPUT_DIR}/prediction.json" \
   --vocab_file="${VOCAB_FILE}" \
   --model_dir="${OUTPUT_DIR}/redace_model"