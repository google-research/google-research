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

# set up the enviroment
python3 -m venv env
source env/bin/activate

pip install -r assessment_plan_modeling/requirements.txt

# run the tests
python3 -m assessment_plan_modeling.ap_parsing.ap_parsing_task_test
python3 -m assessment_plan_modeling.ap_parsing.ap_parsing_utils_test
python3 -m assessment_plan_modeling.ap_parsing.ap_problems_action_items_annotator_test
python3 -m assessment_plan_modeling.ap_parsing.augmentation_lib_test
python3 -m assessment_plan_modeling.ap_parsing.data_lib_test
python3 -m assessment_plan_modeling.ap_parsing.eval_lib_test
python3 -m assessment_plan_modeling.ap_parsing.tokenizer_lib_test
python3 -m assessment_plan_modeling.note_sectioning.note_section_test

# generate data
DATA_DIR="path/to/data"
TFRECORDS_PATH="${DATA_DIR}/ap_parsing_tf_examples/$(date +%Y%m%d)"
python assessment_plan_modeling/ap_parsing/data_gen_main.py \
  --input_note_events="${DATA_DIR}/notes.csv" \
  --input_ratings="${DATA_DIR}/all_model_ratings.csv" \
  --output_path=${TFRECORDS_PATH} \
  --vocab_file="${DATA_DIR}/word_vocab_25K.txt" \
  --section_markers="assessment_plan_modeling/note_sectioning/data/mimic_note_section_markers.json" \
  --n_downsample=100 \
  --max_seq_length=2048

# train model
EXP_TYPE="ap_parsing"
CONFIG_DIR="assessment_plan_modeling/ap_parsing/configs"
MODEL_DIR="${DATA_DIR}/models/model_$(date +%Y%m%d-%H%M)"

TRAIN_DATA="${TFRECORDS_PATH}/train_rated_nonaugmented.tfrecord*"
VAL_DATA="${TFRECORDS_PATH}/val_set.tfrecord*"
PARAMS_OVERRIDE="task.use_crf=true"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},task.train_data.input_path='${TRAIN_DATA}'"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},task.validation_data.input_path='${VAL_DATA}'"
PARAMS_OVERRIDE="${PARAMS_OVERRIDE},trainer.train_steps=5000"

python assessment_plan_modeling/ap_parsing/train.py \
  --experiment=${EXP_TYPE} \
  --config_file="${CONFIG_DIR}/base_ap_parsing_model_config.yaml" \
  --config_file="${CONFIG_DIR}/base_ap_parsing_task_config.yaml" \
  --params_override=${PARAMS_OVERRIDE} \
  --mode=train_and_eval \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr