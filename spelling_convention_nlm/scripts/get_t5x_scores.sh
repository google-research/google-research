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


#
# Script mt5 prompt scoring.

# Location where local scripts are located (including this script).
SCRIPT_DIR="."

# Location where t5x was cloned from github.
T5X_DIR="../../t5x"

# Location of finetuned model checkpoint.
CHECKPOINT="../../t5x_data/checkpoint_1000000"

# Location of model vocabulary.
VOCAB="../../t5x_data/mc4.250000.100extra"
  
# Location of file containing various prompts to score, in tsv format.
INFER_FILE="../data/gen_examples.common_only.16k_sub_multiplied_full.tsv"

# Column for source prompt.
SOURCE_COLUMN=0

# Column for target string to score.
TARGET_COLUMN=2
  
# Number of total columns in input TSV file
TOTAL_COLUMNS=12 

# Directory to store inference output.
OUTPUT_DIR="/tmp/mt5_output"

# Batch size for inference. Can be tuned according to available resources.
BATCH_SIZE=1

# TODO: Is gin_search_paths necessary?
python "${T5X_DIR}"/t5x/infer.py \
  --gin_search_paths="${T5X_DIR}" \
  --gin_file="${SCRIPT_DIR}"/scoring.gin \
  --gin.NUM_FIELDS="${TOTAL_COLUMNS}" \
  --gin.SRC_COL="${SOURCE_COLUMN}" \
  --gin.TGT_COL="${TARGET_COLUMN}" \
  --gin.BATCH_SIZE="${BATCH_SIZE}" \
  --gin.DATA_LOCATION=\""${INFER_FILE}"\" \
  --gin.VOCAB_LOCATION=\""${VOCAB}"\" \
  --gin.CHECKPOINT_PATH=\""${CHECKPOINT}"\" \
  --gin.INFER_OUTPUT_DIR=\""${OUTPUT_DIR}"\" \
  --gin.infer.mode=\"score\" \
  --gin.infer.checkpoint_ds_iter=False


