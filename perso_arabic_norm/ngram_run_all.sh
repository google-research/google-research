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

 -x
#
# Simple word n-gram training script.

EXPERIMENT=reading/01/baselines  # For words.
NUM_TRIALS=100
BASE_DIR=/usr/local/perso_arabic/wiki
EXPERIMENT_DIR=${BASE_DIR}/experiments/${EXPERIMENT}
OUTPUT_DIR=${BASE_DIR}/results/${EXPERIMENT}

declare -a ORDERS=(2 3 4)
NUM_ORDERS=${#ORDERS[@]}

declare -a LANGUAGES=("ks" "pnb" "sd" "ur")
NUM_LANGUAGES=${#LANGUAGES[@]}

for (( i=0; i<${NUM_ORDERS}; i++ )) ; do
  ORDER=${ORDERS[$i]}
  for (( j=0; j<${NUM_LANGUAGES}; j++ )) ; do
    LANGUAGE=${LANGUAGES[$j]}
    python ngram_train_and_eval.py \
      --corpus_file ${EXPERIMENT_DIR}/${LANGUAGE}.txt.bz2 \
      --line_diffs_file ${EXPERIMENT_DIR}/${LANGUAGE}_line_diffs.pickle \
      --order=${ORDER} \
      --num_trials ${NUM_TRIALS} \
      --word_models \
      --output_model_dir ${OUTPUT_DIR}
  done
done
