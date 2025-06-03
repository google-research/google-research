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



# This script generates training data for the BUSTLE model by running many
# enumerative searches. This might take a long time! Consider running within a
# screen session.

set -e

NUM_SEARCHES=${1:-1000}

DIR=$(dirname "$(readlink -f "$0")")
cd "${DIR}"

DATA_DIR=${DIR}/training_data
MODELS_DIR=${DIR}/models
mkdir -p ${DATA_DIR}

# The filename for this dataset.
DATASET_NAME="new_training_data"
# If desired, the old model to use in the search that generates new data.
OLD_MODEL_NAME="1000_epochs__64_64__e2__1e-5"

java -cp ".:lib/*" com.googleresearch.bustle.GenerateData \
--num_searches=${NUM_SEARCHES} \
--num_values_per_search=100 \
--training_data_file=${DATA_DIR}/${DATASET_NAME}.json \
--time_limit=100 \
--max_expressions=1000000 \
--heuristic_reweighting=true \
--model_reweighting=false \
--model_directory=${DATA_DIR}/${OLD_MODEL_NAME} \
--premise_selection=false
