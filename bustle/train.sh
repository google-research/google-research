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


# This script trains a new BUSTLE model given some training data.

set -e

DIR=$(dirname "$(readlink -f "$0")")
cd "${DIR}"
cd ..  # Go to google-research/ so the imports work out.

TRAINING_DATA_FILE=bustle/training_data/new_training_data.json
OUTPUT_DIR=bustle/models/new_trained_model
EPOCHS=${1:-1000}

python -m bustle.bustle_python.train_model \
  --training_data_file=${TRAINING_DATA_FILE} \
  --output_dir=${OUTPUT_DIR} \
  --epochs=${EPOCHS}
