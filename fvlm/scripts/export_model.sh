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

INPUT_DIR=$1
OUTPUT_DIR=$2
FILE_PATH="./export_saved_model.py"
MODEL_NAME="resnet_50"

python3 "${FILE_PATH}" \
 --input_dir="${INPUT_DIR}" \
 --output_dir="${OUTPUT_DIR}" \
 --model_name="${MODEL_NAME}"
