# Copyright 2020 The Google Research Authors.
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

# Save current path.
CURR_DIR=$PWD

# Download the data to the same locations with the current script.
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_DIR="${BASE_DIR}/data"
mkdir -p "${DATA_DIR}"
mkdir -p "${DATA_DIR}/wmt19_metric_task_results"
curl -o "${DATA_DIR}/annotations.zip" http://storage.googleapis.com/gresearch/kobe/data/annotations.zip
curl -o "${DATA_DIR}/wmt19_metric_task_results/sys-level_scores_metrics.csv" http://storage.googleapis.com/gresearch/kobe/data/wmt19_metric_task_results/sys-level_scores_metrics.csv

# Unzip the annotations.
cd "${DATA_DIR}"
unzip "${DATA_DIR}/annotations.zip"

# Return to the saved path.
cd "${CURR_DIR}"

# Run the python script to reproduce results from the paper.
# The script will calculate the KoBE scores for all language pairs and their
# correlations with human judgements, and then print the results as reported in
# the paper.
python "${BASE_DIR}/eval_main.py" "${DATA_DIR}"
