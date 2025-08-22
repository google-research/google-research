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



# --- Configuration ---
PYTHON_MODULE_PATH="causal_evaluation.experiments.evaluate"
DATA_DIRECTORY="./../../data/acs_pums"
N_RESAMPLES=1000
LABEL_COL_NAME="labels"
GROUP_COL_NAME="group"
TASK_COL_NAME_VALUE="task"

# Suffix for prediction and result files
# (The part common to both pred and result files after the task name)
FILE_SUFFIX_PARTS="_5-Year_2018_gradient_boosting.parquet"

# Array of tasks to iterate through
TASKS=(
    "ACSIncome"
    "ACSPublicCoverage"
)

for current_task in "${TASKS[@]}"; do
    echo "--------------------------------------------------------------------"
    echo "Running evaluation for task: $current_task"
    echo "--------------------------------------------------------------------"

    PRED_FILE_NAME="preds_${current_task}${FILE_SUFFIX_PARTS}"
    RESULT_FILE_NAME="metrics_${current_task}${FILE_SUFFIX_PARTS}"

    python -m "${PYTHON_MODULE_PATH}" \
        --data_directory="${DATA_DIRECTORY}" \
        --task="${current_task}" \
        --n_resamples="${N_RESAMPLES}" \
        --pred_file_name="${PRED_FILE_NAME}" \
        --result_file_name="${RESULT_FILE_NAME}" \
        --label_col_name="${LABEL_COL_NAME}" \
        --group_col_name="${GROUP_COL_NAME}" \
        --task_col_name="${TASK_COL_NAME_VALUE}"

    echo "Finished evaluation for task: $current_task"
    echo ""
done

exit 0