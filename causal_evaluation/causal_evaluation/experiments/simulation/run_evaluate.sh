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

# Common arguments
DATA_DIRECTORY="./../../data/simulation"
N_RESAMPLES=1000
NUM_SAMPLES_TRAIN=50000
NUM_SAMPLES_EVAL=20000
LABEL_COL_NAME="y"
GROUP_COL_NAME="a"
TASK_COL_NAME="setting"

# Suffix for prediction and result files
FILE_SUFFIX_PARTS="_${NUM_SAMPLES_TRAIN}_${NUM_SAMPLES_EVAL}_gradient_boosting_gradient_boosting.parquet"

# Array of settings to iterate through
SETTINGS=(
    "covariate_shift"
    "outcome_shift"
    "complex_causal_shift"
    "low_overlap_causal"
    "anticausal_label_shift"
    "anticausal_presentation_shift"
    "complex_anticausal_shift"
)

for setting in "${SETTINGS[@]}"; do
    echo "--------------------------------------------------------------------"
    echo "Running evaluation for setting: $setting"
    echo "--------------------------------------------------------------------"

    PRED_FILE_NAME="sim_samples_eval_${setting}${FILE_SUFFIX_PARTS}"
    RESULT_FILE_NAME="metrics_${setting}${FILE_SUFFIX_PARTS}"

    # Construct the command arguments for the Python script
    # Ensure 'python' or 'python3' is the correct command for your environment
    python -m "${PYTHON_MODULE_PATH}" \
        --data_directory="${DATA_DIRECTORY}" \
        --task="${setting}" \
        --n_resamples="${N_RESAMPLES}" \
        --pred_file_name="${PRED_FILE_NAME}" \
        --result_file_name="${RESULT_FILE_NAME}" \
        --label_col_name="${LABEL_COL_NAME}" \
        --group_col_name="${GROUP_COL_NAME}" \
        --task_col_name="${TASK_COL_NAME}"

    echo "Finished evaluation for setting: $setting"
    echo ""
done

exit 0