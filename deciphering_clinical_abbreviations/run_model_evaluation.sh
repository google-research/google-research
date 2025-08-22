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



SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sh "${SCRIPT_DIR}/download_datasets.sh"
sh "${SCRIPT_DIR}/init_conda.sh"

. ~/.bashrc
echo "Entering conda environment."
conda activate "${SCRIPT_DIR}/.conda_env"

echo "Running model evaluation."
python -m deciphering_clinical_abbreviations.run_model_evaluation \
      --abbreviation_dictionary_path="${SCRIPT_DIR}/datasets/abbreviation_expansion_dictionary.csv" \
      --input_data_path="${SCRIPT_DIR}/datasets/synthetic_snippets.csv" \
      --model_outputs_path="${SCRIPT_DIR}/datasets/t5_11b_elic_outputs_synthetic_snippets.csv" \
      --expansion_equivalences_path="${SCRIPT_DIR}/datasets/expansion_equivalencies.csv"
echo "Evaluation complete."