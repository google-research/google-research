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
sh -i "${SCRIPT_DIR}/init_conda.sh"

. ~/.bashrc
echo "Entering conda environment."
conda activate "${SCRIPT_DIR}/.conda_env"

if [ ! -d "${SCRIPT_DIR}/datasets/reverse_substituted_fake" ]; then
  mkdir "${SCRIPT_DIR}/datasets/reverse_substituted_fake"
  else echo "${SCRIPT_DIR}/datasets/reverse_substituted_fake already exists; Skipping creation."
fi

echo "Running reverse substitution."
python -m deciphering_clinical_abbreviations.run_reverse_substitution \
      --document_dataset_path="${SCRIPT_DIR}/datasets/fake_document_dataset.csv" \
      --abbreviation_dictionary_path="${SCRIPT_DIR}/datasets/abbreviation_expansion_dictionary.csv" \
      --save_filepath="${SCRIPT_DIR}/datasets/reverse_substituted_fake" \
      --expected_replacements_per_expansion=5 \
      --min_snippets_per_substitution_pair=2 \
      --random_seed=1
echo "Reverse substitution complete. Result is located at ${SCRIPT_DIR}/datasets/reverse_substituted_fake"
