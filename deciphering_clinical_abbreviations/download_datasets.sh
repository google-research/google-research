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

if [ ! -d "${SCRIPT_DIR}/datasets" ]; then
  echo "Creating ${SCRIPT_DIR}/datasets folder."
  mkdir "${SCRIPT_DIR}/datasets"
  else echo "${SCRIPT_DIR}/datasets folder already exists; Skipping folder creation."
fi

declare -a filenames=(
  "abbreviation_expansion_dictionary.csv"
  "synthetic_snippets.csv"
  "t5_11b_elic_outputs_synthetic_snippets.csv"
  "expansion_equivalencies.csv")

for filename in "${filenames[@]}"
do
  if [ ! -f "${SCRIPT_DIR}/datasets/${filename}" ]; then
    gsutil cp "gs://gresearch/deciphering_clinical_abbreviations/${filename}" "${SCRIPT_DIR}/datasets/${filename}"
    else echo "${SCRIPT_DIR}/datasets/${filename} file already exists; Skipping download."
  fi
done
