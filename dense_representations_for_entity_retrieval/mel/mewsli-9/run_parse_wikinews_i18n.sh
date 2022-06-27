# Copyright 2022 The Google Research Authors.
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

# Run custom parser to extract entity linked data from WikiNews.
#
# Usage: run_wikinewsi18_parser.sh [input_base_dir] [output_base_dir]

set -eux

INPUT_BASE_DIR_DEFAULT="$(dirname $0)/output/wikiextractor"
INPUT_BASE_DIR="${1:-${INPUT_BASE_DIR_DEFAULT}}"

OUTPUT_BASE_DIR_DEFAULT="$(dirname $0)/output/dataset"
OUTPUT_BASE_DIR="${2:-${OUTPUT_BASE_DIR_DEFAULT}}"

LANG_LIST=(ar de en es fa ja sr ta tr)

MODULE="dense_representations_for_entity_retrieval.mel.wikinews_extractor.parse_wikinews_i18n"

# Point PYTHONPATH to the google_research/ directory.
export PYTHONPATH="${PYTHONPATH}:$(dirname $0)/../../../"

for lang in ${LANG_LIST[@]}; do
  date
  echo ">Parse '${lang}'..."

  output_dir="${OUTPUT_BASE_DIR}/${lang}"
  mkdir -p ${output_dir}

  # Run each language's process in background.
  python -m ${MODULE} \
      --mode=dataset \
      --wikinews_archive="${INPUT_BASE_DIR}/${lang}/*/wiki_*.bz2" \
      --language=${lang} \
      --mention_index_path=${output_dir}/mentions.tsv \
      --doc_index_path=${output_dir}/docs.tsv \
      --output_dir_wiki=${output_dir}/wiki \
      --output_dir_text=${output_dir}/text \
      --logtostderr &> ${output_dir}/log &
  echo
done

wait

date
echo ">Done: ${OUTPUT_BASE_DIR}"
