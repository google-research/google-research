# Copyright 2021 The Google Research Authors.
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

# Run wikiextractor on a WikiNews dump.

# Usage: run_wikiextractor.sh \
#     [wikiextractor_bin] [template_dir] [input_dump_dir] [output_base_dir]
#
# (Pass all args or none of them.)

set -eux

BIN_DEFAULT="$(dirname $0)/../tools/wikiextractor_repo/WikiExtractor.py"
WIKIEXTRACTOR="${1:-${BIN_DEFAULT}}"

OUTPUT_BASE_DIR_DEFAULT="$(dirname $0)/output/wikiextractor"
OUTPUT_BASE_DIR="${4:-${OUTPUT_BASE_DIR_DEFAULT}}"

TEMPLATES_DIR_DEFAULT="${OUTPUT_BASE_DIR}/templates_cache"
TEMPLATES_DIR="${2:-${TEMPLATES_DIR_DEFAULT}}"

INPUT_DUMP_DIR_DEFAULT="$(dirname $0)/output/download"
INPUT_DUMP_DIR="${3:-${INPUT_DUMP_DIR_DEFAULT}}"

LANG_LIST=(ar de en es fa ja sr ta tr)

if [[ ! -x "${WIKIEXTRACTOR}" ]]; then
  echo "! Cannot find runnable wikiextractor at ${WIKIEXTRACTOR}"
  exit 1
fi

mkdir -p "${TEMPLATES_DIR}"

for lang in ${LANG_LIST[@]}; do
  date
  echo ${lang};

  # Filename for templates. Storing them allows faster reruns later if
  # necessary.
  templates="${TEMPLATES_DIR}/${lang}"

  # Output directory for this language.
  output_dir="${OUTPUT_BASE_DIR}/${lang}"

  input="${INPUT_DUMP_DIR}/${lang}wikinews-20190101-pages-articles.xml.bz2"
  mkdir -p ${output_dir}
  ${WIKIEXTRACTOR} \
      --templates "${templates}" \
      --revision \
      --links \
      --sections \
      --json \
      --compress \
      --output ${output_dir} \
      "${input}" \
      &> ${output_dir}/log
done
date

echo ">Done: ${OUTPUT_BASE_DIR}"
