#!/bin/sh
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


#
# Creates an inital romanization lexicon from Bhasha-Abhijnaanam resource,
# then deterimines uncovered pairs to add to supplementary lexicon.
#
# Example usage:
# LOCALBADIR="/path/to/local/bhasha-abhijnaanam/subdirectory"
# ./create_supplementary_lexicons.sh \
#   --input_json="${LOCALBADIR}"/parallel_romanized_train_data.json \
#   --output_dir="${LOCALBADIR}"/language_supplmentary_lexicons \
#   --base_lexicon_dir="${LOCALBADIR}"/dakshina_lexicons
#
# Assumes the names for the Aksharantar lexicons are the same as the source
# language identifiers in the Bhasha-Abhijnaanam resource, e.g., 'Assamese'
# not 'asm'.

INPUT_JSON=""
OUTPUT_DIR=""
BASE_LEXICON_DIR=""

while [[ $# != 0 ]]; do
  # Parses 'option=optarg' word.
  OPT="$(awk 'BEGIN { split(ARGV[1], a, "="); print a[1] }' "${1}")"
  ARG="$(awk 'BEGIN { split(ARGV[1], a, "="); print a[2] }' "${1}")"
  shift
  case "${OPT}" in
    --input_json|-input_json)
      # Full path to Bhasha-Abhijnaanam parallel_romanized_train_data.json file.
      INPUT_JSON="${ARG}" ;;
    --output_dir|-output_dir)
      # Full path to local output directory.
      OUTPUT_DIR="${ARG}" ;;
    --base_lexicon_dir|-base_lexicon_dir)
      # Full path to baseline lexicons, for determining OOVs.
      BASE_LEXICON_DIR="${ARG}" ;;
    *)
      echo "bad option: ${OPT}"
      exit 1 ;;
  esac
done

if [[ -z ${INPUT_JSON} || -z ${OUTPUT_DIR} || -z "${BASE_LEXICON_DIR}" ]]; then
  echo "--input_json --output_dir and --base_lexicon_dir options are required!"
  exit 1
fi

mkdir -p ${OUTPUT_DIR}
TMPDIR="$(mktemp -d -p ${OUTPUT_DIR})"
FILESUFF="parallel_romanized_train_uniq_words.tsv"

# Extracts aligned words from parallel corpus, with language label.
python3 process_raw_ba_parallel_words.py \
  --json_path="${INPUT_JSON}" |\
  LC_ALL=C sort -u >"${TMPDIR}/${FILESUFF}"

# Separates out into individual language lexicons.
cat "${TMPDIR}/${FILESUFF}" |\
  awk '{print $1}' |\
  LC_ALL=C sort -u |\
  while read LANG; do
    grep "^${LANG}" "${TMPDIR}/${FILESUFF}" |\
      awk '{printf("%s\t%s\n",$2,$3)}' \
      >"${TMPDIR}/${LANG}.${FILESUFF}"
  done
rm "${TMPDIR}/${FILESUFF}"

# Processes each language lexicon to create supplementary lexicon.
ls "${TMPDIR}" | sed 's/\..*//g' | while read LANG; do
  python3 filter_covered_words.py \
    --input_lexicon="${TMPDIR}/${LANG}.${FILESUFF}" \
    --baseline_lexicon="${BASE_LEXICON_DIR}/${LANG}".tsv \
    >"${OUTPUT_DIR}/${LANG}".supp.tsv
done

rm -rf "${TMPDIR}"
