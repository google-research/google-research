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

# This script provides a few simple end-to-end regression tests for the
# wikinews_extractor package.
#
# NOTE: This script should be run from the parent directory of
# dense_representations_for_entity_retrieval/.

set -eux

D="dense_representations_for_entity_retrieval"
REQUIREMENTS="${D}/mel/wikinews_extractor/requirements.txt"
TEST_DATA="${D}/mel/wikinews_extractor/testdata"
MODULE="${D}.mel.wikinews_extractor.parse_wikinews_i18n"

# Set up virtual environment.
virtualenv -p python3 ./testenv
source ./testenv/bin/activate
pip install -r ${REQUIREMENTS}

# Prepare bzipped version of test input archive.
bzip2 -f -k ${TEST_DATA}/test.jsonl

# Create temporary output directory (cleared and reused below).
OUTPUT_DIR=$(mktemp -d)

echo "============================================="
echo "raw mode ------------------------------------"
python -m ${MODULE} \
  --language=tr \
  --wikinews_archive=${TEST_DATA}/test.jsonl.bz2 \
  --mode=raw \
  --output_dir_wiki=${OUTPUT_DIR}/wiki \
  --output_dir_text=${OUTPUT_DIR}/text \
  --doc_index_path=${OUTPUT_DIR}/docs.tsv \
  --mention_index_path=${OUTPUT_DIR}/mentions.tsv \
  --logtostderr

diff -q "${OUTPUT_DIR}/docs.tsv" "${TEST_DATA}/expected_docs.tsv"
diff -q "${OUTPUT_DIR}/mentions.tsv" "${TEST_DATA}/expected_mentions.tsv"

echo "============================================="
echo "dataset mode (should succeed)----------------"
rm -rf "${OUTPUT_DIR}"
python -m ${MODULE} \
  --language=tr \
  --wikinews_archive="${TEST_DATA}/test.jsonl.bz2" \
  --mode=dataset \
  --output_dir_wiki="${OUTPUT_DIR}/wiki" \
  --output_dir_text="${OUTPUT_DIR}/text" \
  --doc_index_path="${TEST_DATA}/expected_docs.tsv" \
  --mention_index_path="${TEST_DATA}/dataset_mentions.tsv" \
  --logtostderr

diff -q "${OUTPUT_DIR}/text/tr-8000000" "${TEST_DATA}/expected_clean.txt"

echo "============================================="
echo "dataset mode (should detect mention issue)---"
rm -rf "${OUTPUT_DIR}"
python -m ${MODULE} \
  --language=tr \
  --wikinews_archive="${TEST_DATA}/test.jsonl.bz2" \
  --mode=dataset \
  --output_dir_wiki="${OUTPUT_DIR}/wiki" \
  --output_dir_text="${OUTPUT_DIR}/text" \
  --doc_index_path="${TEST_DATA}/expected_docs.tsv" \
  --mention_index_path="${TEST_DATA}/dataset_mentions_corrupted.tsv" \
  --logtostderr

echo "============================================="
echo "dataset mode (should detect bad doc)---------"
rm -rf "${OUTPUT_DIR}"
python -m ${MODULE} \
  --language=tr \
  --wikinews_archive="${TEST_DATA}/test.jsonl.bz2" \
  --mode=dataset \
  --output_dir_wiki="${OUTPUT_DIR}/wiki" \
  --output_dir_text="${OUTPUT_DIR}/text" \
  --doc_index_path="${TEST_DATA}/docs_corrupted.tsv" \
  --mention_index_path="${TEST_DATA}/dataset_mentions.tsv" \
  --logtostderr

rm -rf "${OUTPUT_DIR}"

echo "============================================="
echo "Tests done."
