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


set -eux

function die {
  echo "${1}"
  exit 1
}

function check_eq {
  if [[ "${1}" != "${2}" ]]
  then
    echo "Expected equality, but values differ:"
    echo $1
    echo $2
    exit 1
  fi
}


SPLIT="dev"
BUCKET="random"

# Find input files.
FAKE_DATASET_DIR="frmt/testdata"
EVAL_BINARY="python -m frmt.evaluate"
TMP="/tmp/"

FAKE_BUCKET_DIR="${FAKE_DATASET_DIR}/${BUCKET}_bucket"
PT_BR_INPUT="${FAKE_BUCKET_DIR}/fake_pt-BR_outputs.tsv"
ZH_CN_INPUT="${FAKE_BUCKET_DIR}/fake_zh-CN_outputs.tsv"
EXPECTED_PT_BR_OUTPUT="${FAKE_BUCKET_DIR}/expected_pt-BR_metrics.tsv"
EXPECTED_ZH_CN_OUTPUT="${FAKE_BUCKET_DIR}/expected_zh-CN_metrics.tsv"
ACTUAL_PT_BR_OUTPUT="${TMP}/actual_pt-BR_metrics.tsv"
ACTUAL_ZH_CN_OUTPUT="${TMP}/actual_zh-CN_metrics.tsv"

${EVAL_BINARY} \
  --prediction_files="${FAKE_BUCKET_DIR}/pt_random_${SPLIT}_en_pt-BR.tsv,${PT_BR_INPUT}" \
  --dataset_dir="${FAKE_DATASET_DIR}" \
  --language="pt" \
  --split="${SPLIT}" \
  --bucket="${BUCKET}" \
  --metric="bleu" \
  --metric="chrf" \
  --output_file="${ACTUAL_PT_BR_OUTPUT}" || die "Failed in fake pt evaluation."

${EVAL_BINARY} \
  --prediction_files="${FAKE_BUCKET_DIR}/zh_random_${SPLIT}_en_zh-CN.tsv,${ZH_CN_INPUT}" \
  --dataset_dir="${FAKE_DATASET_DIR}" \
  --language="zh" \
  --split="${SPLIT}" \
  --bucket="${BUCKET}" \
  --metric="bleu" \
  --metric="chrf" \
  --output_file="${ACTUAL_ZH_CN_OUTPUT}" || die "Failed in fake zh evaluation."

check_eq "$(cat ${ACTUAL_PT_BR_OUTPUT})" "$(cat ${EXPECTED_PT_BR_OUTPUT})"
check_eq "$(cat ${ACTUAL_ZH_CN_OUTPUT})" "$(cat ${EXPECTED_ZH_CN_OUTPUT})"

# Make sure the real data is handled appropriately.
REAL_DATASET_DIR="frmt/dataset/"

REAL_BUCKET_DIR="${REAL_DATASET_DIR}/${BUCKET}_bucket"
for split in "test" "dev"
do
  ${EVAL_BINARY} \
    --prediction_files="${REAL_BUCKET_DIR}/pt_random_${split}_en_pt-BR.tsv" \
    --dataset_dir="${REAL_DATASET_DIR}" \
    --language="pt" \
    --split="${split}" \
    --bucket="${BUCKET}" \
    --metric="bleu" \
    --metric="chrf" || die "Failed in real pt evaluation."
  
  ${EVAL_BINARY} \
    --prediction_files="${REAL_BUCKET_DIR}/zh_random_${split}_en_zh-CN.tsv" \
    --dataset_dir="${REAL_DATASET_DIR}" \
    --language="zh" \
    --split="${split}" \
    --bucket="${BUCKET}" \
    --metric="bleu" \
    --metric="chrf" || die "Failed in real zh evaluation."
done

echo "PASS"
