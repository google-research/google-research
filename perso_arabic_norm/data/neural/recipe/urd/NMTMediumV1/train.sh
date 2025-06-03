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



MODEL=NMTMediumV1
NUM_EPOCHS=8
SOURCE_LANG=urd
TARGET_LANG=eng
MODEL_DIR=run/${MODEL}
TEST_DIR=${MODEL_DIR}/test
TEST_BASENAMES=("test1" "test2")

mkdir -p ${TEST_DIR}
if [ $? -ne 0 ] ; then
  echo "Failed to create test directory"
  exit 1
fi

for epoch in $(seq ${NUM_EPOCHS}) ; do
  echo "========= Training Epoch ${epoch} ========="
  onmt-main --model_type ${MODEL} --config data.yml --auto_config train --with_eval

  echo "========= Inference ${epoch} =============="
  for test_basename in ${TEST_BASENAMES[@]} ; do
    TEST_SOURCE_FILE=${test_basename}.${SOURCE_LANG}
    if [ ! -r ${TEST_SOURCE_FILE} ] ; then
      echo "Source test file ${TEST_SOURCE_FILE} is missing!"
      exit 1
    fi
    TEST_REF_FILE=${test_basename}.${TARGET_LANG}
    if [ ! -r ${TEST_REF_FILE} ] ; then
      echo "Source test file ${TEST_REF_FILE} is missing!"
      exit 1
    fi
    TEST_HYP_FILE=${TEST_DIR}/${test_basename}.${TARGET_LANG}."epoch"${epoch}
    onmt-main --config data.yml --auto_config infer --features_file ${TEST_SOURCE_FILE} --predictions_file ${TEST_HYP_FILE}
    TEST_RESULTS_FILE=${TEST_DIR}/${test_basename}".results.epoch"${epoch}
    sacrebleu ${TEST_REF_FILE} -i ${TEST_HYP_FILE} -m bleu chrf ter -w 4 > ${TEST_RESULTS_FILE}
  done
done
