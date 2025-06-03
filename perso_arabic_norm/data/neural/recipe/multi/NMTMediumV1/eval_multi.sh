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
SOURCE_LANG=mlt
TARGET_LANG=eng
MODEL_DIR=run/${MODEL}
TEST_DIR=${MODEL_DIR}/test
TEST_BASENAMES=("test1")

if [ $? -ne 0 ] ; then
  echo "Directory does not exist!"
  exit 1
fi

# This order is important!
LANGUAGES=("snd" "ckb" "uig" "urd")
LANG_LENGTHS=(1000 2000 2000 2000)

for epoch in $(seq ${NUM_EPOCHS}) ; do
  echo "========= Processing Epoch ${epoch} ========="
  for test_basename in ${TEST_BASENAMES[@]} ; do
    TEST_REF_FILE=${test_basename}.${TARGET_LANG}
    if [ ! -r ${TEST_REF_FILE} ] ; then
      echo "Source test file ${TEST_REF_FILE} is missing!"
      exit 1
    fi
    TEST_HYP_FILE=${TEST_DIR}/${test_basename}.${TARGET_LANG}."epoch"${epoch}
    if [ ! -r ${TEST_HYP_FILE} ] ; then
      echo "Hypothesis test file ${TEST_HYP_FILE} is missing!"
      exit 1
    fi
    LANGS_DIR=${TEST_DIR}/langs/${test_basename}."epoch"${epoch}
    mkdir -p ${LANGS_DIR}
    lang_offset=1
    lang_id=0
    for language in ${LANGUAGES[@]} ; do
      let lang_len=${LANG_LENGTHS[${lang_id}]}
      let lang_end=${lang_offset}+${lang_len}-1
      echo "> Processing ${language}: From ${lang_offset} to ${lang_end}"
      TEST_LANG_REF_FILE=${LANGS_DIR}/${language}.ref
      cat ${TEST_REF_FILE} | head -n +"${lang_end}" | tail -n +"${lang_offset}" > ${TEST_LANG_REF_FILE}
      TEST_LANG_HYP_FILE=${LANGS_DIR}/${language}.hyp
      cat ${TEST_HYP_FILE} | head -n +"${lang_end}" | tail -n +"${lang_offset}" > ${TEST_LANG_HYP_FILE}
      echo "> Computing metrics ..."
      TEST_LANG_RESULTS_FILE=${LANGS_DIR}/${language}.results
      sacrebleu ${TEST_LANG_REF_FILE} -i ${TEST_LANG_HYP_FILE} -m bleu chrf ter -w 4 > ${TEST_LANG_RESULTS_FILE}
      let lang_offset+=${lang_len}
      let lang_id+=1
    done
  done
done
