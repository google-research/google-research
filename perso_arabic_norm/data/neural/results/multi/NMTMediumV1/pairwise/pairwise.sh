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


#
# Pairwise significance testing.

NUM_EPOCHS=8
LANGUAGES=("ckb" "snd" "uig" "urd")

for epoch in $(seq ${NUM_EPOCHS}) ; do
  EPOCH_DIR="test1.epoch"${epoch}
  mkdir -p ${EPOCH_DIR}
  for lang in ${LANGUAGES[@]} ; do
    echo "${lang}: paired-bs ..."
    sacrebleu ../original/languages/${EPOCH_DIR}/${lang}.ref -i ../original/languages/${EPOCH_DIR}/${lang}.hyp ../rewrites/languages/${EPOCH_DIR}/${lang}.hyp -m bleu chrf ter -w 4 --paired-job 20 --paired-bs > ${EPOCH_DIR}/${lang}.paired-bs
    echo "${lang}: paired-ar ..."
    sacrebleu ../original/languages/${EPOCH_DIR}/${lang}.ref -i ../original/languages/${EPOCH_DIR}/${lang}.hyp ../rewrites/languages/${EPOCH_DIR}/${lang}.hyp -m bleu chrf ter -w 4 --paired-job 20 --paired-ar > ${EPOCH_DIR}/${lang}.paired-ar
  done
done
