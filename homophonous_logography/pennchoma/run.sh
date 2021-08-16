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
#
# Computes Penn & Choma measures for some of the languages using the Bible
# corpora.
#
# Example:
# --------
# Assuming that the Bible corpora repository was cloned from
#   https://github.com/christos-c/bible-corpus
# into the directory `${HOME}/bible-corpus/bibles/`, run:
#
#   > pennchoma/run.sh -i ${HOME}/bible-corpus/bibles/ -o /tmp/results

set -euo pipefail

function die() {
  echo "$@" 1>&2 ; exit 1;
}

function usage() {
  echo "Usage: $0 -i BIBLE_DIR -o TARGET_DIR [-n SUBSET]"
}

BIBLE_DIR=
TARGET_DIR=
SUBSET=500
while getopts i:o:n:h flag
do
  case "${flag}" in
    i) BIBLE_DIR=${OPTARG};;
    o) TARGET_DIR=${OPTARG};;
    n) SUBSET=${OPTARG};;
    h | [?]) usage ; exit 0;;
  esac
done

if [ -z "${BIBLE_DIR}" ] ; then
  die "Input directory containing Bible corpora not specified!"
fi
if [ -z "${TARGET_DIR}" ] ; then
  die "Output directory not specified!"
fi

mkdir -p "${TARGET_DIR}" || die "Failed to make ${TARGET_DIR}!"
for iter in 0 1 2 3 4
do
  lang=Chinese
  echo "${TARGET_DIR}/${lang}_ungrouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_ungrouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --notrigrams
  echo "${TARGET_DIR}/${lang}_grouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_grouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --notrigrams \
    --group_docs=6
done
for iter in 0 1 2 3 4
do
  lang=Korean
  echo "${TARGET_DIR}/${lang}_ungrouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_ungrouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --notrigrams
  echo "${TARGET_DIR}/${lang}_grouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_grouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --notrigrams \
    --group_docs=6
done
for iter in 0 1 2 3 4
do
  lang=English
  echo "${TARGET_DIR}/${lang}_ungrouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_ungrouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --trigrams
  echo "${TARGET_DIR}/${lang}_grouped_${iter}.txt"
  python3 pennchoma/penn_choma.py \
    --bible="${BIBLE_DIR}/${lang}.xml" \
    --output="${TARGET_DIR}/${lang}_grouped_${iter}.txt" \
    --random_subset="${SUBSET}" \
    --trigrams \
    --group_docs=6
done
