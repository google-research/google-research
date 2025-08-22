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



IMG_ROOT="/path/to/cache"
POPE_ROOT="/path/to/cache/POPE"
OUTDIR="OUTPUT/"

COCO_IMG_DIR=$IMG_ROOT"/coco/images/val2014"
AOKVQA_IMG_DIR=$IMG_ROOT"/coco/images/val2014"
GQA_IMG_DIR=$IMG_ROOT"/gqa/images/val"

# aokvqa # leave answer-file as it is - it will be overwritten with a new name
python eval_hall_gemini.py \
--image-folder ${AOKVQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/aokvqa/aokvqa_pope_adversarial.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${AOKVQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/aokvqa/aokvqa_pope_popular.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${AOKVQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/aokvqa/aokvqa_pope_random.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

# coco # leave answer-file as it is - it will be overwritten with a new name
python eval_hall_gemini.py \
--image-folder ${COCO_IMG_DIR} \
--question-file ${POPE_ROOT}"/coco/coco_pope_adversarial.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${COCO_IMG_DIR} \
--question-file ${POPE_ROOT}"/coco/coco_pope_popular.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${COCO_IMG_DIR} \
--question-file ${POPE_ROOT}"/coco/coco_pope_random.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

# gqa # leave answer-file as it is - it will be overwritten with a new name
python eval_hall_gemini.py \
--image-folder ${GQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/gqa/gqa_pope_adversarial.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${GQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/gqa/gqa_pope_popular.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"

python eval_hall_gemini.py \
--image-folder ${GQA_IMG_DIR} \
--question-file ${POPE_ROOT}"/gqa/gqa_pope_random.json" \
--answers-file ${OUTDIR}"/gemini.jsonl"