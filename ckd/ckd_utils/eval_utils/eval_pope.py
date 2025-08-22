# coding=utf-8
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

# pylint: disable=missing-module-docstring
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=consider-using-from-import
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


# adopted from https://github.com/DAMO-NLP-SG/VCD/blob/master/experiments/eval/eval_pope.py
# License information: https://github.com/DAMO-NLP-SG/VCD/blob/master/LICENSE

import argparse
import json
import os
from tqdm import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument("--gt_files", type=str, default="data/POPE/coco_pope_popular.json")
# parser.add_argument("--gen_files", type=str, default="answer_files_POPE/llava15_coco_pope_popular_answers_no_cd.jsonl")
# args = parser.parse_args()


def calculate_pope_results(args):
  # open ground truth answers
  gt_files = [
      json.loads(q) for q in open(os.path.expanduser(args.question_file), 'r')
  ]

  # open generated answers
  gen_files = [
      json.loads(q) for q in open(os.path.expanduser(args.answers_file), 'r')
  ]

  # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0
  unknown = 0
  total_questions = len(gt_files)
  yes_answers = 0

  # compare answers
  for index, line in enumerate(gt_files):
    idx = line['question_id']
    gt_answer = line['label']
    assert idx == gen_files[index]['question_id']
    gen_answer = gen_files[index]['text']
    # convert to lowercase
    gt_answer = gt_answer.lower()
    gen_answer = gen_answer.lower()
    # strip
    gt_answer = gt_answer.strip()
    gen_answer = gen_answer.strip()
    # pos = 'yes', neg = 'no'
    if gt_answer == 'yes':
      if 'yes' in gen_answer:
        true_pos += 1
        yes_answers += 1
      else:
        false_neg += 1
    elif gt_answer == 'no':
      if 'no' in gen_answer:
        true_neg += 1
      else:
        yes_answers += 1
        false_pos += 1
    else:
      print(f'Warning: unknown gt_answer: {gt_answer}')
      unknown += 1

  # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
  precision = true_pos / (true_pos + false_pos)
  recall = true_pos / (true_pos + false_neg)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = (true_pos + true_neg) / total_questions
  yes_proportion = yes_answers / total_questions
  unknown_prop = unknown / total_questions
  # report results
  print(f'Precision: {round(precision,4)}')
  print(f'Recall: {round(recall,4)}')
  print(f'F1: {round(f1,4)}')
  print(f'Accuracy: {round(accuracy,4)}')
  print(f'yes: {round(yes_proportion,4)}')
  print(f'unknow: {round(unknown_prop,4)}')

  return {
      'precision': round(precision, 4),
      'recall': round(recall, 4),
      'f1': round(f1, 4),
      'accuracy': round(accuracy, 4),
      'yes_proportion': round(yes_proportion, 4),
      'unknown_prop': round(unknown_prop, 4),
  }
