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

import json
import numpy as np


def calculate_f1_scores(tp, tn, fp, fn):
  """Calculates Macro F1-score, Weighted F1-score, and F1 score.

  Args:
      tp: True positives.
      tn: True negatives.
      fp: False positives.
      fn: False negatives.

  Returns:
      A dictionary containing Macro F1-score, Weighted F1-score, and F1 score.
  """

  # Calculate precision and recall for positive and negative classes
  precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0

  precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
  recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0

  # Calculate F1-scores for positive and negative classes
  f1_pos = (
      2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
      if (precision_pos + recall_pos) > 0
      else 0
  )
  f1_neg = (
      2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
      if (precision_neg + recall_neg) > 0
      else 0
  )

  # Calculate Macro F1-score
  macro_f1 = (f1_pos + f1_neg) / 2

  # Calculate Weighted F1-score
  weighted_f1 = (1269 / 1500) * f1_pos + (231 / 1500) * f1_neg

  # Calculate F1 score
  f1 = (
      2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
      if (precision_pos + recall_pos) > 0
      else 0
  )

  return {'macro_f1': macro_f1}  # only need macro f1 for social interaction


def evaluation(pred_result, annotation):

  total_yes = 0
  total_no = 0
  correct_yes = 0
  correct_no = 0
  res = {}

  for i, _ in enumerate(pred_result):
    key = pred_result[i]['key']

    # social interaction
    signal_sum = annotation[key]['sequence']['person/bbox/social_interaction']

    if np.sum(signal_sum) != 0:
      total_yes += 1
      if 'answer: yes' in pred_result[i]['response']:
        correct_yes += 1
    else:
      total_no += 1
      if 'answer: no' in pred_result[i]['response']:
        correct_no += 1

  total_number = total_yes + total_no
  print(
      f'Yes: correct_yes: {correct_yes}, total_yes: {total_yes}, Acc:'
      f' {correct_yes/total_yes}'
  )
  print(
      f'No: correct_no: {correct_no}, total_no: {total_no}, Acc (Intervention'
      f' Metric): {correct_no/total_no}'
  )
  print(
      f'All: correct_all: {correct_yes + correct_no}, total_all:'
      f' {total_number}, Acc: {(correct_yes + correct_no)/total_number}'
  )

  positive_data_points = total_yes
  negative_data_points = total_no
  TP = correct_yes
  TN = correct_no
  FP = negative_data_points - TN
  FN = positive_data_points - TP

  f1_scores = calculate_f1_scores(TP, TN, FP, FN)
  print(f"Macro F1-score for Social Interaction: {f1_scores['macro_f1']}")

  res['acc_yes'] = correct_yes / total_yes
  res['acc_no_metric_for_intervention'] = correct_no / total_no
  res['acc_all'] = (correct_yes + correct_no) / (total_yes + total_no)
  res['macro_f1_metric_for_social_interation'] = f1_scores['macro_f1']
  res['weighted_f1'] = f1_scores['weighted_f1']
  res['f1'] = f1_scores['f1']

  return res
