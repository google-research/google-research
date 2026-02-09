# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
import os
import numpy as np

"""Evaluate the model predictions.

This script loads the model predictions and the ground truth annotations
and calculates various metrics, including accuracy and F1-score.
"""


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
  # These weights are derived from the dataset distribution.
  weighted_f1 = (1269 / 1500) * f1_pos + (231 / 1500) * f1_neg

  # Calculate F1 score for the positive class
  f1 = (
      2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
      if (precision_pos + recall_pos) > 0
      else 0
  )

  return {
      'macro_f1_for_social_interation': macro_f1,
      'weighted_f1': weighted_f1,
      'f1': f1,
  }


# Load config
with open('config.json', 'r') as f:
  config = json.load(f)
data_base_path = config['data_base_path']

# Path to the prediction results file
result_path = os.path.join(data_base_path, 'logs/gemini2.5_SI_baseline.json')
# Path to the annotation file
annotation_path = os.path.join(
    data_base_path, 'annotation_new_full_social_interaction_focus_v3.json'
)

with open(result_path, 'r') as f:
  output = json.load(f)

with open(annotation_path, 'r') as f:
  annotation = json.load(f)

key_list = list(annotation.keys())

total_yes = 0
total_no = 0
correct_yes = 0
correct_no = 0

for i, _ in enumerate(output):
  key = output[i]['key']

  # The following commented sections are for evaluating different social cues.
  # Uncomment the section corresponding to the cue you want to evaluate.

  # # someone else is talking
  # signal_sum = annotation[key]['sequence']['person/bbox/talking']
  # signal_sum = [t[1:] for t in signal_sum] # someone else
  # if signal_sum > 1:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # # alternating speech turns
  # signal = annotation[key]['sequence']['person/bbox/talking']
  # signal = np.array(signal)
  # signal_sum = np.sum(signal, axis=0)
  # signal_sum = np.sum(signal_sum > 0)
  # if signal_sum > 1:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # talking_to_me
  # signal_sum = annotation[key]['sequence']['person/bbox/talking_to_me']
  # if signal_sum > 1:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # I am talking
  # signal_sum = annotation[key]['sequence']['person/bbox/talking']
  # signal_sum = [t[0] for t in signal_sum] # I talk
  # if np.sum(signal_sum) != 0:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # # personal_space
  # signal_sum = annotation[key]['context']['personal_space']
  # if np.sum(signal_sum) != 0:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # # looking at me
  # signal_sum = annotation[key]['sequence']['person/bbox/looking_at_me']
  # if np.sum(signal_sum) != 0:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # # I look at someone
  # signal_sum = annotation[key]['context']['i_look_at']
  # if np.sum(signal_sum) != 0:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # # focus
  # signal_sum = annotation[key]['context']['focus']
  # if np.sum(signal_sum) != 0:
  #   total_yes += 1
  #   if 'answer: yes' in output[i]["response"]:
  #     correct_yes += 1
  # else:
  #   total_no += 1
  #   if 'answer: no' in output[i]["response"]:
  #     correct_no += 1

  # social interaction
  signal_sum = annotation[key]['sequence']['person/bbox/social_interaction']

  if np.sum(signal_sum) != 0:
    total_yes += 1
    if 'answer: yes' in output[i]['response']:
      correct_yes += 1
  else:
    total_no += 1
    if 'answer: no' in output[i]['response']:
      correct_no += 1

total_number = total_yes + total_no
print(
    f'Yes: correct_yes: {correct_yes}, total_yes: {total_yes}, Acc:'
    f' {correct_yes / total_yes if total_yes > 0 else 0}'
)
print(
    f'No: correct_no: {correct_no}, total_no: {total_no}, Acc (Intervention):'
    f' {correct_no / total_no if total_no > 0 else 0}'
)
print(
    f'All: correct_all: {correct_yes + correct_no}, total_all: {total_number},'
    ' Acc:'
    f' {(correct_yes + correct_no) / total_number if total_number > 0 else 0}'
)

positive_data_points = total_yes
negative_data_points = total_no
TP = correct_yes
TN = correct_no
FP = negative_data_points - TN
FN = positive_data_points - TP

f1_scores = calculate_f1_scores(TP, TN, FP, FN)
print(
    f"Overall social interaction: {f1_scores['macro_f1_for_social_interation']}"
)
