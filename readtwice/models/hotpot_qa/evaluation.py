# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Official evaluation script for HotpotQA dataset.

From
https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
"""

import collections
import re
import string

import numpy as np


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace.

  This normalization is the same as for the TriviaQA dataset.

  Args:
    s: Text

  Returns:
    normalized text
  """

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in [
      'yes', 'no', 'noanswer'
  ] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in [
      'yes', 'no', 'noanswer'
  ] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC

  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics, prediction, gold):
  em = exact_match_score(prediction, gold)
  f1, prec, recall = f1_score(prediction, gold)
  metrics['em'] += float(em)
  metrics['f1'] += f1
  metrics['prec'] += prec
  metrics['recall'] += recall
  return em, prec, recall


def eval_hotpot_qa(groud_truth, predictions, only_span_answer):
  """Evalutes HotpotQA predictions."""
  metrics = {
      'em': 0,
      'f1': 0,
      'prec': 0,
      'recall': 0,
  }
  common = 0
  ignored_questions = 0
  for question_id, reference in groud_truth.items():
    if only_span_answer and reference in ['yes', 'no']:
      ignored_questions += 1
      continue
    if question_id not in predictions:
      continue
    common += 1
    prediction = predictions[question_id]
    # em, prec, recall = update_answer(metrics, prediction, reference)
    update_answer(metrics, prediction, reference)

  for k in metrics:
    metrics[k] /= common
  metrics['common'] = common
  metrics['ignored_questions'] = ignored_questions
  return metrics


def make_predictions_and_eval(ground_truth, predictions):
  """Makes and evaluates HotpotQA predictions."""
  answers = {}
  for key, prediction in predictions.items():
    yesno_logits = np.array(prediction['yesno_logits'])
    answer_type_prediction = np.unravel_index(yesno_logits.argmax(),
                                              yesno_logits.shape)
    answer_type_prediction = answer_type_prediction[1]
    if answer_type_prediction == 0:
      answers[key] = prediction['span_answer']
    elif answer_type_prediction == 1:
      answers[key] = 'yes'
    elif answer_type_prediction == 2:
      answers[key] = 'no'
    else:
      assert False
  return eval_hotpot_qa(ground_truth, answers, only_span_answer=False)
