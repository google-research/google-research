# coding=utf-8
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

"""HotpotQA evaluation library from official script.

(https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)
"""
import collections
import json
import re
import string

_ZERO_METRIC = (0, 0, 0)


def normalize_answer(s):
  """Normalizes answer."""

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
  """Computes F1 score."""
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  if normalized_prediction in [
      'yes', 'no', 'noanswer'
  ] and normalized_prediction != normalized_ground_truth:
    return _ZERO_METRIC
  if normalized_ground_truth in [
      'yes', 'no', 'noanswer'
  ] and normalized_prediction != normalized_ground_truth:
    return _ZERO_METRIC

  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return _ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics, prediction, gold):
  """Update answer span metrics."""
  em = exact_match_score(prediction, gold)
  f1, prec, recall = f1_score(prediction, gold)
  metrics['em'] += float(em)
  metrics['f1'] += f1
  metrics['prec'] += prec
  metrics['recall'] += recall
  return em, prec, recall


def update_sp(metrics, prediction, gold):
  """Update supporting facts metrics."""
  cur_sp_pred = set(map(tuple, prediction))
  gold_sp_pred = set(map(tuple, gold))
  tp, fp, fn = 0, 0, 0
  for e in cur_sp_pred:
    if e in gold_sp_pred:
      tp += 1
    else:
      fp += 1
  for e in gold_sp_pred:
    if e not in cur_sp_pred:
      fn += 1
  prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
  recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
  f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
  em = 1.0 if fp + fn == 0 else 0.0
  metrics['sp_em'] += em
  metrics['sp_f1'] += f1
  metrics['sp_prec'] += prec
  metrics['sp_recall'] += recall
  return em, prec, recall


def evaluate(prediction, gold):
  """Evaluates on prediction data and golden data."""
  metrics = {
      'em': 0,
      'f1': 0,
      'prec': 0,
      'recall': 0,
      'sp_em': 0,
      'sp_f1': 0,
      'sp_prec': 0,
      'sp_recall': 0,
      'joint_em': 0,
      'joint_f1': 0,
      'joint_prec': 0,
      'joint_recall': 0
  }
  for dp in gold:
    cur_id = dp['_id']
    can_eval_joint = True
    if cur_id not in prediction['answer']:
      print('missing answer {}'.format(cur_id))
      can_eval_joint = False
    else:
      em, prec, recall = update_answer(metrics, prediction['answer'][cur_id],
                                       dp['answer'])
    if cur_id not in prediction['sp']:
      print('missing sp fact {}'.format(cur_id))
      can_eval_joint = False
    else:
      sp_em, sp_prec, sp_recall = update_sp(metrics, prediction['sp'][cur_id],
                                            dp['supporting_facts'])

    if can_eval_joint:
      joint_prec = prec * sp_prec
      joint_recall = recall * sp_recall
      if joint_prec + joint_recall > 0:
        joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
      else:
        joint_f1 = 0.
      joint_em = em * sp_em

      metrics['joint_em'] += joint_em
      metrics['joint_f1'] += joint_f1
      metrics['joint_prec'] += joint_prec
      metrics['joint_recall'] += joint_recall

  num_gold = len(gold)
  for k in metrics:
    metrics[k] /= num_gold

  return metrics


def evaluate_using_files(prediction_file, gold_file):
  """Evaluates on prediction file and golden file."""
  with open(prediction_file) as f:
    prediction = json.load(f)
  with open(gold_file) as f:
    gold = json.load(f)
  return evaluate(prediction, gold)


def get_em_counts(prediction, gold):
  """Gets exact match counts for various cases."""
  # For each answer types, there are the following prediction cases.
  # span: correct, yes, no, empty, wrong
  # yes:  span,    yes, no, empty
  # no:   span,    yes, no, empty
  counts = collections.Counter()
  for dp in gold:
    gt_answer = dp['answer']
    gt_answer_normalized = normalize_answer(gt_answer)
    predict_answer = prediction['answer'][dp['_id']]
    predict_answer_normalized = normalize_answer(predict_answer)
    if gt_answer_normalized == 'yes':
      if predict_answer_normalized == 'yes':
        counts['Y_Y'] += 1
      elif predict_answer_normalized == 'no':
        counts['Y_N'] += 1
      elif predict_answer_normalized:
        counts['Y_A'] += 1
      else:
        counts['Y_E'] += 1
    elif gt_answer_normalized == 'no':
      if predict_answer_normalized == 'yes':
        counts['N_Y'] += 1
      elif predict_answer_normalized == 'no':
        counts['N_N'] += 1
      elif predict_answer_normalized:
        counts['N_A'] += 1
      else:
        counts['N_E'] += 1
    else:
      if predict_answer_normalized == 'yes':
        counts['A_Y'] += 1
      elif predict_answer_normalized == 'no':
        counts['A_N'] += 1
      elif predict_answer_normalized == gt_answer_normalized:
        counts['A_A'] += 1
      elif predict_answer_normalized:
        counts['A_B'] += 1
      else:
        counts['A_E'] += 1

  type_counts = {
      'A': sum(v for k, v in counts.items() if k.startswith('A')),
      'Y': sum(v for k, v in counts.items() if k.startswith('Y')),
      'N': sum(v for k, v in counts.items() if k.startswith('N')),
  }
  return {
      **{f'counts/{k}': v for k, v in counts.items()},
      **{f'percentages/{k}': v / type_counts[k[0]] for k, v in counts.items()},
  }
