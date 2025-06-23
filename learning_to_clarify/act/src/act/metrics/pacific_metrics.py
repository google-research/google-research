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

"""Utils for PACIFIC refactored from the official evaluation code located below.

https://raw.githubusercontent.com/dengyang17/PACIFIC/main/UniPCQA/tatqa_utils.py
https://raw.githubusercontent.com/dengyang17/PACIFIC/main/UniPCQA/tatqa_metric.py
"""

import re
import string
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from act.metrics.base_metrics import BaseMetrics


def scale_to_num(scale):
  scale = scale.lower()
  num = 1
  if 'hundred' in scale:
    num = 100
  elif 'thousand' in scale:
    num = 1000
  elif 'million' in scale:
    num = 1000000
  elif 'billion' in scale:
    num = 1000000000
  elif 'percent' in scale:
    num = 0.01
  return num


def extract_one_num_from_str(s):
  s = _clean_num(s)
  r_num = r'([+-]?\d+(\.\d+)?)|([+-]?\.\d+)'
  groups = re.findall(r_num, s)
  if len(groups) == 0:
    return None
  num = groups[0][0]
  if num == '':
    return None
  if '.' in num:
    return float(num)
  return int(num)


EXCLUDE_IN_NUM = '\'"\\$€£¥%(),[]'


def _clean_num(text):
  return ''.join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text):
  try:
    words = ' '.join([_clean_num(w) for w in text.split()]).split()
    if len(words) == 0:
      """1023 or 1 million"""
      return False
    num = float(words[0])
    if np.isnan(num):
      return False
    if len(words) >= 2:
      if scale_to_num(words[1]) == 1:
        return False
    return True
  except ValueError:
    return False


def negative_num_handle(x):
  """:param x:  transform (134) -> -134

  :return:
  """
  all = re.findall('(\([\d.\s]+\))', x.strip())
  if len(all) > 0:
    return -1
  return 1


def percent_num_handle(x):
  """:param x:  transform 12% -> 12/100

  :return:
  """
  all = re.findall('([\d.\s]+%)', x.strip())
  if len(all) > 0:
    return 0.01
  return 1


def word_scale_handle(x):
  """:param x: 1 million = 1,000,000

  :return:
  """
  iter = re.finditer('([\d.]+\s?[a-zA-Z]+)', x)
  for one in iter:
    text = one.group(0).lower()
    scale_val = scale_to_num(text)
    return scale_val
  return 1


def to_number(text):
  num = extract_one_num_from_str(text)
  scale_val = word_scale_handle(text)
  negative_flag = negative_num_handle(text)
  percent_flag = percent_num_handle(text)
  if num is not None:
    return round(num * scale_val * negative_flag * percent_flag, 4)
  return None


def remove_articles(text):
  regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
  return re.sub(regex, ' ', text)


def white_space_fix(text):
  return ' '.join(text.split())


EXCLUDE = set(string.punctuation)


def remove_punc(text):
  if not is_number(text):
    return ''.join(ch for ch in text if ch not in EXCLUDE)
  else:
    return text


def lower(text):
  return text.lower()


def tokenize(text):
  return re.split(' ', str(text))


def normalize_number(text):
  if is_number(text):
    return str(to_number(text))
  else:
    return text


def normalize_answer(text):
  """Lower text and remove punctuation, articles and extra whitespace."""
  parts = [
      white_space_fix(
          remove_articles(normalize_number(remove_punc(lower(token))))
      )
      for token in tokenize(text)
  ]
  parts = [part for part in parts if part.strip()]
  normalized = ' '.join(parts).strip()
  return normalized


STRIPPED_CHARACTERS = string.punctuation + ''.join(['‘', '’', '´', '`', '_'])


def ws_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip().lower()
  if not text:
    return []
  text = white_space_fix(text)
  tokens = text.split()
  tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
  return tokens


def _answer_to_bags(
    answer,
):
  if isinstance(answer, (list, tuple)):
    raw_spans = answer
  else:
    raw_spans = [answer]
  normalized_spans: List[str] = []
  token_bags = []
  for raw_span in raw_spans:
    normalized_span = normalize_answer(str(raw_span))
    normalized_spans.append(normalized_span)
    token_bags.append(set(normalized_span.split()))
  return normalized_spans, token_bags


def _align_bags(predicted, gold):
  """Takes gold and predicted answer sets and first finds the optimal 1-1 alignment

  between them and gets maximum metric values over all the answers.
  """
  scores = np.zeros([len(gold), len(predicted)])
  for gold_index, gold_item in enumerate(gold):
    for pred_index, pred_item in enumerate(predicted):
      scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
  row_ind, col_ind = linear_sum_assignment(-scores)

  max_scores = np.zeros([max(len(gold), len(predicted))])
  for row, column in zip(row_ind, col_ind):
    max_scores[row] = max(max_scores[row], scores[row, column])
  return max_scores


def _compute_f1(predicted_bag, gold_bag):
  intersection = len(gold_bag.intersection(predicted_bag))
  if not predicted_bag:
    precision = 1.0
  else:
    precision = intersection / float(len(predicted_bag))
  if not gold_bag:
    recall = 1.0
  else:
    recall = intersection / float(len(gold_bag))
  f1 = (
      (2 * precision * recall) / (precision + recall)
      if not (precision == 0.0 and recall == 0.0)
      else 0.0
  )
  return f1


def get_metrics(
    predicted,
    gold,
):
  """Takes a predicted answer and a gold answer (that are both either a string or a list of

  strings), and returns exact match and the DROP F1 metric for the prediction.
  If you are
  writing a script for evaluating objects in memory (say, the output of
  predictions during
  validation, or while training), this is the function you want to call, after
  using
  :func:`answer_json_to_strings` when reading the gold answer from the released
  data file.
  """
  predicted_bags = _answer_to_bags(predicted)
  gold_bags = _answer_to_bags(gold)

  if set(predicted_bags[0]) == set(gold_bags[0]) and len(
      predicted_bags[0]
  ) == len(gold_bags[0]):
    exact_match = 1.0
  else:
    exact_match = 0.0

  f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
  f1 = np.mean(f1_per_bag)
  f1 = round(f1, 2)
  return exact_match, f1


class PacificMetrics(BaseMetrics):

  def __init__(
      self,
      drop_f1_threshold=0.9,
  ):
    self.drop_f1_threshold = drop_f1_threshold

  def get_metrics(
      self,
      predicted,
      gold,
  ):
    return get_metrics(predicted, gold)

  def conditon_checker(self, **metadata):
    _, f1 = self.get_metrics(metadata['final_answer'], metadata['gold_target'])
    return f1 < self.drop_f1_threshold
