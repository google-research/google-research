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

"""Contains aggregators for the self-consistency results.

For example, an aggergator that simply comptes majority vote.
"""

from collections import Counter
import dataclasses
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from cisc.src.post_processing import util


class NormalizationType(Enum):
  NONE = 0
  LINEAR = 1
  SOFTMAX = 2


class AggregatorType(Enum):
  SC = 1
  CISC = 2
  TIE = 3
  MAX = 4
  ORACLE = 5
  ORACLE_BY_CONF = 6


@dataclasses.dataclass(frozen=True)
class AggregatorConfig:
  """Holds the configuration for a single eval function."""

  aggregator_type: AggregatorType
  norm_type: NormalizationType | None = None
  temperature: float | None = None

  # The name of the column that holds the confidence values. E.g., logit,
  # verbal, response_probability, etc.
  confidence_col_name: str | None = None

  def __post_init__(self):
    if self.aggregator_type in [
        AggregatorType.CISC,
        AggregatorType.TIE,
        AggregatorType.MAX,
        AggregatorType.ORACLE_BY_CONF,
    ]:
      assert self.confidence_col_name is not None
      assert self.confidence_col_name in [
          'verbal_confidence',
          'logit_confidence',
          'response_probability',
          'binary_confidence',
      ], f'Unsupported confidence column name: {self.confidence_col_name}'

    if self.aggregator_type in [
        AggregatorType.CISC,
        AggregatorType.TIE,
    ]:
      assert self.temperature is not None
      assert self.norm_type is not None

  def config_name(self):
    name = f'{self.aggregator_type.name}'
    if self.confidence_col_name is not None:
      name += f'_{self.confidence_col_name}'
    if self.norm_type is not None:
      name += f'_{self.norm_type.name}'
    if self.temperature is not None:
      name += f'_{self.temperature}'
    return name


def is_none_or_empty_string(x):
  return (x is None) or (x == '') or pd.isna(x)


def majority(answers):
  """Returns the most common answer with its aggregated confidence."""
  answers = [ans for ans in answers if not is_none_or_empty_string(ans)]
  if not answers:
    return '', 0
  ans, cnt = Counter(answers).most_common(1)[0]
  return ans, cnt / len(answers)


def majority_with_conf(
    answers,
    confidences,
    norm_type = NormalizationType.LINEAR,
    temp=1.0,
    only_tie_break=False,
):
  """Majority vote where each trace is weighted by its confidence.

  Args:
    answers: The answers to aggregate.
    confidences: The confidences of each answer.
    norm_type: Whether to use linear normalization or softmax normalization. In
      any case, for this algorithm to work properly, some normalization should
      be applied for the non positive confidences.
    temp: The temperature to use if normalize is True.
    only_tie_break: If true, only used the confidence to break ties in self
      consistency.

  Returns:
    The most common answer and its aggregated confidence.
  """
  if not isinstance(answers, list) and not isinstance(answers, tuple):
    raise ValueError(
        f'answers must be a list or tuple found {type(answers)}.\n{answers}'
    )
  if not isinstance(confidences, list) and not isinstance(confidences, tuple):
    raise ValueError(
        'confidences must be a list or tuple found'
        f' {type(confidences)}.\n{confidences}'
    )
  if len(answers) != len(confidences):
    raise ValueError(
        'The number of answers and confidences must be the same. Got'
        f' {len(answers)} answers and {len(confidences)} confidences.'
    )

  # First, filter nan answers and confidences. Note that nan confidence is
  # different from 0 confidence. The latter is a valid confidence.
  filtered = []
  for ans, conf in zip(answers, confidences):
    if not is_none_or_empty_string(ans) and not is_none_or_empty_string(conf):
      filtered.append((ans, conf))
  if not filtered:
    return '', 0
  answers, confidences = zip(*filtered)

  def _majority_with_tie(answers):
    top2 = Counter(answers).most_common(2)
    ans1, cnt1 = top2[0]
    if len(top2) == 1:
      return ans1, cnt1 / len(answers), False
    _, cnt2 = top2[1]
    is_tie = cnt1 == cnt2
    return ans1, cnt1 / len(answers), is_tie

  if only_tie_break:
    ans, conf, tie = _majority_with_tie(answers)
    if not tie:
      return ans, conf
  # Otherwise, use the confidence to choose the best answer.

  if norm_type == NormalizationType.LINEAR:
    # Always normalize the confidence by reducing the min. This is done to avoid
    # negative confidence, that do not behave nicely in this algorithm. Add a
    # small epsilon to avoid 0 confidences.
    confidences = np.array(confidences) - np.min(confidences)
    confidences = temp * confidences + 1
  elif norm_type == NormalizationType.SOFTMAX:
    # This is the default mode which is suggested by our paper.
    confidences = util.softmax(confidences, temp)
  else:
    assert norm_type == NormalizationType.NONE
    confidences = np.array(confidences)

  # Choose the answer with the highest sum of confidences.
  conf_counter = Counter()
  for ans, conf in zip(answers, confidences):
    conf_counter[ans] += conf
  ans, sum_chosen_conf = conf_counter.most_common(1)[0]
  # In rare cases all the confidences might be 0.
  confidence = sum_chosen_conf / sum(confidences) if sum(confidences) else 0
  return ans, confidence


def max_confidence(answers, confidences):
  """Chooses the answer with the highest confidence."""
  filtered = []
  for ans, conf in zip(answers, confidences):
    if not is_none_or_empty_string(ans) and not is_none_or_empty_string(conf):
      filtered.append((ans, conf))
  if not filtered:
    return '', 0
  answers, confidences = zip(*filtered)
  return answers[np.argmax(confidences)]


def remove_tile(answers, confidences, tile):
  med = np.percentile(confidences, tile)
  filtered_answers = []
  for conf, ans in zip(confidences, answers):
    if conf is not None and conf >= med:
      filtered_answers.append(ans)
  return majority(filtered_answers)[0]


def sort_a_by_b(vals, keys):
  """Sorts the values in vals based on the values in keys."""
  assert len(vals) == len(keys)
  return [x for _, x in sorted(zip(keys, vals), key=lambda pair: pair[0])]


def is_in_any(
    answers,
    confidences,
    golden_label,
    num_traces,
    sort_by_confidence=False,
):
  """Oracle mode. Returns the golden label if it is in the top num_traces."""
  if sort_by_confidence:
    confidences = [
        conf if (conf is not None and not np.isnan(conf)) else 0
        for conf in confidences
    ]
    answers = list(reversed(sort_a_by_b(answers, confidences)))
  # Return the correct answer if it is in the answers list.
  return golden_label if golden_label in answers[:num_traces] else None
