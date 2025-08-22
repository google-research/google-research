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

"""This module is used to run per question evaluation.

That is, the evaluation is done on a unit of a question rather than a single
trace.
"""

import dataclasses
from enum import Enum
import functools
import random
from typing import Any, Callable
import numpy as np
import pandas as pd
import tqdm
from cisc.src.post_processing import aggregators


def group_by_question_id(
    df: pd.DataFrame,
    confidence_cols=(
        'verbal_confidence',
        'logit_confidence',
        'response_probability',
        'binary_confidence',
    ),
) -> pd.DataFrame:
  """Groups the dataframe by question id."""
  for conf_col in confidence_cols:
    nan_cnt = df[conf_col].isna().sum()
    if nan_cnt:
      print(f'WARNING: Found {nan_cnt} NAN {conf_col}.')

  agg_dict = {
      'answer': list,
      'response': 'first',
      'prompt': 'first',
      'golden_label': 'first',
  }
  for conf_col in confidence_cols:
    agg_dict[conf_col] = list
  return df.groupby('question_id').agg(agg_dict)


def run_eval_for_num_traces(
    df,
    num_traces: int,
    eval_func_configs: list[aggregators.AggregatorConfig],
    bootstrap_num,
) -> dict[str, list[float]]:
  """Runs all the eval fuctions specified in in `eval_func_names` on the `df`.

  From each row, sample `bootstrap_num` samples of size `num_traces`, and run
  the eval functions on each of the samples.

  Args:
    df: the data on which to run the eval. Each row represents a single question
      with multiple traces.
    num_traces: number of traces to keep from each row. That is, the size of the
      sample.
    eval_func_configs: the configs of the eval functions to run (e.g., SC or
      CISC).
    bootstrap_num: the number of bootstrap samples to take. Each bootstrap
      sample is of size `num_traces`.

  Returns:
    A dictionary of where keys are the eval function names and the values are
    the scores per question aggregated over all bootstrap samples.
  """
  # For each length we sample multiple times to reduce variance.
  df = df[[
      'answer',
      'verbal_confidence',
      'logit_confidence',
      'binary_confidence',
      'response_probability',
      'golden_label',
  ]].copy()
  rnd = random.Random(17)

  def sample_num_traces(row):
    row = row.copy()
    triplet = list(
        zip(
            row['answer'],
            row['verbal_confidence'],
            row['logit_confidence'],
            row['response_probability'],
            row['binary_confidence'],
        )
    )
    triplet = rnd.sample(triplet, min(num_traces, len(triplet)))
    (
        row['answer'],
        row['verbal_confidence'],
        row['logit_confidence'],
        row['response_probability'],
        row['binary_confidence'],
    ) = zip(*triplet)
    return row

  bootstrap_scores = {config.config_name(): [] for config in eval_func_configs}

  def run_eval(
      metric_name: str, eval_func: Callable[[Any], Any], samples, label
  ):
    answers = np.array([eval_func(s) for s in samples])
    bootstrap_scores[metric_name].append((answers == label).mean())

  for _, row in df.iterrows():
    row = row.to_dict()
    samples = [sample_num_traces(row) for _ in range(bootstrap_num)]
    for config in eval_func_configs:
      match config.aggregator_type:
        case aggregators.AggregatorType.SC:
          run_eval(
              config.config_name(),
              lambda x: aggregators.majority(x['answer'])[0],
              samples=samples,
              label=row['golden_label'],
          )
        case aggregators.AggregatorType.CISC:
          run_eval(
              config.config_name(),
              lambda x: aggregators.majority_with_conf(
                  x['answer'],
                  x[config.confidence_col_name],  # pylint: disable=cell-var-from-loop
                  norm_type=config.norm_type,  # pylint: disable=cell-var-from-loop
                  temp=config.temperature,  # pylint: disable=cell-var-from-loop
              )[0],
              samples=samples,
              label=row['golden_label'],
          )
        case aggregators.AggregatorType.TIE:
          run_eval(
              config.config_name(),
              lambda x: aggregators.majority_with_conf(
                  x['answer'],
                  x[config.confidence_col_name],  # pylint: disable=cell-var-from-loop
                  norm_type=config.norm_type,  # pylint: disable=cell-var-from-loop
                  temp=config.temperature,  # pylint: disable=cell-var-from-loop
                  only_tie_break=True,
              )[0],
              samples=samples,
              label=row['golden_label'],
          )
        case aggregators.AggregatorType.MAX:
          run_eval(
              config.config_name(),
              lambda x: aggregators.max_confidence(
                  x['answer'], x[config.confidence_col_name]  # pylint: disable=cell-var-from-loop
              ),
              samples=samples,
              label=row['golden_label'],
          )
        case aggregators.AggregatorType.ORACLE:
          run_eval(
              config.config_name(),
              lambda x: aggregators.is_in_any(
                  x['answer'],
                  x[config.confidence_col_name],  # pylint: disable=cell-var-from-loop
                  x.golden_label,
                  num_traces=num_traces,
                  sort_by_confidence=False,
              ),
              samples=samples,
              label=row['golden_label'],
          )
        case aggregators.AggregatorType.ORACLE_BY_CONF:
          run_eval(
              config.config_name(),
              lambda x: aggregators.is_in_any(
                  x['answer'],
                  x[config.confidence_col_name],  # pylint: disable=cell-var-from-loop
                  x.golden_label,
                  num_traces=num_traces,
                  sort_by_confidence=True,
              ),
              samples=samples,
              label=row['golden_label'],
          )
  mean_scores = {}
  for metric_name, scores in bootstrap_scores.items():
    mean_scores[metric_name] = scores
  return mean_scores


def score(
    df: pd.DataFrame,
    eval_func_configs: list[aggregators.AggregatorConfig],
    traces_lens: list[int],
    num_bootstrap: int,
    return_per_question_scores=False,
) -> dict[str, list[float] | list[list[float]]]:
  """Runs a bunch of aggregators on the dataframe.

  Args:
    df: The dataframe on which to run the aggregators. It is expected that it is
      already grouped by question id.
    eval_func_configs: The configs of the eval functions to run.
    traces_lens: The lens of the traces in each datapoint. For example if
      `traces_lens` is [1, 3], then on datapoint would be computed with 1 trace
      from each row, and the other datapoint would be computed with 3 traces
      from each row.
    num_bootstrap: The number of bootstrap samples to take. Each bootstrap
      sample is of size `num_traces`.
    return_per_question_scores: If True, returns the scores for each question
      separately. Otherwise, returns the average score over all questions.

  Returns:
    A dictionary of where keys are the eval function names and the values are
    lists of aggregated scores for each datapoint in `traces_lens`.
  """
  # Each time take an increment number of traces. Run eval and add to `stats`.
  stats = {config.config_name(): [] for config in eval_func_configs}
  for length in tqdm.tqdm(traces_lens):
    trace_scores = run_eval_for_num_traces(
        df,
        length,
        eval_func_configs,
        num_bootstrap,
    )
    for metric_name, metric_score in trace_scores.items():
      if return_per_question_scores:
        stats[metric_name].append(metric_score)
      else:
        stats[metric_name].append(np.mean(metric_score).item())

  return stats


def is_none_or_empty_string(x: Any):
  return (x is None) or (x == '') or pd.isna(x)


@dataclasses.dataclass(frozen=True)
class AnswerAndConfidence:
  """Utility class for holding a single trace."""

  answer: str
  confidence: float


def _is_one_correct(answer1, answer2, golden_label):
  """Returns True if only one of the answers is correct and none are none."""
  no_nans = not is_none_or_empty_string(
      answer1
  ) and not is_none_or_empty_string(answer2)
  only_one_correct = (answer1 == golden_label) != (answer2 == golden_label)
  return no_nans and only_one_correct


class _WQDVerdict(Enum):
  """The verdict of the WQD eval."""

  HIGER_BETTER = 1
  LOWER_BETTER = 2
  TIE = 3


def _is_higher_confidence_better(
    answer1, confidence1, confidence2, golden_label
) -> _WQDVerdict:
  """Returns True if the higher confidence trace is better."""
  if np.isclose(confidence1, confidence2):
    return _WQDVerdict.TIE
  if (confidence1 > confidence2) == (answer1 == golden_label):
    return _WQDVerdict.HIGER_BETTER
  return _WQDVerdict.LOWER_BETTER


def _calc_wqd_eval_for_row(
    row: pd.Series,
    confidence_col_name: str,
    num_bootstrap: int,
    rnd: random.Random,
) -> pd.Series:
  """Calculates the two traces eval for a single row."""

  confidences = row[confidence_col_name]
  # Group the traces features from different columns.
  traces = [
      AnswerAndConfidence(answer=a, confidence=c)
      for a, c in list(zip(row.answer, confidences))
  ]
  # Sample 2 traces num_bootstrap times. There are rare cases where there is
  # only one trace, so we skip those cases. This can happen if the model failed
  # to generate an answer almost all the traces.
  samples = [
      rnd.sample(traces, 2) for _ in range(num_bootstrap) if len(traces) > 1
  ]
  # Keep only the samples where exactly one trace is correct.
  samples = [
      s
      for s in samples
      if _is_one_correct(s[0].answer, s[1].answer, row.golden_label)
  ]
  # For each sample, check if the higher confidence trace is better.
  results = [
      _is_higher_confidence_better(
          s[0].answer, s[0].confidence, s[1].confidence, row.golden_label
      )
      for s in samples
  ]

  row['confidence_ties'] = results.count(_WQDVerdict.TIE)
  row['is_higher_better'] = results.count(_WQDVerdict.HIGER_BETTER)
  row['counts'] = len(results)
  return row


@dataclasses.dataclass(frozen=True)
class WQDEvalStats:
  """Holds the stats for the WQD eval."""

  # The number of traces pairs that were left after filtering. We keep only
  # pairs where exactly one is correct, and both are not none.
  num_pairs: int
  # Out of `num_pairs`, how many had the higher confidence trace better.
  num_higher_better: int
  # Out of `num_pairs`, how many had confidence ties. For confidence ties we
  # just arbitrarily choose one of the traces.
  num_confidence_ties: int


def wqd_eval(
    df: pd.DataFrame, confidence_col_name: str, num_bootstrap: int
) -> WQDEvalStats:
  """Samples pairs of traces and checks if higher confidence is better.

  First filters out all rows where both traces are correct or both are wrong.
  Then checks if the higher confidence trace is better.

  Args:
    df: The dataframe on which to run the aggregators. It is expected that it is
      already grouped by question id.
    confidence_col_name: the name of the confidence column to use in `df`.
    num_bootstrap: The number of bootstrap samples to take. Each bootstrap
      sample is of size `num_traces`.

  Returns:
    The fraction of questions for which the higher confidence trace is better.
    Returns None if there no questions with two traces where only one is
    correct.
  """
  # Keep only the columns we need to save memory.
  df = df[['answer', confidence_col_name, 'golden_label']].copy()
  rnd = random.Random(17)
  df = df.apply(
      functools.partial(
          _calc_wqd_eval_for_row,
          num_bootstrap=num_bootstrap,
          confidence_col_name=confidence_col_name,
          rnd=rnd,
      ),
      axis=1,
  )
  total_results = df.counts.sum()
  print(f'total_results after filtering (including bootstrap): {total_results}')

  return WQDEvalStats(
      num_pairs=total_results,
      num_higher_better=df.is_higher_better.sum(),
      num_confidence_ties=df.confidence_ties.sum(),
  )
