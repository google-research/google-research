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

"""Runs post processing for ExperimentResult.

For example, this includes:
1. Extracting confidence scores for the logits.
2. Rounding invalid verbal confidence to 0.
3. Setting invalid answers to an empty string.
4. Optionally filtering out invalid answers.
5. Optionally re-computing the is_correct column.
"""

import dataclasses
from typing import Any
import pandas as pd
from cisc.src import confidence_extraction
from cisc.src import run_lib
from cisc.src.post_processing import util

ExperimentResult = run_lib.ExperimentResult


@dataclasses.dataclass(frozen=True)
class PostProcessingConfig:
  """Holds all the post processing configuration."""

  # If true, removes traces that have invalid answers (E.g., NaN). Note that
  # this should be used with caution, as it would provide better results
  # than they should be.
  filter_answers: bool = False

  # If true, rounds negative confidence to 0. Negative confidence doesn't make
  # sense for our algorithm. This is only applied on the confidence extracted
  # from the log likelihoods. The verbal confidence of the model is always
  # rounded to 0, as it is never expected to be negative.
  round_negative_conf_to_zero: bool = False

  # Re-compute the is_correct column.
  re_compute_is_correct: bool = True


@dataclasses.dataclass
class DebugInfo:
  """Holds debug information about the post processing of a single dataset."""

  num_verbal_confidence_none: int = 0
  num_verbal_confidence_non_float: int = 0
  num_verbal_confidence_out_of_range: int = 0
  num_confidence_logit_non: int = 0
  num_answer_na: int = 0
  num_answer_exceptions: int = 0
  num_total_rows: int = 0

  def __str__(self):
    return '\n'.join([f'{k}: {v}' for k, v in dataclasses.asdict(self).items()])


def df_confidence_from_likelihood(
    df,
    confidence_options,
    round_negative_conf_to_zero,
):
  """Extracts the confidence from log likelihoods by computing the softmax.

  If the confidence likelihoods are not available for a row, sets the
  confidence to None.

  Args:
    df: The dataframe to extract the confidence from.
    confidence_options: the confidence options.
    round_negative_conf_to_zero: If true, rounds negative confidence to 0.

  Returns:
    A series of floats representing the confidence of each row.
  """
  assert 'confidence_likelihoods' in df.columns
  # Currently we do not support logit confidence for multi-option.
  assert len(confidence_options) == 2 or df.confidence_likelihoods.isna().all()

  def get_logit_confidence(row):
    if (
        is_none_or_empty_string(row.answer)
        or row.confidence_likelihoods is None
        or len(row.confidence_likelihoods) != 2
    ):
      return None
    # The first confidence likelihood is the likelihood that the answer is
    # incorrect. The second confidence likelihood is the likelihood that the
    # answer is correct. We return the second softmax value, which is the
    # probability that the answer is correct.
    return util.softmax(
        [row.confidence_likelihoods[0], row.confidence_likelihoods[1]], temp=1.0
    )[1]

  logit_confidence = df.apply(get_logit_confidence, axis=1)
  if round_negative_conf_to_zero:
    logit_confidence = logit_confidence.apply(
        lambda x: max(0, x) if x is not None else None
    )
  return logit_confidence


def is_none_or_empty_string(x):
  return (x is None) or (x == '') or pd.isna(x)


def try_convert_to_float(x):
  if x is None:
    return 0, False
  if isinstance(x, float) or isinstance(x, int):
    return float(x), True
  try:
    x = ''.join(c for c in x if c.isalnum() or c in ['.', '-'])
    return float(x), True
  except:  # pylint: disable=bare-except
    return 0, False


def _process_confidence_inplace(
    df,
    confidence_config,
    config,
    mutable_debug_info,
):
  """Fix verbal confidence and extracts the logit confidence.

  When the verbal confidence is invalid, simply set it to 0. This is a reasoable
  choice as negative confidences are not expected, so this is the lowest value.
  On the other hand, when the answer is invalid we set the confidence to None,
  which is more strict. Aggregators can treat None confidence differently.

  The logit-confidence is extracted from the log-likelihoods. In case the
  confidence likelihoods are not available for a certain row, set the confidence
  to None. Unlike the verbal confidence, the logit confidences can have negative
  values, so we don't want this 0 to be larger than the negative values.

  Args:
    df: The dataframe to process.
    confidence_config: Information about the prompts and optiosns that were used
      for confidence extraction.
    config: The post processing configuration.
    mutable_debug_info: The debug info to update with stats.

  Returns:
    The processed dataframe.
  """
  # First set every invalid confidence to 0.
  converted_to_float = df.verbal_confidence.apply(try_convert_to_float)
  num_non_float = converted_to_float.apply(lambda x: not x[1]).sum()
  df.loc[:, 'verbal_confidence'] = converted_to_float.apply(lambda x: x[0])
  # Assume the options are sorted.

  # If the verbal confidence exists, use the last option as the max confidence.
  max_confidence, is_max_float = 0, False
  if confidence_config.verbal_confidence.confidence_options is not None:
    max_confidence, is_max_float = try_convert_to_float(
        confidence_config.verbal_confidence.confidence_options[-1]
    )
  out_of_range = df.verbal_confidence.apply(
      lambda x: (isinstance(x, float) or isinstance(x, int))
      and (is_max_float)
      and (x < 0 or x > max_confidence)
  )
  df.loc[out_of_range, 'verbal_confidence'] = 0
  df.loc[df.answer.apply(is_none_or_empty_string), 'verbal_confidence'] = None
  mutable_debug_info.num_verbal_confidence_none = (
      df.verbal_confidence.isna().sum()
  )
  mutable_debug_info.num_verbal_confidence_non_float = num_non_float
  mutable_debug_info.num_verbal_confidence_out_of_range = out_of_range.sum()
  df.loc[df.verbal_confidence.isna(), 'verbal_confidence'] = 0

  df = df.copy()
  # Next, deal with the confidence of the log likelihoods column.
  df.loc[:, 'logit_confidence'] = df_confidence_from_likelihood(
      df,
      confidence_config.confidence_likelihoods.confidence_options,
      config.round_negative_conf_to_zero,
  )
  df.loc[:, 'binary_confidence'] = df.logit_confidence.apply(
      lambda x: float((x is not None) and (x > 0.5)),
  )
  mutable_debug_info.num_confidence_logit_non = df.logit_confidence.isna().sum()
  return df


def _process_invalid_answers(
    df,
    filter_out,
    mutable_debug_info,
):
  """Set invalid answers to an empty string and optionaly filter them out.

  In addition prints debug information about the number of invalid answers.

  Args:
    df: The dataframe to process. Note that this function modifies the
      dataframe.
    filter_out: If true, filter out the invalid answers. Otherwise, set them to
      an empty string. Use this setting with caution, as it would provide
      inaccurate results (better than they should be).
    mutable_debug_info: The debug info to update with stats.

  Returns:
    The processed dataframe.
  """
  na = df.answer.apply(is_none_or_empty_string)
  exceptions = ~(df.exception.apply(is_none_or_empty_string))
  mutable_debug_info.num_answer_na = na.sum()
  mutable_debug_info.num_answer_exceptions = exceptions.sum()

  if filter_out:
    return df[~(na | exceptions)]
  else:
    df.loc[na | exceptions, 'answer'] = ''
    return df


def post_process_results_dataframe(
    df,
    confidence_config,
    config,
):
  """Post process the results `df`.

  Args:
    df: The dataframe to process. This function modifies the dataframe. Notes:
      [1] The dataframe is expected to have a row per trace. I.e., multiple
      traces for the same question are already exploded into multiple rows. [2]
      The dataframe is expected to have the following columns:
      'verbal_confidence', 'confidence_likelihoods', 'response_probability',
      'answer', 'is_correct', 'golden_label', 'exception'. [3] If the
      'confidence_likelihoods' column is available, a new column named
      'logit_confidence' would be added, with the confidence extracted from the
      log likelihoods.
    confidence_config: Information about the prompts and optiosns that were used
      for confidence extraction.
    config: The post processing configuration.

  Returns:
    The processed dataframe (which is modified in place).
  """
  for col in [
      'verbal_confidence',
      'confidence_likelihoods',
      'response_probability',
      'answer',
      'is_correct',
      'golden_label',
      'exception',
  ]:
    if col not in df.columns:
      raise ValueError(
          f'Expected column {col} not found in the results dataframe.'
      )
  debug_info = DebugInfo()
  debug_info.num_total_rows = len(df)
  df = _process_invalid_answers(
      df,
      filter_out=config.filter_answers,
      mutable_debug_info=debug_info,
  )
  df = _process_confidence_inplace(df, confidence_config, config, debug_info)

  if config.re_compute_is_correct:
    df.loc[:, 'is_correct'] = df.apply(
        lambda row: row.answer == row.golden_label, axis=1
    )
  return df, debug_info
