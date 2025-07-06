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

"""Utilities used to evaluate models and datasets."""

import dataclasses
import pandas as pd
from cisc.src import run_lib
from cisc.src.post_processing import aggregators
from cisc.src.post_processing import calibration_util
from cisc.src.post_processing import per_question_eval
from cisc.src.post_processing import per_trace_processing


@dataclasses.dataclass(frozen=True)
class ConfidenceMethodsExperiment:
  wqd_eval: per_question_eval.WQDEvalStats
  calibration_metrics: calibration_util.CalibrationMetrics


@dataclasses.dataclass()
class DatasetStats:
  name: str
  # A dictionary mapping from metric name to a list of scores. The list contains
  # one score per number of traces, or a list of scores per question for each
  # number of traces.
  score_stats: dict[str, list[float]] | dict[str, list[list[float]]]
  # Debug info from the post processing.
  debug_info: per_trace_processing.DebugInfo
  # The stats for the table which compares between the confidence methods
  # (e.g., verbal, logit, response_probability).
  confidence_methods_stats: dict[str, ConfidenceMethodsExperiment]


@dataclasses.dataclass(frozen=True)
class TemperatureTuningResult:
  """Temperature tuning result for a specific model and confidence method."""

  # All the tempeatures that were attempted for this model and method, the score
  # of each one of them, and the best temperature.
  attempted_temperatures: list[float]
  attempted_temperature_scores: list[float]
  best_temperature: float


CONFIDENCE_COLS = [
    "verbal_confidence",
    "logit_confidence",
    "response_probability",
    "binary_confidence",
]


def get_confidence_methods_stats(
    per_trace_data,
    per_question_data,
    num_bootstrap,
):
  """Computes the stats for the confidence methods comparasion table."""
  confidence_methods_stats = {}
  for col_name in CONFIDENCE_COLS:
    nans = per_trace_data[col_name].isna()
    print(f"number {col_name} nans: {nans.sum()}")
    data = per_trace_data[~nans]

    if data.empty:
      # Can happen when we did not request this confidence method.
      continue

    # Normalize the confidence col to be between 0 and 1. This is needed for
    # the calibration metrics.
    min_num = min(data[col_name])
    max_num = max(data[col_name])
    if min_num == max_num:
      print(f"Unexpected min_num == max_num for {col_name}. Val is {min_num}")
      if min_num != 0:
        data[col_name] = data[col_name] / min_num
    else:
      min_max_norm = lambda x: (x - min_num) / (max_num - min_num)  # pylint: disable=cell-var-from-loop
      data[col_name] = data[col_name].apply(min_max_norm)

    metrics = calibration_util.fit_and_calculate_calibration_metrics(
        data, col_name
    )
    wqd_eval = per_question_eval.wqd_eval(
        per_question_data, col_name, num_bootstrap
    )
    confidence_methods_stats[col_name] = ConfidenceMethodsExperiment(
        wqd_eval=wqd_eval, calibration_metrics=metrics
    )
  return confidence_methods_stats


def calculate_stats_for_model_and_dataset_path(
    model_name,
    raw_results_path,
    filter_answers,
    round_negative_conf_to_zero,
    re_compute_is_correct,
    aggregator_configs,
    traces_lens,
    num_bootstrap,
    return_per_question_scores,
    file_name,
):
  """Processes the results of a single dataset."""
  print(f"Reading {raw_results_path} ...")
  ds = run_lib.load_dataset_from_disk(
      raw_results_path,
      file_name=file_name,
  )
  print(f"Done reading {raw_results_path} ...")
  return calculate_stats_for_model_and_dataset(
      model_name,
      ds,
      raw_results_path,
      filter_answers,
      round_negative_conf_to_zero,
      re_compute_is_correct,
      aggregator_configs,
      traces_lens,
      num_bootstrap,
      return_per_question_scores,
  )


def calculate_stats_for_model_and_dataset(
    model_name,
    ds,
    raw_results_path,
    filter_answers,
    round_negative_conf_to_zero,
    re_compute_is_correct,
    aggregator_configs,
    traces_lens,
    num_bootstrap,
    return_per_question_scores,
):
  """Processes the results of a single dataset."""
  # Post process on per trace level.
  data, debug_info = per_trace_processing.post_process_results_dataframe(
      ds.get_results_df(),
      confidence_config=ds.experiment_configuration.confidence_config,
      config=per_trace_processing.PostProcessingConfig(
          filter_answers=filter_answers,
          round_negative_conf_to_zero=round_negative_conf_to_zero,
          re_compute_is_correct=re_compute_is_correct,
      ),
  )

  # Post process on per question level.
  gb_data = per_question_eval.group_by_question_id(data)
  try:
    score_stats = per_question_eval.score(
        gb_data,
        eval_func_configs=aggregator_configs,
        traces_lens=traces_lens,
        num_bootstrap=num_bootstrap,
        return_per_question_scores=return_per_question_scores,
    )
  except Exception as e:
    print(
        "Failed to run per_question_eval.score for"
        f" {model_name} {raw_results_path}"
    )
    raise e
  try:
    confidence_methods_stats = get_confidence_methods_stats(
        data, gb_data, num_bootstrap
    )
  except Exception as e:
    print(
        "Failed to run get_confidence_methods_stats for"
        f" {model_name} {raw_results_path}"
    )
    raise e
  return DatasetStats(
      ds.dataset_name,
      score_stats,
      debug_info,
      confidence_methods_stats,
  )
