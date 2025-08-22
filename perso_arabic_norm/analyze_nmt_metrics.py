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

r"""Compare the two sets of measurements that involve NMT quality metrics.

Example:
--------
  RESULTS_DIR=data/neural/results/ckb/NMTSmallV1/
  python analyze_nmt_metrics.py \
    --baseline_metrics_dir ${RESULTS_DIR}/original/ \
    --test_metrics_dir ${RESULTS_DIR}/rewrites/ \
    --metric_file_basenames test1,test2 \
    --num_epochs 8

Dependencies:
-------------
  absl
  numpy
  pandas
  statsmodels
"""

from collections.abc import Sequence
from collections import defaultdict

import io
import json
import logging
import pathlib

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import stat_utils
import statsmodels.stats.api as sms

flags.DEFINE_string(
    "baseline_metrics_dir", "",
    "Directory containing results files in JSON format for the baselines.")

flags.DEFINE_string(
    "test_metrics_dir", "",
    "Directory containing results files in JSON format for the "
    "tested configurations.")

flags.DEFINE_list(
    "metric_file_basenames", ["test1"],
    "List of strings specifying the basenames for the metric files, e.g., "
    "`test1,test2`.")

flags.DEFINE_integer(
    "num_epochs", 8,
    "Number of training epochs corresponding to the number of metrics files.")

FLAGS = flags.FLAGS


def _load_metrics(filename):
  """Loads all the available metrics from a JSON file."""
  metrics = {}
  with open(filename, encoding="utf8") as f:
     for metric in json.load(f):
       if "name" not in metric or "score" not in metric:
         raise ValueError(f"The metrics in {filename} should have botn name "
                          "and score attributes")
       metrics[metric["name"]] = metric["score"]
  return metrics


def _process_dir(directory):
  """Processes all the results files in the supplied directory."""
  results = defaultdict(list)
  for metric_file_basename in FLAGS.metric_file_basenames:
    for epoch in range(1, FLAGS.num_epochs + 1):
      path = pathlib.Path(directory / pathlib.Path(
          f"{metric_file_basename}.results.epoch{epoch}"))
      if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
      results[metric_file_basename].append(_load_metrics(str(path)))
  return results


def _metric_list_to_dicts(all_metrics):
  """Converts a list of metric dicts to a dictionary of lists."""
  result = defaultdict(list)
  for metrics in all_metrics:
    for metric_name in metrics:
      result[metric_name].append(metrics[metric_name])
  return result


def _process_samples(base_samples, test_samples):
  """Compares populations and returns a tuple containing results of tests."""
  # Analyze the metrics using parametric method: t-test, assuming that
  # baseline and test metrics are normally distributed. Use
  # Welch-Satterthwaite t-test.
  ttest = stat_utils.ParameterStats.MeanDifference(base_samples, test_samples)
  all_diffs = list(np.array(test_samples) - np.array(base_samples))
  deltas_percent = []
  for i, diff in enumerate(all_diffs):
    deltas_percent.append(diff / test_samples[i] * 100.0)
  return deltas_percent, ttest


def _process_dirs():
  """Process multiple metrics files from baseline and test directories."""
  base_files = _process_dir(FLAGS.baseline_metrics_dir)
  test_files = _process_dir(FLAGS.test_metrics_dir)
  if not base_files:
    raise ValueError("No files found!")
  if len(base_files) != len(test_files):
    raise ValueError("Mismatching number of test sets!")
  logging.info("Loaded metrics for %d test set.", len(base_files))

  for test_set_name in base_files:
    all_base_metrics = _metric_list_to_dicts(base_files[test_set_name])
    all_test_metrics = _metric_list_to_dicts(test_files[test_set_name])
    if not all_base_metrics:
      raise ValueError(f"No metrics found for {test_set_name}!")
    if len(all_base_metrics) != len(all_test_metrics):
      raise ValueError(f"Mismatching number of metrics for {test_set_name}!")

    print(f"======================== {test_set_name} =========================")
    metrics_delta = {}
    metrics_delta["Epochs"] = [*range(1, FLAGS.num_epochs + 1)]
    for metric_name in all_base_metrics:
      deltas, _ = _process_samples(all_base_metrics[metric_name],
                                   all_test_metrics[metric_name])
      metrics_delta[metric_name] = deltas
    df = pd.DataFrame(data = metrics_delta)
    print(df.to_string(index=False))
    print(df[df.columns[1:]].describe())


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.baseline_metrics_dir or not FLAGS.test_metrics_dir:
    raise app.UsageError("Specify --baseline_metrics_dir [DIR] *and* "
                         "--test_metrics_dir [DIR]!")
  _process_dirs()


if __name__ == "__main__":
  app.run(main)
