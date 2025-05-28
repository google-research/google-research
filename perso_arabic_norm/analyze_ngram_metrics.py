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

r"""Compare the two sets of measurements that involve n-gram LM quality metrics.

Example:
--------

1. For individual report files:

  REPORT_DIR=...
  python analyze_ngram_metrics.py \
    --baseline_metrics_tsv_file ${REPORT_DIR}/baseline_report.tsv \
    --test_metrics_tsv_file ${REPORT_DIR}/test_report.tsv

2. For directories containing multiple reports for multiple languages:

  REPORT_DIR=...
  python analyze_ngram_metrics.py \
    --baseline_metrics_dir ${REPORT_DIR}/baselines/ \
    --test_metrics_dir ${REPORT_DIR}/rewrites/ \
    --language ckb \
    --output_tex_table_file /tmp/comparison_table.tex

Dependencies:
-------------
  absl
  numpy
  pandas
  scipy
  statsmodels
"""

from typing import Sequence

import logging
import os
import pathlib

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import scipy

import utils
import stat_utils

flags.DEFINE_string(
    "baseline_metrics_tsv_file", "",
    "An input text file in tab-separated (TSV) format containing the base "
    "metrics.")

flags.DEFINE_string(
    "test_metrics_tsv_file", "",
    "An input text file in tab-separated (TSV) format containing the base "
    "metrics.")

flags.DEFINE_string(
    "baseline_metrics_dir", "",
    "Directory containing results files in tab-separated (TSV) format for the "
    "baselines.")

flags.DEFINE_string(
    "test_metrics_dir", "",
    "Directory containing results files in tab-separated (TSV) format for the "
    "tested configurations.")

flags.DEFINE_string(
    "language", "",
    "Language code to filter the files by when processing directories with "
    "multiple files.")

flags.DEFINE_string(
    "output_tex_table_file", "",
    "Output file containing all the metrics for a single language as a "
    "`&`-separated table.")

flags.DEFINE_integer(
    "float_precision", 3,
    "Floating point precision.")

FLAGS = flags.FLAGS


def _process_dir(directory):
  """Processes all the results files in the supplied directory."""
  pathlist = pathlib.Path(directory).rglob("*.tsv")
  results = []
  for path in pathlist:
    file_path = str(path)
    filename = os.path.basename(file_path)
    if filename.startswith(FLAGS.language):
      order = utils.ngram_order_from_filename(filename)
      results.append((order, file_path))
  results.sort(key=lambda x: x[0])
  return results


def _process_dirs():
  """Process multiple metrics files from baseline and test directories."""
  base_files = _process_dir(FLAGS.baseline_metrics_dir)
  test_files = _process_dir(FLAGS.test_metrics_dir)
  if len(base_files) != len(test_files):
    raise ValueError("Mismatching number of metrics files!")

  orders = []
  mean_deltas = []
  mean_deltas_percent = []
  ws_ci_low = []
  ws_ci_high = []
  ws_t_stats = []
  ws_p_values = []
  mw_t_stats = []
  mw_p_values = []
  bm_t_stats = []
  bm_p_values = []
  for i in range(len(base_files)):
    base_order, base_file = base_files[i]
    test_order, test_file = test_files[i]
    if base_order != test_order:
      raise ValueError("Mismatching n-gram orders!")
    orders.append(base_order)
    ws_stats, mw_stats, bm_stats = _process_one_pair(base_file, test_file)
    mean_deltas.append(ws_stats.mean)
    mean_deltas_percent.append(ws_stats.mean_percent)
    ws_ci_low.append(ws_stats.confidence_interval[0])
    ws_ci_high.append(ws_stats.confidence_interval[1])
    ws_t_stats.append(ws_stats.t_statistic)
    ws_p_values.append(ws_stats.p_value)
    mw_t_stats.append(mw_stats.statistic)
    mw_p_values.append(mw_stats.pvalue)
    bm_t_stats.append(bm_stats.statistic)
    bm_p_values.append(bm_stats.pvalue)

  return pd.DataFrame(data = {
      "order" : orders,
      "mean_deltas" : mean_deltas,
      "mean_delta_%" : mean_deltas_percent,
      "ws_ci_low" : ws_ci_low,
      "ws_ci_high" : ws_ci_high,
      "ws_t_stat" : ws_t_stats,
      "ws_p_val" : ws_p_values,
      "mw_t_stat" : mw_t_stats,
      "mw_p_val" : mw_p_values,
      "bm_t_stat" : bm_t_stats,
      "bm_p_value" : bm_p_values,
  })


def _process_one_pair(baseline_file, test_file):
  """Compares metrics and returns a tuple containing results of three tests."""
  base_ent = utils.read_entropies(baseline_file)
  test_ent = utils.read_entropies(test_file)

  # Analyze the metrics using parametric method: t-test, assuming that
  # baseline and test entropies are normally distributed. Use
  # Welch-Satterthwaite t-test.
  stats = stat_utils.ParameterStats.MeanDifference(base_ent, test_ent)
  print(f"t-test: {stats}")
  ws_stats = stats

  # Analyze using Mann-Whitney U and Brunner-Munzel non-parametric tests.
  # Note: Unlike the Wilcoxon-Mann-Whitney’s U test, this does not require the
  # assumption of equivariance of two groups.
  if FLAGS.alternative_hypothesis == "less":
    base_ent, test_ent = test_ent, base_ent
  stats = scipy.stats.mannwhitneyu(base_ent, test_ent,
                                   alternative=FLAGS.alternative_hypothesis)
  print(f"Mann-Whitney U: {stats}")
  mw_stats = stats
  stats = scipy.stats.brunnermunzel(base_ent, test_ent,
                                    alternative=FLAGS.alternative_hypothesis)
  print(f"Brunner-Munzel: {stats}")
  bm_stats = stats
  return ws_stats, mw_stats, bm_stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.baseline_metrics_tsv_file and not FLAGS.baseline_metrics_dir:
    raise app.UsageError("Specify --baseline_metrics_tsv_file [FILE] or "
                         "--baseline_metrics_dir [DIR]!!")
  if not FLAGS.test_metrics_tsv_file and not FLAGS.test_metrics_dir:
    raise app.UsageError("Specify --test_metrics_tsv_file [FILE] or "
                         "--test_metrics_dir [DIR]!")

  # Read the metrics from single files.
  if FLAGS.baseline_metrics_tsv_file and FLAGS.test_metrics_tsv_file:
    _process_one_pair(FLAGS.baseline_metrics_tsv_file,
                      FLAGS.test_metrics_tsv_file)
  elif FLAGS.baseline_metrics_dir and FLAGS.test_metrics_dir:
    if not FLAGS.language:
      raise app.UsageError("Specify --language [CODE]!")
    if not FLAGS.output_tex_table_file:
      raise app.UsageError("Specify --output_tex_table_file [FILE]!")
    df = _process_dirs().round(FLAGS.float_precision)
    logging.info("Saving the table to %s ...", FLAGS.output_tex_table_file)
    df.to_csv(FLAGS.output_tex_table_file, sep="&", index=None)


if __name__ == "__main__":
  app.run(main)
