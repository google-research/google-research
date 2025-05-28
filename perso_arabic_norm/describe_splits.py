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

r"""Collects basic stats for training and test splits from the results file.

Example:
--------
  LANGUAGE=...
  cat data/ngrams/results/reading/00/baselines/${LANGUAGE}.*.tsv > /tmp/${LANGUAGE}.tsv
  python describe_splits.py \
    --results_tsv_file /tmp/${LANGUAGE}.tsv

Dependencies:
-------------
  absl
  pandas
  statsmodels
"""

from typing import Sequence

import logging

from absl import app
from absl import flags

import pandas as pd
import statsmodels.stats.api as sms

flags.DEFINE_string(
    "results_tsv_file", "",
    "Results text file in tab-separated (tsv) format.")

FLAGS = flags.FLAGS


def _to_str(stats):
  """Retrieves basic stats from the object."""
  return f"mean: {stats.mean} var: {stats.var} std: {stats.std}"


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.results_tsv_file:
    raise app.UsageError("Specify --results_tsv_file [FILE]!")

  logging.info(f"Reading metrics from {FLAGS.results_tsv_file} ...")
  df = pd.read_csv(FLAGS.results_tsv_file, sep="\t", header=None)
  logging.info(f"Read {df.shape[0]} samples")
  num_train_toks = list(df[0])  # Token can be char or word.
  train_stats = sms.DescrStatsW(num_train_toks)
  logging.info(f"Train stats: {_to_str(train_stats)}")
  num_test_toks = list(df[1])
  test_stats = sms.DescrStatsW(num_test_toks)
  logging.info(f"Test stats: {_to_str(test_stats)}")


if __name__ == "__main__":
  app.run(main)
