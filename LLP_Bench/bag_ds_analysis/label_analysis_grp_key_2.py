# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Performs analysis of bag size distribution and label proportion distribution on dataset with grp key size 1."""
from collections.abc import Sequence
import math
import pickle

from absl import app
from absl import flags
from absl import logging
import analysis_constants
import numpy as np
import pandas as pd


_C1 = flags.DEFINE_integer('c1', 1, 'c1?')
_C2 = flags.DEFINE_integer('c2', 2, 'c2?')


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  saving_dir = '../results/'
  sparse_cols = [analysis_constants.C + str(i) for i in range(1, 27)]
  dense_cols = [analysis_constants.I + str(i) for i in range(1, 14)]
  all_cols = dense_cols + sparse_cols
  list_of_cols_test = ['label'] + all_cols
  criteo_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo.csv',
      usecols=list_of_cols_test,
  )
  logging.info('DataFrame Loaded')
  num_instances = len(criteo_df)
  c1 = analysis_constants.C + str(_C1.value)
  c2 = analysis_constants.C + str(_C2.value)
  df_agg = criteo_df.groupby([c1, c2])
  num_bags = len(df_agg)
  df_filtered = df_agg.filter(lambda x: ((len(x) >= 50) and (len(x) <= 2500)))
  df_filtered_agg = df_filtered.groupby([c1, c2])
  num_filtered_instances = len(df_filtered)
  # pylint: disable=unnecessary-lambda
  bag_sizes = df_filtered_agg.apply(lambda x: len(x))
  num_pos_labels = df_filtered_agg.apply(lambda x: np.sum(x['label']))
  num_filtered_bags = len(bag_sizes)
  logging.info('Grouping and filtering done')
  bag_sizes_sorted = np.sort(bag_sizes)
  bag_size_10_percentile = bag_sizes_sorted[math.floor(0.1 * num_filtered_bags)]
  bag_size_20_percentile = bag_sizes_sorted[math.floor(0.2 * num_filtered_bags)]
  bag_size_30_percentile = bag_sizes_sorted[math.floor(0.3 * num_filtered_bags)]
  bag_size_40_percentile = bag_sizes_sorted[math.floor(0.4 * num_filtered_bags)]
  bag_size_50_percentile = bag_sizes_sorted[math.floor(0.5 * num_filtered_bags)]
  bag_size_60_percentile = bag_sizes_sorted[math.floor(0.6 * num_filtered_bags)]
  bag_size_70_percentile = bag_sizes_sorted[math.floor(0.7 * num_filtered_bags)]
  bag_size_80_percentile = bag_sizes_sorted[math.floor(0.8 * num_filtered_bags)]
  bag_size_85_percentile = bag_sizes_sorted[
      math.floor(0.85 * num_filtered_bags)
  ]
  bag_size_90_percentile = bag_sizes_sorted[math.floor(0.9 * num_filtered_bags)]
  bag_size_95_percentile = bag_sizes_sorted[
      math.floor(0.95 * num_filtered_bags)
  ]
  bag_size_97_percentile = bag_sizes_sorted[
      math.floor(0.97 * num_filtered_bags)
  ]
  bag_size_99_percentile = bag_sizes_sorted[
      math.floor(0.99 * num_filtered_bags)
  ]
  bag_size_99_9_percentile = bag_sizes_sorted[
      math.floor(0.999 * num_filtered_bags)
  ]
  logging.info('Histogram informations computed')
  label_prop = np.array(num_pos_labels / bag_sizes)
  mu = np.sum(df_filtered['label']) / num_filtered_instances
  log_mu = math.log(mu)
  log_1_mu = math.log(1 - mu)
  comb = np.vectorize(
      lambda x, y: math.log(math.comb(x, y)) + y * log_mu + (x - y) * (log_1_mu)
  )
  avg_ll = np.mean(comb(np.array(bag_sizes), np.array(num_pos_labels)))
  data = {
      'c1': c1,
      'c2': c2,
      'num_instances': num_instances,
      'num_bags': num_bags,
      'num_filtered_instances': num_filtered_instances,
      'percentage_instances_left': (
          num_filtered_instances * 100
      ) / num_instances,
      'num_filtered_bags': num_filtered_bags,
      'min_bag_size': bag_sizes.min(),
      'max_bag_size': bag_sizes.max(),
      'mean_bag_size': np.mean(bag_sizes),
      'std_bag_sizes': np.std(bag_sizes),
      'bag_size_10_percentile': bag_size_10_percentile,
      'bag_size_20_percentile': bag_size_20_percentile,
      'bag_size_30_percentile': bag_size_30_percentile,
      'bag_size_40_percentile': bag_size_40_percentile,
      'bag_size_50_percentile': bag_size_50_percentile,
      'bag_size_60_percentile': bag_size_60_percentile,
      'bag_size_70_percentile': bag_size_70_percentile,
      'bag_size_80_percentile': bag_size_80_percentile,
      'bag_size_85_percentile': bag_size_85_percentile,
      'bag_size_90_percentile': bag_size_90_percentile,
      'bag_size_95_percentile': bag_size_95_percentile,
      'bag_size_97_percentile': bag_size_97_percentile,
      'bag_size_99_percentile': bag_size_99_percentile,
      'bag_size_99_9_percentile': bag_size_99_9_percentile,
      'min_label_prop': np.min(label_prop),
      'max_label_prop': np.max(label_prop),
      'mean_label_prop': np.mean(label_prop),
      'std_label_prop': np.std(label_prop),
      'bernoulli_parameter_data_distribution': mu,
      'average_log_likelihood_of_label_distribution': avg_ll,
  }
  pickle.dump(data, saving_dir + 'metrics_dicts/' + c1 + '_' + c2 + '.pkl')
  logging.info('Program Finished')


if __name__ == '__main__':
  app.run(main)
