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

"""Code to generate partial vectors from pi matrix."""

import json
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import mean_map_constants
import numpy as np
import pandas as pd
import tensorflow as tf


_WHICH_SPLIT = flags.DEFINE_integer('which_split', default=0, help='SPLIT idx')

_WHICH_PAIR = flags.DEFINE_integer('which_pair', default=0, help='PAIR idx')

_WHICH_SEGMENT = flags.DEFINE_integer('which_seg', default=0, help='SEG idx')

_WHICH_SIZE = flags.DEFINE_integer('which_size', default=0, help='bag size')

_BAGS_TYPE = flags.DEFINE_enum(
    'bags_type',
    default='feat',
    enum_values=['feat', 'rand', 'feat_rand'],
    help='bags type',
)


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _BAGS_TYPE.value == 'feat':
    split_no = _WHICH_SPLIT.value
    pair = mean_map_constants.PAIRS_LIST[_WHICH_PAIR.value]
    seg_no = _WHICH_SEGMENT.value
    total_num_segment = mean_map_constants.TOTAL_NUM_SEGS
    bags_file = (
        '../../data/bag_ds/split_'
        + str(split_no)
        + '/train/'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '.ftr'
    )
    maps_file = (
        '../../results/mean_map_vectors/size_matrix_map_'
        + str(split_no)
        + '_'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '.json'
    )
    write_file = (
        '../../results/mean_map_vectors/partial_vector_'
        + str(split_no)
        + '_'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '_'
        + str(seg_no)
        + '.json'
    )
  elif _BAGS_TYPE.value == 'rand':
    split_no = _WHICH_SPLIT.value
    bag_size = _WHICH_SIZE.value
    seg_no = _WHICH_SEGMENT.value
    total_num_segment = mean_map_constants.TOTAL_NUM_SEGS
    bags_file = (
        '../../data/bag_ds/split_'
        + str(split_no)
        + '/train/'
        + 'random_'
        + str(bag_size)
        + '.ftr'
    )
    maps_file = (
        '../../results/mean_map_vectors/rand_size_matrix_map_'
        + str(split_no)
        + '_'
        + 'random_'
        + str(bag_size)
        + '.json'
    )
    write_file = (
        '../../results/mean_map_vectors/rand_partial_vector_'
        + str(split_no)
        + '_'
        + 'random_'
        + str(bag_size)
        + '_'
        + str(seg_no)
        + '.json'
    )
  else:
    split_no = _WHICH_SPLIT.value
    pair = mean_map_constants.PAIRS_LIST[_WHICH_PAIR.value]
    bag_size_index = _WHICH_SIZE.value
    total_num_segment = mean_map_constants.LIST_NUM_SEGS[bag_size_index]
    bag_size = mean_map_constants.LIST_SIZES[bag_size_index]
    seg_no = _WHICH_SEGMENT.value
    bags_file = (
        '../../data/bag_ds/split_'
        + str(split_no)
        + '/train/feature_random_'
        + str(bag_size)
        + '_'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '.ftr'
    )
    maps_file = (
        '../../results/mean_map_vectors/feat_rand_size_matrix_map_'
        + str(split_no)
        + '_'
        + str(bag_size)
        + '_'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '.json'
    )
    write_file = (
        '../../results/mean_map_vectors/feat_rand_partial_vector_'
        + str(split_no)
        + '_'
        + str(bag_size)
        + '_'
        + 'C'
        + str(pair[0])
        + '_'
        + 'C'
        + str(pair[1])
        + '_'
        + str(seg_no)
        + '.json'
    )
  logging.info('Bags file %s', bags_file)
  logging.info('Maps file %s', maps_file)

  df_bags = pd.read_feather(bags_file)

  num_bags = len(df_bags.index)

  with open(maps_file, 'r') as fp:
    map_to_read = json.load(fp)

  assert num_bags == map_to_read['num_bags']

  pre_mul_matrix = map_to_read['pi_transpose_pi_inverse_x_pi_transpose']

  start_index = seg_no * (num_bags // total_num_segment)

  end_index = (seg_no + 1) * (num_bags // total_num_segment)

  if seg_no == (total_num_segment - 1):
    end_index = num_bags

  logging.info('num_bags: %d', num_bags)

  logging.info('start_index: %d, end_index: %d', start_index, end_index)

  feature_cols = mean_map_constants.FEATURE_COLS

  offsets = mean_map_constants.OFFSETS

  num_total_features = 1000254 + len(offsets)

  dense_shape = [num_total_features]

  values = np.ones([39], dtype=np.float32)

  def get_vector(indices):
    list_of_indices = [[i] for i in indices]
    return tf.sparse.SparseTensor(list_of_indices, values, dense_shape)

  def get_avg_vector(indices_list):
    sum_tensor = tf.sparse.SparseTensor(
        np.empty((0, 1), dtype=np.int64), [], dense_shape
    )
    no_of_tensors = len(indices_list)
    for indices in indices_list:
      sum_tensor = tf.sparse.add(sum_tensor, get_vector(indices))
    return sum_tensor / no_of_tensors

  p_y_equals_1 = (df_bags['label_count'].sum() * 1.0) / (
      df_bags['bag_size'].sum()
  )

  # Now compute the partial pi_transpose_pi_inverse_x_pi_transpose * mu_set

  pi_transpose_pi_inverse_x_pi_transpose_x_mu_set = tf.sparse.SparseTensor(
      np.empty((0, 1), dtype=np.int64), [], dense_shape
  )

  for i in range(start_index, end_index):
    if i % 10 == 0:
      logging.info('Done i = %d', i)

    row = df_bags.iloc[[i]]

    list_col = []

    for colname in feature_cols:
      list_col.append(row[colname].to_list()[0])

    n_array = np.array(list_col, dtype=np.int32)
    bag_x = np.transpose(n_array)

    avg_vector = get_avg_vector(bag_x.tolist())

    pi_transpose_pi_inverse_x_pi_transpose_x_mu_set = tf.sparse.add(
        pi_transpose_pi_inverse_x_pi_transpose_x_mu_set,
        avg_vector
        * (
            pre_mul_matrix[1, i] * p_y_equals_1
            - pre_mul_matrix[0, i] * (1 - p_y_equals_1)
        ),
    )

  partial_vec = tf.sparse.to_dense(
      pi_transpose_pi_inverse_x_pi_transpose_x_mu_set
  ).numpy()

  with open(write_file, 'w') as fp:
    json.dump(partial_vec, fp)


if __name__ == '__main__':
  app.run(main)
