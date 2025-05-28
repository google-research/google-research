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

"""Code to generate pi matrix."""

import json
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import mean_map_constants
import numpy as np
import pandas as pd


_WHICH_SPLIT = flags.DEFINE_integer('which_split', default=0, help='SPLIT idx')

_WHICH_PAIR = flags.DEFINE_integer('which_pair', default=0, help='SPLIT idx')

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
    bag_size = None
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
  elif _BAGS_TYPE.value == 'rand':
    split_no = _WHICH_SPLIT.value
    pair = None
    bag_size = _WHICH_SIZE.value
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
  else:
    split_no = _WHICH_SPLIT.value
    pair = mean_map_constants.PAIRS_LIST[_WHICH_PAIR.value]
    bag_size = _WHICH_SIZE.value
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
  logging.info('Bags file %s', bags_file)

  df_bags = pd.read_feather(bags_file)

  num_bags = len(df_bags.index)

  logging.info('num_bags: %d', num_bags)

  pi_i_y = np.zeros((num_bags, 2))

  for i in range(num_bags):
    pi_i_y[i][1] = (df_bags['label_count'][i] * 1.0) / df_bags['bag_size'][i]
    pi_i_y[i][0] = 1 - pi_i_y[i][1]

  logging.info('Iteration over bags done')

  pi_transpose_pi = np.matmul(np.transpose(pi_i_y), pi_i_y)

  pi_transpose_pi_inverse = np.linalg.inv(pi_transpose_pi)

  pi_transpose_pi_inverse_x_pi_transpose = np.matmul(
      pi_transpose_pi_inverse, np.transpose(pi_i_y)
  )

  map_to_write = {}

  map_to_write['num_bags'] = num_bags
  map_to_write['pi_transpose_pi_inverse_x_pi_transpose'] = (
      pi_transpose_pi_inverse_x_pi_transpose
  )

  with open(maps_file, 'w') as fp:
    json.dump(map_to_write, fp)


if __name__ == '__main__':
  app.run(main)
