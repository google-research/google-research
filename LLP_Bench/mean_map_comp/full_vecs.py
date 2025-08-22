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

"""Code to generate full vector from partial vector."""

import json
from typing import Sequence

from absl import app
from absl import flags
import mean_map_constants
import numpy as np


_WHICH_SPLIT = flags.DEFINE_integer('which_split', default=0, help='SPLIT idx')

_WHICH_PAIR = flags.DEFINE_integer('which_pair', default=0, help='PAIR idx')

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
    total_num_segment = mean_map_constants.TOTAL_NUM_SEGS
    bag_size = None
  elif _BAGS_TYPE.value == 'rand':
    split_no = _WHICH_SPLIT.value
    bag_size = _WHICH_SIZE.value
    total_num_segment = mean_map_constants.TOTAL_NUM_SEGS
    pair = None
  else:
    split_no = _WHICH_SPLIT.value
    pair = mean_map_constants.PAIRS_LIST[_WHICH_PAIR.value]
    bag_size_index = _WHICH_SIZE.value
    total_num_segment = mean_map_constants.LIST_NUM_SEGS[bag_size_index]
    bag_size = mean_map_constants.LIST_SIZES[bag_size_index]

  offsets = mean_map_constants.OFFSETS
  num_total_features = 1000254 + len(offsets)
  full_vec = np.zeros(num_total_features)
  for seg_no in range(total_num_segment):
    if _BAGS_TYPE.value == 'feat':
      partial_vec_file = (
          '../../results/mean_map_vectors/partial_vector_'
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
      partial_vec_file = (
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
      partial_vec_file = (
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
          + '.json'
      )
    with open(partial_vec_file, 'r') as fp:
      partial_vec = json.load(fp)
    full_vec = full_vec + partial_vec
  if _BAGS_TYPE.value == 'feat':
    full_vec_file = (
        '../../results/mean_map_vectors/full_vector_'
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
    full_vec_file = (
        '../../results/mean_map_vectors/rand_full_vector_'
        + str(split_no)
        + '_'
        + 'random_'
        + str(bag_size)
        + '.json'
    )
  else:
    full_vec_file = (
        '../../results/mean_map_vectors/feat_rand_full_vector_'
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
  with open(full_vec_file, 'w') as fp:
    json.dump(full_vec, fp)


if __name__ == '__main__':
  app.run(main)
