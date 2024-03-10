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

"""Code to generate full vector from partial vector for random bags datasets."""
import pickle
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import numpy as np
import random_mean_map_constants


_PARTIAL_VECTOR_READ_DIR = '../../results/mean_map_vectors/rand_partial_vector_'

_FULL_VECTOR_WRITE_DIR = '../../results/mean_map_vectors/rand_full_vector_'

_WHICH_SPLIT = flags.DEFINE_integer('which_split', default=0, help='SPLIT idx')

_WHICH_SIZE = flags.DEFINE_integer('which_size', default=0, help='PAIR idx')


TOTAL_NUM_SEGMENTS = 40


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  split_no = _WHICH_SPLIT.value

  bag_size = _WHICH_SIZE.value

  logging.info('Split: %d', split_no)
  logging.info('bag_size: %d', bag_size)

  offsets = random_mean_map_constants.OFFSETS

  num_total_features = 1000254 + len(offsets)

  full_vec = np.zeros(num_total_features)

  for seg_no in range(TOTAL_NUM_SEGMENTS):
    partial_vec_file = (
        _PARTIAL_VECTOR_READ_DIR
        + str(split_no)
        + '_'
        + 'random_'
        + str(bag_size)
        + '_'
        + str(seg_no)
        + '.pkl'
    )

    logging.info('partial_vec_file %s', partial_vec_file)

    partial_vec = pickle.load(partial_vec_file)

    full_vec = full_vec + partial_vec

  full_vec_file = (
      _FULL_VECTOR_WRITE_DIR
      + str(split_no)
      + '_'
      + 'random_'
      + str(bag_size)
      + '.pkl'
  )

  logging.info('full_vec_file %s', full_vec_file)

  pickle.dump(full_vec, full_vec_file)


if __name__ == '__main__':
  app.run(main)
