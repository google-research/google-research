# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collaborative Filtering MovieLens dataset pre-processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf
from hyperbolic.utils.preprocess import process_dataset
from hyperbolic.utils.preprocess import save_as_pickle


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path',
    default='data/ml-1m/ratings.dat',
    help='Path to raw dataset')
flags.DEFINE_string(
    'save_dir_path',
    default='data/ml-1m/',
    help='Path to saving directory')


def movielens_to_dict(dataset_file):
  """Maps raw dataset file to a Dictonary.

  Args:
    dataset_file: Path to file containing interactions in a format
      uid::iid::rate::time.

  Returns:
    Dictionary containing users as keys, and a numpy array of items the user
    interacted with, sorted by the time of interaction.
  """
  all_examples = {}
  with tf.gfile.Open(dataset_file, 'r') as lines:
    for line in lines:
      line = line.strip('\n').split('::')
      uid = int(line[0])-1
      iid = int(line[1])-1
      timestamp = int(line[3])
      if uid in all_examples:
        all_examples[uid].append((iid, timestamp))
      else:
        all_examples[uid] = [(iid, timestamp)]
  for uid in all_examples:
    sorted_items = sorted(all_examples[uid], key=lambda p: p[1])
    all_examples[uid] = np.array([pair[0] for pair in sorted_items
                                 ]).astype('int64')
  return all_examples


def main(_):
  dataset_path = FLAGS.dataset_path
  save_path = FLAGS.save_dir_path
  sorted_dict = movielens_to_dict(dataset_path)
  dataset_examples = process_dataset(sorted_dict)
  save_as_pickle(save_path, dataset_examples)


if __name__ == '__main__':
  app.run(main)
