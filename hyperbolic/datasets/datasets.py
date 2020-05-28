# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Dataset class for loading and processing CF datasets."""

import os
import pickle as pkl

import numpy as np
import tensorflow.compat.v2 as tf


class DatasetClass(object):
  """CF dataset class."""

  def __init__(self, data_path, debug):
    """Creates CF dataset object for data loading.

    Args:
      data_path: Path to directory containing train/valid/test pickle files
        produced by process.py.
      debug: boolean indicating whether to use debug mode or not. If true, the
        dataset will only contain 1000 examples for debugging.
    """
    self.data_path = data_path
    self.debug = debug
    self.data = {}
    for split in ['train', 'test', 'valid']:
      file_path = os.path.join(self.data_path, split + '.pickle')
      with tf.gfile.Open(file_path, 'rb') as in_file:
        self.data[split] = pkl.load(in_file)
    filters_file = tf.gfile.Open(
        os.path.join(self.data_path, 'to_skip.pickle'), 'rb')
    self.to_skip = pkl.load(filters_file)
    filters_file.close()
    max_axis = np.max(
        np.concatenate(
            (self.data['train'], self.data['valid'], self.data['test']),
            axis=0),
        axis=0)
    self.n_users = int(max_axis[0] + 1)
    self.n_items = int(max_axis[1] + 1)

  def get_filters(self,):
    """Return filter dict to compute ranking metrics in the filtered setting."""
    return self.to_skip

  def get_examples(self, split):
    """Get examples in a split.

    Args:
      split: String indicating the split to use (train/valid/test).

    Returns:
      examples: tf.data.Dataset containing CF pairs in a split.
    """
    examples = self.data[split]
    if self.debug:
      examples = examples[:1000]
      examples = examples.astype(np.int64)
    tf_dataset = tf.data.Dataset.from_tensor_slices(examples)
    if split == 'train':
      buffer_size = examples.shape[0]
      tf_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    return tf_dataset

  def get_shape(self):
    """Returns CF dataset shape."""
    return self.n_users, self.n_items
