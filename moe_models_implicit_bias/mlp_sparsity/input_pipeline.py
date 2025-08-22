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

"""Input pipeline for testing natural sparsity patterns on deep MLPs on random data.
"""

import numpy as np
import tensorflow as tf


def create_split(config, local_batch_size):
  """Prepares the dataset for training/evaluating the model."""
  ot = (tf.float32, tf.float32)
  os = (tf.TensorShape([config.dim]), tf.TensorShape([1]))
  od = RandomNoiseDataset(config)
  ds = tf.data.Dataset.from_generator(od, output_types=ot, output_shapes=os)
  options = tf.data.Options()
  options.autotune.enabled = True
  ds = ds.with_options(options)
  ds = ds.repeat()
  ds = ds.shuffle(8 * local_batch_size, seed=0)
  ds = ds.batch(local_batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


class RandomNoiseDataset:
  """Dataset where the features and the label is random noise.
  """

  def __init__(self, config):
    self.config = config
    self.len = config.num_samples
    self.x = np.random.randn(self.len, config.dim)
    self.y = np.random.randn(self.len, 1)

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    return tf.convert_to_tensor(self.x[idx]), tf.convert_to_tensor(self.y[idx])

  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)
