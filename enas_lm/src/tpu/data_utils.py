# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Data input pipeline for TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf

gfile = tf.gfile


def _pad_to_batch(batch_size, data):
  """Pad `Tensor`s in data so that `N == batch_size` and return `mask`."""
  x = data['x']

  curr_batch_size = tf.shape(x)[0]
  if curr_batch_size == batch_size:
    masks = tf.ones([batch_size], dtype=tf.float32)
    return data, masks

  batch_diff = batch_size - curr_batch_size
  padded_data = {}
  for key, val in data.items():
    val = tf.pad(val, [[0, batch_diff]] + [[0, 0]] * (val.shape.ndims - 1))
    val.set_shape([batch_size] + val.shape.as_list()[1:])
    padded_data[key] = val
  masks = tf.pad(tf.ones([curr_batch_size], dtype=tf.float32),
                 [[0, batch_diff]])
  masks.set_shape([batch_size])
  return padded_data, masks


def input_fn(params):
  """For `TPUEstimator`."""
  with gfile.GFile(params.data_path, 'rb') as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp)
    tf.logging.info('-' * 80)
    tf.logging.info('train_size: {0}'.format(np.size(x_train)))
    tf.logging.info('valid_size: {0}'.format(np.size(x_valid)))
    tf.logging.info(' test_size: {0}'.format(np.size(x_test)))

  def _build_dataset(data, batch_size, bptt_steps):
    """Create LM dataset from a `data` tensor."""
    num_batches = np.size(data) // batch_size
    data = np.reshape(data[:batch_size*num_batches], [batch_size, num_batches])
    data = np.transpose(data)
    dataset = tf.data.Dataset.from_tensor_slices({'x': data[:-1],
                                                  'y': data[1:]})
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=bptt_steps, drop_remainder=True)
    def pad_to_batch(data):
      padded_data, masks = _pad_to_batch(bptt_steps, data)
      return padded_data, masks
    dataset = dataset.map(map_func=pad_to_batch)
    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training
    return dataset

  if params.task_mode == 'train':
    return _build_dataset(x_train, params.train_batch_size, params.bptt_steps)
  elif params.task_mode == 'valid':
    return _build_dataset(x_valid, params.eval_batch_size, params.bptt_steps)
  elif params.task_mode == 'test':
    return _build_dataset(x_test, params.eval_batch_size, params.bptt_steps)
  else:
    raise ValueError('Unknown task_mode {0}'.format(params.task_mode))

