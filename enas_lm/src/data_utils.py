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

"""Load picked Penn Treebank data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def input_producer(raw_data, batch_size, num_steps, shuffle=False,
                   randomize=False, random_len=False):
  """Produces graph-based input for Penn Treebank.

  Args:
    raw_data: np tensor of size [num_words].
    batch_size: self-explained.
    num_steps: number of BPTT steps.
    shuffle: whether to shuffle sentences.
    randomize: use random segments instead of the continuous corpus.
    random_len: random sequence len.

  Returns:
    If `random_len` is set, return op that represents whether we have reached
      the end of a sequence.
    Otherwise, return number of batches in an epoch.
  """

  num_batches_per_epoch = ((np.size(raw_data) // batch_size) - 1) // num_steps
  raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)

  data_len = tf.size(raw_data)
  batch_len = data_len // batch_size
  data = tf.reshape(raw_data[0 : batch_size * batch_len],
                    [batch_size, batch_len])

  epoch_size = (batch_len - 1) // num_steps
  with tf.device('/cpu:0'):
    epoch_size = tf.identity(epoch_size, name='epoch_size')

    if random_len:
      start_idx = tf.Variable(0, name='start_idx', dtype=tf.int32,
                              trainable=False)
      base_bptt = tf.cond(
          tf.random_uniform(shape=(), minval=0., maxval=1.) < 0.95,
          lambda: tf.cast(num_steps, dtype=tf.float32),
          lambda: tf.cast(num_steps, dtype=tf.float32) / 2.)
      seq_len = tf.random.truncated_normal(shape=(), mean=base_bptt, stddev=5.,
                                           dtype=tf.float32)
      seq_len = tf.cast(seq_len, dtype=tf.int32)
      seq_len = tf.minimum(seq_len, num_steps + 20)  # seq_len <= bptt + 40
      seq_len = tf.minimum(seq_len, batch_len - start_idx - 1)
      end_idx = start_idx + seq_len

      x = data[:, start_idx : end_idx]
      y = data[:, start_idx+1 : end_idx+1]

      with tf.control_dependencies([x, y]):
        with tf.control_dependencies([tf.assign(start_idx, end_idx)]):
          should_reset = tf.greater_equal(end_idx, batch_len - 3)

      reset_start_idx = tf.assign(start_idx, 0)
      return (x, y, num_batches_per_epoch, reset_start_idx, should_reset,
              base_bptt)

    if randomize:
      i = tf.random_uniform([1], minval=0, maxval=batch_len - num_steps,
                            dtype=tf.int32)
      x = tf.strided_slice(data, [0, i], [batch_size, i + num_steps])
      y = tf.strided_slice(data, [0, i + 1], [batch_size, i + num_steps + 1])
    else:
      i = tf.train.range_input_producer(epoch_size, shuffle=shuffle).dequeue()
      x = tf.strided_slice(
          data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
      y = tf.strided_slice(
          data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])

    return x, y, num_batches_per_epoch

