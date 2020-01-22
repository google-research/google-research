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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.tpu import tpu_function


def np_gumbel(np_random, shape, temperature, u_low=1e-9, u_high=1.0):
  u = np_random.uniform(low=u_low, high=u_high, size=shape)
  return -np.log(-np.log(u)) * temperature


def gumbel(shape, temperature, u_low=1e-9, u_high=1.0):
  u = tf.random_uniform(shape=shape, minval=u_low, maxval=u_high)
  return -tf.log(-tf.log(u)) * temperature


def chans_to_rows(x):
  """Stack channels vertically into a 1-channel image with 3x the height."""
  B, H, W, C, D = x.shape  # pylint: disable=invalid-name
  x = tf.transpose(x, [0, 3, 1, 2, 4])
  assert x.shape == (B, C, H, W, D)
  x = tf.reshape(x, [B, C * H, W, D])
  return x


def rows_to_chans(x, channels):
  B, H, W, D = x.shape  # pylint: disable=invalid-name
  assert H % channels == 0
  x = tf.reshape(x, [B, channels, H // channels, W, D])
  x = tf.transpose(x, [0, 2, 3, 1, 4])
  return x


def chans_to_interleaved_cols(x):
  B, H, W, C, D = x.shape  # pylint: disable=invalid-name
  return tf.reshape(x, [B, H, W * C, D])


def interleaved_cols_to_chans(x, channels):
  B, H, W, D = x.shape  # pylint: disable=invalid-name
  return tf.reshape(x, [B, H, W // channels, channels, D])


def get_warmed_up_lr(max_lr, warmup, global_step):
  if warmup == 0:
    return max_lr
  return max_lr * tf.minimum(
      tf.cast(global_step, tf.float32) / float(warmup), 1.0)


def make_train_op(optimizer, loss, trainable_variables, global_step,
                  grad_clip_norm):
  num_cores = tpu_function.get_tpu_context().number_of_shards

  # compute scaled gradient
  grads_and_vars = optimizer.compute_gradients(
      loss / float(num_cores), var_list=trainable_variables)

  # clip gradient
  clipped_grads, gnorm = tf.clip_by_global_norm(
      [g for (g, _) in grads_and_vars], grad_clip_norm / float(num_cores))
  grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

  # optimize
  optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  return train_op, gnorm


def rms(variables):
  return tf.sqrt(
      sum([tf.reduce_sum(tf.square(v)) for v in variables]) /
      sum(int(np.prod(v.shape.as_list())) for v in variables))
