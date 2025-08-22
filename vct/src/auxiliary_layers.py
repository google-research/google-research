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

"""Auxiliary layers."""

from typing import Optional

import tensorflow as tf


def _shift_to_the_right(x,
                        pad = None):
  """Returns essentially [pad, x[:-1]] but using the right dimensions."""
  *dims, _, c = x.shape
  expected_pad_shape = (*dims, 1, c)
  if pad is None:
    pad = tf.zeros(expected_pad_shape, dtype=x.dtype)
  elif pad.shape != expected_pad_shape:
    raise ValueError(f"Invalid shape: {pad.shape} != {expected_pad_shape}")
  return tf.concat((pad, x[Ellipsis, :-1, :]), axis=-2)


def make_embedding_layer(num_channels,
                         d_model):
  """Creates an embedding layer."""
  scale = 1 / num_channels**0.5
  resample_kernel_init = tf.random_uniform_initializer(-scale, scale)
  resample_bias_init = tf.random_uniform_initializer(-scale, scale)
  return tf.keras.layers.Dense(
      units=d_model,
      kernel_initializer=resample_kernel_init,
      bias_initializer=resample_bias_init,
      use_bias=True,
      name="linear_reduction")


class StartSym(tf.keras.layers.Layer):
  """Helper to learn a "zero" symbol, i.e., the first symbol to feed."""

  def __init__(self, num_channels):
    super().__init__()

    def initializer(shape, dtype):
      return tf.random.uniform(shape, -3, 3, dtype, seed=42)

    self.sym = self.add_weight(
        shape=(num_channels,),
        initializer=initializer,
        trainable=True,
        name="sym",
    )

  def call(self, x):
    """Prefixes `x` with the learned start symbol."""
    b, _, c = x.shape
    return _shift_to_the_right(x, self.sym * tf.ones((b, 1, c)))


class LearnedPosition(tf.keras.layers.Layer):
  """Single learned positional encoding."""

  def __init__(
      self,
      name,
      seq_len,
      d_model,
  ):
    super().__init__()
    self._emb = self.add_weight(
        initializer=tf.random_normal_initializer(stddev=0.02),
        trainable=True,
        dtype=tf.float32,
        shape=[seq_len, d_model],
        name=name)
    self._seq_len = seq_len
    self._d_model = d_model

  def __call__(self, tensor):
    """Adds positional encodings to `tensor`."""
    expected = (self._seq_len, self._d_model)
    if tensor.shape[-2:] != expected:
      raise ValueError(f"Invalid shape, {tensor.shape[-2:]} != {expected}")
    return tensor + self._emb
