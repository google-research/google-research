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

"""Neural network utilities and layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import numpy as np
import tensorflow as tf


class Module(tf.Module):

  @property
  def variable_scope(self):
    scope_name = self._scope_name
    if scope_name.endswith('/'):
      scope_name = scope_name[:-1]
    return tf.variable_scope(scope_name)


def nonlinearity(x):
  return x * tf.sigmoid(1.702 * x)


def flatten(x):
  return tf.reshape(x, [int(x.shape[0]), -1])


def shift(x, axis, num):
  """Shift a tensor (image) by padding.

  Inserts padding on one side and drops data on the other side.

  Args:
    x: input tensor
    axis: Shifting axis
    num: Number of pixels to shift. Positive means to pad at the beginning,
      negative means to pad at the end.

  Returns:
    Shifted tensor. Same shape as `x`
  """
  paddings = [([max(num, 0), -min(num, 0)] if i == axis else [0, 0])
              for i in range(len(x.shape))]
  slices = tuple([(slice(-num if num < 0 else None, -num if num > 0 else None)
                   if i == axis else slice(None)) for i in range(len(x.shape))])
  out = tf.pad(x[slices], paddings)
  assert out.shape == x.shape
  return out


def shift_down(imgs):
  assert len(imgs.shape) == 4
  return shift(imgs, axis=1, num=1)


def shift_right(imgs):
  assert len(imgs.shape) == 4
  return shift(imgs, axis=2, num=1)


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


def attn_nd(q, k, v, time_axis, feat_axis, masked):
  assert q.shape == k.shape == v.shape
  assert time_axis != feat_axis
  num_axes = len(q.shape)
  head_dim, num_timesteps = q.shape[feat_axis], q.shape[time_axis]
  letters = string.ascii_lowercase[:num_axes]
  assert len(letters) == num_axes, 'too many axes'

  q_str, k_str, w_str = map(list, [letters] * 3)
  k_str[time_axis] = k_str[time_axis].upper()
  del w_str[feat_axis]
  w_str.append(k_str[time_axis])

  w = _einsum(q_str, k_str, w_str, q, k) / np.sqrt(int(head_dim))
  if masked:
    mask_shape = [1] * len(w.shape)
    mask_shape[time_axis] = mask_shape[-1] = num_timesteps
    ts = tf.range(num_timesteps, dtype=tf.int32)
    mask = ts[:, None] >= ts[None, :]
    mask = tf.reshape(tf.cast(mask, w.dtype), mask_shape)
    w = w * mask - 1e9 * (1 - mask)
  w = tf.nn.softmax(w)
  return _einsum(w_str, k_str, q_str, w, v)


class Dense(Module):

  def __init__(self, in_dim, num_units, init_scale=1.0, name=None):
    super(Dense, self).__init__(name=name)
    if not isinstance(num_units, (tuple, list)):
      num_units = [num_units]
    self.num_units = num_units = list(num_units)
    self.in_dim = in_dim
    with self.variable_scope:
      self.w = tf.get_variable(
          'w',
          shape=[in_dim, int(np.prod(num_units))],
          initializer=tf.initializers.variance_scaling(scale=init_scale)
      )
      self.b = tf.get_variable(
          'b',
          shape=[int(np.prod(num_units))],
          initializer=tf.zeros_initializer())

  @Module.with_name_scope
  def __call__(self, x):
    assert x.shape[-1] == self.in_dim
    y = (
        contract_inner(x, tf.reshape(self.w, [self.in_dim] + self.num_units)) +
        tf.reshape(self.b, self.num_units))
    assert y.shape == x.shape[:-1] + self.num_units
    return y


class Conv2d(Module):

  def __init__(self,
               in_dim,
               num_units,
               filter_size=(3, 3),
               init_scale=1.0,
               name=None):
    super(Conv2d, self).__init__(name=name)

    assert len(filter_size) == 2

    with self.variable_scope:
      self.w = tf.get_variable(
          'w',
          shape=list(filter_size) + [in_dim, num_units],
          initializer=tf.initializers.variance_scaling(scale=init_scale))
      self.b = tf.get_variable(
          'b', shape=[num_units], initializer=tf.zeros_initializer())

  @Module.with_name_scope
  def __call__(self, x):
    return tf.nn.conv2d(x, self.w, strides=1, padding='SAME') + self.b


class LayerNorm(Module):

  def __init__(self, dim, eps=1e-5, name=None):
    super(LayerNorm, self).__init__(name=name)
    self.eps = eps
    with self.variable_scope:
      self.g = tf.get_variable(
          'g', shape=[dim], initializer=tf.ones_initializer())
      self.b = tf.get_variable(
          'b', shape=[dim], initializer=tf.zeros_initializer())

  @Module.with_name_scope
  def __call__(self, x):
    assert [x.shape[-1]] == self.g.shape == self.b.shape
    u = tf.reduce_mean(x, axis=-1, keepdims=True)
    v = tf.reduce_mean(tf.squared_difference(x, u), axis=-1, keepdims=True)
    return (x - u) * tf.rsqrt(v + self.eps) * self.g + self.b
