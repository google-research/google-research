# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Common ops."""
import string
import numpy as np

import tensorflow.compat.v1 as tf


##### Initializers
WEIGHT_INITIALIZER = tf.random_normal_initializer(stddev=0.02)
BIAS_INITIALIZER = tf.zeros_initializer


##### Common modules
def layer_norm(x, begin_axis=-1, eps=1e-5, name=None, gamma_initializer=None):
  """Layer normalization."""
  shape = x.shape.as_list()
  norm_shape = shape[begin_axis:]
  axes = list(range(len(shape)))[begin_axis:]
  with tf.variable_scope(name, default_name='norm'):
    gamma = tf.get_variable(
        'gamma',
        shape=norm_shape,
        initializer=gamma_initializer or tf.initializers.ones(),
        dtype=x.dtype)
    beta = tf.get_variable(
        'beta',
        shape=norm_shape,
        initializer=tf.initializers.zeros(),
        dtype=x.dtype)
    mean, var = tf.nn.moments(x, axes, keepdims=True)
    return (x - mean) * tf.rsqrt(var + eps) * gamma + beta


def gelu(x):
  """GeLU activation function."""
  return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def dropout(x, is_training, rate=0):
  """Dropout with rate."""
  x = tf.keras.layers.Dropout(rate)(x, training=is_training)
  return x


def trail_dense(x, output_shape, begin_axis=-1, bias=True, name=None,
                kernel_initializer=WEIGHT_INITIALIZER,
                bias_initializer=BIAS_INITIALIZER):
  """A dense layer that projects x[begin_axis:] to output_shape."""
  if isinstance(output_shape, int):
    output_shape = [output_shape]
  else:
    output_shape = list(output_shape)

  input_shape = x.shape.as_list()
  input_rank = len(input_shape)
  shared_size = begin_axis % input_rank
  i_only_size = input_rank - shared_size
  o_only_size = len(output_shape)

  assert input_rank + o_only_size < len(string.ascii_lowercase)
  einsum_str = string.ascii_lowercase[:input_rank + o_only_size]

  offset = 0
  shared_str = einsum_str[offset:offset+shared_size]
  offset += shared_size
  i_only_str = einsum_str[offset:offset+i_only_size]
  offset += i_only_size
  o_only_str = einsum_str[offset:offset+o_only_size]

  input_str = '{}{}'.format(shared_str, i_only_str)
  output_str = '{}{}'.format(shared_str, o_only_str)
  weight_str = '{}{}'.format(i_only_str, o_only_str)
  weight_shape = input_shape[begin_axis:] + output_shape

  # Actual computation
  with tf.variable_scope(name, default_name='dense'):
    weight = tf.get_variable(
        'weight',
        shape=weight_shape,
        initializer=kernel_initializer,
        dtype=x.dtype)
    einsum_expr = '{},{}->{}'.format(input_str, weight_str, output_str)
    output = tf.einsum(einsum_expr, x, weight)

    if bias:
      bias = tf.get_variable(
          'bias',
          shape=output_shape,
          initializer=bias_initializer,
          dtype=x.dtype)
      output += bias

  return output


def ffn(h, is_training, dropffn=0.1, expansion_rate=4):
  """FFN."""
  hidden_size = h.shape[-1].value
  h = trail_dense(h, hidden_size * expansion_rate)
  h = gelu(h)
  h = dropout(h, is_training, rate=dropffn)
  h = trail_dense(h, hidden_size)
  return h


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training or survival_prob is None or survival_prob == 1.0:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size], dtype=inputs.dtype)
  for _ in range(inputs.shape.rank - 1):
    random_tensor = tf.expand_dims(random_tensor, axis=-1)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


def preprocess(x, config):
  """Preprocess: pre-normalization or post-normalization."""
  pre_norm = getattr(config, 'pre_norm', True)
  if pre_norm:
    shortcut = x
    normed = layer_norm(x, name='pre_norm')
  else:
    normed = layer_norm(x, name='sum_norm')
    shortcut = normed
  return shortcut, normed


def postprocess(shortcut, x, config, is_training):
  """Postprocess: dropout + stochstic residual."""
  x = dropout(x, is_training, config.dropout)
  x = drop_connect(x, is_training, getattr(config, 'survival_prob', None))
  x = shortcut + x
  return x

