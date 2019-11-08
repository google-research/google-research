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

"""Based on implementation of TGM layer proposed by.

Temporal Gaussian Mixture Layer for Videos
AJ Piergiovanni and Michael S. Ryoo, ICML 2019
https://arxiv.org/abs/1803.06316

and extended iTGM layer in

Evolving Space-Time Neural Architectures for Videos
AJ Piergiovanni, A. Angelova, A. Toshev, and M. S. Ryoo, ICCV 2019
https://arxiv.org/abs/1811.10636
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib.slim import initializers as contrib_slim_initializers
from tensorflow.contrib.slim import utils as contrib_slim_utils

add_arg_scope = contrib_framework.add_arg_scope


def n_element_tuple(ary, int_or_tuple):
  """Converts `int_or_tuple` to an n-ary tuple.

  This functions normalizes the input value by always returning a tuple. If a
  single value is provided, the value is broadcoast.

  Args:
    ary: The size of the expected tuple.
    int_or_tuple: A list of `ary` ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with `ary` values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if not isinstance(ary, int) or ary < 1:
    raise ValueError('`ary` must be a positive integer')

  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != ary:
      raise ValueError(
          'Must be a list with %d elements: %s' % (ary, int_or_tuple))
    return tuple([int(x) for x in int_or_tuple])

  if isinstance(int_or_tuple, int):
    return tuple([int(int_or_tuple)] * ary)

  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == ary:
      return tuple([x for x in int_or_tuple])

  raise ValueError('Must be an int, a list with %d elements or a TensorShape of'
                   ' length %d' % (ary, ary))


def get_filters(length, num, scope, init=1, dtype=tf.float32):
  """Gets the filters based on gaussian or cauchy distribution.

  Gaussian and Cauchy distributions are very similar, we find that cauchy can
  converge more quickly.

  Args:
    length: The temporal length of the filter
    num: number of distributions
    scope: variable scope
    init: std variance
    dtype: layer type
  Returns:
    the filters
  """
  with tf.variable_scope(scope):
    # create slim variables for the center and std of distribution
    center = contrib_slim.model_variable(
        'tgm-center',
        shape=[num],
        initializer=tf.initializers.random_normal(0, 0.5))
    gamma = contrib_slim.model_variable(
        'tgm-gamma',
        shape=[num],
        initializer=tf.initializers.random_normal(0, init))

    # create gaussians (eqs from paper)
    center = tf.cast(tf.tanh(center), dtype)
    gamma = tf.cast(tf.tanh(gamma), dtype)

    center = tf.expand_dims((length - 1) * (center + 1) / 2, -1)
    gamma = tf.expand_dims(
        tf.expand_dims(tf.exp(1.5 - 2 * tf.abs(gamma)), -1), -1)

    a = tf.expand_dims(tf.cast(tf.zeros(num), dtype), -1)
    a += center

    b = tf.cast(tf.range(length), dtype)
    f = b - tf.expand_dims(a, -1)
    f = f / gamma

    f = np.pi * gamma * tf.square(f) + 1
    f = 1.0 / f
    f = f / tf.expand_dims(tf.reduce_sum(f, axis=2) + 1e-6, -1)
    return tf.squeeze(f)


@add_arg_scope
def tgm_3d_conv(
    inputs,
    num_outputs,
    kernel_size,
    num,
    stride=1,
    padding='SAME',
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    trainable=True,
    scope=None,
    weights_regularizer=None,
    outputs_collection=None,
    weights_initializer=contrib_slim_initializers.xavier_initializer(),
    dtype=tf.float32):
  """iTGM inflated 3D convoltuion.

  Args:
    inputs: input tensor
    num_outputs: number of output channels
    kernel_size: size of kernel (T, H, W)
    num: number of gaussians
    stride: stride of layer int or (T,H,W)
    padding: SAME or VALID
    activation_fn: activation function to apply
    normalizer_fn: normalization fn (e.g., batch norm)
    normalizer_params: params of normalization fn
    trainable: train parameters
    scope: variable scope
    weights_regularizer: weight regularizer
    outputs_collection: graph collection to store outputs
    weights_initializer: weight initialization
    dtype: dtype of layer
  Returns:
    output tensor after iTGM conv.
  """

  with tf.variable_scope(scope, 'Conv3d', [inputs]) as sc:
    num_filters_in = contrib_slim_utils.last_dimension(
        inputs.get_shape(), min_rank=5)
    length, kernel_h, kernel_w = n_element_tuple(3, kernel_size)
    stride_d, stride_h, stride_w = n_element_tuple(3, stride)

    spatial_weight_shape = [1, kernel_h, kernel_w, num_filters_in, num_outputs]
    weight_collection = contrib_slim_utils.get_variable_collections(
        None, 'weights')

    spatial_kernel = contrib_slim.model_variable(
        'weights',
        shape=spatial_weight_shape,
        dtype=inputs.dtype.base_dtype,
        initializer=weights_initializer,
        regularizer=weights_regularizer,
        collections=weight_collection,
        trainable=trainable)

    if length > 1:
      # for now, set these to be the same
      # (i.e., number of filters after 2D conv)
      # though we could be more creative here and have
      # more/less filters intermediatly.
      c_in = num_filters_in
      c_out = num_outputs
      mixing_weights = contrib_slim.model_variable(
          'soft-attn',
          shape=[c_in * c_out, num],
          initializer=tf.initializers.truncated_normal())

      # N x L
      with tf.variable_scope('tgm-f'):
        k = get_filters(length, num, scope='tgm-f', init=0.1, dtype=dtype)

      # apply mixing weights to gaussians
      mw = tf.nn.softmax(mixing_weights, dim=1)

      # now L x num_outputs
      k = tf.transpose(tf.matmul(mw, k))

      # make this Lx1x1x1xO
      k = tf.cast(tf.reshape(k, (length, 1, 1, c_in, c_out)), dtype)

      # 2D spatial conv 1x1x1x1xO
      spatial_kernel = tf.cast(spatial_kernel, dtype)
      k = spatial_kernel * k
      outputs = tf.nn.conv3d(
          inputs,
          k,
          strides=[1, stride_d, stride_h, stride_w, 1],
          padding=padding)

    else:
      outputs = tf.nn.conv3d(
          inputs,
          spatial_kernel,
          strides=[1, stride_d, stride_h, stride_w, 1],
          padding=padding)
    if normalizer_fn:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return contrib_slim_utils.collect_named_outputs(outputs_collection,
                                                    sc.original_name_scope,
                                                    outputs)
