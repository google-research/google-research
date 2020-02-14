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

# Lint as: python3

r"""Tensorflow layers with added variables for parameter masking.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from pruning_identified_exemplars.pruning_tools import core_layers as core
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import add_to_collections
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as tf_variables
# pylint: enable=g-direct-tensorflow-import


def variance_scaling_initializer(factor=2.0,
                                 mode='FAN_IN',
                                 uniform=False,
                                 seed=None,
                                 dtype=dtypes.float32):
  """Returns an initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. This initializer use the following formula:

  ```python
    if mode='FAN_IN': # Count only number of input connections.
      n = fan_in
    elif mode='FAN_OUT': # Count only number of output connections.
      n = fan_out
    elif mode='FAN_AVG': # Average number of inputs and output connections.
      n = (fan_in + fan_out)/2.0

      truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
  ```

  * To get [Delving Deep into Rectifiers](
     http://arxiv.org/pdf/1502.01852v1.pdf) (also know as the "MSRA
     initialization"), use (Default):<br/>
    `factor=2.0 mode='FAN_IN' uniform=False`
  * To get [Convolutional Architecture for Fast Feature Embedding](
     http://arxiv.org/abs/1408.5093), use:<br/>
    `factor=1.0 mode='FAN_IN' uniform=True`
  * To get [Understanding the difficulty of training deep feedforward neural
    networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
    use:<br/>
    `factor=1.0 mode='FAN_AVG' uniform=True.`
  * To get `xavier_initializer` use either:<br/>
    `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
    `factor=1.0 mode='FAN_AVG' uniform=False`.

  Args:
    factor: Float.  A multiplicative factor.
    mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
    uniform: Whether to use uniform or normal distributed random initialization.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with unit variance.

  Raises:
    ValueError: if `dtype` is not a floating point type.
    TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
  """
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')
  if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
    raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]' % mode)

  def _initializer(shape, dtype=dtype, partition_info=None):
    """Initializer function."""
    del partition_info
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
    if uniform:
      # To get stddev = math_ops.sqrt(factor / n) need to adjust for uniform.
      limit = math_ops.sqrt(3.0 * factor / n)
      return random_ops.random_uniform(shape, -limit, limit, dtype, seed=seed)
    else:
      # To get stddev = math_ops.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math_ops.sqrt(1.3 * factor / n)
      return random_ops.truncated_normal(
          shape, 0.0, trunc_stddev, dtype, seed=seed)

  return _initializer


def xavier_initializer(uniform=True, seed=None, dtype=dtypes.float32):
  """Returns an initializer performing "Xavier" initialization for weights."""
  return variance_scaling_initializer(
      factor=1.0, mode='FAN_AVG', uniform=uniform, seed=seed, dtype=dtype)


def get_variable_collections(variables_collections, name):
  if isinstance(variables_collections, dict):
    variable_collections = variables_collections.get(name, None)
  else:
    variable_collections = variables_collections
  return variable_collections


def collect_named_outputs(collections, alias, outputs):
  """Add `Tensor` outputs tagged with alias to collections."""
  if collections:
    append_tensor_alias(outputs, alias)
    add_to_collections(collections, outputs)
  return outputs


def append_tensor_alias(tensor, alias):
  """Append an alias to the list of aliases of the tensor.

  Args:
    tensor: A `Tensor`.
    alias: String, to add to the list of aliases of the tensor.

  Returns:
    The tensor with a new alias appended to its list of aliases.
  """
  # Remove ending '/' if present.
  if alias[-1] == '/':
    alias = alias[:-1]
  if hasattr(tensor, 'aliases'):
    tensor.aliases.append(alias)
  else:
    tensor.aliases = [alias]
  return tensor


def _model_variable_getter(getter,
                           name,
                           shape=None,
                           dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           rename=None,
                           use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  name_components = name.split('/')
  short_name = name_components[-1]
  if rename and short_name in rename:
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return tf_variables.model_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      collections=collections,
      trainable=trainable,
      caching_device=caching_device,
      partitioner=partitioner,
      custom_getter=getter,
      use_resource=use_resource)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""

  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)

  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = get_variable_collections(collections_set,
                                         collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf1.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in tf.get_collection(collection):
        tf.add_to_collection(collection, var)


def masked_convolution(inputs,
                       num_outputs,
                       kernel_size,
                       stride=1,
                       padding='SAME',
                       data_format=None,
                       rate=1,
                       activation_fn=tf.nn.relu,
                       normalizer_fn=None,
                       normalizer_params=None,
                       weights_initializer=xavier_initializer(),
                       weights_regularizer=None,
                       biases_initializer=tf.zeros_initializer(),
                       biases_regularizer=None,
                       reuse=None,
                       variables_collections=None,
                       outputs_collections=None,
                       trainable=True,
                       scope=None):
  """Adds an 2D convolution followed by a optional normalizer layer."""
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = _build_variable_getter({
      'bias': 'biases',
      'kernel': 'weights'
  })

  with tf1.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = tf.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    if input_rank == 4:
      layer_class = core.MaskedConv2D
    else:
      raise ValueError('Sparse Convolution not supported for input with rank',
                       input_rank)

    if data_format is None or data_format == 'NHWC':
      df = 'channels_last'
    elif data_format == 'NCHW':
      df = 'channels_first'
    else:
      raise ValueError('Unsupported data format', data_format)

    layer = layer_class(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        data_format=df,
        dilation_rate=rate,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return collect_named_outputs(outputs_collections, sc.original_name_scope,
                                 outputs)


masked_conv2d = masked_convolution
