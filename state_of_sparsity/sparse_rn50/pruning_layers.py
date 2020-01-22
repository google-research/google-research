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

"""Tensorflow layers with parameters for implementing pruning."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import state_of_sparsity.layers.l0_regularization as l0
import state_of_sparsity.layers.variational_dropout as vd
from tensorflow.contrib.framework.python.ops import variables  # pylint: disable=g-direct-tensorflow-import
from tensorflow.contrib.model_pruning.python.layers import layers  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import init_ops  # pylint: disable=g-direct-tensorflow-import


def get_model_variables(getter,
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
  """This ensure variables are retrieved in a consistent way for core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
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


def variable_getter(rename=None):
  """Ensures scope is respected and consistently used."""

  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return get_model_variables(getter, *args, **kwargs)

  return layer_variable_getter


# TODO(tgale): The variable naming for vd and l0 is very similar and
# makes the API for the following sparse layers a bit confusing. Try
# and clean this up and make it clearer which parameters are needed
# for each technique.
def sparse_conv2d(x,
                  units,
                  kernel_size,
                  activation=None,
                  use_bias=False,
                  kernel_initializer=None,
                  kernel_regularizer=None,
                  bias_initializer=None,
                  biases_regularizer=None,
                  sparsity_technique='baseline',
                  log_sigma2_initializer=None,
                  log_alpha_initializer=None,
                  normalizer_fn=None,
                  strides=(1, 1),
                  padding='SAME',
                  threshold=3.0,
                  clip_alpha=None,
                  data_format='channels_last',
                  is_training=False,
                  name=None):
  """Function that constructs conv2d with any desired pruning method.

  Args:
    x: Input, float32 tensor.
    units: Int representing size of output tensor.
    kernel_size: The size of the convolutional window, int of list of ints.
    activation: If None, a linear activation is used.
    use_bias: Boolean specifying whether bias vector should be used.
    kernel_initializer: Initializer for the convolution weights.
    kernel_regularizer: Regularization method for the convolution weights.
    bias_initializer: Initalizer of the bias vector.
    biases_regularizer: Optional regularizer for the bias vector.
    sparsity_technique: Method used to introduce sparsity.
           ['threshold', 'variational_dropout', 'l0_regularization']
    log_sigma2_initializer: Specified initializer of the log_sigma2 term used
      in variational dropout.
    log_alpha_initializer: Specified initializer of the log_alpha term used
      in l0 regularization.
    normalizer_fn: function used to transform the output activations.
    strides: stride length of convolution, a single int is expected.
    padding: May be populated as 'VALID' or 'SAME'.
    threshold: Theshold for masking variational dropout log alpha at test time.
    clip_alpha: Int that specifies range for clippling variational dropout
      log alpha values.
    data_format: Either 'channels_last', 'channels_first'.
    is_training: Boolean specifying whether it is training or eval.
    name: String speciying name scope of layer in network.

  Returns:
    Output: activations.

  Raises:
    ValueError: If the rank of the input is not greater than 2.
  """

  if data_format == 'channels_last':
    data_format_channels = 'NHWC'
  elif data_format == 'channels_first':
    data_format_channels = 'NCHW'
  else:
    raise ValueError('Not a valid channel string:', data_format)

  layer_variable_getter = variable_getter({
      'bias': 'biases',
      'kernel': 'weights',
  })
  input_rank = x.get_shape().ndims
  if input_rank != 4:
    raise ValueError('Rank not supported {}'.format(input_rank))

  with tf.variable_scope(
      name, 'Conv', [x], custom_getter=layer_variable_getter) as sc:

    input_shape = x.get_shape().as_list()
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Convolution` '
                       'should be defined. Found `None`.')

    pruning_methods = ['threshold']

    if sparsity_technique in pruning_methods:
      return layers.masked_conv2d(
          inputs=x,
          num_outputs=units,
          kernel_size=kernel_size[0],
          stride=strides[0],
          padding=padding,
          data_format=data_format_channels,
          rate=1,
          activation_fn=activation,
          weights_initializer=kernel_initializer,
          weights_regularizer=kernel_regularizer,
          normalizer_fn=normalizer_fn,
          normalizer_params=None,
          biases_initializer=bias_initializer,
          biases_regularizer=biases_regularizer,
          outputs_collections=None,
          trainable=True,
          scope=sc)

    elif sparsity_technique == 'variational_dropout':
      vd_conv = vd.layers.Conv2D(
          num_outputs=units,
          kernel_size=kernel_size,
          strides=strides,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          log_sigma2_initializer=log_sigma2_initializer,
          is_training=is_training,
          use_bias=use_bias,
          padding=padding,
          data_format=data_format_channels,
          clip_alpha=clip_alpha,
          threshold=threshold,
          trainable=True,
          name=sc)
      return vd_conv.apply(x)
    elif sparsity_technique == 'l0_regularization':
      l0_conv = l0.layers.Conv2D(
          num_outputs=units,
          kernel_size=kernel_size,
          strides=strides,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          log_alpha_initializer=log_alpha_initializer,
          is_training=is_training,
          use_bias=use_bias,
          padding=padding,
          data_format=data_format_channels,
          trainable=True,
          name=sc)
      return l0_conv.apply(x)
    elif sparsity_technique == 'baseline':
      return tf.layers.conv2d(
          inputs=x,
          filters=units,
          kernel_size=kernel_size,
          strides=strides,
          padding=padding,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          data_format=data_format,
          name=name)
    else:
      raise ValueError(
          'Unsupported sparsity technique {}'.format(sparsity_technique))


def sparse_fully_connected(x,
                           units,
                           activation=None,
                           use_bias=True,
                           kernel_initializer=None,
                           kernel_regularizer=None,
                           bias_initializer=init_ops.zeros_initializer(),
                           biases_regularizer=None,
                           sparsity_technique='baseline',
                           log_sigma2_initializer=None,
                           log_alpha_initializer=None,
                           threshold=3.0,
                           clip_alpha=None,
                           is_training=False,
                           name=None):
  """Constructs sparse_fully_connected with any desired pruning method.

  Args:
    x: Input, float32 tensor.
    units: Int representing size of output tensor.
    activation: If None, a linear activation is used.
    use_bias: Boolean specifying whether bias vector should be used.
    kernel_initializer: Initializer for the convolution weights.
    kernel_regularizer: Regularization method for the convolution weights.
    bias_initializer: Initalizer of the bias vector.
    biases_regularizer: Optional regularizer for the bias vector.
    sparsity_technique: Method used to introduce sparsity. ['baseline',
      'threshold', 'variational_dropout', 'l0_regularization']
    log_sigma2_initializer: Specified initializer of the log_sigma2 term used
      in variational dropout.
    log_alpha_initializer: Specified initializer of the log_alpha term used
      in l0 regularization.
    threshold: Threshold for masking variational dropout log alpha at test time.
    clip_alpha: Int that specifies range for clippling variational dropout
      log alpha values.
    is_training: Boolean specifying whether it is training or eval.
    name: String speciying name scope of layer in network.

  Returns:
    Output: activations.

  Raises:
    ValueError: If the rank of the input is not greater than 2.
  """

  layer_variable_getter = variable_getter({
      'bias': 'biases',
      'kernel': 'weights',
  })

  with tf.variable_scope(
      name, 'Dense', [x], custom_getter=layer_variable_getter) as sc:

    input_shape = x.get_shape().as_list()
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')

    pruning_methods = ['threshold']

    if sparsity_technique in pruning_methods:
      return layers.masked_fully_connected(
          inputs=x,
          num_outputs=units,
          activation_fn=activation,
          weights_initializer=kernel_initializer,
          weights_regularizer=kernel_regularizer,
          biases_initializer=bias_initializer,
          biases_regularizer=biases_regularizer,
          outputs_collections=None,
          trainable=True,
          scope=sc)

    elif sparsity_technique == 'variational_dropout':
      vd_fc = vd.layers.FullyConnected(
          num_outputs=units,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          log_sigma2_initializer=log_sigma2_initializer,
          is_training=is_training,
          use_bias=use_bias,
          clip_alpha=clip_alpha,
          threshold=threshold,
          trainable=True,
          name=sc)
      return vd_fc.apply(x)
    elif sparsity_technique == 'l0_regularization':
      l0_fc = l0.layers.FullyConnected(
          num_outputs=units,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          log_alpha_initializer=log_alpha_initializer,
          is_training=is_training,
          use_bias=use_bias,
          trainable=True,
          name=sc)
      return l0_fc.apply(x)
    elif sparsity_technique == 'baseline':
      return tf.layers.dense(
          inputs=x,
          units=units,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_initializer=bias_initializer,
          bias_regularizer=biases_regularizer,
          name=name)
    else:
      raise ValueError(
          'Unsupported sparsity technique {}'.format(sparsity_technique))
