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

# Lint as: python2, python3
"""Utilities for translating V3 model_spec objects to rematlib Layers.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from typing import Any, List, Mapping, Optional, Text, Tuple, TypeVar, Union
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import depthwise_initializers
from tunas import mobile_search_space_v3
from tunas import schema
from tunas import search_space_utils
from tunas.rematlib import layers


_SQUEEZE_AND_EXCITE_RATIO = 0.25


def _to_filters_mask(value):
  """Convert a Tensor or int into a rank-1 float32 mask."""
  if isinstance(value, tf.Tensor):
    # If `value` is already a float32 Tensor, conver_to_tensor() will be a
    # no-op. If value has a dtype other than float32, we'll raise an exception.
    return tf.convert_to_tensor(value, tf.float32)
  else:
    return tf.ones([int(value)], dtype=tf.float32)


def _l2_regularizer(decay):
  # Divide by 2 for backwards compatibility w/ tf.contrib.layers.l2_regularizer.
  return tf.keras.regularizers.l2(decay / 2)


_T = TypeVar('_T', int, tf.Tensor)


def _compute_filters_for_multiplier(
    multiplier,
    input_filters_or_mask,
    filters_base):
  """Convert a FilterMultiplier to an integer (or int Tensor) filter size."""
  if isinstance(input_filters_or_mask, int):
    input_filters = input_filters_or_mask
    return search_space_utils.scale_filters(
        input_filters, multiplier.scale, filters_base)
  elif isinstance(input_filters_or_mask, tf.Tensor):
    input_filters = tf.reduce_sum(tf.cast(input_filters_or_mask, tf.int32))
    return search_space_utils.tf_scale_filters(
        input_filters, multiplier.scale, filters_base)
  else:
    raise ValueError('Unsupported type for input_filters_or_mask: {}'.format(
        input_filters_or_mask))


def _compute_filters(
    value,
    input_filters_or_mask,
    filters_base):
  """Compute the absolute number of filters from an int or FilterMultiplier."""
  if isinstance(value, int):
    return value
  elif isinstance(value, basic_specs.FilterMultiplier):
    if input_filters_or_mask is None:
      raise ValueError(
          'Relative filter multipliers are not supported when '
          'input_filters_or_mask is None.')
    if filters_base is None:
      raise ValueError(
          'Relative filter multipliers are not supported when '
          'filters_base is None.')
    return  _compute_filters_for_multiplier(
        value, input_filters_or_mask, filters_base)
  else:
    raise ValueError(
        'Filters choices must be integers or FilterMultiplier objects: {}'
        .format(value))


def _oneof_filters_to_int_or_mask(
    value,
    input_filters_or_mask = None,
    filters_base = None
):
  """Convert a OneOf or int to either an int or a rank-1 float mask."""
  if isinstance(value, schema.OneOf):
    choices = value.choices
    mask = value.mask
  elif isinstance(value, (int, basic_specs.FilterMultiplier)):
    choices = [value]
    mask = None
  else:
    raise ValueError('Must be a OneOf or FilterMultiplier: {}'.format(value))

  # Generate a list of candidate filter sizes. Each filter size can either be
  # an int or a scalar int Tensor.
  scaled_choices = []  # type: List[Union[int, tf.Tensor]]
  for choice in choices:
    scaled_choices.append(
        _compute_filters(choice, input_filters_or_mask, filters_base))

  # Compute the largest possible number of input filters as an integer.
  if input_filters_or_mask is None or isinstance(input_filters_or_mask, int):
    max_input_filters = input_filters_or_mask
  else:
    # input_filters_or_mask must be a tf.Tensor in this case.
    max_input_filters = int(input_filters_or_mask.shape[-1])

  # Compute the largest possible number of output filters as an integer.
  max_output_filters = 0
  for choice in choices:
    # Note: current_filters should always be an integer (rather than a Tensor)
    # because `max_input_filters` is an integer.
    current_filters = _compute_filters(choice, max_input_filters, filters_base)
    max_output_filters = max(max_output_filters, current_filters)

  # Return an integer (if possible) or a mask (if we can't infer the exact
  # number of filters at graph construction time.
  if len(scaled_choices) == 1:
    selection = scaled_choices[0]  # type: Union[int, tf.Tensor]
    if isinstance(selection, tf.Tensor):
      return tf.sequence_mask(selection, max_output_filters, dtype=tf.float32)
    else:
      return selection
  else:
    selection_index = tf.argmax(mask)
    selection = tf.gather(scaled_choices, selection_index)
    return tf.sequence_mask(selection, max_output_filters, dtype=tf.float32)


def _get_activation(
    spec):
  """Get a rematlib Layer corresponding to a given activation function."""
  if spec == mobile_search_space_v3.RELU:
    result = layers.ReLU()
  elif spec == mobile_search_space_v3.RELU6:
    result = layers.ReLU6()
  elif spec == mobile_search_space_v3.SWISH6:
    result = layers.Swish6()
  elif spec == mobile_search_space_v3.SIGMOID:
    result = layers.Sigmoid()
  else:
    raise ValueError('Unrecognized activation function: {}'.format(spec))

  return result


# Primitive (low-level) layers.
def _batch_norm(params,
                filters_or_mask):
  if isinstance(filters_or_mask, int):
    return layers.BatchNorm(
        epsilon=params['batch_norm_epsilon'],
        stateful=not params['force_stateless_batch_norm'])
  else:
    return layers.MaskedStatelessBatchNorm(
        mask=filters_or_mask, epsilon=params['batch_norm_epsilon'])


def _conv_with_fixed_kernel(params,
                            input_filters_or_mask,
                            output_filters_or_mask,
                            kernel_size,
                            strides = (1, 1),
                            activation = None,
                            use_batch_norm = True):
  """Construct a Conv2D, followed by a batch norm and optional activation."""
  result = []
  if (isinstance(input_filters_or_mask, int) and
      isinstance(output_filters_or_mask, int)):
    result.append(
        layers.Conv2D(
            filters=output_filters_or_mask,
            kernel_size=kernel_size,
            kernel_initializer=params['kernel_initializer'],
            kernel_regularizer=params['kernel_regularizer'],
            use_bias=not use_batch_norm,
            strides=strides))
  else:
    result.append(
        layers.MaskedConv2D(
            input_mask=_to_filters_mask(input_filters_or_mask),
            output_mask=_to_filters_mask(output_filters_or_mask),
            kernel_size=kernel_size,
            kernel_initializer=params['kernel_initializer'],
            kernel_regularizer=params['kernel_regularizer'],
            use_bias=not use_batch_norm,
            strides=strides))

  if use_batch_norm:
    result.append(_batch_norm(params, output_filters_or_mask))
  if activation is not None:
    result.append(activation)
  return layers.Sequential(result)


def _conv(params,
          input_filters_or_mask,
          output_filters_or_mask,
          kernel_size,
          strides = (1, 1),
          activation = None,
          use_batch_norm = True):
  """Conv2D + batch norm + activation, optionally searching over kernel size."""
  if isinstance(kernel_size, schema.OneOf) and len(kernel_size.choices) > 1:
    choices = []
    for kernel_size_value in kernel_size.choices:
      choices.append(
          _conv_with_fixed_kernel(
              params=params,
              input_filters_or_mask=input_filters_or_mask,
              output_filters_or_mask=output_filters_or_mask,
              kernel_size=kernel_size_value,
              strides=strides,
              activation=activation,
              use_batch_norm=use_batch_norm))
    return layers.maybe_switch_v2(kernel_size.mask, choices)
  else:
    if isinstance(kernel_size, schema.OneOf):
      kernel_size_value = kernel_size.choices[0]
    else:
      kernel_size_value = kernel_size
    return _conv_with_fixed_kernel(
        params=params,
        input_filters_or_mask=input_filters_or_mask,
        output_filters_or_mask=output_filters_or_mask,
        kernel_size=kernel_size_value,
        strides=strides,
        activation=activation,
        use_batch_norm=use_batch_norm)


def _depthwise_conv_with_fixed_kernel(
    params,
    filters_or_mask,
    kernel_size,
    strides = (1, 1),
    activation = None):
  """Build a depthwise conv, batch norm, and optional activation."""
  result = []
  if isinstance(filters_or_mask, int):
    result.append(
        layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            depthwise_initializer=params['depthwise_initializer'],
            depthwise_regularizer=params['kernel_regularizer'],
            strides=strides))
  else:
    result.append(
        layers.MaskedDepthwiseConv2D(
            kernel_size=kernel_size,
            mask=filters_or_mask,
            depthwise_initializer=params['depthwise_initializer'],
            depthwise_regularizer=params['kernel_regularizer'],
            strides=strides))

  result.append(_batch_norm(params, filters_or_mask))
  if activation is not None:
    result.append(activation)
  return layers.Sequential(result)


def _depthwise_conv(
    params,
    filters_or_mask,
    kernel_size,
    strides = (1, 1),
    activation = None):
  """Depthwise conv + BN + activation, optionally searching over kernel size."""
  if isinstance(kernel_size, schema.OneOf) and len(kernel_size.choices) > 1:
    choices = []
    for kernel_size_value in kernel_size.choices:
      choices.append(
          _depthwise_conv_with_fixed_kernel(
              params=params,
              filters_or_mask=filters_or_mask,
              kernel_size=kernel_size_value,
              strides=strides,
              activation=activation))
    return layers.maybe_switch_v2(kernel_size.mask, choices)
  else:
    if isinstance(kernel_size, schema.OneOf):
      kernel_size_value = kernel_size.choices[0]
    else:
      kernel_size_value = kernel_size
    return _depthwise_conv_with_fixed_kernel(
        params=params,
        filters_or_mask=filters_or_mask,
        kernel_size=kernel_size_value,
        strides=strides,
        activation=activation)


def _squeeze_and_excite(params,
                        input_filters_or_mask,
                        inner_activation,
                        gating_activation):
  """Generate a squeeze-and-excite layer."""
  # We provide two code paths:
  # 1. For the case where the number of input filters is known at graph
  #    construction time, and input_filters_or_mask is an int. This typically
  #    happens during stand-alone model training.
  # 2. For the case where the number of input filters is not known until
  #    runtime, and input_filters_or_mask is a 1D float tensor. This often
  #    happens during an architecture search.
  if isinstance(input_filters_or_mask, int):
    input_filters = input_filters_or_mask
    hidden_filters = search_space_utils.make_divisible(
        input_filters * _SQUEEZE_AND_EXCITE_RATIO,
        divisor=params['filters_base'])

    return layers.ParallelProduct([
        layers.Identity(),
        layers.Sequential([
            layers.GlobalAveragePool(keepdims=True),
            layers.Conv2D(
                filters=hidden_filters,
                kernel_size=(1, 1),
                kernel_initializer=params['kernel_initializer'],
                kernel_regularizer=params['kernel_regularizer'],
                use_bias=True),
            inner_activation,
            layers.Conv2D(
                filters=input_filters,
                kernel_size=(1, 1),
                kernel_initializer=params['kernel_initializer'],
                kernel_regularizer=params['kernel_regularizer'],
                use_bias=True),
            gating_activation,
        ]),
    ])
  else:
    input_mask = input_filters_or_mask
    input_filters = tf.reduce_sum(input_mask)
    hidden_filters = search_space_utils.tf_make_divisible(
        input_filters * _SQUEEZE_AND_EXCITE_RATIO,
        divisor=params['filters_base'])

    max_input_filters = int(input_mask.shape[0])
    max_hidden_filters = search_space_utils.make_divisible(
        max_input_filters * _SQUEEZE_AND_EXCITE_RATIO,
        divisor=params['filters_base'])

    hidden_mask = tf.sequence_mask(
        hidden_filters, max_hidden_filters, dtype=tf.float32)

    return layers.ParallelProduct([
        layers.Identity(),
        layers.Sequential([
            layers.GlobalAveragePool(keepdims=True),
            layers.MaskedConv2D(
                input_mask=input_mask,
                output_mask=hidden_mask,
                kernel_size=(1, 1),
                kernel_initializer=params['kernel_initializer'],
                kernel_regularizer=params['kernel_regularizer'],
                use_bias=True),
            inner_activation,
            layers.MaskedConv2D(
                input_mask=hidden_mask,
                output_mask=input_mask,
                kernel_size=(1, 1),
                kernel_initializer=params['kernel_initializer'],
                kernel_regularizer=params['kernel_regularizer'],
                use_bias=True),
            gating_activation,
        ])
    ])


def _maybe_squeeze_and_excite(
    params,
    input_filters_or_mask,
    inner_activation,
    gating_activation,
    enabled
):
  """Generate a squeeze-and-excite layer or identity function."""
  def make_squeeze_and_excite():
    return _squeeze_and_excite(
        params=params,
        input_filters_or_mask=input_filters_or_mask,
        inner_activation=inner_activation,
        gating_activation=gating_activation)

  # We use explicit bool comparisons to make sure a user doesn't pass in a
  # bad configuration like enabled=OneOf([False, True, 42]).
  if isinstance(enabled, bool):
    return make_squeeze_and_excite() if enabled else layers.Identity()
  elif isinstance(enabled, schema.OneOf):
    options = []
    for choice in enabled.choices:
      options.append(make_squeeze_and_excite() if choice else layers.Identity())
    return layers.maybe_switch_v2(enabled.mask, options)
  else:
    raise ValueError('Unsupported value for "enabled": {}'.format(enabled))


######################################################################
# FUNCTIONS THAT CORRESPOND TO NAMEDTUPLES IN MOBILE_SEARCH_SPACE_V3 #
######################################################################
def _build_conv(params,
                layer_spec,
                input_filters,
                output_filters):
  return _conv(
      params,
      input_filters_or_mask=_oneof_filters_to_int_or_mask(input_filters),
      output_filters_or_mask=_oneof_filters_to_int_or_mask(output_filters),
      kernel_size=layer_spec.kernel_size,
      strides=layer_spec.strides,
      activation=None,
      use_batch_norm=layer_spec.use_batch_norm)


def _build_separable_conv(params,
                          layer_spec,
                          input_filters,
                          output_filters):
  return layers.Sequential([
      _depthwise_conv(
          params,
          filters_or_mask=_oneof_filters_to_int_or_mask(input_filters),
          kernel_size=layer_spec.kernel_size,
          strides=layer_spec.strides,
          activation=_get_activation(layer_spec.activation)),
      _conv(
          params,
          input_filters_or_mask=_oneof_filters_to_int_or_mask(input_filters),
          output_filters_or_mask=_oneof_filters_to_int_or_mask(output_filters),
          kernel_size=(1, 1),
          activation=None),
  ])


def _build_depthwise_bottleneck(
    params,
    layer_spec,
    input_filters,
    output_filters,
    filters_base):
  """Construct a bottleneck layer with a depthwise conv in the middle."""
  input_filters_or_mask = _oneof_filters_to_int_or_mask(input_filters)
  output_filters_or_mask = _oneof_filters_to_int_or_mask(output_filters)
  expansion_filters_or_mask = _oneof_filters_to_int_or_mask(
      layer_spec.expansion_filters, input_filters_or_mask, filters_base)

  result = [
      _conv(
          params,
          input_filters_or_mask=input_filters_or_mask,
          output_filters_or_mask=expansion_filters_or_mask,
          kernel_size=(1, 1),
          activation=_get_activation(layer_spec.activation)),
      _depthwise_conv(
          params,
          filters_or_mask=expansion_filters_or_mask,
          kernel_size=layer_spec.kernel_size,
          strides=layer_spec.strides,
          activation=_get_activation(layer_spec.activation)),
  ]

  result.append(
      _maybe_squeeze_and_excite(
          params,
          expansion_filters_or_mask,
          _get_activation(layer_spec.se_inner_activation),
          _get_activation(layer_spec.se_gating_activation),
          layer_spec.use_squeeze_and_excite))

  result.append(
      _conv(
          params,
          input_filters_or_mask=expansion_filters_or_mask,
          output_filters_or_mask=output_filters_or_mask,
          kernel_size=(1, 1),
          activation=None))

  return layers.Sequential(result)


def _build_oneof(params,
                 layer_spec,
                 input_filters,
                 output_filters,
                 filters_base):
  """Select one of N possible choices."""
  if len(layer_spec.choices) > 1 and not params['force_stateless_batch_norm']:
    raise ValueError(
        'force_stateless_batch_norm must be true for models containing '
        'Switch layers (e.g., when performing architecture searches).')
  choices = [
      _build_layer(params, choice, input_filters, output_filters, filters_base)
      for choice in layer_spec.choices
  ]
  return layers.maybe_switch_v2(layer_spec.mask, choices)


def _build_residual_spec(params,
                         layer_spec,
                         input_filters,
                         output_filters,
                         filters_base):
  """Builds a layers.Layer implementation for ResidualSpec specification."""
  def can_optimize_residual_spec(layer):
    """Returns true if we can replace residual layer with an identity."""
    can_optimize_oneof = (isinstance(layer, schema.OneOf) and
                          len(layer.choices) == 1 and
                          isinstance(layer.choices[0], basic_specs.ZeroSpec))
    can_optimize_zerospec = isinstance(layer, basic_specs.ZeroSpec)
    return can_optimize_oneof or can_optimize_zerospec
  if can_optimize_residual_spec(layer_spec.layer):
    return layers.Identity()
  layer = _build_layer(params, layer_spec.layer, input_filters,
                       output_filters, filters_base)
  return layers.ParallelSum([layer, layers.Identity()])


def _build_layer(params,
                 layer_spec,
                 input_filters,
                 output_filters,
                 filters_base):
  """Create a one of N possible types of layers within the body of a network."""
  if isinstance(layer_spec, mobile_search_space_v3.ResidualSpec):
    return _build_residual_spec(params, layer_spec, input_filters,
                                output_filters, filters_base)
  elif isinstance(layer_spec, mobile_search_space_v3.ConvSpec):
    return _build_conv(params, layer_spec, input_filters, output_filters)
  elif isinstance(layer_spec, mobile_search_space_v3.SeparableConvSpec):
    return _build_separable_conv(params, layer_spec, input_filters,
                                 output_filters)
  elif isinstance(layer_spec, mobile_search_space_v3.DepthwiseBottleneckSpec):
    return _build_depthwise_bottleneck(params, layer_spec, input_filters,
                                       output_filters, filters_base)
  elif isinstance(layer_spec, mobile_search_space_v3.GlobalAveragePoolSpec):
    return layers.GlobalAveragePool(keepdims=True)
  elif isinstance(layer_spec, mobile_search_space_v3.ActivationSpec):
    return _get_activation(layer_spec)
  elif isinstance(layer_spec, basic_specs.ZeroSpec):
    return layers.Zeros()
  elif isinstance(layer_spec, schema.OneOf):
    return _build_oneof(
        params, layer_spec, input_filters, output_filters, filters_base)
  else:
    raise ValueError('Unsupported layer_spec type: {}'.format(
        type(layer_spec)))


def _make_head(params):
  """Construct a classification model head."""
  result = []
  if params['dropout_rate'] > 0:
    result.append(layers.Dropout(params['dropout_rate']))
  result.append(
      layers.Conv2D(filters=params['num_classes'],
                    kernel_size=(1, 1),
                    kernel_initializer=params['dense_initializer'],
                    # kernel_regularizer is not used for the final dense layer:
                    kernel_regularizer=None,
                    use_bias=True))
  result.append(layers.GlobalAveragePool(keepdims=False))
  return layers.Sequential(result)


def _build_model(params,
                 model_spec):
  """Translate a ConvTowerSpec namedtuple into a rematlib Layer."""
  input_filters = schema.OneOf([3], basic_specs.FILTERS_TAG)
  layer = None
  result = []
  endpoints = []
  for block_spec in model_spec.blocks:
    for layer_spec in block_spec.layers:
      if isinstance(layer_spec, mobile_search_space_v3.DetectionEndpointSpec):
        if layer is None:
          raise ValueError(
              'The first layer of the network cannot be a detection endpoint.')
        endpoints.append(layer)
      else:
        output_filters = block_spec.filters
        if isinstance(output_filters, int):
          output_filters = schema.OneOf(
              [output_filters], basic_specs.FILTERS_TAG)

        layer = _build_layer(
            params, layer_spec, input_filters, output_filters,
            model_spec.filters_base)
        input_filters = output_filters
        result.append(layer)

  result.append(_make_head(params))

  # Build the model
  model = layers.Sequential(result, aux_outputs=endpoints)
  return model


def get_model(
    model_spec,
    kernel_initializer=tf.initializers.he_normal(),
    depthwise_initializer=depthwise_initializers.depthwise_he_normal(),
    dense_initializer=tf.initializers.random_normal(stddev=0.01),
    kernel_regularizer=_l2_regularizer(0.00004),
    batch_norm_epsilon = 0.001,
    force_stateless_batch_norm = False,
    dropout_rate = 0,
    num_classes = 1001):
  """Get a new Layer representing an architecture or one-shot model.

  Args:
    model_spec: basic_specs.ConvTowerSpec namedtuple controlling the
        search space to use.
    kernel_initializer: TF initializer to use for the kernels of ordinary
        convolutions.
    depthwise_initializer: TF initializer to use for the kernels of depthwise
        convolutions.
    dense_initializer: TF initializer to use for the final dense
        (fully connected) layer of the network.
    kernel_regularizer: TF regularizer to apply to kernels.
    batch_norm_epsilon: Positive float, the value of `epsilon` to use in batch
        normalization to prevent division by zero.
    force_stateless_batch_norm: Boolean. If true, we will run all batch norm
        ops in 'training' mode, even at eval time.
    dropout_rate: Float between 0 and 1. The fraction of elements to drop
        immediately before the final 1x1 convolution.
    num_classes: An integer. The expected number of output classes.

  Returns:
    A layers.Sequential object whose aux_inputs are detection endpoints.
  """
  if dropout_rate < 0 or dropout_rate > 1:
    raise ValueError('dropout_rate must be between 0 and 1: {:f}'
                     .format(dropout_rate))

  if model_spec.filters_base < 1:
    raise ValueError('filters_base must be a positive integer: {:s}'
                     .format(str(model_spec.filters_base)))

  params = {
      'filters_base': model_spec.filters_base,
      'kernel_initializer': kernel_initializer,
      'depthwise_initializer': depthwise_initializer,
      'dense_initializer': dense_initializer,
      'kernel_regularizer': kernel_regularizer,
      'batch_norm_epsilon': batch_norm_epsilon,
      'force_stateless_batch_norm': force_stateless_batch_norm,
      'dropout_rate': dropout_rate,
      'num_classes': num_classes,
  }
  return _build_model(params, model_spec)
