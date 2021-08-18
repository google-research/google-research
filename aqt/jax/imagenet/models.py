# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Flax implementation of ResNet V1 with optional quantization."""

import dataclasses
import functools
import typing
from typing import Any, Optional, Tuple, Type, Iterable

from flax import linen as nn
import jax.numpy as jnp

from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import quant_config
from aqt.jax.flax import struct as flax_struct


dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


class BPReLU(nn.Module):
  """Biased PReLU that learns the origin point of PReLU.

  Attributes:
    dtype: the dtype of the computation (default: bfloat16).
    negative_slope_init: initializer function for the negative slope.
    bias_init: initializer function for both x and y biases.
  """
  init_bias: float = 0.0
  init_slope: float = 0.25
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, inputs):
    """Apply a threshold and negative slope to inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    params_shape = (inputs.shape[-1],)
    slope_init_fn = lambda key, shape: jnp.ones(shape) * self.init_slope
    negative_slope = self.param('negative_slope', slope_init_fn, params_shape)
    bias_init_fn = lambda key, shape: jnp.ones(shape) * self.init_bias
    bias_x = self.param('bias_x', bias_init_fn, params_shape)
    bias_y = self.param('bias_y', bias_init_fn, params_shape)
    inputs = inputs - bias_x
    y = jnp.where(inputs >= 0, inputs, negative_slope * inputs) - bias_y
    return jnp.asarray(y, self.dtype)


# A function zoo that contains all candidates of activation functions
act_function_zoo = {
    'relu': nn.relu,
    'leaky_relu': nn.leaky_relu,
    'soft_sign': nn.soft_sign,
    'celu': nn.celu,
    'swish': nn.swish,
    'elu': nn.elu,
    'gelu': nn.gelu,
    'bprelu': lambda x: BPReLU()(x),  # pylint: disable=unnecessary-lambda
    'none': lambda x: x,
}


def shortcuts_ch_shrink(x, out_features, method):
  """Match the number of channels in the shortcuts in the 1st conv1x1 layer."""
  in_features = x.shape[-1]
  num_ch_avg = in_features // out_features
  assert out_features * num_ch_avg == in_features, (
      'in_features needs to be a whole multiple of out_features')
  dim_nwh = x.shape[0:3]
  if method == 'consecutive':
    x = jnp.reshape(x, dim_nwh + (out_features, num_ch_avg))
    return jnp.average(x, axis=4)
  elif method == 'every_n':
    x = jnp.reshape(x, dim_nwh + (num_ch_avg, out_features))
    return jnp.average(x, axis=3)
  elif method == 'none':
    # return all zeros to represent no shortcut
    return jnp.zeros(dim_nwh + (out_features,), dtype=x.dtype)
  else:
    raise ValueError('Unsupported channel shrinking shortcut function type.')


def shortcuts_ch_expand(x, out_features, method):
  """Match the number of channels in the shortcuts in the 3rd conv1x1 layer."""
  in_features = x.shape[-1]
  assert in_features < out_features and out_features % in_features == 0, (
      'Number of in_features should be smaller '
      'than number of out_features')
  ch_multiplier = out_features // in_features
  if method == 'tile':
    return jnp.tile(x, reps=(1, 1, 1, ch_multiplier))
  elif method == 'repeat':
    return jnp.repeat(x, repeats=ch_multiplier, axis=3)
  elif method == 'zeropad':
    # pad all zeros after the input feature maps
    return jnp.pad(
        x,
        pad_width=((0, 0), (0, 0), (0, 0), (0, out_features - in_features)),
        mode='constant')
  elif method == 'none':
    # return all zeros to represent no shortcut
    return jnp.zeros(x.shape[0:3] + (out_features,), dtype=x.dtype)
  else:
    raise ValueError('Unsupported channel duplication function type.')


def spatial_downsample(x, strides, method):
  """Match the spatial resolution in the shortcuts in the 2nd conv3x3 layer."""
  function_zoo = {
      'max_pool':
          functools.partial(nn.max_pool, window_shape=(3, 3), padding='SAME'),
      'avg_pool':
          functools.partial(nn.avg_pool, window_shape=(3, 3), padding='SAME'),
      'none':
          None,
  }
  assert method in function_zoo.keys(), 'Unsupported shortcut spatial method.'
  if method == 'none':
    # return all zeros to represent no shortcut
    out_dim_n = x.shape[0]
    out_dim_w = x.shape[1] // strides[0]
    out_dim_h = x.shape[2] // strides[1]
    out_dim_c = x.shape[3]
    output_shape = (out_dim_n, out_dim_w, out_dim_h, out_dim_c)
    return jnp.zeros(output_shape, dtype=x.dtype)
  else:
    return x if strides == (1, 1) else function_zoo[method](x, strides=strides)


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  @dataclass  # pylint: disable=missing-class-docstring
  class HParams:
    # 'conv_proj' is only used in between different stages of residual blocks
    #   where the Tensor shape needs to be matched for the next stage.
    #   When it is not needed, it is set to None in hparams.
    conv_proj: Optional[aqt_flax_layers.ConvAqt.HParams]
    conv_1: aqt_flax_layers.ConvAqt.HParams
    conv_2: aqt_flax_layers.ConvAqt.HParams
    conv_3: aqt_flax_layers.ConvAqt.HParams
    act_function: str
    shortcut_ch_shrink_method: str
    shortcut_ch_expand_method: str
    shortcut_spatial_method: str

  hparams: HParams
  filters: int
  quant_context: quant_config.QuantContext
  strides: Tuple[int, int]
  train: bool
  dtype: Type[Any]

  @nn.compact
  def __call__(
      self,
      inputs,
  ):
    """Applies a residual block consisting of Conv-Batch Norm-ReLU chains."""
    filters = self.filters
    strides = self.strides
    hparams = self.hparams
    dtype = self.dtype
    train = self.train
    quant_context = self.quant_context
    assert hparams.act_function in act_function_zoo.keys(
    ), 'Activation function type is not supported.'
    act_function = act_function_zoo[hparams.act_function]

    needs_projection = inputs.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=dtype)

    conv = functools.partial(
        aqt_flax_layers.ConvAqt,
        use_bias=False,
        dtype=dtype,
        quant_context=quant_context,
        paxis_name='batch',
        train=train)

    residual = inputs
    if needs_projection:
      if hparams.conv_proj is None:
        raise ValueError(
            'hparams.conv_proj cannot be None if needs_projection is True.')
      assert hparams.conv_proj is not None
      residual = conv(
          features=filters * 4,
          kernel_size=(1, 1),
          strides=strides,
          name='proj_conv',
          hparams=hparams.conv_proj)(
              residual)
      residual = batch_norm(name='proj_bn')(residual)
    else:
      if hparams.conv_proj is not None:
        raise ValueError(
            'hparams.conv_proj must be None if needs_projection is False.')

    shortcut1 = shortcuts_ch_shrink(
        inputs, out_features=filters, method=hparams.shortcut_ch_shrink_method)
    y = conv(
        features=filters,
        kernel_size=(1, 1),
        name='conv1',
        hparams=hparams.conv_1)(
            inputs)
    y = batch_norm(name='bn1')(y)
    y = y + shortcut1
    del shortcut1
    y = act_function(y)
    shortcut2 = spatial_downsample(
        y, strides=strides, method=hparams.shortcut_spatial_method)
    y = conv(
        features=filters,
        kernel_size=(3, 3),
        strides=strides,
        name='conv2',
        hparams=hparams.conv_2)(
            y)
    y = batch_norm(name='bn2')(y)
    y = y + shortcut2
    del shortcut2
    y = act_function(y)
    shortcut3 = shortcuts_ch_expand(
        y, out_features=filters * 4, method=hparams.shortcut_ch_expand_method)
    y = conv(
        features=filters * 4,
        kernel_size=(1, 1),
        name='conv3',
        hparams=hparams.conv_3)(
            y)

    y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(y)
    y = y + shortcut3
    del shortcut3
    output = act_function(residual + y)
    return output


class ResNet(nn.Module):
  """ResNetV1 with optional quantization."""

  @dataclass
  class HParams:
    dense_layer: aqt_flax_layers.DenseAqt.HParams
    conv_init: aqt_flax_layers.ConvAqt.HParams
    residual_blocks: Tuple[ResidualBlock.HParams, Ellipsis]
    filter_multiplier: float
    act_function: str

  num_classes: int
  hparams: HParams
  quant_context: quant_config.QuantContext
  num_filters: int
  train: bool
  dtype: Type[Any]

  @nn.compact
  def __call__(
      self,
      inputs,
  ):
    """Applies ResNet model. Number of residual blocks inferred from hparams."""
    num_classes = self.num_classes
    hparams = self.hparams
    num_filters = self.num_filters
    dtype = self.dtype
    assert hparams.act_function in act_function_zoo.keys(
    ), 'Activation function type is not supported.'

    x = aqt_flax_layers.ConvAqt(
        features=num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=dtype,
        name='init_conv',
        train=self.train,
        quant_context=self.quant_context,
        paxis_name='batch',
        hparams=hparams.conv_init,
    )(
        inputs)
    x = nn.BatchNorm(
        use_running_average=not self.train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=dtype,
        name='init_bn')(
            x)
    if hparams.act_function == 'relu':
      x = nn.relu(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    else:
      # TODO(yichi): try adding other activation functions here
      # Use avg pool so that for binary nets, the distribution is symmetric.
      x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    filter_multiplier = hparams.filter_multiplier
    for i, block_hparams in enumerate(hparams.residual_blocks):
      proj = block_hparams.conv_proj
      # For projection layers (unless it is the first layer), strides = (2, 2)
      if i > 0 and proj is not None:
        filter_multiplier *= 2
        strides = (2, 2)
      else:
        strides = (1, 1)
      x = ResidualBlock(
          filters=int(num_filters * filter_multiplier),
          hparams=block_hparams,
          quant_context=self.quant_context,
          strides=strides,
          train=self.train,
          dtype=dtype)(
              x)
    if hparams.act_function == 'none':
      # The DenseAQT below is not binarized.
      # If removing the activation functions, there will be no act function
      # between the last residual block and the dense layer.
      # So add a ReLU in that case.
      # TODO(yichi): try BPReLU
      x = nn.relu(x)
    else:
      pass
    x = jnp.mean(x, axis=(1, 2))

    x = aqt_flax_layers.DenseAqt(
        features=num_classes,
        dtype=dtype,
        train=self.train,
        quant_context=self.quant_context,
        paxis_name='batch',
        hparams=hparams.dense_layer,
    )(x, padding_mask=None)

    x = jnp.asarray(x, dtype)
    output = nn.log_softmax(x)
    return output
