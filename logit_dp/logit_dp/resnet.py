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

"""Stateless copy of Resnet18 from Haiku's net library."""

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class StatelessBatchNorm(hk.Module):
  """Normalizes inputs to maintain a mean of 0 and stddev of 1.

  Variation of hk.BatchNorm without any moving averages (to avoid forwarding
  unncessary state variables).
  """

  def __init__(
      self,
      create_scale,
      create_offset,
      eps = 1e-5,
      scale_init = None,
      offset_init = None,
      axis = None,
      cross_replica_axis = None,
      cross_replica_axis_index_groups = None,
      data_format = "channels_last",
      name = None,
  ):
    """Constructs a BatchNorm module.

    Args:
      create_scale: Whether to include a trainable scaling factor.
      create_offset: Whether to include a trainable offset.
      eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
        as in the paper and Sonnet.
      scale_init: Optional initializer for gain (aka scale). Can only be set if
        ``create_scale=True``. By default, ``1``.
      offset_init: Optional initializer for bias (aka offset). Can only be set
        if ``create_offset=True``. By default, ``0``.
      axis: Which axes to reduce over. The default (``None``) signifies that all
        but the channel axis should be normalized. Otherwise this is a list of
        axis indices which will have normalization statistics calculated.
      cross_replica_axis: If not ``None``, it should be a string (or sequence of
        strings) representing the axis name(s) over which this module is being
        run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
        argument means that batch statistics are calculated across all replicas
        on the named axes.
      cross_replica_axis_index_groups: Specifies how devices are grouped. Valid
        only within ``jax.pmap`` collectives.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``. See :func:`get_channel_index`.
      name: The module name.
    """
    super().__init__(name=name)
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`")
    if not create_offset and offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`")
    if (
        cross_replica_axis is None
        and cross_replica_axis_index_groups is not None
    ):
      raise ValueError(
          "`cross_replica_axis` name must be specified"
          "if `cross_replica_axis_index_groups` are used."
      )

    self.create_scale = create_scale
    self.create_offset = create_offset
    self.eps = eps
    self.scale_init = scale_init or jnp.ones
    self.offset_init = offset_init or jnp.zeros
    self.axis = axis
    self.cross_replica_axis = cross_replica_axis
    self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
    self.channel_index = hk.get_channel_index(data_format)

  def __call__(
      self,
      inputs,
      scale = None,
      offset = None,
  ):
    """Computes the normalized version of the input.

    Args:
      inputs: An array, where the data format is ``[..., C]``.
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_scale=True``.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_offset=True``.

    Returns:
      The array, normalized across all but the last dimension.
    """
    if self.create_scale and scale is not None:
      raise ValueError(
          "Cannot pass `scale` at call time if `create_scale=True`."
      )
    if self.create_offset and offset is not None:
      raise ValueError(
          "Cannot pass `offset` at call time if `create_offset=True`."
      )

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    mean = jnp.mean(inputs, axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
    if self.cross_replica_axis:
      mean = jax.lax.pmean(
          mean,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
      mean_of_squares = jax.lax.pmean(
          mean_of_squares,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
    var = mean_of_squares - jnp.square(mean)

    w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
    w_dtype = inputs.dtype

    if self.create_scale:
      scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
    elif scale is None:
      scale = np.ones([], dtype=w_dtype)

    if self.create_offset:
      offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
    elif offset is None:
      offset = np.zeros([], dtype=w_dtype)

    eps = jax.lax.convert_element_type(self.eps, var.dtype)
    inv = scale * jax.lax.rsqrt(var + eps)
    return (inputs - mean) * inv + offset


FloatStrOrBool = Union[str, float, bool]


class BlockV2(hk.Module):
  """ResNet V2 block with optional bottleneck and batch normalization."""

  def __init__(
      self,
      channels,
      stride,
      use_projection,
      bn_config,
      use_batch_norm,
      bottleneck,
      name = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    def normalization_fn(name):
      if use_batch_norm:
        return StatelessBatchNorm(name=name, **bn_config)
      else:
        return lambda x: x

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv",
      )
      self.proj_batchnorm = normalization_fn("shortcut_stateless_batchnorm")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0",
    )

    sbn_0 = normalization_fn("stateless_batchnorm_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1",
    )

    sbn_1 = normalization_fn("stateless_batchnorm_1")

    layers = ((conv_0, sbn_0), (conv_1, sbn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2",
      )

      # NOTE: Some implementations of ResNet50 v2 suggest initializing
      # gamma/scale here to zeros.
      sbn_2 = normalization_fn("stateless_batchnorm_2")
      layers = layers + ((conv_2, sbn_2),)

    self.layers = layers

  def __call__(self, inputs):
    x = shortcut = inputs
    for i, (conv_i, sbn_i) in enumerate(self.layers):
      x = sbn_i(x)
      x = jax.nn.relu(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = conv_i(x)
    return x + shortcut


class BlockGroup(hk.Module):
  """Higher level block for ResNet implementation."""

  def __init__(
      self,
      channels,
      num_blocks,
      stride,
      bn_config,
      bottleneck,
      use_projection,
      use_batch_norm,
      name = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(
              channels=channels,
              stride=(1 if i else stride),
              use_projection=(i == 0 and use_projection),
              bottleneck=bottleneck,
              bn_config=bn_config,
              use_batch_norm=use_batch_norm,
              name="block_%d" % i,
          )
      )

  def __call__(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
  """ResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (64, 128, 256, 512),
          "use_projection": (False, True, True, True),
      },
      1: {
          "blocks_per_group": 2,
          "bottleneck": False,
          "channels_per_group": 64,
          "use_projection": True,
      },
  }

  BlockGroup = BlockGroup  # pylint: disable=invalid-name
  BlockV2 = BlockV2  # pylint: disable=invalid-name

  def __init__(
      self,
      blocks_per_group,
      num_classes,
      bn_config = None,
      bottleneck = True,
      channels_per_group = (256, 512, 1024, 2048),
      use_projection = (True, True, True, True),
      logits_config = None,
      name = None,
      use_batch_norm = False,
      initial_conv_config = None,
      strides = (1, 2, 2, 2),
  ):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number of
        channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      use_batch_norm: Specifies if batch normalization is used.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride of
        convolutions for each block in each group.
    """
    super().__init__(name=name)

    bn_config = dict(bn_config or {})
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    logits_config = dict(logits_config or {})
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")
    check_length(4, strides, "strides")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 64)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    self.block_groups = []
    for i, stride in enumerate(strides):
      self.block_groups.append(
          BlockGroup(
              channels=channels_per_group[i],
              num_blocks=blocks_per_group[i],
              stride=stride,
              bn_config=bn_config,
              bottleneck=bottleneck,
              use_projection=use_projection[i],
              use_batch_norm=use_batch_norm,
              name="block_group_%d" % i,
          )
      )

    self.logits = hk.Linear(num_classes, **logits_config)

  def __call__(self, inputs):
    out = inputs
    out = self.initial_conv(out)

    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
    )

    for block_group in self.block_groups:
      out = block_group(out)

    out = jax.nn.relu(out)
    out = jnp.mean(out, axis=(1, 2))
    return self.logits(out)


class ResNet18(ResNet):
  """ResNet18."""

  def __init__(
      self,
      num_classes,
      bn_config = None,
      logits_config = None,
      name = None,
      initial_conv_config = None,
      strides = (1, 2, 2, 2),
      use_batch_norm = False,
  ):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
      strides: A sequence of length 4 that indicates the size of stride of
        convolutions for each block in each group.
    """
    super().__init__(
        num_classes=num_classes,
        bn_config=bn_config,
        initial_conv_config=initial_conv_config,
        strides=strides,
        logits_config=logits_config,
        name=name,
        use_batch_norm=use_batch_norm,
        **ResNet.CONFIGS[18],
    )
