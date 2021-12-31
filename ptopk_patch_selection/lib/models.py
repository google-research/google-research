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

"""File to define model architectures."""
from typing import Tuple

from flax.deprecated import nn
from jax import numpy as jnp
import jax.nn
from lib import utils  # pytype: disable=import-error


class SimpleCNNImageClassifier(nn.Module):
  """Simple convolutional network with dense classifier on top."""

  def apply(self, x):
    x = nn.Conv(x, features=32, kernel_size=(3, 3), name="conv")
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten.
    x = nn.Dense(x, 128, name="fc")
    return x


class MLP(nn.Module):
  """Simple MLP."""

  def apply(self, x, hidden_size, output_size, act_fn=nn.relu):
    dense = nn.DenseGeneral.partial(axis=-1)
    x = dense(x, hidden_size, name="fc_hid")
    x = act_fn(x)
    x = dense(x, output_size, name="fc_out")
    return x


class CNN(nn.Module):
  """Simple CNN."""

  def apply(self,
            x,
            features,
            kernel_size=(3, 3),
            act_fn=nn.relu,
            num_layers=3,
            padding="constant"):
    conv2d = nn.Conv.partial(
        features=features, kernel_size=kernel_size, padding="VALID")
    padding_size = (kernel_size[0] - 1) // 2
    for i in range(num_layers - 1):
      if padding is not None:
        x = jnp.pad(
            x, ((0, 0), (padding_size, padding_size),
                (padding_size, padding_size), (0, 0)),
            mode=padding)
      x = conv2d(x, name=f"conv_{i+1}")
      x = act_fn(x)
    if padding is not None:
      x = jnp.pad(
          x, ((0, 0), (padding_size, padding_size),
              (padding_size, padding_size), (0, 0)),
          mode=padding)
    x = conv2d(x, name=f"conv_{num_layers}")
    return x


class CNNDecoder(nn.Module):
  """Simple CNN decoder."""

  def apply(self,
            x,
            features,
            kernel_size=(5, 5),
            act_fn=nn.relu,
            num_layers=5,
            padding="SAME"):
    conv2d = nn.ConvTranspose.partial(
        features=features, kernel_size=kernel_size, padding=padding)
    for i in range(num_layers - 1):
      x = act_fn(conv2d(x, name=f"conv_{i+1}"))
    x = conv2d(x, name=f"conv_{num_layers}")
    return x


class BasicBlock(nn.Module):
  """Basic (non-bottleneck) ResNet-v1 block."""

  def apply(self,
            x,
            filters,
            strides=(1, 1),
            train=True):
    norm_layer = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)
    conv3x3 = nn.Conv.partial(kernel_size=(3, 3), padding="SAME", bias=False)
    conv1x1 = nn.Conv.partial(kernel_size=(1, 1), padding="SAME", bias=False)
    needs_projection = x.shape[-1] != filters * 1 or strides != (1, 1)

    identity = x
    out = conv3x3(x, filters, strides=strides, name="conv1")
    out = norm_layer(out, name="bn1")
    out = jax.nn.relu(out)

    out = conv3x3(out, filters, strides=(1, 1), name="conv2")
    out = norm_layer(out, name="bn2")

    if needs_projection:
      x = conv1x1(x, filters, strides=strides, name="proj_conv")
      identity = norm_layer(x, name="proj_bn")

    out += identity
    out = jax.nn.relu(out)

    return out


class BottleneckBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self,
            x,
            filters,
            strides=(1, 1),
            use_attention=False,
            attention_kwargs=None,
            train=True):
    del use_attention, attention_kwargs
    batch_norm = nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.9, epsilon=1e-5)
    conv = nn.Conv.partial(bias=False)
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)

    y = conv(x, filters, kernel_size=(1, 1), strides=(1, 1), name="conv1")
    y = batch_norm(y, name="bn1")
    y = jax.nn.relu(y)
    # Normal resent 3x3 conv
    y = conv(y, filters, kernel_size=(3, 3), strides=strides, name="conv2")
    y = batch_norm(y, name="bn2")
    y = jax.nn.relu(y)
    y = conv(y, filters * 4, kernel_size=(1, 1), strides=(1, 1), name="conv3")
    y = batch_norm(y, name="bn3", scale_init=jax.nn.initializers.zeros)
    if needs_projection:
      x = conv(
          x, filters * 4, kernel_size=(1, 1), strides=strides, name="proj_conv")
      x = batch_norm(x, name="proj_bn")
    return jax.nn.relu(x + y)


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(self,
            x,
            num_filters=64,
            block_sizes=(3, 4, 6, 3),
            train=True,
            block=BottleneckBlock,
            small_inputs=False):
    if small_inputs:
      x = nn.Conv(
          x,
          num_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
          bias=False,
          name="init_conv")
    else:
      x = nn.Conv(
          x,
          num_filters,
          kernel_size=(7, 7),
          strides=(2, 2),
          bias=False,
          name="init_conv")
    x = nn.BatchNorm(
        x, use_running_average=not train, epsilon=1e-5, name="init_bn")
    if not small_inputs:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = block(x, num_filters * 2**i, strides=strides, train=train)

    return x


ResNet18 = ResNet.partial(num_filters=64,
                          block_sizes=(2, 2, 2, 2),
                          block=BasicBlock,
                          small_inputs=False)


ResNet50 = ResNet.partial(num_filters=64,
                          block_sizes=(3, 4, 6, 3),
                          block=BottleneckBlock,
                          small_inputs=False)


class ConstantPositionEmbedding(nn.Module):
  """A constant (not learnable) position embedding."""

  def apply(self, x):
    """Returns a tensor that goes from 0 to +1 in every input axis.

    Args:
      x: Input array, of shape [b, ..., r], where the intermediate dimensions
        are the input dimensionality. For 2D inputs, this will be [b, w, h, r].

    Returns:
      Output array of shape [b, ..., d], where d is the rank of x, and every
      other axis has the same shape as x, but the inputs stretch from 0 to +1.
    """
    # Create position embedding: for us, this is a constant tensor that
    # ranges from [0, +1] in each input dimension.
    # TODO(unterthiner): ugly hack until the flax API supports constant state
    if not hasattr(self, "_pos_embedding"):
      pos_embedding = utils.create_grid(x.shape[1:-1], [0.0, 1.0])
      self._pos_embedding = pos_embedding[None, Ellipsis]  # Add batch axis.
    return self._pos_embedding


class ATSFeatureNetwork(nn.Module):
  """The feature network used for traffic signs dataset by ATS.

  Paper: https://arxiv.org/abs/1905.03711
  Code: https://github.com/idiap/attention-sampling
  """

  def apply(self,
            x,
            strides=(1, 2, 2, 2),
            filters=(32, 32, 32, 32),
            train=True):
    """This is an adaptation of a ResNetv2 used in ATS (see links above).

    Note that the size of each block is fixed to 1 and the first block is only
    a convolution.

    Args:
      x: Input tensor of shape (b, h, w,  c).
      strides: Strides of the blocks.
      filters: Number of filters of each block.
      train: Whether the module is being trained.

    Returns:
      The global averaged and normalized vector representation of each image.
    """
    norm_layer = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)
    conv3x3 = nn.Conv.partial(kernel_size=(3, 3), padding="SAME", bias=False)

    # Make strides a pair of integer instead of an int
    strides = [(s, s) if isinstance(s, int) else s for s in strides]

    x = conv3x3(x, features=filters[0], strides=strides[0])

    for s, f in zip(strides[1:], filters[1:]):
      x = BasicBlockv2(x, stride=s, filters=f, train=train)

    x = norm_layer(x)
    x = nn.relu(x)

    # Global average pooling and l2 normalize.
    x = x.mean(axis=(1, 2))

    return x


class BasicBlockv2(nn.Module):
  """Basic ResNet block taken from ATS paper.

  See
  https://github.com/idiap/attention-sampling/blob/master/scripts/speed_limits.py#L378
  for reference.
  """

  def apply(self, x, *, stride, filters, train):
    norm_layer = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5)
    conv3x3 = nn.Conv.partial(kernel_size=(3, 3), padding="SAME", bias=False)
    conv1x1 = nn.Conv.partial(kernel_size=(1, 1), padding="SAME", bias=False)

    x = norm_layer(x)
    x = nn.relu(x)
    identity = x
    needs_projection = x.shape[-1] != filters or stride != (1, 1)
    if needs_projection:
      identity = conv1x1(x, features=filters, strides=stride)

    x = conv3x3(x, features=filters, strides=stride)
    x = norm_layer(x)
    x = nn.relu(x)
    x = conv3x3(x, features=filters, strides=(1, 1))

    x += identity
    return x
