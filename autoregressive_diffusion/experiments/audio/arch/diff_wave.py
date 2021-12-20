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

"""DiffWave architecture.

Ported from PyTorch to JAX from
  https://github.com/philsyn/DiffWave-unconditional/blob/master/WaveNet.py
"""

from typing import Any, Callable, Iterable, Optional, Tuple

from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np

from autoregressive_diffusion.model.architecture_components import input_embedding
from autoregressive_diffusion.model.architecture_components import layers

Array = jnp.ndarray
Shape = Iterable[int]
Dtype = Any
PRNGKey = Array
InitializerFn = Callable[[PRNGKey, Shape, Dtype], Array]


class ResBlock(nn.Module):
  """Step-conditioned Residual block."""
  features: int
  kernel_size: Tuple[int] = (3,)
  kernel_dilation: Tuple[int] = (1,)
  skip_features: Optional[int] = None
  kernel_init: InitializerFn = nn.initializers.kaiming_normal()
  activation: Callable[[Array], Array] = jax.nn.swish
  is_causal: bool = False

  @nn.compact
  def __call__(self, x, t_embed):
    """Apply the residual block.

    Args:
      x: Inputs of shape [batch, <spatial>, features].
      t_embed: Embedded time steps of shape [batch, dim].

    Returns:
      Mapped inputs of shape [batch, <spatial>, features] for the output and
      skip connections.
    """
    in_features = x.shape[-1]
    if in_features != self.features:
      raise ValueError(
          f'DiffWave ResBlock requires the same number of input ({in_features})'
          f'and output ({self.features}) features.')

    h = x
    if t_embed is not None:
      # Project time step embedding.
      t_embed = nn.Dense(
          in_features,
          name='step_proj')(
              self.activation(t_embed))
      # Reshape to [batch, 1, ..., 1, in_features] for broadcast.
      t_embed = jnp.reshape(
          t_embed,
          (-1,) + (1,) * len(self.kernel_size) + (in_features,))
      h += t_embed

    # Dilated gated conv.
    u = layers.CausalConv(
        self.features,
        self.kernel_size,
        kernel_dilation=self.kernel_dilation,
        kernel_init=self.kernel_init,
        padding='VALID' if self.is_causal else 'SAME',
        is_causal=self.is_causal,
        name='dilated_tanh')(
            h)
    v = layers.CausalConv(
        self.features,
        self.kernel_size,
        kernel_dilation=self.kernel_dilation,
        kernel_init=self.kernel_init,
        padding='VALID' if self.is_causal else 'SAME',
        is_causal=self.is_causal,
        name='dilated_sigmoid')(
            h)
    y = jax.nn.tanh(u) * jax.nn.sigmoid(v)

    # Residual and skip convs.
    residual = nn.Conv(
        self.features,
        (1,) * len(self.kernel_size),
        kernel_init=self.kernel_init,
        name='residual')(
            y)
    skip = nn.Conv(
        self.skip_features or self.features,
        (1,) * len(self.kernel_size),
        kernel_init=self.kernel_init,
        name='skip')(
            y)

    return (x + residual) / np.sqrt(2.), skip


class ResGroup(nn.Module):
  """Residual group with skip connection aggregation and dilation cycling.

  Attributes:
    num_blocks: Number of residual blocks.
    features: Number of ResBlock features.
    skip_features: Number of ResBlock skip connection features.
    kernel_size: Kernel size for ResBlock-s.
    kernel_init: Convolutional kernel initializer.
    dilation_cycle: Dilation cycling length.
    is_causal: Whether to use a causal architecture.
  """
  num_blocks: int
  features: int
  skip_features: Optional[int] = None
  kernel_size: Tuple[int] = (3,)
  kernel_init: InitializerFn = nn.initializers.kaiming_normal()
  dilation_cycle: int = 12  # Max dilation is 2 ** 11 = 2048.
  is_causal: bool = False

  @nn.compact
  def __call__(self, x, t_embed):
    """Apply a residual group.

    Args:
      x: Inputs of shape [batch, <spatial>, features].
      t_embed: Embedded time steps of shape [batch, dim].

    Returns:
      Mapped inputs of shape [batch, <spatial>, skip_features]
    """
    y = 0.
    for i in range(self.num_blocks):
      x, skip = ResBlock(
          features=self.features,
          skip_features=self.skip_features,
          kernel_size=self.kernel_size,
          kernel_dilation=(2 ** (i % self.dilation_cycle),),
          kernel_init=self.kernel_init,
          is_causal=self.is_causal)(
              x, t_embed)
      y += skip
    y /= np.sqrt(self.num_blocks)
    return y


class DiffWave(nn.Module):
  """DiffWave network architecture.

  Attributes:
    num_blocks: Number of residual blocks.
    features: Number of ResBlock features.
    max_time: Number of generation steps (i.e. data dimensionality).
    num_classes: Number of output classes.
    output_features: Number of output features.
    skip_features: Number of ResBlock skip connection features.
    kernel_size: Kernel size for ResBlock-s.
    kernel_init: Convolutional kernel initializer.
    dilation_cycle: ResGroup dilation cycling length.
    is_causal: Whether to use the causal architecture.
  """
  num_blocks: int
  features: int
  max_time: int
  num_classes: int
  output_features: Optional[int] = 1
  skip_features: Optional[int] = None
  kernel_size: Tuple[int] = (3,)
  kernel_init: InitializerFn = nn.initializers.kaiming_normal()
  dilation_cycle: int = 12
  is_causal: bool = False

  @nn.compact
  def __call__(self, x, t, mask, train,
               context = None):
    """Apply the WaveDiff network.

    Args:
      x: Inputs of shape [batch, <spatial>, features].
      t: Time steps of shape [batch].
      mask: Array of the same shape as `x` giving the auto-regressive mask.
      train: If True, the model is ran in training. *Not* used in this
        architecture.
      context: Unused.

    Returns:
      Mapped inputs of shape [batch, <spatial>, skip_features]
    """
    assert context is None

    # Sinusoidal features + MLP for time step embedding.
    # Note: this differs from the DiffWave embedding in several ways:
    # * Time embeddings have different dimensionality: 128-512-512
    #   vs 256-1024-1024.
    # * First convlution has kernel size 3 instead of 1.
    h, t_embed = input_embedding.InputProcessingAudio(
        num_classes=self.num_classes,
        num_channels=self.features,
        max_time=self.max_time,
        is_causal=self.is_causal)(
            x, t, mask, train)
    del x, t, mask

    h = nn.relu(h)
    h = ResGroup(
        num_blocks=self.num_blocks,
        features=self.features,
        skip_features=self.skip_features,
        kernel_size=self.kernel_size,
        dilation_cycle=self.dilation_cycle,
        kernel_init=self.kernel_init,
        is_causal=self.is_causal,
        name='res_group')(
            h, t_embed)

    # Final convolution.
    h = nn.Conv(
        features=self.skip_features or self.features,
        kernel_size=(1,) * len(self.kernel_size),
        kernel_init=self.kernel_init,
        name='flower_conv')(
            h)
    h = nn.relu(h)
    if self.output_features:
      h = nn.Conv(
          features=self.output_features,
          kernel_size=(1,) * len(self.kernel_size),
          kernel_init=nn.initializers.zeros,
          name='class_conv')(
              h)

    return h
