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

"""Common networks."""
import chex
from flax import linen as nn
from jax import numpy as jnp
from pvn.utils import mesh_utils


class NatureDqnEncoder(nn.Module):
  """An encoder network for use with Atari."""

  num_features: int = 512
  width_multiplier: float = 1.0
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  apply_final_relu: bool = True

  @nn.compact
  def __call__(self, x):
    chex.assert_type(x, self.dtype)

    initializer = nn.initializers.xavier_uniform()
    x = nn.Conv(
        features=int(32 * self.width_multiplier),
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=int(64 * self.width_multiplier),
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=int(64 * self.width_multiplier),
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    x = nn.Dense(
        features=self.num_features,
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    if self.apply_final_relu:
      x = nn.relu(x)
    return x


class NatureRndNetwork(nn.Module):
  """A modified Nature DQN network that outputs a single scalar value."""

  features: int
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    chex.assert_type(x, self.dtype)

    initializer = nn.initializers.xavier_uniform()
    x = NatureDqnEncoder(dtype=self.dtype, param_dtype=self.param_dtype)(x)
    x = nn.Dense(
        features=self.features,
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    return x


class ResidualBlock(nn.Module):
  """Stack of pooling and convolutional blocks with residual connections."""

  num_channels: int
  num_blocks: int
  use_max_pooling: bool = True
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    chex.assert_type(x, self.dtype)

    initializer = nn.initializers.xavier_uniform()
    conv_out = nn.Conv(
        features=self.num_channels,
        kernel_init=initializer,
        kernel_size=(3, 3),
        strides=1,
        padding='SAME',
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    if self.use_max_pooling:
      conv_out = nn.max_pool(
          conv_out, window_shape=(3, 3), padding='SAME', strides=(2, 2)
      )

    for _ in range(self.num_blocks):
      block_input = conv_out
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_channels,
          kernel_init=initializer,
          kernel_size=(3, 3),
          strides=1,
          padding='SAME',
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(conv_out)
      conv_out = nn.relu(conv_out)
      conv_out = nn.Conv(
          features=self.num_channels,
          kernel_init=initializer,
          kernel_size=(3, 3),
          strides=1,
          padding='SAME',
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(conv_out)
      conv_out += block_input

    return conv_out


class ImpalaEncoder(nn.Module):
  """Impala Network which also outputs penultimate representation layers."""

  width_multiplier: float = 1.0
  stack_sizes: tuple[int, Ellipsis] = (16, 32, 32)
  num_blocks: int = 2
  num_features: int = 512
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    chex.assert_type(x, self.dtype)

    initializer = nn.initializers.xavier_uniform()

    for stack_size in self.stack_sizes:
      x = ResidualBlock(
          num_channels=int(stack_size * self.width_multiplier),
          num_blocks=self.num_blocks,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)

    x = nn.relu(x)
    x = x.reshape(-1)

    x = nn.Dense(
        features=int(self.num_features),
        kernel_init=initializer,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x)
    x = nn.relu(x)
    return x


class DsmNetwork(nn.Module):
  """A network that predicts DSM action-values and thresholds DSM rewards."""

  num_actions: int
  num_auxiliary_tasks: int
  encoder: nn.Module
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  input_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, obs):
    initializer = nn.initializers.xavier_uniform()

    obs = obs.astype(self.input_dtype) / 255.0
    phi = self.encoder(obs)

    vmap_action_preds = nn.vmap(
        nn.Dense,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=0,
        axis_size=self.num_auxiliary_tasks,
    )
    action_preds = vmap_action_preds(
        features=self.num_actions,
        kernel_init=initializer,
        param_dtype=self.param_dtype,
        name='aux_tasks',
    )(phi)
    action_preds = mesh_utils.with_sharding_constraint(
        action_preds, mesh_utils.create_partition_spec('model')
    )

    return action_preds
