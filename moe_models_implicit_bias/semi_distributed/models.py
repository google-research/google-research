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

"""Multilayer Perceptron based models and variants with Mixture of Experts."""

import functools

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


class MLP(nn.Module):
  """Multilayer perceptron model."""
  config: ml_collections.ConfigDict
  mu: np.ndarray

  @nn.compact
  def __call__(self, x, train = True):

    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5)

    for i, feat in enumerate(self.config.widths + [1]):
      x = nn.Dense(
          feat,
          name=f'layers_{i}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      if i != len(self.config.widths + [1]) - 1:
        x = norm()(x)
        x = nn.relu(x)
        z = x

    _ = norm()(x)

    return x / 4, z


class LabelModel(nn.Module):
  """A binary classification model.

  Uses an MLP backbone.
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, train = True):

    widths = [1000, 1000, 1]

    if self.config.label_rank > 0:
      x = nn.Dense(
          self.config.label_rank,
          name='layers_rank',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)

    for i, feat in enumerate(widths):
      x = nn.Dense(
          feat,
          name=f'layers_{i}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      if i != len(widths) - 1:
        x = nn.relu(x)

    return x


class ExpertMLP(nn.Module):
  """A Multilayer perceptron model with mixture of experts."""
  config: ml_collections.ConfigDict
  mu: np.ndarray

  @nn.compact
  def __call__(self, x, expert_scale=1, train = True):
    config = self.config
    norm = functools.partial(
        nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5)

    if self.config.inp_type == 'mog':
      expert = nn.Dense(
          self.config.num_experts,
          name='expert_1',
          use_bias=False,
          kernel_init=nn.initializers.variance_scaling(
              scale=1.0, mode='fan_in', distribution='truncated_normal'))(
                  x)
    elif self.config.inp_type == 'mos':
      expert = nn.Dense(
          12 * self.config.num_experts, name='expert_1', use_bias=False)(
              x)
      expert = norm()(expert)
      expert = nn.relu(expert)
      expert = nn.Dense(
          12 * self.config.num_experts, name='expert_2', use_bias=False)(
              x)
      expert = norm()(expert)
      expert = nn.relu(expert)
      expert = nn.Dense(
          self.config.num_experts, name='expert_3', use_bias=True)(
              x)

    if config.top_1 == 0:
      expert = nn.softmax(expert_scale * expert)
    elif config.top_1 == 1:
      expert = nn.softmax(expert_scale * expert) * nn.softmax(1000 * expert)
    elif config.top_1 == 2:
      expert = jax.lax.stop_gradient(nn.softmax(1000 * expert))

    if config.router_only_scale:
      scale_param = self.param('scale_param', nn.initializers.zeros, (1))
      expert = (1. + scale_param *
                (config.dim**.5)) * jax.lax.stop_gradient(expert)

    if self.config.depth == 0:
      weights_2 = self.param('layers_1', nn.initializers.zeros,
                             (self.config.num_experts, config.out_dim))
      x2 = weights_2[None, :] * (config.dim**.5)
      x = jnp.repeat(x2, x.shape[0], axis=0)

      x = jnp.einsum('ijk,ij->ik', x, expert)
      # x = jnp.reshape(x, (x.shape[0]))

    if self.config.depth == 1:
      x = nn.Dense(
          self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    if self.config.depth == 2:
      weights_2 = self.param(
          'weights_2',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[0], self.config.num_experts))

      x = nn.Dense(
          self.config.widths[0] * self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.reshape(
          x, (x.shape[0], self.config.num_experts, self.config.widths[0]))

      x = jnp.einsum('ijk,kj->ij', x, weights_2)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    if config.depth == 3:
      weights_2 = self.param(
          'weights_2',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[0],
           self.config.widths[1] * self.config.num_experts))

      weights_3 = self.param(
          'weights_3',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[1], self.config.num_experts))

      x = nn.Dense(
          self.config.widths[0] * self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.reshape(
          x, (x.shape[0], self.config.num_experts, self.config.widths[0]))

      weights_2_view = jnp.reshape(weights_2,
                                   (config.num_experts, config.widths[0], -1))

      x = jnp.einsum('ijk,jkw->ijw', x, weights_2_view)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.einsum('ijw,wj->ij', x, weights_3)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    if config.depth == 4:
      weights_2 = self.param(
          'weights_2',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[0],
           self.config.widths[1] * self.config.num_experts))

      weights_3 = self.param(
          'weights_3',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[1],
           self.config.widths[2] * self.config.num_experts))

      weights_4 = self.param(
          'weights_4',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[2], self.config.num_experts))

      x = nn.Dense(
          self.config.widths[0] * self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.reshape(
          x, (x.shape[0], self.config.num_experts, self.config.widths[0]))

      weights_2_view = jnp.reshape(weights_2,
                                   (config.num_experts, config.widths[0], -1))

      weights_3_view = jnp.reshape(weights_3,
                                   (config.num_experts, config.widths[1], -1))

      x = jnp.einsum('ijk,jkw->ijw', x, weights_2_view)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.einsum('ijk,jkw->ijw', x, weights_3_view)
      x = norm()(x)
      x = nn.relu(x)
      x = jnp.einsum('ijw,wj->ij', x, weights_4)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    return x / 4, expert


class ExpertMLPNoBN(nn.Module):
  """A Multilayer perceptron model with mixture of experts.

  Does not use batch norm.
  """
  config: ml_collections.ConfigDict
  mu: np.ndarray

  @nn.compact
  def __call__(self, x, expert_scale=1, train = True):
    config = self.config

    if self.config.inp_type == 'mog':
      expert = nn.Dense(
          self.config.num_experts,
          name='expert_1',
          use_bias=False,
          kernel_init=nn.initializers.variance_scaling(
              scale=1.0, mode='fan_in', distribution='truncated_normal'))(
                  x)
    elif self.config.inp_type == 'mos':
      expert = nn.Dense(
          4 * self.config.num_experts, name='expert_1', use_bias=False)(
              x)
      expert = nn.relu(expert)
      expert = nn.Dense(
          self.config.num_experts, name='expert_2', use_bias=False)(
              x)

    if config.top_1 == 0:
      expert = nn.softmax(expert_scale * expert)
    elif config.top_1 == 1:
      expert = nn.softmax(expert_scale * expert) * nn.softmax(1000 * expert)
    elif config.top_1 == 2:
      expert = jax.lax.stop_gradient(nn.softmax(1000 * expert))

    if self.config.depth == 1:
      x = nn.Dense(
          self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    if self.config.depth == 2:
      weights_2 = self.param(
          'weights_2',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[0], self.config.num_experts))

      x = nn.Dense(
          self.config.widths[0] * self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = nn.relu(x)
      x = jnp.reshape(
          x, (x.shape[0], self.config.num_experts, self.config.widths[0]))

      x = jnp.einsum('ijk,kj->ij', x, weights_2)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    if config.depth == 3:
      weights_2 = self.param(
          'weights_2',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[0],
           self.config.widths[1] * self.config.num_experts))

      weights_3 = self.param(
          'weights_3',
          nn.initializers.lecun_uniform(),  # Initialization function
          (self.config.widths[1], self.config.num_experts))

      x = nn.Dense(
          self.config.widths[0] * self.config.num_experts,
          name=f'layers_{1}',
          use_bias=False,
          kernel_init=nn.initializers.lecun_uniform())(
              x)
      x = nn.relu(x)
      x = jnp.reshape(
          x, (x.shape[0], self.config.num_experts, self.config.widths[0]))

      weights_2_view = jnp.reshape(weights_2,
                                   (config.num_experts, config.widths[0], -1))

      x = jnp.einsum('ijk,jkw->ijw', x, weights_2_view)
      x = nn.relu(x)
      x = jnp.einsum('ijw,wj->ij', x, weights_3)
      x = jnp.einsum('ij,ij->i', x, expert)
      x = jnp.reshape(x, (x.shape[0], 1))

    return x / 4
