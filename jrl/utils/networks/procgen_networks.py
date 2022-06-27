# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Networks used for Procgen.
"""

import gin
from collections import OrderedDict
import functools
import numpy as np

from typing import Optional

import haiku as hk
import acme
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp

base = networks_lib.base
atari = networks_lib.atari
distributional = networks_lib.distributional


@gin.register
def build_procgen_actor_fn(
    num_dimensions,):
  """Deep Atari Torso (3 resnet blocks with fc 256 relu on top), then another
  fc 256 relu on top then categorical."""
  def _actor_fn(obs):
    layers = [atari.DeepAtariTorso()]
    layers.extend([
        hk.Linear(256),
        jax.nn.relu,
        distributional.CategoricalHead(int(num_dimensions),)
    ])
    return hk.Sequential(layers)(obs)

  return _actor_fn


class DeepAtariTorsoWithoutFC(base.Module):
  """Deep torso for Atari, from the IMPALA paper. Removes the FC + relu."""

  def __init__(self, name = 'deep_atari_torso_without_fc'):
    super().__init__(name=name)
    layers = []
    for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
      conv = hk.Conv2D(
          num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME')
      pooling = functools.partial(
          hk.max_pool,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME')
      layers.append(conv)
      layers.append(pooling)

      for j in range(num_blocks):
        block = atari.ResidualBlock(
            num_channels, name='residual_{}_{}'.format(i, j))
        layers.append(block)

    layers.extend([
        jax.nn.relu,
        hk.Flatten(),
    ])
    self._network = hk.Sequential(layers)

  def __call__(self, x):
    return self._network(x)


@gin.register
def build_procgen_critic_neck(
    num_dimensions,
    num_critics,):
  def _critic_fn(obs):
    preds = []
    for _ in range(num_critics):
      layers = [
          hk.Linear(256),
          jax.nn.relu,
          hk.Linear(256),
          jax.nn.relu,
          hk.Linear(num_dimensions),
      ]
      preds.append(hk.Sequential(layers)(obs))
    return jnp.stack(preds, axis=-1)

  return _critic_fn
