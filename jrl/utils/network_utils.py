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

"""
Some useful network components.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk


class ResidualLayerNormBlock(hk.Module):
  def __init__(
      self,
      layer_sizes,
      activation,
      with_bias=True,
      w_init=None,
      b_init=None,
      name = 'ResidualLayerNormBlock'):
    super().__init__(name=name)

    self._mlp = hk.nets.MLP(
        output_sizes=layer_sizes,
        w_init=w_init,
        b_init=b_init,
        with_bias=with_bias,
        activation=activation,
        activate_final=False)
    self._layer_norm = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True)

  def __call__(self, x):
    y = self._mlp(x)
    return self._layer_norm(x + y)


def build_tree_deep_ensemble_critic(w_init, b_init, use_double_q=False):
  NUM_LAYERS = 3
  NUM_HEADS_PER_LAYER = 4
  HID_DIM = 256

  def tree_deep_ensemble_critic(x):
    splits = [x]
    for _ in range(NUM_LAYERS):
      new_splits = []
      for h in splits:
        h = hk.Linear(
            NUM_HEADS_PER_LAYER * HID_DIM,
            w_init=w_init,
            b_init=b_init,)(h)
        h = jax.nn.relu(h)
        h = jnp.split(h, NUM_HEADS_PER_LAYER, axis=-1)
        new_splits.extend(h)
      splits = new_splits

    h = jnp.concatenate(splits, axis=-1)
    vs = []
    v_dim = 1 if not use_double_q else 2
    for s in splits:
      vs.append(hk.Linear(v_dim, w_init=w_init, b_init=b_init)(s))
    v = jnp.concatenate(vs, axis=-1)
    return v, h

  return tree_deep_ensemble_critic


def build_efficient_tree_deep_ensemble_critic(w_init, b_init, use_double_q=False):
  raise NotImplementedError()
  NUM_LAYERS = 3
  NUM_HEADS_PER_LAYER = 4
  HID_DIM = 256

  def tree_deep_ensemble_critic(x):
    splits = [x]
    for _ in range(NUM_LAYERS):
      new_splits = []
      for h in splits:
        h = jnp.split(h, NUM_HEADS_PER_LAYER, axis=0)
        for sub_h in h:
          sub_h = hk.Linear(
              HID_DIM,
              w_init=w_init,
              b_init=b_init,)(sub_h)
          sub_h = jax.nn.relu(sub_h)
          new_splits.append(sub_h)
      splits = new_splits

    h = jnp.concatenate(splits, axis=-1)
    vs = []
    v_dim = 1 if not use_double_q else 2
    for s in splits:
      vs.append(hk.Linear(v_dim, w_init=w_init, b_init=b_init)(s))
    v = jnp.concatenate(vs, axis=-1)
    return v, h

  return tree_deep_ensemble_critic
