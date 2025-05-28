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

"""Layers in jraph/flax."""

from typing import Any, Callable
from flax import linen as nn
import jax
import jax.numpy as jnp


class PReLU(nn.Module):
  """A PReLU Layer."""
  init_fn: Callable[[Any], Any] = nn.initializers.uniform()

  @nn.compact
  def __call__(self, x):
    leakage = self.param('leakage', self.init_fn, [1])
    return jnp.maximum(0, x) + leakage * jnp.minimum(0, x)


class Activation(nn.Module):
  """Activation function."""
  activation: str

  def setup(self):
    if self.activation == 'ReLU':
      self.act_fn = nn.relu
    elif self.activation == 'SeLU':
      self.act_fn = jax.nn.selu
    elif self.activation == 'PReLU':
      self.act_fn = PReLU()
    else:
      raise 'Activation not recognized'

  def __call__(self, x):
    return self.act_fn(x)


class Bilinear(nn.Module):
  """A Bilinear Layer."""
  init_fn: Callable[[Any], Any] = nn.initializers.normal()

  @nn.compact
  def __call__(self, x_l, x_r):
    kernel = self.param('kernel', self.init_fn, [x_l.shape[-1], x_r.shape[-1]])
    return x_l @ kernel @ jnp.transpose(x_r)


class EucCluster(nn.Module):
  """Learnable KMeans Clustering."""
  num_reps: int
  init_fn: Callable[[Any], Any] = nn.initializers.normal()

  @nn.compact
  def __call__(self, x):
    centers = self.param('centers', self.init_fn, [self.num_reps, x.shape[-1]])
    dists = jnp.sqrt(pairwise_sqeuc_dists(x, centers))
    return jnp.argmin(dists, axis=0), jnp.min(dists, axis=1), centers


@jax.jit
def dgi_readout(node_embs):
  return jax.nn.sigmoid(jnp.mean(node_embs, axis=0))


def subtract_mean(embs):
  return embs - jnp.mean(embs, axis=0)


def divide_by_l2_norm(embs):
  norm = jnp.linalg.norm(embs, axis=1, keepdims=True)
  return embs / norm


@jax.jit
def normalize(node_embs):
  return divide_by_l2_norm(subtract_mean(node_embs))


@jax.jit
def pairwise_sqeuc_dists(x, y):
  """Pairwise square Euclidean distances."""
  n = x.shape[0]
  m = y.shape[0]
  x_exp = jnp.expand_dims(x, axis=1).repeat(m, axis=1).reshape(n * m, -1)
  y_exp = jnp.expand_dims(y, axis=0).repeat(n, axis=0).reshape(n * m, -1)
  return jnp.sum(jnp.power(x_exp - y_exp, 2), axis=1).reshape(n, m)
