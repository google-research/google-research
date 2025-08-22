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

"""Jax implementation of Fast Weight Layers."""

import functools

from flax import linen as nn
import jax
from jax import numpy as jnp


def exclusive_cumsum(a, axis):
  mask = 1 - nn.one_hot(0, a.shape[axis], dtype=a.dtype)
  mask = mask.reshape(
      ([1] * axis) + [a.shape[axis]] + ([1] * (a.ndim - axis - 1)))
  return mask * jnp.roll(jnp.cumsum(a, axis), 1, axis)


def causal_mask(x):
  seq_len = x.shape[-2]
  return jnp.tri(seq_len, seq_len, -1)


def _local_quadratic_attn(q, k, v):
  qk = jnp.einsum("bgns,bgms->bgnm", q, k) * causal_mask(q)
  return jnp.einsum("bgnm,bgme->bgne", qk, v)


def _global_linear_attn(q, k, v):
  kv = jnp.einsum("bgcs,bgce->bgse", k, v)
  kv = exclusive_cumsum(kv, axis=1)
  return jnp.einsum("bgcs,bgse->bgce", q, kv)


def mixed_chunk_attn(q, k, v, n_chunks):
  """Mixed chunk linear attention from https://arxiv.org/pdf/2202.10447.pdf."""
  if n_chunks == 1:
    # if 1 chunk we do regular linear attention
    qk = jnp.einsum("blh,bmh->blm", q, k) * causal_mask(q)
    return jnp.einsum("blm,bmp->blp", qk, v)
  batch_size, seq_len = q.shape[:2]
  chunked_shape = [batch_size, n_chunks, seq_len // n_chunks, -1]
  q = jnp.reshape(q, chunked_shape)
  v = jnp.reshape(v, chunked_shape)
  k = jnp.reshape(k, chunked_shape)
  attn = _global_linear_attn(q, k, v) + _local_quadratic_attn(q, k, v)
  return jnp.reshape(attn, [batch_size, seq_len, -1])


class FWBlock(nn.Module):
  """NN layers with fast-weight gradient updates for text generation."""

  size: int
  vocab_size: int
  attn_chunks: int

  def setup(self):
    self.layers = [
        FWDense(size=4 * self.size, attn_chunks=self.attn_chunks),
        FWSquaredRelu(),
        FWDense(size=self.size, attn_chunks=self.attn_chunks),
        FWLayerNorm()]
    self.step_sizes = [
        self.param("step_size_" + str(i),
                   nn.initializers.constant(0.01), tuple())
        for i in range(len(self.layers))
    ]
    self.unembed = nn.Dense(self.vocab_size)

  def get_logits(self, fwl_output):
    return self.unembed(fwl_output)

  def slow_weight_fwd(self, x):
    for l in self.layers:
      x = l(x)
    return self.get_logits(x)

  def fast_weight_fwd(self, x, sw_xs, grads):
    for l, sw_x, grad, step_size in zip(
        self.layers, sw_xs, grads, self.step_sizes):
      x = l.fast_weight_fwd(x, sw_x, grad, step_size)
    return self.get_logits(x)

  @functools.partial(jax.grad, argnums=-1, has_aux=True)
  def slow_weight_fwd_bwd(self, x, labels, weights, perturbations):
    sw_xs = []
    for l, p in zip(self.layers, perturbations):
      sw_xs.append(x)
      x = l.fwd_with_perturbation(x, p)
    logits = self.get_logits(x)
    log_probs = -jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
    loss = jnp.sum(weights * log_probs / jnp.maximum(jnp.sum(weights), 1e-8))
    return loss, sw_xs

  def __call__(self, x, labels, weights):
    if self.is_initializing():
      return self.slow_weight_fwd(x)
    perturbations = [l.get_perturbation(x) for l in self.layers]
    grads, sw_xs = self.slow_weight_fwd_bwd(x, labels, weights, perturbations)
    return self.fast_weight_fwd(x, sw_xs, grads)


class FWModule(nn.Module):
  """A flax module using fast weights."""

  def replicated_params(self, x):
    return jax.tree.map(
        lambda v: jnp.tile(v, (x.shape[0], x.shape[1], 1)), self.variables)

  def get_perturbation(self, x):
    return jax.tree.map(jnp.zeros_like, self.replicated_params(x))

  def fwd_with_perturbation(self, x, perturbation):
    new_params = jax.tree.map(
        lambda v, p: v + p, self.replicated_params(x), perturbation)
    return jax.vmap(jax.vmap(self.apply))(new_params, x)

  def fast_weight_fwd(self, x, sw_x, grad, step_size):
    return self.fwd_with_perturbation(x, jax.tree_util.tree_map(
        lambda g: -exclusive_cumsum(g, axis=1) * step_size, grad))


class FWLayerNorm(FWModule):
  @nn.compact
  def __call__(self, x):
    return nn.LayerNorm()(x)


class FWSquaredRelu(FWModule):
  @nn.compact
  def __call__(self, x):
    return jnp.square(nn.relu(x))


class FWDense(nn.Module):
  """Dense layer using fast weights.

  This implementation is much more efficient than simply using a FWModule with
  nn.Dense because it takes advantage of weight matrix gradients being rank one
  to express the output as linear causal attention.
  """

  size: int
  attn_chunks: int

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.size)(x)

  def get_perturbation(self, x):
    return jnp.zeros(list(x.shape[:-1]) + [self.size])

  def fwd_with_perturbation(self, x, perturbation):
    return self(x) + perturbation

  def fast_weight_fwd(self, x, sw_x, grad, step_size):
    output = self(x)
    output -= step_size * mixed_chunk_attn(x, sw_x, grad, self.attn_chunks)
    return output
