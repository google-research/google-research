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

"""VQVAE implementation."""

from typing import Optional

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


from latent_programmer.models import base_models

INF = np.array(1.0e7)
EPS = np.array(1.0e-7)


class VectorQuantizerEMA(nn.Module):
  """Flax module representing the VQVAE embedding.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Slightly modified to use ema when computing the embedding vectors.
  """
  config: base_models.TransformerConfig
  num_embeddings: int
  commitment_cost: float = 0.25
  decay: float = 0.99

  def setup(self):
    cfg = self.config
    embedding_shape = [cfg.emb_dim, self.num_embeddings]

    key = jax.random.PRNGKey(0)
    self.counter = self.variable(
        'vqvae', 'counter', nn.initializers.zeros, key, [])
    self.vq_emb = self.variable(
        'vqvae', 'emb',
        nn.initializers.variance_scaling(1.0, 'fan_in', 'uniform'),
        key, embedding_shape)
    self.ema_emb = self.variable(
        'vqvae', 'ema_emb', nn.initializers.zeros, key, embedding_shape)
    self.cluster_sizes = self.variable(
        'vqvae', 'cs', nn.initializers.zeros, key, [self.num_embeddings])

  def quantize(self,
               latent_indices,
               soft_quantize = False):
    """Returns embedding tensor for a batch of indices."""
    embeddings = self.vq_emb.value
    w = embeddings.swapaxes(1, 0)
    if soft_quantize:
      # Given logits over latent states instead.
      return jnp.dot(nn.softmax(latent_indices), w)
    else:
      return w[latent_indices]

  def attend(self, query):
    """Attend over the embedding using a query array."""
    embeddings = self.vq_emb.value
    w = embeddings.swapaxes(1, 0)
    return lax.dot_general(
        query, w, (((query.ndim - 1,), (1,)), ((), ())))

  def __call__(self,
               query,
               train = True,
               emb_mask = None,
               padding_mask = None):
    """Quantizes query array using VQ discretization bottleneck."""
    flat_query = jnp.reshape(query, [-1, query.shape[-1]])

    is_initialized = self.has_variable('vqvae', 'counter')
    if not is_initialized:
      self.ema_emb.value = self.vq_emb.value
    embeddings = self.vq_emb.value

    distances = (jnp.sum(flat_query**2, 1, keepdims=True)
                 - 2 * jnp.dot(flat_query, embeddings)
                 + jnp.sum(embeddings**2, 0, keepdims=True))
    if emb_mask is not None:  # Mask out some embeddings i.e. pad, BOS, EOS.
      # emb_mask shape == [batch_size, num_embeddings,]
      distances += INF * (1 - emb_mask)

    encoding_indices = jnp.argmin(distances, axis=1)
    encodings = jax.nn.one_hot(
        encoding_indices, self.num_embeddings, dtype=distances.dtype)

    encoding_indices = jnp.reshape(encoding_indices, query.shape[:-1])
    quantized = embeddings.T[encoding_indices]

    e_latent_loss = jnp.mean(
        jnp.square(lax.stop_gradient(quantized) - query),
        axis=-1,
        keepdims=True)

    if train and is_initialized:
      self.counter.value += 1

      dw = jnp.matmul(flat_query.T, encodings)

      decay = lax.convert_element_type(self.decay, dw.dtype)
      # Update ema_cluster_size and ema_emb
      one = jnp.ones([], dw.dtype)
      self.cluster_sizes.value = (self.cluster_sizes.value * self.decay +
                                  jnp.sum(encodings, axis=0) * (one - decay))
      self.ema_emb.value = self.ema_emb.value * decay + dw * (one - decay)

      # Assign updated ema_emb to emb
      updated_ema_emb = self.ema_emb.value

      n = jnp.sum(self.cluster_sizes.value)
      updated_ema_cluster_size = ((self.cluster_sizes.value + EPS) /
                                  (n + self.num_embeddings * EPS) * n)

      normalised_updated_ema_w = (
          updated_ema_emb / jnp.reshape(updated_ema_cluster_size, [1, -1]))
      self.vq_emb.value = normalised_updated_ema_w

    if padding_mask is not None:
      encoding_indices *= padding_mask
      e_latent_loss *= padding_mask[Ellipsis, None]

    loss = self.commitment_cost * e_latent_loss.sum()
    quantized = query + lax.stop_gradient(quantized - query)

    indices = lax.stop_gradient(encoding_indices).astype(jnp.int32)
    return {
        'latents': quantized,
        'loss': loss,
        'latent_indices': indices,}
