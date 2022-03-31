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

"""K-Means utilities used for one way of implementing the SNR regularizer."""

import jax
import jax.numpy as jnp


def _kmeans_update_step(
    data, # N x dim
    prev_centroids, # K x dim
    prev_counts,
    key,
    iters=100,
    decay=0.5,
    counts_decay=0.8, # only used for checking dead threshold
    dead_tolerance = 0.01,
):
  eps = 1e-16
  dead_threshold = max(
      (dead_tolerance * data.shape[0]) / prev_centroids.shape[0],
      1.,
  )

  def _dead_centroid_fix(operand):
    counts, centroids, key = operand
    min_idx = jnp.argmin(counts)
    max_idx = jnp.argmax(counts)
    avg_count = (counts[min_idx, 0] + counts[max_idx, 0]) / 2.
    new_counts = counts.at[min_idx, 0].set(avg_count)
    new_counts = new_counts.at[max_idx, 0].set(avg_count)

    replacement = centroids[max_idx, :]
    replacement = replacement + jax.random.normal(key, replacement.shape)
    new_centroids = centroids.at[min_idx, :].set(replacement)

    return new_counts, new_centroids


  def _iter_condition(state):
    i, _, _, _ = state
    return i < iters

  def _iter_body(state):
    i, centroids, counts, key = state

    centroids_norm = jnp.sum(centroids ** 2, axis=-1, keepdims=True) # K x 1
    data_norm = jnp.sum(data ** 2, axis=-1, keepdims=True) # N x 1
    dot_product = jnp.matmul(data, jnp.transpose(centroids)) # N x K
    distances = data_norm + jnp.transpose(centroids_norm) - 2 * dot_product # N x K

    # centroids_norm = jnp.sum(centroids ** 2, axis=-1, keepdims=True) # K x 1
    # data_norm = jnp.sum(data ** 2, axis=-1, keepdims=True) # N x 1
    # dot_product = jnp.matmul(
    #     data / data_norm,
    #     jnp.transpose(centroids / centroids_norm)) # N x K
    # distances = -1. * dot_product # N x K

    labels = jnp.argmin(distances, axis=1)
    one_hot = jax.nn.one_hot(labels, prev_counts.shape[0])
    # labels = one_hot * norm
    labels = one_hot # N x K
    dw = jnp.matmul(jnp.transpose(one_hot), data) # K x dim
    count = jnp.expand_dims(jnp.sum(one_hot, axis=0), axis=-1) # K x 1
    dw /= (count + eps)
    centroids = decay * centroids + (1 - decay) * dw

    key, sub_key = jax.random.split(key)
    counts = counts_decay * counts + (1 - counts_decay) * count
    counts, centroids = jax.lax.cond(
        pred=jnp.min(counts) < dead_threshold,
        true_fun=_dead_centroid_fix,
        false_fun=lambda operand: (operand[0], operand[1]),
        operand=(counts, centroids, sub_key)
    )

    return (i + 1, centroids, count, key)

  init_state = tuple([0, prev_centroids, prev_counts, key])
  _, new_centroids, new_counts, _ = jax.lax.while_loop(_iter_condition, _iter_body, init_state)

  return new_centroids, new_counts
