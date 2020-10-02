# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Functions for running EM on GMMs."""
from functools import partial

from . import util

import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy


@partial(jit, static_argnums=1)
def em(X, k, T, key):
  n, _ = X.shape
  # Initialize centroids using k-means++ scheme
  centroids = kmeans_pp_init(X, k, key)
  # [k, d, d]
  covs = jnp.array([jnp.cov(X, rowvar=False)]*k)
  # centroid mixture weights, [k]
  log_weights = -jnp.ones(k)*jnp.log(n)

  def update_centroids_body(unused_t, state):
    centroids, covs, log_weights, _ = state
    # E step
    # [n, k]
    log_ps = vmap(
        jscipy.stats.multivariate_normal.logpdf,
        in_axes=(None, 0, 0))(X, centroids, covs)
    log_ps = log_ps.T
    # [n, 1]
    log_zs = jscipy.special.logsumexp(
        log_ps + log_weights[jnp.newaxis, :], axis=1, keepdims=True)

    # [n, k]
    log_mem_weights = log_ps + log_weights[jnp.newaxis, :] - log_zs
    # M step
    # [k]
    log_ns = jscipy.special.logsumexp(log_mem_weights, axis=0)
    # [k]
    log_weights = log_ns - jnp.log(n)
    # Compute new centroids
    # [k, d]
    centroids = jnp.sum(
        (X[:, jnp.newaxis, :] * jnp.exp(log_mem_weights)[:, :, jnp.newaxis]) /
        jnp.exp(log_ns[jnp.newaxis, :, jnp.newaxis]),
        axis=0)
    # [n, k, d]
    centered_x = X[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :]
    # [n, k, d, d]
    outers = jnp.einsum('...i,...j->...ij', centered_x, centered_x)
    weighted_outers = outers * jnp.exp(log_mem_weights[Ellipsis, jnp.newaxis,
                                                       jnp.newaxis])
    covs = jnp.sum(
        weighted_outers, axis=0) / jnp.exp(log_ns[:, jnp.newaxis, jnp.newaxis])
    return (centroids, covs, log_weights, log_mem_weights)

  out_centroids, out_covs, _, log_mem_weights = jax.lax.fori_loop(
      0, T, update_centroids_body,
      (centroids, covs, log_weights, jnp.zeros([n, k])))
  return out_centroids, out_covs, log_mem_weights


@partial(jit, static_argnums=1)
def kmeans_pp_init(X, k, key):
  keys = jax.random.split(key, num=k)
  n, d = X.shape
  centroids = jnp.ones([k, d]) * jnp.inf
  centroids = jax.ops.index_update(centroids, 0,
                                   X[jax.random.randint(keys[0], [], 0, n), :])
  dist = lambda x, y: jnp.linalg.norm(x - y, axis=0, keepdims=False)

  def for_body(i, centroids):
    dists = vmap(vmap(dist, in_axes=(None, 0)), in_axes=(0, None))(X, centroids)
    min_square_dists = jnp.square(jnp.min(dists, axis=1))
    new_centroid_ind = jax.random.categorical(keys[i],
                                              jnp.log(min_square_dists))
    centroids = jax.ops.index_update(centroids, i, X[new_centroid_ind, :])
    return centroids

  return jax.lax.fori_loop(1, k, for_body, centroids)


def em_accuracy(xs, cs, k, key, num_iterations=25):
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  em_predicted_cs = jnp.argmax(em_log_membership_weights, axis=1)
  return util.permutation_invariant_accuracy(em_predicted_cs, cs, k)


def em_map(xs, k, key, num_iterations=25):
  """Computes EM's MAP estimate of cluster assignments."""
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  return jnp.argmax(em_log_membership_weights, axis=1)


def em_pairwise_metrics(xs, cs, k, key, num_iterations=25):
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  em_predicted_cs = jnp.argmax(em_log_membership_weights, axis=1)
  em_pairwise_preds = util.to_pairwise_preds(em_predicted_cs)
  true_pairwise_cs = util.to_pairwise_preds(cs)
  acc = jnp.mean(em_pairwise_preds == true_pairwise_cs)
  f1 = util.binary_f1(em_pairwise_preds, true_pairwise_cs)
  return acc, f1
