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

"""Utilities for computing the MICo loss."""

import functools
import gin
import jax
from jax import custom_jvp
import jax.numpy as jnp


EPSILON = 1e-9


# The following two functions were borrowed from
# https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py
# as they resolve the instabilities observed when using `jnp.arccos`.
@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return jnp.sqrt(jnp.maximum(x, tol))


@_sqrt.defjvp
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  return square_root, jnp.where(x > safe_tol, x_dot / (2 * square_root), 0.)


def l2(x, y):
  return jnp.linalg.norm(x - y)


def cosine_distance(x, y):
  numerator = jnp.sum(x * y)
  denominator = jnp.sqrt(jnp.sum(x**2)) * jnp.sqrt(jnp.sum(y**2))
  cos_similarity = numerator / (denominator + EPSILON)
  return jnp.arctan2(_sqrt(1. - cos_similarity**2), cos_similarity)


def squarify(x):
  batch_size = x.shape[0]
  if len(x.shape) > 1:
    representation_dim = x.shape[-1]
    return jnp.reshape(jnp.tile(x, batch_size),
                       (batch_size, batch_size, representation_dim))
  return jnp.reshape(jnp.tile(x, batch_size), (batch_size, batch_size))


@gin.configurable
def representation_distances(first_representations, second_representations,
                             distance_fn, beta=0.1,
                             return_distance_components=False):
  """Compute distances between representations.

  This will compute the distances between two representations.

  Args:
    first_representations: first est of representations to use.
    second_representations: second set of representations to use.
    distance_fn: function to use for computing representatiion distances.
    beta: float, weight given to cosine distance between representations.
    return_distance_components: bool, whether to return the components used for
      computing the distance.

  Returns:
    The distances between representations, combining the average of the norm of
    the representations and the angular distances.
  """
  batch_size = first_representations.shape[0]
  representation_dim = first_representations.shape[-1]
  first_squared_reps = squarify(first_representations)
  first_squared_reps = jnp.reshape(first_squared_reps,
                                   [batch_size**2, representation_dim])
  second_squared_reps = squarify(second_representations)
  second_squared_reps = jnp.transpose(second_squared_reps, axes=[1, 0, 2])
  second_squared_reps = jnp.reshape(second_squared_reps,
                                    [batch_size**2, representation_dim])
  base_distances = jax.vmap(distance_fn, in_axes=(0, 0))(first_squared_reps,
                                                         second_squared_reps)
  norm_average = 0.5 * (jnp.sum(jnp.square(first_squared_reps), -1) +
                        jnp.sum(jnp.square(second_squared_reps), -1))
  if return_distance_components:
    return norm_average + beta * base_distances, norm_average, base_distances
  return norm_average + beta * base_distances


def absolute_reward_diff(r1, r2):
  return jnp.abs(r1 - r2)


@gin.configurable
def target_distances(representations, rewards, distance_fn, cumulative_gamma):
  """Target distance using the metric operator."""
  next_state_distances = representation_distances(
      representations, representations, distance_fn)
  squared_rews = squarify(rewards)
  squared_rews_transp = jnp.transpose(squared_rews)
  squared_rews = squared_rews.reshape((squared_rews.shape[0]**2))
  squared_rews_transp = squared_rews_transp.reshape(
      (squared_rews_transp.shape[0]**2))
  reward_diffs = absolute_reward_diff(squared_rews, squared_rews_transp)
  return (
      jax.lax.stop_gradient(
          reward_diffs + cumulative_gamma * next_state_distances))
