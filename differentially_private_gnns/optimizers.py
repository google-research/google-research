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

"""Optimizers."""

import chex
import jax
import jax.numpy as jnp
import optax


def clip_by_norm(updates,
                 l2_norms_threshold):
  """Standard clipping by L2 norm."""

  grad_norms = jax.tree.map(
      jax.vmap(jnp.linalg.norm),
      updates)
  divisors = jax.tree.map(
      lambda g_norm, l2_norm_clip: jnp.maximum(g_norm / l2_norm_clip, 1.0),
      grad_norms, l2_norms_threshold)
  return jax.tree.map(
      jax.vmap(lambda g, div: g / div),
      updates, divisors)


def dp_aggregate(
    l2_norms_threshold,
    base_sensitivity,
    noise_multiplier,
    init_rng,
):
  """Aggregates gradients based on the DP-SGD algorithm.

  This method clips per-example gradients to some l2 norm, sums them up,
  and adds noise to the sum.

  WARNING: Unlike other transforms, `dp_aggregate` expects
  the input updates to have a batch dimension in the 0th axis. That is, this
  function expects per-example gradients as input (which are easy to obtain in
  JAX using `jax.vmap`). It can still be composed with other transformations as
  long as it is the first in the chain.
  Further, each per-example gradient must already be divided by the batch size.

  References:
    [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

  Args:
    l2_norms_threshold: max L2 norm of the per-example gradients for each layer.
    base_sensitivity: ratio of sensitivity to the clipping norm.
    noise_multiplier: ratio of noise standard deviation to the sensitivity.
    init_rng: initial jax.random.PRNGKey

  Returns:
    A `GradientTransformation`.
  """
  noise_stds = jax.tree.map(
      lambda l2_norm_clip: l2_norm_clip * base_sensitivity * noise_multiplier,
      l2_norms_threshold)

  def init_fn(params):
    del params
    return optax.DifferentiallyPrivateAggregateState(
        rng_key=init_rng)

  def update_fn(updates, state, params):
    del params
    grads_flat, grads_treedef = jax.tree.flatten(updates)
    batch_size = grads_flat[0].shape[0]

    if any(g.ndim == 0 or batch_size != g.shape[0] for g in grads_flat):
      raise ValueError(
          'Unlike other transforms, `dp_aggregate` expects'
          ' `updates` to have a batch dimension in the 0th axis. That is, this'
          ' function expects per-example gradients as input.')

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    rng_tree = jax.tree.unflatten(grads_treedef, rngs)

    clipped_updates = clip_by_norm(updates, l2_norms_threshold)
    summed_updates = jax.tree.map(
        lambda g: jnp.sum(g, axis=0),
        clipped_updates)
    noise = jax.tree.map(
        lambda g, std, rng: (std * jax.random.normal(rng, g.shape, g.dtype)),
        summed_updates, noise_stds, rng_tree)
    noisy_updates = jax.tree.map(lambda g, noise: (g + noise), summed_updates,
                                 noise)
    return (noisy_updates,
            optax.DifferentiallyPrivateAggregateState(rng_key=new_key))

  return optax.GradientTransformation(init_fn, update_fn)


def dpsgd(learning_rate, l2_norms_threshold,
          base_sensitivity, noise_multiplier,
          init_rng, momentum,
          nesterov):
  """A differentially-private version of SGD."""
  return optax.chain(
      dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
                   init_rng), optax.sgd(learning_rate, momentum, nesterov))


def dpadam(learning_rate, l2_norms_threshold,
           base_sensitivity, noise_multiplier,
           init_rng):
  """A differentially-private version of Adam."""
  return optax.chain(
      dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
                   init_rng), optax.adam(learning_rate))
