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

"""Miscellaneous utilities."""

import jax
import jax.numpy as jnp


def joint_from_samples(sampler,
                       logits_1,
                       logits_2,
                       rng,
                       num_samples,
                       loop_size=None):
  """Constructs a coupling matrix from a sampler that samples one-hots.

  Args:
    sampler: Function (logits_1, logits_2, rng) -> float[M, M] with a single
      index nonzero, representing the sampled pair.
    logits_1: First logits as a float[M] vector.
    logits_2: Second logits as a float[M] vector.
    rng: PRNGKey
    num_samples: How many samples to average over.
    loop_size: Optional maximum size to execute in a single batched computation.
      Set this if you want to reduce memory usage, in exchange for possibly
      longer computation time.

  Returns:
    Matrix float[M,M] averaged over many samples.
  """
  # JAX note: jax.vmap computes many samples in parallel
  logits_1 = jnp.array(logits_1)
  logits_2 = jnp.array(logits_2)
  if loop_size is None:
    return jnp.mean(
        jax.vmap(lambda key: sampler(logits_1, logits_2, key))(jax.random.split(
            rng, num_samples)),
        axis=0)
  else:
    assert num_samples % loop_size == 0

    def go(i, counts):
      return counts + jnp.sum(
          jax.vmap(lambda key: sampler(logits_1, logits_2, key))(
              jax.random.split(jax.random.fold_in(rng, i), loop_size)),
          axis=0)

    counts = jax.lax.fori_loop(0, num_samples // loop_size, go,
                               jnp.zeros([10, 10]))
    return counts / num_samples


def counterfactual_gumbels(logits, observed_sample, rng):
  """Samples Gumbels conditioned on an argmax using top-down sampling.

  Args:
    logits: Vector of logits for the distribution of interest.
    observed_sample: Sampled index from this distribution.
    rng: PRNGKey

  Returns:
    A vector of gumbels, sampled conditioned on
    `observed_sample == argmax_i (logits[i] + gumbels[i])`
  """
  dim, = logits.shape
  logits = jax.nn.log_softmax(logits)
  uniforms = jax.random.uniform(
      rng, shape=(dim,), minval=jnp.finfo(logits.dtype).tiny, maxval=1.)
  max_gumbel_shifted = -jnp.log(-jnp.log(uniforms[observed_sample]))
  gumbels_shifted = logits - jnp.log(
      jnp.exp(logits - max_gumbel_shifted) - jnp.log(uniforms))
  gumbels_shifted = gumbels_shifted.at[observed_sample].set(max_gumbel_shifted)
  gumbels_unshifted = gumbels_shifted - logits
  return gumbels_unshifted


def gumbel_max_sampler(logits_1, logits_2, rng):
  """Samples from a Gumbel-max coupling."""
  gumbels = jax.random.gumbel(rng, logits_1.shape)
  x = jnp.argmax(gumbels + logits_1)
  y = jnp.argmax(gumbels + logits_2)
  return jnp.zeros([10, 10]).at[x, y].set(1.)


def sampler_from_common_random_numbers(single_sampler,
                                       first_kwargs=None,
                                       second_kwargs=None,
                                       dim=10):
  """Helper function to take a sampler and turn it into a coupling.

  Args:
    single_sampler: Function (logits, key) -> int
    first_kwargs: Optional keyword args for sampling from the first logits.
    second_kwargs: Optional keyword args for sampling from the second logits.
    dim: Dimension of the matrix to return.

  Returns:
    float[M, M] with a single index nonzero, representing the sampled pair.
  """

  def joint_sampler(logits_1, logits_2, key):
    x = single_sampler(logits_1, key, **(first_kwargs or {}))
    y = single_sampler(logits_2, key, **(second_kwargs or {}))
    return jnp.zeros([dim, dim]).at[x, y].set(1.)

  return joint_sampler


def independent_coupling(logits_1, logits_2):
  """Constructs the matrix for an independent coupling."""
  return jnp.exp(logits_1)[:, None] * jnp.exp(logits_2)[None, :]


def inverse_cdf_coupling(logits_1, logits_2):
  """Constructs the matrix for an inverse CDF coupling."""
  dim, = logits_1.shape
  p1 = jnp.exp(logits_1)
  p2 = jnp.exp(logits_2)
  p1_bins = jnp.concatenate([jnp.array([0.]), jnp.cumsum(p1)])
  p2_bins = jnp.concatenate([jnp.array([0.]), jnp.cumsum(p2)])

  # Value in bin (i, j): overlap between bin ranges
  def get(i, j):
    left = jnp.maximum(p1_bins[i], p2_bins[j])
    right = jnp.minimum(p1_bins[i + 1], p2_bins[j + 1])
    return jnp.where(left < right, right - left, 0.0)

  return jax.vmap(lambda i: jax.vmap(lambda j: get(i, j))(jnp.arange(dim)))(
      jnp.arange(dim))


def permuted_inverse_cdf_coupling(logits_1, logits_2, permutation_seed=1):
  """Constructs the matrix for an inverse CDF coupling under a permutation."""
  dim, = logits_1.shape
  perm = jnp.argsort(
      jax.random.uniform(jax.random.PRNGKey(permutation_seed), shape=[dim]))
  invperm = jnp.argsort(perm)
  p1 = jnp.exp(logits_1)[perm]
  p2 = jnp.exp(logits_2)[perm]
  p1_bins = jnp.concatenate([jnp.array([0.]), jnp.cumsum(p1)])
  p2_bins = jnp.concatenate([jnp.array([0.]), jnp.cumsum(p2)])

  # Value in bin (i, j): overlap between bin ranges
  def get(i, j):
    left = jnp.maximum(p1_bins[i], p2_bins[j])
    right = jnp.minimum(p1_bins[i + 1], p2_bins[j + 1])
    return jnp.where(left < right, right - left, 0.0)

  return jax.vmap(lambda i: jax.vmap(lambda j: get(i, j))(invperm))(invperm)


def maximal_coupling_sampler(logits_p, logits_q, rng):
  """Samples from a maximal coupling.

  Based on the algorithm described in
    https://colcarroll.github.io/couplings/static/maximal_couplings.html

  Args:
    logits_p: First logits as a float[M] vector.
    logits_q: Second logits as a float[M] vector.
    rng: PRNGKey

  Returns:
    float[M, M] with a single index nonzero, representing the sampled pair.
  """
  p = jnp.exp(logits_p)
  q = jnp.exp(logits_q)
  x_rng, rng = jax.random.split(rng)
  x = jax.random.categorical(x_rng, logits_p)
  w_rng, rng = jax.random.split(rng)
  w = jax.random.uniform(w_rng) * p[x]

  is_case_1 = w < q[x]

  def case_1():
    return jnp.zeros([10, 10]).at[x, x].set(1.)

  def case_2():

    def while_cond(state):
      done, _, _ = state
      return jnp.logical_not(done)

    def while_body(state):
      _, _, rng = state
      y_rng, rng = jax.random.split(rng)
      y = jax.random.categorical(y_rng, logits_q)
      w_rng, rng = jax.random.split(rng)
      w = jax.random.uniform(w_rng) * q[y]
      return (w > p[y], jnp.zeros([10, 10]).at[x, y].set(1.), rng)

    _, result, _ = jax.lax.while_loop(while_cond, while_body,
                                      (is_case_1, jnp.zeros([10, 10]), rng))
    return result

  return jnp.where(is_case_1, case_1(), case_2())
