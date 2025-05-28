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

"""Contains utilities for specific model choices."""

from typing import Union

from absl import logging
import chex
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import sklearn.decomposition
import tensorflow as tf


def get_timestep_embedding(timesteps,
                           embedding_dim,
                           max_time=1000.,
                           dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    max_time: float: largest time input
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  chex.assert_rank(timesteps, 1)  # and timesteps.dtype == tf.int32
  timesteps *= (1000. / max_time)

  half_dim = embedding_dim // 2
  emb = np.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))

  chex.assert_shape(emb, (timesteps.shape[0], embedding_dim))
  return emb


def central_crop(inputs, target_shape):
  """Returns a central crop in axis (1, 2).

  Args:
    inputs: nd-array; Inputs in shape of `[bs, height, width, channels]'.
    target_shape: tuple(int); Target shape after crop.

  Returns:
    Cropped image.
  """
  h, w = target_shape[1:3]
  assert h <= inputs.shape[1], f"{h} > {inputs.shape[1]}"
  assert w <= inputs.shape[2], f"{w} > {inputs.shape[2]}"
  h0 = (inputs.shape[1] - h) // 2
  w0 = (inputs.shape[2] - w) // 2
  return inputs[:, h0:(h0 + h), w0:(w0 + w)]


@flax.struct.dataclass
class CachedMatrixPowerState:
  """Matrix power state that caches powers of two."""
  cache: chex.Array

  @staticmethod
  def precompute(matrix,
                 max_power = 100,
                 precision=jax.lax.Precision.DEFAULT,
                 use_numpy = False):
    """Builds state for computing efficient matrix_power vector products.

    Args:
      matrix: an [N,N] matrix to compute powers of.
      max_power: the maximum power to support matrix powers for.
      precision: precision of matmuls while generating state.
      use_numpy: if True, will use maximum precision with numpy float64.

    Returns:
      a Jax array of size [ceil(log_2(max_power)), N, N] which amortizes matrix
      power computation.
    """
    max_steps = np.ceil(np.log2(max_power)).astype(np.int32)

    if use_numpy:
      original_dtype = matrix.dtype
      matrix = np.asarray(matrix, np.float64)
      buffer = np.zeros((max_steps,) + matrix.shape, np.float64)

      for i in range(max_steps):
        buffer[i] = matrix
        matrix = np.dot(matrix, matrix)

      return CachedMatrixPowerState(jnp.asarray(buffer, dtype=original_dtype))

    def _state_init_body_fn(current_arr, _):
      new_arr = jnp.dot(current_arr, current_arr, precision=precision)
      return new_arr, new_arr

    _, arrs = jax.lax.scan(_state_init_body_fn, matrix, None, max_steps - 1)

    return CachedMatrixPowerState(jnp.concatenate([matrix[None], arrs]))

  def matrix_power_multiply(self,
                            x,
                            power,
                            transpose=False,
                            precision=jax.lax.Precision.DEFAULT):
    """Computes matrix vector product jnp.linalg.matrix_power(M, power) @ x.

    Args:
      x: the vector to multiply with the matrix M.
      power: the power to raise M to. Note that this power must be less than or
        equal to max_power given to init_matrix_power_state.
      transpose: if True, computes the product with M_T^power instead.
      precision: precision with which matrix multiplcations are performed.

    Returns:
      the matrix-vector product jnp.linalg.matrix_power(M, power) @ x.
    """
    chex.assert_rank(x, 1)

    num_steps = self.cache.shape[0]

    cache = self.cache
    if transpose:
      cache = jnp.moveaxis(self.cache, -1, -2)

    @chex.dataclass
    class PowerState:
      step: int
      current_power: int
      current_x: chex.Array

    def power_body_fn(state, arr):
      power = 2**state.step
      cond = state.current_power >= power
      new_x = jnp.dot(arr, state.current_x, precision=precision)
      new_x = jnp.where(cond, new_x, state.current_x)
      new_power = jnp.where(cond, state.current_power - power,
                            state.current_power)

      return PowerState(
          step=state.step - 1,
          current_power=new_power,
          current_x=new_x,
      ), None

    init_state = PowerState(
        step=num_steps - 1, current_power=power, current_x=x)
    final_state, _ = jax.lax.scan(
        power_body_fn, init_state, cache, num_steps, reverse=True)

    ## sanity check to make sure product was actually correct.
    return jnp.where(final_state.current_power == 0, final_state.current_x,
                     jnp.full_like(final_state.current_x, jnp.nan))

  def matrix_power(self,
                   power,
                   precision=jax.lax.Precision.DEFAULT):
    """Computes matrix power jnp.linalg.matrix_power(M, power) efficiently.

    Args:
      power: the power to raise M to. Note that this power must be less than or
        equal to max_power given to init_matrix_power_state.
      precision: precision with which matrix multiplcations are performed.

    Returns:
      the matrix-power of M to the power power.
    """
    num_steps = self.cache.shape[0]

    @chex.dataclass
    class PowerState:
      step: int
      current_power: int
      current_mat: chex.Array

    def power_body_fn(state, arr):
      power = 2**state.step
      cond = state.current_power >= power
      new_mat = jnp.dot(arr, state.current_mat, precision=precision)
      new_mat = jnp.where(cond, new_mat, state.current_mat)
      new_power = jnp.where(cond, state.current_power - power,
                            state.current_power)

      return PowerState(
          step=state.step - 1,
          current_power=new_power,
          current_mat=new_mat,
      ), None

    init_state = PowerState(
        step=num_steps - 1,
        current_power=power,
        current_mat=jnp.eye(self.cache.shape[-1]))
    final_state, _ = jax.lax.scan(
        power_body_fn, init_state, self.cache, num_steps, reverse=True)

    ## sanity check to make sure product was actually correct.
    return jnp.where(final_state.current_power == 0, final_state.current_mat,
                     jnp.full_like(final_state.current_mat, jnp.nan))


@flax.struct.dataclass
class LazyMatrixPowerState:
  """Lazy on-demand computer of matrix powers."""
  base_matrix: chex.Array

  def matrix_power_multiply(self,
                            x,
                            power,
                            transpose=False,
                            precision=jax.lax.Precision.HIGHEST):
    """Computes matrix vector product jnp.linalg.matrix_power(M, power) @ x.

    Args:
      x: the matrix or vector to multiply with the matrix M.
      power: the power to raise M to. Note that this power must be less than or
        equal to max_power given to init_matrix_power_state.
      transpose: if True, computes the product with M_T^power instead.
      precision: precision with which matrix multiplcations are performed.

    Returns:
      the matrix-vector product jnp.linalg.matrix_power(M, power) @ x.
    """
    chex.assert_rank(x, {1, 2})

    if transpose:
      base_matrix = self.base_matrix.T
    else:
      base_matrix = self.base_matrix

    z = base_matrix
    n, bit = jnp.divmod(power, 2)
    r = jnp.where(bit, jnp.dot(z, x, precision=precision), x)

    def cond(state):
      n, _, _ = state
      return n > 0

    def body(state):
      n, z, r = state
      z = jnp.dot(z, z, precision=precision)
      n, bit = jnp.divmod(n, 2)
      r = jnp.where(bit, jnp.dot(z, r, precision=precision), r)
      return n, z, r

    _, _, result = jax.lax.while_loop(cond, body, (n, z, r))
    return result

  def matrix_power(self,
                   power,
                   precision=jax.lax.Precision.HIGHEST):
    """Computes matrix power jnp.linalg.matrix_power(M, power) efficiently.

    Args:
      power: the power to raise M to. Note that this power must be less than or
        equal to max_power given to init_matrix_power_state.
      precision: precision with which matrix multiplcations are performed.

    Returns:
      the matrix-power of M to the power power.
    """
    return self.matrix_power_multiply(
        x=jnp.eye(self.base_matrix.shape[0]), power=power, precision=precision)


def get_embedding(params):
  return params["params"]["embedder"]["embedding"]


def get_nearest_neighbors(embeddings,
                          num_chunks = 16,
                          k = 10,
                          return_distances = False,
                          include_self=False):
  """Computes the nearest neighbors for a set of word embeddings in chunks.

  Args:
    embeddings: [num embeddings, dimension], a Jax array containing word
      embeddings.
    num_chunks (int): the number of chunks to use to split the computation. If
      an OOM occurs, increase this number.
    k (int): the number of nearest neighbors to return.
    return_distances: if True, will return distances to the top k neighbors.
    include_self: if True, includes self as a nearest neighbor.

  Returns:
    an integer array of nearest neighbor indices, and optionally, an array of
      floats of the same shape.
  """
  embeddings = jnp.asarray(embeddings)

  if (num_chunks > embeddings.shape[0]) or (num_chunks < 1):
    raise ValueError(
        "num_chunks must be smaller than the number of embeddings and greater "
        "or equal to 1.")

  if embeddings.ndim != 2:
    raise ValueError("embeddings must have dimension 2 (num_embeddings, dim).")

  interval = np.ceil(embeddings.shape[0] / num_chunks).astype(np.int32)

  def nn_body_fn(start_idx, _):
    embed_slice = jax.lax.dynamic_slice(embeddings, (start_idx, 0),
                                        (interval, embeddings.shape[1]))[:,
                                                                         None]
    distances = jnp.linalg.norm(embed_slice - embeddings[None, :], axis=-1)
    if include_self:
      neighbors = distances.argsort(axis=-1)[:, :k]
    else:
      neighbors = distances.argsort(axis=-1)[:, 1:1 + k]

    if return_distances:
      distances = jax.vmap(lambda v, i: v[i])(distances, neighbors)
      return start_idx + interval, (neighbors, distances)
    else:
      return start_idx + interval, neighbors

  _, result = jax.lax.scan(nn_body_fn, 0, None, num_chunks)

  def _reshape(arr):
    arr = arr.reshape((num_chunks * interval, -1))
    arr = arr[:embeddings.shape[0]]
    return arr

  return jax.tree.map(_reshape, result)


def naive_expm(matrix, iterations=10):
  # Horrible approximation: e^A ~= I + A
  # Then correct it by computing (e^(A/2^k))^(2^k)
  tiny_approx = jnp.eye(matrix.shape[0]) + matrix / (2.0**iterations)

  def step(_, mat):
    return jnp.dot(mat, mat, precision=jax.lax.Precision.HIGHEST)

  result = jax.lax.fori_loop(0, iterations, step, tiny_approx)
  return result


def transition_rate_expm(matrix, target_diagonal=1e-3, renormalize_cols=True):
  """Slightly improved expm for transition rate matrices.

  A transition rate matrix will always have columns that sum to zero, and will
  have nonnegative entries everywhere except the diagonal. We can ensure some
  stability by controlling the magnitude of the diagonal elements and
  renormalizing during each squaring to reduce error.

  Args:
    matrix: The matrix to compute a matrix exponential for.
    target_diagonal: Maximum magnitude of the diagonal elements for which it is
      "safe" to approximate e(tA) as I + tA. Will automatically perform more
      iterations until this is small enough to be a good approximation.
    renormalize_cols: Whether to renormalize the columns of the result, with the
      assumption that the rate matrix summed to zero across the columns. This
      property should always hold, so renormalizing can prevent errors from
      exploding.

  Returns:
    Approximation of expm(matrix).
  """
  max_diag = jnp.max(-jnp.diag(matrix))
  # Each iteration halves the diagonal. How many do we need to get to at or
  # below the target diagonal?
  iterations_for_diagonal = jnp.ceil(
      jnp.log2(max_diag) - jnp.log2(target_diagonal))
  # Make sure we're also squaring enough so that every element has a chance of
  # transitioning to every other element, in theory.
  iterations_for_mixing = jnp.ceil(jnp.log2(matrix.shape[0]))
  iterations = jnp.maximum(iterations_for_diagonal,
                           iterations_for_mixing).astype(jnp.int32)

  # Locally linear approximation: e^A ~= I + A
  # First divide by 2^iterations so that this approximation is accurate.
  tiny_approx = jnp.eye(matrix.shape[0]) + matrix / (2.0**iterations)

  def step(i, mat):
    del i

    updated = jnp.dot(mat, mat, precision=jax.lax.Precision.HIGHEST)
    if renormalize_cols:
      updated = updated / jnp.sum(updated, axis=0, keepdims=True)
    return updated

  result = jax.lax.fori_loop(0, iterations, step, tiny_approx)
  return result


def partition_embeddings_hierarchically(embedding_matrix,
                                        special_tokens_at_front=0):
  """Partition embeddings into power-of-two-sized subsets by PCA.

  Repeatedly bisects the space by computing the first principal component of
  the embeddings, then sorting them along that component and splitting into
  two halves.

  Args:
    embedding_matrix: Matrix of shape [vocab_size, embedding_dim]. Vocab size
      MUST be a power of 2.
    special_tokens_at_front: How many special tokens there are at the beginning
      of the vocab. These will always be kept at the beginning. (Note that they
      do still count toward the size of the first partition, for computational
      convenience. If you want them to not be counted, strip them out before
      calling this function.)

  Returns:
    A permutation vector with sorted indices of the embedding matrix, such
    that the first half of the vector is the first hierarchical subset, the
    first half of that first half is the next level of the hierarchy, and more
    generally tokens that share common prefixes in a binary representation are
    more similar to each other.
  """
  return _partition_embeddings_hierarchically(
      embedding_matrix,
      np.arange(embedding_matrix.shape[0]),
      force_keep_at_front=special_tokens_at_front)


def _partition_embeddings_hierarchically(embedding_matrix,
                                         indices,
                                         force_keep_at_front=0):
  """Helper function for hierarchical partitioning."""

  length = embedding_matrix.shape[0]
  if length == 1:
    return indices

  relevant_embeddings = embedding_matrix[force_keep_at_front:]

  # Project onto principal component.
  projected = sklearn.decomposition.PCA(1).fit_transform(
      relevant_embeddings).squeeze(1)
  projected_ixs = np.argsort(projected) + force_keep_at_front

  split = length // 2
  if split <= force_keep_at_front:
    # More than half of this region is special tokens, just take all of them.
    first_partitioned = indices[:split]
    second_half = np.concatenate(
        [np.arange(split, force_keep_at_front), projected_ixs])
    second_partitioned = _partition_embeddings_hierarchically(
        embedding_matrix[second_half],
        indices[second_half],
        force_keep_at_front=force_keep_at_front - split)
  else:
    # Sort each half, keeping trakc of the special tokens in the first half.
    first_half = np.concatenate([
        np.arange(force_keep_at_front),
        projected_ixs[:split - force_keep_at_front]
    ])
    first_partitioned = _partition_embeddings_hierarchically(
        embedding_matrix[first_half],
        indices[first_half],
        force_keep_at_front=force_keep_at_front)
    second_half = projected_ixs[split - force_keep_at_front:]
    second_partitioned = _partition_embeddings_hierarchically(
        embedding_matrix[second_half], indices[second_half])

  assert first_partitioned.shape == second_partitioned.shape
  return np.concatenate([first_partitioned, second_partitioned])


@gin.configurable
def load_from_numpy(filename):
  """Gin helper to load files from numpy."""
  with tf.io.gfile.Open(filename, "rb") as fp:
    return np.load(fp)


def compute_relative_information_removal(transition_matrix,
                                         initial_distribution,
                                         use_perplexity=False):
  """Computes removal of (mutual) information after applying a transition matrix.

    I(x_t; x_0) = [ log p(x_0, x_t) - log p(x_0) - log p(x_t)]
                = H(x_0) + H(x_t) - H(x_0, x_t)
         result = 1 - I(x_t; x_0) / H(x_0)
                = 1 - (H(x_0) + H(x_t) - H(x_0, x_t)) / H(x_0)
                = (H(x_0, x_t) - H(x_t)) / H(x_0)

  Args:
    transition_matrix: float32 matrix such that transition_matrix[i, j] = p(x_t
      = i | x_0 = j)
    initial_distribution: float32 matrix reprezenting p(x_0)
    use_perplexity: Use conditional perplexity(ish) instead of MI. Assumes
      convergence to uniform.

  Returns:
    Normalized information removal, which should be zero for the identity
    matrix,
    and 1 for a transition matrix which does not depend on the initial state.
  """
  # Normalizations for stability
  log_transition = jnp.log(transition_matrix)
  log_transition = (
      log_transition -
      jax.scipy.special.logsumexp(log_transition, axis=0, keepdims=True))
  log_initial = jnp.log(initial_distribution)
  log_initial = (
      log_initial -
      jax.scipy.special.logsumexp(log_initial, axis=0, keepdims=True))
  log_joint = log_initial[None, :] + log_transition
  log_marginal_after = jax.scipy.special.logsumexp(log_joint, axis=1)

  joint_entropy = -jnp.sum(
      jnp.where(log_joint == -np.inf, 0.0,
                jnp.exp(log_joint) * log_joint))
  initial_entropy = -jnp.sum(
      jnp.where(log_initial == -np.inf, 0.0,
                jnp.exp(log_initial) * log_initial))
  marginal_after_entropy = -jnp.sum(
      jnp.where(log_marginal_after == -np.inf, 0.0,
                jnp.exp(log_marginal_after) * log_marginal_after))

  if use_perplexity:
    dim = initial_distribution.shape[0]
    conditional_perplexity = jnp.exp(joint_entropy - initial_entropy)
    return (conditional_perplexity - 1) / (dim - 1)
  else:
    information_removal = (joint_entropy -
                           marginal_after_entropy) / initial_entropy
    return information_removal


def compute_information_removal_samples_closed_form(builder_fn,
                                                    initial_distribution,
                                                    min_exponent=1e-4,
                                                    max_exponent=1e5,
                                                    interpolation_steps=256):
  """Compute mutual information by evaluating a closed form estimate.

  Chooses interpolation steps, then evaluates mutual information for each one.

  Args:
    builder_fn: Function that, given a float exponent parameter, returns a
      transition matrix T[i, j] = p(x_t = i | x_0 = j) representing a matrix
      exponetial with the given exponent.
    initial_distribution: Initial distribution of tokens.
    min_exponent: Smallest non-zero exponent to try.
    max_exponent: Largest exponent to try.
    interpolation_steps: How many interpolation steps to try.

  Returns:
    exponents: Array of exponents for which we computed relative mutual
      information removal.
    information_removals: Array of the information removal for each exponent.
  """
  query_exponents = jnp.geomspace(min_exponent, max_exponent,
                                  interpolation_steps)

  def step(exponent):
    return compute_relative_information_removal(
        builder_fn(exponent), initial_distribution)

  information_removals = jax.lax.map(step, query_exponents)
  return query_exponents, information_removals


@gin.configurable
def compute_information_removal_samples_by_squaring(rate_matrix,
                                                    initial_distribution,
                                                    min_exponent=1e-4,
                                                    max_exponent=1e5,
                                                    interpolation_steps=256,
                                                    use_perplexity=False):
  """Compute mutual information using repeated squaring.

  Reduces a bunch of repeated work by evaluating power-of-two exponents using
  repeated squaring, starting from a few different test offsets to fill the
  gaps between powers of two.

  Args:
    rate_matrix: Transition rate matrix of shape [vocab_size, vocab_size]
    initial_distribution: Initial distribution of tokens.
    min_exponent: Smallest non-zero exponent to try.
    max_exponent: Largest exponent to try.
    interpolation_steps: Minimum number of interpolation steps to try.
    use_perplexity: Use conditional perplexity(ish) instead of MI

  Returns:
    exponents: Array of exponents for which we computed relative mutual
      information removal.
    information_removals: Array of the information removal for each exponent.
  """
  # How many powers of two do we need to fill the range?
  powers_of_two = 1 + jnp.ceil(jnp.log2(max_exponent) -
                               jnp.log2(min_exponent)).astype(jnp.int32)
  # How many shifts should we evaluate between each power of two? For instance,
  # in addition to evaluating at 1, 2, 4, 8, 16, 32 we might also evaluate at
  # 3/2, 3, 6, 12, 24, 48. Increasing interpolation steps will increase this.
  shifts = jnp.ceil(interpolation_steps / powers_of_two).astype(jnp.int32)

  # Figure out the base exponents (1 and 3/2 in the above example, but there
  # may be more)
  base_exponents = jnp.exp2(
      jnp.log2(min_exponent) + jnp.linspace(0, 1, shifts, endpoint=False))

  def from_base(base_exponent):
    base_matrix = transition_rate_expm(base_exponent * rate_matrix)

    def step(mat, i):
      exponent = base_exponent * (2.0**i)
      info_removal = compute_relative_information_removal(
          mat, initial_distribution, use_perplexity=use_perplexity)
      new_mat = jnp.dot(mat, mat, precision=jax.lax.Precision.HIGHEST)
      new_mat = new_mat / jnp.sum(new_mat, axis=0, keepdims=True)
      return new_mat, (exponent, info_removal)

    _, (exponents, info_removals) = jax.lax.scan(
        step, init=base_matrix, xs=jnp.arange(powers_of_two))
    return exponents, info_removals

  exponents, info_removals = jax.lax.map(from_base, base_exponents)
  return exponents.reshape([-1]), info_removals.reshape([-1])


@gin.configurable
def build_mutual_information_schedule(schedule_steps,
                                      exponents,
                                      information_removals,
                                      allow_out_of_bounds=False,
                                      kind="linear"):  # "warn"
  """Compute a mutual-information-based schedule by interpolation.

  Args:
    schedule_steps: Desired number of steps in the schedule.
    exponents: Array of exponents for which we computed relative mutual
      information removal.
    information_removals: Array of the information removal for each exponent.
    allow_out_of_bounds: Whether to allow interpolation for mutual information
      values that are not encountered before `max_exponent`. If True, clips the
      schedule so that it ends at the mutual info for `max_exponent` instead of
      at the desired (near-one) amount of mutual information removal. If False,
      throws an error.
    kind: one of ['linear', 'cosine']. Used to determine the schedule used.

  Returns:
    schedule_info_removals: float32[schedule_steps] array giving the amount of
      relative information removal at each point in the schedule. Will linearly
      interpolate between 0 and 1, not including either endpoint, unless this
      goes out of bounds and `allow_out_of_bounds=True`, in which case it may
      linearly interpolate to some value smaller than 1. Note that this may
      not be exactly correct due to the interpolation, but it should be close.
    schedule_exponents: float32[schedule_steps] array with the exponents
      needed to obtain each level of information removal. Note that this array
      does NOT include zero or infinity at the beginning/end, which are needed
      to obtain zero or one information removal. The caller should take care of
      padding so that the schedule takes the appropriate number of steps, for
      instance by adding zero to the front and ensuring that the sequence is
      replaced by a mask at the last step.
  """
  exponents = np.array(exponents)
  information_removals = np.array(information_removals)
  # Sort by exponent.
  permutation = np.argsort(exponents)
  exponents = exponents[permutation]
  information_removals = information_removals[permutation]
  # Fix out-of-order information removals due to numerical error.
  cmax_info_removal = np.maximum.accumulate(information_removals)
  bad = information_removals <= np.concatenate([[0], cmax_info_removal[:-1]])
  exponents = exponents[~bad]
  information_removals = information_removals[~bad]
  # Add zero at the start.
  exponents = np.concatenate([[0], exponents])
  information_removals = np.concatenate([[0], information_removals])

  # Interpolate monotonically so that our exponents are non-decreasing
  interpolator = scipy.interpolate.PchipInterpolator(
      information_removals, exponents, extrapolate=False)

  if kind == "linear":
    schedule_info_removals = np.linspace(0, 1, schedule_steps + 2)[1:-1]

  elif kind == "cosine":
    s = 0.008

    def cosine_fn(step):
      return jnp.cos((step / schedule_steps + s) / (1 + s) * jnp.pi / 2)

    schedule_info_removals = 1 - cosine_fn(np.arange(schedule_steps))
  else:
    raise ValueError(f"kind {kind} is not supported.")

  if schedule_info_removals[-1] > information_removals[-1]:
    if allow_out_of_bounds:
      if allow_out_of_bounds == "warn":
        logging.warning(
            "build_mutual_information_schedule: Requested mutual "
            "information removal value %s for "
            "schedule was larger than largest observed value "
            "%s. Clipping schedule to this largest "
            "observed value; consider increasing extrapolation range.",
            schedule_info_removals[-1], information_removals[-1])
      schedule_info_removals = (
          np.linspace(0, information_removals[-1], schedule_steps + 1)[1:])
    else:
      raise ValueError(
          "Requested mutual information removal value "
          f"{schedule_info_removals[-1]} for schedule was larger than largest "
          f"observed value {information_removals[-1]}")

  schedule_exponents = interpolator(schedule_info_removals)
  return schedule_info_removals, schedule_exponents
