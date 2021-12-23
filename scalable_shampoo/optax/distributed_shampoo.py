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

# An implementation of distributed Shampoo optimizer from:
#
#  Scalable Second Order Optimization for Deep Learning
#  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
#  Preprint Paper: https://arxiv.org/abs/2002.09018
#
# This implementation moves computation of inverse pth root back to the
# accelerator (if higher precision is available).
#
# Authors: Rohan Anil (rohananil at google dot com)
#    &     Vineet Gupta (vineet at google dot com)
#

"""Distributed Shampoo Implementation."""

import enum
import functools
import itertools
from typing import Any, NamedTuple

import chex
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

# pylint:disable=no-value-for-parameter


class ParameterStats(NamedTuple):
  """State associated to each parameter of the model being trained."""
  diagonal_statistics: chex.Array  # Accumulator for diagonal preconditioner
  statistics: chex.Array  # Statistics
  preconditioners: chex.Array  # Preconditioners
  diagonal_momentum: chex.Array  # Momentum for the diagonal preconditioner
  momentum: chex.Array  # Momentum for the shampoo preconditioner


class ShampooState(NamedTuple):
  count: chex.Array
  stats: Any


class GraftingType(enum.IntEnum):
  SGD = 1
  ADAGRAD = 2
  RMSPROP = 3


def power_iteration(
    matrix,
    num_iters=100,
    error_tolerance=1e-6,
    precision=lax.Precision.HIGHEST):
  r"""Power iteration algorithm.

  The power iteration algorithm takes a symmetric PSD matrix `A`, and produces
  a scalar `\lambda` , which is the greatest (in absolute value) eigenvalue
  of `A`, and a vector v, which is the corresponding eigenvector of `A`.

  References:
    [Wikipedia, 2021](https://en.wikipedia.org/wiki/Power_iteration)

  Args:
    matrix: the symmetric PSD matrix.
    num_iters: Number of iterations.
    error_tolerance: Iterative exit condition.
    precision: precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise)
      b) lax.Precision.HIGH (increased precision, slower)
      c) lax.Precision.HIGHEST (best possible precision, slowest)

  Returns:
    eigen vector, eigen value
  """
  matrix_size = matrix.shape[-1]
  def _iter_condition(state):
    i, unused_v, unused_s, unused_s_v, run_step = state
    return jnp.logical_and(i < num_iters, run_step)

  def _iter_body(state):
    """One step of power iteration."""
    i, new_v, s, s_v, unused_run_step = state
    new_v = new_v / jnp.linalg.norm(new_v)

    s_v = jnp.einsum('ij,j->i', matrix, new_v, precision=precision)
    s_new = jnp.einsum('i,i->', new_v, s_v, precision=precision)
    return (i + 1, s_v, s_new, s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance))

  # Figure out how to use step as seed for random.
  v_0 = np.random.uniform(-1.0, 1.0, matrix_size).astype(matrix.dtype)

  init_state = tuple([0, v_0, jnp.zeros([], dtype=matrix.dtype), v_0, True])
  _, v_out, s_out, _, _ = lax.while_loop(
      _iter_condition, _iter_body, init_state)
  v_out = v_out / jnp.linalg.norm(v_out)
  return v_out, s_out


def matrix_inverse_pth_root(
    matrix,
    p,
    num_iters=100,
    ridge_epsilon=1e-6,
    error_tolerance=1e-6,
    precision=lax.Precision.HIGHEST):
  """Computes `matrix^(-1/p)`, where `p` is a positive integer.

  This function uses the Coupled newton iterations algorithm for
  the computation of a matrix's inverse pth root.


  References:
    [Functions of Matrices, Theory and Computation,
     Nicholas J Higham, Pg 184, Eq 7.18](
     https://epubs.siam.org/doi/book/10.1137/1.9780898717778)

  Args:
    matrix: the symmetric PSD matrix whose power it to be computed
    p: exponent, for p a positive integer.
    num_iters: Maximum number of iterations.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.
    error_tolerance: Error indicator, useful for early termination.
    precision: precision XLA related flag, the available options are:
      a) lax.Precision.DEFAULT (better step time, but not precise)
      b) lax.Precision.HIGH (increased precision, slower)
      c) lax.Precision.HIGHEST (best possible precision, slowest)

  Returns:
    matrix^(-1/p)
  """

  # We use float32 for the matrix inverse pth root.
  # Switch to f64 if you have hardware that supports it.
  matrix_size = matrix.shape[0]
  alpha = jnp.asarray(-1.0 / p, jnp.float32)
  identity = jnp.eye(matrix_size, dtype=jnp.float32)
  _, max_ev = power_iteration(
      matrix=matrix, num_iters=100,
      error_tolerance=1e-6, precision=precision)
  ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-16)

  def _unrolled_mat_pow_1(mat_m):
    """Computes mat_m^1."""
    return mat_m

  def _unrolled_mat_pow_2(mat_m):
    """Computes mat_m^2."""
    return jnp.matmul(mat_m, mat_m, precision=precision)

  def _unrolled_mat_pow_4(mat_m):
    """Computes mat_m^4."""
    mat_pow_2 = _unrolled_mat_pow_2(mat_m)
    return jnp.matmul(
        mat_pow_2, mat_pow_2, precision=precision)

  def _unrolled_mat_pow_8(mat_m):
    """Computes mat_m^4."""
    mat_pow_4 = _unrolled_mat_pow_4(mat_m)
    return jnp.matmul(
        mat_pow_4, mat_pow_4, precision=precision)

  def mat_power(mat_m, p):
    """Computes mat_m^p, for p == 1, 2, 4 or 8.

    Args:
      mat_m: a square matrix
      p: a positive integer

    Returns:
      mat_m^p
    """
    # We unrolled the loop for performance reasons.
    exponent = jnp.round(jnp.log2(p))
    return lax.switch(
        jnp.asarray(exponent, jnp.int32), [
            _unrolled_mat_pow_1,
            _unrolled_mat_pow_2,
            _unrolled_mat_pow_4,
            _unrolled_mat_pow_8,
        ], (mat_m))

  def _iter_condition(state):
    (i, unused_mat_m, unused_mat_h, unused_old_mat_h, error,
     run_step) = state
    error_above_threshold = jnp.logical_and(
        error > error_tolerance, run_step)
    return jnp.logical_and(i < num_iters, error_above_threshold)

  def _iter_body(state):
    (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = jnp.matmul(mat_power(mat_m_i, p), mat_m, precision=precision)
    new_mat_h = jnp.matmul(mat_h, mat_m_i, precision=precision)
    new_error = jnp.max(jnp.abs(new_mat_m - identity))
    # sometimes error increases after an iteration before decreasing and
    # converging. 1.2 factor is used to bound the maximal allowed increase.
    return (i + 1, new_mat_m, new_mat_h, mat_h, new_error,
            new_error < error * 1.2)

  if matrix_size == 1:
    resultant_mat_h = (matrix + ridge_epsilon)**alpha
    error = 0
  else:
    damped_matrix = matrix + ridge_epsilon * identity

    z = (1 + p) / (2 * jnp.linalg.norm(damped_matrix))
    new_mat_m_0 = damped_matrix * z
    new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
    iters, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(
        _iter_condition, _iter_body, init_state)
    error = jnp.max(jnp.abs(mat_m - identity))
    is_converged = jnp.asarray(convergence, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    resultant_mat_h = jnp.asarray(resultant_mat_h, matrix.dtype)
  return resultant_mat_h, error


def merge_small_dims(shape_to_merge, max_dim):
  """Merge small dimensions.

  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]

  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.

  Returns:
    Merged shape.
  """
  resulting_shape = []
  product = 1
  for d in shape_to_merge:
    if product * d <= max_dim:
      product *= d
    else:
      if product > 1:
        resulting_shape.append(product)
      product = d
  if product > 1:
    resulting_shape.append(product)
  return resulting_shape


def pad_matrix(mat, max_size):
  """Pad a matrix to a max_size.

  Args:
    mat: a matrix to pad.
    max_size: matrix size requested.

  Returns:
    Given M returns [[M, 0], [0, I]]
  """
  size = mat.shape[0]
  assert size <= max_size
  if size == max_size:
    return mat
  pad_size = max_size - size
  zs1 = jnp.zeros([size, pad_size], dtype=mat.dtype)
  zs2 = jnp.zeros([pad_size, size], dtype=mat.dtype)
  eye = jnp.eye(pad_size, dtype=mat.dtype)
  mat = jnp.concatenate([mat, zs1], 1)
  mat = jnp.concatenate([mat, jnp.concatenate([zs2, eye], 1)], 0)
  return mat


def efficient_cond(predicate, compute_fn, init_state, *args, **kwargs):
  """Avoids wasteful buffer allocation with XLA."""

  def _iter_body(unused_state):
    results = compute_fn(*args, **kwargs)
    return tuple([False] + list(results))

  def _iter_condition(state):
    return state[0]

  results = jax.lax.while_loop(_iter_condition, _iter_body,
                               tuple([predicate] + init_state))
  return tuple(results[1:])


class BlockPartitioner:
  """Partitions a tensor into smaller tensors."""

  def __init__(self, param, block_size):
    self._shape = param.shape
    self._splits = []
    split_sizes = []
    # We split params into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(param.shape):
      if 0 < block_size < d:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d - 1) // block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        split_sizes.append(sizes)
      else:
        split_sizes.append(np.array([d], dtype=np.int32))
    self._num_splits = len(split_sizes)
    self._preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])

  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return self._num_splits

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, indices) in self._splits:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
      tensors = tensors_local
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            jnp.concatenate(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, param, block_size, best_effort_shape_interpretation):
    self._original_shape = param.shape
    self._transformed_shape = param.shape
    if best_effort_shape_interpretation:
      self._transformed_shape = merge_small_dims(self._original_shape,
                                                 block_size)
    reshaped_param = jnp.reshape(param, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_param, block_size)

  def statistics_from_grad(self, grad):
    """Compute statistics from gradients.

    Args:
      grad: Gradient to compute statistics from.

    Returns:
      A list of gradient statistics for each partition.
    """
    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    stats = []
    for g in partitioned_grads:
      g_stats = []
      rank = len(g.shape)
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = jnp.tensordot(g, g, axes=(axes, axes))
        g_stats.append(stat)
      stats.extend(g_stats)
    return stats

  def shapes_for_preconditioners(self):
    """Returns shape from statistics."""
    return self._partitioner.shapes_for_preconditioners()

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    return 2 * len(self._transformed_shape)

  def preconditioned_grad(self, grad, preconditioners):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.
      preconditioners: A list of preconditioners to apply.

    Returns:
      A preconditioned gradient.
    """

    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, g in enumerate(partitioned_grads):
      preconditioners_for_grad = preconditioners[i * num_splits:(i + 1) *
                                                 num_splits]
      rank = len(g.shape)
      precond_g = g
      for j in range(rank):
        precond_g = jnp.tensordot(
            precond_g, preconditioners_for_grad[j], axes=[[0], [0]])
      preconditioned_partitioned_grads.append(precond_g)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return jnp.reshape(merged_grad, self._original_shape)


def distributed_shampoo(learning_rate,
                        block_size,
                        beta1=0.9,
                        beta2=0.999,
                        diagonal_epsilon=1e-10,
                        matrix_epsilon=1e-6,
                        weight_decay=0.0,
                        start_preconditioning_step=1,
                        preconditioning_compute_steps=1,
                        statistics_compute_steps=1,
                        best_effort_shape_interpretation=True,
                        graft_type=GraftingType.SGD,
                        nesterov=True,
                        exponent_override=0,
                        batch_axis_name=None,
                        inverse_failure_threshold=0.1,
                        moving_average_for_momentum=False,
                        skip_preconditioning_dim_size_gt=4096,
                        clip_by_scaled_gradient_norm=None,
                        precision=lax.Precision.HIGHEST):
  """Distributed Shampoo optimizer.

  Distributed Shampoo is a second-order preconditioned method (concretely, a
  variant of full-matrix Adagrad), that provides significant convergence and
  wall-clock time improvements compared to conventional first-order methods,
  and that has been shown to scale to large state-of-the-art deep learning
  models.

  References:
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer

    Preprint: https://arxiv.org/abs/2002.09018

  Args:
    learning_rate: the step size used to update the parameters.
    block_size: Block size for large layers (if > 0). Preconditioning compute
      operation is cubic in the dimension of the tensor. Block size allows us to
      chunk the layers into sub-layers of maximal dimension dictated by this
      value. Use 128 as default (increase if you have compute budget).
    beta1: momentum parameter.
    beta2: second moment averaging parameter.
    diagonal_epsilon: epsilon for diagonal adagrad (only if layerwise grafting
      to AdaGrad is enabled).
    matrix_epsilon: epsilon to add to statistics before computing inverse pth
      root. If you are running in f32 precision for inverse pth root
      (recommended today) this can go upto 1e-6. If you have latest hardware
      with native f64 precision, set this upto 1e-12.
    weight_decay: Weight decay for regularization.
    start_preconditioning_step: When to start Shampoo update before which
      diagonal update is used. This is because we dont have enough information
      to do stable inverse.
    preconditioning_compute_steps: How often to compute preconditioner.
      Performance tuning params for controlling memory and compute requirements.
      Ideally set this and statistics_compute_steps params to 1.
    statistics_compute_steps: How often to compute statistics.
    best_effort_shape_interpretation: If there are some small dimensions,
      collapse them e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if
      block = 1024, [1, 2, 768, 1, 2048] --> [2, 768, 2048]
    graft_type: Grafting is a technique to fix the layerwise scale of Shampoo
      optimizer. This allows us to plugin the Shampoo optimizer into settings
      where SGD/AdaGrad is already well tuned. Available options are:
        GraftingType.SGD and GraftingType.ADAGRAD.
    nesterov: Nesterov momentum.
    exponent_override: Override the exponent used in matrix inverse.
    batch_axis_name: labeled axis over pmap for dataparallel training the
      optimizer used for.
    inverse_failure_threshold: numerics are hard and inverses fail sometimes; we
      determine that using this threshold.
    moving_average_for_momentum: Whether to use moving average for momentum
      instead of exponential moving average.
    skip_preconditioning_dim_size_gt: Skip if preconditioning dim size is
        greater than this value.
    clip_by_scaled_gradient_norm: Clip by scaled gradient norm (only useful
      when using RMSProp Grafting).
    precision: precision XLA related flag, the available options are: a)
      lax.Precision.DEFAULT (better step time, but not precise) b)
      lax.Precision.HIGH (increased precision, slower) c) lax.Precision.HIGHEST
      (best possible precision, slowest)

  Returns:
    a GradientTransformation.
  """

  def init_fn(params):
    """Initialise the optimiser's state."""

    def _init(param):
      preconditioner = Preconditioner(param, block_size,
                                      best_effort_shape_interpretation)
      statistics = []
      preconditioners = []
      if not _skip_preconditioning(param):
        shapes = preconditioner.shapes_for_preconditioners()
        statistics = [matrix_epsilon * jnp.eye(s[0]) for s in shapes]
        preconditioners = [jnp.eye(s[0]) for s in shapes]

      adagrad_statistics = []
      if graft_type == GraftingType.ADAGRAD or graft_type == GraftingType.RMSPROP:
        adagrad_statistics = jnp.zeros_like(param)
      return ParameterStats(adagrad_statistics, statistics, preconditioners,
                            jnp.zeros_like(param), jnp.zeros_like(param))

    return ShampooState(
        count=jnp.zeros([], jnp.int32), stats=jax.tree_map(_init, params))

  def _skip_preconditioning(param):
    return len(param.shape) < 1 or any(
        [s > skip_preconditioning_dim_size_gt for s in param.shape])

  def _compute_stats(grad, state, param, step):
    """Compute per-parameter statistics."""
    preconditioner = Preconditioner(param, block_size,
                                    best_effort_shape_interpretation)
    new_statistics = [[]] * len(state.statistics)
    w1 = beta2
    w2 = beta2 if beta2 == 1.0 else (1.0 - beta2)
    if not _skip_preconditioning(param):

      def compute_updated_statistics():
        new_stats = preconditioner.statistics_from_grad(grad)
        new_stats_accumulators = []
        for stat, stat_accumulator in zip(new_stats, state.statistics):
          new_stats_accumulators.append(w1 * stat_accumulator + w2 * stat)
        return new_stats_accumulators

      if statistics_compute_steps > 1:
        perform_step = step % statistics_compute_steps == 0
        init_state = state.statistics
        new_statistics = list(
            efficient_cond(perform_step, compute_updated_statistics,
                           init_state))
      else:
        new_statistics = compute_updated_statistics()
    return ParameterStats(state.diagonal_statistics, new_statistics,
                          state.preconditioners, state.diagonal_momentum,
                          state.momentum)

  def _compute_preconditioners(states, params, step):
    """Compute preconditioners for statistics."""
    statistics = []
    num_statistics_per_state = []
    original_shapes = []
    exponents = []
    max_size = 0
    prev_preconditioners = []
    for state, param in zip(states, params):
      num_statistics = len(state.statistics)
      num_statistics_per_state.append(num_statistics)
      original_shapes_for_state = []
      if num_statistics > 0:
        preconditioner = Preconditioner(param, block_size,
                                        best_effort_shape_interpretation)
        for statistic in state.statistics:
          exponents.append(preconditioner.exponent_for_preconditioner(
          ) if exponent_override == 0 else exponent_override)
          original_shapes_for_state.append(statistic.shape)
          max_size = max(max_size, statistic.shape[0])
        statistics.extend(state.statistics)
        prev_preconditioners.extend(state.preconditioners)
        original_shapes.extend(original_shapes_for_state)
    num_statistics = len(statistics)

    if not batch_axis_name:
      num_devices = jax.local_device_count()
    else:
      num_devices = lax.psum(1, batch_axis_name)

    # Pad statistics and exponents to next multiple of num_devices.
    packed_statistics = [
        pad_matrix(stat, max_size) for stat in statistics
    ]
    to_pad = -num_statistics % num_devices
    packed_statistics.extend([
        jnp.eye(max_size, dtype=packed_statistics[0].dtype)
        for _ in range(to_pad)
    ])
    exponents.extend([1 for _ in range(to_pad)])

    if not packed_statistics:
      return states
    # Batch statistics and exponents so that so that leading axis is
    # num_devices.
    def _batch(statistics, exponents, num_devices):
      assert len(statistics) == len(exponents)
      n = len(statistics)
      b = int(n / num_devices)
      batched_statistics = [
          jnp.stack(statistics[idx:idx + b]) for idx in range(0, n, b)
      ]
      batched_exponents = [
          jnp.stack(exponents[idx:idx + b]) for idx in range(0, n, b)
      ]
      return jnp.stack(batched_statistics), jnp.stack(batched_exponents)

    # Unbatch values across leading axis and return a list of elements.
    def _unbatch(batched_values):
      b1, b2 = batched_values.shape[0], batched_values.shape[1]
      results = []
      for v_array in jnp.split(batched_values, indices_or_sections=b1, axis=0):
        v_array = jnp.squeeze(v_array)
        # b2 = batches (number of preconditioner computation) per core.
        if b2 > 1:
          for v in jnp.split(v_array, indices_or_sections=b2, axis=0):
            results.append(jnp.squeeze(v))
        else:
          results.append(v_array)
      return results

    all_statistics, all_exponents = _batch(packed_statistics, exponents,
                                           num_devices)

    def _matrix_inverse_pth_root(xs, ps):
      mi_pth_root = functools.partial(
          matrix_inverse_pth_root,
          ridge_epsilon=matrix_epsilon,
          precision=precision)
      preconditioners, errors = jax.vmap(mi_pth_root)(xs, ps)
      return preconditioners, errors

    if not batch_axis_name:
      preconditioners, errors = jax.pmap(_matrix_inverse_pth_root)(
          all_statistics, all_exponents)
      preconditioners_flat = _unbatch(preconditioners)
      errors_flat = _unbatch(errors)
    else:

      def _internal_inverse_pth_root_all():
        preconditioners = jnp.array(all_statistics)
        current_replica = lax.axis_index(batch_axis_name)
        preconditioners, errors = _matrix_inverse_pth_root(
            all_statistics[current_replica], all_exponents[current_replica])
        preconditioners = jax.lax.all_gather(preconditioners, batch_axis_name)
        errors = jax.lax.all_gather(errors, batch_axis_name)
        preconditioners_flat = _unbatch(preconditioners)
        errors_flat = _unbatch(errors)
        return preconditioners_flat, errors_flat

      if preconditioning_compute_steps == 1:
        preconditioners_flat, errors_flat = _internal_inverse_pth_root_all()
      else:
        # Passing statistics instead of preconditioners as they are similarly
        # shaped tensors, as error we are passing is the threshold these will
        # be ignored.
        preconditioners_init = packed_statistics
        errors_init = ([inverse_failure_threshold] * len(packed_statistics))
        init_state = [preconditioners_init, errors_init]
        perform_step = step % preconditioning_compute_steps == 0
        preconditioners_flat, errors_flat = efficient_cond(
            perform_step, _internal_inverse_pth_root_all, init_state)

    def _skip(error):
      condition = jnp.logical_or(
          jnp.isnan(error), error >= inverse_failure_threshold)
      return condition.astype(error.dtype)

    def _select_preconditioner(error, new_p, old_p):
      return lax.cond(
          _skip(error), lambda _: old_p, lambda _: new_p, operand=None)

    new_preconditioners_flat = []
    for p, shape, prev_p, error in zip(preconditioners_flat, original_shapes,
                                       prev_preconditioners, errors_flat):
      new_preconditioners_flat.append(
          _select_preconditioner(error, p[:shape[0], :shape[1]], prev_p))

    assert len(states) == len(num_statistics_per_state)
    assert len(new_preconditioners_flat) == num_statistics

    # Add back empty preconditioners so we that we can set the optimizer state.
    preconditioners_for_states = []
    idx = 0
    for num_statistics, state in zip(num_statistics_per_state, states):
      if num_statistics == 0:
        preconditioners_for_states.append([])
      else:
        preconditioners_for_state = new_preconditioners_flat[idx:idx +
                                                             num_statistics]
        assert len(state.statistics) == len(preconditioners_for_state)
        preconditioners_for_states.append(preconditioners_for_state)
        idx += num_statistics
    new_states = []
    for state, new_preconditioners in zip(states, preconditioners_for_states):
      new_states.append(
          ParameterStats(state.diagonal_statistics, state.statistics,
                         new_preconditioners, state.diagonal_momentum,
                         state.momentum))

    return new_states

  def _transform_grad(grad, state, param, step):
    """Transform per-parameter gradients."""
    preconditioner = Preconditioner(param, block_size,
                                    best_effort_shape_interpretation)
    sgd_update = grad
    new_diagonal_statistics = state.diagonal_statistics
    if graft_type == GraftingType.ADAGRAD:
      new_diagonal_statistics = state.diagonal_statistics + jnp.square(grad)
      adagrad_update = grad / (
          jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)
      grafting_update = adagrad_update
    elif graft_type == GraftingType.RMSPROP:
      w1 = beta2
      w2 = beta2 if beta2 == 1.0 else (1.0 - beta2)
      new_diagonal_statistics = (
          w1 * state.diagonal_statistics + w2 * jnp.square(grad))
      rmsprop_update = grad / (
          jnp.sqrt(new_diagonal_statistics) + diagonal_epsilon)

      if clip_by_scaled_gradient_norm:
        scaled_grad_norm = jnp.linalg.norm(rmsprop_update) / (
            jnp.sqrt(float(rmsprop_update.size)))
        clipping_denom = jnp.maximum(
            1., scaled_grad_norm / clip_by_scaled_gradient_norm)
        rmsprop_update /= clipping_denom

      grafting_update = rmsprop_update
    else:
      grafting_update = sgd_update

    precond_grad = grad
    if not _skip_preconditioning(param):
      precond_grad = preconditioner.preconditioned_grad(precond_grad,
                                                        state.preconditioners)
    else:
      precond_grad = grafting_update

    grafting_update_norm = jnp.linalg.norm(grafting_update)
    precond_grad_norm = jnp.linalg.norm(precond_grad)

    multiplier = (grafting_update_norm / (precond_grad_norm + 1e-16))
    shampoo_update = precond_grad * multiplier

    shampoo_update_with_wd = shampoo_update
    grafting_update_with_wd = grafting_update
    if weight_decay != 0:
      shampoo_update_with_wd = shampoo_update + weight_decay * param
      grafting_update_with_wd = grafting_update + weight_decay * param

    w = (1.0 - beta1) if moving_average_for_momentum else 1.0
    shampoo_update_with_wd_momentum = (
        state.momentum * beta1 + w * shampoo_update_with_wd)
    grafting_update_with_wd_momentum = (
        state.diagonal_momentum * beta1 + w * grafting_update_with_wd)

    run_shampoo = (step >= start_preconditioning_step).astype(
        grafting_update_with_wd_momentum.dtype)

    momentum_update = (
        run_shampoo * shampoo_update_with_wd_momentum +
        (1.0 - run_shampoo) * grafting_update_with_wd_momentum)

    wd_update = (
        run_shampoo * shampoo_update_with_wd +
        (1.0 - run_shampoo) * grafting_update_with_wd)

    if nesterov:
      momentum_update = w * wd_update + beta1 * momentum_update

    lr = learning_rate
    if callable(learning_rate):
      lr = learning_rate(step)
    transformed_update = -1.0 * lr * momentum_update

    param_stats = ParameterStats(new_diagonal_statistics, state.statistics,
                                 state.preconditioners,
                                 grafting_update_with_wd_momentum,
                                 shampoo_update_with_wd_momentum)
    return transformed_update, param_stats

  def update_fn(grads, state, params):
    """Transform the input gradient and update all statistics.

    Args:
      grads: the gradient tensors for the parameters.
      state: a named tuple containing the state of the optimizer
      params: the parameters that should be updated.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    params_flat, treedef = jax.tree_flatten(params)
    stats_flat = treedef.flatten_up_to(state.stats)
    grads_flat = treedef.flatten_up_to(grads)

    new_stats_flat = jax.tree_multimap(
        lambda g, s, p: _compute_stats(g, s, p, state.count), grads_flat,
        stats_flat, params_flat)
    new_stats_flat = _compute_preconditioners(new_stats_flat, params_flat,
                                              state.count)

    outputs = jax.tree_multimap(
        lambda g, s, p: _transform_grad(g, s, p, state.count), grads_flat,
        new_stats_flat, params_flat)
    updates_flat, new_stats_flat = list(zip(*outputs)) if outputs else ((), ())

    updates = jax.tree_unflatten(treedef, updates_flat)
    new_stats = jax.tree_unflatten(treedef, new_stats_flat)

    new_state = ShampooState(
        count=state.count+1, stats=new_stats)
    return updates, new_state

  return optax.GradientTransformation(init_fn, update_fn)
