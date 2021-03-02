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

"""Distributed Shampoo Implementation."""
# An implementation of distributed Shampoo optimizer from:
#
#  Towards Practical Second Order Optimization for Deep Learning
#  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
#  Preprint Paper: https://openreview.net/forum?id=Sc8cY4Jpi3s
#
# This implementation moves computation of inverse pth root back to the
# accelerator (if higher precision is available). We will present the details
# in an ArXiv note soon.
#
# This implementation has been verified to work on ResNet-50 training to 75.9%
# accuracy which is the MLPerf benchmark at 32K batch size. At the time of
# writing this comment it achieves this in 1729 steps whereas the best known
# first order method trains in 2512 steps.
#
# Authors: Rohan Anil (rohananil at google dot com)
#    &     Vineet Gupta (vineet at google dot com)
#
import enum
import itertools

from absl import logging
from flax import struct
from flax.optim.base import OptimizerDef
from flax.optim.base import OptimizerState
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp

# Precision to use for matrix inverse pth root. Switch to f64 if you have
# hardware that supports it.
_INVERSE_PTH_ROOT_DATA_TYPE = jnp.float32

# Numerics are hard. Inverses fail sometimes. We determine that using this
# threshold.
_INVERSE_PTH_ROOT_FAILURE_THRESHOLD = 0.1

# Inverse pth root precision (XLA related) flag.
#
# Options are:
# a. lax.Precision.DEFAULT (Better step time, but not precise)
# b. lax.Precision.HIGH (Increased precision, slower)
# c. lax.Precision.HIGHEST (Best possible precision, slowest)
#
_INVERSE_PTH_ROOT_PRECISION = lax.Precision.HIGHEST


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. Moreover this
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned.
class LayerwiseGrafting(enum.Enum):
  SGD = 1
  ADAGRAD = 2


@struct.dataclass
class _ShampooHyperParams:
  """Shampoo hyperparameters."""

  learning_rate: float
  # Momentum (in Heavy-Ball or Nesterov, if nesterov is True).
  beta1: onp.ndarray
  # Parameter for exponential moving average of Shampoo second moment statistics
  # if set == 1.0, then sums statistics instead of moving average.
  beta2: onp.ndarray
  # Only set if using Layerwise grafting mode to adagrad. This is the epsilon
  # for adagrad update.
  diagonal_eps: float

  # Epsilon to add to statistics before computing inverse pth root. If you are
  # running in f32 precision for inverse pth root (recommended today)
  # this can go upto 1e-6. If you have latest hardware with native f64 precision
  # set this upto 1e-12.
  matrix_eps: float

  # Weight decay parameter for regularization.
  weight_decay: float

  # When to start Shampoo update before which diagonal update is used. This is
  # because we do not have enough information to compute a stable inverse.
  start_preconditioning_step: int

  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner. Ideally set both params to 1.
  preconditioning_compute_steps: int
  # How often to compute statistics.
  statistics_compute_steps: int

  # Whether to even try preconditioning large layers.
  no_preconditioning_for_layers_with_dim_gt: int

  # Block size for large layers (if > 0).
  block_size: int

  # if there are some small dimensions, collapse them:
  # e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if block = 1024
  # [1, 2, 768, 1, 2048] --> [2, 768, 2048]
  best_effort_shape_interpretation: bool

  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int

  # Nesterov momentum
  nesterov: bool
  # Exponent override (if > 0):
  exponent_override: int
  # Batch axis name (for data parallel code).
  batch_axis_name: str


class BlockPartitioner:
  """Partitions a tensor into smaller tensors."""

  def __init__(self, param, hps):
    self._shape = param.shape
    self._splits = []
    self._split_sizes = []
    # We split params into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(param.shape):
      if hps.block_size > 0 and d > hps.block_size:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d-1) // hps.block_size
        indices = (onp.arange(nsplit, dtype=onp.int32) + 1) * hps.block_size
        sizes = onp.ones(nsplit + 1, dtype=onp.int32) * hps.block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        self._split_sizes.append(sizes)
      else:
        self._split_sizes.append(onp.array([d], dtype=onp.int32))
    self._preconditioner_shapes = []
    for t in itertools.product(*self._split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])

  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return len(self._split_sizes)

  def partition(self, tensor):
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    for (i, indices) in self._splits:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(jnp.split(t, indices, axis=i))
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

  def __init__(self, param, hps):
    self._hps = hps
    self._original_shape = param.shape
    self._transformed_shape = param.shape
    if hps.best_effort_shape_interpretation:
      self._transformed_shape = []
      # if there are some small dimensions, collapse them:
      # e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if block = 1024
      # [1, 2, 768, 1, 2048] --> [2, 768, 2048]
      product = 1
      for d in self._original_shape:
        if product * d <= hps.block_size:
          product *= d
        else:
          if product > 1: self._transformed_shape.append(product)
          product = d
      if product > 1:
        self._transformed_shape.append(product)
      print('Shampoo: changing shape from: to:', self._original_shape,
            self._transformed_shape)

    reshaped_param = jnp.reshape(param, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_param, hps)

  def statistics_from_grad(self, grad):
    """Compute statistics from gradients."""
    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    stats = []
    for grad in partitioned_grads:
      grad_stats = []
      rank = len(grad.shape)
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = jnp.tensordot(grad, grad, axes=(axes, axes))
        grad_stats.append(stat)
      stats.extend(grad_stats)
    return stats

  def shapes_for_preconditioners(self):
    """Compute shape from statistics."""
    return self._partitioner.shapes_for_preconditioners()

  def exponent_for_preconditioner(self):
    return 2 * len(self._transformed_shape)

  def preconditioned_grad(self, grad, preconditioners):
    """Precondition the gradient with the given preconditioners."""

    reshaped_grad = jnp.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, grad in enumerate(partitioned_grads):
      preconditioners_for_grad = preconditioners[i * num_splits:(i + 1) *
                                                 num_splits]
      rank = len(grad.shape)
      precond_grad = grad
      for j in range(rank):
        precond_grad = jnp.tensordot(
            precond_grad, preconditioners_for_grad[j], axes=[[0], [0]])
      preconditioned_partitioned_grads.append(precond_grad)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return jnp.reshape(merged_grad, self._original_shape)


@struct.dataclass
class _ShampooDefaultParamState:
  """Shampoo default parameter state."""

  # Accumulator for diagonal preconditioner
  diagonal_statistics: onp.ndarray
  # Statistics
  statistics: onp.ndarray
  # Preconditioners
  preconditioners: onp.ndarray
  # Momentum for the diagonal preconditioner
  diagonal_momentum: onp.ndarray
  # Momentum for the shampoo preconditioner
  momentum: onp.ndarray


def power_iter(mat_g, error_tolerance=1e-6, num_iters=100):
  """Power iteration.

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
  mat_g_size = mat_g.shape[-1]
  def _iter_condition(state):
    i, unused_v, unused_s, unused_s_v, run_step = state
    return jnp.logical_and(i < num_iters, run_step)

  def _iter_body(state):
    """One step of power iteration."""
    i, new_v, s, s_v, unused_run_step = state
    new_v = new_v / jnp.linalg.norm(new_v)

    s_v = jnp.einsum(
        'ij,j->i', mat_g, new_v, precision=_INVERSE_PTH_ROOT_PRECISION)
    s_new = jnp.einsum(
        'i,i->', new_v, s_v, precision=_INVERSE_PTH_ROOT_PRECISION)
    return (i + 1, s_v, s_new, s_v,
            jnp.greater(jnp.abs(s_new - s), error_tolerance))

  # Figure out how to use step as seed for random.
  v_0 = onp.random.uniform(-1.0, 1.0, mat_g_size).astype(mat_g.dtype)

  init_state = tuple([0, v_0, jnp.zeros([], dtype=mat_g.dtype), v_0, True])
  num_iters, v_out, s_out, _, _ = lax.while_loop(
      _iter_condition, _iter_body, init_state)
  v_out = v_out / jnp.linalg.norm(v_out)
  return v_out, s_out, num_iters


def matrix_inverse_pth_root(mat_g,
                            p,
                            iter_count=100,
                            error_tolerance=1e-6,
                            ridge_epsilon=1e-6):
  """Computes mat_g^(-1/p), where p is a positive integer.

  Coupled newton iterations for matrix inverse pth root.

  Args:
    mat_g: the symmetric PSD matrix whose power it to be computed
    p: exponent, for p a positive integer.
    iter_count: Maximum number of iterations.
    error_tolerance: Error indicator, useful for early termination.
    ridge_epsilon: Ridge epsilon added to make the matrix positive definite.

  Returns:
    mat_g^(-1/p)
  """
  mat_g_size = mat_g.shape[0]
  alpha = jnp.asarray(-1.0 / p, _INVERSE_PTH_ROOT_DATA_TYPE)
  identity = jnp.eye(mat_g_size, dtype=_INVERSE_PTH_ROOT_DATA_TYPE)
  _, max_ev, _ = power_iter(mat_g, mat_g.shape[0], 100)
  ridge_epsilon = ridge_epsilon * jnp.maximum(max_ev, 1e-16)

  def _unrolled_mat_pow_1(mat_m):
    """Computes mat_m^1."""
    return mat_m

  def _unrolled_mat_pow_2(mat_m):
    """Computes mat_m^2."""
    return jnp.matmul(mat_m, mat_m, precision=_INVERSE_PTH_ROOT_PRECISION)

  def _unrolled_mat_pow_4(mat_m):
    """Computes mat_m^4."""
    mat_pow_2 = _unrolled_mat_pow_2(mat_m)
    return jnp.matmul(
        mat_pow_2, mat_pow_2, precision=_INVERSE_PTH_ROOT_PRECISION)

  def _unrolled_mat_pow_8(mat_m):
    """Computes mat_m^4."""
    mat_pow_4 = _unrolled_mat_pow_4(mat_m)
    return jnp.matmul(
        mat_pow_4, mat_pow_4, precision=_INVERSE_PTH_ROOT_PRECISION)

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
    return jnp.logical_and(i < iter_count,
                           jnp.logical_or(error > error_tolerance, run_step))

  def _iter_body(state):
    (i, mat_m, mat_h, unused_old_mat_h, error, unused_run_step) = state
    mat_m_i = (1 - alpha) * identity + alpha * mat_m
    new_mat_m = jnp.matmul(
        mat_power(mat_m_i, p), mat_m, precision=_INVERSE_PTH_ROOT_PRECISION)
    new_mat_h = jnp.matmul(
        mat_h, mat_m_i, precision=_INVERSE_PTH_ROOT_PRECISION)
    new_error = jnp.max(jnp.abs(new_mat_m - identity))
    # sometimes error increases after an iteration before decreasing and
    # converging. 1.2 factor is used to bound the maximal allowed increase.
    return (i + 1, new_mat_m, new_mat_h, mat_h, new_error,
            new_error < error * 1.2)

  if mat_g_size == 1:
    resultant_mat_h = (mat_g + ridge_epsilon)**alpha
    error = 0
  else:
    damped_mat_g = mat_g + ridge_epsilon * identity

    z = (1 + p) / (2 * jnp.linalg.norm(damped_mat_g))
    new_mat_m_0 = damped_mat_g * z
    new_error = jnp.max(jnp.abs(new_mat_m_0 - identity))
    new_mat_h_0 = identity * jnp.power(z, 1.0 / p)
    init_state = tuple(
        [0, new_mat_m_0, new_mat_h_0, new_mat_h_0, new_error, True])
    _, mat_m, mat_h, old_mat_h, error, convergence = lax.while_loop(
        _iter_condition, _iter_body, init_state)
    error = jnp.max(jnp.abs(mat_m - identity))
    is_converged = jnp.asarray(convergence, old_mat_h.dtype)
    resultant_mat_h = is_converged * mat_h + (1 - is_converged) * old_mat_h
    resultant_mat_h = jnp.asarray(resultant_mat_h, mat_g.dtype)
  return resultant_mat_h, error


class Shampoo(OptimizerDef):
  """Shampoo optimizer.

    Towards Practical Second Order Optimization for Deep Learning
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer, Preprint

    Paper: https://openreview.net/forum?id=Sc8cY4Jpi3s
  """

  def __init__(self,
               learning_rate = None,
               beta1=0.9,
               beta2=0.999,
               diagonal_epsilon=1e-10,
               matrix_epsilon=1e-6,
               weight_decay=0.0,
               start_preconditioning_step=1,
               preconditioning_compute_steps=1,
               statistics_compute_steps=1,
               no_preconditioning_for_layers_with_dim_gt=8192,
               block_size=128,
               best_effort_shape_interpretation=True,
               graft_type=LayerwiseGrafting.SGD,
               nesterov=True,
               exponent_override=0,
               batch_axis_name=None):
    """Constructor for the Shampoo optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
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
        Performance tuning params for controlling memory and compute
        requirements. Ideally set both params to 1.
      statistics_compute_steps: How often to compute statistics.
      no_preconditioning_for_layers_with_dim_gt: Run diagonal method for if any
        of the dim is larger than this value.
      block_size: Block size for large layers (if > 0). Preconditioning compute
        operation is cubic in the dimension of the tensor. Block size allows us
        to chunk the layers into sub-layers of maximal dimension dictated by
        this value. Use 128 as default (increase if you have compute budget).
      best_effort_shape_interpretation:
      graft_type: Options are: LayerwiseGrafting.SGD, LayerwiseGrafting.ADAGRAD
      nesterov: Nesterov momentum.
      exponent_override: Override the exponent used in matrix inverse.
      batch_axis_name: labeled axis over pmap for dataparallel training the
        optimizer used for.
    """
    hps = _ShampooHyperParams(
        learning_rate,
        beta1,
        beta2,
        diagonal_epsilon,
        matrix_epsilon,
        weight_decay,
        start_preconditioning_step,
        preconditioning_compute_steps,
        statistics_compute_steps,
        no_preconditioning_for_layers_with_dim_gt,
        block_size,
        best_effort_shape_interpretation,
        graft_type=graft_type,
        nesterov=nesterov,
        exponent_override=exponent_override,
        batch_axis_name=batch_axis_name)
    print(hps)
    super().__init__(hps)

  def init_param_state(self, param):
    """Initialize parameter state."""
    rank = len(param.shape)
    preconditioner = Preconditioner(param, self.hyper_params)
    statistics = []
    preconditioners = []
    if rank >= 1:
      shapes = preconditioner.shapes_for_preconditioners()
      statistics = [
          self.hyper_params.matrix_eps * jnp.eye(s[0]) for s in shapes
      ]
      preconditioners = [jnp.eye(s[0]) for s in shapes]

    return _ShampooDefaultParamState(
        jnp.zeros_like(param),
        statistics,
        preconditioners,
        jnp.zeros_like(param),
        jnp.zeros_like(param))

  def _skip_preconditioning(self, param, hps):
    return len(param.shape) < 1

  def fast_cond(self, predicate, compute_fn, init_state, *args, **kwargs):
    """Avoids wasteful buffer allocation with XLA."""

    def _iter_body(unused_state):
      results = compute_fn(*args, **kwargs)
      return tuple([False] + list(results))

    def _iter_condition(state):
      return state[0]

    results = lax.while_loop(_iter_condition, _iter_body,
                             tuple([predicate] + init_state))
    return tuple(results[1:])

  def compute_shampoo_statistics(self, step, hps, param, state, grad):
    """Compute statistics."""
    logging.info(param.shape)
    preconditioner = Preconditioner(param, hps)
    assert hps.learning_rate is not None, 'no learning rate provided.'
    new_statistics = [[]] * len(state.statistics)
    w1 = hps.beta2
    w2 = hps.beta2 if hps.beta2 == 1.0 else (1.0 - hps.beta2)
    if not self._skip_preconditioning(param, hps):
      def compute_updated_statistics():
        new_stats = preconditioner.statistics_from_grad(grad)
        new_stats_accumulators = []
        for stat, stat_accumulator in zip(new_stats, state.statistics):
          new_stats_accumulators.append(w1 * stat_accumulator + w2 * stat)
        return new_stats_accumulators

      if hps.statistics_compute_steps > 1:
        perform_step = step % hps.statistics_compute_steps == 0
        init_state = state.statistics
        new_statistics = list(
            self.fast_cond(perform_step, compute_updated_statistics,
                           init_state))
      else:
        new_statistics = compute_updated_statistics()
    new_state = _ShampooDefaultParamState(state.diagonal_statistics,
                                          new_statistics, state.preconditioners,
                                          state.diagonal_momentum,
                                          state.momentum)
    return new_state

  def compute_preconditioners_from_statistics(self, states, hps, step):
    """Compute preconditioners for statistics."""
    statistics = []
    num_statistics_per_state = []
    original_shapes = []
    exponents = []
    max_size = 0
    prev_preconditioners = []
    for state in states:
      num_statistics = len(state.statistics)
      num_statistics_per_state.append(num_statistics)
      original_shapes_for_state = []
      if num_statistics > 0:
        for statistic in state.statistics:
          exponents.append(2 * num_statistics if hps.exponent_override ==
                           0 else hps.exponent_override)
          original_shapes_for_state.append(statistic.shape)
          max_size = max(max_size, statistic.shape[0])
        statistics.extend(state.statistics)
        prev_preconditioners.extend(state.preconditioners)
        original_shapes.extend(original_shapes_for_state)
    num_statistics = len(statistics)

    def pack(mat, max_size):
      """Pack a matrix to a max_size for inverse on TPUs with static shapes.

      Args:
        mat: Matrix for computing inverse pth root.
        max_size: Matrix size to pack to.

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

    if not hps.batch_axis_name:
      num_devices = jax.local_device_count()
    else:
      num_devices = lax.psum(1, hps.batch_axis_name)

    # Pad statistics and exponents to next multiple of num_devices.
    packed_statistics = [pack(stat, max_size) for stat in statistics]
    to_pad = -num_statistics % num_devices
    packed_statistics.extend([
        jnp.eye(max_size, dtype=packed_statistics[0].dtype)
        for _ in range(to_pad)
    ])
    exponents.extend([1 for _ in range(to_pad)])

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
      for v_array in jnp.split(batched_values, b1, 0):
        for v in jnp.split(jnp.squeeze(v_array), b2, 0):
          results.append(jnp.squeeze(v))
      return results

    all_statistics, all_exponents = _batch(packed_statistics, exponents,
                                           num_devices)

    def _matrix_inverse_pth_root(xs, ps):
      mi_pth_root = lambda x, y: matrix_inverse_pth_root(  # pylint: disable=g-long-lambda
          x, y, ridge_epsilon=hps.matrix_eps)
      preconditioners, errors = jax.vmap(mi_pth_root)(xs, ps)
      return preconditioners, errors

    if not hps.batch_axis_name:
      preconditioners, errors = jax.pmap(_matrix_inverse_pth_root)(
          all_statistics, all_exponents)
      preconditioners_flat = _unbatch(preconditioners)
      errors_flat = _unbatch(errors)
    else:

      def _internal_inverse_pth_root_all():
        preconditioners = jnp.array(all_statistics)
        current_replica = lax.axis_index(hps.batch_axis_name)
        preconditioners, errors = _matrix_inverse_pth_root(
            all_statistics[current_replica], all_exponents[current_replica])
        preconditioners = jax.lax.all_gather(preconditioners,
                                             hps.batch_axis_name)
        errors = jax.lax.all_gather(errors, hps.batch_axis_name)
        preconditioners_flat = _unbatch(preconditioners)
        errors_flat = _unbatch(errors)
        return preconditioners_flat, errors_flat

      if hps.preconditioning_compute_steps == 1:
        preconditioners_flat, errors_flat = _internal_inverse_pth_root_all()
      else:
        # Passing statistics instead of preconditioners as they are similarly
        # shaped tensors, as error we are passing is the threshold these will
        # be ignored.
        preconditioners_init = packed_statistics
        errors_init = ([_INVERSE_PTH_ROOT_FAILURE_THRESHOLD] *
                       len(packed_statistics))
        init_state = [preconditioners_init, errors_init]
        perform_step = step % hps.preconditioning_compute_steps == 0
        preconditioners_flat, errors_flat = self.fast_cond(
            perform_step, _internal_inverse_pth_root_all, init_state)

    def _skip(error):
      return jnp.logical_or(
          jnp.isnan(error),
          error >= _INVERSE_PTH_ROOT_FAILURE_THRESHOLD).astype(error.dtype)

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
          _ShampooDefaultParamState(state.diagonal_statistics, state.statistics,
                                    new_preconditioners,
                                    state.diagonal_momentum, state.momentum))

    return new_states

  def apply_per_param_gradient(self, step, hps, param, state, grad):
    """Apply per-parameter gradients."""
    logging.info(param.shape)
    preconditioner = Preconditioner(param, hps)
    assert hps.learning_rate is not None, 'no learning rate provided.'
    new_diagonal_statistics = state.diagonal_statistics + jnp.square(grad)
    adagrad_update = grad / (
        jnp.sqrt(new_diagonal_statistics) + hps.diagonal_eps)
    sgd_update = grad
    if hps.graft_type == LayerwiseGrafting.ADAGRAD:
      grafting_update = adagrad_update
    else:
      grafting_update = sgd_update

    precond_grad = grad
    if not self._skip_preconditioning(param, hps):
      precond_grad = preconditioner.preconditioned_grad(precond_grad,
                                                        state.preconditioners)
    else:
      precond_grad = grafting_update

    grafting_update_norm = jnp.linalg.norm(grafting_update)
    precond_grad_norm = jnp.linalg.norm(precond_grad)
    shampoo_update = precond_grad * (
        grafting_update_norm / (precond_grad_norm + 1e-16))

    shampoo_update_with_wd = shampoo_update
    grafting_update_with_wd = grafting_update
    if hps.weight_decay != 0:
      shampoo_update_with_wd = shampoo_update + hps.weight_decay * param
      grafting_update_with_wd = grafting_update + hps.weight_decay * param

    shampoo_update_with_wd_momentum = (
        state.momentum * hps.beta1 + shampoo_update_with_wd)
    grafting_update_with_wd_momentum = (
        state.diagonal_momentum * hps.beta1 + grafting_update_with_wd)

    run_shampoo = (step >= hps.start_preconditioning_step).astype(
        grafting_update_with_wd_momentum.dtype)

    momentum_update = (
        run_shampoo * shampoo_update_with_wd_momentum +
        (1.0 - run_shampoo) * grafting_update_with_wd_momentum)

    wd_update = (
        run_shampoo * shampoo_update_with_wd +
        (1.0 - run_shampoo) * grafting_update_with_wd)

    if hps.nesterov:
      momentum_update = wd_update + hps.beta1 * momentum_update

    new_param = param - hps.learning_rate * momentum_update
    new_state = _ShampooDefaultParamState(new_diagonal_statistics,
                                          state.statistics,
                                          state.preconditioners,
                                          grafting_update_with_wd_momentum,
                                          shampoo_update_with_wd_momentum)
    return new_param, new_state

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies a gradient for a set of parameters.

    Args:
      hyper_params: a named tuple of hyper parameters.
      params: the parameters that should be updated.
      state: a named tuple containing the state of the optimizer
      grads: the gradient tensors for the parameters.

    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    step = state.step
    params_flat, treedef = jax.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)

    new_states_flat = [
        self.compute_shampoo_statistics(step, hyper_params, param, state, grad)
        for param, state, grad in zip(params_flat, states_flat, grads_flat)
    ]

    new_states_flat = self.compute_preconditioners_from_statistics(
        new_states_flat, hyper_params, step)

    out = [
        self.apply_per_param_gradient(step, hyper_params, param, state, grad)
        for param, state, grad in zip(params_flat, new_states_flat, grads_flat)
    ]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = OptimizerState(step + 1, new_param_states)
    return new_params, new_state
