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

"""Distributed Shampoo Implementation."""
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

from typing import Any

from flax import struct
from flax.core import unfreeze
from flax.optim.base import OptimizerDef
from flax.optim.base import OptimizerState
import jax
from jax import lax
import jax.experimental.pjit as pjit
import numpy as onp
import optax

from scalable_shampoo.optax import distributed_shampoo as optax_distributed_shampoo

GraftingType = optax_distributed_shampoo.GraftingType


@struct.dataclass
class _DistributedShampooHyperParams:
  """DistributedShampoo hyperparameters."""
  # Learning rate for distributed shampoo.
  learning_rate: float

  # Block size for large layers (if > 0).
  block_size: int

  # Momentum (in Heavy-Ball or Nesterov, if nesterov is True).
  beta1: onp.ndarray = 0.9
  # Parameter for exponential moving average of Shampoo second moment statistics
  # if set == 1.0, then sums statistics instead of moving average.
  beta2: onp.ndarray = 0.999
  # Only set if using Layerwise grafting mode to adagrad. This is the epsilon
  # for adagrad update.
  diagonal_epsilon: float = 1e-10

  # Epsilon to add to statistics before computing inverse pth root. If you are
  # running in f32 precision for inverse pth root (recommended today)
  # this can go upto 1e-6. If you have latest hardware with native f64 precision
  # set this upto 1e-12.
  matrix_epsilon: float = 1e-6

  # Weight decay parameter for regularization.
  weight_decay: float = 0.0

  # When to start Shampoo update before which diagonal update is used. This is
  # because we do not have enough information to compute a stable inverse.
  start_preconditioning_step: int = 5

  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner. Ideally set both params to 1.
  preconditioning_compute_steps: int = 1
  # How often to compute statistics.
  statistics_compute_steps: int = 1

  # if there are some small dimensions, collapse them:
  # e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if block = 1024
  # [1, 2, 768, 1, 2048] --> [2, 768, 2048]
  best_effort_shape_interpretation: bool = True

  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int = 1

  # Nesterov momentum
  nesterov: bool = True

  # Exponent override (if > 0):
  exponent_override: int = 0

  # Batch axis name (for data parallel code).
  batch_axis_name: str = None

  # PartitionSpec which describes how to shard the statistics and preconditioner
  # tensors.
  statistics_partition_spec: pjit.PartitionSpec = None
  preconditioner_partition_spec: pjit.PartitionSpec = None

  # Number of devices in pjit.
  num_devices_for_pjit: int = None

  # Shard optimizer states.
  shard_optimizer_states: bool = False

  # Quantization mode for memory usage reduction.
  best_effort_memory_usage_reduction: bool = False

  # Default value for ignoring preconditioners that are incorrect.
  inverse_failure_threshold: float = 0.1

  # HB vs EMA
  moving_average_for_momentum: bool = False

  # Avoids preconditioning large layers to reduce overall memory usage if any
  # of the dimensions are greater than this value.
  skip_preconditioning_dim_size_gt: int = 4096

  # Clipping based on RMS
  clip_by_scaled_gradient_norm: float = None

  # Precision used for training
  precision: Any = lax.Precision.HIGHEST


class DistributedShampoo(OptimizerDef):
  """Distributed Shampoo optimizer.

    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer

    Preprint: https://arxiv.org/abs/2002.09018
  """

  def __init__(
      self,
      learning_rate,
      block_size,
      beta1=0.9,
      beta2=0.999,
      diagonal_epsilon=1e-10,
      matrix_epsilon=1e-6,
      weight_decay=0.0,
      start_preconditioning_step=5,
      preconditioning_compute_steps=1,
      statistics_compute_steps=1,
      best_effort_shape_interpretation=True,
      graft_type=GraftingType.SGD,
      nesterov=True,
      exponent_override=0,
      # Pass pmap 'batch axis name' in pmap mode.
      batch_axis_name=None,
      ### Only set following 3 params in pjit/spmd mode.
      ### WARNING: Experimental
      statistics_partition_spec=None,
      preconditioner_partition_spec=None,
      num_devices_for_pjit=None,
      shard_optimizer_states=False,
      ###
      ### Experimental memory reduction mode
      best_effort_memory_usage_reduction=False,
      ###
      inverse_failure_threshold=0.1,
      moving_average_for_momentum=False,
      skip_preconditioning_dim_size_gt=4096,
      clip_by_scaled_gradient_norm=None,
      precision=lax.Precision.HIGHEST):
    r"""Constructor of DistributedShampoo.

    Args:
      learning_rate: the step size used to update the parameters.
      block_size: Block size for large layers (if > 0). Preconditioning compute
        operation is cubic in the dimension of the tensor. Block size allows us
        to chunk the layers into sub-layers of maximal dimension dictated by
        this value. Use 128 as default (increase if you have compute budget).
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
        requirements. Ideally set this and statistics_compute_steps params to 1.
      statistics_compute_steps: How often to compute statistics.
      best_effort_shape_interpretation: If there are some small dimensions,
        collapse them e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if
        block = 1024, [1, 2, 768, 1, 2048] --> [2, 768, 2048]
      graft_type: Grafting is a technique to fix the layerwise scale of Shampoo
        optimizer. This allows us to plugin the Shampoo optimizer into settings
        where SGD/AdaGrad is already well tuned.
      nesterov: Nesterov momentum.
      exponent_override: Override the exponent used in matrix inverse.
      batch_axis_name: labeled axis over pmap for data-parallel training the
        optimizer used for.
      statistics_partition_spec: PartitionSpec to be used in sharded mode.
      preconditioner_partition_spec: PartitionSpec to be used in sharded mode.
      num_devices_for_pjit: Number of devices to parallelize over when using
        pjit.
      shard_optimizer_states: Shard optimizer states to save memory in model
        parallel training.
      best_effort_memory_usage_reduction: Best effort memory usage reduction.
        diagonal_statistics -> jnp.bfloat16 momentum buffers (2x) -> jnp.int8
        statistics, preconditioners -> jnp.int16 + diagonals
      inverse_failure_threshold: numerics are hard and inverses fail sometimes;
        we determine that using this threshold.
      moving_average_for_momentum: Whether to use moving average for momentum
        instead of exponential moving average.
      skip_preconditioning_dim_size_gt: Skip if preconditioning dim size is
        greater than this value.
      clip_by_scaled_gradient_norm: Clip by scaled gradient norm (only useful
        when using RMSProp Grafting).
      precision: precision XLA related flag, the available options are: a)
        lax.Precision.DEFAULT (better step time, but not precise) b)
        lax.Precision.HIGHg4 fi (increased precision, slower) c)
        lax.Precision.HIGHEST (best possible precision, slowest).
    """
    self._hps = _DistributedShampooHyperParams(
        learning_rate=learning_rate,
        block_size=block_size,
        beta1=beta1,
        beta2=beta2,
        diagonal_epsilon=diagonal_epsilon,
        matrix_epsilon=matrix_epsilon,
        weight_decay=weight_decay,
        start_preconditioning_step=start_preconditioning_step,
        preconditioning_compute_steps=preconditioning_compute_steps,
        statistics_compute_steps=statistics_compute_steps,
        best_effort_shape_interpretation=best_effort_shape_interpretation,
        graft_type=graft_type,
        nesterov=nesterov,
        exponent_override=exponent_override,
        batch_axis_name=batch_axis_name,
        statistics_partition_spec=statistics_partition_spec,
        preconditioner_partition_spec=preconditioner_partition_spec,
        num_devices_for_pjit=num_devices_for_pjit,
        shard_optimizer_states=shard_optimizer_states,
        best_effort_memory_usage_reduction=best_effort_memory_usage_reduction,
        inverse_failure_threshold=inverse_failure_threshold,
        moving_average_for_momentum=moving_average_for_momentum,
        skip_preconditioning_dim_size_gt=skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=clip_by_scaled_gradient_norm,
        precision=precision)
    super().__init__(self._hps)
    self.distributed_shampoo = self._optax_gradient_transformation(self._hps)

  def _optax_gradient_transformation(self, hps):
    return optax_distributed_shampoo.distributed_shampoo(
        learning_rate=hps.learning_rate,
        block_size=hps.block_size,
        beta1=hps.beta1,
        beta2=hps.beta2,
        diagonal_epsilon=hps.diagonal_epsilon,
        matrix_epsilon=hps.matrix_epsilon,
        weight_decay=hps.weight_decay,
        start_preconditioning_step=hps.start_preconditioning_step,
        preconditioning_compute_steps=hps.preconditioning_compute_steps,
        statistics_compute_steps=hps.statistics_compute_steps,
        best_effort_shape_interpretation=hps.best_effort_shape_interpretation,
        graft_type=hps.graft_type,
        nesterov=hps.nesterov,
        exponent_override=hps.exponent_override,
        batch_axis_name=hps.batch_axis_name,
        statistics_partition_spec=hps.statistics_partition_spec,
        preconditioner_partition_spec=hps.preconditioner_partition_spec,
        num_devices_for_pjit=hps.num_devices_for_pjit,
        shard_optimizer_states=hps.shard_optimizer_states,
        best_effort_memory_usage_reduction=hps
        .best_effort_memory_usage_reduction,
        inverse_failure_threshold=hps.inverse_failure_threshold,
        moving_average_for_momentum=hps.moving_average_for_momentum,
        skip_preconditioning_dim_size_gt=hps.skip_preconditioning_dim_size_gt,
        clip_by_scaled_gradient_norm=hps.clip_by_scaled_gradient_norm,
        precision=hps.precision)

  def _optax_state_to_optim_state(self, optax_state):
    return OptimizerState(optax_state.count, optax_state.stats)

  def _optim_state_to_optax_state(self, optim_state):
    return optax_distributed_shampoo.ShampooState(optim_state.step,
                                                  optim_state.param_states)

  def init_state(self, params):
    init_fn = self.distributed_shampoo.init
    if self._hps.shard_optimizer_states:
      init_state = self.distributed_shampoo.init(None)
      init_fn = init_state.init_fn

    opt_state = init_fn(params=params)
    return self._optax_state_to_optim_state(opt_state)

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

    self.distributed_shampoo = self._optax_gradient_transformation(hyper_params)
    optax_state = self._optim_state_to_optax_state(state)
    unapplied_updates, new_optax_state = self.distributed_shampoo.update(
        grads, optax_state, params)
    new_params = optax.apply_updates(params, unapplied_updates)
    new_state = self._optax_state_to_optim_state(new_optax_state)
    return new_params, new_state

  def derive_logical_axes(self, optimizer, param_logical_axes):
    """Returns PartitionSpec associated with optimizer states.

    Args:
      optimizer: A flax.Optim optimizer.
      param_logical_axes: Pytree of pjit.PartitionSpec associated with params.
    """
    assert self._hps.shard_optimizer_states
    optimizer_dict = optimizer.state_dict()
    optimizer_logical_axes = jax.tree_map(lambda x: None,
                                          optimizer.state_dict())
    optimizer_logical_axes['target'] = param_logical_axes
    init_state = self.distributed_shampoo.init(None)
    pspec_fn = init_state.pspec_fn
    optimizer_logical_axes['state']['param_states'] = pspec_fn(
        optimizer_dict['target'], param_logical_axes,
        self._hps.statistics_partition_spec)
    return optimizer.restore_state(unfreeze(optimizer_logical_axes))
