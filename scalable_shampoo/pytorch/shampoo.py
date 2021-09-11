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

"""Pytorch implementation of Shampoo."""

from __future__ import print_function

import enum
import itertools

from dataclasses import dataclass
import matrix_functions
import numpy as np
import torch
import torch.optim as optim


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
# but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
class LayerwiseGrafting(enum.IntEnum):
  NONE = 0
  SGD = 1
  ADAGRAD = 2


@dataclass
class ShampooHyperParams:
  """Shampoo hyper parameters."""
  beta2: float = 1.0
  diagonal_eps: float = 1e-6
  matrix_eps: float = 1e-12
  weight_decay: float = 0.0
  inverse_exponent_override: int = 0  # fixed exponent for preconditioner, if >0
  start_preconditioning_step: int = 1
  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner.
  preconditioning_compute_steps: int = 1
  # How often to compute statistics.
  statistics_compute_steps: int = 1
  # Block size for large layers (if > 0).
  # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
  # Block size should be as large as feasible under memory/time constraints.
  block_size: int = 128
  # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
  # 12 x [1024, 512] L and R statistics. Disabled by default which results in
  # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
  best_effort_shape_interpretation: bool = True
  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int = LayerwiseGrafting.SGD
  # Nesterov momentum
  nesterov: bool = True


class Graft:
  """Base class to perform grafting onto Shampoo. This class does no grafting.
  """

  def __init__(self, hps, unused_var):
    self.hps = hps

  def add_statistics(self, grad):
    pass

  def precondition_gradient(self, grad):
    return grad

  def update_momentum(self, update, unused_beta1):
    return update


class SGDGraft(Graft):
  """Graft using SGD+momentum.

  momentum maintains an exponentially weighted moving average of gradients.
  """

  def __init__(self, hps, var):
    super(SGDGraft, self).__init__(hps, var)
    self.momentum = torch.zeros_like(var.data, device=var.get_device())

  def update_momentum(self, update, beta1):
    self.momentum.mul_(beta1).add_(update)
    return self.momentum


class AdagradGraft(SGDGraft):
  """Graft using Adagrad.

  Essentially an implementation of Adagrad with momentum.
  """

  def __init__(self, hps, var):
    super(AdagradGraft, self).__init__(hps, var)
    self.statistics = torch.zeros_like(var.data, device=var.get_device())

  def add_statistics(self, grad):
    self.statistics.add_(grad * grad)

  def precondition_gradient(self, grad):
    return grad / (torch.sqrt(self.statistics) + self.hps.diagonal_eps)


class BlockPartitioner:
  """Partitions a tensor into smaller tensors for preconditioning.

    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.
  """

  def __init__(self, var, hps):
    self._shape = var.shape
    self._splits = []
    self._split_sizes = []
    split_sizes = []
    # We split var into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(var.shape):
      if hps.block_size > 0 and d > hps.block_size:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d-1) // hps.block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        self._split_sizes.append((i, sizes))
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
    for (i, sizes) in self._split_sizes:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(
            torch.split(t, tuple(sizes), dim=i))
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
            torch.cat(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim):
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


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, var, hps):
    self._hps = hps
    self._original_shape = var.shape
    self._transformed_shape = var.shape
    if hps.best_effort_shape_interpretation:
      self._transformed_shape = _merge_small_dims(
          self._original_shape, hps.block_size)

    reshaped_var = torch.reshape(var, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_var, hps)
    shapes = self._partitioner.shapes_for_preconditioners()
    rank = len(self._transformed_shape)
    device = var.get_device()
    if rank <= 1:
      self.statistics = []
      self.preconditioners = []
    else:
      eps = self._hps.matrix_eps
      self.statistics = [eps * torch.eye(s[0], device=device) for s in shapes]
      self.preconditioners = [torch.eye(s[0], device=device) for s in shapes]

  def add_statistics(self, grad):
    """Compute statistics from gradients and add to the correct state entries.

    Args:
      grad: Gradient to compute statistics from.
    """
    if not self.statistics: return
    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    w1 = self._hps.beta2
    w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
    rank = len(self._transformed_shape)
    for j, grad in enumerate(partitioned_grads):
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = torch.tensordot(grad, grad, [axes, axes])
        self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    if self._hps.inverse_exponent_override > 0:
      return self._hps.inverse_exponent_override
    return 2 * len(self._transformed_shape)

  def compute_preconditioners(self):
    """Compute L^{-1/exp} for each stats matrix L."""
    exp = self.exponent_for_preconditioner()
    eps = self._hps.matrix_eps
    for i, stat in enumerate(self.statistics):
      self.preconditioners[i] = matrix_functions.ComputePower(
          stat, exp, ridge_epsilon=eps)

  def preconditioned_grad(self, grad):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.

    Returns:
      A preconditioned gradient.
    """
    if not self.preconditioners: return grad
    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, grad in enumerate(partitioned_grads):
      preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) *
                                                      num_splits]
      rank = len(grad.shape)
      precond_grad = grad
      for j in range(rank):
        preconditioner = preconditioners_for_grad[j]
        precond_grad = torch.tensordot(
            precond_grad, preconditioner, [[0], [0]])
      preconditioned_partitioned_grads.append(precond_grad)
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads)
    return torch.reshape(merged_grad, self._original_shape)


STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'
GRAFT = 'graft'


class Shampoo(optim.Optimizer):
  """The Shampoo optimizer."""

  def __init__(self,
               params,
               lr=1.0,
               momentum=0.9,
               hyperparams=ShampooHyperParams()):
    defaults = dict(lr=lr, momentum=momentum)
    self.hps = hyperparams
    super(Shampoo, self).__init__(params, defaults)

  def init_var_state(self, var, state):
    """Initialize the PyTorch state of for a single variable."""
    state[STEP] = 0
    state[MOMENTUM] = torch.zeros_like(var.data, device=var.get_device())
    state[PRECONDITIONER] = Preconditioner(var, self.hps)
    if self.hps.graft_type == LayerwiseGrafting.ADAGRAD:
      state[GRAFT] = AdagradGraft(self.hps, var)
    elif self.hps.graft_type == LayerwiseGrafting.SGD:
      state[GRAFT] = SGDGraft(self.hps, var)
    else:
      state[GRAFT] = Graft(self.hps, var)

  def step(self, closure=None):
    hps = self.hps
    for group in self.param_groups:
      lr = group['lr']
      for p in group['params']:
        if p.grad is None: continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Shampoo does not support sparse yet')
        state = self.state[p]
        if not state:
          self.init_var_state(p, state)
        state[STEP] += 1

        preconditioner = state[PRECONDITIONER]
        graft = state[GRAFT]

        # Gather statistics, compute preconditioners
        graft.add_statistics(grad)
        if state[STEP] % hps.statistics_compute_steps == 0:
          preconditioner.add_statistics(grad)
        if state[STEP] % hps.preconditioning_compute_steps == 0:
          preconditioner.compute_preconditioners()

        # Precondition gradients
        graft_grad = graft.precondition_gradient(grad)
        shampoo_grad = grad
        if state[STEP] >= self.hps.start_preconditioning_step:
          shampoo_grad = preconditioner.preconditioned_grad(grad)

        # Grafting
        graft_norm = torch.norm(graft_grad)
        shampoo_norm = torch.norm(shampoo_grad)
        shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))

        # Weight decay
        if self.hps.weight_decay != 0.0:
          shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
          graft_grad.add_(p.data, alpha=self.hps.weight_decay)

        # Momentum and Nesterov momentum, if needed
        state[MOMENTUM].mul_(group['momentum']).add_(shampoo_grad)
        graft_momentum = graft.update_momentum(grad, group['momentum'])

        if state[STEP] >= self.hps.start_preconditioning_step:
          momentum_update = state[MOMENTUM]
          wd_update = shampoo_grad
        else:
          momentum_update = graft_momentum
          wd_update = graft_grad

        if hps.nesterov:
          momentum_update.mul_(group['momentum']).add_(wd_update)

        # Final update
        p.data.add_(momentum_update, alpha=-lr)
