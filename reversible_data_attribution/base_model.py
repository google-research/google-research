# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Base model container and training trajectory tracker for data attribution.

This module implements AttributionBaseModel, which encapsulates:
1. Initial model architecture and starting random seed.
2. Adam optimizer hyperparameters (lr, beta_1, beta_2, eps).
3. Single training execution with automatic checkpointing of model weights,
   momentum, variance, and gradients at specified step intervals.
4. Counterfactual retraining with specified sample indices removed.
"""

from collections.abc import Callable, Sequence
import copy
import dataclasses
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import tqdm

from reversible_data_attribution import reversible
from reversible_data_attribution import utils


@dataclasses.dataclass
class CheckpointState:
  """State captured at a specific training step."""

  step: int
  model: nn.Module
  momentum: list[torch.Tensor]
  variance: list[torch.Tensor]
  gradients: list[torch.Tensor] | None = None


class AttributionBaseModel:
  """Container for the base model, its training state, and counterfactual retraining.

  This class encapsulates a neural network model, its loss function, and its
  Adam optimizer hyperparameters. It records the complete optimization
  trajectory (checkpoints, momentum, variance, gradients) during a single
  training run and provides service methods for counterfactual retraining.
  """

  def __init__(
      self,
      model,
      loss_fn = nn.functional.cross_entropy,
      lr = 0.001,
      beta_1 = 0.9,
      beta_2 = 0.999,
      eps = 1e-8,
      optimizer = None,
      seed = 0,
  ):
    """Initializes the AttributionBaseModel.

    Args:
      model: The initial PyTorch neural network module.
      loss_fn: The loss function to use during training (e.g.
        nn.functional.cross_entropy or nn.CrossEntropyLoss()).
      lr: Learning rate for the optimizer.
      beta_1: Exponential decay rate for first moment estimates (momentum).
      beta_2: Exponential decay rate for second moment estimates (variance).
      eps: Epsilon parameter for numerical stability in Adam.
      optimizer: Optional pre-configured PyTorch optimizer. If provided,
        hyperparameters will be extracted or checked from it.
      seed: Starting random seed for deterministic training and retraining.
    """
    self._seed = seed
    self._initial_model = copy.deepcopy(model).cpu()
    self._model = copy.deepcopy(model)
    self._loss_fn = loss_fn
    self._lr = lr
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._eps = eps
    self._device: torch.device | str = 'cpu'

    if optimizer is not None:
      self._optimizer = optimizer
      if optimizer.param_groups:
        group = optimizer.param_groups[0]
        if 'lr' in group:
          self._lr = group['lr']
        if 'betas' in group:
          self._beta_1, self._beta_2 = group['betas']
        if 'eps' in group:
          self._eps = group['eps']
    else:
      self._optimizer = torch.optim.Adam(
          self._model.parameters(),
          lr=self._lr,
          betas=(self._beta_1, self._beta_2),
          eps=self._eps,
      )

    self._checkpoints: dict[int, CheckpointState] = {}
    self._train_loader: Any = None
    self._num_epochs: int = 0
    self._total_steps: int = 0
    self._step_idx: int = 0
    self._is_trained: bool = False

  # --- Getter methods ---

  def beta_1(self):
    """Returns the beta_1 hyperparameter for Adam."""
    return self._beta_1

  def beta_2(self):
    """Returns the beta_2 hyperparameter for Adam."""
    return self._beta_2

  def eps(self):
    """Returns the epsilon hyperparameter for Adam."""
    return self._eps

  def epsilon(self):
    """Alias for eps()."""
    return self._eps

  def lr(self):
    """Returns the learning rate."""
    return self._lr

  def learning_rate(self):
    """Alias for lr()."""
    return self._lr

  def optimizer(self):
    """Returns the optimizer instance."""
    return self._optimizer

  def model(self):
    """Returns the current model (trained if train_model was called)."""
    return self._model

  def initial_model(self):
    """Returns a deepcopy of the initial model before training."""
    return copy.deepcopy(self._initial_model)

  def loss_fn(self):
    """Returns the loss function."""
    return self._loss_fn

  def seed(self):
    """Returns the initial seed."""
    return self._seed

  def train_loader(self):
    """Returns the training data loader used during train_model."""
    return self._train_loader

  def num_epochs(self):
    """Returns the number of training epochs."""
    return self._num_epochs

  def total_steps(self):
    """Returns the total number of optimization steps performed."""
    return self._total_steps

  def is_trained(self):
    """Returns True if train_model has been run."""
    return self._is_trained

  def checkpoints(self):
    """Returns the dictionary of recorded checkpoints keyed by step index."""
    return self._checkpoints

  # --- Compatibility helper getters for influence functions ---

  def get_checkpoint_models(self):
    """Returns a dictionary mapping step -> nn.Module checkpoint."""
    return {k: v.model for k, v in self._checkpoints.items()}

  def get_momentum_dict(self):
    """Returns a dictionary mapping step -> list of momentum tensors."""
    return {k: v.momentum for k, v in self._checkpoints.items()}

  def get_variance_dict(self):
    """Returns a dictionary mapping step -> list of variance tensors."""
    return {k: v.variance for k, v in self._checkpoints.items()}

  def get_gradients_dict(self):
    """Returns a dictionary mapping step -> list of gradient tensors."""
    return {
        k: v.gradients
        for k, v in self._checkpoints.items()
        if v.gradients is not None
    }

  # --- Main Training Method ---

  def train_model(
      self,
      train_loader,
      num_epochs,
      checkpoint_freq = None,
      save_all_ckpts = False,
      device = 'cpu',
      save_gradients = True,
  ):
    """Runs the training loop once and records checkpoints and trajectory metadata.

    Args:
      train_loader: Training DataLoader, Dataset, or sequence of batches.
      num_epochs: Number of training epochs.
      checkpoint_freq: Step interval at which to record checkpoints. Step 0 and
        the final step are always recorded. If None, only step 0 and the final
        step are recorded.
      device: Device to run computation on ('cpu' or 'cuda').
      save_gradients: Whether to record parameter gradients at each checkpoint.

    Returns:
      The trained final PyTorch model.

    Raises:
      RuntimeError: If train_model is called more than once on the same instance.
    """
    if self._is_trained:
      raise RuntimeError(
          'AttributionBaseModel has already been trained. The base model cannot'
          ' be trained more than once.'
      )

    # Set seeds for deterministic training
    random.seed(self._seed)
    np.random.seed(self._seed)
    torch.manual_seed(self._seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self._seed)

    self._device = device
    self._train_loader = train_loader
    self._num_epochs = num_epochs

    device_obj = torch.device(device)
    self._model = self._model.to(device_obj)
    self._model.train()

    # Re-initialize Adam optimizer on device parameters
    self._optimizer = torch.optim.Adam(
        self._model.parameters(),
        lr=self._lr,
        betas=(self._beta_1, self._beta_2),
        eps=self._eps,
    )

    batches = utils.prepare_batches(train_loader)
    num_batches = len(batches)
    total_steps = num_epochs * num_batches
    self._total_steps = total_steps

    # Record Step 0 checkpoint
    init_model_cpu = copy.deepcopy(self._model).cpu()
    init_momentum = [
        torch.zeros_like(p).cpu() for p in self._model.parameters()
    ]
    init_variance = [
        torch.zeros_like(p).cpu() for p in self._model.parameters()
    ]
    self._checkpoints[0] = CheckpointState(
        step=0,
        model=init_model_cpu,
        momentum=init_momentum,
        variance=init_variance,
        gradients=None,
    )

    step_idx = 0
    for _ in tqdm.tqdm(range(num_epochs)):
      for x_batch, y_batch, _ in batches:
        step_idx += 1
        x_batch = x_batch.to(device_obj)
        y_batch = y_batch.to(device_obj)

        self._optimizer.zero_grad()
        y_hat = self._model(x_batch)
        loss = self._loss_fn(y_hat, y_batch)
        grads: tuple[torch.Tensor, Ellipsis] = torch.autograd.grad(
            loss, list(self._model.parameters()), create_graph=False
        )

        for p, g in zip(self._model.parameters(), grads):
          p.grad = g.detach().clone()
        self._optimizer.step()

        # Check if we should record this step
        is_checkpoint_step = save_all_ckpts
        if checkpoint_freq is not None and checkpoint_freq > 0:
          if step_idx % checkpoint_freq == 0:
            is_checkpoint_step = True
        if step_idx == total_steps:
          is_checkpoint_step = True

        if is_checkpoint_step:
          cur_momentum = [
              self._optimizer.state[p]['exp_avg'].detach().cpu().clone()
              if p in self._optimizer.state
              and 'exp_avg' in self._optimizer.state[p]
              else torch.zeros_like(p).cpu()
              for p in self._model.parameters()
          ]
          cur_variance = [
              self._optimizer.state[p]['exp_avg_sq'].detach().cpu().clone()
              if p in self._optimizer.state
              and 'exp_avg_sq' in self._optimizer.state[p]
              else torch.zeros_like(p).cpu()
              for p in self._model.parameters()
          ]
          cur_grads = (
              [g.detach().cpu() for g in grads] if save_gradients else None
          )

          ckpt = CheckpointState(
              step=step_idx,
              model=copy.deepcopy(self._model).cpu(),
              momentum=cur_momentum,
              variance=cur_variance,
              gradients=cur_grads,
          )
          self._checkpoints[step_idx] = ckpt

    # Alias step -1 to the final checkpoint for convenience
    if total_steps in self._checkpoints:
      self._checkpoints[-1] = self._checkpoints[total_steps]

    self._step_idx = total_steps
    self._is_trained = True
    return self._model

  def train_model_continue(
      self,
      batches,
      num_steps = None,
      indices_to_remove = None,
      checkpoint_freq = None,
      save_all_ckpts = False,
      device = None,
      save_gradients = True,
      batch_scale = True,
  ):
    """Continues training for a specified number of steps or batches on active samples.

    Maintains the existing Adam optimizer state (momentum, variance, and global
    step count) and records window checkpoints and batches.

    Args:
      batches: Sequence of batches or DataLoader to draw batches from.
      num_steps: Optional maximum number of optimization steps to execute.
      indices_to_remove: Set or sequence of dataset indices to exclude/0-mask.
      checkpoint_freq: Frequency at which to save checkpoints.
      save_all_ckpts: Whether to save checkpoints at every single step.
      device: Optional computation device override (defaults to current device).
      save_gradients: Whether to record parameter gradients in checkpoints.
      batch_scale: Whether to scale batch loss by (num_kept / orig_batch_size).

    Returns:
      A tuple of:
        - The current PyTorch model.
        - List of global step indices executed during this segment.
        - List of the actual batches (x, y, indices) executed in this segment.
    """
    dev = device if device is not None else self._device
    device_obj = torch.device(dev)
    self._device = dev
    self._model = self._model.to(device_obj)
    self._model.train()

    if 0 not in self._checkpoints:
      random.seed(self._seed)
      np.random.seed(self._seed)
      torch.manual_seed(self._seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(self._seed)

      self._optimizer = torch.optim.Adam(
          self._model.parameters(),
          lr=self._lr,
          betas=(self._beta_1, self._beta_2),
          eps=self._eps,
      )
      init_model_cpu = copy.deepcopy(self._model).cpu()
      init_momentum = [
          torch.zeros_like(p).cpu() for p in self._model.parameters()
      ]
      init_variance = [
          torch.zeros_like(p).cpu() for p in self._model.parameters()
      ]
      self._checkpoints[0] = CheckpointState(
          step=0,
          model=init_model_cpu,
          momentum=init_momentum,
          variance=init_variance,
          gradients=None,
      )
      self._step_idx = 0

    if indices_to_remove is None:
      skip_set = set()
    elif isinstance(indices_to_remove, int):
      skip_set = {indices_to_remove}
    else:
      skip_set = set(indices_to_remove)

    prepared_batches = utils.prepare_batches(batches)
    executed_steps: list[int] = []
    executed_batches: list[Any] = []

    steps_taken = 0
    for x_batch, y_batch, indices in prepared_batches:
      if num_steps is not None and steps_taken >= num_steps:
        break

      orig_batch_size = x_batch.shape[0]
      if indices is not None:
        if isinstance(indices, torch.Tensor):
          idx_list: list[int] = indices.tolist()
        else:
          idx_list = list(indices)
        keep_mask = [
            pos for pos, idx in enumerate(idx_list) if idx not in skip_set
        ]
      else:
        idx_list = []
        keep_mask = list(range(orig_batch_size))

      if not keep_mask:
        continue

      x_batch = x_batch.to(device_obj)
      y_batch = y_batch.to(device_obj)

      if len(keep_mask) < orig_batch_size:
        mask_tensor = torch.tensor(keep_mask, device=device_obj)
        x_exec = x_batch[mask_tensor]
        y_exec = y_batch[mask_tensor]
        indices_exec = (
            [idx_list[pos] for pos in keep_mask]
            if indices is not None
            else None
        )
        scale = (
            (len(keep_mask) / float(orig_batch_size)) if batch_scale else 1.0
        )
      else:
        x_exec = x_batch
        y_exec = y_batch
        indices_exec = indices
        scale = 1.0

      self._step_idx += 1
      self._total_steps = max(self._total_steps, self._step_idx)
      steps_taken += 1
      cur_step = self._step_idx

      self._optimizer.zero_grad()
      y_hat = self._model(x_exec)
      loss = self._loss_fn(y_hat, y_exec) * scale
      grads = torch.autograd.grad(
          loss, list(self._model.parameters()), create_graph=False
      )

      for p, g in zip(self._model.parameters(), grads):
        p.grad = g.detach().clone()
      self._optimizer.step()

      is_checkpoint_step = save_all_ckpts
      if checkpoint_freq is not None and checkpoint_freq > 0:
        if cur_step % checkpoint_freq == 0:
          is_checkpoint_step = True

      if is_checkpoint_step:
        cur_momentum = [
            self._optimizer.state[p]['exp_avg'].detach().cpu().clone()
            if p in self._optimizer.state
            and 'exp_avg' in self._optimizer.state[p]
            else torch.zeros_like(p).cpu()
            for p in self._model.parameters()
        ]
        cur_variance = [
            self._optimizer.state[p]['exp_avg_sq'].detach().cpu().clone()
            if p in self._optimizer.state
            and 'exp_avg_sq' in self._optimizer.state[p]
            else torch.zeros_like(p).cpu()
            for p in self._model.parameters()
        ]
        cur_grads = (
            [g.detach().cpu() for g in grads] if save_gradients else None
        )

        ckpt = CheckpointState(
            step=cur_step,
            model=copy.deepcopy(self._model).cpu(),
            momentum=cur_momentum,
            variance=cur_variance,
            gradients=cur_grads,
        )
        self._checkpoints[cur_step] = ckpt

      executed_steps.append(cur_step)
      executed_batches.append(
          (x_exec.cpu(), y_exec.cpu(), indices_exec)
      )

    self._is_trained = True
    if self._step_idx in self._checkpoints:
      self._checkpoints[-1] = self._checkpoints[self._step_idx]

    return self._model, executed_steps, executed_batches

  # --- Counterfactual Retraining Service Method ---

  def retrain_remove_indices(
      self,
      indices_to_remove,
      train_loader = None,
      num_epochs = None,
      device = None,
      batch_scale = True,
  ):
    """Retrains a fresh model from scratch with specified sample indices removed.

    Uses the exact same initial model state, seed, optimizer hyperparameters,
    and loss function as the base model.

    Args:
      indices_to_remove: Index or collection of dataset indices to exclude.
      train_loader: Optional train_loader override (defaults to
        self.train_loader).
      num_epochs: Optional number of epochs override (defaults to
        self.num_epochs).
      device: Optional computation device override (defaults to self._device).
      batch_scale: Whether to scale batch loss by (num_kept / orig_batch_size)
        when samples in a batch are excluded.

    Returns:
      The retrained PyTorch nn.Module.
    """
    if isinstance(indices_to_remove, int):
      skip_set = {indices_to_remove}
    else:
      skip_set = set(indices_to_remove)

    loader = train_loader if train_loader is not None else self._train_loader
    if loader is None:
      raise ValueError(
          'train_loader must be provided or train_model must be run first.'
      )

    epochs = num_epochs if num_epochs is not None else self._num_epochs
    dev = device if device is not None else self._device
    device_obj = torch.device(dev)

    # Deterministic reset to the same initial seed
    random.seed(self._seed)
    np.random.seed(self._seed)
    torch.manual_seed(self._seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self._seed)

    model_cf = copy.deepcopy(self._initial_model).to(device_obj)
    model_cf.train()

    optimizer_cf = torch.optim.Adam(
        model_cf.parameters(),
        lr=self._lr,
        betas=(self._beta_1, self._beta_2),
        eps=self._eps,
    )

    batches = utils.prepare_batches(loader)

    for _ in range(epochs):
      for j, (x_batch, y_batch, indices) in enumerate(batches):
        x_batch = x_batch.to(device_obj)
        y_batch = y_batch.to(device_obj)
        orig_batch_size = x_batch.shape[0]

        # Determine which samples to keep in this batch
        if indices is not None:
          if isinstance(indices, torch.Tensor):
            idx_list = indices.tolist()
          else:
            idx_list = list(indices)
          keep_mask = [
              pos for pos, idx in enumerate(idx_list) if idx not in skip_set
          ]
        else:
          batch_start = sum(b[0].shape[0] for b in batches[:j])
          keep_mask = [
              pos
              for pos in range(orig_batch_size)
              if (batch_start + pos) not in skip_set
          ]

        if not keep_mask:
          continue

        if len(keep_mask) < orig_batch_size:
          mask_tensor = torch.tensor(keep_mask, device=device_obj)
          x_batch = x_batch[mask_tensor]
          y_batch = y_batch[mask_tensor]
          scale = (
              (len(keep_mask) / float(orig_batch_size)) if batch_scale else 1.0
          )
        else:
          scale = 1.0

        optimizer_cf.zero_grad()
        y_hat = model_cf(x_batch)
        loss = self._loss_fn(y_hat, y_batch) * scale
        loss.backward()
        optimizer_cf.step()

    return model_cf

  def retrain_with_quantize(
      self,
      train_loader,
      num_epochs,
      checkpoint_freq = None,
      device = 'cpu',
      save_all_ckpts = False,
      save_gradients = True,
      quantization_scale = 1000000,
  ):
    """Retrains model with fixed-point / quantized representations.

    Performs training with integer-scaled momentum and variance buffers and
    quantized weights to enable exact reversible backward reconstruction.

    Args:
      train_loader: Training DataLoader, Dataset, or sequence of batches.
      num_epochs: Number of training epochs.
      checkpoint_freq: Step interval at which to record checkpoints. Step 0 and
        the final step (-1) are always recorded. If None, only step 0 and step
        -1 are recorded (unless save_all_ckpts is True).
      device: Device to run computation on ('cpu' or 'cuda').
      save_all_ckpts: Whether to record a checkpoint at every training step.
      save_gradients: Whether to record parameter gradients at each checkpoint.
      quantization_scale: Fixed-point integer scaling factor.

    Returns:
      Tuple of (quantized_model, quantized_checkpoints, momentum_buffer,
      variance_buffer).
    """
    # Set seeds for deterministic training
    random.seed(self._seed)
    np.random.seed(self._seed)
    torch.manual_seed(self._seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self._seed)

    self._device = device
    self._train_loader = train_loader
    self._num_epochs = num_epochs

    device_obj = torch.device(device)
    model = copy.deepcopy(self._initial_model)
    model = model.to(device=device_obj, dtype=torch.float64)
    model.train()
    split_sizes = [p.numel() for p in model.parameters()]
    param_dtype = torch.float64

    batches = utils.prepare_batches(train_loader)
    num_batches = len(batches)
    total_steps = num_epochs * num_batches
    self._total_steps = total_steps
    num_params = sum(p.numel() for p in model.parameters())

    momentum_buffer = reversible.GPUReversibleTransform(
        num_params,
        gamma=self._beta_1,
        max_steps=total_steps,
        device=device_obj,
    )
    variance_buffer = reversible.GPUReversibleTransform(
        num_params,
        gamma=self._beta_2,
        max_steps=total_steps,
        device=device_obj,
        mode='ceil',
    )

    scaled_momentum = torch.zeros(
        num_params, dtype=torch.int64, device=device_obj
    )
    # Initialize to non-zero 1 (or 0)
    scaled_variance = torch.ones(
        num_params, dtype=torch.int64, device=device_obj
    )

    # First we store initial model weights as a flat int64 vector
    flat_weights_int = torch.cat([
        torch.round(w.data.to(torch.float64).flatten() * quantization_scale).to(
            torch.int64
        )
        for w in model.parameters()
    ])
    # Update model parameters to quantized initial values
    flat_weights_float = flat_weights_int.to(param_dtype) / quantization_scale
    offset = 0
    for p in model.parameters():
      numel = p.numel()
      p.data.copy_(flat_weights_float[offset : offset + numel].view_as(p))
      offset += numel

    # Record Step 0 checkpoint
    init_model_cpu = copy.deepcopy(model).cpu()
    unscaled_momentum_0 = scaled_momentum.to(torch.float64) / quantization_scale
    unscaled_variance_0 = scaled_variance.to(torch.float64) / quantization_scale
    init_momentum = [
        t.view(p.shape).to(torch.float64).cpu()
        for t, p in zip(
            torch.split(unscaled_momentum_0, split_sizes),
            model.parameters(),
        )
    ]
    init_variance = [
        t.view(p.shape).to(torch.float64).cpu()
        for t, p in zip(
            torch.split(unscaled_variance_0, split_sizes),
            model.parameters(),
        )
    ]

    # We need a new set of checkpoints for the retrained model.
    quantized_checkpoints = {}
    quantized_checkpoints[0] = CheckpointState(
        step=0,
        model=init_model_cpu,
        momentum=init_momentum,
        variance=init_variance,
        gradients=None,
    )

    step_idx = 0
    for _ in tqdm.tqdm(range(num_epochs)):
      for x_batch, y_batch, _ in batches:
        step_idx += 1
        x_batch = x_batch.to(device=device_obj, dtype=param_dtype)
        y_batch = y_batch.to(device_obj)
        model.zero_grad(set_to_none=True)
        y_hat = model(x_batch)
        loss = self._loss_fn(y_hat, y_batch)
        grads: tuple[torch.Tensor, Ellipsis] = torch.autograd.grad(
            loss, list(model.parameters()), create_graph=False
        )
        for p, g in zip(model.parameters(), grads):
          p.grad = g.detach().clone()

        grad_flat = torch.cat([g.detach().flatten() for g in grads]).to(
            torch.float64
        )
        # Check if we should record this step
        is_checkpoint_step = save_all_ckpts
        if checkpoint_freq is not None and checkpoint_freq > 0:
          if step_idx % checkpoint_freq == 0:
            is_checkpoint_step = True
        if step_idx == total_steps:
          is_checkpoint_step = True

        # Scaled gradients in int64 on device
        grad_scaled_m = torch.round(
            (1.0 - self._beta_1) * grad_flat * quantization_scale
        ).to(torch.int64)
        grad_scaled_v = torch.ceil(
            (1.0 - self._beta_2) * (grad_flat ** 2) * quantization_scale
        ).to(torch.int64)

        # Buffer decay on GPU
        scaled_momentum = (
            momentum_buffer.forward_decay(scaled_momentum) + grad_scaled_m
        )
        scaled_variance = (
            variance_buffer.forward_decay(scaled_variance) + grad_scaled_v
        )

        unscaled_momentum = (
            scaled_momentum.to(torch.float64) / quantization_scale
        )
        unscaled_variance = (
            scaled_variance.to(torch.float64) / quantization_scale
        )

        # step_idx is already incremented so it's actually t+1
        bc1 = 1.0 - (self._beta_1 ** step_idx)
        bc2 = 1.0 - (self._beta_2 ** step_idx)
        unscaled_momentum_hat = unscaled_momentum / bc1
        unscaled_variance_hat = unscaled_variance / bc2

        update_limit = utils.compute_update_limit(
            self._beta_1, self._beta_2, step_idx
        )
        r_unscaled = unscaled_momentum_hat / (
            self._eps + torch.sqrt(unscaled_variance_hat)
        )
        r_clamped = torch.clamp(r_unscaled, -update_limit, update_limit)
        update_vec = self._lr * r_clamped
        update_vec_int = torch.round(update_vec * quantization_scale).to(
            torch.int64
        )
        flat_weights_int.sub_(update_vec_int)
        flat_weights_float = (
            flat_weights_int.to(param_dtype) / quantization_scale
        )
        offset = 0
        for p in model.parameters():
          numel = p.numel()
          p.data.copy_(flat_weights_float[offset : offset + numel].view_as(p))
          offset += numel

        if is_checkpoint_step:
          cur_momentum = [
              t.view(p.shape).to(torch.float64).cpu()
              for t, p in zip(
                  torch.split(unscaled_momentum, split_sizes),
                  model.parameters(),
              )
          ]
          cur_variance = [
              t.view(p.shape).to(torch.float64).cpu()
              for t, p in zip(
                  torch.split(unscaled_variance, split_sizes),
                  model.parameters(),
              )
          ]
          cur_grads = (
              [g.detach().cpu() for g in grads] if save_gradients else None
          )

          ckpt = CheckpointState(
              step=step_idx,
              model=copy.deepcopy(model).cpu(),
              momentum=cur_momentum,
              variance=cur_variance,
              gradients=cur_grads,
          )
          quantized_checkpoints[step_idx] = ckpt

    # Alias step -1 to the final checkpoint for convenience
    if total_steps in quantized_checkpoints:
      quantized_checkpoints[-1] = quantized_checkpoints[total_steps]

    self._is_trained = True
    return model, quantized_checkpoints, momentum_buffer, variance_buffer

  # Backward-compatible alias
  retrain_quantized = retrain_with_quantize


# Backward-compatible and descriptive alias
BaseModel = AttributionBaseModel
