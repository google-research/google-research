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

"""General purpose utility functions for influence function computation and PyTorch batching."""

from collections.abc import Callable
import contextlib
import copy
import gc
from typing import Any, Sequence

from absl import logging
import numpy as np
import scipy.signal
import torch
from torch import nn

from reversible_data_attribution import reversible


def garbage_collect(device = None):
  """Runs Python garbage collection and empties CUDA memory cache if available."""
  _ = device
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def log_cuda_memory(
    prefix, device = None
):
  """Logs allocated, reserved, and peak CUDA memory if CUDA is available, or a CPU/CUDA-unavailable message otherwise."""
  if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    logging.info(
        '%s - CUDA memory: allocated=%.2f MB, reserved=%.2f MB,'
        ' max_allocated=%.2f MB',
        prefix,
        allocated,
        reserved,
        max_allocated,
    )
  else:
    logging.info(
        '%s - CUDA memory: CUDA is not available (device=%s)', prefix, device
    )


def get_dynamic_chunk_size(
    num_params,
    dtype = torch.float32,
    device = None,
    max_memory_mb = 1024.0,  # Max cap per chunk buffer (e.g., 1 GB)
    memory_fraction = 0.25,  # Max fraction of free GPU VRAM to use
    num_state_tensors = 1,
):
  """Dynamically computes the sample chunk size based on available GPU memory and parameter count."""
  device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  bytes_per_sample = (
      num_params
      * torch.empty(0, dtype=dtype).element_size()
      * max(1, num_state_tensors)
  )
  if bytes_per_sample == 0:
    return 1
  if device.type == 'cuda':
    # Query free GPU memory in bytes
    free_mem, _ = torch.cuda.mem_get_info(device)
    # Target budget: fraction of free VRAM capped at max_memory_mb
    target_budget_bytes = min(free_mem * memory_fraction, max_memory_mb * 1024 * 1024)
  else:
    # CPU fallback
    target_budget_bytes = max_memory_mb * 1024 * 1024
  chunk_size = int(target_budget_bytes // bytes_per_sample)
  return max(1, chunk_size)


@contextlib.contextmanager
def sdpa_math_context(enable = True):
  """Context manager to scope the SDPA MATH backend for higher-order derivatives (Hessian/HVP).

  When enable=True, forces PyTorch SDPA to use the composite MATH backend which
  supports double backward (autograd graph differentiation). When enable=False,
  leaves the default SDPA backend configuration untouched (enabling
  FlashAttention
  and Mem-Efficient Attention for maximum speed).
  """
  if not enable:
    yield
    return

  if (
      hasattr(torch, 'nn')
      and hasattr(torch.nn, 'attention')
      and hasattr(torch.nn.attention, 'sdpa_kernel')
      and hasattr(torch.nn.attention, 'SDPBackend')
  ):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
      yield
  elif (
      hasattr(torch, 'backends')
      and hasattr(torch.backends, 'cuda')
      and hasattr(torch.backends.cuda, 'sdp_kernel')
  ):
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
      yield
  else:
    yield


def prepare_batches(data_loader):
  """Converts a data_loader (DataLoader, Iterable, or Sequence) into a list of (x_batch, y_batch, indices)."""
  """Args:
    data_loader: The data loader to prepare.
  Returns:
    A list of (x_batch, y_batch, indices) tuples.
  """
  batches = []
  for batch in data_loader:
    indices = None
    if isinstance(batch, (tuple, list)):
      x_batch, y_batch = batch[0], batch[1]
      if len(batch) > 2:
        indices = batch[2]
    elif isinstance(batch, dict):
      if 'image' in batch:
        x_batch = batch['image']
      elif 'x' in batch:
        x_batch = batch['x']
      elif 'features' in batch:
        x_batch = batch['features']
      elif 'input' in batch:
        x_batch = batch['input']
      else:
        raise ValueError(
            f'Unknown input key in batch dict: {list(batch.keys())}'
        )

      if 'label' in batch:
        y_batch = batch['label']
      elif 'y' in batch:
        y_batch = batch['y']
      elif 'targets' in batch:
        y_batch = batch['targets']
      elif 'target' in batch:
        y_batch = batch['target']
      else:
        raise ValueError(
            f'Unknown target key in batch dict: {list(batch.keys())}'
        )

      if 'index' in batch:
        indices = batch['index']
      elif 'indices' in batch:
        indices = batch['indices']
      elif 'idx' in batch:
        indices = batch['idx']
    else:
      raise TypeError(f'Unsupported batch type: {type(batch)}')

    if isinstance(x_batch, np.ndarray):
      x_batch = torch.from_numpy(x_batch).float()
    if isinstance(y_batch, np.ndarray):
      y_batch = torch.from_numpy(y_batch)
    if isinstance(indices, (np.ndarray, list, tuple)):
      indices = torch.tensor(indices)

    batches.append((x_batch, y_batch, indices))
  return batches


_prepare_batches = prepare_batches


def compute_batch_start_offsets(
    batches,
):
  """Precomputes cumulative start offsets for each batch in batches."""
  offsets = [0]
  for b in batches[:-1]:
    offsets.append(offsets[-1] + b[0].shape[0])
  return offsets


def get_batch_active_indices(
    batches,
    j,
    candidate_set,
    batch_start_offsets = None,
):
  """Returns list of (dataset_idx, batch_pos) for samples in batch j that are in candidate_set.

  Args:
    batches: List of (x_batch, y_batch, indices) tuples.
    j: Batch index.
    candidate_set: Set of target dataset indices.
    batch_start_offsets: Optional precomputed list of cumulative start offsets
      for each batch.

  Returns:
    List of (dataset_idx, batch_pos) tuples.
  """
  x_batch, _, indices = batches[j]
  batch_size = x_batch.shape[0]
  if indices is not None:
    if isinstance(indices, torch.Tensor):
      indices_list = indices.tolist()
    else:
      indices_list = list(indices)
    active = [
        (int(idx), pos)
        for pos, idx in enumerate(indices_list)
        if int(idx) in candidate_set
    ]
  else:
    batch_start = (
        batch_start_offsets[j]
        if batch_start_offsets is not None
        else sum(b[0].shape[0] for b in batches[:j])
    )
    active = [
        (batch_start + pos, pos)
        for pos in range(batch_size)
        if (batch_start + pos) in candidate_set
    ]
  return active


def get_batch_and_ind(batches, j, ind):
  """Returns (x_batch, y_batch, ind_batch) for batch j, removing datapoint ind if applicable."""
  x_batch, y_batch, indices = batches[j]
  if ind is None:
    return x_batch, y_batch, None

  if indices is not None:
    if isinstance(indices, (list, tuple)):
      indices = torch.tensor(indices)
    mask = indices == ind
    if mask.any():
      ind_batch = mask.nonzero(as_tuple=False)[0].item()
    else:
      ind_batch = None
  else:
    batch_start = sum(b[0].shape[0] for b in batches[:j])
    ind_batch = ind - batch_start
    if ind_batch < 0 or ind_batch >= x_batch.shape[0]:
      ind_batch = None

  return x_batch, y_batch, ind_batch


_get_batch_and_ind = get_batch_and_ind


def compute_update_limit(
    beta_1,
    beta_2,
    step
):
  """
  Computes the update limit for Adam. The idea is that the ratio can be written in the form
  (sum (a_t g_t)) / (sqrt(sum b_t g_t^2)+epsilon), so by Cauchy-Schwarz, we can bound
  (sum a_t g_t)^2 <= (sum b_tg_t^2) * (sum a_t^2 / b_t), or
  update_limit <= sqrt(sum a_t^2 / b_t) (this is before multiplying by learning rate).
  In our case, a_t = (1-beta_1) * beta_1^(t-1) / (1 - beta_1^T),
  and b_t = (1-beta_2)*beta_2^(t-1) / (1 - beta_2^T), T = number of steps.
  Therefore, we have
  update_limit**2 <= (1-beta_1)**2 / (1 - beta_2) * (1 - beta_2**T) / (1 - beta_1**T)**2
  * (1 + beta_1**2/beta_2 + beta_1**4/beta_2**2 + ... + beta_1**(2T-2)/beta_2**(T-1)),
  the last one can be simplified into (1 - r^T) / (1 - r) where r = beta_1^2 / beta_2.
  Of course, this is only in the condition where beta_1^2 < beta_2.

  Args:
    beta_1: The first beta parameter.
    beta_2: The second beta parameter.
    step: The current step.

  Returns:
    The update limit for Adam.
  """
  r = beta_1 ** 2 / beta_2
  tol = 1e-10
  factor1 = (1 - beta_2 ** step) / ((1 - beta_1 ** step) ** 2)
  factor2 = (1 - beta_1)**2 / (1 - beta_2)
  factor3 = (1 - r**step) / (1 - r)
  if r < 1 - tol:
    return np.sqrt(factor1 * factor2 * factor3)
  elif abs(r - 1) <= tol:
    return 1.00
  else:
    logging.info("Warning: r = %s > 1", r)
    return np.sqrt(factor1 * factor2 * (r**step - 1) / (r - 1))

def compute_metadata_adam_forward(
    x_batch,
    y_batch,
    model,
    loss_fn,
    momentum,
    variance,
    lr,
    beta_1,
    beta_2,
    eps,
    step,
    device = 'cpu',
):
  """Forward step: given that we have the metadata at time t, compute the same for time t + 1."""
  device_obj = torch.device(device)
  model = model.to(device_obj)
  x_batch = x_batch.to(device_obj)
  y_batch = y_batch.to(device_obj)
  momentum = [m.to(device_obj) for m in momentum]
  variance = [v.to(device_obj) for v in variance]

  gradient = compute_gradient(
      x_batch, y_batch, model, loss_fn, create_graph=False, retain_graph=False
  )
  gradient_detached = [g.detach() for g in gradient]

  momentum_new = [
      (beta_1 * m + (1 - beta_1) * g).detach()
      for m, g in zip(momentum, gradient_detached)
  ]
  momentum_hat = [m / (1 - beta_1**step) for m in momentum_new]
  variance_new = [
      (beta_2 * v + (1 - beta_2) * (g**2)).detach()
      for v, g in zip(variance, gradient_detached)
  ]
  variance_hat = [v / (1 - beta_2**step) for v in variance_new]

  theta_old = list(model.parameters())
  theta_new = [
      (p - lr * m_h / (torch.sqrt(v_h) + eps)).detach()
      for p, m_h, v_h in zip(theta_old, momentum_hat, variance_hat)
  ]
  model.zero_grad(set_to_none=True)
  for p in model.parameters():
    p.requires_grad_(True)
  del gradient, gradient_detached
  return theta_new, momentum_new, variance_new


def update_model_parameters(
    model, params
):
  """In-place updates model parameters with a sequence of parameter tensors."""
  with torch.no_grad():
    for p, v in zip(model.parameters(), params):
      p.copy_(v)


def compute_metadata_adam_backward(
    x_batch,
    y_batch,
    model,
    loss_fn,
    momentum,
    variance,
    lr,
    beta_1,
    beta_2,
    eps,
    step,
    device = 'cpu',
    momentum_buffer = None,
    variance_buffer = None,
    scale = 1000000,
):
  """Backward step: given that we have the metadata at time t + 1, compute the same for time t."""
  device_obj = torch.device(device)
  model = model.to(device_obj)
  x_batch = x_batch.to(device_obj)
  y_batch = y_batch.to(device_obj)
  momentum = [m.to(device_obj) for m in momentum]
  variance = [v.to(device_obj) for v in variance]

  split_sizes = [p.numel() for p in model.parameters()]

  with torch.no_grad():
    bc1 = 1.0 - (beta_1 ** step)
    bc2 = 1.0 - (beta_2 ** step)
    momentum_hat = [m / bc1 for m in momentum]
    variance_hat = [v / bc2 for v in variance]
    theta_current = list(model.parameters())
    update_limit = compute_update_limit(beta_1, beta_2, step)
    momentum_variance_ratio = [
        m_h / (torch.sqrt(v_h) + eps)
        for m_h, v_h in zip(momentum_hat, variance_hat)
    ]
    # We only clamp if we're not using the reversible transform.
    if momentum_buffer is None or variance_buffer is None:
      update_vec = [
          torch.round(
              (torch.clamp(r, -update_limit, update_limit) * lr) * scale
          )
          / scale
          for r in momentum_variance_ratio
      ]
      theta_prev = [
          (p + u).detach() for p, u in zip(theta_current, update_vec)
      ]
    else:
      update_vec_int = [
          torch.round(
              torch.clamp(r.to(torch.float64), -update_limit, update_limit)
              * lr
              * scale
          ).to(torch.int64)
          for r in momentum_variance_ratio
      ]
      theta_current_int = [
          torch.round(p.to(torch.float64) * scale).to(torch.int64)
          for p in theta_current
      ]
      theta_prev_int = [
          p + u for p, u in zip(theta_current_int, update_vec_int)
      ]
      theta_prev = [
          p_int.to(p_orig.dtype) / scale
          for p_int, p_orig in zip(theta_prev_int, theta_current)
      ]
      update_vec = [
          u_int.to(p_orig.dtype) / scale
          for u_int, p_orig in zip(update_vec_int, theta_current)
      ]
    for t in update_vec:
      if torch.isnan(t).any():
        raise ValueError(f'update_vec contains NaN at step {step}')
    for t in theta_prev:
      if torch.isnan(t).any():
        raise ValueError(f'theta_prev contains NaN at step {step}')
  model_prev = copy.deepcopy(model)
  update_model_parameters(model_prev, theta_prev)
  for p in model_prev.parameters():
    if torch.isnan(p).any():
      raise ValueError(f'model_prev contains NaN at step {step}')
  gradient_prev = compute_gradient(
      x_batch,
      y_batch,
      model_prev,
      loss_fn,
      create_graph=False,
      retain_graph=False,
  )
  for g in gradient_prev:
    if torch.isnan(g).any():
      raise ValueError(f'gradient_prev contains NaN at step {step}')

  gradient_prev_detached = [g.detach() for g in gradient_prev]
  gradient_prev_flat = torch.cat(
      [g.flatten() for g in gradient_prev_detached]
  ).to(torch.float64)

  if momentum_buffer is not None:
    momentum_flat = torch.cat([m.flatten() for m in momentum])
    scaled_momentum_t = torch.round(
        momentum_flat.to(torch.float64) * scale
    ).to(torch.int64)
    grad_scaled_m = torch.round(
        (1.0 - beta_1) * gradient_prev_flat * scale
    ).to(torch.int64)
    target_m_int = scaled_momentum_t - grad_scaled_m
    momentum_prev_int = momentum_buffer.backward_decay(target_m_int)
    momentum_prev_flat = momentum_prev_int.to(torch.float64) / scale
    momentum_prev = [
        t.view(p.shape).to(device=device_obj, dtype=p.dtype)
        for t, p in zip(
            torch.split(momentum_prev_flat, split_sizes),
            model_prev.parameters(),
        )
    ]
    for m in momentum_prev:
      if torch.isnan(m).any():
        raise ValueError(f'momentum_prev contains NaN at step {step}')
  else:
    momentum_prev = [
        ((m - (1 - beta_1) * g) / beta_1).detach()
        for m, g in zip(momentum, gradient_prev_detached)
    ]

  if variance_buffer is not None:
    variance_flat = torch.cat([v.flatten() for v in variance])
    scaled_variance_t = torch.round(
        variance_flat.to(torch.float64) * scale
    ).to(torch.int64)
    grad_scaled_v = torch.ceil(
        (1.0 - beta_2) * (gradient_prev_flat**2) * scale
    ).to(torch.int64)
    target_v_int = torch.clamp_min(scaled_variance_t - grad_scaled_v, 0)
    variance_prev_int = variance_buffer.backward_decay(target_v_int)
    variance_prev_flat = torch.clamp(
        variance_prev_int.to(torch.float64) / scale, min=0.0
    )
    variance_prev = [
        t.view(p.shape).to(device=device_obj, dtype=p.dtype)
        for t, p in zip(
            torch.split(variance_prev_flat, split_sizes),
            model_prev.parameters(),
        )
    ]
    for v in variance_prev:
      if torch.isnan(v).any():
        raise ValueError(f'variance_prev contains NaN at step {step}')
  else:
    variance_prev = [
        torch.clamp(
            (v - (1 - beta_2) * (g**2)) / beta_2, min=eps
        ).detach()
        for v, g in zip(variance, gradient_prev_detached)
    ]

  model_prev.zero_grad(set_to_none=True)
  for p in model_prev.parameters():
    p.requires_grad_(True)
  del gradient_prev, gradient_prev_detached
  return model_prev, momentum_prev, variance_prev


def compute_validation_gradient(
    model,
    criterion = nn.functional.cross_entropy,
    val_loader = None,
    device = 'cpu',
    flatten = False,
):
  """Computes average gradient over the validation dataset.

  Args:
    model: The PyTorch neural network module.
    criterion: Loss function.
    val_loader: Validation DataLoader or sequence of batches.
    device: Target computation device.
    flatten: If True, returns a single flattened 1D tensor of all gradients.

  Returns:
    List of gradient tensors per parameter (or single 1D tensor if flatten=True).
  """
  device_obj = torch.device(device)
  model = model.to(device_obj)
  model.eval()
  u = [torch.zeros_like(p, device=device_obj) for p in model.parameters()]
  total_samples = 0

  val_batches = prepare_batches(val_loader)
  for x_val, y_val, _ in val_batches:
    param_dtype = next(model.parameters()).dtype
    x_val = x_val.to(device=device_obj, dtype=param_dtype)
    y_val = y_val.to(device_obj)

    batch_size = x_val.shape[0]
    total_samples += batch_size

    out = model(x_val)
    loss = criterion(out, y_val) * batch_size
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
      for j, p in enumerate(model.parameters()):
        if p.grad is not None:
          u[j] += p.grad.data

  if total_samples > 0:
    for j in range(len(u)):
      u[j] /= total_samples

  if flatten:
    return torch.cat([g.flatten() for g in u]).detach()
  return u


def compute_gradient(
    x_batch,
    y_batch,
    model,
    loss_fn,
    ind_remove = None,
    create_graph = True,
    retain_graph = True,
):
  """Computes the gradient of the loss w.r.t. the parameters of the model.

  Optionally removes one datapoint from the batch.

  Args:
    x_batch: Batch of input data.
    y_batch: Batch of target data.
    model: The model to use for prediction.
    loss_fn: The loss function to use.
    ind_remove: The index of the datapoint to remove.
    create_graph: Whether to create autograd graph.
    retain_graph: Whether to retain autograd graph.

  Returns:
    The gradient of the loss w.r.t. the parameters of the model.
  """
  params = list(model.parameters())
  if params:
    device = params[0].device
    dtype = params[0].dtype
    x_batch = x_batch.to(device=device, dtype=dtype)
    y_batch = y_batch.to(device)

  for p in params:
    if torch.isnan(p).any():
      raise ValueError(f'params contains NaN before compute_gradient')

  scale = 1.00
  batch_size = x_batch.shape[0]
  if ind_remove is not None:
    x_batch = torch.cat([x_batch[:ind_remove], x_batch[ind_remove + 1 :]])
    y_batch = torch.cat([y_batch[:ind_remove], y_batch[ind_remove + 1 :]])
    scale = 1 - 1.0 / batch_size
  with sdpa_math_context(enable=create_graph):
    y_hat = model(x_batch)
    loss = loss_fn(y_hat, y_batch) * scale
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    for g in grads:
      if torch.isnan(g).any():
        raise ValueError(f'grads contains NaN')

  return grads


def compute_hessian_from_gradient(
    model_or_params,
    gradient,
    retain_graph = True,
):
  """Computes the Hessian of the loss w.r.t. parameters, given the gradient.

  Args:
    model_or_params: The model or sequence of parameter tensors.
    gradient: Sequence of gradient tensors w.r.t. model parameters.
    retain_graph: Whether to retain the autograd computational graph.

  Returns:
    The Hessian tensor.
  """
  if isinstance(model_or_params, nn.Module):
    params = list(model_or_params.parameters())
  else:
    params = list(model_or_params)

  flat_grads = torch.cat([g.flatten() for g in gradient])
  num_params = flat_grads.shape[0]
  hessian = torch.zeros((num_params, num_params), device=flat_grads.device)
  for i in range(num_params):
    keep = retain_graph or (i < num_params - 1)
    grad_i = torch.autograd.grad(flat_grads[i], params, retain_graph=keep)
    hessian[i] = torch.cat([g.flatten() for g in grad_i]).detach()
  return hessian.detach()


def compute_hessian_prod_from_gradient(
    model_or_params,
    gradient,
    vec,
    nan_impute_value = 0.0,
    retain_graph = True,
):
  """Computes Hessian-vector product of the loss w.r.t. model parameters.

  This is done by taking the dot product of the gradient with the vector, and
  then computing the gradient of the dot product with respect to the parameters.

  Args:
    model_or_params: The model or sequence of parameter tensors.
    gradient: Sequence of gradient tensors w.r.t. model parameters.
    vec: Vector to multiply with Hessian.
    nan_impute_value: Constant to impute if the Hessian-vector product contains
      NaNs after autograd. Defaults to 0.0.
    random_mask: Optional random mask to apply to the gradients and vector.
    retain_graph: Whether to retain the autograd computational graph.

  Returns:
    The Hessian-vector product tensor.
  """
  if isinstance(model_or_params, nn.Module):
    params = list(model_or_params.parameters())
  else:
    params = list(model_or_params)

  flat_grads = torch.cat([g.flatten() for g in gradient])
  vec = vec.to(dtype=flat_grads.dtype, device=flat_grads.device)
  prod = torch.dot(flat_grads, vec)
  vec_list = torch.autograd.grad(prod, params, retain_graph=retain_graph)
  h_prod = torch.cat([v.flatten() for v in vec_list])
  if nan_impute_value is not None:
    h_prod = torch.nan_to_num(
        h_prod,
        nan=nan_impute_value,
        posinf=nan_impute_value,
        neginf=nan_impute_value,
    )
  return h_prod.detach()


def generate_random_mask(
    model_or_params,
    prob,
    seed = None,
    device = None,
):
  """Generates a random boolean mask for model parameters with Bernoulli(prob).

  Args:
    model_or_params: A PyTorch nn.Module or sequence of parameter tensors.
    prob: Float Bernoulli probability p in [0.0, 1.0] of keeping each parameter
      coordinate.
    seed: Optional random seed for reproducibility.
    device: Target device for mask tensors.

  Returns:
    List of boolean torch.Tensor masks with signature [b for b in
    model.parameters()].
  """
  params = (
      list(model_or_params.parameters())
      if isinstance(model_or_params, nn.Module)
      else list(model_or_params)
  )
  gen = None
  if seed is not None:
    gen = torch.Generator()
    gen.manual_seed(seed)
  mask = []
  for p in params:
    dev = device if device is not None else p.device
    if gen is not None:
      r = (torch.rand(p.shape, generator=gen) < prob).to(dev)
    else:
      r = torch.rand(p.shape, device=dev) < prob
    mask.append(r)
  return mask


def get_grad_diff(
    x_batch,
    y_batch,
    model,
    loss_func,
    theta_diff,
    ind_remove,
    first_order = False,
    use_pearlmutter = False,
    random_mask = None,
):
  """Computes the gradient difference for removed datapoints.

  That is, using the formula g_t^{[-j]} - g_t = H_t (theta_t^{[-j]} - theta_t) -
  1/|B_t| * g_t(z_j) * 1[z_j in B_t].

  Supports evaluating m counterfactual indices at once in a vectorized fashion.

  Args:
    x_batch: Batch of input data.
    y_batch: Batch of target data.
    model: The model to use for prediction.
    loss_func: The loss function to use.
    theta_diff: List of parameter tensors or 2D tensor (m x p) / (m x p')
      representing the difference in parameters.
    ind_remove: The index or sequence of indices of datapoints to remove.
    first_order: Whether to use first-order approximation for the Hessian and
      gradient.
    use_pearlmutter: Whether to use Pearlmutter's method for computing the
      Hessian-vector product.
    random_mask: Optional random mask to apply to the gradients.
      Note: if random_mask is not None, then theta_diff must have also been
        masked.

  Returns:
    Tuple of (grads_list, diff_grad) tensors (or lists if theta_diff was a list
    and m=1).
  """
  params = list(model.parameters())
  if not params:
    raise ValueError('Model has no parameters.')
  device = params[0].device
  x_batch = x_batch.to(device)
  y_batch = y_batch.to(device)
  batch_size = x_batch.shape[0]

  mask_flat = None
  if random_mask is not None:
    if isinstance(random_mask, torch.Tensor):
      mask_flat = random_mask.to(device).bool()
    else:
      mask_flat = torch.cat(
          [r.to(device).flatten() for r in random_mask]
      ).bool()

  is_list_input = isinstance(theta_diff, list)
  if is_list_input:
    theta_diff_flat = torch.cat(
        [u.to(device).flatten() for u in theta_diff]
    ).unsqueeze(0)
  elif isinstance(theta_diff, torch.Tensor):
    if theta_diff.dim() == 1:
      theta_diff_flat = theta_diff.to(device).unsqueeze(0)
    else:
      theta_diff_flat = theta_diff.to(device)
  else:
    raise TypeError(f'Unsupported theta_diff type: {type(theta_diff)}')

  if ind_remove is None:
    ind_remove_list = [None] * theta_diff_flat.shape[0]
  elif isinstance(ind_remove, int):
    ind_remove_list = [ind_remove] * theta_diff_flat.shape[0]
  else:
    ind_remove_list = list(ind_remove)

  m = max(theta_diff_flat.shape[0], len(ind_remove_list))
  if theta_diff_flat.shape[0] == 1 and m > 1:
    theta_diff_flat = theta_diff_flat.expand(m, -1)
  elif len(ind_remove_list) == 1 and m > 1:
    ind_remove_list = ind_remove_list * m

  # Compute base gradient on full batch (1 graph)
  base_grads = compute_gradient(
      x_batch,
      y_batch,
      model,
      loss_func,
      None,
      create_graph=True,
      retain_graph=True,
  )
  flat_base_grad = torch.cat([g.flatten() for g in base_grads])
  if mask_flat is not None:
    flat_base_grad_masked = flat_base_grad[mask_flat]
  else:
    flat_base_grad_masked = flat_base_grad

  # Identify unique indices within this batch to remove
  unique_inds = {
      idx
      for idx in ind_remove_list
      if idx is not None and 0 <= idx < batch_size
  }

  grads_remove_dict = {}
  if unique_inds:
    # If not first_order, we need create_graph=True for removed points so H_t^{[-j]} = H_t - (1/|B_t|) \nabla^2 \ell(z_j)
    # If first_order, create_graph=False is sufficient since only 1st-order subtraction is performed.
    need_graph = not first_order
    for idx in unique_inds:
      input_remove = x_batch[idx : idx + 1]
      target_remove = y_batch[idx : idx + 1]
      loss_batch_remove = loss_func(model(input_remove), target_remove)
      grads_remove = torch.autograd.grad(
          loss_batch_remove,
          params,
          create_graph=need_graph,
          retain_graph=need_graph,
      )
      flat_gr = torch.cat([g.flatten() for g in grads_remove])
      if mask_flat is not None:
        flat_gr = flat_gr[mask_flat]
      grads_remove_dict[idx] = flat_gr

  if use_pearlmutter:
    hvp_list = []
    for k in range(m):
      idx = ind_remove_list[k]
      if not first_order and idx is not None and idx in grads_remove_dict:
        g_k = (
            flat_base_grad_masked - (1.0 / batch_size) * grads_remove_dict[idx]
        )
      else:
        g_k = flat_base_grad_masked
      prod_k = (theta_diff_flat[k] * g_k).sum()
      vec_list_k = torch.autograd.grad(prod_k, params, retain_graph=(k < m - 1))
      flat_hvp_k = torch.cat([v.flatten() for v in vec_list_k])
      hvp_list.append(flat_hvp_k)

    hvp_mat = torch.stack(hvp_list, dim=0)
    if mask_flat is not None:
      diff_grad = hvp_mat[:, mask_flat].detach()
    else:
      diff_grad = hvp_mat.detach()
    del hvp_list, hvp_mat
  else:
    if (
        not first_order
        and len(ind_remove_list) > 0
        and ind_remove_list[0] is not None
    ):
      base_grad_for_hess = compute_gradient(
          x_batch,
          y_batch,
          model,
          loss_func,
          ind_remove_list[0],
          create_graph=True,
      )
    else:
      base_grad_for_hess = base_grads
    explicit_hessian = compute_hessian_from_gradient(
        model, base_grad_for_hess, retain_graph=False
    )
    if mask_flat is not None:
      sub_hessian = explicit_hessian[mask_flat][:, mask_flat]
      diff_grad = (theta_diff_flat @ sub_hessian.T).detach()
    else:
      diff_grad = (theta_diff_flat @ explicit_hessian.T).detach()

  # Subtract the first-order removal term: -(1/|B_t|) * g_t(z_j)
  if unique_inds:
    diff_grad = diff_grad.clone()
    for k in range(m):
      idx = ind_remove_list[k]
      if idx is not None and idx in grads_remove_dict:
        diff_grad[k] = (
            diff_grad[k] - grads_remove_dict[idx].detach() / batch_size
        )

  if isinstance(model, nn.Module):
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
      p.requires_grad_(True)

  grads_return = flat_base_grad_masked.unsqueeze(0).expand(m, -1).detach()
  del base_grads, flat_base_grad, flat_base_grad_masked, grads_remove_dict

  if is_list_input and m == 1:
    if random_mask is None:
      split_sizes = [p.numel() for p in params]
      grads_return_list = [
          g.reshape(p.shape).clone()
          for g, p in zip(torch.split(grads_return[0], split_sizes), params)
      ]
      diff_grad_list = [
          d.reshape(p.shape).clone()
          for d, p in zip(torch.split(diff_grad[0], split_sizes), params)
      ]
      del grads_return, diff_grad, theta_diff_flat
      return grads_return_list, diff_grad_list

    else:
      if isinstance(random_mask, torch.Tensor):
        split_sizes = [int(random_mask.sum().item())]
      else:
        split_sizes = [int(r.sum().item()) for r in random_mask]
      grads_return_list = [
          g.clone() for g in torch.split(grads_return[0], split_sizes)
      ]
      diff_grad_list = [
          d.clone() for d in torch.split(diff_grad[0], split_sizes)
      ]
      del grads_return, diff_grad, theta_diff_flat
      return grads_return_list, diff_grad_list

  del theta_diff_flat
  return grads_return, diff_grad


def add_label_noise(
    labels,
    noise_rate,
    num_classes = 10,
    seed = None,
):
  """Randomly flips labels for a subset of the dataset.

  Args:
    labels: Input labels as numpy array or PyTorch tensor of shape (N,).
    noise_rate: Fraction of labels to flip (between 0.0 and 1.0).
    num_classes: Total number of target classes.
    seed: Optional random seed for reproducibility.

  Returns:
    Tuple of (corrupted_labels, corrupted_indices):
      corrupted_labels: Tensor or array with corrupted labels.
      corrupted_indices: 1D numpy array of indices that were modified.
  """
  if noise_rate <= 0.0:
    return labels, np.array([], dtype=int)

  is_torch = isinstance(labels, torch.Tensor)
  device = None
  dtype = None
  if is_torch:
    labels_np = labels.cpu().numpy().copy()
    device = labels.device
    dtype = labels.dtype
  else:
    labels_np = labels.copy()

  n_samples = len(labels_np)
  num_corrupt = int(round(noise_rate * n_samples))
  if num_corrupt <= 0:
    return labels, np.array([], dtype=int)

  rng = np.random.default_rng(seed)
  corrupted_indices = rng.choice(n_samples, size=num_corrupt, replace=False)
  corrupted_indices = np.sort(corrupted_indices)

  for idx in corrupted_indices:
    old_label = int(labels_np[idx])
    if num_classes > 1:
      possible_classes = [c for c in range(num_classes) if c != old_label]
      labels_np[idx] = rng.choice(possible_classes)

  if is_torch:
    return (
        torch.from_numpy(labels_np).to(device=device, dtype=dtype),
        corrupted_indices,
    )
  return labels_np, corrupted_indices


def add_gaussian_noise(
    inputs,
    noise_rate,
    std = 0.1,
    clip_min = 0.0,
    clip_max = 1.0,
    seed = None,
):
  """Adds Gaussian noise to a subset of the inputs.

  Args:
    inputs: Input images/features as numpy array or PyTorch tensor of shape (N,
      ...).
    noise_rate: Fraction of samples to add noise to (between 0.0 and 1.0).
    std: Standard deviation of Gaussian noise.
    clip_min: Optional minimum value for clipping.
    clip_max: Optional maximum value for clipping.
    seed: Optional random seed for reproducibility.

  Returns:
    Tuple of (corrupted_inputs, corrupted_indices):
      corrupted_inputs: Tensor or array with corrupted inputs.
      corrupted_indices: 1D numpy array of indices that were modified.
  """
  if noise_rate <= 0.0 or std <= 0.0:
    return inputs, np.array([], dtype=int)

  is_torch = isinstance(inputs, torch.Tensor)
  device = None
  dtype = None
  if is_torch:
    inputs_np = inputs.cpu().numpy().copy()
    device = inputs.device
    dtype = inputs.dtype
  else:
    inputs_np = inputs.copy()

  n_samples = len(inputs_np)
  num_corrupt = int(round(noise_rate * n_samples))
  if num_corrupt <= 0:
    return inputs, np.array([], dtype=int)

  rng = np.random.default_rng(seed)
  corrupted_indices = rng.choice(n_samples, size=num_corrupt, replace=False)
  corrupted_indices = np.sort(corrupted_indices)

  for idx in corrupted_indices:
    noise = rng.normal(loc=0.0, scale=std, size=inputs_np[idx].shape).astype(
        inputs_np.dtype
    )
    inputs_np[idx] = inputs_np[idx] + noise
    if clip_min is not None or clip_max is not None:
      inputs_np[idx] = np.clip(
          inputs_np[idx],
          a_min=clip_min if clip_min is not None else -np.inf,
          a_max=clip_max if clip_max is not None else np.inf,
      )

  if is_torch:
    return (
        torch.from_numpy(inputs_np).to(device=device, dtype=dtype),
        corrupted_indices,
    )
  return inputs_np, corrupted_indices

# Add high/low pass filters.
def apply_pass_filter(
    inputs,
    noise_rate,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    clip_min = 0.0,
    clip_max = 1.0,
    seed = None,
    btype = 'lowpass',
):
  """Applies a low-pass or high-pass Butterworth filter to a subset of inputs.

  Args:
    inputs: Input images/features as numpy array or PyTorch tensor of shape (N,
      ...).
    noise_rate: Fraction of samples to filter (between 0.0 and 1.0).
    cutoff_freq: Cutoff frequency for the Butterworth filter (must be <
      sampling_rate / 2).
    filter_order: Order of the Butterworth filter.
    sampling_rate: Sampling frequency.
    clip_min: Optional minimum value for clipping.
    clip_max: Optional maximum value for clipping.
    seed: Optional random seed for sample corruption selection.
    btype: Filter type, either 'lowpass' (or 'low') or 'highpass' (or 'high').

  Returns:
    Tuple of (corrupted_inputs, corrupted_indices):
      corrupted_inputs: Tensor or array with corrupted inputs.
      corrupted_indices: 1D numpy array of indices that were modified.
  """
  if noise_rate <= 0.0:
    return inputs, np.array([], dtype=int)

  is_torch = isinstance(inputs, torch.Tensor)
  device = None
  dtype = None
  if is_torch:
    inputs_np = inputs.cpu().numpy().copy()
    device = inputs.device
    dtype = inputs.dtype
  else:
    inputs_np = inputs.copy()

  n_samples = len(inputs_np)
  num_corrupt = int(round(noise_rate * n_samples))
  if num_corrupt <= 0:
    return inputs, np.array([], dtype=int)

  rng = np.random.default_rng(seed)
  corrupted_indices = rng.choice(n_samples, size=num_corrupt, replace=False)
  corrupted_indices = np.sort(corrupted_indices)

  norm_btype = 'lowpass' if 'low' in btype.lower() else 'highpass'
  sos = scipy.signal.butter(
      N=filter_order,
      Wn=cutoff_freq,
      btype=norm_btype,
      output='sos',
      fs=sampling_rate,
  )

  for idx in corrupted_indices:
    filtered = scipy.signal.sosfiltfilt(sos, inputs_np[idx], axis=-1)
    if clip_min is not None or clip_max is not None:
      filtered = np.clip(
          filtered,
          a_min=clip_min if clip_min is not None else -np.inf,
          a_max=clip_max if clip_max is not None else np.inf,
      )
    inputs_np[idx] = filtered.astype(inputs_np.dtype)

  if is_torch:
    return (
        torch.from_numpy(inputs_np).to(device=device, dtype=dtype),
        corrupted_indices,
    )
  return inputs_np, corrupted_indices


def apply_dataset_noise(
    x_train,
    y_train,
    noise_type = 'none',
    noise_rate = 0.1,
    num_classes = 10,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    clip_min = 0.0,
    clip_max = 1.0,
    seed = 42,
):
  """Applies requested noise type to a subset of the training dataset.

  Args:
    x_train: Training inputs of shape (N, ...).
    y_train: Training labels of shape (N,).
    noise_type: One of 'none', 'label_flip', 'gaussian', 'both', 'lowpass' (or
      'low_pass'), 'highpass' (or 'high_pass').
    noise_rate: Fraction of training samples to corrupt (0.0 to 1.0).
    num_classes: Number of target classes for label flipping.
    gaussian_std: Standard deviation for Gaussian noise.
    cutoff_freq: Cutoff frequency for lowpass / highpass filtering.
    filter_order: Filter order for lowpass / highpass filtering.
    sampling_rate: Sampling rate for lowpass / highpass filtering.
    clip_min: Minimum clipping value for image inputs after adding noise.
    clip_max: Maximum clipping value for image inputs after adding noise.
    seed: Random seed.

  Returns:
    Tuple of (x_train_corrupted, y_train_corrupted, corrupted_indices):
      x_train_corrupted: Inputs with noise applied if requested.
      y_train_corrupted: Labels with flipping applied if requested.
      corrupted_indices: Sorted 1D numpy array of unique indices that were
      corrupted.
  """
  noise_type = noise_type.lower()
  if noise_rate <= 0.0 or noise_type == 'none':
    return x_train, y_train, np.array([], dtype=int)

  corrupted_x_indices = np.array([], dtype=int)
  corrupted_y_indices = np.array([], dtype=int)

  x_train_corrupted = x_train
  y_train_corrupted = y_train

  if noise_type in ('label_flip', 'both'):
    y_train_corrupted, corrupted_y_indices = add_label_noise(
        labels=y_train,
        noise_rate=noise_rate,
        num_classes=num_classes,
        seed=seed,
    )

  if noise_type in ('gaussian', 'both'):
    input_seed = seed + 1000 if seed is not None else None
    x_train_corrupted, corrupted_x_indices = add_gaussian_noise(
        inputs=x_train,
        noise_rate=noise_rate,
        std=gaussian_std,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=input_seed,
    )

  if noise_type in ('lowpass', 'highpass', 'low_pass', 'high_pass'):
    input_seed = seed + 1000 if seed is not None else None
    btype = 'lowpass' if 'low' in noise_type else 'highpass'
    x_train_corrupted, corrupted_x_indices = apply_pass_filter(
        inputs=x_train,
        noise_rate=noise_rate,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=input_seed,
        btype=btype,
    )

  all_corrupted = np.unique(
      np.concatenate([corrupted_x_indices, corrupted_y_indices])
  )
  return x_train_corrupted, y_train_corrupted, np.sort(all_corrupted)
