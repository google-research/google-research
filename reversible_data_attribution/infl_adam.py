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

"""This file contains the implementation of influence function under Adam optimizer, which contains the following functionalities:

- Forward update: computing the counterfactual based on direct iteration (t -> t
+ 1).
- Recursive update: computing the counterfactual based on recursive iteration (t
-> t - 1).
"""

import copy
from typing import Any, Sequence

from absl import logging
import torch
from torch import nn

from reversible_data_attribution import base_model
from reversible_data_attribution import reversible
from reversible_data_attribution import utils

_prepare_batches = utils._prepare_batches
_get_batch_and_ind = utils._get_batch_and_ind
prepare_batches = utils.prepare_batches
get_batch_and_ind = utils.get_batch_and_ind
compute_gradient = utils.compute_gradient
compute_hessian_from_gradient = utils.compute_hessian_from_gradient
compute_hessian_prod_from_gradient = utils.compute_hessian_prod_from_gradient


def one_step_update(
    x_batch,
    y_batch,
    model,
    loss_func,
    theta_diff,
    momentum,
    momentum_diff,
    variance,
    variance_diff,
    lr,
    beta_1,
    beta_2,
    eps,
    ind,
    remove_dv = False,
    step = 1,
    dv_option = 'first_order',
    theta_option = 'first_order',
    random_mask = None,
):
  """Goes from t to t+1 for counterfactual parameters under Adam with bias-correction.

  Supports evaluating m counterfactual indices simultaneously using (m x p) / (m
  x p') tensors.

  This is done by computing Delta theta_{t+1} - Delta theta_{t} = lr * (m_{t+1}
  * Delta_j (v_{t+1}) / 2 * (sqrt(v_{t+1}) + eps)) - lr * Delta_j(m_{t+1}) /
  (sqrt(v_{t+1}) + eps) where Delta_j denotes the counterfactual difference
  under index j.

  If remove_dv is True, then the dv term is removed from the update, i.e. we
  only keep
  - lr * Delta_j(m_{t+1}) / (sqrt(v_{t+1}) + eps).

  Options to estimate changes:
  1. dv (Difference of variance):
  - 'first_order': 2 * g * dg (purely first order)
  - 'exact': 2 * g * dg + dg^2 (exact change in squared gradient)
  2. Change in theta:
  - 'first_order': Purely first order Taylor expansion
  - 'second_order': Regularized second order
  - 'exact': Exact change
  3. Remove dv term in the update or not (remove_dv=True/False).

  Args:
    x_batch: Batch of input data.
    y_batch: Batch of target data.
    model: The model to use for prediction.
    loss_func: The loss function to use.
    theta_diff: List of param tensors or 2D (m x p) tensor representing
      theta_{t}^{[-j]} - theta_{t}.
    momentum: List of parameter tensors or (1 x p) / (p,) tensor representing
      the momentum at time t + 1.
    momentum_diff: List of param tensors or 2D (m x p) tensor representing
      momentum_{t}^{[-j]} - momentum_{t}.
    variance: List of parameter tensors or (1 x p) / (p,) tensor representing
      the variance at time t + 1.
    variance_diff: List of param tensors or 2D (m x p) tensor representing
      variance_{t}^{[-j]} - variance_{t}.
    lr: Learning rate.
    beta_1: Beta 1 hyperparameter.
    beta_2: Beta 2 hyperparameter.
    eps: Epsilon value for numerical stability.
    ind: The index or sequence of indices of datapoints to remove.
    remove_dv: Whether to remove the dv term.
    step: The current step number.
    dv_option: Option for estimating change in variance ('first_order' or
      'exact').
    theta_option: Option for estimating change in theta ('first_order',
      'second_order', or 'exact').
    random_mask: Random masking to compress the gradient dimension.

  Returns:
    Tuple of (theta_diff_new, momentum_diff_new, variance_diff_new) tensors (or
    lists if list was passed).
  """
  params = list(model.parameters())
  device = params[0].device if params else x_batch.device
  is_list_input = isinstance(theta_diff, list)

  if is_list_input:
    theta_diff_mat = torch.cat(
        [t.to(device).flatten() for t in theta_diff]
    ).unsqueeze(0)
    momentum_diff_mat = torch.cat(
        [m.to(device).flatten() for m in momentum_diff]
    ).unsqueeze(0)
    variance_diff_mat = torch.cat(
        [v.to(device).flatten() for v in variance_diff]
    ).unsqueeze(0)
  else:
    theta_diff_mat = (
        theta_diff.to(device)
        if theta_diff.dim() == 2
        else theta_diff.to(device).unsqueeze(0)
    )
    momentum_diff_mat = (
        momentum_diff.to(device)
        if momentum_diff.dim() == 2
        else momentum_diff.to(device).unsqueeze(0)
    )
    variance_diff_mat = (
        variance_diff.to(device)
        if variance_diff.dim() == 2
        else variance_diff.to(device).unsqueeze(0)
    )

  if isinstance(momentum, list):
    flat_m = torch.cat([m.to(device).flatten() for m in momentum])
    flat_v = torch.cat([v.to(device).flatten() for v in variance])
    if random_mask is not None and flat_m.numel() > theta_diff_mat.shape[1]:
      if isinstance(random_mask, torch.Tensor):
        mask_flat = random_mask.to(device).bool()
      else:
        mask_flat = torch.cat(
            [r.to(device).flatten() for r in random_mask]
        ).bool()
      flat_m = flat_m[mask_flat]
      flat_v = flat_v[mask_flat]
    momentum_base = flat_m.unsqueeze(0)
    variance_base = flat_v.unsqueeze(0)
  else:
    momentum_base = (
        momentum.to(device)
        if momentum.dim() == 2
        else momentum.to(device).unsqueeze(0)
    )
    variance_base = (
        variance.to(device)
        if variance.dim() == 2
        else variance.to(device).unsqueeze(0)
    )

  gradients, diff_grad = utils.get_grad_diff(
      x_batch,
      y_batch,
      model,
      loss_func,
      theta_diff_mat,
      ind,
      first_order=(theta_option == 'first_order'),
      use_pearlmutter=True,
      random_mask=random_mask,
  )
  if isinstance(gradients, list):
    gradients = torch.cat([g.flatten() for g in gradients]).unsqueeze(0)
  if isinstance(diff_grad, list):
    diff_grad = torch.cat([d.flatten() for d in diff_grad]).unsqueeze(0)

  assert dv_option in [
      'first_order',
      'exact',
  ], f'dv_option must be first_order or exact, got {dv_option}'
  assert theta_option in ['first_order', 'second_order', 'exact'], (
      'theta_option must be first_order, second_order, or exact, got'
      f' {theta_option}'
  )

  if dv_option == 'first_order':
    diff_grad_sq = 2 * gradients * diff_grad
  else:
    diff_grad_sq = 2 * gradients * diff_grad + (diff_grad**2)

  momentum_diff_new = beta_1 * momentum_diff_mat + (1 - beta_1) * diff_grad
  variance_diff_new = beta_2 * variance_diff_mat + (1 - beta_2) * diff_grad_sq

  bc1 = 1.0 - beta_1**step
  bc2 = 1.0 - beta_2**step

  momentum_hat = momentum_base / bc1
  variance_hat = variance_base / bc2
  momentum_diff_hat = momentum_diff_new / bc1
  variance_diff_hat = variance_diff_new / bc2

  if theta_option == 'first_order':
    d1 = (
        momentum_hat
        * variance_diff_hat
        / (2 * (torch.sqrt(variance_hat) + eps) ** 3)
    )
    d2 = momentum_diff_hat / (torch.sqrt(variance_hat) + eps)
  elif theta_option == 'second_order':
    variance_cf_hat = variance_hat + variance_diff_hat
    denom = (
        (torch.sqrt(variance_hat) + eps) ** 3
        + (torch.sqrt(torch.relu(variance_cf_hat)) + eps) ** 3
        + eps
    )
    d1 = momentum_hat * variance_diff_hat / denom
    d2 = momentum_diff_hat / (torch.sqrt(torch.relu(variance_cf_hat)) + eps)
  else:
    d1 = None
    d2 = None

  if remove_dv:
    if d2 is None:
      d2 = momentum_diff_hat / (torch.sqrt(variance_hat) + eps)
    theta_additional_diff = -d2
  elif step == 0:
    momentum_hat_cf = momentum_hat + momentum_diff_hat
    variance_hat_cf = torch.relu(variance_hat + variance_diff_hat)
    theta_additional_diff = momentum_hat / (
        torch.sqrt(variance_hat) + eps
    ) - momentum_hat_cf / (torch.sqrt(variance_hat_cf) + eps)
  else:
    if theta_option in ['first_order', 'second_order']:
      theta_additional_diff = d1 - d2
    else:
      variance_cf_hat = variance_hat + variance_diff_hat
      theta_additional_diff = momentum_hat / (
          torch.sqrt(variance_hat) + eps
      ) - (momentum_hat + momentum_diff_hat) / (
          torch.sqrt(torch.relu(variance_cf_hat)) + eps
      )

  if torch.isnan(theta_additional_diff).any():
    raise ValueError('theta_additional_diff is NaN')

  theta_diff_new = (theta_diff_mat + lr * theta_additional_diff).detach()
  momentum_diff_new = momentum_diff_new.detach()
  variance_diff_new = variance_diff_new.detach()

  if is_list_input:
    if random_mask is None:
      split_sizes = [p.numel() for p in params]
      theta_ret = [
          t.reshape(p.shape).clone()
          for t, p in zip(torch.split(theta_diff_new[0], split_sizes), params)
      ]
      mom_ret = [
          m.reshape(p.shape).clone()
          for m, p in zip(
              torch.split(momentum_diff_new[0], split_sizes), params
          )
      ]
      var_ret = [
          v.reshape(p.shape).clone()
          for v, p in zip(
              torch.split(variance_diff_new[0], split_sizes), params
          )
      ]
    else:
      if isinstance(random_mask, torch.Tensor):
        split_sizes = [int(random_mask.sum().item())]
      else:
        split_sizes = [int(r.sum().item()) for r in random_mask]
      theta_ret = [
          t.clone() for t in torch.split(theta_diff_new[0], split_sizes)
      ]
      mom_ret = [
          m.clone() for m in torch.split(momentum_diff_new[0], split_sizes)
      ]
      var_ret = [
          v.clone() for v in torch.split(variance_diff_new[0], split_sizes)
      ]
    del theta_diff_new, momentum_diff_new, variance_diff_new
    del gradients, diff_grad, diff_grad_sq
    del momentum_hat, variance_hat, momentum_diff_hat, variance_diff_hat
    if d1 is not None:
      del d1
    if d2 is not None:
      del d2
    del theta_additional_diff
    return theta_ret, mom_ret, var_ret

  del gradients, diff_grad, diff_grad_sq
  del momentum_hat, variance_hat, momentum_diff_hat, variance_diff_hat
  if d1 is not None:
    del d1
  if d2 is not None:
    del d2
  del theta_additional_diff
  return theta_diff_new, momentum_diff_new, variance_diff_new


def _extract_adam_context(
    base_model_obj,
):
  """Extracts required data structures and hyperparameters from base_model."""
  if hasattr(base_model_obj, 'train_loader'):
    data_loader = base_model_obj.train_loader()
    checkpoints = base_model_obj.get_checkpoint_models()
    momentum_lst = (
        base_model_obj.get_momentum_dict()
        if hasattr(base_model_obj, 'get_momentum_dict')
        else None
    )
    variance_lst = (
        base_model_obj.get_variance_dict()
        if hasattr(base_model_obj, 'get_variance_dict')
        else None
    )
    criterion = (
        base_model_obj.loss_fn()
        if hasattr(base_model_obj, 'loss_fn')
        else None
    )
    lr = base_model_obj.lr() if hasattr(base_model_obj, 'lr') else 0.01
    beta_1 = base_model_obj.beta_1() if hasattr(base_model_obj, 'beta_1') else 0.9
    beta_2 = base_model_obj.beta_2() if hasattr(base_model_obj, 'beta_2') else 0.999
    eps = base_model_obj.eps() if hasattr(base_model_obj, 'eps') else 1e-8
  elif isinstance(base_model_obj, dict):
    checkpoints = base_model_obj.get('checkpoints', base_model_obj)
    momentum_lst = base_model_obj.get('momentum_lst')
    variance_lst = base_model_obj.get('variance_lst')
    hp = base_model_obj.get('hyperparams', {})
    lr = hp.get('lr', 0.01)
    beta_1 = hp.get('beta_1', 0.9)
    beta_2 = hp.get('beta_2', 0.999)
    eps = hp.get('eps', 1e-8)
    data_loader = base_model_obj.get('train_loader') or base_model_obj.get('batches')
    criterion = base_model_obj.get('loss_fn') or base_model_obj.get('criterion')
  else:
    raise TypeError(f'Unsupported base_model type: {type(base_model_obj)}')

  if criterion is None:
    criterion = nn.functional.cross_entropy

  return (
      data_loader,
      checkpoints,
      criterion,
      momentum_lst,
      variance_lst,
      lr,
      beta_1,
      beta_2,
      eps,
  )


def forward_update(
    base_model,
    ind = None,
    num_steps = None,
    remove_dv = False,
    dv_option = 'first_order',
    theta_option = 'first_order',
    random_mask = None,
    device = None,
):
  """Computes forward update for counterfactual parameter difference under Adam.

  Supports evaluating m counterfactual indices at once in a vectorized fashion.

  This is done by iterating from t = 0 to num_steps - 1, and computing
  theta_{t+1}^{[-j]} - theta_{t+1} using the one_step_update function.

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    ind: The index or sequence of indices of datapoints to remove.
    num_steps: The number of steps to take.
    remove_dv: Whether to remove the dv term.
    dv_option: Option for estimating change in variance ('first_order' or
      'exact').
    theta_option: Option for estimating change in theta ('first_order',
      'second_order', or 'exact').
    random_mask: Random masking to compress the gradient dimension.
    device: Computation device.

  Returns:
    Tuple of (theta_diff, momentum_diff, variance_diff) tensors (or lists if
    single index was passed).
  """
  (
      data_loader,
      checkpoints,
      criterion,
      momentum_lst,
      variance_lst,
      lr,
      beta_1,
      beta_2,
      eps,
  ) = _extract_adam_context(base_model)
  if criterion is None:
    criterion = nn.functional.cross_entropy
  batches = utils._prepare_batches(data_loader)
  num_batches = len(batches)
  if num_batches == 0:
    raise ValueError('data_loader is empty.')

  if isinstance(checkpoints, dict):
    if 0 in checkpoints:
      ckpt_0 = checkpoints[0]
    elif -1 in checkpoints and len(checkpoints) == 1:
      ckpt_0 = checkpoints[-1]
    else:
      ckpt_0 = next(iter(checkpoints.values()))
  elif isinstance(checkpoints, nn.Module):
    ckpt_0 = checkpoints
  else:
    ckpt_0 = checkpoints[0]

  if isinstance(momentum_lst, dict):
    if 0 in momentum_lst:
      m_0 = momentum_lst[0]
    elif -1 in momentum_lst and len(momentum_lst) == 1:
      m_0 = momentum_lst[-1]
    else:
      m_0 = next(iter(momentum_lst.values()))
  elif momentum_lst is not None:
    m_0 = momentum_lst[0]
  else:
    m_0 = None

  if isinstance(variance_lst, dict):
    if 0 in variance_lst:
      v_0 = variance_lst[0]
    elif -1 in variance_lst and len(variance_lst) == 1:
      v_0 = variance_lst[-1]
    else:
      v_0 = next(iter(variance_lst.values()))
  elif variance_lst is not None:
    v_0 = variance_lst[0]
  else:
    v_0 = None

  if device is None:
    device = next(ckpt_0.parameters()).device
  else:
    device = torch.device(device)

  is_batched = isinstance(ind, (list, tuple, torch.Tensor)) and not isinstance(
      ind, int
  )
  if is_batched:
    ind_list = list(ind)
    m = len(ind_list)
  elif ind is None:
    ind_list = [None]
    m = 1
  else:
    ind_list = [ind]
    m = 1

  num_params = sum(p.numel() for p in ckpt_0.parameters())
  if random_mask is not None:
    if isinstance(random_mask, torch.Tensor):
      mask_flat = random_mask.to(device).bool()
    else:
      mask_flat = torch.cat(
          [r.to(device).flatten() for r in random_mask]
      ).bool()
    p_dim = int(mask_flat.sum().item())
  else:
    mask_flat = None
    p_dim = num_params

  param_dtype = next(ckpt_0.parameters()).dtype
  elem_size = torch.empty(0, dtype=param_dtype).element_size()
  num_state_tensors = 2 if remove_dv else 3
  total_state_bytes = m * p_dim * elem_size * num_state_tensors
  max_gpu_state_bytes = 256 * 1024 * 1024  # 256 MB threshold

  # Keep batched state tensors in CPU RAM when they exceed GPU memory budget,
  # streaming GPU-sized chunks during update iterations.
  if is_batched and (
      device.type == 'cuda' and total_state_bytes > max_gpu_state_bytes
  ):
    momentum_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
    variance_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
    theta_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
  else:
    try:
      momentum_diff = torch.zeros((m, p_dim), device=device, dtype=param_dtype)
      variance_diff = torch.zeros((m, p_dim), device=device, dtype=param_dtype)
      theta_diff = torch.zeros((m, p_dim), device=device, dtype=param_dtype)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
      momentum_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
      variance_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
      theta_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)

  if num_steps is not None:
    total_steps = num_steps
  elif isinstance(base_model, dict) and base_model.get('num_steps') is not None:
    total_steps = base_model['num_steps']
  elif hasattr(base_model, 'total_steps') and callable(base_model.total_steps) and base_model.total_steps() > 0:
    total_steps = base_model.total_steps()
  elif isinstance(checkpoints, (list, tuple)) and len(checkpoints) > 1:
    total_steps = len(checkpoints) - 1
  elif isinstance(checkpoints, dict):
    pos_keys = [k for k in checkpoints.keys() if isinstance(k, int)]
    if -1 in pos_keys:
      total_steps = num_steps if num_steps is not None else (base_model.get('num_steps') if isinstance(base_model, dict) and base_model.get('num_steps') is not None else (max([k for k in pos_keys if k > 0]) if any(k > 0 for k in pos_keys) else 1))
    else:
      valid_keys = [k for k in pos_keys if k > 0]
      if valid_keys:
        total_steps = max(valid_keys)
      else:
        raise ValueError(
            'num_steps must be specified when using dict or single checkpoint in'
            ' forward_update.'
        )
  else:
    total_steps = 1

  has_all_ckpts = (
      isinstance(checkpoints, (list, tuple))
      and len(checkpoints) >= total_steps + 1
  ) or (
      isinstance(checkpoints, dict)
      and all(k in checkpoints for k in range(total_steps + 1))
  )
  has_all_momentum = (
      isinstance(momentum_lst, (list, tuple))
      and len(momentum_lst) >= total_steps + 1
  ) or (
      isinstance(momentum_lst, dict)
      and all(k in momentum_lst for k in range(total_steps + 1))
  )
  has_all_variance = (
      isinstance(variance_lst, (list, tuple))
      and len(variance_lst) >= total_steps + 1
  ) or (
      isinstance(variance_lst, dict)
      and all(k in variance_lst for k in range(total_steps + 1))
  )
  use_on_the_fly_metadata = not (
      has_all_ckpts and has_all_momentum and has_all_variance
  )

  curr_model = (
      copy.deepcopy(ckpt_0).to(device) if use_on_the_fly_metadata else None
  )
  curr_momentum = (
      (
          [m.to(device) for m in m_0]
          if m_0 is not None
          else [torch.zeros_like(p, device=device) for p in ckpt_0.parameters()]
      )
      if use_on_the_fly_metadata
      else []
  )
  curr_variance = (
      (
          [v.to(device) for v in v_0]
          if v_0 is not None
          else [torch.zeros_like(p, device=device) for p in ckpt_0.parameters()]
      )
      if use_on_the_fly_metadata
      else []
  )

  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  for t in range(total_steps):
    if t % 100 == 0:
      utils.log_cuda_memory(
          f'infl_adam.forward_update step t={t}', device=device
      )
    j = t % num_batches
    x_batch, y_batch, batch_indices = batches[j]
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    has_ckpt_t = (isinstance(checkpoints, dict) and t in checkpoints) or (
        isinstance(checkpoints, (list, tuple))
        and len(checkpoints) >= total_steps + 1
        and t < len(checkpoints)
    )

    ckpt_t = None
    was_on_cpu = False
    if use_on_the_fly_metadata:
      if has_ckpt_t and t > 0:
        ckpt_t = (
            checkpoints[t] if isinstance(checkpoints, dict) else checkpoints[t]
        )
        curr_model = copy.deepcopy(ckpt_t).to(device)
        if isinstance(momentum_lst, dict) and t in momentum_lst:
          curr_momentum = [m.to(device) for m in momentum_lst[t]]
        elif isinstance(momentum_lst, (list, tuple)) and len(momentum_lst) > t:
          curr_momentum = [m.to(device) for m in momentum_lst[t]]
        if isinstance(variance_lst, dict) and t in variance_lst:
          curr_variance = [v.to(device) for v in variance_lst[t]]
        elif isinstance(variance_lst, (list, tuple)) and len(variance_lst) > t:
          curr_variance = [v.to(device) for v in variance_lst[t]]
      model = curr_model if curr_model is not None else ckpt_0.to(device)
      theta_next, momentum_next, variance_next = (
          utils.compute_metadata_adam_forward(
              x_batch,
              y_batch,
              model,
              criterion,
              curr_momentum,
              curr_variance,
              lr,
              beta_1,
              beta_2,
              eps,
              step=t + 1,
              device=device,
          )
      )
      momentum = momentum_next
      variance = variance_next
    else:
      theta_next, momentum_next, variance_next = None, None, None
      ckpt_t = (
          checkpoints[t] if isinstance(checkpoints, dict) else checkpoints[t]
      )
      was_on_cpu = hasattr(ckpt_t, 'parameters') and any(
          p.device.type == 'cpu' for p in ckpt_t.parameters()
      )
      model = ckpt_t.to(device)
      momentum = [m.to(device) for m in momentum_lst[t + 1]]
      variance = [v.to(device) for v in variance_lst[t + 1]]

    if batch_indices is not None:
      if isinstance(batch_indices, torch.Tensor):
        batch_indices_list = batch_indices.tolist()
      else:
        batch_indices_list = list(batch_indices)
      idx_to_pos = {int(idx): pos for pos, idx in enumerate(batch_indices_list)}
      ind_batch_list = [
          idx_to_pos.get(int(idx), None) if idx is not None else None
          for idx in ind_list
      ]
    else:
      batch_start = batch_start_offsets[j]
      batch_size = x_batch.shape[0]
      ind_batch_list = [
          (int(idx) - batch_start)
          if (idx is not None and 0 <= int(idx) - batch_start < batch_size)
          else None
          for idx in ind_list
      ]
    ind_arg = ind_batch_list if is_batched else ind_batch_list[0]

    if is_batched:
      chunk_size = utils.get_dynamic_chunk_size(
          num_params=p_dim,
          dtype=theta_diff.dtype,
          device=device,
          max_memory_mb=1024.0,
          num_state_tensors=num_state_tensors,
      )

      for chunk_start in range(0, m, chunk_size):
        chunk_end = min(chunk_start + chunk_size, m)
        chunk_positions = ind_batch_list[chunk_start:chunk_end]
        chunk_theta = theta_diff[chunk_start:chunk_end].to(device)
        chunk_mom = momentum_diff[chunk_start:chunk_end].to(device)
        chunk_var = variance_diff[chunk_start:chunk_end].to(device)
        updated_theta, updated_mom, updated_var = one_step_update(
            x_batch,
            y_batch,
            model,
            criterion,
            chunk_theta,
            momentum,
            chunk_mom,
            variance,
            chunk_var,
            lr,
            beta_1,
            beta_2,
            eps,
            chunk_positions,
            remove_dv=remove_dv,
            step=t + 1,
            dv_option=dv_option,
            theta_option=theta_option,
            random_mask=random_mask,
        )
        theta_diff[chunk_start:chunk_end] = updated_theta.to(theta_diff.device)
        momentum_diff[chunk_start:chunk_end] = updated_mom.to(
            momentum_diff.device
        )
        variance_diff[chunk_start:chunk_end] = updated_var.to(
            variance_diff.device
        )
        del (
            chunk_theta,
            chunk_mom,
            chunk_var,
            updated_theta,
            updated_mom,
            updated_var,
        )
    else:
      chunk_theta = theta_diff.to(device)
      chunk_mom = momentum_diff.to(device)
      chunk_var = variance_diff.to(device)
      theta_diff, momentum_diff, variance_diff = one_step_update(
          x_batch,
          y_batch,
          model,
          criterion,
          chunk_theta,
          momentum,
          chunk_mom,
          variance,
          chunk_var,
          lr,
          beta_1,
          beta_2,
          eps,
          ind_arg,
          remove_dv=remove_dv,
          step=t + 1,
          dv_option=dv_option,
          theta_option=theta_option,
          random_mask=random_mask,
      )
      del chunk_theta, chunk_mom, chunk_var

    theta_diff = theta_diff.detach()
    momentum_diff = momentum_diff.detach()
    variance_diff = variance_diff.detach()

    if isinstance(model, nn.Module):
      model.zero_grad(set_to_none=True)
      for p in model.parameters():
        p.requires_grad_(True)

    if (
        use_on_the_fly_metadata
        and curr_model is not None
        and theta_next is not None
        and momentum_next is not None
        and variance_next is not None
        and not (isinstance(checkpoints, dict) and (t + 1) in checkpoints)
    ):
      utils.update_model_parameters(curr_model, theta_next)
      curr_momentum = momentum_next
      curr_variance = variance_next

    if was_on_cpu and ckpt_t is not None and hasattr(ckpt_t, 'to'):
      ckpt_t.to('cpu')

    del x_batch, y_batch, ind_batch_list, model
    if not use_on_the_fly_metadata:
      del momentum, variance
    if theta_next is not None:
      del theta_next

  if curr_model is not None:
    del curr_model, curr_momentum, curr_variance

  utils.garbage_collect(device=device)

  if is_batched:
    if (
        theta_diff.device.type == 'cpu'
        and total_state_bytes > max_gpu_state_bytes
    ):
      return theta_diff, momentum_diff, variance_diff
    try:
      return (
          theta_diff.to(device),
          momentum_diff.to(device),
          variance_diff.to(device),
      )
    except (torch.cuda.OutOfMemoryError, RuntimeError):
      return theta_diff, momentum_diff, variance_diff
  else:
    if random_mask is None:
      split_sizes = [p.numel() for p in ckpt_0.parameters()]
      theta_ret = [
          t.reshape(p.shape).clone()
          for t, p in zip(
              torch.split(theta_diff[0], split_sizes), ckpt_0.parameters()
          )
      ]
      mom_ret = [
          m.reshape(p.shape).clone()
          for m, p in zip(
              torch.split(momentum_diff[0], split_sizes), ckpt_0.parameters()
          )
      ]
      var_ret = [
          v.reshape(p.shape).clone()
          for v, p in zip(
              torch.split(variance_diff[0], split_sizes), ckpt_0.parameters()
          )
      ]
    else:
      if isinstance(random_mask, torch.Tensor):
        split_sizes = [int(random_mask.sum().item())]
      else:
        split_sizes = [int(r.sum().item()) for r in random_mask]
      theta_ret = [t.clone() for t in torch.split(theta_diff[0], split_sizes)]
      mom_ret = [m.clone() for m in torch.split(momentum_diff[0], split_sizes)]
      var_ret = [v.clone() for v in torch.split(variance_diff[0], split_sizes)]
    del theta_diff, momentum_diff, variance_diff
    return theta_ret, mom_ret, var_ret


def recursive_update_nodv(
    base_model,
    inds = None,
    query_vec = None,
    num_steps = None,
    gradients_lst = None,
    device = None,
    momentum_buffer = None,
    variance_buffer = None,
    reversible_scale = 1000000000000,
    start_step = 0,
):
  """Computes recursive update for counterfactual parameter difference under Adam (no dv).

  This is done by iterating from t = num_steps - 1 down to 0, by writing
  Delta theta_{t+1} in terms of Delta theta_t, gradient / momentum / variance at
  time t + 1.
  Uses the identity that Delta theta_{t} can be approximated by linear
  combination of gradients at time t' for t' <= t.

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    inds: The index or sequence of indices of datapoints to remove.
    query_vec: The query vector to use for the update.
    num_steps: The number of steps to take.
    gradients_lst: List of gradient tensors.
    device: Computation device.
    momentum_buffer: Optional reversible momentum transform.
    variance_buffer: Optional reversible variance transform.
    reversible_scale: Fixed-point scaling factor.
    start_step: Step index before which gradient effects are ignored.

  Returns:
    Dictionary mapping index to counterfactual parameter or loss change.
  """
  if momentum_buffer is not None:
    momentum_buffer = copy.deepcopy(momentum_buffer)
  if variance_buffer is not None:
    variance_buffer = copy.deepcopy(variance_buffer)

  (
      data_loader,
      checkpoints,
      criterion,
      momentum_lst,
      variance_lst,
      lr,
      beta_1,
      beta_2,
      eps,
  ) = _extract_adam_context(base_model)
  if criterion is None:
    criterion = nn.functional.cross_entropy
  batches = utils.prepare_batches(data_loader)
  num_batches = len(batches)
  if num_batches == 0:
    raise ValueError('data_loader is empty.')

  if isinstance(checkpoints, dict):
    if -1 in checkpoints:
      ckpt_T = checkpoints[-1]
    elif num_steps is not None and num_steps in checkpoints:
      ckpt_T = checkpoints[num_steps]
    else:
      pos_keys = [
          k for k in checkpoints.keys() if isinstance(k, int) and k >= 0
      ]
      ckpt_T = (
          checkpoints[max(pos_keys)]
          if pos_keys
          else next(iter(checkpoints.values()))
      )
  elif isinstance(checkpoints, nn.Module):
    ckpt_T = checkpoints
  else:
    ckpt_T = checkpoints[-1]

  if isinstance(momentum_lst, dict):
    if -1 in momentum_lst:
      m_T = momentum_lst[-1]
    elif num_steps is not None and num_steps in momentum_lst:
      m_T = momentum_lst[num_steps]
    else:
      pos_keys = [
          k for k in momentum_lst.keys() if isinstance(k, int) and k >= 0
      ]
      m_T = (
          momentum_lst[max(pos_keys)]
          if pos_keys
          else next(iter(momentum_lst.values()))
      )
  elif momentum_lst is not None:
    m_T = momentum_lst[-1]
  else:
    m_T = None

  if isinstance(variance_lst, dict):
    if -1 in variance_lst:
      v_T = variance_lst[-1]
    elif num_steps is not None and num_steps in variance_lst:
      v_T = variance_lst[num_steps]
    else:
      pos_keys = [
          k for k in variance_lst.keys() if isinstance(k, int) and k >= 0
      ]
      v_T = (
          variance_lst[max(pos_keys)]
          if pos_keys
          else next(iter(variance_lst.values()))
      )
  elif variance_lst is not None:
    v_T = variance_lst[-1]
  else:
    v_T = None

  if device is None:
    if query_vec is not None:
      device = query_vec.device
    else:
      device = next(ckpt_T.parameters()).device
  else:
    device = torch.device(device)
  num_params = sum(p.numel() for p in ckpt_T.parameters())

  if inds is None:
    ind_list = []
  elif isinstance(inds, int):
    ind_list = [inds]
  else:
    ind_list = list(inds)

  param_dtype = next(ckpt_T.parameters()).dtype
  theta_diff_dict = {
      idx: (
          torch.tensor(0.0, device=device, dtype=param_dtype)
          if query_vec is not None
          else torch.zeros(num_params, device=device, dtype=param_dtype)
      )
      for idx in ind_list
  }
  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  u1 = (
      query_vec.clone().to(device=device, dtype=param_dtype)
      if query_vec is not None
      else None
  )
  u2 = (
      torch.zeros(num_params, device=device, dtype=param_dtype)
      if query_vec is not None
      else None
  )
  n1 = (
      torch.eye(num_params, device=device, dtype=param_dtype)
      if query_vec is None
      else None
  )
  n2 = (
      torch.zeros((num_params, num_params), device=device, dtype=param_dtype)
      if query_vec is None
      else None
  )

  if num_steps is not None:
    total_steps = num_steps
  elif isinstance(checkpoints, (list, tuple)) and len(checkpoints) > 1:
    total_steps = len(checkpoints) - 1
  elif isinstance(checkpoints, dict):
    pos_keys = [k for k in checkpoints.keys() if isinstance(k, int) and k > 0]
    if pos_keys:
      total_steps = max(pos_keys)
    else:
      raise ValueError(
          'num_steps must be specified when using dict or single checkpoint in'
          ' recursive_update_nodv.'
      )
  else:
    total_steps = 1

  curr_model = copy.deepcopy(ckpt_T).to(device)
  curr_momentum = (
      [m.to(device) for m in m_T]
      if m_T is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_T.parameters()]
  )
  curr_variance = (
      [v.to(device) for v in v_T]
      if v_T is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_T.parameters()]
  )

  for t in range(total_steps - 1, start_step - 1, -1):
    if t % 100 == 0:
      utils.log_cuda_memory(
          f'infl_adam.recursive_update_nodv step t={t}', device=device
      )
    step = t + 1
    bc1 = 1.0 - beta_1**step
    bc2 = 1.0 - beta_2**step
    effective_lr = lr / bc1

    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    batch_size_t = x_batch.shape[0]
    param_dtype = next(curr_model.parameters()).dtype
    x_batch = x_batch.to(device=device, dtype=param_dtype)
    y_batch = y_batch.to(device)

    variance_curr = [v.to(device) for v in curr_variance]

    has_ckpt_t = (
        (isinstance(checkpoints, dict) and t in checkpoints)
        or (
            isinstance(checkpoints, (list, tuple))
            and len(checkpoints) >= total_steps + 1
            and t < len(checkpoints)
        )
    )
    has_mom_t = (
        momentum_lst is not None
        and (
            (isinstance(momentum_lst, dict) and t in momentum_lst)
            or (
                isinstance(momentum_lst, (list, tuple))
                and len(momentum_lst) >= total_steps + 1
                and t < len(momentum_lst)
            )
        )
    )
    has_var_t = (
        variance_lst is not None
        and (
            (isinstance(variance_lst, dict) and t in variance_lst)
            or (
                isinstance(variance_lst, (list, tuple))
                and len(variance_lst) >= total_steps + 1
                and t < len(variance_lst)
            )
        )
    )

    if (
        momentum_buffer is None
        and variance_buffer is None
        and has_ckpt_t
        and has_mom_t
        and has_var_t
    ):
      model_prev = checkpoints[t].to(device)
      momentum_prev = [m.to(device) for m in momentum_lst[t]]
      variance_prev = [v.to(device) for v in variance_lst[t]]
      model_t = model_prev
    else:
      model_prev, momentum_prev, variance_prev = (
          utils.compute_metadata_adam_backward(
              x_batch,
              y_batch,
              curr_model,
              criterion,
              curr_momentum,
              curr_variance,
              lr,
              beta_1,
              beta_2,
              eps,
              step=step,
              device=device,
              momentum_buffer=momentum_buffer,
              variance_buffer=variance_buffer,
              scale=reversible_scale,
          )
      )
      if has_ckpt_t and has_mom_t and has_var_t:
        model_prev = checkpoints[t].to(device)
        momentum_prev = [m.to(device) for m in momentum_lst[t]]
        variance_prev = [v.to(device) for v in variance_lst[t]]
      model_t = model_prev.to(device)
      variance_prev = [v.to(device) for v in variance_prev]

    has_grad_t = (
        gradients_lst is not None
        and (
            (isinstance(gradients_lst, dict) and t in gradients_lst)
            or (
                isinstance(gradients_lst, (list, tuple))
                and t < len(gradients_lst)
            )
        )
    )
    if has_grad_t:
      gradients_list = [g.to(device) for g in gradients_lst[t]]
    else:
      gradients_list = utils.compute_gradient(
          x_batch, y_batch, model_t, criterion, None
      )

    # Cauchy-Schwarz theoretical limit on |m| / sqrt(v)
    update_limit_curr = utils.compute_update_limit(beta_1, beta_2, step)
    momentum_curr_flat = torch.cat(
        [m.contiguous().view(-1) for m in curr_momentum]
    )
    min_std_curr = momentum_curr_flat.abs() / (bc1 * update_limit_curr)
    v_curr_flat = torch.cat([v.contiguous().view(-1) for v in variance_curr])
    std_curr = torch.maximum(torch.sqrt(v_curr_flat / bc2), min_std_curr)
    s = std_curr + eps

    if u1 is not None:
      target = (effective_lr * (1 - beta_1) * (u1 / s)).detach()
    elif n1 is not None:
      target = (effective_lr * (1 - beta_1) * (n1 @ torch.diag(1 / s))).detach()
    else:
      raise RuntimeError('Neither u1 nor n1 initialized.')

    active_pairs = utils.get_batch_active_indices(
        batches, j, candidate_set, batch_start_offsets
    )
    if active_pairs:
      active_dataset_indices = [p[0] for p in active_pairs]
      active_batch_positions = [p[1] for p in active_pairs]

      num_active = len(active_batch_positions)
      # Dynamically determine chunk size based on current GPU memory & model size
      chunk_size = utils.get_dynamic_chunk_size(
          num_params=num_params,
          dtype=target.dtype,
          device=device,
          max_memory_mb=1024.0,
      )

      # Process active samples in GPU-sized chunks
      for chunk_start in range(0, num_active, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_active)
        chunk_positions = active_batch_positions[chunk_start:chunk_end]
        chunk_dataset_indices = active_dataset_indices[chunk_start:chunk_end]
        curr_chunk_len = len(chunk_positions)
        # Preallocate chunk buffer directly on GPU to avoid torch.stack overhead
        flat_grads_chunk = torch.empty(
            (curr_chunk_len, num_params),
            dtype=target.dtype,
            device=device,
        )
        for local_idx, pos in enumerate(chunk_positions):
          x_batch_remove = x_batch[pos : pos + 1]
          y_batch_remove = y_batch[pos : pos + 1]
          y_pred_batch_remove = model_t(x_batch_remove)
          loss_batch_remove = criterion(y_pred_batch_remove, y_batch_remove)
          gradients_remove = torch.autograd.grad(
              loss_batch_remove, model_t.parameters(), create_graph=False
          )
          flat_grads_chunk[local_idx] = torch.cat(
              [g.contiguous().view(-1) for g in gradients_remove]
          )
        # High-performance batched GPU matrix-vector / matrix-matrix multiplication
        if query_vec is not None:
          # (curr_chunk_len, num_params) @ (num_params,) -> (curr_chunk_len,)
          dot_products = (
              torch.mv(flat_grads_chunk, target) / batch_size_t
          ).detach()
          for idx, dot_val in zip(chunk_dataset_indices, dot_products):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + dot_val).detach()
        else:
          # (curr_chunk_len, num_params) @ (num_params, K) -> (curr_chunk_len, K)
          diff_batch = (
              flat_grads_chunk @ target.T / batch_size_t
          ).detach()
          for idx, diff_row in zip(chunk_dataset_indices, diff_batch):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + diff_row).detach()
        del flat_grads_chunk

    if t == 0 or t == start_step:
      if isinstance(model_t, nn.Module):
        model_t.zero_grad(set_to_none=True)
      del x_batch, y_batch, model_t
      break

    if t > 0:
      bc2_prev = 1.0 - beta_2**t
      s_prev = [torch.sqrt(v / bc2_prev) + eps for v in variance_prev]
    else:
      s_prev = [torch.full_like(v, eps) for v in variance_curr]

    s_prev = torch.cat([u.contiguous().view(-1) for u in s_prev])
    r = s_prev / s

    step_prev = t
    bc1_prev = 1.0 - beta_1**step_prev if step_prev > 0 else 1.0
    a_k = (bc1_prev / bc1) if t > 0 else 0.0

    retain_last_graph = gradients_lst is not None

    if u1 is not None and u2 is not None:
      hessian_prod = utils.compute_hessian_prod_from_gradient(
          model_t, gradients_list, u1 / s, retain_graph=retain_last_graph
      )
      u1_new = (
          u2
          + u1
          + beta_1 * a_k * (u1 * r)
          - effective_lr * (1 - beta_1) * hessian_prod
      )
      u2_new = -beta_1 * a_k * (u1 * r)
      u1 = u1_new
      u2 = u2_new
    elif n1 is not None and n2 is not None:
      h_mat = utils.compute_hessian_from_gradient(
          model_t, gradients_list, retain_graph=retain_last_graph
      )
      a_mat = (
          torch.eye(num_params, device=device)
          + beta_1 * a_k * torch.diag(r)
          - effective_lr * (1 - beta_1) * (torch.diag(1 / s) @ h_mat)
      )
      b_mat = -beta_1 * a_k * torch.diag(r)
      n1_new = n2 + n1 @ a_mat
      n2_new = n1 @ b_mat
      n1 = n1_new
      n2 = n2_new

    curr_model = model_prev
    curr_momentum = momentum_prev
    curr_variance = variance_prev

    if isinstance(model_t, nn.Module):
      model_t.zero_grad(set_to_none=True)
      for p in model_t.parameters():
        p.requires_grad_(True)

    del x_batch, y_batch, model_t, gradients_list
    utils.garbage_collect(device=device)

  return theta_diff_dict


def recursive_update(
    base_model,
    inds = None,
    query_vec = None,
    num_steps = None,
    gradients_lst = None,
    device = None,
    momentum_buffer = None,
    variance_buffer = None,
    reversible_scale = 1000000000000,
    start_step = 0,
    eps_hvp = 1e-4,
    clip_factor = 5.0,
    calib_steps = 100,
):
  """Computes recursive update for counterfactual parameter difference under Adam (with dv).

  This is done by iterating from t = num_steps - 1 down to 0, by writing
  Delta theta_{t+1} in terms of Delta theta_t, gradient / momentum / variance at
  time t + 1.
  Uses the identity that Delta theta_{t} can be approximated by linear
  combination of gradients at time t' for t' <= t.

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    inds: The index or sequence of indices of datapoints to remove.
    query_vec: The query vector to use for the update.
    num_steps: The number of steps to take.
    gradients_lst: List of gradient tensors.
    device: Computation device.
    momentum_buffer: Optional reversible momentum transform.
    variance_buffer: Optional reversible variance transform.
    reversible_scale: Fixed-point scaling factor.
    start_step: Step index before which gradient effects are ignored.
    eps_hvp: Curvature damping floor for second-order Hessian denominator.
    clip_factor: Headroom factor for adaptive state/HVP magnitude clipping.
    calib_steps: Number of initial backward steps to calibrate max magnitudes.

  Returns:
    Dictionary mapping index to counterfactual parameter or loss change.
  """
  if momentum_buffer is not None:
    momentum_buffer = copy.deepcopy(momentum_buffer)
  if variance_buffer is not None:
    variance_buffer = copy.deepcopy(variance_buffer)

  (
      data_loader,
      checkpoints,
      criterion,
      momentum_lst,
      variance_lst,
      lr,
      beta_1,
      beta_2,
      eps,
  ) = _extract_adam_context(base_model)
  if criterion is None:
    criterion = nn.functional.cross_entropy
  batches = utils.prepare_batches(data_loader)
  num_batches = len(batches)
  if num_batches == 0:
    raise ValueError('data_loader is empty.')

  if isinstance(checkpoints, dict):
    if -1 in checkpoints:
      ckpt_T = checkpoints[-1]
    elif num_steps is not None and num_steps in checkpoints:
      ckpt_T = checkpoints[num_steps]
    else:
      pos_keys = [
          k for k in checkpoints.keys() if isinstance(k, int) and k >= 0
      ]
      ckpt_T = (
          checkpoints[max(pos_keys)]
          if pos_keys
          else next(iter(checkpoints.values()))
      )
  elif isinstance(checkpoints, nn.Module):
    ckpt_T = checkpoints
  else:
    ckpt_T = checkpoints[-1]

  if isinstance(momentum_lst, dict):
    if -1 in momentum_lst:
      m_T = momentum_lst[-1]
    elif num_steps is not None and num_steps in momentum_lst:
      m_T = momentum_lst[num_steps]
    else:
      pos_keys = [
          k for k in momentum_lst.keys() if isinstance(k, int) and k >= 0
      ]
      m_T = (
          momentum_lst[max(pos_keys)]
          if pos_keys
          else next(iter(momentum_lst.values()))
      )
  elif momentum_lst is not None:
    m_T = momentum_lst[-1]
  else:
    m_T = None

  if isinstance(variance_lst, dict):
    if -1 in variance_lst:
      v_T = variance_lst[-1]
    elif num_steps is not None and num_steps in variance_lst:
      v_T = variance_lst[num_steps]
    else:
      pos_keys = [
          k for k in variance_lst.keys() if isinstance(k, int) and k >= 0
      ]
      v_T = (
          variance_lst[max(pos_keys)]
          if pos_keys
          else next(iter(variance_lst.values()))
      )
  elif variance_lst is not None:
    v_T = variance_lst[-1]
  else:
    v_T = None

  if device is None:
    if query_vec is not None:
      device = query_vec.device
    else:
      device = next(ckpt_T.parameters()).device
  else:
    device = torch.device(device)
  num_params = sum(p.numel() for p in ckpt_T.parameters())

  if inds is None:
    ind_list = []
  elif isinstance(inds, int):
    ind_list = [inds]
  else:
    ind_list = list(inds)

  param_dtype = next(ckpt_T.parameters()).dtype
  theta_diff_dict = {
      idx: (
          torch.tensor(0.0, device=device, dtype=param_dtype)
          if query_vec is not None
          else torch.zeros(num_params, device=device, dtype=param_dtype)
      )
      for idx in ind_list
  }
  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  # The following are the base cases for the recursion.
  mat_p = (
      query_vec.clone().to(device=device, dtype=param_dtype)
      if query_vec is not None
      else torch.eye(num_params, device=device, dtype=param_dtype)
  )
  mat_q = (
      torch.zeros(num_params, device=device, dtype=param_dtype)
      if query_vec is not None
      else torch.zeros((num_params, num_params), device=device, dtype=param_dtype)
  )
  mat_r = (
      torch.zeros(num_params, device=device, dtype=param_dtype)
      if query_vec is not None
      else torch.zeros((num_params, num_params), device=device, dtype=param_dtype)
  )

  if num_steps is not None:
    total_steps = num_steps
  elif isinstance(checkpoints, (list, tuple)) and len(checkpoints) > 1:
    total_steps = len(checkpoints) - 1
  elif isinstance(checkpoints, dict):
    pos_keys = [k for k in checkpoints.keys() if isinstance(k, int) and k > 0]
    if pos_keys:
      total_steps = max(pos_keys)
    else:
      raise ValueError(
          'num_steps must be specified when using dict or single checkpoint in'
          ' recursive_update.'
      )
  else:
    total_steps = 1

  curr_model = copy.deepcopy(ckpt_T).to(device)
  curr_momentum = (
      [m.to(device) for m in m_T]
      if m_T is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_T.parameters()]
  )
  curr_variance = (
      [v.to(device) for v in v_T]
      if v_T is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_T.parameters()]
  )

  max_hvp_a_calib = 0.0
  max_hvp_b_calib = 0.0
  max_mat_p_calib = 0.0
  max_mat_q_calib = 0.0
  max_mat_r_calib = 0.0

  for t in range(total_steps - 1, start_step - 1, -1):
    if t % 100 == 0:
      utils.log_cuda_memory(
          f'infl_adam.recursive_update step t={t}', device=device
      )
    # First, we get all the normalization ratios.
    step = t + 1
    bc1 = 1.0 - beta_1**step
    bc1_prev = 1.0 - beta_1 ** (step - 1)
    bc2 = 1.0 - beta_2**step
    bc2_prev = 1.0 - beta_2 ** (step - 1)
    a_k1 = (bc1_prev / bc1) if t > 0 else 0.0
    a_k2 = (bc2_prev / bc2) if t > 0 else 0.0

    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    batch_size_t = x_batch.shape[0]
    param_dtype = next(curr_model.parameters()).dtype
    x_batch = x_batch.to(device=device, dtype=param_dtype)
    y_batch = y_batch.to(device)

    momentum_curr_list = [m.to(device) for m in curr_momentum]
    momentum_curr = torch.cat(
        [m.contiguous().view(-1) for m in momentum_curr_list]
    )
    variance_curr = [v.to(device) for v in curr_variance]

    has_ckpt_t = (
        (isinstance(checkpoints, dict) and t in checkpoints)
        or (
            isinstance(checkpoints, (list, tuple))
            and len(checkpoints) >= total_steps + 1
            and t < len(checkpoints)
        )
    )
    has_mom_t = (
        momentum_lst is not None
        and (
            (isinstance(momentum_lst, dict) and t in momentum_lst)
            or (
                isinstance(momentum_lst, (list, tuple))
                and len(momentum_lst) >= total_steps + 1
                and t < len(momentum_lst)
            )
        )
    )
    has_var_t = (
        variance_lst is not None
        and (
            (isinstance(variance_lst, dict) and t in variance_lst)
            or (
                isinstance(variance_lst, (list, tuple))
                and len(variance_lst) >= total_steps + 1
                and t < len(variance_lst)
            )
        )
    )
    if (
        momentum_buffer is None
        and variance_buffer is None
        and has_ckpt_t
        and has_mom_t
        and has_var_t
    ):
      model_prev = checkpoints[t].to(device)
      momentum_prev = [m.to(device) for m in momentum_lst[t]]
      variance_prev = [v.to(device) for v in variance_lst[t]]
      model_t = model_prev
    else:
      model_prev, momentum_prev, variance_prev = (
          utils.compute_metadata_adam_backward(
              x_batch,
              y_batch,
              curr_model,
              criterion,
              curr_momentum,
              curr_variance,
              lr,
              beta_1,
              beta_2,
              eps,
              step=step,
              device=device,
              momentum_buffer=momentum_buffer,
              variance_buffer=variance_buffer,
              scale=reversible_scale,
          )
      )
      if has_ckpt_t and has_mom_t and has_var_t:
        model_prev = checkpoints[t].to(device)
        momentum_prev = [m.to(device) for m in momentum_lst[t]]
        variance_prev = [v.to(device) for v in variance_lst[t]]
      for p in model_prev.parameters():
        if torch.isnan(p).any():
          raise ValueError(f'NaN found in model_prev parameters at step {t}')
      model_t = model_prev.to(device)
      variance_prev = [v.to(device) for v in variance_prev]

    has_grad_t = (
        gradients_lst is not None
        and (
            (isinstance(gradients_lst, dict) and t in gradients_lst)
            or (
                isinstance(gradients_lst, (list, tuple))
                and t < len(gradients_lst)
            )
        )
    )
    if has_grad_t:
      gradients_list = [g.to(device) for g in gradients_lst[t]]
    else:
      gradients_list = utils.compute_gradient(
          x_batch, y_batch, model_t, criterion, None
      )

    momentum_hat = momentum_curr / bc1

    gradient_curr = torch.cat(
        [g.to(device).contiguous().view(-1) for g in gradients_list]
    )

    # Cauchy-Schwarz theoretical limit on |m| / sqrt(v)
    update_limit_curr = utils.compute_update_limit(beta_1, beta_2, step)
    min_std_curr = momentum_curr.abs() / (bc1 * update_limit_curr)
    v_curr_flat = torch.cat([v.contiguous().view(-1) for v in variance_curr])
    std_curr = torch.maximum(torch.sqrt(v_curr_flat / bc2), min_std_curr)
    s_curr = std_curr + eps

    momentum_prev_flat = torch.cat(
        [m.contiguous().view(-1) for m in momentum_prev]
    )
    v_prev_flat = torch.cat([v.contiguous().view(-1) for v in variance_prev])
    if step > 1:
      update_limit_prev = utils.compute_update_limit(beta_1, beta_2, step - 1)
      min_std_prev = momentum_prev_flat.abs() / (bc1_prev * update_limit_prev)
      std_prev = torch.maximum(torch.sqrt(v_prev_flat / bc2_prev), min_std_prev)
      s_prev = std_prev + eps
    else:
      s_prev = torch.full_like(s_curr, eps)

    s_curr_hvp = torch.clamp(s_curr, min=eps_hvp)
    s_ratio = torch.clamp(s_prev / s_curr, min=0.0, max=5.0)

    vec_a1 = a_k2 * beta_2 * (s_ratio ** 3)
    vec_a2 = a_k1 * beta_1 * s_ratio
    vec_c1 = (1 - beta_2) / bc2 * gradient_curr / (s_curr_hvp**3)
    vec_c2 = (1 - beta_1) / bc1 / s_curr
    if torch.isnan(vec_c1).any():
      raise ValueError(f'NaN found in vec_c1 at step {t}')
    if torch.isnan(vec_c2).any():
      raise ValueError(f'NaN found in vec_c2 at step {t}')
    if ((t + 1) % 100 == 0):
      logging.info('step %d', t)
      logging.info('gradient curr magnitude: %f', gradient_curr.abs().max().item())
      logging.info('vec a1 magnitude: %f', vec_a1.abs().max().item())
      logging.info('vec a2 magnitude: %f', vec_a2.abs().max().item())
      logging.info('vec c1 magnitude: %f', vec_c1.abs().max().item())
      logging.info('vec c2 magnitude: %f', vec_c2.abs().max().item())
      logging.info('momentum magnitude: %f', momentum_hat.abs().max().item())

    # These two "coefficient" vector will be used a lot later.
    gamma_1 = (
        lr * momentum_hat * mat_p + mat_q
        if query_vec is not None
        else lr * mat_p @ torch.diag(momentum_hat) + mat_q
    )
    if torch.isnan(gamma_1).any():
      raise ValueError(f'NaN found in gamma_1 at step {t}')
    gamma_2 = -lr * mat_p + mat_r
    if torch.isnan(gamma_2).any():
      raise ValueError(f'NaN found in gamma_2 at step {t}')
    if ((t + 1) % 100 == 0):
      logging.info('gamma_1 magnitude: %f', gamma_1.abs().max().item())
      logging.info('gamma_2 magnitude: %f', gamma_2.abs().max().item())
    if torch.abs(gamma_1).max().item() > 10 or torch.abs(gamma_2).max().item() > 10:
      logging.warning(
          'Gamma 1 magnitude: %f, Gamma 2 magnitude: %f, Momentum hat magnitude: %f, mat_p magnitude: %f, mat_q magnitude: %f, mat_r magnitude: %f at step %d',
          gamma_1.abs().max().item(),
          gamma_2.abs().max().item(),
          momentum_hat.abs().max().item(),
          mat_p.abs().max().item(),
          mat_q.abs().max().item(),
          mat_r.abs().max().item(),
          t,
      )

    if query_vec is not None:
      target = (-gamma_1 * vec_c1 - gamma_2 * vec_c2).detach()
      if t == total_steps - 1:
        assert torch.allclose(gamma_1, lr * query_vec * momentum_hat)
        assert torch.allclose(gamma_2, -lr * query_vec)
    else:
      target = (
          -gamma_1 @ torch.diag(vec_c1) - gamma_2 @ torch.diag(vec_c2)
      ).detach()
      if t == total_steps - 1:
        assert torch.allclose(mat_p, torch.eye(num_params, device=device))
        assert torch.allclose(
            mat_q, torch.zeros((num_params, num_params), device=device)
        )
        assert torch.allclose(
            mat_r, torch.zeros((num_params, num_params), device=device)
        )
        assert torch.allclose(gamma_1, lr * torch.diag(momentum_hat))
        assert torch.allclose(
            gamma_2, -lr * torch.eye(num_params, device=device)
        )

    active_pairs = utils.get_batch_active_indices(
        batches, j, candidate_set, batch_start_offsets
    )
    if active_pairs:
      active_dataset_indices = [p[0] for p in active_pairs]
      active_batch_positions = [p[1] for p in active_pairs]

      num_active = len(active_batch_positions)
      # Dynamically determine chunk size based on current GPU memory & model size
      chunk_size = utils.get_dynamic_chunk_size(
          num_params=num_params,
          dtype=target.dtype,
          device=device,
          max_memory_mb=1024.0,
      )

      # Process active samples in GPU-sized chunks
      for chunk_start in range(0, num_active, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_active)
        chunk_positions = active_batch_positions[chunk_start:chunk_end]
        chunk_dataset_indices = active_dataset_indices[chunk_start:chunk_end]
        curr_chunk_len = len(chunk_positions)
        # Preallocate chunk buffer directly on GPU to avoid torch.stack overhead
        flat_grads_chunk = torch.empty(
            (curr_chunk_len, num_params),
            dtype=target.dtype,
            device=device,
        )
        for local_idx, pos in enumerate(chunk_positions):
          # print(local_idx, curr_chunk_len)
          x_batch_remove = x_batch[pos : pos + 1]
          y_batch_remove = y_batch[pos : pos + 1]
          y_pred_batch_remove = model_t(x_batch_remove)
          loss_batch_remove = criterion(y_pred_batch_remove, y_batch_remove)
          gradients_remove = torch.autograd.grad(
              loss_batch_remove, model_t.parameters(), create_graph=False
          )
          flat_grads_chunk[local_idx] = torch.cat(
              [g.contiguous().view(-1) for g in gradients_remove]
          )
          if torch.isnan(flat_grads_chunk[local_idx]).any():
            raise ValueError(
                f'NaN found in flat_grads_chunk at step {t}, chunk {chunk_start},'
                f' local_idx {local_idx}'
            )

        # High-performance batched GPU matrix-vector / matrix-matrix multiplication
        if query_vec is not None:
          # (curr_chunk_len, num_params) @ (num_params,) -> (curr_chunk_len,)
          dot_products = (
              torch.mv(flat_grads_chunk, target) / batch_size_t
          ).detach()
          if torch.isnan(dot_products).any():
            print(f'NaN found in dot_products at step {t}, chunk {chunk_start}')
            print(flat_grads_chunk.min(), flat_grads_chunk.max())
            print(target.min(), target.max())
            logging.info(
                'NaN found in dot_products at step %d, chunk %d',
                t,
                chunk_start,
            )
            raise ValueError(f'NaN found in dot_products at step {t}, chunk {chunk_start}')
          for idx, dot_val in zip(chunk_dataset_indices, dot_products):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + dot_val).detach()
        else:
          # (curr_chunk_len, num_params) @ (num_params, K) -> (curr_chunk_len, K)
          diff_batch = (
              flat_grads_chunk @ target.T / batch_size_t
          ).detach()
          for idx, diff_row in zip(chunk_dataset_indices, diff_batch):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + diff_row).detach()
        del flat_grads_chunk

    if t == 0 or t == start_step:
      if isinstance(model_t, nn.Module):
        model_t.zero_grad(set_to_none=True)
      del x_batch, y_batch, model_t
      break

    step_prev = t
    retain_last_graph = gradients_lst is not None

    # Now the fun part: updating mat_p, mat_q, mat_r.
    if query_vec is not None:
      hvp_in_vec = gradient_curr * gamma_1 / ((s_curr_hvp**3) + eps)
      hvp_in_mag = hvp_in_vec.abs().max().item()
      if hvp_in_mag > 100:
        logging.warning(
            'HVP input vector magnitude: %f at step %d (gamma_1 mag=%f, grad mag=%f)',
            hvp_in_mag,
            t,
            gamma_1.abs().max().item(),
            gradient_curr.abs().max().item(),
        )

      hvp_a = (
          utils.compute_hessian_prod_from_gradient(
              model_t,
              gradients_list,
              hvp_in_vec,
              retain_graph=True,
          )
          / bc2
      )
      hvp_b = (
          utils.compute_hessian_prod_from_gradient(
              model_t,
              gradients_list,
              gamma_2 / s_curr,
              retain_graph=retain_last_graph,
          )
          / bc1
      )
      if torch.isnan(hvp_a).any():
        raise ValueError(f'NaN found in hvp_a at step {t}')
      if torch.isnan(hvp_b).any():
        raise ValueError(f'NaN found in hvp_b at step {t}')
      if ((t + 1) % 100 == 0):
        logging.info('hvp_a magnitude: %f', hvp_a.abs().max().item())
        logging.info('hvp_b magnitude: %f', hvp_b.abs().max().item())

      mat_p_new = mat_p + (1 - beta_2) * hvp_a + (1 - beta_1) * hvp_b
      mat_q_new = gamma_1 * vec_a1
      mat_r_new = gamma_2 * vec_a2

      back_step_idx = (total_steps - 1) - t
      if clip_factor is not None and clip_factor > 0:
        if back_step_idx < calib_steps:
          max_hvp_a_calib = max(max_hvp_a_calib, hvp_a.abs().max().item())
          max_hvp_b_calib = max(max_hvp_b_calib, hvp_b.abs().max().item())
          max_mat_p_calib = max(max_mat_p_calib, mat_p_new.abs().max().item())
          max_mat_q_calib = max(max_mat_q_calib, mat_q_new.abs().max().item())
          max_mat_r_calib = max(max_mat_r_calib, mat_r_new.abs().max().item())
        else:
          thresh_hvp_a = clip_factor * max(max_hvp_a_calib, 1e-4)
          thresh_hvp_b = clip_factor * max(max_hvp_b_calib, 1e-4)
          thresh_mat_p = clip_factor * max(max_mat_p_calib, 1e-4)
          thresh_mat_q = clip_factor * max(max_mat_q_calib, 1e-4)
          thresh_mat_r = clip_factor * max(max_mat_r_calib, 1e-4)

          if back_step_idx == calib_steps:
            logging.info(
                'Adaptive clipping calibration complete at step %d: '
                'thresh_hvp_a=%.6f, thresh_hvp_b=%.6f, thresh_mat_p=%.6f, '
                'thresh_mat_q=%.6f, thresh_mat_r=%.6f',
                t,
                thresh_hvp_a,
                thresh_hvp_b,
                thresh_mat_p,
                thresh_mat_q,
                thresh_mat_r,
            )

          hvp_a_mag = hvp_a.abs().max().item()
          if hvp_a_mag > thresh_hvp_a:
            logging.warning(
                'Adaptive clipping hvp_a at step %d: mag=%f > thresh=%f (calib_max=%f)',
                t,
                hvp_a_mag,
                thresh_hvp_a,
                max_hvp_a_calib,
            )
            hvp_a = torch.clamp(hvp_a, -thresh_hvp_a, thresh_hvp_a)

          hvp_b_mag = hvp_b.abs().max().item()
          if hvp_b_mag > thresh_hvp_b:
            logging.warning(
                'Adaptive clipping hvp_b at step %d: mag=%f > thresh=%f (calib_max=%f)',
                t,
                hvp_b_mag,
                thresh_hvp_b,
                max_hvp_b_calib,
            )
            hvp_b = torch.clamp(hvp_b, -thresh_hvp_b, thresh_hvp_b)

          mat_p_new = mat_p + (1 - beta_2) * hvp_a + (1 - beta_1) * hvp_b
          mat_p_mag = mat_p_new.abs().max().item()
          if mat_p_mag > thresh_mat_p:
            logging.warning(
                'Adaptive clipping mat_p_new at step %d: mag=%f > thresh=%f (calib_max=%f)',
                t,
                mat_p_mag,
                thresh_mat_p,
                max_mat_p_calib,
            )
            mat_p_new = torch.clamp(mat_p_new, -thresh_mat_p, thresh_mat_p)

          mat_q_mag = mat_q_new.abs().max().item()
          if mat_q_mag > thresh_mat_q:
            logging.warning(
                'Adaptive clipping mat_q_new at step %d: mag=%f > thresh=%f (calib_max=%f)',
                t,
                mat_q_mag,
                thresh_mat_q,
                max_mat_q_calib,
            )
            mat_q_new = torch.clamp(mat_q_new, -thresh_mat_q, thresh_mat_q)

          mat_r_mag = mat_r_new.abs().max().item()
          if mat_r_mag > thresh_mat_r:
            logging.warning(
                'Adaptive clipping mat_r_new at step %d: mag=%f > thresh=%f (calib_max=%f)',
                t,
                mat_r_mag,
                thresh_mat_r,
                max_mat_r_calib,
            )
            mat_r_new = torch.clamp(mat_r_new, -thresh_mat_r, thresh_mat_r)

      if torch.isnan(mat_p_new).any():
        raise ValueError(f'NaN found in mat_p_new at step {t}')
      if torch.isnan(mat_q_new).any():
        raise ValueError(f'NaN found in mat_q_new at step {t}')
      if torch.isnan(mat_r_new).any():
        raise ValueError(f'NaN found in mat_r_new at step {t}')
      if ((t + 1) % 100 == 0):
        logging.info('mat_p_new magnitude: %f', mat_p_new.abs().max().item())
        logging.info('mat_q_new magnitude: %f', mat_q_new.abs().max().item())
        logging.info('mat_r_new magnitude: %f', mat_r_new.abs().max().item())

    else:
      h_mat = utils.compute_hessian_from_gradient(
          model_t, gradients_list, retain_graph=retain_last_graph
      )
      mat_b1 = (
          (1 - beta_2) * torch.diag(gradient_curr / (s_curr_hvp**3)) @ h_mat / bc2
      )
      mat_b2 = (1 - beta_1) * torch.diag(1 / s_curr) @ h_mat / bc1
      hvp_a = gamma_1 @ mat_b1
      hvp_b = gamma_2 @ mat_b2
      mat_p_new = mat_p + hvp_a + hvp_b
      mat_q_new = torch.diag(vec_a1) @ gamma_1
      mat_r_new = torch.diag(vec_a2) @ gamma_2
    mat_p = mat_p_new.detach()
    mat_q = mat_q_new.detach()
    mat_r = mat_r_new.detach()

    curr_model = model_prev
    curr_momentum = momentum_prev
    curr_variance = variance_prev

    if isinstance(model_t, nn.Module):
      model_t.zero_grad(set_to_none=True)
      for p in model_t.parameters():
        p.requires_grad_(True)

    del x_batch, y_batch, model_t, gradients_list
    del (
        momentum_curr,
        gradient_curr,
        s_curr,
        s_prev,
        vec_a1,
        vec_a2,
        vec_c1,
        vec_c2,
        gamma_1,
        gamma_2,
        target,
    )
    del mat_p_new, mat_q_new, mat_r_new
    if query_vec is not None:
      del hvp_a, hvp_b
    else:
      del h_mat, mat_b1, mat_b2, hvp_a, hvp_b
    utils.garbage_collect(device=device)

  return theta_diff_dict


def tracin(
    base_model,
    query_vec = None,
    inds = None,
    num_steps = None,
    device = None,
    ind = None,
):
  r"""Computes the influence of training points on the final model according to TracIn for Adam.

  Reference: "LESS: Selecting Influential Data for Targeted Instruction Tuning"
  (Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen,
  ICML 2024, https://arxiv.org/pdf/2402.04333).

  For each training point z_i at step t in batch B_t, TracIn for Adam attributes
  parameter displacement via the Adam-preconditioned gradient:
    \Delta \theta_t(z_i) = \frac{\eta}{|B_t|} \frac{1 - \beta_1}{1 -
    \beta_1^{t+1}} \frac{\nabla \ell(z_i, \theta_t)}{\sqrt{v_{t+1} / (1 -
    \beta_2^{t+1})} + \epsilon}.
  With query vector u (e.g. \nabla \ell(z_test, \theta_T)), the influence is:
    TracIn_{Adam}(z_i) = \sum_{t: z_i \in B_t} u^\top \Delta \theta_t(z_i).

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    query_vec: Query vector (required, e.g. test loss gradient).
    inds: Index or sequence of indices of training points to evaluate.
    num_steps: Number of training steps to iterate over.
    device: Computation device.
    ind: Single index (for backward compatibility).

  Returns:
    Dictionary mapping each index in inds to its scalar TracIn influence score.
  """
  (
      data_loader,
      checkpoints,
      criterion,
      momentum_lst,
      variance_lst,
      lr,
      beta_1,
      beta_2,
      eps,
  ) = _extract_adam_context(base_model)


  if criterion is None:
    criterion = nn.functional.cross_entropy
  if query_vec is None:
    raise ValueError('query_vec must be provided for TracIn.')

  batches = utils.prepare_batches(data_loader)
  num_batches = len(batches)
  if num_batches == 0:
    raise ValueError('data_loader is empty.')

  if inds is None and ind is not None:
    inds = ind

  if inds is None:
    ind_list = []
  elif isinstance(inds, int):
    ind_list = [inds]
  else:
    ind_list = list(inds)

  if device is None:
    device = query_vec.device
  else:
    device = torch.device(device)

  query_vec = query_vec.to(device)

  if num_steps is not None:
    total_steps = num_steps
  elif isinstance(base_model, dict) and base_model.get('num_steps') is not None:
    total_steps = base_model['num_steps']
  elif hasattr(base_model, 'total_steps') and callable(base_model.total_steps) and base_model.total_steps() > 0:
    total_steps = base_model.total_steps()
  elif isinstance(checkpoints, (list, tuple)) and len(checkpoints) > 1:
    total_steps = len(checkpoints) - 1
  elif isinstance(checkpoints, dict):
    pos_keys = [k for k in checkpoints.keys() if isinstance(k, int)]
    if -1 in pos_keys:
      total_steps = num_steps if num_steps is not None else (base_model.get('num_steps') if isinstance(base_model, dict) and base_model.get('num_steps') is not None else (max([k for k in pos_keys if k > 0]) if any(k > 0 for k in pos_keys) else 1))
    else:
      valid_keys = [k for k in pos_keys if k > 0]
      total_steps = max(valid_keys) if valid_keys else 1
  else:
    total_steps = 1

  if isinstance(checkpoints, dict):
    if 0 in checkpoints:
      ckpt_0 = checkpoints[0]
    elif -1 in checkpoints and len(checkpoints) == 1:
      ckpt_0 = checkpoints[-1]
    else:
      ckpt_0 = next(iter(checkpoints.values()))
  elif isinstance(checkpoints, nn.Module):
    ckpt_0 = checkpoints
  else:
    ckpt_0 = checkpoints[0]

  if isinstance(momentum_lst, dict):
    if 0 in momentum_lst:
      m_0 = momentum_lst[0]
    elif -1 in momentum_lst and len(momentum_lst) == 1:
      m_0 = momentum_lst[-1]
    else:
      m_0 = next(iter(momentum_lst.values()))
  elif momentum_lst is not None:
    m_0 = momentum_lst[0]
  else:
    m_0 = None

  if isinstance(variance_lst, dict):
    if 0 in variance_lst:
      v_0 = variance_lst[0]
    elif -1 in variance_lst and len(variance_lst) == 1:
      v_0 = variance_lst[-1]
    else:
      v_0 = next(iter(variance_lst.values()))
  elif variance_lst is not None:
    v_0 = variance_lst[0]
  else:
    v_0 = None

  has_all_ckpts = (
      (
          isinstance(checkpoints, (list, tuple))
          and len(checkpoints) >= total_steps + 1
      )
      or (
          isinstance(checkpoints, dict)
          and all(k in checkpoints for k in range(total_steps + 1))
      )
  ) and (
      variance_lst is None
      or (
          isinstance(variance_lst, (list, tuple))
          and len(variance_lst) >= total_steps + 1
      )
      or (
          isinstance(variance_lst, dict)
          and all(k in variance_lst for k in range(1, total_steps + 1))
      )
  )
  use_on_the_fly_metadata = not has_all_ckpts

  curr_model = copy.deepcopy(ckpt_0).to(device) if use_on_the_fly_metadata else None
  curr_momentum = (
      [m.to(device) for m in m_0]
      if m_0 is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_0.parameters()]
  ) if use_on_the_fly_metadata else []
  curr_variance = (
      [v.to(device) for v in v_0]
      if v_0 is not None
      else [torch.zeros_like(p, device=device) for p in ckpt_0.parameters()]
  ) if use_on_the_fly_metadata else []

  theta_diff_dict = {idx: torch.tensor(0.0, device=device) for idx in ind_list}

  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  for t in range(total_steps):
    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    has_ckpt_t = (
        (isinstance(checkpoints, dict) and t in checkpoints)
        or (
            isinstance(checkpoints, (list, tuple))
            and len(checkpoints) > t
        )
    )

    if use_on_the_fly_metadata:
      if has_ckpt_t:
        model = (checkpoints[t] if isinstance(checkpoints, dict) else checkpoints[t]).to(device)
        curr_model = copy.deepcopy(model)
        if isinstance(momentum_lst, dict) and t in momentum_lst:
          curr_momentum = [m.to(device) for m in momentum_lst[t]]
        elif isinstance(momentum_lst, (list, tuple)) and len(momentum_lst) > t:
          curr_momentum = [m.to(device) for m in momentum_lst[t]]
        if isinstance(variance_lst, dict) and t in variance_lst:
          curr_variance = [v.to(device) for v in variance_lst[t]]
        elif isinstance(variance_lst, (list, tuple)) and len(variance_lst) > t:
          curr_variance = [v.to(device) for v in variance_lst[t]]
      else:
        model = curr_model if curr_model is not None else ckpt_0.to(device)

      theta_next, momentum_next, variance_next = (
          utils.compute_metadata_adam_forward(
              x_batch,
              y_batch,
              model,
              criterion,
              curr_momentum,
              curr_variance,
              lr,
              beta_1,
              beta_2,
              eps,
              step=t + 1,
              device=device,
          )
      )
      v_curr = variance_next
    else:
      theta_next, momentum_next, variance_next = None, None, None
      model = checkpoints[t].to(device)
      if variance_lst is not None:
        v_curr = [v.to(device) for v in variance_lst[t + 1]]
      else:
        v_curr = [torch.ones_like(p, device=device) for p in model.parameters()]

    batch_size_t = x_batch.shape[0]
    num_params = sum(p.numel() for p in model.parameters())

    step = t + 1
    bc1 = 1.0 - beta_1**step
    bc2 = 1.0 - beta_2**step
    effective_lr = (lr / bc1) * (1.0 - beta_1) / batch_size_t

    s = [torch.sqrt(v / bc2) + eps for v in v_curr]
    flat_s = torch.cat([u_v.contiguous().view(-1) for u_v in s])

    active_pairs = utils.get_batch_active_indices(
        batches, j, candidate_set, batch_start_offsets
    )
    if active_pairs:
      active_dataset_indices = [p[0] for p in active_pairs]
      active_batch_positions = [p[1] for p in active_pairs]

      num_active = len(active_batch_positions)
      # Dynamically determine chunk size based on current GPU memory & model size
      chunk_size = utils.get_dynamic_chunk_size(
          num_params=num_params,
          dtype=flat_s.dtype,
          device=device,
          max_memory_mb=1024.0,
      )

      # Process active samples in GPU-sized chunks
      for chunk_start in range(0, num_active, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_active)
        chunk_positions = active_batch_positions[chunk_start:chunk_end]
        chunk_dataset_indices = active_dataset_indices[chunk_start:chunk_end]
        curr_chunk_len = len(chunk_positions)
        # Preallocate chunk buffer directly on GPU to avoid torch.stack overhead
        flat_grads_chunk = torch.empty(
            (curr_chunk_len, num_params),
            dtype=flat_s.dtype,
            device=device,
        )
        for local_idx, pos in enumerate(chunk_positions):
          x_batch_remove = x_batch[pos : pos + 1]
          y_batch_remove = y_batch[pos : pos + 1]
          y_pred_batch_remove = model(x_batch_remove)
          loss_batch_remove = criterion(y_pred_batch_remove, y_batch_remove)
          gradients_remove = torch.autograd.grad(
              loss_batch_remove, model.parameters(), create_graph=False
          )
          flat_grads_chunk[local_idx] = torch.cat(
              [g.contiguous().view(-1) for g in gradients_remove]
          )
        scaled_grads_chunk = effective_lr * (
            flat_grads_chunk / flat_s.unsqueeze(0)
        )
        cos_sims_chunk = torch.cosine_similarity(
            query_vec.unsqueeze(0), scaled_grads_chunk, dim=1
        ).detach()

        for idx, sim_val in zip(chunk_dataset_indices, cos_sims_chunk):
          theta_diff_dict[idx] = (theta_diff_dict[idx] + sim_val).detach()
        del flat_grads_chunk, scaled_grads_chunk, cos_sims_chunk

    if isinstance(model, nn.Module):
      model.zero_grad(set_to_none=True)
      for p in model.parameters():
        p.requires_grad_(True)

    if (
        use_on_the_fly_metadata
        and curr_model is not None
        and theta_next is not None
        and momentum_next is not None
        and variance_next is not None
    ):
      curr_model = copy.deepcopy(curr_model)
      utils.update_model_parameters(curr_model, theta_next)
      curr_momentum = momentum_next
      curr_variance = variance_next

    del x_batch, y_batch, model

  utils.garbage_collect(device=device)
  return theta_diff_dict
