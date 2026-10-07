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

import copy
from typing import Any, Sequence

import torch
from torch import nn

from reversible_data_attribution import base_model
from reversible_data_attribution import utils


def one_step_update(
    x_batch,
    y_batch,
    model,
    loss_func,
    theta_diff,
    lr,
    ind,
    random_mask = None,
):
  """Goes from t to t+1 for counterfactual parameters under SGD."""
  is_list_input = isinstance(theta_diff, list)
  params = list(model.parameters())
  device = params[0].device if params else x_batch.device

  if is_list_input:
    theta_diff_mat = torch.cat(
        [u.to(device).flatten() for u in theta_diff]
    ).unsqueeze(0)
  elif isinstance(theta_diff, torch.Tensor):
    theta_diff_mat = (
        theta_diff.to(device)
        if theta_diff.dim() == 2
        else theta_diff.to(device).unsqueeze(0)
    )
  else:
    raise TypeError(f'Unsupported theta_diff type: {type(theta_diff)}')

  grads_list, diff_grad = utils.get_grad_diff(
      x_batch,
      y_batch,
      model,
      loss_func,
      theta_diff_mat,
      ind,
      use_pearlmutter=True,
      random_mask=random_mask,
  )
  del grads_list
  if isinstance(diff_grad, list):
    diff_grad = torch.cat([d.flatten() for d in diff_grad]).unsqueeze(0)

  theta_diff_new = (theta_diff_mat - diff_grad * lr).detach()

  if is_list_input:
    if random_mask is None:
      split_sizes = [p.numel() for p in params]
      theta_ret = [
          t.reshape(p.shape)
          for t, p in zip(torch.split(theta_diff_new[0], split_sizes), params)
      ]
    else:
      if isinstance(random_mask, torch.Tensor):
        split_sizes = [int(random_mask.sum().item())]
      else:
        split_sizes = [int(r.sum().item()) for r in random_mask]
      theta_ret = list(torch.split(theta_diff_new[0], split_sizes))
    return theta_ret

  return theta_diff_new


def _extract_sgd_context(
    base_model_obj,
):
  """Extracts required data structures and hyperparameters from base_model for SGD."""
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
    lr = hp.get('lr', base_model_obj.get('lr', 0.01))
    beta_1 = hp.get('beta_1', base_model_obj.get('beta_1', 0.9))
    beta_2 = hp.get('beta_2', base_model_obj.get('beta_2', 0.999))
    eps = hp.get('eps', base_model_obj.get('eps', 1e-8))
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
      lr,
      beta_1,
      beta_2,
      eps,
      momentum_lst,
      variance_lst,
  )


def forward_update(
    base_model,
    ind = None,
    num_steps = None,
    random_mask = None,
    device = None,
):
  """Computes forward update for counterfactual parameter difference.

  Supports running forward_update on multiple indices simultaneously.

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    ind: Index or sequence of indices of datapoints to remove.
    num_steps: Number of steps to take.
    random_mask: Random mask for parameter compression.
    device: Computation device.
  """
  (
      data_loader,
      checkpoints,
      criterion,
      lr,
      beta_1,
      beta_2,
      eps,
      momentum_lst,
      variance_lst,
  ) = _extract_sgd_context(base_model)
  if criterion is None:
    criterion = nn.functional.cross_entropy
  batches = utils.prepare_batches(data_loader)
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
  total_theta_bytes = m * p_dim * elem_size
  max_gpu_theta_bytes = 256 * 1024 * 1024  # 256 MB threshold

  # Keep batched theta_diff in CPU RAM when it exceeds GPU memory budget,
  # streaming GPU-sized chunks during update iterations.
  if is_batched and (
      device.type == 'cuda' and total_theta_bytes > max_gpu_theta_bytes
  ):
    theta_diff = torch.zeros((m, p_dim), device='cpu', dtype=param_dtype)
  else:
    try:
      theta_diff = torch.zeros((m, p_dim), device=device, dtype=param_dtype)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
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
      total_steps = max(valid_keys) if valid_keys else 1
  else:
    total_steps = 1

  has_all_ckpts = (
      isinstance(checkpoints, (list, tuple))
      and len(checkpoints) >= total_steps + 1
  ) or (
      isinstance(checkpoints, dict)
      and all(k in checkpoints for k in range(total_steps + 1))
  )
  use_on_the_fly_metadata = not has_all_ckpts

  m_0 = None
  if isinstance(momentum_lst, dict):
    if 0 in momentum_lst:
      m_0 = momentum_lst[0]
    elif -1 in momentum_lst and len(momentum_lst) == 1:
      m_0 = momentum_lst[-1]
    else:
      m_0 = next(iter(momentum_lst.values()))
  elif momentum_lst is not None and len(momentum_lst) > 0:
    m_0 = momentum_lst[0]

  v_0 = None
  if isinstance(variance_lst, dict):
    if 0 in variance_lst:
      v_0 = variance_lst[0]
    elif -1 in variance_lst and len(variance_lst) == 1:
      v_0 = variance_lst[-1]
    else:
      v_0 = next(iter(variance_lst.values()))
  elif variance_lst is not None and len(variance_lst) > 0:
    v_0 = variance_lst[0]

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

  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  for t in range(total_steps):
    if t % 100 == 0:
      utils.log_cuda_memory(
          f'infl_sgd.forward_update step t={t}', device=device
      )
    j = t % num_batches
    x_batch, y_batch, batch_indices = batches[j]
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    has_ckpt_t = (
        (isinstance(checkpoints, dict) and t in checkpoints)
        or (
            isinstance(checkpoints, (list, tuple))
            and len(checkpoints) >= total_steps + 1
            and t < len(checkpoints)
        )
    )

    ckpt_t = None
    was_on_cpu = False
    if use_on_the_fly_metadata:
      if has_ckpt_t:
        ckpt_t = (
            checkpoints[t] if isinstance(checkpoints, dict) else checkpoints[t]
        )
        curr_model = copy.deepcopy(ckpt_t).to(device)
        model = curr_model
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
    else:
      theta_next, momentum_next, variance_next = None, None, None
      ckpt_t = checkpoints[t]
      was_on_cpu = hasattr(ckpt_t, 'parameters') and any(
          p.device.type == 'cpu' for p in ckpt_t.parameters()
      )
      model = ckpt_t.to(device)

    if batch_indices is not None:
      if isinstance(batch_indices, torch.Tensor):
        batch_indices_list = batch_indices.tolist()
      else:
        batch_indices_list = list(batch_indices)
      idx_to_pos = {
          int(idx): pos for pos, idx in enumerate(batch_indices_list)
      }
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
      # Dynamically determine chunk size based on current GPU memory & model size
      chunk_size = utils.get_dynamic_chunk_size(
          num_params=p_dim,
          dtype=theta_diff.dtype,
          device=device,
          max_memory_mb=1024.0,
      )

      # Process active samples in GPU-sized chunks
      for chunk_start in range(0, m, chunk_size):
        chunk_end = min(chunk_start + chunk_size, m)
        chunk_positions = ind_batch_list[chunk_start:chunk_end]
        chunk_theta = theta_diff[chunk_start:chunk_end].to(device)
        updated_chunk = one_step_update(
            x_batch,
            y_batch,
            model,
            criterion,
            chunk_theta,
            lr,
            chunk_positions,
            random_mask=random_mask,
        )
        theta_diff[chunk_start:chunk_end] = updated_chunk.to(theta_diff.device)
        del chunk_theta, updated_chunk
    else:
      chunk_theta = theta_diff.to(device)
      theta_diff = one_step_update(
          x_batch,
          y_batch,
          model,
          criterion,
          chunk_theta,
          lr,
          ind_arg,
          random_mask=random_mask,
      )
      del chunk_theta
    theta_diff = theta_diff.detach()

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

    del x_batch, y_batch, model
    if theta_next is not None:
      del theta_next, momentum_next, variance_next

  if curr_model is not None:
    del curr_model, curr_momentum, curr_variance

  utils.garbage_collect(device=device)
  if is_batched:
    if (
        theta_diff.device.type == 'cpu'
        and total_theta_bytes > max_gpu_theta_bytes
    ):
      return theta_diff
    try:
      return theta_diff.to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
      return theta_diff
  else:
    if random_mask is None:
      split_sizes = [p.numel() for p in ckpt_0.parameters()]
      theta_ret = [
          t.reshape(p.shape).clone()
          for t, p in zip(
              torch.split(theta_diff[0], split_sizes), ckpt_0.parameters()
          )
      ]
    else:
      if isinstance(random_mask, torch.Tensor):
        split_sizes = [int(random_mask.sum().item())]
      else:
        split_sizes = [int(r.sum().item()) for r in random_mask]
      theta_ret = [t.clone() for t in torch.split(theta_diff[0], split_sizes)]
    del theta_diff
    return theta_ret


def recursive_update(
    base_model,
    inds = None,
    query_vec = None,
    num_steps = None,
    gradients_lst = None,
    device = None,
    start_step = 0,
):
  """Computes recursive update for counterfactual parameter difference.

  Args:
    base_model: AttributionBaseModel instance (or dictionary of artifacts).
    inds: Index or sequence of indices of datapoints to remove.
    query_vec: Optional query vector.
    num_steps: Number of steps.
    gradients_lst: Optional gradients list.
    device: Computation device.
    start_step: Step index before which gradient effects are ignored.
  """
  data_loader, checkpoints, criterion, lr, *_ = _extract_sgd_context(base_model)
  if criterion is None:
    criterion = nn.functional.cross_entropy
  batches = utils.prepare_batches(data_loader)
  num_batches = len(batches)
  if num_batches == 0:
    raise ValueError('data_loader is empty.')

  if isinstance(checkpoints, dict):
    if 0 in checkpoints:
      ckpt_ref = checkpoints[0]
    elif -1 in checkpoints:
      ckpt_ref = checkpoints[-1]
    else:
      ckpt_ref = next(iter(checkpoints.values()))
  elif isinstance(checkpoints, nn.Module):
    ckpt_ref = checkpoints
  else:
    ckpt_ref = checkpoints[0]

  if device is None:
    if query_vec is not None:
      device = query_vec.device
    else:
      device = next(ckpt_ref.parameters()).device
  else:
    device = torch.device(device)
  num_params = sum(p.numel() for p in ckpt_ref.parameters())

  if inds is None:
    ind_list = []
  elif isinstance(inds, int):
    ind_list = [inds]
  else:
    ind_list = list(inds)

  u = query_vec.clone().to(device) if query_vec is not None else None
  m_mat = torch.eye(num_params, device=device) if query_vec is None else None

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

  m_lst = {}

  for t in range(total_steps - 1, start_step - 1, -1):
    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    batch_size_t = x_batch.shape[0]
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    if u is not None:
      target = lr * u
    elif m_mat is not None:
      target = lr * m_mat
    else:
      raise RuntimeError('Neither u nor m_mat is initialized.')
    m_lst[t] = target
    if t == 0 or t == start_step:
      del x_batch, y_batch
      break

    if isinstance(checkpoints, (list, tuple)) and len(checkpoints) > t:
      model_t = checkpoints[t].to(device)
    elif isinstance(checkpoints, dict) and t in checkpoints:
      model_t = checkpoints[t].to(device)
    elif isinstance(checkpoints, dict) and -1 in checkpoints:
      model_t = checkpoints[-1].to(device)
    elif isinstance(checkpoints, nn.Module):
      model_t = checkpoints.to(device)
    else:
      model_t = (
          checkpoints[0]
          if (isinstance(checkpoints, dict) and 0 in checkpoints)
          else next(iter(checkpoints.values()))
      ).to(device)

    if gradients_lst is not None:
      gradients_list = [g.to(device) for g in gradients_lst[t]]
    else:
      gradients_list = utils.compute_gradient(
          x_batch, y_batch, model_t, criterion, None
      )

    if u is not None:
      hessian_prod = utils.compute_hessian_prod_from_gradient(
          model_t, gradients_list, u
      )
      u = u - lr * hessian_prod
    elif m_mat is not None:
      h_mat = utils.compute_hessian_from_gradient(model_t, gradients_list)
      m_mat = m_mat - lr * (h_mat @ m_mat)

    del x_batch, y_batch, model_t, gradients_list

  utils.garbage_collect(device=device)

  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  theta_diff_dict = {
      idx: (
          torch.tensor(0.0, device=device)
          if query_vec is not None
          else torch.zeros(num_params, device=device)
      )
      for idx in ind_list
  }

  for t in range(start_step, total_steps):
    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    if isinstance(checkpoints, (list, tuple)) and len(checkpoints) > t:
      model = checkpoints[t].to(device)
    elif isinstance(checkpoints, dict) and t in checkpoints:
      model = checkpoints[t].to(device)
    elif isinstance(checkpoints, dict) and -1 in checkpoints:
      model = checkpoints[-1].to(device)
    elif isinstance(checkpoints, nn.Module):
      model = checkpoints.to(device)
    else:
      model = (
          checkpoints[0]
          if (isinstance(checkpoints, dict) and 0 in checkpoints)
          else next(iter(checkpoints.values()))
      ).to(device)

    batch_size_t = x_batch.shape[0]

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
          dtype=m_lst[t].dtype,
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
            dtype=m_lst[t].dtype,
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

        # High-performance batched GPU matrix-vector / matrix-matrix multiplication
        if query_vec is not None:
          dot_products = (
              torch.mv(flat_grads_chunk, m_lst[t]) / batch_size_t
          ).detach()
          for idx, dot_val in zip(chunk_dataset_indices, dot_products):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + dot_val).detach()
        else:
          diff_batch = (flat_grads_chunk @ m_lst[t].T / batch_size_t).detach()
          for idx, diff_row in zip(chunk_dataset_indices, diff_batch):
            theta_diff_dict[idx] = (theta_diff_dict[idx] + diff_row).detach()

        del flat_grads_chunk

    del x_batch, y_batch, model

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
  r"""Computes the influence of training points on the final model according to TracIn (SGD).

  Reference: "Estimating Training Data Influence by Tracing Gradient Descent"
  (Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan, NeurIPS 2020,
  https://arxiv.org/pdf/2002.08484).

  For each training point z_i at step t in batch B_t, TracIn attributes the
  parameter
  change:
    \Delta \theta_t(z_i) = (lr / |B_t|) * \nabla \ell(z_i, \theta_t).
  With query vector u (e.g. \nabla \ell(z_test, \theta_T)), the influence is:
    TracIn(z_i) = \sum_{t: z_i \in B_t} (lr / |B_t|) * u^\top \nabla
    \ell(z_i, \theta_t).

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
      lr,
      beta_1,
      beta_2,
      eps,
      momentum_lst,
      variance_lst,
  ) = _extract_sgd_context(base_model)
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

  has_all_ckpts = (
      isinstance(checkpoints, (list, tuple))
      and len(checkpoints) >= total_steps + 1
  ) or (
      isinstance(checkpoints, dict)
      and all(k in checkpoints for k in range(total_steps + 1))
  )
  use_on_the_fly_metadata = not has_all_ckpts

  m_0 = None
  if isinstance(momentum_lst, dict):
    if 0 in momentum_lst:
      m_0 = momentum_lst[0]
    elif -1 in momentum_lst and len(momentum_lst) == 1:
      m_0 = momentum_lst[-1]
    else:
      m_0 = next(iter(momentum_lst.values()))
  elif momentum_lst is not None and len(momentum_lst) > 0:
    m_0 = momentum_lst[0]

  v_0 = None
  if isinstance(variance_lst, dict):
    if 0 in variance_lst:
      v_0 = variance_lst[0]
    elif -1 in variance_lst and len(variance_lst) == 1:
      v_0 = variance_lst[-1]
    else:
      v_0 = next(iter(variance_lst.values()))
  elif variance_lst is not None and len(variance_lst) > 0:
    v_0 = variance_lst[0]

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

  candidate_set = set(ind_list)
  batch_start_offsets = utils.compute_batch_start_offsets(batches)

  theta_diff_dict = {idx: torch.tensor(0.0, device=device) for idx in ind_list}

  for t in range(total_steps):
    j = t % num_batches
    x_batch, y_batch, _ = utils.get_batch_and_ind(batches, j, None)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    has_ckpt_t = (
        (isinstance(checkpoints, dict) and t in checkpoints)
        or (
            isinstance(checkpoints, (list, tuple))
            and len(checkpoints) >= total_steps + 1
            and t < len(checkpoints)
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
    else:
      theta_next, momentum_next, variance_next = None, None, None
      model = checkpoints[t].to(device)

    batch_size_t = x_batch.shape[0]
    num_params = sum(p.numel() for p in model.parameters())

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
          dtype=query_vec.dtype,
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
            dtype=query_vec.dtype,
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

        # High-performance batched GPU matrix-vector multiplication
        dot_products = (
            (lr / batch_size_t) * torch.mv(flat_grads_chunk, query_vec)
        ).detach()

        for idx, dot_val in zip(chunk_dataset_indices, dot_products):
          theta_diff_dict[idx] = (theta_diff_dict[idx] + dot_val).detach()

        del flat_grads_chunk

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
      curr_model = copy.deepcopy(curr_model)
      utils.update_model_parameters(curr_model, theta_next)
      curr_momentum = momentum_next
      curr_variance = variance_next

    del x_batch, y_batch, model

  utils.garbage_collect(device=device)
  return theta_diff_dict
