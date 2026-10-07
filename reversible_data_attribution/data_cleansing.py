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

"""Evaluation on Data Cleansing (Section 7.2 of arXiv:1906.08473).

This module integrates directly with infl_sgd.py and infl_adam.py to calculate
influence scores and perform high-level evaluation of data cleansing (retraining
with chosen indices removed and measuring changes in loss and accuracy).
"""

from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import csv
import gc
import glob
import io
import json
import os
import random
from typing import Any

from absl import logging
import numpy as np
import sklearn.ensemble
import sklearn.metrics
import torch
from torch import nn

from reversible_data_attribution import base_model
from reversible_data_attribution import infl_adam
from reversible_data_attribution import infl_sgd
from reversible_data_attribution import pipeline
from reversible_data_attribution import reversible
from reversible_data_attribution import utils


def get_adam_method_name(random_masking = None):
  """Returns the method name for Adam exact forward, with masking suffix if random_masking is supplied."""
  if random_masking is None or random_masking <= 0:
    return 'adam_exact'
  if random_masking >= 0.01:
    pct = random_masking * 100
    pct_str = f'{int(pct)}%' if pct.is_integer() else f'{pct:g}%'
    return f'adam_masked_{pct_str}'
  else:
    sci_str = (
        f'{random_masking:.0e}'.replace('e-0', 'e-')
        .replace('e+0', 'e+')
        .replace('e+', 'e')
    )
    return f'adam_masked_{sci_str}'


def compute_partitioned_corrupted_metrics(
    scores,
    corrupted_indices,
    is_outlier_score = False,
):
  """Computes partitioned score metrics and ROC-AUC for corrupted vs clean samples."""
  num_samples = len(scores)
  corrupted_set = (
      set(corrupted_indices) if corrupted_indices is not None else set()
  )

  y_true = np.zeros(num_samples, dtype=np.int32)
  for idx in corrupted_set:
    if 0 <= idx < num_samples:
      y_true[idx] = 1

  clean_mask = y_true == 0
  corrupted_mask = y_true == 1

  mean_clean = float(np.mean(scores[clean_mask])) if np.any(clean_mask) else 0.0
  std_clean = float(np.std(scores[clean_mask])) if np.any(clean_mask) else 0.0
  mean_corrupted = (
      float(np.mean(scores[corrupted_mask])) if np.any(corrupted_mask) else 0.0
  )
  std_corrupted = (
      float(np.std(scores[corrupted_mask])) if np.any(corrupted_mask) else 0.0
  )

  # For ROC-AUC: higher y_score must correspond to higher likelihood of corruption (class 1).
  # For outlier scores (AE/ISO), higher score = corrupted -> y_score = scores
  # For counterfactuals (\Delta Loss), more negative score = corrupted -> y_score = -scores
  y_score = scores if is_outlier_score else -scores

  if 0 < len(corrupted_set) < num_samples:
    try:
      roc_auc = float(sklearn.metrics.roc_auc_score(y_true, y_score))
    except Exception:
      roc_auc = 0.5
  else:
    roc_auc = 0.5

  return {
      'mean_score_clean': mean_clean,
      'std_score_clean': std_clean,
      'mean_score_corrupted': mean_corrupted,
      'std_score_corrupted': std_corrupted,
      'score_diff_corrupted_minus_clean': mean_corrupted - mean_clean,
      'roc_auc': roc_auc,
  }


compute_validation_gradient = utils.compute_validation_gradient


def compute_tracin_sgd_influence(
    base_model_obj,
    val_loader = None,
    query_vec = None,
    candidate_indices = None,
    num_steps = None,
    device = 'cpu',
):
  """Computes TracIn (SGD) influence scores using infl_sgd.tracin.

  Args:
    base_model_obj: AttributionBaseModel instance or artifact dict.
    val_loader: Validation DataLoader (used to compute query gradient if
      query_vec is not given).
    query_vec: Optional precomputed query gradient vector.
    candidate_indices: Optional subset of sample indices to evaluate.
    num_steps: Optional number of training steps.
    device: Computation device.

  Returns:
    Array of shape (N_train,) or dict mapping index to score for
    candidate_indices.
  """
  if hasattr(base_model_obj, 'train_loader'):
    data_loader = base_model_obj.train_loader()
    checkpoints = base_model_obj.get_checkpoint_models()
    criterion = base_model_obj.loss_fn()
  elif isinstance(base_model_obj, dict):
    data_loader = base_model_obj.get('train_loader') or base_model_obj.get('batches')
    checkpoints = base_model_obj.get('checkpoints', base_model_obj)
    criterion = base_model_obj.get('loss_fn') or base_model_obj.get('criterion')
  else:
    raise TypeError(f'Unsupported base_model_obj type: {type(base_model_obj)}')

  if criterion is None:
    criterion = nn.functional.cross_entropy
  if data_loader is None:
    raise ValueError('data_loader is missing in base_model_obj.')

  num_train = (
      len(data_loader.dataset)
      if hasattr(data_loader, 'dataset')
      else len(data_loader)
  )
  if query_vec is None:
    if val_loader is None:
      raise ValueError(
          'Either val_loader or query_vec must be provided for TracIn (SGD)'
          ' influence.'
      )
    ckpt_ref = (
        checkpoints[-1]
        if isinstance(checkpoints, (list, tuple, dict)) and -1 in checkpoints or isinstance(checkpoints, (list, tuple))
        else (checkpoints[max(checkpoints.keys())] if isinstance(checkpoints, dict) else checkpoints)
    )
    u_params = compute_validation_gradient(
        ckpt_ref, criterion, val_loader, device
    )
    query_vec = torch.cat([g.contiguous().view(-1) for g in u_params])

  target_indices = (
      list(candidate_indices)
      if candidate_indices is not None
      else list(range(num_train))
  )
  results_dict = infl_sgd.tracin(
      base_model=base_model_obj,
      query_vec=query_vec,
      inds=target_indices,
      num_steps=num_steps,
      device=device,
  )
  if candidate_indices is not None:
    return {i: results_dict[i].item() for i in target_indices}
  return np.array(
      [results_dict[i].item() for i in range(num_train)], dtype=np.float32
  )


def compute_tracin_adam_influence(
    base_model_obj,
    val_loader = None,
    query_vec = None,
    candidate_indices = None,
    num_steps = None,
    device = 'cpu',
):
  """Computes TracIn (Adam) influence scores using infl_adam.tracin.

  Args:
    base_model_obj: AttributionBaseModel instance or artifact dict.
    val_loader: Validation DataLoader (used to compute query gradient if
      query_vec is not given).
    query_vec: Optional precomputed query gradient vector.
    candidate_indices: Optional subset of sample indices to evaluate.
    num_steps: Optional number of training steps.
    device: Computation device.

  Returns:
    Array of shape (N_train,) or dict mapping index to score for
    candidate_indices.
  """
  if hasattr(base_model_obj, 'train_loader'):
    data_loader = base_model_obj.train_loader()
    checkpoints = base_model_obj.get_checkpoint_models()
    criterion = base_model_obj.loss_fn()
  elif isinstance(base_model_obj, dict):
    data_loader = base_model_obj.get('train_loader') or base_model_obj.get('batches')
    checkpoints = base_model_obj.get('checkpoints', base_model_obj)
    criterion = base_model_obj.get('loss_fn') or base_model_obj.get('criterion')
  else:
    raise TypeError(f'Unsupported base_model_obj type: {type(base_model_obj)}')

  if criterion is None:
    criterion = nn.functional.cross_entropy
  if data_loader is None:
    raise ValueError('data_loader is missing in base_model_obj.')

  num_train = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader)
  if query_vec is None:
    if val_loader is None:
      raise ValueError(
          'Either val_loader or query_vec must be provided for TracIn (Adam)'
          ' influence.'
      )
    ckpt_ref = (
        checkpoints[-1]
        if isinstance(checkpoints, (list, tuple, dict)) and -1 in checkpoints or isinstance(checkpoints, (list, tuple))
        else (checkpoints[max(checkpoints.keys())] if isinstance(checkpoints, dict) else checkpoints)
    )
    u_params = compute_validation_gradient(
        ckpt_ref, criterion, val_loader, device
    )
    query_vec = torch.cat([g.contiguous().view(-1) for g in u_params])

  target_indices = (
      list(candidate_indices)
      if candidate_indices is not None
      else list(range(num_train))
  )
  results_dict = infl_adam.tracin(
      base_model=base_model_obj,
      query_vec=query_vec,
      inds=target_indices,
      num_steps=num_steps,
      device=device,
  )
  if candidate_indices is not None:
    return {i: results_dict[i].item() for i in target_indices}
  return np.array(
      [results_dict[i].item() for i in range(num_train)], dtype=np.float32
  )


def compute_icml_influence(
    final_model,
    train_loader,
    val_loader,
    criterion,
    lr = 0.005,
    alpha = 0.01,
    num_epochs = 2,
    device = 'cpu',
):
  """Computes ICML (Koh & Liang 2017) Hessian-based influence scores.

  Solves v = H^{-1} u via stochastic optimization on 0.5 v^T H v + 0.5 alpha v^T
  v - u^T v.
  """
  final_model = final_model.to(device)
  u = compute_validation_gradient(final_model, criterion, val_loader, device)

  v = [
      torch.zeros_like(p, requires_grad=True, device=device)
      for p in final_model.parameters()
  ]
  optimizer = torch.optim.SGD(v, lr=lr)

  for _ in range(num_epochs):
    final_model.eval()
    for batch in train_loader:
      if isinstance(batch, (list, tuple)):
        x_tr, y_tr = batch[0].to(device), batch[1].to(device)
      else:
        x_tr, y_tr = batch['x'].to(device), batch['y'].to(device)

      with utils.sdpa_math_context(enable=True):
        out = final_model(x_tr)
        loss = criterion(out, y_tr)
        grad_params = torch.autograd.grad(
            loss, final_model.parameters(), create_graph=True
        )
        vg = sum(torch.sum(vv * g) for vv, g in zip(v, grad_params))
        final_model.zero_grad()
        vgrad_params = torch.autograd.grad(
            vg, final_model.parameters(), create_graph=True
        )

      loss_i = torch.tensor(0.0, device=device)
      for vgp, vv, uu in zip(vgrad_params, v, u):
        loss_i += 0.5 * torch.sum(vgp * vv + alpha * vv * vv) - torch.sum(
            uu * vv
        )

      optimizer.zero_grad()
      loss_i.backward()
      optimizer.step()

  v_detached = [vv.detach() for vv in v]
  icml_influences = []

  for batch in train_loader:
    if isinstance(batch, (list, tuple)):
      x_tr, y_tr = batch[0].to(device), batch[1].to(device)
    else:
      x_tr, y_tr = batch['x'].to(device), batch['y'].to(device)

    for i in range(x_tr.shape[0]):
      xi = x_tr[i : i + 1]
      yi = y_tr[i : i + 1]
      out_i = final_model(xi)
      loss_i = criterion(out_i, yi)
      final_model.zero_grad()
      loss_i.backward()

      infl_val = 0.0
      with torch.no_grad():
        for j, p in enumerate(final_model.parameters()):
          if p.grad is not None:
            infl_val += torch.sum(v_detached[j] * p.grad.data).item()
      icml_influences.append(infl_val)

  return np.array(icml_influences, dtype=np.float32)


class SimpleAutoEncoder(nn.Module):
  """Simple Autoencoder for image anomaly detection."""

  def __init__(self, input_dim, hidden_dim = 64):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
    )
    self.decoder = nn.Sequential(
        nn.Linear(hidden_dim // 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
        nn.Sigmoid(),
    )

  def forward(self, x):
    latent = self.encoder(x)
    return self.decoder(latent)


def compute_autoencoder_outlier_scores(
    train_loader,
    num_epochs = 10,
    lr = 0.001,
    device = 'cpu',
):
  """Trains an autoencoder on training data and returns reconstruction MSE per sample."""
  first_batch = next(iter(train_loader))
  x_sample = (
      first_batch[0]
      if isinstance(first_batch, (list, tuple))
      else first_batch['x']
  )
  x_flat = x_sample.view(x_sample.shape[0], -1)
  input_dim = x_flat.shape[1]

  ae = SimpleAutoEncoder(input_dim).to(device)
  optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
  criterion = nn.MSELoss()

  ae.train()
  for _ in range(num_epochs):
    for batch in train_loader:
      x_tr = batch[0] if isinstance(batch, (list, tuple)) else batch['x']
      x_tr = x_tr.view(x_tr.shape[0], -1).to(device)
      recon = ae(x_tr)
      loss = criterion(recon, x_tr)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  ae.eval()
  scores = []
  with torch.no_grad():
    for batch in train_loader:
      x_tr = batch[0] if isinstance(batch, (list, tuple)) else batch['x']
      x_tr = x_tr.view(x_tr.shape[0], -1).to(device)
      recon = ae(x_tr)
      mse = torch.mean((recon - x_tr) ** 2, dim=1)
      scores.extend(mse.cpu().numpy().tolist())

  return np.array(scores, dtype=np.float32)


def compute_isolation_forest_scores(
    train_loader,
    seed = 0,
):
  """Fits IsolationForest on training data and returns anomaly scores."""
  all_x = []
  for batch in train_loader:
    x_tr = batch[0] if isinstance(batch, (list, tuple)) else batch['x']
    all_x.append(x_tr.view(x_tr.shape[0], -1).cpu().numpy())
  x_mat = np.vstack(all_x)

  clf = sklearn.ensemble.IsolationForest(random_state=seed)
  clf.fit(x_mat)
  scores = -clf.score_samples(x_mat)
  return scores.astype(np.float32)


def retrain_model_with_indices_removed(
    model_fn,
    train_dataset,
    indices_to_remove,
    num_epochs,
    batch_size,
    lr,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    device = 'cpu',
):
  """High-level function: retrains a model from scratch with specified indices removed."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  model = model_fn().to(device)

  if optimizer_type.lower() == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps
    )
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  skip_set = set(indices_to_remove)
  n_total = len(train_dataset)
  kept_indices = [i for i in range(n_total) if i not in skip_set]

  if not kept_indices:
    return model

  model.train()
  for epoch in range(num_epochs):
    for start_idx in range(0, n_total, batch_size):
      end_idx = min(start_idx + batch_size, n_total)
      orig_batch_size = end_idx - start_idx
      chunk = [i for i in range(start_idx, end_idx) if i not in skip_set]

      if not chunk:
        continue

      if hasattr(train_dataset, 'tensors'):
        x_batch = train_dataset.tensors[0][chunk].to(device)
        y_batch = train_dataset.tensors[1][chunk].to(device)
      else:
        x_batch_list = []
        y_batch_list = []
        for idx in chunk:
          item = train_dataset[idx]
          if isinstance(item, (list, tuple)):
            x_batch_list.append(item[0])
            y_batch_list.append(item[1])
          else:
            x_batch_list.append(item['x'])
            y_batch_list.append(item['y'])

        x_batch = torch.stack(x_batch_list).to(device)
        if isinstance(y_batch_list[0], torch.Tensor):
          y_batch = torch.stack(y_batch_list).to(device)
        else:
          y_batch = torch.tensor(y_batch_list, device=device)

      out = model(x_batch)
      raw_loss = criterion(out, y_batch)
      loss = raw_loss * (len(chunk) / orig_batch_size)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  is_cf = bool(indices_to_remove)
  loop_type = (
      f'counterfactual model (indices_to_remove={list(indices_to_remove)})'
      if is_cf
      else 'full model'
  )
  utils.log_cuda_memory(f'After training loop with {loop_type}', device=device)
  return model


def retrain_model_with_indices_removed_randomized(
    model_fn,
    train_dataset,
    indices_to_remove = (),
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    device = 'cpu',
):
  """Retrains a model from scratch with specified indices removed, randomizing both initial point and batch structure with a seed."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

  model = model_fn().to(device)

  if optimizer_type.lower() == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps
    )
  else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

  skip_set = set(indices_to_remove) if indices_to_remove is not None else set()
  n_total = len(train_dataset)
  kept_indices = [i for i in range(n_total) if i not in skip_set]

  if not kept_indices:
    return model

  generator = torch.Generator()
  generator.manual_seed(seed)

  is_tensor_dataset = hasattr(train_dataset, 'tensors')

  model.train()
  for epoch in range(num_epochs):
    full_perm = torch.randperm(n_total, generator=generator).tolist()
    shuffled_indices = [idx for idx in full_perm if idx not in skip_set]
    num_shuffled = len(shuffled_indices)

    for start_idx in range(0, num_shuffled, batch_size):
      batch_indices = shuffled_indices[start_idx : start_idx + batch_size]
      if not batch_indices:
        continue

      if is_tensor_dataset:
        batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long)
        x_batch = train_dataset.tensors[0][batch_idx_tensor].to(device)
        y_batch = train_dataset.tensors[1][batch_idx_tensor].to(device)
      else:
        x_batch_list = []
        y_batch_list = []
        for idx in batch_indices:
          item = train_dataset[idx]
          if isinstance(item, (list, tuple)):
            x_batch_list.append(item[0])
            y_batch_list.append(item[1])
          elif isinstance(item, dict):
            x_key = (
                'x'
                if 'x' in item
                else (
                    'image'
                    if 'image' in item
                    else ('features' if 'features' in item else 'input')
                )
            )
            y_key = (
                'y'
                if 'y' in item
                else (
                    'label'
                    if 'label' in item
                    else ('targets' if 'targets' in item else 'target')
                )
            )
            x_batch_list.append(item[x_key])
            y_batch_list.append(item[y_key])
          else:
            raise TypeError(f'Unsupported dataset item type: {type(item)}')

        if isinstance(x_batch_list[0], torch.Tensor):
          x_batch = torch.stack(x_batch_list).to(device)
        else:
          x_batch = torch.tensor(np.array(x_batch_list), device=device)

        if isinstance(y_batch_list[0], torch.Tensor):
          y_batch = torch.stack(y_batch_list).to(device)
        else:
          y_batch = torch.tensor(np.array(y_batch_list), device=device)

      out = model(x_batch)
      loss = criterion(out, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  is_cf = bool(indices_to_remove)
  loop_type = (
      'randomized counterfactual model'
      f' (indices_to_remove={list(indices_to_remove)}, seed={seed})'
      if is_cf
      else f'randomized full model (seed={seed})'
  )
  utils.log_cuda_memory(f'After training loop with {loop_type}', device=device)
  return model


retrain_model_randomized = retrain_model_with_indices_removed_randomized


def evaluate_model_performance(
    model,
    data_loader,
    criterion = nn.functional.cross_entropy,
    device = 'cpu',
):
  """Evaluates model loss and classification accuracy on a dataset."""
  model = model.to(device)
  model.eval()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  with torch.no_grad():
    for batch in data_loader:
      if isinstance(batch, (list, tuple)):
        x_b, y_b = batch[0].to(device), batch[1].to(device)
      else:
        x_b, y_b = batch['x'].to(device), batch['y'].to(device)

      batch_size = x_b.shape[0]
      total_samples += batch_size

      out = model(x_b)
      loss = criterion(out, y_b) * batch_size
      total_loss += loss.item()

      if out.ndim > 1 and out.shape[1] > 1:
        preds = out.argmax(dim=1)
        correct += (preds == y_b).sum().item()

  avg_loss = total_loss / max(1, total_samples)
  accuracy = correct / max(1, total_samples)
  return avg_loss, accuracy


def evaluate_splits(
    model,
    train_loader = None,
    val_loader = None,
    test_loader = None,
    criterion = nn.functional.cross_entropy,
    device = 'cpu',
):
  """Evaluates an existing model across train, val, and test splits."""
  metrics = {}
  if train_loader is not None:
    metrics['train'] = evaluate_model_performance(
        model, train_loader, criterion, device
    )
  if val_loader is not None:
    metrics['val'] = evaluate_model_performance(
        model, val_loader, criterion, device
    )
  if test_loader is not None:
    metrics['test'] = evaluate_model_performance(
        model, test_loader, criterion, device
    )
  return metrics


def retrain_and_evaluate(
    model_fn,
    train_dataset,
    train_loader = None,
    val_loader = None,
    test_loader = None,
    indices_to_remove = (),
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    device = 'cpu',
):
  """Retrains a model with specified indices removed and evaluates across splits."""
  cleansed_model = retrain_model_with_indices_removed(
      model_fn=model_fn,
      train_dataset=train_dataset,
      indices_to_remove=indices_to_remove,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )
  metrics = evaluate_splits(
      cleansed_model,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      criterion=criterion,
      device=device,
  )
  del cleansed_model
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return metrics


def retrain_and_evaluate_randomized(
    model_fn,
    train_dataset,
    train_loader = None,
    val_loader = None,
    test_loader = None,
    indices_to_remove = (),
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    device = 'cpu',
):
  """Retrains a model with randomized batches and initial point, then evaluates across splits."""
  cleansed_model = retrain_model_with_indices_removed_randomized(
      model_fn=model_fn,
      train_dataset=train_dataset,
      indices_to_remove=indices_to_remove,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )
  metrics = evaluate_splits(
      cleansed_model,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      criterion=criterion,
      device=device,
  )
  del cleansed_model
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return metrics


def compute_scores_for_method(
    method,
    base_model_obj,
    val_loader = None,
    optimizer_type = 'adam',
    random_mask = None,
    candidate_indices = None,
    max_workers = 1,
    num_steps = None,
    device = 'cpu',
    use_reversible_transform = False,
    start_step = 0,
    quantized_base_model_obj = None,
    quantized_checkpoints = None,
    momentum_buffer = None,
    variance_buffer = None,
    quantization_scale = 1000000000000,
):
  """Computes influence or outlier scores for a single method."""
  if momentum_buffer is None and isinstance(base_model_obj, dict):
    momentum_buffer = base_model_obj.get('momentum_buffer')
  if variance_buffer is None and isinstance(base_model_obj, dict):
    variance_buffer = base_model_obj.get('variance_buffer')
  if quantized_checkpoints is None and isinstance(base_model_obj, dict):
    quantized_checkpoints = base_model_obj.get('quantized_checkpoints')
  if quantized_base_model_obj is None and isinstance(base_model_obj, dict):
    quantized_base_model_obj = base_model_obj.get('quantized_base_model_obj')
  if (
      quantization_scale == 1000000000000
      and isinstance(base_model_obj, dict)
      and 'quantization_scale' in base_model_obj
  ):
    quantization_scale = base_model_obj.get('quantization_scale', 1000000000000)

  if hasattr(base_model_obj, 'train_loader'):
    train_loader = base_model_obj.train_loader()
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
    lr = base_model_obj.lr() if hasattr(base_model_obj, 'lr') else 0.05
    beta_1 = (
        base_model_obj.beta_1()
        if hasattr(base_model_obj, 'beta_1')
        else 0.9
    )
    beta_2 = (
        base_model_obj.beta_2()
        if hasattr(base_model_obj, 'beta_2')
        else 0.999
    )
    eps = base_model_obj.eps() if hasattr(base_model_obj, 'eps') else 1e-8
    criterion = (
        base_model_obj.loss_fn()
        if hasattr(base_model_obj, 'loss_fn')
        else None
    )
    seed = base_model_obj.seed() if hasattr(base_model_obj, 'seed') else 0
  elif isinstance(base_model_obj, dict):
    train_loader = base_model_obj.get('train_loader') or base_model_obj.get('batches')
    checkpoints = base_model_obj.get('checkpoints', base_model_obj)
    momentum_lst = base_model_obj.get('momentum_lst')
    variance_lst = base_model_obj.get('variance_lst')
    hp = base_model_obj.get('hyperparams', {})
    lr = hp.get('lr', 0.05)
    beta_1 = hp.get('beta_1', 0.9)
    beta_2 = hp.get('beta_2', 0.999)
    eps = hp.get('eps', 1e-8)
    criterion = base_model_obj.get('loss_fn') or base_model_obj.get('criterion')
    seed = hp.get('seed', 0)
  else:
    raise TypeError(f'Unsupported base_model_obj type: {type(base_model_obj)}')

  if criterion is None:
    criterion = nn.functional.cross_entropy
  if train_loader is None:
    raise ValueError('train_loader is missing in base_model_obj.')

  if hasattr(train_loader, 'dataset'):
    num_train = len(train_loader.dataset)
  elif isinstance(train_loader, Sequence):
    num_train = sum(
        b[0].shape[0] if isinstance(b, (tuple, list)) else len(b['x'])
        for b in train_loader
    )
  else:
    num_train = len(train_loader)

  target_indices = (
      list(candidate_indices)
      if candidate_indices is not None
      else list(range(num_train))
  )

  if method.startswith('adam'):
    if momentum_lst is None or variance_lst is None:
      raise ValueError(f'momentum_lst and variance_lst required for {method}.')
    if method in ('adam', 'adam_recursive'):
      mode = 'recursive'
      remove_dv = False
    elif method == 'adam_recursive_nodv':
      mode = 'recursive_nodv'
      remove_dv = True
    elif (
        method in ('adam_forward', 'adam_forward_masked', 'adam_exact')
        or method.startswith('adam_masked')
        or 'exact' in method
        or 'forward' in method
    ):
      mode = 'forward'
      remove_dv = 'nodv' in method
    else:
      mode = (
          'forward'
          if ('forward' in method or 'exact' in method or 'masked' in method)
          else 'recursive'
      )
      remove_dv = 'nodv' in method

    if mode in ('recursive', 'recursive_nodv') and use_reversible_transform:
      if quantized_base_model_obj is not None:
        target_base_model = quantized_base_model_obj
        q_ckpts = (
            target_base_model.get_checkpoint_models()
            if hasattr(target_base_model, 'get_checkpoint_models')
            else target_base_model.get('checkpoints', target_base_model)
        )
      elif quantized_checkpoints is not None:
        if isinstance(quantized_checkpoints, dict):
          q_ckpt_models = {
              k: (v.model if hasattr(v, 'model') else v)
              for k, v in quantized_checkpoints.items()
          }
          q_mom = {
              k: (v.momentum if hasattr(v, 'momentum') else v)
              for k, v in quantized_checkpoints.items()
              if hasattr(v, 'momentum')
          }
          q_var = {
              k: (v.variance if hasattr(v, 'variance') else v)
              for k, v in quantized_checkpoints.items()
              if hasattr(v, 'variance')
          }
        else:
          q_ckpt_models = quantized_checkpoints
          q_mom = momentum_lst
          q_var = variance_lst

        target_base_model = {
            'train_loader': train_loader,
            'checkpoints': q_ckpt_models,
            'momentum_lst': q_mom if q_mom else momentum_lst,
            'variance_lst': q_var if q_var else variance_lst,
            'loss_fn': criterion,
            'hyperparams': {
                'lr': lr,
                'beta_1': beta_1,
                'beta_2': beta_2,
                'eps': eps,
                'seed': seed,
            },
        }
        q_ckpts = q_ckpt_models
      else:
        target_base_model = base_model_obj
        q_ckpts = checkpoints

      ckpt_ref_rec = (
          q_ckpts[-1]
          if (
              isinstance(q_ckpts, (list, tuple, dict))
              and -1 in q_ckpts
              or isinstance(q_ckpts, (list, tuple))
          )
          else (
              q_ckpts[max(q_ckpts.keys())]
              if isinstance(q_ckpts, dict)
              else q_ckpts
          )
      )
      u_params = compute_validation_gradient(
          ckpt_ref_rec, criterion, val_loader, device
      )
      query_vec = torch.cat([g.contiguous().view(-1) for g in u_params])

      if mode == 'recursive':
        results_dict = infl_adam.recursive_update(
            base_model=target_base_model,
            inds=target_indices,
            query_vec=query_vec,
            num_steps=num_steps,
            device=device,
            momentum_buffer=momentum_buffer,
            variance_buffer=variance_buffer,
            reversible_scale=quantization_scale,
            start_step=start_step,
        )
      else:
        results_dict = infl_adam.recursive_update_nodv(
            base_model=target_base_model,
            inds=target_indices,
            query_vec=query_vec,
            num_steps=num_steps,
            device=device,
            momentum_buffer=momentum_buffer,
            variance_buffer=variance_buffer,
            reversible_scale=quantization_scale,
            start_step=start_step,
        )

      if candidate_indices is not None:
        return {i: results_dict[i].item() for i in target_indices}
      return np.array(
          [results_dict[i].item() for i in range(num_train)], dtype=np.float32
      )

    ckpt_ref = (
        checkpoints[-1]
        if isinstance(checkpoints, (list, tuple, dict)) and -1 in checkpoints or isinstance(checkpoints, (list, tuple))
        else (checkpoints[max(checkpoints.keys())] if isinstance(checkpoints, dict) else checkpoints)
    )
    u_params = compute_validation_gradient(
        ckpt_ref, criterion, val_loader, device
    )
    query_vec = torch.cat([g.contiguous().view(-1) for g in u_params])

    if mode == 'recursive':
      results_dict = infl_adam.recursive_update(
          base_model=base_model_obj,
          inds=target_indices,
          query_vec=query_vec,
          num_steps=num_steps,
          device=device,
          start_step=start_step,
      )
      if candidate_indices is not None:
        return {i: results_dict[i].item() for i in target_indices}
      return np.array(
          [results_dict[i].item() for i in range(num_train)], dtype=np.float32
      )
    elif mode == 'recursive_nodv':
      results_dict = infl_adam.recursive_update_nodv(
          base_model=base_model_obj,
          inds=target_indices,
          query_vec=query_vec,
          num_steps=num_steps,
          device=device,
          start_step=start_step,
      )
      if candidate_indices is not None:
        return {i: results_dict[i].item() for i in target_indices}
      return np.array(
          [results_dict[i].item() for i in range(num_train)], dtype=np.float32
      )
    elif mode == 'forward':
      if random_mask is not None:
        random_mask_flat = torch.cat([r.flatten() for r in random_mask])
        query_vec_forward = query_vec[random_mask_flat]
      else:
        query_vec_forward = query_vec

      dv_opt = 'first_order' if 'first_order' in method else 'exact'
      theta_opt = 'first_order' if 'first_order' in method else 'exact'

      # Chunk target_indices into batches of size m <= 64 (decoupled from train batch size)
      m = min(getattr(train_loader, 'batch_size', None) or 64, 64)
      infl_arr = np.zeros(num_train, dtype=np.float32)

      def _eval_chunk_forward(chunk):
        if not chunk:
          return []
        res = infl_adam.forward_update(
            base_model=base_model_obj,
            ind=chunk,
            num_steps=num_steps,
            remove_dv=remove_dv,
            dv_option=dv_opt,
            theta_option=theta_opt,
            random_mask=random_mask,
            device=device,
        )
        theta_diff = res[0] if isinstance(res, tuple) else res
        target_dev = (
            theta_diff.device
            if isinstance(theta_diff, torch.Tensor)
            else (theta_diff[0].device if theta_diff else device)
        )
        q_vec = query_vec_forward.to(target_dev)
        if isinstance(theta_diff, torch.Tensor) and theta_diff.dim() == 2:
          infl_vals = torch.mv(theta_diff, q_vec).cpu().tolist()
        elif isinstance(theta_diff, torch.Tensor):
          infl_vals = [torch.dot(q_vec, theta_diff).item()]
        else:
          flat_theta_diff = torch.cat(
              [p.contiguous().view(-1) for p in theta_diff]
          )
          infl_vals = [torch.dot(q_vec, flat_theta_diff).item()]
          del flat_theta_diff
        del res, theta_diff, q_vec
        utils.garbage_collect(device=device)
        return list(zip(chunk, infl_vals))

      chunks = [
          target_indices[i : i + m] for i in range(0, len(target_indices), m)
      ]

      if max_workers > 1 and len(chunks) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
          chunk_results = list(executor.map(_eval_chunk_forward, chunks))
      else:
        chunk_results = [_eval_chunk_forward(chunk) for chunk in chunks]

      if candidate_indices is not None:
        return {
            idx: infl_val
            for chunk_res in chunk_results
            for idx, infl_val in chunk_res
        }

      for chunk_res in chunk_results:
        for idx, infl_val in chunk_res:
          if 0 <= idx < num_train:
            infl_arr[idx] = infl_val
      return infl_arr

  elif method == 'tracin_adam' or (
      method == 'tracin' and optimizer_type == 'adam'
  ):
    return compute_tracin_adam_influence(
        base_model_obj=base_model_obj,
        val_loader=val_loader,
        candidate_indices=candidate_indices,
        num_steps=num_steps,
        device=device,
    )

  elif method == 'tracin_sgd' or (
      method == 'tracin' and optimizer_type == 'sgd'
  ):
    return compute_tracin_sgd_influence(
        base_model_obj=base_model_obj,
        val_loader=val_loader,
        candidate_indices=candidate_indices,
        num_steps=num_steps,
        device=device,
    )

  elif method in ('sgd', 'sgd_all', 'sgd_last'):
    ckpt_ref = (
        checkpoints[-1]
        if isinstance(checkpoints, (list, tuple, dict)) and -1 in checkpoints or isinstance(checkpoints, (list, tuple))
        else (checkpoints[max(checkpoints.keys())] if isinstance(checkpoints, dict) else checkpoints)
    )
    u_params = compute_validation_gradient(
        ckpt_ref, criterion, val_loader, device
    )
    query_vec = torch.cat([g.contiguous().view(-1) for g in u_params])

    if num_steps is not None:
      total_steps = num_steps
    elif isinstance(checkpoints, (list, tuple)) and len(checkpoints) > 1:
      total_steps = len(checkpoints) - 1
    elif isinstance(checkpoints, dict):
      pos_keys = [k for k in checkpoints.keys() if isinstance(k, int) and k > 0]
      total_steps = max(pos_keys) if pos_keys else 1
    else:
      total_steps = 1

    has_all_ckpts = (
        isinstance(checkpoints, (list, tuple))
        and len(checkpoints) >= total_steps + 1
    ) or (
        isinstance(checkpoints, dict)
        and all(k in checkpoints for k in range(total_steps + 1))
    )

    if has_all_ckpts:
      results_dict = infl_sgd.recursive_update(
          base_model=base_model_obj,
          inds=target_indices,
          query_vec=query_vec,
          num_steps=num_steps,
          device=device,
          start_step=start_step,
      )
      if candidate_indices is not None:
        return {i: results_dict[i].item() for i in target_indices}
      return np.array(
          [results_dict[i].item() for i in range(num_train)], dtype=np.float32
      )
    else:
      if random_mask is not None:
        random_mask_flat = torch.cat([r.flatten() for r in random_mask])
        query_vec_forward = query_vec[random_mask_flat]
      else:
        query_vec_forward = query_vec

      # Chunk target_indices into batches of size m <= 64 (decoupled from train batch size)
      m = min(getattr(train_loader, 'batch_size', None) or 64, 64)
      infl_arr = np.zeros(num_train, dtype=np.float32)

      def _eval_chunk_forward_sgd(chunk):
        if not chunk:
          return []
        res = infl_sgd.forward_update(
            base_model=base_model_obj,
            ind=chunk,
            num_steps=num_steps,
            random_mask=random_mask,
            device=device,
        )
        theta_diff = res[0] if isinstance(res, tuple) else res
        target_dev = (
            theta_diff.device
            if isinstance(theta_diff, torch.Tensor)
            else (theta_diff[0].device if theta_diff else device)
        )
        q_vec = query_vec_forward.to(target_dev)
        if isinstance(theta_diff, torch.Tensor) and theta_diff.dim() == 2:
          infl_vals = torch.mv(theta_diff, q_vec).cpu().tolist()
        elif isinstance(theta_diff, torch.Tensor):
          infl_vals = [torch.dot(q_vec, theta_diff).item()]
        else:
          flat_theta_diff = torch.cat(
              [p.contiguous().view(-1) for p in theta_diff]
          )
          infl_vals = [torch.dot(q_vec, flat_theta_diff).item()]
          del flat_theta_diff
        del res, theta_diff, q_vec
        utils.garbage_collect(device=device)
        return list(zip(chunk, infl_vals))

      chunks = [
          target_indices[i : i + m] for i in range(0, len(target_indices), m)
      ]

      if max_workers > 1 and len(chunks) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
          chunk_results = list(executor.map(_eval_chunk_forward_sgd, chunks))
      else:
        chunk_results = [_eval_chunk_forward_sgd(chunk) for chunk in chunks]

      if candidate_indices is not None:
        return {
            idx: infl_val
            for chunk_res in chunk_results
            for idx, infl_val in chunk_res
        }

      for chunk_res in chunk_results:
        for idx, infl_val in chunk_res:
          if 0 <= idx < num_train:
            infl_arr[idx] = infl_val
      return infl_arr

  elif method == 'icml':
    ckpt_ref = (
        checkpoints[-1]
        if isinstance(checkpoints, (list, tuple, dict)) and -1 in checkpoints or isinstance(checkpoints, (list, tuple))
        else (checkpoints[max(checkpoints.keys())] if isinstance(checkpoints, dict) else checkpoints)
    )
    scores = compute_icml_influence(
        ckpt_ref,
        train_loader,
        val_loader,
        criterion,
        lr=lr,
        device=device,
    )
    if candidate_indices is not None:
      return {i: float(scores[i]) for i in candidate_indices}
    return scores

  elif method == 'ae':
    scores = compute_autoencoder_outlier_scores(train_loader, device=device)
    if candidate_indices is not None:
      return {i: float(scores[i]) for i in candidate_indices}
    return scores

  elif method == 'iso':
    scores = compute_isolation_forest_scores(train_loader, seed=seed)
    if candidate_indices is not None:
      return {i: float(scores[i]) for i in candidate_indices}
    return scores

  elif method == 'random':
    rng = np.random.default_rng(seed)
    scores = rng.random(num_train).astype(np.float32)
    if candidate_indices is not None:
      return {i: float(scores[i]) for i in candidate_indices}
    return scores

  else:
    raise ValueError(f"Unknown influence/cleansing method: '{method}'")


def compute_scores_for_methods(
    methods,
    base_model_obj,
    val_loader = None,
    optimizer_type = 'adam',
    random_mask = None,
    candidate_indices = None,
    max_workers = 1,
    num_steps = None,
    device = 'cpu',
    use_reversible_transform = False,
    start_step = 0,
    quantized_base_model_obj = None,
    quantized_checkpoints = None,
    momentum_buffer = None,
    variance_buffer = None,
    quantization_scale = 1000000000000,
):
  """Computes scores for a list of methods, sharing intermediate computations when possible."""
  scores_dict = {}

  for m_name in methods:
    logging.info('Computing scores for method: %s', m_name)
    scores_dict[m_name] = compute_scores_for_method(
        method=m_name,
        base_model_obj=base_model_obj,
        val_loader=val_loader,
        optimizer_type=optimizer_type,
        random_mask=random_mask,
        candidate_indices=candidate_indices,
        max_workers=max_workers,
        num_steps=num_steps,
        device=device,
        use_reversible_transform=use_reversible_transform,
        start_step=start_step,
        quantized_base_model_obj=quantized_base_model_obj,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=quantization_scale,
    )
    logging.info('Done computing scores for method: %s', m_name)
  return scores_dict


def select_indices_for_removal(
    scores,
    method,
    candidate_indices = None,
    k = None,
    threshold = 0.0,
    max_removals = None,
):
  """Selects training sample indices for removal based on method scores.

  For outlier methods ('ae', 'iso'), higher scores indicate anomaly.
  For influence methods, lower (negative) scores indicate loss reduction
  (removing sample improves val loss).
  """
  num_samples = len(scores)
  if candidate_indices is None:
    if isinstance(scores, (dict, Mapping)):
      candidate_indices = list(scores.keys())
    else:
      candidate_indices = list(range(num_samples))

  is_outlier = method in ('ae', 'iso')

  if k is not None:
    cand_scores = [(i, float(scores[i])) for i in candidate_indices]
    cand_scores.sort(key=lambda item: item[1], reverse=is_outlier)
    return [i for i, _ in cand_scores[:k]]

  # Threshold / automatic selection
  if is_outlier:
    cand_scores = [(i, float(scores[i])) for i in candidate_indices]
    cand_scores.sort(key=lambda item: item[1], reverse=True)
    if max_removals is not None:
      return [i for i, _ in cand_scores[:max_removals]]
    else:
      default_k = max(1, int(len(candidate_indices) * 0.1))
      return [i for i, _ in cand_scores[:default_k]]
  else:
    negative_indices = [i for i in candidate_indices if scores[i] < threshold]
    negative_indices.sort(key=lambda i: scores[i])
    if max_removals is not None and len(negative_indices) > max_removals:
      return negative_indices[:max_removals]
    return negative_indices


def evaluate_cleansing_for_indices(
    model_fn,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    indices_to_remove,
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    retrain_seeds = None,
    device = 'cpu',
):
  """High-level function: takes a set of indices, retrains model, and evaluates change in loss/acc."""
  retrain_kwargs = dict(
      model_fn=model_fn,
      train_dataset=train_dataset,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )
  base_metrics = retrain_and_evaluate(indices_to_remove=[], **retrain_kwargs)
  cleansed_metrics = retrain_and_evaluate(
      indices_to_remove=indices_to_remove, **retrain_kwargs
  )

  res = {
      'baseline_val': base_metrics['val'],
      'baseline_train': base_metrics['train'],
      'baseline_test': base_metrics['test'],
      'cleansed_val': cleansed_metrics['val'],
      'cleansed_train': cleansed_metrics['train'],
      'cleansed_test': cleansed_metrics['test'],
      'loss_change_val': (
          cleansed_metrics['val'][0] - base_metrics['val'][0],
          cleansed_metrics['val'][1] - base_metrics['val'][1],
      ),
      'loss_change_test': (
          cleansed_metrics['test'][0] - base_metrics['test'][0],
          cleansed_metrics['test'][1] - base_metrics['test'][1],
      ),
  }
  if retrain_seeds:
    for s in retrain_seeds:
      rand_kw = dict(retrain_kwargs)
      rand_kw['seed'] = s
      r_cleansed = retrain_and_evaluate_randomized(
          indices_to_remove=indices_to_remove, **rand_kw
      )
      res[f'cleansed_val_seed_{s}'] = r_cleansed['val']
      res[f'cleansed_train_seed_{s}'] = r_cleansed['train']
      res[f'cleansed_test_seed_{s}'] = r_cleansed['test']
      res[f'loss_change_val_seed_{s}'] = (
          r_cleansed['val'][0] - base_metrics['val'][0],
          r_cleansed['val'][1] - base_metrics['val'][1],
      )
      res[f'loss_change_test_seed_{s}'] = (
          r_cleansed['test'][0] - base_metrics['test'][0],
          r_cleansed['test'][1] - base_metrics['test'][1],
      )
  return res


def evaluate_data_cleansing(
    model_fn,
    checkpoints,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    scores_dict = None,
    corrupted_indices = None,
    momentum_lst = None,
    variance_lst = None,
    criterion = nn.functional.cross_entropy,
    lr = 0.05,
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    optimizer_type = 'adam',
    num_epochs = 20,
    batch_size = 64,
    k_list = (1, 3, 6, 10, 30, 60, 100),
    methods = (
        'adam_recursive',
        'adam_recursive_nodv',
        'adam_exact',
        'tracin_adam',
        'tracin_sgd',
        'sgd_all',
        'sgd_last',
        'icml',
        'random',
        'ae',
        'iso',
    ),
    candidate_indices = None,
    seed = 0,
    retrain_seeds = None,
    random_mask = None,
    max_workers = 1,
    num_steps = None,
    device = 'cpu',
    eval_auto = True,
    auto_threshold = 0.0,
    max_auto_removals = None,
    eval_oracle = True,
    return_details = False,
    use_reversible_transform = False,
    start_step = 0,
    ignore_first = False,
    quantized_base_model_obj = None,
    quantized_checkpoints = None,
    momentum_buffer = None,
    variance_buffer = None,
    quantization_scale = 1000000,
):
  """Runs Data Cleansing evaluation across methods and k values (both fixed k and automated criterion)."""
  if scores_dict is None:
    scores_dict = {}
  else:
    scores_dict = dict(scores_dict)

  num_batches = len(utils.prepare_batches(train_loader))
  effective_start_step = num_batches if ignore_first else start_step

  missing_methods = [m for m in methods if m not in scores_dict]
  if missing_methods:
    logging.info('Computing scores for missing methods: %s', missing_methods)
    base_model_artifacts = {
        'train_loader': train_loader,
        'checkpoints': checkpoints,
        'momentum_lst': momentum_lst,
        'variance_lst': variance_lst,
        'loss_fn': criterion,
        'hyperparams': {
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'eps': eps,
            'seed': seed,
        },
    }
    new_scores = compute_scores_for_methods(
        methods=missing_methods,
        base_model_obj=base_model_artifacts,
        val_loader=val_loader,
        optimizer_type=optimizer_type,
        random_mask=random_mask,
        candidate_indices=candidate_indices,
        max_workers=max_workers,
        num_steps=num_steps,
        device=device,
        use_reversible_transform=use_reversible_transform,
        start_step=effective_start_step,
        quantized_base_model_obj=quantized_base_model_obj,
        quantized_checkpoints=quantized_checkpoints,
        momentum_buffer=momentum_buffer,
        variance_buffer=variance_buffer,
        quantization_scale=quantization_scale,
    )
    scores_dict.update(new_scores)

  num_train = len(train_dataset)
  if candidate_indices is None:
    candidate_indices = list(range(num_train))

  results = {}
  retrain_cache: dict[
      tuple[tuple[int, Ellipsis], int, bool], dict[str, tuple[float, float]]
  ] = {}

  retrain_kwargs = dict(
      model_fn=model_fn,
      train_dataset=train_dataset,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )

  def _get_retrain_metrics(indices, s, randomized):
    key = (tuple(sorted(indices)), s, randomized)
    if key not in retrain_cache:
      kw = dict(retrain_kwargs)
      kw['seed'] = s
      if randomized:
        retrain_cache[key] = retrain_and_evaluate_randomized(
            indices_to_remove=indices, **kw
        )
      else:
        retrain_cache[key] = retrain_and_evaluate(
            indices_to_remove=indices, **kw
        )
    return retrain_cache[key]

  # Baseline evaluation (k=0 removed)
  base_eval = evaluate_splits(
      checkpoints[-1],
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      criterion=criterion,
      device=device,
  )
  tr_loss, tr_acc = base_eval['train']
  val_loss, val_acc = base_eval['val']
  te_loss, te_acc = base_eval['test']

  baseline_dict = {
      'train_loss': tr_loss,
      'train_acc': tr_acc,
      'val_loss': val_loss,
      'val_acc': val_acc,
      'test_loss': te_loss,
      'test_acc': te_acc,
  }
  results['baseline'] = (
      tr_loss,
      val_loss,
      te_loss,
      tr_acc,
      val_acc,
      te_acc,
  )

  # Evaluate exact removal of corrupted indices if provided
  corrupted_set = (
      set(corrupted_indices) if corrupted_indices is not None else set()
  )
  oracle_metrics = None
  oracle_delta = None
  randomized_oracle = {}
  randomized_oracle_delta = {}

  if eval_oracle and corrupted_set:
    c_metrics = _get_retrain_metrics(
        sorted(list(corrupted_set)), s=seed, randomized=False
    )
    o_tr_loss, o_tr_acc = c_metrics['train']
    o_val_loss, o_val_acc = c_metrics['val']
    o_te_loss, o_te_acc = c_metrics['test']

    results[('remove_corrupted', len(corrupted_set))] = (
        o_tr_loss,
        o_val_loss,
        o_te_loss,
        o_tr_acc,
        o_val_acc,
        o_te_acc,
    )
    oracle_metrics = {
        'train_loss': o_tr_loss,
        'train_acc': o_tr_acc,
        'val_loss': o_val_loss,
        'val_acc': o_val_acc,
        'test_loss': o_te_loss,
        'test_acc': o_te_acc,
    }
    oracle_delta = {
        'train_loss_change': o_tr_loss - tr_loss,
        'val_loss_change': o_val_loss - val_loss,
        'val_acc_change': o_val_acc - val_acc,
        'test_loss_change': o_te_loss - te_loss,
        'test_acc_change': o_te_acc - te_acc,
    }

    if retrain_seeds:
      for s in retrain_seeds:
        r_c_metrics = _get_retrain_metrics(
            sorted(list(corrupted_set)), s=s, randomized=True
        )
        r_o_tr_loss, r_o_tr_acc = r_c_metrics['train']
        r_o_val_loss, r_o_val_acc = r_c_metrics['val']
        r_o_te_loss, r_o_te_acc = r_c_metrics['test']
        results[(f'remove_corrupted_seed_{s}', len(corrupted_set))] = (
            r_o_tr_loss,
            r_o_val_loss,
            r_o_te_loss,
            r_o_tr_acc,
            r_o_val_acc,
            r_o_te_acc,
        )
        randomized_oracle[s] = {
            'train_loss': r_o_tr_loss,
            'train_acc': r_o_tr_acc,
            'val_loss': r_o_val_loss,
            'val_acc': r_o_val_acc,
            'test_loss': r_o_te_loss,
            'test_acc': r_o_te_acc,
        }
        randomized_oracle_delta[s] = {
            'train_loss_change': r_o_tr_loss - tr_loss,
            'val_loss_change': r_o_val_loss - val_loss,
            'val_acc_change': r_o_val_acc - val_acc,
            'test_loss_change': r_o_te_loss - te_loss,
            'test_acc_change': r_o_te_acc - te_acc,
        }

  detection_metrics_dict = {}
  auto_cleansing_dict = {}

  # Resolve effective numeric k_list (ensuring k = len(corrupted_set) is included without duplicate runs)
  numeric_k_list = []
  seen_k = set()
  for k in k_list:
    if k == 'auto' or k is None:
      continue
    k_int = int(k)
    if k_int not in seen_k:
      seen_k.add(k_int)
      numeric_k_list.append(k_int)

  if corrupted_set and len(corrupted_set) not in seen_k:
    numeric_k_list.append(len(corrupted_set))
    numeric_k_list.sort()

  # Evaluate each method for top-k and automatic criterion
  for method in methods:
    if method not in scores_dict:
      continue
    scores = scores_dict[method]

    # 1. Fixed-k evaluations
    for k_int in numeric_k_list:
      if k_int > num_train:
        continue
      top_k_indices = select_indices_for_removal(
          scores, method, candidate_indices=candidate_indices, k=k_int
      )
      c_metrics = _get_retrain_metrics(top_k_indices, s=seed, randomized=False)
      results[(method, k_int)] = (
          c_metrics['train'][0],
          c_metrics['val'][0],
          c_metrics['test'][0],
          c_metrics['train'][1],
          c_metrics['val'][1],
          c_metrics['test'][1],
      )
      if retrain_seeds:
        for s in retrain_seeds:
          r_metrics = _get_retrain_metrics(top_k_indices, s=s, randomized=True)
          results[(f'{method}_seed_{s}', k_int)] = (
              r_metrics['train'][0],
              r_metrics['val'][0],
              r_metrics['test'][0],
              r_metrics['train'][1],
              r_metrics['val'][1],
              r_metrics['test'][1],
          )

    # 2. Automated criterion-based evaluation
    if eval_auto or ('auto' in k_list):
      auto_indices = select_indices_for_removal(
          scores,
          method,
          candidate_indices=candidate_indices,
          threshold=auto_threshold,
          max_removals=max_auto_removals,
      )
      c_metrics = _get_retrain_metrics(auto_indices, s=seed, randomized=False)
      c_tr_loss, c_tr_acc = c_metrics['train']
      c_val_loss, c_val_acc = c_metrics['val']
      c_te_loss, c_te_acc = c_metrics['test']

      results[(method, 'auto')] = (
          c_tr_loss,
          c_val_loss,
          c_te_loss,
          c_tr_acc,
          c_val_acc,
          c_te_acc,
      )

      # Detection metrics against ground-truth corrupted indices
      removed_set = set(auto_indices)
      tp = len(removed_set & corrupted_set)
      num_removed = len(removed_set)
      num_corrupted = len(corrupted_set)
      precision = tp / max(1, num_removed) if num_removed > 0 else 0.0
      recall = tp / max(1, num_corrupted) if num_corrupted > 0 else 0.0
      f1 = (
          (2 * precision * recall / (precision + recall))
          if (precision + recall) > 0
          else 0.0
      )
      part_metrics = compute_partitioned_corrupted_metrics(
          scores,
          corrupted_set,
          is_outlier_score=(method in ('ae', 'iso')),
      )
      det_dict = {
          'precision': precision,
          'recall': recall,
          'f1_score': f1,
          'true_positives': tp,
          'num_corrupted': num_corrupted,
          'num_removed': num_removed,
      }
      det_dict.update(part_metrics)
      method_auto = {
          'baseline': baseline_dict,
          'cleansed': {
              'train_loss': c_tr_loss,
              'train_acc': c_tr_acc,
              'val_loss': c_val_loss,
              'val_acc': c_val_acc,
              'test_loss': c_te_loss,
              'test_acc': c_te_acc,
          },
          'delta': {
              'train_loss_change': c_tr_loss - tr_loss,
              'val_loss_change': c_val_loss - val_loss,
              'val_acc_change': c_val_acc - val_acc,
              'test_loss_change': c_te_loss - te_loss,
              'test_acc_change': c_te_acc - te_acc,
          },
          'selected_indices': auto_indices,
          'estimated_loss_changes': {
              i: float(scores[i]) for i in candidate_indices
          },
          'detection_metrics': det_dict,
      }
      if oracle_metrics is not None:
        method_auto['oracle'] = oracle_metrics
        method_auto['oracle_delta'] = oracle_delta

      if retrain_seeds:
        randomized_cleansed = {}
        randomized_delta = {}
        for s in retrain_seeds:
          r_metrics = _get_retrain_metrics(auto_indices, s=s, randomized=True)
          r_tr_l, r_tr_a = r_metrics['train']
          r_val_l, r_val_a = r_metrics['val']
          r_te_l, r_te_a = r_metrics['test']
          results[(f'{method}_seed_{s}', 'auto')] = (
              r_tr_l,
              r_val_l,
              r_te_l,
              r_tr_a,
              r_val_a,
              r_te_a,
          )
          randomized_cleansed[s] = {
              'train_loss': r_tr_l,
              'train_acc': r_tr_a,
              'val_loss': r_val_l,
              'val_acc': r_val_a,
              'test_loss': r_te_l,
              'test_acc': r_te_a,
          }
          randomized_delta[s] = {
              'train_loss_change': r_tr_l - tr_loss,
              'val_loss_change': r_val_l - val_loss,
              'val_acc_change': r_val_a - val_acc,
              'test_loss_change': r_te_l - te_loss,
              'test_acc_change': r_te_a - te_acc,
          }
        method_auto['randomized_cleansed'] = randomized_cleansed
        method_auto['randomized_delta'] = randomized_delta
        if eval_oracle and corrupted_set:
          method_auto['randomized_oracle'] = randomized_oracle
          method_auto['randomized_oracle_delta'] = randomized_oracle_delta

      auto_cleansing_dict[method] = method_auto
    elif corrupted_set:
      # Compute detection metrics without retraining if auto cleansing retraining is disabled
      auto_indices = select_indices_for_removal(
          scores,
          method,
          candidate_indices=candidate_indices,
          threshold=auto_threshold,
          max_removals=max_auto_removals,
      )
      removed_set = set(auto_indices)
      tp = len(removed_set & corrupted_set)
      num_removed = len(removed_set)
      num_corrupted = len(corrupted_set)
      precision = tp / max(1, num_removed) if num_removed > 0 else 0.0
      recall = tp / max(1, num_corrupted) if num_corrupted > 0 else 0.0
      f1 = (
          (2 * precision * recall / (precision + recall))
          if (precision + recall) > 0
          else 0.0
      )
      part_metrics = compute_partitioned_corrupted_metrics(
          scores,
          corrupted_set,
          is_outlier_score=(method in ('ae', 'iso')),
      )
      det_dict = {
          'precision': precision,
          'recall': recall,
          'f1_score': f1,
          'true_positives': tp,
          'num_corrupted': num_corrupted,
          'num_removed': num_removed,
      }
      det_dict.update(part_metrics)
      detection_metrics_dict[method] = det_dict

  if return_details:
    ret_details = {
        'results': results,
        'baseline': baseline_dict,
        'auto_cleansing': auto_cleansing_dict,
        'detection_metrics': detection_metrics_dict,
    }
    if oracle_metrics is not None:
      ret_details['oracle'] = oracle_metrics
      ret_details['oracle_delta'] = oracle_delta
    if retrain_seeds and eval_oracle and corrupted_set:
      ret_details['randomized_oracle'] = randomized_oracle
      ret_details['randomized_oracle_delta'] = randomized_oracle_delta
    return ret_details

  return results


def evaluate_corruption_and_cleansing(
    model_fn,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    checkpoints,
    scores = None,
    corrupted_indices = None,
    momentum_lst = None,
    variance_lst = None,
    optimizer_type = 'adam',
    influence_method = 'adam_recursive',
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    candidate_indices = None,
    max_removals = None,
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    momentum = 0.0,
    criterion = nn.functional.cross_entropy,
    eval_oracle = True,
    seed = 0,
    retrain_seeds = None,
    random_mask = None,
    device = 'cpu',
):
  r"""Executes corruption and counterfactual data cleansing experiment.

  Unified single-method wrapper delegating to evaluate_data_cleansing.
  """
  scores_dict = {influence_method: scores} if scores is not None else None
  details = evaluate_data_cleansing(
      model_fn=model_fn,
      checkpoints=checkpoints,
      train_dataset=train_dataset,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      scores_dict=scores_dict,
      corrupted_indices=corrupted_indices,
      momentum_lst=momentum_lst,
      variance_lst=variance_lst,
      criterion=criterion,
      lr=lr,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      optimizer_type=optimizer_type,
      num_epochs=num_epochs,
      batch_size=batch_size,
      k_list=[],
      methods=[influence_method],
      candidate_indices=candidate_indices,
      seed=seed,
      retrain_seeds=retrain_seeds,
      random_mask=random_mask,
      device=device,
      eval_auto=True,
      max_auto_removals=max_removals,
      eval_oracle=eval_oracle,
      return_details=True,
  )
  return details['auto_cleansing'].get(influence_method, {})


def save_scores(
    scores,
    output_path,
):
  """Saves influence or counterfactual scores to a file (npz, pt, json, or npy)."""
  parent_dir = os.path.dirname(output_path)
  if parent_dir:
    pipeline._make_dirs(parent_dir)
  if output_path.endswith('.pt') or output_path.endswith('.pth'):
    buf = io.BytesIO()
    torch.save(scores, buf)
    with pipeline._open_file(output_path, 'wb') as f:
      f.write(buf.getvalue())
  elif output_path.endswith('.npz'):
    buf = io.BytesIO()
    if isinstance(scores, (dict, Mapping)):
      formatted_dict = {}
      for k, v in scores.items():
        if isinstance(v, dict):
          indices = np.array(list(v.keys()), dtype=np.int64)
          vals = np.array(list(v.values()), dtype=np.float32)
          formatted_dict[k] = vals
          formatted_dict[f'{k}__indices'] = indices
          formatted_dict[f'{k}__values'] = vals
        else:
          formatted_dict[k] = np.asarray(v, dtype=np.float32)
      np.savez_compressed(buf, **formatted_dict)
    else:
      np.savez_compressed(buf, scores=np.asarray(scores, dtype=np.float32))
    with pipeline._open_file(output_path, 'wb') as f:
      f.write(buf.getvalue())
  elif output_path.endswith('.npy'):
    buf = io.BytesIO()
    np.save(buf, np.asarray(scores, dtype=np.float32))
    with pipeline._open_file(output_path, 'wb') as f:
      f.write(buf.getvalue())
  elif output_path.endswith('.json'):

    def _json_convert(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      if isinstance(obj, (np.floating, float)):
        return float(obj)
      if isinstance(obj, (np.integer, int)):
        return int(obj)
      if isinstance(obj, (dict, Mapping)):
        return {str(k): _json_convert(v) for k, v in obj.items()}
      return obj

    with pipeline._open_file(output_path, 'w') as f:
      json.dump(_json_convert(scores), f, indent=2)
  else:
    buf = io.BytesIO()
    torch.save(scores, buf)
    with pipeline._open_file(output_path, 'wb') as f:
      f.write(buf.getvalue())


def load_scores(
    input_path,
):
  """Loads influence or counterfactual scores from a file (npz, pt, npy, json)."""
  if input_path.endswith('.pt') or input_path.endswith('.pth'):
    with pipeline._open_file(input_path, 'rb') as f:
      buf = io.BytesIO(f.read())
    loaded = torch.load(buf, map_location='cpu')
    if isinstance(loaded, dict):
      res = {}
      for k, v in loaded.items():
        if isinstance(v, torch.Tensor):
          res[k] = v.cpu().numpy().astype(np.float32)
        elif isinstance(v, np.ndarray):
          res[k] = v.astype(np.float32)
        elif isinstance(v, dict):
          res[k] = v
        else:
          res[k] = np.asarray(v, dtype=np.float32)
      return res
    elif isinstance(loaded, torch.Tensor):
      return loaded.cpu().numpy().astype(np.float32)
    return loaded

  elif input_path.endswith('.npz'):
    with pipeline._open_file(input_path, 'rb') as f:
      buf = io.BytesIO(f.read())
    npz_data = np.load(buf)
    res = {}
    keys = list(npz_data.files)
    paired_indices = {
        k.replace('__indices', ''): k for k in keys if k.endswith('__indices')
    }
    for k in keys:
      if k.endswith('__indices') or k.endswith('__values'):
        continue
      res[k] = npz_data[k].astype(np.float32)
    for method_name, idx_key in paired_indices.items():
      val_key = f'{method_name}__values'
      if val_key in keys:
        indices = npz_data[idx_key]
        vals = npz_data[val_key]
        res[method_name] = {int(i): float(v) for i, v in zip(indices, vals)}
    return res

  elif input_path.endswith('.npy'):
    with pipeline._open_file(input_path, 'rb') as f:
      buf = io.BytesIO(f.read())
    return np.load(buf).astype(np.float32)

  elif input_path.endswith('.json'):
    with pipeline._open_file(input_path, 'r') as f:
      loaded = json.load(f)
    if isinstance(loaded, dict):
      res = {}
      for k, v in loaded.items():
        if isinstance(v, list):
          res[k] = np.array(v, dtype=np.float32)
        elif isinstance(v, dict):
          res[k] = {int(sub_k): float(sub_v) for sub_k, sub_v in v.items()}
        else:
          res[k] = v
      return res
    elif isinstance(loaded, list):
      return np.array(loaded, dtype=np.float32)
    return loaded

  else:
    with pipeline._open_file(input_path, 'rb') as f:
      return torch.load(f, map_location='cpu')


def _glob_files(pattern):
  if hasattr(pipeline, '_glob'):
    try:
      matches = pipeline._glob(pattern)
      if matches:
        return list(matches)
    except Exception:
      pass
  return list(glob.glob(pattern))


def _is_dir(path):
  if hasattr(pipeline, '_is_dir'):
    try:
      return bool(pipeline._is_dir(path))
    except Exception:
      pass
  if hasattr(pipeline, 'gfile') and pipeline.gfile is not None:
    try:
      if hasattr(pipeline.gfile, 'IsDirectory'):
        return bool(pipeline.gfile.IsDirectory(path))
      elif hasattr(pipeline.gfile, 'isdir'):
        return bool(pipeline.gfile.isdir(path))
    except Exception:
      pass
  return os.path.isdir(path)


def merge_sharded_scores(
    shard_files_or_dir,
    num_train = None,
):
  """Merges sharded score files (from parallel Map workers) into a consolidated scores dictionary."""
  if isinstance(shard_files_or_dir, str):
    is_directory = _is_dir(shard_files_or_dir)
    if (
        not is_directory
        and not any(
            shard_files_or_dir.endswith(ext)
            for ext in ('.npz', '.pt', '.json', '.npy')
        )
        and '*' not in shard_files_or_dir
    ):
      is_directory = True

    if is_directory:
      file_patterns = [
          os.path.join(shard_files_or_dir, 'scores_shard_*.pt'),
          os.path.join(shard_files_or_dir, 'scores_shard_*.npz'),
          os.path.join(shard_files_or_dir, 'scores_shard_*.json'),
          os.path.join(shard_files_or_dir, 'shard_*_scores.pt'),
          os.path.join(shard_files_or_dir, 'shard_*_scores.npz'),
          os.path.join(shard_files_or_dir, 'shard_*_scores.json'),
          os.path.join(shard_files_or_dir, '*.pt'),
          os.path.join(shard_files_or_dir, '*.npz'),
          os.path.join(shard_files_or_dir, '*.json'),
          os.path.join(shard_files_or_dir, 'shards', '*.pt'),
          os.path.join(shard_files_or_dir, 'shards', '*.npz'),
          os.path.join(shard_files_or_dir, 'shards', '*.json'),
      ]
      shard_files = []
      for pat in file_patterns:
        matches = _glob_files(pat)
        if matches:
          shard_files.extend(matches)
      shard_files = sorted(list(set(shard_files)))
    else:
      shard_files = _glob_files(shard_files_or_dir)
  else:
    shard_files = list(shard_files_or_dir)

  ignored_metadata_files = {
      'corrupted_indices.json',
      'config.json',
      'hparams.json',
      'metadata.json',
      'summary.json',
  }
  shard_files = [
      f
      for f in shard_files
      if os.path.basename(f) not in ignored_metadata_files
  ]

  if not shard_files:
    raise FileNotFoundError(f'No shard files found for {shard_files_or_dir}')

  merged_dict: dict[str, dict[int, float]] = {}

  for s_file in shard_files:
    data = load_scores(s_file)
    if isinstance(data, dict):
      for method_name, val in data.items():
        if method_name not in merged_dict:
          merged_dict[method_name] = {}
        if isinstance(val, dict):
          for idx, sc in val.items():
            merged_dict[method_name][int(idx)] = float(sc)
        elif isinstance(val, (np.ndarray, list)):
          for idx, sc in enumerate(val):
            if not np.isnan(sc):
              merged_dict[method_name][int(idx)] = float(sc)
    elif isinstance(data, (np.ndarray, list)):
      if 'default' not in merged_dict:
        merged_dict['default'] = {}
      for idx, sc in enumerate(data):
        if not np.isnan(sc):
          merged_dict['default'][int(idx)] = float(sc)

  final_scores: dict[str, np.ndarray] = {}
  for method_name, idx_map in merged_dict.items():
    max_idx = max(idx_map.keys()) if idx_map else -1
    total_len = num_train if num_train is not None else (max_idx + 1)
    arr = np.full(total_len, np.nan, dtype=np.float32)
    for idx, sc in idx_map.items():
      if 0 <= idx < total_len:
        arr[idx] = sc
    if np.any(np.isnan(arr)):
      missing_count = int(np.isnan(arr).sum())
      logging.warning(
          'Method %s has %d missing indices out of %d. Filling with 0.0.',
          method_name,
          missing_count,
          total_len,
      )
      arr = np.nan_to_num(arr, nan=0.0)
    final_scores[method_name] = arr

  return final_scores


def compute_leave_one_out_counterfactuals(
    model_fn,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    sample_indices = None,
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    optimizer_type = 'adam',
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    seed = 0,
    retrain_seeds = None,
    max_workers = 1,
    device = 'cpu',
):
  """Computes exact leave-one-out counterfactual loss changes for specified sample_indices with parallelization support."""
  num_train = len(train_dataset)
  target_indices = (
      list(sample_indices)
      if sample_indices is not None
      else list(range(num_train))
  )

  retrain_kwargs = dict(
      model_fn=model_fn,
      train_dataset=train_dataset,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )

  base_metrics = retrain_and_evaluate(indices_to_remove=[], **retrain_kwargs)
  base_val_l, base_val_a = base_metrics['val']
  base_test_l, base_test_a = base_metrics['test']

  r_base_metrics = {}
  if retrain_seeds:
    for s in retrain_seeds:
      rand_kw = dict(retrain_kwargs)
      rand_kw['seed'] = s
      r_base_metrics[s] = retrain_and_evaluate_randomized(
          indices_to_remove=[], **rand_kw
      )

  def _eval_single_sample(idx):
    m_i = retrain_and_evaluate(indices_to_remove=[idx], **retrain_kwargs)
    val_l, val_a = m_i['val']
    test_l, test_a = m_i['test']
    rand_res = {}
    if retrain_seeds:
      for s in retrain_seeds:
        rand_kw = dict(retrain_kwargs)
        rand_kw['seed'] = s
        m_i_s = retrain_and_evaluate_randomized(
            indices_to_remove=[idx], **rand_kw
        )
        rand_res[s] = (m_i_s['val'][0], m_i_s['test'][0])
    return idx, val_l, test_l, val_a, test_a, rand_res

  if max_workers > 1 and len(target_indices) > 1:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      eval_results = list(executor.map(_eval_single_sample, target_indices))
  else:
    eval_results = [_eval_single_sample(idx) for idx in target_indices]

  val_losses: dict[int, float] = {}
  test_losses: dict[int, float] = {}
  val_loss_changes: dict[int, float] = {}
  test_loss_changes: dict[int, float] = {}
  randomized_cf_losses: dict[int, dict[str, dict[int, float]]] = {}
  if retrain_seeds:
    for s in retrain_seeds:
      randomized_cf_losses[s] = {
          'val_losses': {},
          'test_losses': {},
          'val_loss_changes': {},
          'test_loss_changes': {},
      }

  for idx, val_l, test_l, val_a, test_a, rand_res in eval_results:
    val_losses[idx] = val_l
    test_losses[idx] = test_l
    val_loss_changes[idx] = val_l - base_val_l
    test_loss_changes[idx] = test_l - base_test_l
    if retrain_seeds:
      for s in retrain_seeds:
        r_val_l, r_test_l = rand_res[s]
        r_b_val_l = r_base_metrics[s]['val'][0]
        r_b_test_l = r_base_metrics[s]['test'][0]
        randomized_cf_losses[s]['val_losses'][idx] = r_val_l
        randomized_cf_losses[s]['test_losses'][idx] = r_test_l
        randomized_cf_losses[s]['val_loss_changes'][idx] = r_val_l - r_b_val_l
        randomized_cf_losses[s]['test_loss_changes'][idx] = (
            r_test_l - r_b_test_l
        )

  return {
      'baseline': {
          'val_loss': base_val_l,
          'val_acc': base_val_a,
          'test_loss': base_test_l,
          'test_acc': base_test_a,
      },
      'indices': target_indices,
      'val_losses': val_losses,
      'test_losses': test_losses,
      'val_loss_changes': val_loss_changes,
      'test_loss_changes': test_loss_changes,
      'randomized_cf_losses': randomized_cf_losses,
  }


def evaluate_ground_truth_counterfactuals_and_cleansing(
    model_fn,
    train_dataset,
    train_loader,
    val_loader,
    test_loader,
    checkpoints,
    scores_dict = None,
    test_scores_dict = None,
    corrupted_indices = None,
    momentum_lst = None,
    variance_lst = None,
    optimizer_type = 'adam',
    methods = (
        'adam_recursive',
        'adam_recursive_nodv',
        'adam_exact',
        'sgd_all',
        'sgd_last',
        'icml',
    ),
    sample_indices = None,
    num_epochs = 10,
    batch_size = 64,
    lr = 0.05,
    momentum = 0.0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    criterion = nn.functional.cross_entropy,
    k_list = (1, 3, 6, 10, 30, 60, 100),
    seed = 0,
    retrain_seeds = None,
    random_mask = None,
    max_workers = 1,
    device = 'cpu',
    output_dir = None,
    use_reversible_transform = False,
    start_step = 0,
    ignore_first = False,
    quantized_base_model_obj = None,
    quantized_checkpoints = None,
    momentum_buffer = None,
    variance_buffer = None,
    quantization_scale = 1000000,
):
  """Runs leave-one-out ground truth counterfactual models and compares estimation with cleansing performance."""
  num_train = len(train_dataset)
  baseline_model = checkpoints[-1]
  target_indices = (
      list(sample_indices)
      if sample_indices is not None
      else list(range(num_train))
  )

  base_metrics = evaluate_splits(
      baseline_model,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      criterion=criterion,
      device=device,
  )
  base_val_l, base_val_a = base_metrics['val']
  base_test_l, base_test_a = base_metrics['test']

  retrain_kwargs = dict(
      model_fn=model_fn,
      train_dataset=train_dataset,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      num_epochs=num_epochs,
      batch_size=batch_size,
      lr=lr,
      optimizer_type=optimizer_type,
      momentum=momentum,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      criterion=criterion,
      seed=seed,
      device=device,
  )

  # Baseline evaluation for each retrain seed if randomized retraining is enabled
  r_base_metrics = {}
  if retrain_seeds:
    for s in retrain_seeds:
      rand_kw = dict(retrain_kwargs)
      rand_kw['seed'] = s
      r_base_metrics[s] = retrain_and_evaluate_randomized(
          indices_to_remove=[], **rand_kw
      )

  # 1. Train ground truth leave-one-out counterfactual models
  logging.info(
      'Running ground truth leave-one-out counterfactual models for %d'
      ' samples (max_workers=%d)...',
      len(target_indices),
      max_workers,
  )

  def _eval_single_loo_sample(idx):
    m_i = retrain_and_evaluate(indices_to_remove=[idx], **retrain_kwargs)
    val_l, _ = m_i['val']
    test_l, _ = m_i['test']
    rand_res = {}
    if retrain_seeds:
      for s in retrain_seeds:
        rand_kw = dict(retrain_kwargs)
        rand_kw['seed'] = s
        m_i_s = retrain_and_evaluate_randomized(
            indices_to_remove=[idx], **rand_kw
        )
        rand_res[s] = (m_i_s['val'][0], m_i_s['test'][0])
    return idx, val_l, test_l, rand_res

  if max_workers > 1 and len(target_indices) > 1:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      eval_results = list(executor.map(_eval_single_loo_sample, target_indices))
  else:
    eval_results = [_eval_single_loo_sample(i) for i in target_indices]

  val_loss_map = {idx: vl for idx, vl, _, _ in eval_results}
  test_loss_map = {idx: tl for idx, _, tl, _ in eval_results}
  rand_res_map = {idx: rr for idx, _, _, rr in eval_results}

  actual_val_losses = [
      val_loss_map.get(i, base_val_l) for i in range(num_train)
  ]
  actual_test_losses = [
      test_loss_map.get(i, base_test_l) for i in range(num_train)
  ]
  actual_val_loss_changes = [
      val_loss_map.get(i, base_val_l) - base_val_l for i in range(num_train)
  ]
  actual_test_loss_changes = [
      test_loss_map.get(i, base_test_l) - base_test_l for i in range(num_train)
  ]

  randomized_actual_cf_losses = {}
  if retrain_seeds:
    for s in retrain_seeds:
      randomized_actual_cf_losses[s] = {
          'val_losses': [],
          'test_losses': [],
          'val_loss_changes': [],
          'test_loss_changes': [],
      }
      r_b_val_l = r_base_metrics[s]['val'][0]
      r_b_test_l = r_base_metrics[s]['test'][0]
      for i in range(num_train):
        if i in rand_res_map and s in rand_res_map[i]:
          r_val_l, r_test_l = rand_res_map[i][s]
        else:
          r_val_l, r_test_l = r_b_val_l, r_b_test_l
        randomized_actual_cf_losses[s]['val_losses'].append(r_val_l)
        randomized_actual_cf_losses[s]['test_losses'].append(r_test_l)
        randomized_actual_cf_losses[s]['val_loss_changes'].append(
            r_val_l - r_b_val_l
        )
        randomized_actual_cf_losses[s]['test_loss_changes'].append(
            r_test_l - r_b_test_l
        )

  actual_val_loss_changes_arr = np.array(
      actual_val_loss_changes, dtype=np.float32
  )
  actual_test_loss_changes_arr = np.array(
      actual_test_loss_changes, dtype=np.float32
  )

  # 2. Get method scores (using precomputed scores_dict if passed)
  if scores_dict is None:
    scores_dict = {}
  else:
    scores_dict = dict(scores_dict)

  num_batches = len(utils.prepare_batches(train_loader))
  effective_start_step = num_batches if ignore_first else start_step

  method_estimates = {}
  base_model_artifacts = {
      'train_loader': train_loader,
      'checkpoints': checkpoints,
      'momentum_lst': momentum_lst,
      'variance_lst': variance_lst,
      'loss_fn': criterion,
      'hyperparams': {
          'lr': lr,
          'beta_1': beta_1,
          'beta_2': beta_2,
          'eps': eps,
          'seed': seed,
      },
  }
  for m_name in methods:
    if (
        m_name.startswith('adam')
        or m_name == 'tracin_adam'
        or (m_name == 'tracin' and optimizer_type == 'adam')
    ) and (momentum_lst is None or variance_lst is None):
      continue

    if m_name in scores_dict:
      val_scores = scores_dict[m_name]
    else:
      val_scores = compute_scores_for_method(
          method=m_name,
          base_model_obj=base_model_artifacts,
          val_loader=val_loader,
          optimizer_type=optimizer_type,
          random_mask=random_mask,
          candidate_indices=sample_indices,
          max_workers=max_workers,
          device=device,
          use_reversible_transform=use_reversible_transform,
          start_step=effective_start_step,
          quantized_base_model_obj=quantized_base_model_obj,
          quantized_checkpoints=quantized_checkpoints,
          momentum_buffer=momentum_buffer,
          variance_buffer=variance_buffer,
          quantization_scale=quantization_scale,
      )
      scores_dict[m_name] = val_scores

    if test_scores_dict is not None and m_name in test_scores_dict:
      test_scores = test_scores_dict[m_name]
    elif m_name in ('ae', 'iso'):
      test_scores = val_scores.copy()
    elif m_name == 'random':
      rng = np.random.default_rng(seed)
      test_scores = rng.random(num_train).astype(np.float32)
    else:
      test_scores = compute_scores_for_method(
          method=m_name,
          base_model_obj=base_model_artifacts,
          val_loader=test_loader,
          optimizer_type=optimizer_type,
          random_mask=random_mask,
          candidate_indices=sample_indices,
          max_workers=max_workers,
          device=device,
          use_reversible_transform=use_reversible_transform,
          start_step=effective_start_step,
          quantized_base_model_obj=quantized_base_model_obj,
          quantized_checkpoints=quantized_checkpoints,
          momentum_buffer=momentum_buffer,
          variance_buffer=variance_buffer,
          quantization_scale=quantization_scale,
      )

    method_estimates[m_name] = {
        'val_scores': val_scores,
        'test_scores': test_scores,
    }

  # 3. Summary comparison using Ground Truth Actual Counterfactuals
  corrupted_set = (
      set(corrupted_indices) if corrupted_indices is not None else set()
  )

  actual_negative_indices = [
      i for i in range(num_train) if actual_val_loss_changes_arr[i] < 0.0
  ]
  actual_negative_indices.sort(key=lambda i: actual_val_loss_changes_arr[i])

  actual_cf_metrics = retrain_and_evaluate(
      indices_to_remove=actual_negative_indices, **retrain_kwargs
  )
  c_val_l, _ = actual_cf_metrics['val']
  c_te_l, c_te_a = actual_cf_metrics['test']

  removed_set = set(actual_negative_indices)
  tp = len(removed_set & corrupted_set)
  prec = tp / max(1, len(removed_set)) if removed_set else 0.0
  rec = tp / max(1, len(corrupted_set)) if corrupted_set else 0.0
  f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

  act_part = compute_partitioned_corrupted_metrics(
      actual_val_loss_changes_arr, corrupted_set, is_outlier_score=False
  )

  summary_rows = [{
      'method': 'actual_cf',
      'val_loss_mae': 0.0,
      'test_loss_mae': 0.0,
      'val_loss_corr': 1.0,
      'test_loss_corr': 1.0,
      'cleansed_val_loss': c_val_l,
      'cleansed_test_loss': c_te_l,
      'cleansed_test_acc': c_te_a,
      'val_loss_change': c_val_l - base_val_l,
      'test_loss_change': c_te_l - base_test_l,
      'test_acc_change': c_te_a - base_test_a,
      'detection_precision': prec,
      'detection_recall': rec,
      'detection_f1': f1,
      'mean_val_loss_change_clean': act_part['mean_score_clean'],
      'mean_val_loss_change_corrupted': act_part['mean_score_corrupted'],
      'roc_auc': act_part['roc_auc'],
  }]

  oracle_corrupted_results = None
  if corrupted_set:
    logging.info(
        'Retraining model with exactly the %d corrupted indices removed...',
        len(corrupted_set),
    )
    corr_metrics = retrain_and_evaluate(
        indices_to_remove=sorted(list(corrupted_set)), **retrain_kwargs
    )
    corr_tr_l, corr_tr_a = corr_metrics['train']
    corr_val_l, corr_val_a = corr_metrics['val']
    corr_te_l, corr_te_a = corr_metrics['test']

    oracle_corrupted_results = {
        'train_loss': corr_tr_l,
        'train_acc': corr_tr_a,
        'val_loss': corr_val_l,
        'val_acc': corr_val_a,
        'test_loss': corr_te_l,
        'test_acc': corr_te_a,
        'val_loss_change': corr_val_l - base_val_l,
        'test_loss_change': corr_te_l - base_test_l,
        'test_acc_change': corr_te_a - base_test_a,
    }

    summary_rows.append({
        'method': 'remove_corrupted',
        'val_loss_mae': 0.0,
        'test_loss_mae': 0.0,
        'val_loss_corr': 1.0,
        'test_loss_corr': 1.0,
        'cleansed_val_loss': corr_val_l,
        'cleansed_test_loss': corr_te_l,
        'cleansed_test_acc': corr_te_a,
        'val_loss_change': corr_val_l - base_val_l,
        'test_loss_change': corr_te_l - base_test_l,
        'test_acc_change': corr_te_a - base_test_a,
        'detection_precision': 1.0,
        'detection_recall': 1.0,
        'detection_f1': 1.0,
        'mean_val_loss_change_clean': act_part['mean_score_clean'],
        'mean_val_loss_change_corrupted': act_part['mean_score_corrupted'],
        'roc_auc': 1.0,
    })

  for m_name, ests in method_estimates.items():
    val_sc = ests['val_scores']
    test_sc = ests['test_scores']

    v_mae = float(np.mean(np.abs(val_sc - actual_val_loss_changes_arr)))
    t_mae = float(np.mean(np.abs(test_sc - actual_test_loss_changes_arr)))

    v_std = np.std(val_sc) * np.std(actual_val_loss_changes_arr)
    t_std = np.std(test_sc) * np.std(actual_test_loss_changes_arr)
    v_corr = (
        float(np.corrcoef(val_sc, actual_val_loss_changes_arr)[0, 1])
        if v_std > 1e-12
        else 0.0
    )
    t_corr = (
        float(np.corrcoef(test_sc, actual_test_loss_changes_arr)[0, 1])
        if t_std > 1e-12
        else 0.0
    )

    m_selected_indices = select_indices_for_removal(val_sc, m_name)
    m_removed_set = set(m_selected_indices)
    m_tp = len(m_removed_set & corrupted_set)
    m_prec = m_tp / max(1, len(m_removed_set)) if m_removed_set else 0.0
    m_rec = m_tp / max(1, len(corrupted_set)) if corrupted_set else 0.0
    m_f1 = (
        (2 * m_prec * m_rec / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0
    )
    m_part = compute_partitioned_corrupted_metrics(
        val_sc, corrupted_set, is_outlier_score=(m_name in ('ae', 'iso'))
    )

    summary_rows.append({
        'method': m_name,
        'val_loss_mae': v_mae,
        'test_loss_mae': t_mae,
        'val_loss_corr': v_corr,
        'test_loss_corr': t_corr,
        'detection_precision': m_prec,
        'detection_recall': m_rec,
        'detection_f1': m_f1,
        'mean_val_loss_change_clean': m_part['mean_score_clean'],
        'mean_val_loss_change_corrupted': m_part['mean_score_corrupted'],
        'roc_auc': m_part['roc_auc'],
    })

  # 4. Top-k Cleansing for actual_cf and remove_corrupted
  topk_actual_results = {}
  if oracle_corrupted_results is not None:
    topk_actual_results[('remove_corrupted', len(corrupted_set))] = (
        oracle_corrupted_results['train_loss'],
        oracle_corrupted_results['val_loss'],
        oracle_corrupted_results['test_loss'],
        oracle_corrupted_results['train_acc'],
        oracle_corrupted_results['val_acc'],
        oracle_corrupted_results['test_acc'],
    )
    if retrain_seeds:
      for s in retrain_seeds:
        rand_kw = dict(retrain_kwargs)
        rand_kw['seed'] = s
        r_corr_metrics = retrain_and_evaluate_randomized(
            indices_to_remove=sorted(list(corrupted_set)), **rand_kw
        )
        r_corr_tr_l, r_corr_tr_a = r_corr_metrics['train']
        r_corr_val_l, r_corr_val_a = r_corr_metrics['val']
        r_corr_te_l, r_corr_te_a = r_corr_metrics['test']
        topk_actual_results[
            (f'remove_corrupted_seed_{s}', len(corrupted_set))
        ] = (
            r_corr_tr_l,
            r_corr_val_l,
            r_corr_te_l,
            r_corr_tr_a,
            r_corr_val_a,
            r_corr_te_a,
        )

  numeric_k_list = []
  seen_k = set()
  for k in k_list:
    if k == 'auto' or k is None:
      continue
    k_int = int(k)
    if k_int not in seen_k:
      seen_k.add(k_int)
      numeric_k_list.append(k_int)

  if corrupted_set and len(corrupted_set) not in seen_k:
    numeric_k_list.append(len(corrupted_set))
    numeric_k_list.sort()

  ranked_actual = np.argsort(actual_val_loss_changes_arr)
  for k in numeric_k_list:
    if k > num_train:
      continue
    top_k_indices = ranked_actual[:k].tolist()
    m_k_metrics = retrain_and_evaluate(
        indices_to_remove=top_k_indices, **retrain_kwargs
    )
    tr_l, tr_a = m_k_metrics['train']
    val_l, val_a = m_k_metrics['val']
    te_l, te_a = m_k_metrics['test']
    topk_actual_results[('actual_cf', k)] = (
        tr_l,
        val_l,
        te_l,
        tr_a,
        val_a,
        te_a,
    )
    if retrain_seeds:
      for s in retrain_seeds:
        rand_kw = dict(retrain_kwargs)
        rand_kw['seed'] = s
        r_m_k_metrics = retrain_and_evaluate_randomized(
            indices_to_remove=top_k_indices, **rand_kw
        )
        r_tr_l, r_tr_a = r_m_k_metrics['train']
        r_val_l, r_val_a = r_m_k_metrics['val']
        r_te_l, r_te_a = r_m_k_metrics['test']
        topk_actual_results[(f'actual_cf_seed_{s}', k)] = (
            r_tr_l,
            r_val_l,
            r_te_l,
            r_tr_a,
            r_val_a,
            r_te_a,
        )

  # 5. Save CSVs if output_dir is given
  if output_dir:
    pipeline._make_dirs(output_dir)

    detail_csv = os.path.join(
        output_dir, 'counterfactual_estimation_comparison.csv'
    )
    fieldnames = [
        'sample_index',
        'is_corrupted',
        'actual_val_loss',
        'actual_test_loss',
        'actual_val_loss_change',
        'actual_test_loss_change',
    ]
    for m_name in method_estimates:
      fieldnames.extend([
          f'est_val_loss_{m_name}',
          f'est_test_loss_{m_name}',
          f'est_val_loss_change_{m_name}',
          f'est_test_loss_change_{m_name}',
          f'val_loss_abs_err_{m_name}',
          f'test_loss_abs_err_{m_name}',
      ])

    with pipeline._open_file(detail_csv, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(fieldnames)
      for i in range(num_train):
        row = [
            i,
            int(i in corrupted_set),
            actual_val_losses[i],
            actual_test_losses[i],
            actual_val_loss_changes[i],
            actual_test_loss_changes[i],
        ]
        for m_name, ests in method_estimates.items():
          val_sc_i = float(ests['val_scores'][i])
          test_sc_i = float(ests['test_scores'][i])
          est_val_l = base_val_l + val_sc_i
          est_test_l = base_test_l + test_sc_i
          val_err = abs(val_sc_i - actual_val_loss_changes[i])
          test_err = abs(test_sc_i - actual_test_loss_changes[i])
          row.extend([
              est_val_l,
              est_test_l,
              val_sc_i,
              test_sc_i,
              val_err,
              test_err,
          ])
        writer.writerow(row)

    summary_csv = os.path.join(
        output_dir, 'counterfactual_estimation_summary.csv'
    )
    with pipeline._open_file(summary_csv, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
      writer.writeheader()
      writer.writerows(summary_rows)

  ret = {
      'baseline': {
          'val_loss': base_val_l,
          'val_acc': base_val_a,
          'test_loss': base_test_l,
          'test_acc': base_test_a,
      },
      'actual_cf_losses': {
          'val_losses': actual_val_losses,
          'test_losses': actual_test_losses,
          'val_loss_changes': actual_val_loss_changes,
          'test_loss_changes': actual_test_loss_changes,
      },
      'summary_rows': summary_rows,
      'topk_actual_results': topk_actual_results,
  }
  if retrain_seeds:
    ret['randomized_actual_cf_losses'] = randomized_actual_cf_losses
  if oracle_corrupted_results is not None:
    ret['oracle_corrupted'] = oracle_corrupted_results
  return ret
