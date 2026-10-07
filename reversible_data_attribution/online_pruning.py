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

"""Online data selection and pruning framework for deep neural networks.

This module implements:
1. Windowed counterfactual data attribution over local training intervals [m-t,
m].
2. The N = A + B + C online pruning curriculum (Warmup -> Progressive Pruning ->
Post-Pruning).
3. Metric evaluation for data selection quality (ROC-AUC / F1 on corrupted
labels)
   and downstream generalization (online model vs retrained model).
"""

from collections.abc import Callable, Sequence
import dataclasses
import random
from typing import Any

import numpy as np
import sklearn.metrics
import torch
from torch import nn

from reversible_data_attribution import base_model
from reversible_data_attribution import infl_adam
from reversible_data_attribution import utils


@dataclasses.dataclass
class OnlinePruningConfig:
  """Configuration for online training data pruning."""

  warmup_epochs: int = 1
  pruning_epochs: int = 2
  post_pruning_epochs: int = 1
  prune_budget: float = 0.10
  window_size: int = 10
  score_type: str = (
      'counterfactual'  # 'counterfactual', 'tracin', 'loss', 'random'
  )
  dv_option: str = 'first_order'
  theta_option: str = 'first_order'
  batch_scale: bool = True
  save_all_ckpts: bool = True
  checkpoint_freq: int | None = 1
  device: str = 'cpu'


@dataclasses.dataclass
class OnlinePruningResult:
  """Results returned from an online pruning run."""

  online_model: nn.Module
  retrained_model: nn.Module | None
  base_model: base_model.AttributionBaseModel
  pruned_indices: set[int]
  metrics: dict[str, float]
  history: list[dict[str, Any]]


compute_validation_gradient = utils.compute_validation_gradient


def compute_windowed_scores(
    base_model_obj,
    window_steps,
    window_batches,
    val_loader,
    score_type = 'counterfactual',
    dv_option = 'first_order',
    theta_option = 'first_order',
    device = 'cpu',
):
  """Computes data attribution scores for all samples seen in a training window.

  Args:
    base_model_obj: AttributionBaseModel containing recorded checkpoints.
    window_steps: List of global step indices executed in this window.
    window_batches: List of (x, y, indices) batches executed in this window.
    val_loader: Validation DataLoader for target objective evaluation.
    score_type: 'counterfactual', 'counterfactual_backward' (or 'recursive'),
      'tracin', 'loss', or 'random'.
    dv_option: 'first_order' or 'exact' for variance differences in Adam.
    theta_option: 'first_order' or 'second_order' for parameter updates.
    device: Computation device.

  Returns:
    Dictionary mapping dataset sample index -> score (lower score = more
    harmful).
  """
  device_obj = torch.device(device)
  ckpts = base_model_obj.checkpoints()
  loss_fn = base_model_obj.loss_fn()
  lr = base_model_obj.lr()
  beta_1 = base_model_obj.beta_1()
  beta_2 = base_model_obj.beta_2()
  eps = base_model_obj.eps()

  # Identify all unique sample indices present in this window
  sample_set = set()
  for _, _, indices in window_batches:
    if indices is not None:
      if isinstance(indices, torch.Tensor):
        sample_set.update(indices.tolist())
      else:
        sample_set.update(list(indices))

  if not sample_set:
    return {}

  if score_type == 'random':
    return {idx: random.random() for idx in sample_set}

  latest_step = (
      window_steps[-1] if window_steps else base_model_obj.total_steps()
  )
  latest_model = (
      ckpts[latest_step].model.to(device_obj)
      if latest_step in ckpts
      else base_model_obj.model().to(device_obj)
  )

  if score_type == 'loss':
    # Loss-based scoring: higher loss = worse score = more likely to prune
    scores = {}
    for x_b, y_b, indices in window_batches:
      if indices is None:
        continue
      x_b = x_b.to(device_obj)
      y_b = y_b.to(device_obj)
      idx_list = (
          indices.tolist()
          if isinstance(indices, torch.Tensor)
          else list(indices)
      )
      with torch.no_grad():
        preds = latest_model(x_b)
        sample_losses = [
            loss_fn(preds[i : i + 1], y_b[i : i + 1]).item()
            for i in range(len(idx_list))
        ]
      for sample_id, loss_val in zip(idx_list, sample_losses):
        scores[sample_id] = -float(loss_val)
    return scores

  # Validation loss gradient for Taylor / counterfactual projection
  val_grad = utils.compute_validation_gradient(
      latest_model,
      criterion=loss_fn,
      val_loader=val_loader,
      device=device_obj,
      flatten=True,
  )

  if score_type in ['counterfactual_backward', 'recursive']:
    start_g = window_steps[0] - 1 if window_steps and window_steps[0] > 0 else 0
    end_g = window_steps[-1] if window_steps else base_model_obj.total_steps()

    # Populate checkpoints, momentum, and variance for steps start_g ... end_g.
    # Checkpoints at start_g, ..., end_g - 1 are the prior models (model_prev),
    # while checkpoint at end_g is the terminal model (ckpt_T).
    step_ckpts = {}
    step_mom = {}
    step_var = {}
    for g in range(start_g, end_g + 1):
      if g in ckpts:
        step_ckpts[g] = ckpts[g].model
        step_mom[g] = ckpts[g].momentum
        step_var[g] = ckpts[g].variance
      elif g == 0:
        step_ckpts[0] = base_model_obj.initial_model()
        step_mom[0] = [torch.zeros_like(p) for p in step_ckpts[0].parameters()]
        step_var[0] = [torch.zeros_like(p) for p in step_ckpts[0].parameters()]
      else:
        step_ckpts[g] = latest_model
        step_mom[g] = [torch.zeros_like(p) for p in latest_model.parameters()]
        step_var[g] = [torch.zeros_like(p) for p in latest_model.parameters()]

    # Construct aligned_batches of length end_g such that for any global step
    # t in [start_g, end_g - 1], t % len(aligned_batches) == t maps to the
    # exact batch in window_batches executed at transition t -> t + 1.
    aligned_batches = [window_batches[0]] * start_g + list(window_batches)

    context_dict = {
        'checkpoints': step_ckpts,
        'momentum_lst': step_mom,
        'variance_lst': step_var,
        'batches': aligned_batches,
        'hyperparams': {
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'eps': eps,
        },
        'loss_fn': loss_fn,
    }
    scores_dict = infl_adam.recursive_update(
        base_model=context_dict,
        inds=list(sample_set),
        query_vec=val_grad,
        start_step=start_g,
        num_steps=end_g,
        device=device_obj,
    )
    return {
        idx: float(val.item()) if isinstance(val, torch.Tensor) else float(val)
        for idx, val in scores_dict.items()
    }

  if score_type == 'tracin':
    # 1-step gradient dot product
    scores = {}
    for x_b, y_b, indices in window_batches:
      if indices is None:
        continue
      x_b = x_b.to(device_obj)
      y_b = y_b.to(device_obj)
      idx_list = (
          indices.tolist()
          if isinstance(indices, torch.Tensor)
          else list(indices)
      )
      for i, sample_id in enumerate(idx_list):
        sample_x = x_b[i : i + 1]
        sample_y = y_b[i : i + 1]
        y_hat = latest_model(sample_x)
        loss = loss_fn(y_hat, sample_y)
        grads = torch.autograd.grad(
            loss, list(latest_model.parameters()), create_graph=False
        )
        flat_g = torch.cat([g.flatten() for g in grads]).detach()
        dot_score = torch.dot(val_grad, flat_g).item()
        scores[sample_id] = float(dot_score)
    return scores

  # score_type == 'counterfactual' (TSLOO forward windowed parameter tracking)
  scores = {}
  for sample_id in sample_set:
    # Initialize zero perturbations at window start
    theta_diff = None
    mom_diff = None
    var_diff = None

    for s, step_g in enumerate(window_steps):
      x_b, y_b, indices = window_batches[s]
      x_b = x_b.to(device_obj)
      y_b = y_b.to(device_obj)

      # Check if this sample was in the batch
      ind_in_batch = None
      if indices is not None:
        idx_list = (
            indices.tolist()
            if isinstance(indices, torch.Tensor)
            else list(indices)
        )
        if sample_id in idx_list:
          ind_in_batch = idx_list.index(sample_id)

      # Base checkpoint for transition step_g - 1 -> step_g is step_g - 1
      prev_step = step_g - 1
      if prev_step in ckpts:
        step_model = ckpts[prev_step].model.to(device_obj)
      elif prev_step == 0:
        step_model = base_model_obj.initial_model().to(device_obj)
      elif step_g in ckpts:
        step_model = ckpts[step_g].model.to(device_obj)
      else:
        step_model = latest_model

      if step_g in ckpts:
        step_mom = [m.to(device_obj) for m in ckpts[step_g].momentum]
        step_var = [v.to(device_obj) for v in ckpts[step_g].variance]
      elif prev_step in ckpts:
        step_mom = [m.to(device_obj) for m in ckpts[prev_step].momentum]
        step_var = [v.to(device_obj) for v in ckpts[prev_step].variance]
      else:
        step_mom = [torch.zeros_like(p) for p in step_model.parameters()]
        step_var = [torch.zeros_like(p) for p in step_model.parameters()]

      if theta_diff is None:
        num_params = sum(p.numel() for p in step_model.parameters())
        theta_diff = torch.zeros((1, num_params), device=device_obj)
        mom_diff = torch.zeros((1, num_params), device=device_obj)
        var_diff = torch.zeros((1, num_params), device=device_obj)

      theta_diff, mom_diff, var_diff = infl_adam.one_step_update(
          x_batch=x_b,
          y_batch=y_b,
          model=step_model,
          loss_func=loss_fn,
          theta_diff=theta_diff,
          momentum=step_mom,
          momentum_diff=mom_diff,
          variance=step_var,
          variance_diff=var_diff,
          lr=lr,
          beta_1=beta_1,
          beta_2=beta_2,
          eps=eps,
          ind=ind_in_batch,
          step=step_g,
          dv_option=dv_option,
          theta_option=theta_option,
      )

    # Compute counterfactual loss change via Taylor projection:
    # S(j) ≈ ∇L(θ_m, Z_val)^T (θ_m^{[-j]} - θ_m)
    cf_score = torch.dot(val_grad, theta_diff.flatten()).item()
    scores[sample_id] = float(cf_score)

  return scores


def evaluate_model_performance(
    model,
    data_loader,
    loss_fn = nn.functional.cross_entropy,
    device = 'cpu',
):
  """Computes average loss and accuracy of model on data_loader."""
  device_obj = torch.device(device)
  model = model.to(device_obj)
  model.eval()

  batches = utils.prepare_batches(data_loader)
  if not batches:
    return 0.0, 0.0

  total_loss = 0.0
  total_correct = 0
  total_samples = 0

  with torch.no_grad():
    for x_b, y_b, _ in batches:
      x_b = x_b.to(device_obj)
      y_b = y_b.to(device_obj)
      batch_size = x_b.shape[0]
      total_samples += batch_size

      preds = model(x_b)
      loss = loss_fn(preds, y_b).item()
      total_loss += loss * batch_size

      if preds.ndim > 1 and preds.shape[1] > 1:
        correct = (preds.argmax(dim=1) == y_b).sum().item()
      else:
        correct = ((preds > 0.0).long() == y_b).sum().item()
      total_correct += correct

  avg_loss = total_loss / max(1, total_samples)
  accuracy = total_correct / max(1, total_samples)
  return avg_loss, accuracy


def run_online_pruning(
    model,
    train_loader,
    val_loader,
    test_loader = None,
    config = None,
    loss_fn = nn.functional.cross_entropy,
    lr = 0.001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    eps = 1e-8,
    corrupted_indices = None,
    seed = 42,
):
  """Runs the complete online data pruning curriculum.

  Curriculum structure:
    Phase A (Warmup): No pruning, establishes baseline trajectory.
    Phase B (Pruning): Periodically computes windowed scores and 0-masks worst
    data.
    Phase C (Post-Pruning): Trains on the pruned subset to convergence.

  Args:
    model: PyTorch neural network module.
    train_loader: Training DataLoader or dataset with sample indices.
    val_loader: Validation DataLoader for target objective.
    test_loader: Optional test DataLoader for final test generalization.
    config: OnlinePruningConfig settings.
    loss_fn: Loss function.
    lr: Learning rate for Adam.
    beta_1: Adam beta_1.
    beta_2: Adam beta_2.
    eps: Adam epsilon.
    corrupted_indices: Optional known noisy/flipped sample indices for ROC-AUC
      evaluation.
    seed: Random seed.

  Returns:
    OnlinePruningResult containing the online model, retrained model, pruned
    indices,
    and metrics.
  """
  cfg = config if config is not None else OnlinePruningConfig()
  device_obj = torch.device(cfg.device)

  # Initialize AttributionBaseModel container
  base = base_model.AttributionBaseModel(
      model=model,
      loss_fn=loss_fn,
      lr=lr,
      beta_1=beta_1,
      beta_2=beta_2,
      eps=eps,
      seed=seed,
  )

  all_batches = utils.prepare_batches(train_loader)
  all_sample_ids = set()
  for _, _, indices in all_batches:
    if indices is not None:
      if isinstance(indices, torch.Tensor):
        all_sample_ids.update(indices.tolist())
      else:
        all_sample_ids.update(list(indices))

  total_samples = len(all_sample_ids)
  total_to_prune = int(total_samples * cfg.prune_budget)
  prune_per_epoch = (
      max(1, total_to_prune // cfg.pruning_epochs)
      if cfg.pruning_epochs > 0
      else 0
  )

  pruned_indices: set[int] = set()
  history: list[dict[str, Any]] = []

  # --- Phase A: Warmup ---
  for ep in range(cfg.warmup_epochs):
    base.train_model_continue(
        batches=all_batches,
        indices_to_remove=pruned_indices,
        save_all_ckpts=cfg.save_all_ckpts,
        checkpoint_freq=cfg.checkpoint_freq,
        device=cfg.device,
        batch_scale=cfg.batch_scale,
    )
    val_loss, val_acc = evaluate_model_performance(
        base.model(), val_loader, loss_fn=loss_fn, device=device_obj
    )
    history.append({
        'phase': 'A_warmup',
        'epoch': ep + 1,
        'active_samples': total_samples - len(pruned_indices),
        'val_loss': val_loss,
        'val_acc': val_acc,
    })

  # --- Phase B: Progressive Pruning ---
  for ep in range(cfg.pruning_epochs):
    epoch_scores: dict[int, float] = {}

    # Process epoch in chunks of window_size steps
    batch_idx = 0
    while batch_idx < len(all_batches):
      window_batch_slice = all_batches[batch_idx : batch_idx + cfg.window_size]
      batch_idx += cfg.window_size

      _, win_steps, win_batches = base.train_model_continue(
          batches=window_batch_slice,
          indices_to_remove=pruned_indices,
          save_all_ckpts=cfg.save_all_ckpts,
          checkpoint_freq=cfg.checkpoint_freq,
          device=cfg.device,
          batch_scale=cfg.batch_scale,
      )

      if win_steps and win_batches:
        win_scores = compute_windowed_scores(
            base_model_obj=base,
            window_steps=win_steps,
            window_batches=win_batches,
            val_loader=val_loader,
            score_type=cfg.score_type,
            dv_option=cfg.dv_option,
            theta_option=cfg.theta_option,
            device=cfg.device,
        )
        epoch_scores.update(win_scores)

    # Sort remaining active candidates and prune the lowest scoring samples
    active_candidates = [
        (idx, score)
        for idx, score in epoch_scores.items()
        if idx not in pruned_indices
    ]
    active_candidates.sort(key=lambda x: x[1])

    # Prune worst k samples
    k_prune = min(prune_per_epoch, len(active_candidates))
    newly_pruned = {idx for idx, _ in active_candidates[:k_prune]}
    pruned_indices.update(newly_pruned)

    val_loss, val_acc = evaluate_model_performance(
        base.model(), val_loader, loss_fn=loss_fn, device=device_obj
    )
    history.append({
        'phase': 'B_pruning',
        'epoch': cfg.warmup_epochs + ep + 1,
        'active_samples': total_samples - len(pruned_indices),
        'pruned_in_epoch': len(newly_pruned),
        'val_loss': val_loss,
        'val_acc': val_acc,
    })

  # --- Phase C: Post-Pruning Stabilization ---
  for ep in range(cfg.post_pruning_epochs):
    base.train_model_continue(
        batches=all_batches,
        indices_to_remove=pruned_indices,
        save_all_ckpts=cfg.save_all_ckpts,
        checkpoint_freq=cfg.checkpoint_freq,
        device=cfg.device,
        batch_scale=cfg.batch_scale,
    )
    val_loss, val_acc = evaluate_model_performance(
        base.model(), val_loader, loss_fn=loss_fn, device=device_obj
    )
    history.append({
        'phase': 'C_post_pruning',
        'epoch': cfg.warmup_epochs + cfg.pruning_epochs + ep + 1,
        'active_samples': total_samples - len(pruned_indices),
        'val_loss': val_loss,
        'val_acc': val_acc,
    })

  online_model = base.model()

  # --- Retrained Model (Trained from scratch on pruned subset) ---
  total_epochs = (
      cfg.warmup_epochs + cfg.pruning_epochs + cfg.post_pruning_epochs
  )
  retrained_model = base.retrain_remove_indices(
      indices_to_remove=pruned_indices,
      train_loader=train_loader,
      num_epochs=total_epochs,
      device=cfg.device,
      batch_scale=cfg.batch_scale,
  )

  # Compute Final Metrics
  metrics: dict[str, float] = {}

  # 1. Data Selection Quality (ROC-AUC / F1 vs corrupted indices)
  if corrupted_indices is not None and all_sample_ids:
    corrupted_set = set(corrupted_indices)
    y_true = np.array(
        [1 if idx in corrupted_set else 0 for idx in sorted(all_sample_ids)]
    )
    y_pred = np.array(
        [1 if idx in pruned_indices else 0 for idx in sorted(all_sample_ids)]
    )

    precision = float(
        sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)
    )
    recall = float(
        sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)
    )
    f1 = float(sklearn.metrics.f1_score(y_true, y_pred, zero_division=0))
    metrics['data_precision'] = precision
    metrics['data_recall'] = recall
    metrics['data_f1'] = f1

  # 2. Generalization Performance
  val_loss_online, val_acc_online = evaluate_model_performance(
      online_model, val_loader, loss_fn=loss_fn, device=device_obj
  )
  val_loss_retrain, val_acc_retrain = evaluate_model_performance(
      retrained_model, val_loader, loss_fn=loss_fn, device=device_obj
  )

  metrics['online_val_loss'] = val_loss_online
  metrics['online_val_acc'] = val_acc_online
  metrics['retrain_val_loss'] = val_loss_retrain
  metrics['retrain_val_acc'] = val_acc_retrain

  if test_loader is not None:
    test_loss_online, test_acc_online = evaluate_model_performance(
        online_model, test_loader, loss_fn=loss_fn, device=device_obj
    )
    test_loss_retrain, test_acc_retrain = evaluate_model_performance(
        retrained_model, test_loader, loss_fn=loss_fn, device=device_obj
    )
    metrics['online_test_loss'] = test_loss_online
    metrics['online_test_acc'] = test_acc_online
    metrics['retrain_test_loss'] = test_loss_retrain
    metrics['retrain_test_acc'] = test_acc_retrain

  return OnlinePruningResult(
      online_model=online_model,
      retrained_model=retrained_model,
      base_model=base,
      pruned_indices=pruned_indices,
      metrics=metrics,
      history=history,
  )
