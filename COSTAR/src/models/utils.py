# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utilities."""

import copy
from typing import List
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import nn
import torch.nn.functional as F

deepcopy = copy.deepcopy
Function = torch.autograd.Function


def grad_reverse(x, scale=1.0):
  """Gradient reversal."""

  class ReverseGrad(Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx, x):
      return x

    @staticmethod
    def backward(ctx, grad_output):
      return scale * grad_output.neg()

  return ReverseGrad.apply(x)


class FilteringMlFlowLogger(MLFlowLogger):
  """Filtering MlFlow logger."""

  def __init__(self, filter_submodels = None, **kwargs):
    super().__init__(**kwargs)
    self.filter_submodels = filter_submodels

  @rank_zero_only
  def log_hyperparams(self, params):
    params = deepcopy(params)
    for filter_submodel in self.filter_submodels:
      if filter_submodel in params.model:
        params.model.pop(filter_submodel)
    super().log_hyperparams(params)


class FilteringWandbLogger(WandbLogger):
  """Filtering wandb logger."""

  def __init__(self, filter_submodels = None, **kwargs):
    super().__init__(**kwargs)
    self.filter_submodels = filter_submodels

  @rank_zero_only
  def log_hyperparams(self, params):
    params = deepcopy(params)
    for filter_submodel in self.filter_submodels:
      if filter_submodel in params.model:
        params.model.pop(filter_submodel)
    super().log_hyperparams(params)


def bce(treatment_pred, current_treatments, mode, weights=None):
  """BCE loss."""
  if mode == 'multiclass':
    return F.cross_entropy(
        treatment_pred.permute(0, 2, 1),
        current_treatments.permute(0, 2, 1),
        reduce=False,
        weight=weights,
    )
  elif mode == 'multilabel':
    return F.binary_cross_entropy_with_logits(
        treatment_pred, current_treatments, reduce=False, weight=weights
    ).mean(dim=-1)
  else:
    raise NotImplementedError()


class BRTreatmentOutcomeHead(nn.Module):
  """Used by CRN, EDCT, MultiInputTransformer."""

  def __init__(
      self,
      seq_hidden_units,
      br_size,
      fc_hidden_units,
      dim_treatments,
      dim_outcome,
      alpha=0.0,
      update_alpha=True,
      balancing='grad_reverse',
      alpha_prev_treat=0.0,
      update_alpha_prev_treat=False,
      alpha_age=0.0,
      update_alpha_age=False,
  ):
    super().__init__()

    self.seq_hidden_units = seq_hidden_units
    self.br_size = br_size
    self.fc_hidden_units = fc_hidden_units
    self.dim_treatments = dim_treatments
    self.dim_outcome = dim_outcome
    self.alpha = alpha if not update_alpha else 0.0
    self.alpha_max = alpha
    self.balancing = balancing

    self.alpha_prev_treat = (
        alpha_prev_treat if not update_alpha_prev_treat else 0.0
    )
    self.alpha_prev_treat_max = alpha_prev_treat
    self.alpha_age = alpha_age if not update_alpha_age else 0.0
    self.alpha_age_max = alpha_age

    self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
    self.elu1 = nn.ELU()

    self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
    self.elu2 = nn.ELU()
    self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

    self.linear4 = nn.Linear(
        self.br_size + self.dim_treatments, self.fc_hidden_units
    )
    self.elu3 = nn.ELU()
    self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

    self.treatment_head_params = ['linear2', 'linear3']

    if self.alpha_prev_treat_max > 0:
      self.linear6 = nn.Linear(self.br_size, self.fc_hidden_units)
      self.elu6 = nn.ELU()
      self.linear7 = nn.Linear(self.fc_hidden_units, self.dim_treatments)
      self.treatment_head_params.extend(['linear6', 'linear7'])

    if self.alpha_age_max > 0:
      self.linear8 = nn.Linear(self.br_size, self.fc_hidden_units)
      self.elu8 = nn.ELU()
      self.linear9 = nn.Linear(self.fc_hidden_units, 1)
      self.treatment_head_params.extend(['linear8', 'linear9'])

  def build_treatment(self, br, detached=False):
    if detached:
      br = br.detach()

    if self.balancing == 'grad_reverse':
      br = grad_reverse(br, self.alpha)

    br = self.elu2(self.linear2(br))
    treatment = self.linear3(
        br
    )  # Softmax is encapsulated into F.cross_entropy()
    return treatment

  def build_domain_label(self, br, detached=False):
    if detached:
      br = br.detach()
    if self.balancing == 'grad_reverse':
      br = grad_reverse(br, self.alpha)

    if self.alpha_prev_treat_max > 0:
      ret = self.linear7(self.elu6(self.linear6(br)))
      return ret
    if self.alpha_age_max > 0:
      ret = self.linear9(self.elu8(self.linear8(br)))
      return ret

  def build_outcome(self, br, current_treatment):
    x = torch.cat((br, current_treatment), dim=-1)
    x = self.elu3(self.linear4(x))
    outcome = self.linear5(x)
    return outcome

  def build_br(self, seq_output):
    br = self.elu1(self.linear1(seq_output))
    return br


class BRTreatmentOutcomeTLearnerHead(BRTreatmentOutcomeHead):
  """Used by CRN, EDCT, MultiInputTransformer.

  Replace the S-Learner in outcome prediction with a T-Learner.
  """

  def __init__(
      self,
      seq_hidden_units,
      br_size,
      fc_hidden_units,
      dim_treatments,
      dim_outcome,
      alpha=0.0,
      update_alpha=True,
      balancing='grad_reverse',
      alpha_prev_treat=0.0,
      update_alpha_prev_treat=False,
      alpha_age=0.0,
      update_alpha_age=False,
      is_one_hot_treatment=True,
  ):
    super().__init__(
        seq_hidden_units,
        br_size,
        fc_hidden_units,
        dim_treatments,
        dim_outcome,
        alpha,
        update_alpha,
        balancing,
        alpha_prev_treat,
        update_alpha_prev_treat,
        alpha_age,
        update_alpha_age,
    )
    self.is_one_hot_treatment = (
        is_one_hot_treatment  # treatment is either one-hot or binary list
    )
    if self.is_one_hot_treatment:
      self.treatment_type_num = dim_treatments
    else:
      self.treatment_type_num = 2**dim_treatments

    outcome_heads = []
    for _ in range(self.treatment_type_num):
      outcome_heads.append(
          nn.Sequential(
              nn.Linear(self.br_size, self.fc_hidden_units),
              nn.ELU(),
              nn.Linear(self.fc_hidden_units, self.dim_outcome),
          )
      )
    self.outcome_heads = nn.ModuleList(outcome_heads)

  def build_outcome(self, br, current_treatment):
    outcome_all_heads = torch.stack(
        [h(br) for h in self.outcome_heads], dim=-1
    )  # [B, T, D, H]
    if self.is_one_hot_treatment:
      head_selector = current_treatment.unsqueeze(-1)  # [B, T, H, 1]
      outcome = torch.matmul(outcome_all_heads, head_selector).squeeze(-1)
    else:
      multiplier = torch.tensor(
          [2**p for p in range(self.dim_treatments - 1, -1, -1)],
          dtype=current_treatment.dtype,
      ).to(current_treatment.device)
      head_selector = torch.round(
          (current_treatment * multiplier).sum(-1)
      ).long()  # [B, T]
      head_selector = (
          head_selector.unsqueeze(-1)
          .expand(-1, -1, self.dim_outcome)
          .unsqueeze(-1)
      )  # [B, T, D, 1]
      outcome = torch.gather(
          outcome_all_heads, dim=3, index=head_selector
      ).squeeze(-1)
    return outcome


class BROutcomeHead(nn.Module):
  """Used by CTRaw.

  Predict outcome only from (covariates, prev_treatments, prev_outcomes,
  curr_treatments)
  """

  def __init__(
      self,
      seq_hidden_units,
      br_size,
      fc_hidden_units,
      dim_treatments,
      dim_outcome,
      alpha=0.0,
      update_alpha=True,
      balancing='grad_reverse',
      alpha_prev_treat=0.0,
      update_alpha_prev_treat=False,
      alpha_age=0.0,
      update_alpha_age=False,
  ):
    super().__init__()

    self.seq_hidden_units = seq_hidden_units
    self.br_size = br_size
    self.fc_hidden_units = fc_hidden_units
    self.dim_treatments = dim_treatments
    self.dim_outcome = dim_outcome
    self.alpha = alpha if not update_alpha else 0.0
    self.alpha_max = alpha
    self.balancing = balancing

    self.alpha_prev_treat = (
        alpha_prev_treat if not update_alpha_prev_treat else 0.0
    )
    self.alpha_prev_treat_max = alpha_prev_treat
    self.alpha_age = alpha_age if not update_alpha_age else 0.0
    self.alpha_age_max = alpha_age

    self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
    self.elu1 = nn.ELU()

    self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
    self.elu2 = nn.ELU()
    self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

    self.linear4 = nn.Linear(self.br_size, self.fc_hidden_units)
    self.elu3 = nn.ELU()
    self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

    self.treatment_head_params = ['linear2', 'linear3']

    if self.alpha_prev_treat_max > 0:
      self.linear6 = nn.Linear(self.br_size, self.fc_hidden_units)
      self.elu6 = nn.ELU()
      self.linear7 = nn.Linear(self.fc_hidden_units, self.dim_treatments)
      self.treatment_head_params.extend(['linear6', 'linear7'])

    if self.alpha_age_max > 0:
      self.linear8 = nn.Linear(self.br_size, self.fc_hidden_units)
      self.elu8 = nn.ELU()
      self.linear9 = nn.Linear(self.fc_hidden_units, 1)
      self.treatment_head_params.extend(['linear8', 'linear9'])

  def build_treatment(self, br, detached=False):
    if detached:
      br = br.detach()

    if self.balancing == 'grad_reverse':
      br = grad_reverse(br, self.alpha)

    br = self.elu2(self.linear2(br))
    treatment = self.linear3(
        br
    )  # Softmax is encapsulated into F.cross_entropy()
    return treatment

  def build_domain_label(self, br, detached=False):
    if detached:
      br = br.detach()
    if self.balancing == 'grad_reverse':
      br = grad_reverse(br, self.alpha)

    if self.alpha_prev_treat_max > 0:
      ret = self.linear7(self.elu6(self.linear6(br)))
      return ret
    if self.alpha_age_max > 0:
      ret = self.linear9(self.elu8(self.linear8(br)))
      return ret

  def build_outcome(self, br, current_treatment):
    _ = current_treatment
    x = br
    x = self.elu3(self.linear4(x))
    outcome = self.linear5(x)
    return outcome

  def build_br(self, seq_output):
    br = self.elu1(self.linear1(seq_output))
    return br


class ROutcomeVitalsHead(nn.Module):
  """Used by G-Net."""

  def __init__(
      self,
      seq_hidden_units,
      r_size,
      fc_hidden_units,
      dim_outcome,
      dim_vitals,
      num_comp,
      comp_sizes,
  ):
    super().__init__()

    self.seq_hidden_units = seq_hidden_units
    self.r_size = r_size
    self.fc_hidden_units = fc_hidden_units
    self.dim_outcome = dim_outcome
    self.dim_vitals = dim_vitals
    self.num_comp = num_comp
    self.comp_sizes = comp_sizes

    self.linear1 = nn.Linear(self.seq_hidden_units, self.r_size)
    self.elu1 = nn.ELU()

    # Conditional distribution networks init
    self.cond_nets = []
    add_input_dim = 0
    for comp in range(self.num_comp):
      linear2 = nn.Linear(self.r_size + add_input_dim, self.fc_hidden_units)
      elu2 = nn.ELU()
      linear3 = nn.Linear(self.fc_hidden_units, self.comp_sizes[comp])
      self.cond_nets.append(nn.Sequential(linear2, elu2, linear3))

      add_input_dim += self.comp_sizes[comp]

    self.cond_nets = nn.ModuleList(self.cond_nets)

  def build_r(self, seq_output):
    r = self.elu1(self.linear1(seq_output))
    return r

  def build_outcome_vitals(self, r):
    vitals_outcome_pred = []
    for cond_net in self.cond_nets:
      out = cond_net(r)
      r = torch.cat((out, r), dim=-1)
      vitals_outcome_pred.append(out)
    return torch.cat(vitals_outcome_pred, dim=-1)


class AlphaRise(Callback):
  """Exponential alpha rise."""

  def __init__(self, rate='exp'):
    self.rate = rate

  def on_epoch_end(self, trainer, pl_module):
    if pl_module.hparams.exp.update_alpha:
      assert hasattr(pl_module, 'br_treatment_outcome_head')
      try:
        p = float(pl_module.current_epoch + 1) / float(
            pl_module.hparams.exp.max_epochs
        )
      except ZeroDivisionError:
        p = 1.0
      if self.rate == 'lin':
        pl_module.br_treatment_outcome_head.alpha = (
            p * pl_module.br_treatment_outcome_head.alpha_max
        )
      elif self.rate == 'exp':
        pl_module.br_treatment_outcome_head.alpha = (
            2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        ) * pl_module.br_treatment_outcome_head.alpha_max
      else:
        raise NotImplementedError()


def clip_normalize_stabilized_weights(
    stabilized_weights, active_entries, multiple_horizons=False
):
  """Used by RMSNs."""
  active_entries = active_entries.astype(bool)
  stabilized_weights[~np.squeeze(active_entries)] = np.nan
  sw_tilde = np.clip(
      stabilized_weights,
      np.nanquantile(stabilized_weights, 0.01),
      np.nanquantile(stabilized_weights, 0.99),
  )
  if multiple_horizons:
    sw_tilde = sw_tilde / np.nanmean(sw_tilde, axis=0, keepdims=True)
  else:
    sw_tilde = sw_tilde / np.nanmean(sw_tilde)

  sw_tilde[~np.squeeze(active_entries)] = 0.0
  sw_tilde[np.isnan(sw_tilde)] = 0.0
  return sw_tilde
