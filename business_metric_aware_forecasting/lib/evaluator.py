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

"""Evaluator object.

Computes all evaluation metrics.

Evaluation metrics include forecasting and inventory performance metrics.
"""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import numpy as np
from scipy.stats import norm
import torch
from utils.eval_utils import get_ragged_mean
from utils.eval_utils import get_ragged_sum
from utils.eval_utils import get_ragged_var


class Evaluator(object):
  """Evaluator class. Handles differentiable computation of metrics."""

  def __init__(
      self, first_cutoff, scale01, device, target_dims, no_safety_stock
  ):
    self.first_cutoff = first_cutoff
    self.scale01 = scale01
    self.device = device
    self.target_dims = target_dims
    self.no_safety_stock = no_safety_stock

  def _extract_target_dims(self, batch):
    """Extracts the slices corresponding to the target of interest.

    Args:
      batch: the batch data dictionary

    Returns:
      the batch data dictionary, sliced to extract the target
    """
    new_batch = batch.copy()
    for k, v in batch.items():
      if k in [
          'x',
          'x_scale',
          'x_offset',
          'model_inputs',
          'model_targets',
          'eval_inputs',
          'eval_targets',
      ]:
        if len(v.shape) == 2:
          v = v[:, self.target_dims]
        elif len(v.shape) == 3:
          v = v[:, :, self.target_dims]
        elif len(v.shape) == 4:
          v = v[:, :, :, self.target_dims]
        else:
          raise NotImplementedError('Unexpected number of dims: ', v.shape)
      new_batch[k] = v
    return new_batch

  def _rescale(self, arr, x_scale, x_offset):
    """Scale the array back up to its original values.

    Args:
      arr: scaled array
      x_scale: scale
      x_offset: offset

    Returns:
      array in its original range of values
    """
    if not x_scale.shape:
      return (arr * x_scale) + x_offset
    shape = arr.shape
    assert shape[0] == x_scale.shape[0]

    # repeat scale and offset to match up with arr shape
    to_expand = len(shape) - len(x_scale.shape)
    for _ in range(to_expand):
      x_scale = x_scale.unsqueeze(-1)
      x_offset = x_offset.unsqueeze(-1)
    x_scale = x_scale.repeat(1, *shape[1:])
    x_offset = x_offset.repeat(1, *shape[1:])
    return (arr * x_scale) + x_offset

  def _get_lengths_from_time_mask(self, time_mask):
    forecast_horizon_lengths = (
        time_mask[:, :, :, :, -1].sum(dim=2).unsqueeze(2).type(torch.int64)
    )  # D is always last
    time_lengths = (
        (time_mask[:, :, 0, :, -1] > 0)
        .float()
        .sum(dim=1)
        .unsqueeze(1)
        .type(torch.int64)
    )
    return forecast_horizon_lengths, time_lengths

  def compute_mse(
      self,
      preds,
      unfolded_actual_imputed,
      forecast_horizon_lengths,
      time_lengths,
      series_mean=True,
  ):
    """Compute the mean squared error, taking sequence lengths into account.

    Args:
      preds: predictions tensor
      unfolded_actual_imputed: actual values tensor, unfolded to be the same
        shape as predictions, and imputed to avoid issues with autodiff
      forecast_horizon_lengths: lengths of each forecast horizon
      time_lengths: lengths of each series
      series_mean: whether to take the mean across series

    Returns:
      mean squared error
    """
    squared_errs = (preds - unfolded_actual_imputed) ** 2  # N x T x L x D

    # handle first cutoff
    squared_errs = squared_errs[:, self.first_cutoff :, :]
    forecast_horizon_lengths = forecast_horizon_lengths[
        :, self.first_cutoff :, :
    ]
    time_lengths = time_lengths - self.first_cutoff

    # get average along forecasting horizon
    mse = get_ragged_mean(
        squared_errs, lens=forecast_horizon_lengths, axis=-2, device=self.device
    )
    # get average along time
    mse = get_ragged_mean(mse, lens=time_lengths, axis=-2, device=self.device)
    # get average across all series
    if series_mean:
      mse = mse.mean()
    return mse

  def _get_std_e(
      self,
      preds,
      unfolded_actual_imputed,
      unfolded_time_mask,
      eps=1e-5,
  ):
    """Compute the standard deviation over previous forecast errors.

    Args:
      preds: predictions tensor (N x T x L x D)
      unfolded_actual_imputed: actual values tensor, unfolded to be the same
        shape as predictions, and imputed to avoid issues with autodiff
      unfolded_time_mask: times mask, the [:,:,:,t] slice corresponds to whether
        the corresponding timepoint has passed
      eps: small constant for stability

    Returns:
      tensor of standard deviations
    """
    N, T, _, _ = preds.shape

    squared_errs = (preds - unfolded_actual_imputed) ** 2
    squared_errs = squared_errs.unsqueeze(-1).repeat(1, 1, 1, 1, T)
    masked_errs = squared_errs * unfolded_time_mask

    # handle first cutoff
    masked_errs = masked_errs[:, :, :, :, self.first_cutoff :]
    mask = unfolded_time_mask[:, :, :, :, self.first_cutoff :]
    mask_denom_nonzero = mask.sum(2).sum(1)  # takes errors per timestep
    mask_denom_nonzero = (mask_denom_nonzero != 0).float()
    mask_denom_nonzero = mask.sum(2).sum(1) + (
        1 - mask_denom_nonzero
    )  # fills in a 1 wherever it's 0

    avg_per_time = masked_errs.sum(2).sum(1) / mask_denom_nonzero  # N x T
    avg_per_time = torch.concat(
        [torch.zeros((N, 1, 1)).to(self.device), avg_per_time[:, :, :-1]],
        axis=2,
    )  # start @ 0
    std_e = torch.sqrt(
        avg_per_time + eps
    )  # square root causes some problems if MSE is 0
    std_e = std_e.permute(0, 2, 1)
    return std_e

  def compute_forecasting_metrics(
      self,
      preds,
      actual_batch,
      eps=1,
      periodicity=12,
      series_mean=True,
      rolling_eval=False,
      min0=False,
  ):
    """Computes forecasting metrics.

    Args:
      preds: predicted values
      actual_batch: batch with actual values
      eps: small constant for stability
      periodicity: number of timepoints in a period
      series_mean: whether to take mean across series
      rolling_eval: whether evaluation in performed in a roll-forward manner
      min0: whether to cut off predictions at 0 as the minimum (e.g. since
        negative demand is impossible)

    Returns:
      dictionary of forecasting metrics
    """
    N, T, L, D = preds.shape

    x_scale = actual_batch['x_scale']
    x_offset = actual_batch['x_offset']

    if 'eval_targets' in actual_batch:  # dealing with windowed input
      x_imputed = actual_batch['x']  # should have 144 timepoints for m3
      unfolded_actual_imputed = actual_batch['eval_targets']
      forecast_horizon_lengths, time_lengths = self._get_lengths_from_time_mask(
          actual_batch['eval_target_times_mask']
      )
    else:
      x_imputed = actual_batch['x_imputed']
      unfolded_actual_imputed = actual_batch['unfolded_actual_imputed']
      forecast_horizon_lengths = actual_batch['forecast_horizon_lengths']
      time_lengths = actual_batch['time_lengths']

    forecast_horizon_lengths = forecast_horizon_lengths[
        :, self.first_cutoff :, :
    ]
    time_lengths = time_lengths - self.first_cutoff

    if self.scale01:
      preds = self._rescale(preds, x_scale, x_offset)
      x_imputed = self._rescale(x_imputed, x_scale, x_offset)
      unfolded_actual_imputed = self._rescale(
          unfolded_actual_imputed, x_scale, x_offset
      )
      if min0:
        preds = torch.nn.functional.relu(preds)
    if x_imputed.min() < 0 or unfolded_actual_imputed.min() < 0:
      raise NotImplementedError(
          'unexpected value in x_imputed or unfolded_actual_imputed'
      )

    test_actual = unfolded_actual_imputed[:, self.first_cutoff :, :]
    test_preds = preds[:, self.first_cutoff :, :]

    # MSE
    mse = self.compute_mse(
        test_preds,
        test_actual,
        forecast_horizon_lengths,
        time_lengths,
        series_mean=series_mean,
    )

    # MPE
    mpe = get_ragged_mean(
        (test_actual - test_preds) / (test_actual + eps),
        lens=forecast_horizon_lengths,
        axis=-2,
        device=self.device,
    )
    mpe = get_ragged_mean(mpe, lens=time_lengths, axis=-2, device=self.device)
    if series_mean:
      mpe = mpe.mean()

    # sMAPE
    smape = get_ragged_mean(
        (test_actual - test_preds).abs() * 2.0 / (test_actual.abs() + eps),
        lens=forecast_horizon_lengths,
        axis=-2,
        device=self.device,
    )
    smape = get_ragged_mean(
        smape, lens=time_lengths, axis=-2, device=self.device
    )

    if series_mean:
      smape = smape.mean()

    # MASE
    ae = (unfolded_actual_imputed - preds).abs()
    ae = ae[:, self.first_cutoff :, :]

    if 'eval_targets' in actual_batch:
      full_N, full_T, full_D = (
          x_imputed.shape
      )  # expect x_imputed 2nd dim to match original timescale so times correct
      scale = torch.zeros((full_N, full_T, full_D)).to(self.device)
      scale[:, periodicity:] = (
          x_imputed[:, periodicity:] - x_imputed[:, :-periodicity]
      ).abs()
      scale = torch.cumsum(scale, dim=1)

      scale_ct = torch.zeros((full_N, full_T, full_D)).to(self.device)
      scale_ct[:, periodicity:] = (
          torch.arange(1, full_T - periodicity + 1)
          .unsqueeze(-1)
          .unsqueeze(0)
          .repeat(full_N, 1, full_D)
      )

      scale = scale / scale_ct
      if rolling_eval:  # each sample is actually a decoding point
        num_start_ts, num_roll_ts, _, num_dims = ae.shape  # t1, t2, l, d

        # figure out scaling factor corresponding to each element of ae
        start_ts = (actual_batch['target_times'][:, 0, 0] - 1).type(torch.int64)
        scales_unrolled = torch.cat(
            [
                scale,
                torch.ones(num_start_ts, num_roll_ts - 1, num_dims).to(
                    self.device
                )
                * 1e18,
            ],
            axis=1,
        ).unfold(1, num_roll_ts, 1)
        scales_unrolled = scales_unrolled.permute(0, 1, 3, 2)
        scales_unrolled = torch.cat(
            [
                scales_unrolled,
                torch.ones(
                    scales_unrolled.shape[0],
                    1,
                    scales_unrolled.shape[2],
                    scales_unrolled.shape[3],
                ).to(self.device),
            ],
            axis=1,
        )

        start_ts = torch.clamp(start_ts, max=scales_unrolled.shape[1] - 1)
        scale = torch.gather(
            scales_unrolled,
            1,
            start_ts.unsqueeze(-2)
            .unsqueeze(-2)
            .repeat(1, 1, scales_unrolled.shape[2], 1),
        ).squeeze(1)
      else:
        first_cutoff = int(actual_batch['target_times'].min().item()) - 1
        if 'max_t_cutoff' in actual_batch:
          max_t_cutoff = actual_batch['max_t_cutoff']
          scale = scale[:, first_cutoff:max_t_cutoff]
        else:
          scale = scale[:, first_cutoff:]
    else:
      assert periodicity <= self.first_cutoff
      scale = torch.zeros((N, T)).to(self.device)
      scale[:, periodicity:] = (
          unfolded_actual_imputed[:, periodicity:, 0]
          - unfolded_actual_imputed[:, :-periodicity, 0]
      ).abs()
      scale = torch.cumsum(scale, dim=1)

      scale_ct = torch.zeros((N, T)).to(self.device)
      scale_ct[:, periodicity:] = torch.arange(1, T - periodicity + 1)

      scale = scale / scale_ct
      scale = scale[:, self.first_cutoff :]

    nans = np.empty((N, L - 1, D))
    nans[:] = np.nan
    nans = torch.from_numpy(nans).float().to(self.device)
    scale = torch.cat([scale, nans], dim=1).unfold(1, L, 1).permute(0, 1, 3, 2)
    scale = torch.nan_to_num(scale, nan=1.0)

    if scale.shape[1] < ae.shape[1]:
      print(scale.shape, ae.shape)
      ones = torch.ones(
          ae.shape[0], ae.shape[1] - scale.shape[1], ae.shape[2], ae.shape[3]
      )
      scale = torch.cat([scale, ones], dim=1)
    ase = ae / scale

    mase = get_ragged_mean(
        ase, lens=forecast_horizon_lengths, axis=-2, device=self.device
    )
    mase = get_ragged_mean(mase, lens=time_lengths, axis=-2, device=self.device)
    if series_mean:
      mase = mase.mean()

    forecasting_metrics = {
        'mse': mse,
        'mpe': mpe,
        'smape': smape,
        'mase': mase,
    }
    return forecasting_metrics

  def _get_lagged(self, matrix, lag=1, same_size=True):
    N, _, D = matrix.shape  # N x T x D
    pad = torch.zeros((N, lag, D)).to(self.device)
    lagged = torch.concat([pad, matrix], axis=1)
    if same_size:
      lagged = lagged[:, :-lag, :]
    return lagged

  def compute_inventory_metrics(
      self,
      preds,
      actual_batch,
      target_service_level=0.95,
      unit_holding_cost=1,
      unit_var_o_cost=1.0 / 100000.0,
      unit_stockout_cost=1,
      series_mean=True,
      quantile_loss=None,
      naive_metrics=None,
      min0=False,
  ):
    """Computes inventory metrics.

    Args:
      preds: predicted values
      actual_batch: batch with actual values
      target_service_level: service level to use for safety stock calculation
      unit_holding_cost: cost per unit held
      unit_var_o_cost: cost per unit order variance
      unit_stockout_cost: cost per unit stockout
      series_mean:  whether to take mean across series
      quantile_loss: quantile loss objective, if relevant
      naive_metrics: baseline metrics
      min0: whether to cut off predictions at 0 as the minimum (e.g. since
        negative demand is impossible)

    Returns:

    """

    x_scale = actual_batch['x_scale']
    x_offset = actual_batch['x_offset']
    _, _, lead_time, target_D = actual_batch['eval_targets'].shape  # N, T, L, D

    if 'eval_targets' in actual_batch:  # dealing with windowed input
      x_imputed = actual_batch['eval_targets'][:, :, 0]
      unfolded_actual_imputed = actual_batch['eval_targets']
      forecast_horizon_lengths, time_lengths = self._get_lengths_from_time_mask(
          actual_batch['eval_target_times_mask']
      )
      unfolded_time_mask = actual_batch['eval_target_times_mask']
    else:
      x_imputed = actual_batch['x_imputed']
      unfolded_actual_imputed = actual_batch['unfolded_actual_imputed']
      time_lengths = actual_batch['time_lengths']
      forecast_horizon_lengths = actual_batch['forecast_horizon_lengths']
      unfolded_time_mask = actual_batch['unfolded_time_mask']

    time_lengths = time_lengths - self.first_cutoff
    if self.scale01:
      preds = self._rescale(preds, x_scale, x_offset)
      unfolded_actual_imputed = self._rescale(
          unfolded_actual_imputed, x_scale, x_offset
      )
      x_imputed = self._rescale(x_imputed, x_scale, x_offset)
      if min0:
        preds = torch.nn.functional.relu(preds)

    preds = preds * unfolded_time_mask[:, :, :, :, -1]
    lead_forecasts = preds.sum(axis=2)  # N x T
    lead_forecasts = lead_forecasts[:, self.first_cutoff :]
    if quantile_loss or self.no_safety_stock:
      safety_stocks = torch.zeros(lead_forecasts.shape).to(self.device)
    else:
      std_e = self._get_std_e(
          preds,
          unfolded_actual_imputed,
          unfolded_time_mask,
          eps=1e-5,
      )  # N x T
      std_e = std_e * lead_time  # approximate lead time std_e
      safety_const = norm.ppf(target_service_level)
      safety_stocks = safety_const * std_e  # N x T

    inventory_positions = (
        self._get_lagged(lead_forecasts)
        + self._get_lagged(safety_stocks)
        - x_imputed[:, self.first_cutoff :]
    )

    orders = lead_forecasts + safety_stocks - inventory_positions

    recent_demand = (
        self._get_lagged(x_imputed, lag=lead_time - 1, same_size=False)
        .unfold(1, lead_time, 1)
        .permute(0, 1, 3, 2)
    )
    # works because there's at least lead time worth of real obs
    recent_horizon_lengths = torch.cat(
        [
            torch.ones(x_imputed.shape[0], lead_time - 1, 1, target_D).to(
                self.device
            )
            * lead_time,
            forecast_horizon_lengths[:, : -(lead_time - 1), :, :],
        ],
        axis=1,
    ).type(torch.int64)
    recent_demand = get_ragged_sum(
        recent_demand, recent_horizon_lengths, device=self.device, axis=2
    )

    net_inventory_levels = (
        self._get_lagged(lead_forecasts, lag=lead_time)
        + self._get_lagged(safety_stocks, lag=lead_time)
        - recent_demand
    )

    work_in_progress = inventory_positions - net_inventory_levels

    holding_cost = (
        torch.nn.functional.relu(net_inventory_levels) * unit_holding_cost
    )
    holding_cost = get_ragged_mean(
        holding_cost, time_lengths, device=self.device, axis=1
    )
    if series_mean:
      holding_cost = holding_cost.mean()

    soft_holding_cost = (
        torch.nn.functional.softplus(net_inventory_levels) * unit_holding_cost
    )
    soft_holding_cost = get_ragged_mean(
        soft_holding_cost, time_lengths, device=self.device, axis=1
    )
    if series_mean:
      soft_holding_cost = soft_holding_cost.mean()

    var_o = get_ragged_var(
        orders,
        torch.maximum(time_lengths, torch.Tensor([0]).to(self.device)).type(
            torch.int64
        ),
        device=self.device,
        axis=1,
    )  # avg of variance of orders for each series
    if series_mean:
      var_o = var_o.mean()

    var_o_cost = var_o * unit_var_o_cost

    # proportion of orders that are negative
    prop_neg_orders = get_ragged_mean(
        (orders < 0).float(), time_lengths, device=self.device, axis=1
    )
    if series_mean:
      prop_neg_orders = prop_neg_orders.mean()

    # how often stockout occurs
    achieved_service_level = get_ragged_mean(
        (net_inventory_levels >= 0).float(),
        time_lengths,
        device=self.device,
        axis=1,
    )
    if series_mean:
      achieved_service_level = achieved_service_level.mean()

    soft_alpha = torch.sigmoid(net_inventory_levels * 1e2)
    soft_alpha = get_ragged_mean(
        soft_alpha, time_lengths, device=self.device, axis=1
    )
    if series_mean:
      soft_alpha = soft_alpha.mean()

    # stockout cost
    stockout_cost = (
        torch.nn.functional.relu(-net_inventory_levels) * unit_stockout_cost
    )
    stockout_cost = get_ragged_mean(
        stockout_cost, time_lengths, device=self.device, axis=1
    )
    if series_mean:
      stockout_cost = stockout_cost.mean()

    # compute rms
    rms = torch.sqrt(
        (
            holding_cost**2
            + var_o**2
            + (1.0 / (achieved_service_level + 1e-5)) ** 2
        )
        / 3.0
    )

    # cost
    total_cost = holding_cost + stockout_cost + var_o_cost

    inventory_values = {
        'inventory_positions': inventory_positions,
        'net_inventory_levels': net_inventory_levels,
        'work_in_progress': work_in_progress,
        'safety_stocks': safety_stocks,
        'orders': orders,
        'lead_forecasts': lead_forecasts,
        'unfolded_actual_imputed': unfolded_actual_imputed,
        'unfolded_time_mask': unfolded_time_mask,
        'time_lengths': time_lengths,
        'demand': x_imputed,
    }

    inventory_metrics = {
        'holding_cost': holding_cost,
        'soft_holding_cost': soft_holding_cost,
        'var_o': var_o,
        'var_o_cost': var_o_cost,
        'prop_neg_orders': prop_neg_orders,
        'achieved_service_level': achieved_service_level,
        'soft_achieved_service_level': soft_alpha,
        'stockout_cost': stockout_cost,
        'rms': rms,
        'total_cost': total_cost,
        'inventory_values': inventory_values,
    }

    # compute scaled_rms
    if naive_metrics:
      scaled_holding_cost = holding_cost / (naive_metrics['holding_cost'] + 1)
      scaled_var_o = var_o / (naive_metrics['var_o'] + 1)
      scaled_achieved_service_level = achieved_service_level / (
          naive_metrics['achieved_service_level'] + 0.1
      )
      scaled_rms = torch.sqrt(
          (
              scaled_holding_cost**2
              + scaled_var_o**2
              + (1.0 / (scaled_achieved_service_level + 0.1)) ** 2
          )
          / 3.0
      )

      rel_holding_cost = (holding_cost - naive_metrics['holding_cost']) / (
          naive_metrics['holding_cost'] + 1
      )
      rel_var_o = (var_o - naive_metrics['var_o']) / (
          naive_metrics['var_o'] + 1
      )
      rel_achieved_service_level = (
          (1.0 / (achieved_service_level + 0.1))
          - (1.0 / (naive_metrics['achieved_service_level'] + 0.1))
      ) / (1.0 / (naive_metrics['achieved_service_level'] + 0.1))
      rel_stockout_cost = (stockout_cost - naive_metrics['stockout_cost']) / (
          naive_metrics['stockout_cost'] + 1
      )
      rel_rms_avg = (
          torch.sigmoid(rel_holding_cost)
          + torch.sigmoid(rel_var_o)
          + torch.sigmoid(rel_achieved_service_level)
      ) / 3.0
      rel_rms_2 = (
          (torch.sigmoid(rel_holding_cost) ** 2)
          + (torch.sigmoid(rel_var_o) ** 2)
          + (torch.sigmoid(rel_achieved_service_level) ** 2)
      )
      rel_rms_3 = (
          (torch.sigmoid(rel_holding_cost) ** 3)
          + (torch.sigmoid(rel_var_o) ** 3)
          + (torch.sigmoid(rel_achieved_service_level) ** 3)
      )
      rel_rms_5 = (
          (torch.sigmoid(rel_holding_cost) ** 5)
          + (torch.sigmoid(rel_var_o) ** 5)
          + (torch.sigmoid(rel_achieved_service_level) ** 5)
      )
      rel_rms_logsumexp = torch.logsumexp(
          torch.cat(
              [
                  torch.sigmoid(rel_holding_cost).unsqueeze(0),
                  torch.sigmoid(rel_var_o).unsqueeze(0),
                  torch.sigmoid(rel_achieved_service_level).unsqueeze(0),
              ],
              dim=0,
          ),
          dim=0,
      )

      rel_rms_stockout_2 = (
          (torch.sigmoid(rel_holding_cost) ** 2)
          + (torch.sigmoid(rel_var_o) ** 2)
          + (torch.sigmoid(rel_stockout_cost) ** 2)
      )
      rel_rms_stockout_3 = (
          (torch.sigmoid(rel_holding_cost) ** 3)
          + (torch.sigmoid(rel_var_o) ** 3)
          + (torch.sigmoid(rel_stockout_cost) ** 3)
      )
      rel_rms_stockout_5 = (
          (torch.sigmoid(rel_holding_cost) ** 5)
          + (torch.sigmoid(rel_var_o) ** 5)
          + (torch.sigmoid(rel_stockout_cost) ** 5)
      )

      inventory_metrics['scaled_rms'] = scaled_rms
      inventory_metrics['rel_rms_avg'] = rel_rms_avg
      inventory_metrics['rel_rms_2'] = rel_rms_2
      inventory_metrics['rel_rms_3'] = rel_rms_3
      inventory_metrics['rel_rms_5'] = rel_rms_5
      inventory_metrics['rel_rms_logsumexp'] = rel_rms_logsumexp
      inventory_metrics['rel_rms_stockout_2'] = rel_rms_stockout_2
      inventory_metrics['rel_rms_stockout_3'] = rel_rms_stockout_3
      inventory_metrics['rel_rms_stockout_5'] = rel_rms_stockout_5

    return inventory_metrics

  def compute_all_metrics(
      self,
      preds,
      actual_batch,
      target_service_level,
      unit_holding_cost,
      unit_stockout_cost,
      unit_var_o_cost,
      series_mean,
      quantile_loss,
      naive_model,
      scale_by_naive_model=False,
      rolling_eval=False,
      min0=False,
  ):
    """Given predictions, computes all metrics of interest.

    Args:
      preds: predicted values
      actual_batch: batch with actual values
      target_service_level: service level to use for safety stock calculation
      unit_holding_cost: cost per unit held
      unit_stockout_cost: cost per unit stockout
      unit_var_o_cost: cost per unit order variance
      series_mean:  whether to take mean across series
      quantile_loss: quantile loss objective, if relevant
      naive_model: baseline model
      scale_by_naive_model: whether to scale performance by baseline model
      rolling_eval: whether evaluation is roll-forward
      min0: whether to cut off predictions at 0 as the minimum (e.g. since
        negative demand is impossible)

    Returns:
    """
    actual_batch = self._extract_target_dims(actual_batch)

    all_metrics = {}

    _, T, _, _ = preds.shape  # N x T x L x D

    immediate_series_mean = False

    # compute naive model metrics
    naive_all_metrics = {}
    with torch.no_grad():
      naive_preds = naive_model(actual_batch, in_eval=True)
      naive_preds = naive_preds[:, :T, :, :]
      naive_inventory_metrics = self.compute_inventory_metrics(
          naive_preds,
          actual_batch,
          target_service_level=target_service_level,
          unit_holding_cost=unit_holding_cost,
          unit_stockout_cost=unit_stockout_cost,
          unit_var_o_cost=unit_var_o_cost,
          series_mean=immediate_series_mean,
      )
      naive_forecasting_metrics = self.compute_forecasting_metrics(
          naive_preds,
          actual_batch,
          series_mean=immediate_series_mean,
          rolling_eval=rolling_eval,
      )
      naive_all_metrics.update(naive_inventory_metrics)
      naive_all_metrics.update(naive_forecasting_metrics)

    # compute inventory metrics
    inventory_metrics = self.compute_inventory_metrics(
        preds,
        actual_batch,
        target_service_level=target_service_level,
        unit_holding_cost=unit_holding_cost,
        unit_stockout_cost=unit_stockout_cost,
        unit_var_o_cost=unit_var_o_cost,
        series_mean=immediate_series_mean,
        quantile_loss=quantile_loss,
        naive_metrics=naive_all_metrics,
    )

    for metric_name, metric_val in inventory_metrics.items():
      if metric_name == 'inventory_values':
        all_metrics[metric_name] = metric_val
        continue
      if scale_by_naive_model:
        metric_val = metric_val / (naive_all_metrics[metric_name] + 1e-5)
      if (not immediate_series_mean) and series_mean:
        metric_val = metric_val.mean()
      all_metrics[metric_name] = metric_val

    # compute forecasting metrics
    forecasting_metrics = self.compute_forecasting_metrics(
        preds,
        actual_batch,
        series_mean=immediate_series_mean,
        rolling_eval=rolling_eval,
    )

    for metric_name, metric_val in forecasting_metrics.items():
      if scale_by_naive_model and metric_name != 'mpe':
        metric_val = metric_val / (naive_all_metrics[metric_name] + 1e-5)
      if (not immediate_series_mean) and series_mean:
        metric_val = metric_val.mean()
      all_metrics[metric_name] = metric_val

    # add quantile loss
    if quantile_loss:
      if self.scale01:
        preds = self._rescale(
            preds, actual_batch['x_scale'], actual_batch['x_offset']
        )
        targets = self._rescale(
            actual_batch['eval_targets'],
            actual_batch['x_scale'],
            actual_batch['x_offset'],
        )
        if min0:
          preds = torch.nn.functional.relu(preds)
      forecast_horizon_lengths, time_lengths = self._get_lengths_from_time_mask(
          actual_batch['eval_target_times_mask']
      )
      qloss = quantile_loss(
          preds, targets, forecast_horizon_lengths, time_lengths
      )
      if series_mean:
        qloss = qloss.mean()
      all_metrics['quantile_loss'] = qloss

    return all_metrics
