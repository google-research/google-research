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

"""NaiveScalingBaseline model.

Baseline model which repeats the previous observation as its prediction, scaled
by some learned scaling factor alpha.
"""


import torch


class NaiveScalingBaseline(torch.nn.Module):
  """NaiveScalingBaseline pytorch module.

  Given an observed sequence as input = (1,2,3,4),
  and for a desired lead time to make forecasts for L = 3,
  this model will forecast: 4 x alpha, 4 x alpha, 4 x alpha.
  Alpha is a learned parameter with initial value = 1.

  We expect that the final value of alpha should be a positive number
  close to 1, however different downstream metrics may push alpha further
  or closer to 1. Deviations from 1 might roughly be interpreted as follows:

  * 0 < alpha < 1: underforecasting may be beneficial for downstream metrics
  * alpha > 1: overforecasting may be beneficial for downstream metrics
  """

  def __init__(
      self,
      forecasting_horizon,
      init_alpha=1.0,
      periodicity=1,
      frozen=False,
      device=torch.device('cpu'),
      target_dims=(0,),
  ):
    """Initializes the NaiveScalingBaseline.

    Args:
      forecasting_horizon: number of timesteps to forecast (i.e. number of times
        to repeat the last observation)
      init_alpha: initial value of alpha
      periodicity: number of timesteps corresponding to one period
      frozen: whether alpha should be frozen or learned
      device: device to perform computations on
      target_dims: input dimensions corresponding to target values
    """
    super().__init__()
    self.forecasting_horizon = forecasting_horizon
    self.init_alpha = init_alpha
    self.periodicity = periodicity
    self.frozen = frozen

    self.alpha = torch.nn.Parameter(torch.Tensor([init_alpha]).to(device))
    self.alpha.requires_grad = not frozen
    self.alpha = self.alpha.to(device)

    self.target_dims = list(target_dims)

    self.device = device

  def forward(self, batch, in_eval=False):
    """Makes forecasts for the next self.forecasting_horizon timesteps.

    Args:
      batch: batch dictionary of data from a WindowedSequenceDataset. Should
        contain an 'inputs' key which maps to a tensor of size N x T x L, where
        N is number of samples in the batch, T is the number of timepoints, and
        L is the lead time.
      in_eval: whether in evaluation mode or prediction mode (influences whether
        there is an additional dimension corresponding to unrolled timepoints)

    Returns:
      predictions for subsequent timesteps
    """
    if in_eval:
      inputs = batch['eval_inputs']

      if self.periodicity != 1:
        # get most recent period worth of inputs
        last = inputs[:, :, -self.periodicity :] * self.alpha
        # repeat this period if output size is longer than period
        last = last.repeat(
            1, 1, int(self.forecasting_horizon / float(self.periodicity)) + 1, 1
        )
        last = last[:, :, : self.forecasting_horizon]  # cut off at output size
      else:
        last = inputs[:, :, -1] * self.alpha
        last = last.unsqueeze(-1).repeat(1, 1, self.forecasting_horizon)
    else:
      inputs = batch['model_inputs']

      if self.periodicity != 1:
        # get most recent period worth of inputs
        last = inputs[:, -self.periodicity :, :] * self.alpha
        # repeat this period if output size is longer than period
        last = last.repeat(
            1, int(self.forecasting_horizon / float(self.periodicity)) + 1, 1
        )
        last = last[:, : self.forecasting_horizon, :]  # cut off at output size
      else:
        last = inputs[:, -1, :] * self.alpha
        last = last.unsqueeze(-2).repeat(1, self.forecasting_horizon, 1)

    if len(last.shape) == 3:
      return last[:, :, self.target_dims]
    elif len(last.shape) == 4:
      return last[:, :, :, self.target_dims]
    else:
      raise NotImplementedError('unexpected number of dims: ', len(last.shape))

  def __str__(self):
    return (
        f'NaiveScalingBaseline(forecasting_horizon={self.forecasting_horizon},'
        f' init_alpha={self.init_alpha}, periodicity={self.periodicity},'
        f' frozen={self.frozen})'
    )
