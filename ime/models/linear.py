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

"""Class that implement Linear."""
import torch.nn as nn


class Linear(nn.Module):
  """Simple Linear Model.
  """

  def __init__(self, n_forecasts, n_lags):
    """Initializes a  ARNet instance.

    Args:
      n_forecasts: Number of time steps to forecast
      n_lags: Lags (past time steps) used to make forecast
    """
    super(Linear, self).__init__()
    self.n_lags = n_lags
    self.n_forecasts = n_forecasts
    self.fc = nn.Linear(n_lags, n_forecasts)
    nn.init.kaiming_normal_(self.fc.weight, mode="fan_in")

  def forward(self, x):
    """Forward pass for Linear.

    Args:
      x: A tensor of shape `(batch_size, n_lags)

    Returns:
      output: Forecast a tensor of shape `(batch_size, n_forecasts)`
    """
    output = self.fc(x).squeeze()
    return output
