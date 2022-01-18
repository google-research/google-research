# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Class that implement  ARNet."""
import torch
import torch.nn as nn


class  ARNet(nn.Module):

  """Auto Regressive model as described in https://arxiv.org/abs/1911.12436.
  """

  def __init__(self, n_forecasts, n_lags, device):
    """Initializes a  ARNet instance.

    Args:
      n_forecasts: Number of time steps to forecast
      n_lags: Lags (past time steps) used to make forecast
      device: Device used  by the model
    """
    super(ARNet, self).__init__()
    self.n_lags = n_lags
    self.device = device
    self.n_forecasts = n_forecasts
    self.fc = nn.Linear(n_lags, 1, bias=False)
    nn.init.kaiming_normal_(self.fc.weight, mode="fan_in")

  def forward(self, x, true_output):
    """Forward pass for  ARNet.

    Args:
      x: A tensor of shape `(batch_size, n_lags)
      true_output: Actual forecast this is used for teacher forcing during
        training

    Returns:
      output: Forecast a tensor of shape `(batch_size, n_forecasts)`
    """
    output = torch.zeros((x.shape[0], self.n_forecasts)).to(self.device)
    output[:, 0] = self.fc(x).squeeze()

    if self.n_forecasts > self.n_lags:
      # If the forecast larger the lags than use orignal input and shift untill
      # the orginal inputs are done than use true output (teacher forecing).
      for i in range(1, self.n_lags):
        output[:,
               i] = self.fc(torch.cat((x[:, i:], true_output[:, :i]),
                                      dim=1)).squeeze()
      for i in range(0, self.n_forecasts - self.n_lags):
        output[:, i] = self.fc(true_output[:, i:i + self.n_lags]).squeeze()
    else:
      for i in range(1, self.n_forecasts):
        output[:,
               i] = self.fc(torch.cat((x[:, i:], true_output[:, :i]),
                                      dim=1)).squeeze()

    return output

  def predict(self, x):
    """Function used during testing to make predictions in an auto regressive style.

    Args:
      x : A tensor of shape `(batch_size, n_lags)
    Returns:
      output: Forecast a tensor of shape `(batch_size, n_forecasts)`
    """

    output = torch.zeros((x.shape[0], self.n_forecasts)).to(self.device)

    output[:, 0] = self.fc(x).squeeze()
    if self.n_forecasts > self.n_lags:
      # If the forecast larger the lags than use orignal input and shift untill
      # the orginal inputs are done than the input will only contain forecasted
      # values
      for i in range(1, self.n_lags):
        output[:, i] = self.fc(torch.cat((x[:, i:], output[:, :i]),
                                         dim=1)).squeeze()
      for i in range(0, self.n_forecasts - self.n_lags):
        output[:,
               self.n_lags + i] = self.fc(output[:,
                                                 i:i + self.n_lags]).squeeze()
    else:
      for i in range(1, self.n_forecasts):
        output[:, i] = self.fc(torch.cat((x[:, i:], output[:, :i]),
                                         dim=1)).squeeze()
    return output
