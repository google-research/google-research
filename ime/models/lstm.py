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

"""Class that implement LSTM."""
import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTMime(nn.Module):
  """An implementation  of LSTM.

      This LSTM is used by interpretable mixture of expert as assigner module.
  """

  def __init__(self,
               input_size,
               num_experts,
               out_len,
               d_model=512,
               layers=3,
               dropout=0.0,
               device=torch.device('cuda:0')
               ):
    """Initializes a LSTMime instance.

    Args:
      input_size: Input features dimension
      num_experts: Number of experts
      out_len:  Forecasting horizon
      d_model: Hidden layer dimension
      layers: Number of LSTM layers.
      dropout: Fraction of neurons affected by Dropout (default=0.0).
      device: Device used  by the model

    """

    super(LSTMime, self).__init__()
    self.hidden_size = d_model
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=d_model,
                        num_layers=layers, batch_first=True)
    self.fc1 = nn.Linear(d_model, d_model)
    self.fc2 = nn.Linear(d_model, num_experts)
    self.fc3 = nn.Linear(d_model, out_len)
    self.drop_out = nn.Dropout(dropout)
    self.device = device
    self.num_layers = layers

  def forward(self, x):
    """Forward pass for LSTMime.

    Args:
      x: A tensor of shape `(batch_size, seqence_length, input_size)`

    Returns:
      output: Output used for expert classification a tensor of shape
      `(batch_size, num_experts, 1)`
      reg_out: Regression output the forecast a tensor of `(batch_size,
      out_len)`
    """
    h_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size)).to(self.device)

    c_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size)).to(self.device)

    out, (_, _) = self.lstm(x, (h_0, c_0))
    out = self.drop_out(out)

    out = out[:, -1, :]
    out = self.fc1(out)
    output = self.fc2(out)
    reg_out = self.fc3(out)
    return output.unsqueeze(-1), reg_out


class LSTM(nn.Module):
  """An implementation  of LSTM for forecasting.
  """

  def __init__(self,
               input_size,
               out_len,
               d_model=512,
               layers=3,
               dropout=0.0,
               device=torch.device('cuda:0')):
    """Initializes a LSTM instance.

    Args:
      input_size: Input features dimension
      out_len:  Forecasting horizon
      d_model: Hidden layer dimension
      layers: Number of LSTM layers.
      dropout: Fraction of neurons affected by Dropout (default=0.0).
      device: Device used  by the model
    """
    super(LSTM, self).__init__()
    self.hidden_size = d_model

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=d_model,
                        num_layers=layers, batch_first=True)

    self.fc1 = nn.Linear(d_model, d_model)
    self.fc2 = nn.Linear(d_model, out_len)
    self.drop_out = nn.Dropout(dropout)
    self.device = device
    self.num_layers = layers

  def forward(self, x):
    """Forward pass for LSTM.

    Args:
      x: A tensor of shape `(batch_size, seqence_length, input_size)`

    Returns:
      output: The forecast, a tensor of shape `(batch_size, out_len, 1)`
    """
    h_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size)).to(self.device)

    c_0 = Variable(torch.zeros(
        self.num_layers, x.size(0), self.hidden_size)).to(self.device)
    out, (_, _) = self.lstm(x, (h_0, c_0))
    out = self.drop_out(out)
    out = out[:, -1, :]
    out = self.fc1(out)
    output = self.fc2(out)

    return output.unsqueeze(-1)
