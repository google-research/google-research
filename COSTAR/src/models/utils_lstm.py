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

"""LSTM utils."""

import torch
from torch import nn


class VariationalLSTM(nn.Module):
  """Variational LSTM layer in Pytorch."""

  def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
    super().__init__()

    self.lstm_layers = [
        nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
    ]
    if num_layer > 1:
      self.lstm_layers += [
          nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
          for _ in range(num_layer - 1)
      ]
    self.lstm_layers = nn.ModuleList(self.lstm_layers)

    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate

  def forward(self, x, init_states=None):
    for lstm_cell in self.lstm_layers:
      # Customised LSTM-cell for variational LSTM dropout
      # (Tensorflow-like implementation)
      if init_states is None:  # Encoder - init states are zeros
        hx, cx = torch.zeros((x.shape[0], self.hidden_size)).type_as(
            x
        ), torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
      else:  # Decoder init states are br of encoder
        hx, cx = init_states, init_states

      # Variational dropout - sampled once per batch
      out_dropout = torch.bernoulli(
          hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)
      ) / (1 - self.dropout_rate)
      h_dropout = torch.bernoulli(
          hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)
      ) / (1 - self.dropout_rate)
      c_dropout = torch.bernoulli(
          cx.data.new(cx.data.size()).fill_(1 - self.dropout_rate)
      ) / (1 - self.dropout_rate)

      output = []
      for t in range(x.shape[1]):
        hx, cx = lstm_cell(x[:, t, :], (hx, cx))
        if lstm_cell.training:
          out = hx * out_dropout
          hx, cx = hx * h_dropout, cx * c_dropout
        else:
          out = hx
        output.append(out)

      x = torch.stack(output, dim=1)

    return x
