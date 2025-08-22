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

"""LSTM Encoder-Decoder architecture."""

# allow capital letter names for dimensions to improve clarity (e.g. N, T, D)
# pylint: disable=invalid-name

import random

from data_formatting.favorita_embedder import FavoritaEmbedder
import torch
from torch import nn


class LSTMEncoder(nn.Module):
  """LSTM Encoder."""

  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers,
      batch_first=True,
      device=torch.device('cpu'),
  ):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.device = device
    self.hidden = None

    self.lstm = (
        nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        .float()
        .to(device)
    )

  def forward(self, x_input):
    # output, hidden state, cell state
    output, self.hidden = self.lstm(x_input)
    return output, self.hidden

  def init_hidden(self, batch_size):
    return (
        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        ),
        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        ),
    )


class LSTMDecoder(nn.Module):
  """LSTM Decoder."""

  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers,
      output_size,
      batch_first=True,
      device=torch.device('cpu'),
  ):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.device = device

    self.lstm = (
        nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
        )
        .float()
        .to(device)
    )
    self.linear = nn.Linear(hidden_size, output_size).to(device)

  def forward(self, x_input, encoder_hidden_states):
    lstm_out, self.hidden = self.lstm(x_input.float(), encoder_hidden_states)
    output = self.linear(lstm_out)
    return output, self.hidden


class LstmEncoderLstmDecoder(nn.Module):
  """LSTM Encoder-Decoder architecture."""

  def __init__(
      self,
      input_size,
      hidden_size,
      num_layers,
      forecasting_horizon,
      training_prediction='teacher_forcing',
      teacher_forcing_ratio=0.5,
      scale01=False,
      device=torch.device('cpu'),
      target_dims=(0,),
      embedding_dim=None,
  ):
    """Constructor.

    Args:
      input_size: size of input dimension
      hidden_size: number of hidden units
      num_layers: number of layers in both the encoder and decoder
      forecasting_horizon: number of timepoints to forecast
      training_prediction: whether to use teacher_forcing, recursive, or
        mixed_teacher_forcing in training
      teacher_forcing_ratio: probability teacher forcing is used
      scale01: whether predictions are scaled between 0 and 1
      device: device to perform computations on
      target_dims: dimension of input corresponding to the desired target
      embedding_dim: size of embeddings
    """
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.forecasting_horizon = forecasting_horizon
    self.training_prediction = training_prediction
    self.teacher_forcing_ratio = teacher_forcing_ratio
    self.scale01 = scale01
    self.device = device
    self.target_dims = target_dims
    self.embedder = None
    if embedding_dim is not None:
      self.embedder = FavoritaEmbedder(embedding_dim=10, device=device).float()

    self.encoder = LSTMEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        device=device,
    )

    self.decoder = LSTMDecoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=input_size,
        batch_first=True,
        device=device,
    )

    self.output_layer = torch.nn.Linear(input_size, len(target_dims)).float()
    self.sigmoid = torch.nn.Sigmoid().to(device)

  def _embed(self, tensor):
    embedding = self.embedder(tensor.float())
    embedding = embedding.permute(0, 2, 1, 3)
    N, T, D, E = embedding.shape
    embedding = embedding.reshape(N, T, D * E)
    return embedding

  def forward(self, batch, in_eval=False):
    if in_eval:
      inputs = batch['eval_inputs']
      targets = batch['eval_targets']
    else:
      inputs = batch['model_inputs']
      targets = batch['model_targets']

    x_input = inputs.float()

    if self.embedder is not None:
      x_input = self._embed(x_input)

    # pass through encoder
    _, enc_hidden = self.encoder(x_input)

    # prepare first input for decoder
    # take the last observation of the window
    dec_input = x_input[:, -1, :].unsqueeze(1)
    dec_hidden = enc_hidden

    assert x_input.shape[-1] == self.input_size

    # decoding under different modes
    outputs = torch.zeros(
        targets.shape[0], self.forecasting_horizon, self.input_size
    ).to(self.device)
    mode = self.training_prediction
    if mode == 'teacher_forcing':  # each sequence entirely teacher or recursive
      if random.random() < self.teacher_forcing_ratio:
        mode = 'teacher'
      else:
        mode = 'recursive'

    for t in range(self.forecasting_horizon):
      dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
      assert dec_output.shape[1] == 1
      outputs[:, t] = dec_output.squeeze(1)

      if mode == 'recursive':
        dec_input = dec_output
      elif mode == 'teacher':
        dec_input = targets[:, t, :].unsqueeze(1)
        if self.embedder is not None:
          dec_input = self._embed(dec_input.float())
      elif (
          mode == 'mixed_teacher_forcing'
      ):  # each sequence is a mix of teacher and recursive
        if random.random() < self.teacher_forcing_ratio:
          dec_input = targets[:, t, :].unsqueeze(1)
          if self.embedder is not None:
            dec_input = self._embed(dec_input)
        else:
          dec_input = dec_output

    if self.embedder is not None:
      outputs = self.output_layer(outputs.float())
    else:
      outputs = outputs[:, :, self.target_dims]

    if self.scale01:
      outputs = self.sigmoid(outputs)
    return outputs
