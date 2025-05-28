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

# pylint: disable=function-missing-types

"""Latent graph forecaster model that integrates."""
import sys

from typing import Any, Dict, Tuple

import numpy as np
sys.path.insert(0, '../src/')
from src.lgf_cell import LGFCell  # pylint: disable=g-import-not-at-top
from src.lgf_cell import Seq2SeqAttrs

from pytorch_lightning import LightningModule  # pylint: disable=g-bad-import-order
import torch
from torch import nn
from torch import optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


class Encoder(LightningModule, Seq2SeqAttrs):
  """Encoder module.

  Attributes:
    embedding: embedding layer for the input
    lgf_layers: latent graph forecaster layer
  """

  def __init__(self, adj_mx, args):
    super().__init__()

    Seq2SeqAttrs.__init__(self, args)
    self.embedding = nn.Linear(self.input_dim, self.rnn_units)
    torch.nn.init.normal_(self.embedding.weight)

    self.lgf_layers = nn.ModuleList(
        [LGFCell(adj_mx, args) for _ in range(self.num_rnn_layers)])
    # self.batch_norm  = nn.BatchNorm1d(self.num_nodes)
    self.dropout = nn.Dropout(self.dropout)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  def forward(self, inputs,
              hidden_state = None):
    """Encoder forward pass.

    Args:
        inputs (tensor): [batch_size, self.num_nodes, self.input_dim]
        hidden_state (tensor): [num_layers, batch_size, self.rnn_units]
          optional, zeros if not provided

    Returns:
        output: # shape (batch_size, num_nodes,  self.rnn_units)
        hidden_state # shape (num_layers, batch_size, num_nodes,
        self.rnn_units)
          (lower indices mean lower layers)
    """
    linear_weights = self.embedding.weight
    if torch.any(torch.isnan(linear_weights)):
      print('weight nan')
    embedded = self.embedding(inputs)
    embedded = self.tanh(embedded)

    output = self.dropout(embedded)

    if hidden_state is None:
      hidden_state = torch.zeros((self.num_rnn_layers, self.batch_size,
                                  self.num_nodes, self.rnn_units),
                                 device=device)
    hidden_states = []
    for layer_num, dcgru_layer in enumerate(self.lgf_layers):
      next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
      hidden_states.append(next_hidden_state)
      output = next_hidden_state

    # output = self.batch_norm(output)
    if self.activation == 'relu':
      output = self.relu(output)
    elif self.activation == 'tanh':
      output = self.tanh(output)
    elif self.activation == 'linear':
      pass

    return output, torch.stack(
        hidden_states)  # runs in O(num_layers) so not too slow


class Decoder(LightningModule, Seq2SeqAttrs):
  """Decoder module.

  Attributes:
    embedding: embedding layer for the output
    lgf_layers: latent graph forecaster layer
    fc_out: fully connected output layer
  """

  def __init__(self, adj_mx, args):
    # super().__init__(is_training, adj_mx, **model_kwargs)
    super().__init__()
    Seq2SeqAttrs.__init__(self, args)

    self.embedding = nn.Linear(self.output_dim, self.rnn_units)

    self.lgf_layers = nn.ModuleList(
        [LGFCell(adj_mx, args) for _ in range(self.num_rnn_layers)])

    # self.batch_norm = nn.BatchNorm1d(self.num_nodes)

    self.fc_out = nn.Linear(self.rnn_units, self.output_dim)
    self.dropout = nn.Dropout(self.dropout)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, inputs,
              hidden_state = None):
    """Decoder forward pass.

    Args:
        inputs: [batch_size, self.num_nodes, self.output_dim]
        hidden_state: [num_layers, batch_size, self.hidden_state_size] optional,
          zeros if not provided

    Returns:
        output: [batch_size, self.num_nodes,  self.output_dim]
          hidden_state: [num_layers, batch_size, self.hidden_state_size]
          (lower indices mean lower layers)
    """
    embedded = self.tanh(self.embedding(inputs))
    output = self.dropout(embedded)

    hidden_states = []
    for layer_num, dcgru_layer in enumerate(self.lgf_layers):
      next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
      hidden_states.append(next_hidden_state)
      output = next_hidden_state

    # output = self.batch_norm(output)

    output = self.fc_out(output.view(-1, self.rnn_units))
    output = output.view(-1, self.num_nodes, self.output_dim)

    if self.activation == 'relu':
      output = self.relu(output)
    elif self.activation == 'tanh':
      output = self.tanh(output)
    elif self.activation == 'linear':
      pass

    return output, torch.stack(hidden_states)


class LGF(LightningModule, Seq2SeqAttrs):
  """Lightning module for Latent Graph Forecaster model.

  Attributes:
    adj_mx: initialize if learning the graph, load the graph if known
    encoder: encoder module
    decoder: decoder module
  """

  def __init__(self,
               adj_mx,
               args,
               config=None):
    super().__init__()
    Seq2SeqAttrs.__init__(self, args)
    if self.filter_type == 'learned':
      # a global learnable graph
      print('create graph')
      self.adj_mx = nn.Parameter(adj_mx.to(device))  # initialization
    else:
      print('initialize graph')
      self.adj_mx = adj_mx

    self.encoder = Encoder(self.adj_mx, args)
    self.decoder = Decoder(self.adj_mx, args)

    # define loss function
    self.loss = nn.L1Loss()

    if config is not None:
      self.hidden_dim = config['hidden_dim']

  def _compute_sampling_threshold(self, batches_seen):
    return self.cl_decay_steps / (
        self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

  def forward(self,
              inputs,
              labels = None,
              batches_seen = None):
    """LGF forward pass.

    Args:
        inputs: [seq_len, batch_size, num_nodes, input_dim]
        labels: [horizon, batch_size, num_nodes, output_dim]
        batches_seen: batches seen till now

    Returns:
        output: [self.horizon, batch_size, self.num_nodes,
        self.output_dim]
    """

    # reshape [batch, seq_len, num_nodes, dim]
    #           -- > [seq_len, batch, num_nodes, dim]

    inputs.to(device)
    inputs = inputs.permute(1, 0, 2, 3)
    if labels is not None:
      labels.to(device)
      labels = labels.permute(1, 0, 2, 3)

    encoder_hidden_state = None
    for t in range(self.input_len):
      # if torch.any(torch.isnan(inputs[t])):
      #     print('encoder time step', t)
      _, encoder_hidden_state = self.encoder(inputs[t], encoder_hidden_state)

    encoder_hidden_state.detach()

    decoder_hidden_state = encoder_hidden_state

    decoder_input = torch.zeros(
        (self.batch_size, self.num_nodes, self.decoder.output_dim),
        device=device)
    outputs = []
    for t in range(self.output_len):
      # if torch.any(torch.isnan(decoder_input)):
      #     print('decoder time step', t)
      decoder_output, decoder_hidden_state = self.decoder(
          decoder_input, decoder_hidden_state)
      outputs.append(decoder_output)

      # teacher_forcing_ratio = 0 if labels!=None
      # else 0 # self._compute_sampling_threshold(batches_seen)
      # teacher_force = random.random() < teacher_forcing_ratio
      # if teacher forcing, use actual next token as next input;
      # if not, use predicted token
      # decoder_input = labels[t,:,:,:self.output_dim] if teacher_force
      # else decoder_output
      decoder_input = decoder_output
      # if self.training and self.use_curriculum_learning:
      #     c = np.random.uniform(0, 1)
      #     if c < self._compute_sampling_threshold(batches_seen):
      #         decoder_input = labels[t]
    outputs = torch.stack(outputs)
    outputs.detach()

    del encoder_hidden_state
    del decoder_hidden_state

    # self._logger.debug("Decoder complete")
    # if batches_seen == 0:
    #     self._logger.info(
    #         "Total trainable parameters {}".format(count_parameters(self))
    #     )
    # permute back
    outputs = outputs.permute(1, 0, 2, 3)
    return outputs

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)

  def training_step(self, batch,
                    batch_idx):
    x, y = batch
    loss = self.loss(self(x, None), y)
    return {'loss': loss, 'log': {'train_loss': loss.detach()}}

  def validation_step(self, batch,
                      batch_idx):
    x, y = batch
    loss = self.loss(self(x, None), y)
    self.log('val_loss', loss)
    return {'val_loss': loss, 'log': {'val_loss': loss.detach()}}

  def test_step(self, batch, batch_idx):
    x, y = batch
    loss = self.loss(self(x, None), y)
    self.log('test_loss', loss)
    return {'test_loss': loss, 'log': {'test_loss': loss.detach()}}

  def predict_step(self, batch,
                   batch_idx):
    x, y = batch
    if x.shape[0] != self.batch_size:
      return None
    return (self(x, None), y)

  def validation_epoch_end(self, outputs):
    val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
    # show val_acc in progress bar but only log val_loss
    results = {
        'progress_bar': {
            'val_loss': val_loss_mean.item()
        },
        'log': {
            'val_loss': val_loss_mean.item()
        },
        'val_loss': val_loss_mean.item()
    }
    return results
