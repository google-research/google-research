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
"""GATRNN model.

Defines the encoder, decoder, and complete GATRNN model architecture.
The gumbel softmax sampling function is also implemented to enable
gradient passing through the predicted graph.
"""

import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch import nn
import torch.nn.functional as F

from editable_graph_temporal.model import gat_cell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
  """Sample from the Gumbel distribution.

  Args:
    shape: shape of the random variables to be sampled.
    eps: a small value used to avoid doing logarithms on zero.

  Returns:
    [batch_size, n_class] sample from the Gumbel distribution.
  """
  uniform_rand = torch.rand(shape).to(device)
  return -torch.autograd.Variable(
      torch.log(-torch.log(uniform_rand + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution.

  Args:
    logits: [batch_size, n_class] unnormalized log-probs.
      temperature: non-negative scalar.
    eps: a small value used to avoid doing logarithms on zero.

  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
  """
  sample = sample_gumbel(logits.size(), eps=eps)
  y = logits + sample
  return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.

  Args:
    logits: [batch_size, n_class] unnormalized log-probs.
    temperature: non-negative scalar.
    hard: if True, take argmax, but differentiate w.r.t. soft sample y.
    eps: a small value used to avoid doing logarithms on zero.

  Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it
      will be a probabilitiy distribution that sums to 1 across classes.
  """
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
  if hard:
    shape = logits.size()
    _, k = y_soft.data.max(-1)
    y_hard = torch.zeros(*shape).to(device)
    y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
  else:
    y = y_soft
  return y


class Encoder(LightningModule, gat_cell.Seq2SeqAttrs):
  """Implements GATRNN encoder model.

  Encodes the input time series sequence to the hidden vector.
  """

  def __init__(self, args):
    """Instantiates the GATRNN encoder model.

    Args:
      args: python argparse.ArgumentParser class, we only use model-related
        arguments here.
    """
    super().__init__()
    self._initialize_arguments(args)
    self.embedding = nn.Linear(self.input_dim, self.rnn_units)
    torch.nn.init.normal_(self.embedding.weight)

    self.gat_layers = nn.ModuleList(
        [gat_cell.GATGRUCell(args) for _ in range(self.num_rnn_layers)])
    self.dropout = nn.Dropout(self.dropout)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()

  def forward(self, inputs, adj, global_embs, hidden_state=None):
    r"""Encoder forward pass.

    Args:
      inputs: input one-step time series, with shape (batch_size,
        self.num_nodes, self.input_dim).
      adj: adjacency matrix, with shape (self.num_nodes, self.num_nodes).
      global_embs: global embedding matrix, with shape (self.num_nodes,
        self.rnn_units).
      hidden_state (tensor): hidden vectors, with shape (num_layers, batch_size,
        self.rnn_units) optional, zeros if not provided.

    Returns:
      output: outputs, with shape (batch_size, self.num_nodes,
        self.rnn_units).
      hidden_state: output hidden vectors, with shape (num_layers,
        batch_size, self.num_nodes, self.rnn_units),
        (lower indices mean lower layers).
    """
    linear_weights = self.embedding.weight
    if torch.any(torch.isnan(linear_weights)):
      print("weight nan")
    embedded = self.embedding(inputs)
    embedded = self.tanh(embedded)

    output = self.dropout(embedded)

    if hidden_state is None:
      hidden_state = torch.zeros((self.num_rnn_layers, inputs.shape[0],
                                  self.num_nodes, self.rnn_units),
                                 device=device)
    hidden_states = []
    for layer_num, gat_layer in enumerate(self.gat_layers):
      next_hidden_state = gat_layer(output, hidden_state[layer_num], adj,
                                    global_embs)
      hidden_states.append(next_hidden_state)
      output = next_hidden_state

    # output = self.batch_norm(output)
    if self.activation == "relu":
      output = self.relu(output)
    elif self.activation == "tanh":
      output = self.tanh(output)
    elif self.activation == "linear":
      pass

    return output, torch.stack(
        hidden_states)  # runs in O(num_layers) so not too slow


class Decoder(LightningModule, gat_cell.Seq2SeqAttrs):
  """Implements GATRNN encoder model.

  Decodes the input hidden vector to the output time series sequence.
  """

  def __init__(self, args):
    """Instantiates the GATRNN encoder model.

    Args:
      args: python argparse.ArgumentParser class, we only use model-related
        arguments here.
    """
    super().__init__()
    self._initialize_arguments(args)
    self.embedding = nn.Linear(self.output_dim, self.rnn_units)

    self.gat_layers = nn.ModuleList(
        [gat_cell.GATGRUCell(args) for _ in range(self.num_rnn_layers)])
    self.fc_out = nn.Linear(self.rnn_units, self.output_dim)
    self.dropout = nn.Dropout(self.dropout)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, inputs, adj, global_embs, hidden_state=None):
    r"""Decoder forward pass.

    Args:
      inputs: input one-step time series, with shape (batch_size,
        self.num_nodes, self.output_dim).
      adj: adjacency matrix, with shape (self.num_nodes, self.num_nodes).
      global_embs: global embedding matrix, with shape (self.num_nodes,
        self.rnn_units).
      hidden_state (tensor): hidden vectors, with shape (num_layers, batch_size,
        self.rnn_units) optional, zeros if not provided.

    Returns:
      output: outputs, with shape (batch_size, self.num_nodes,
        self.output_dim).
      hidden_state: output hidden vectors, with shape (num_layers,
        batch_size, self.num_nodes, self.rnn_units),
        (lower indices mean lower layers).
    """
    embedded = self.tanh(self.embedding(inputs))
    output = self.dropout(embedded)
    # output = embedded

    hidden_states = []
    for layer_num, gat_layer in enumerate(self.gat_layers):
      next_hidden_state = gat_layer(output, hidden_state[layer_num], adj,
                                    global_embs)
      hidden_states.append(next_hidden_state)
      output = next_hidden_state

    # output = self.batch_norm(output)

    output = self.fc_out(output.view(-1, self.rnn_units))
    output = output.view(-1, self.num_nodes, self.output_dim)

    if self.activation == "relu":
      output = self.relu(output)
    elif self.activation == "tanh":
      output = self.tanh(output)
    elif self.activation == "linear":
      pass

    return output, torch.stack(hidden_states)


class GATRNN(LightningModule, gat_cell.Seq2SeqAttrs):
  """Implements the GATRNN model."""

  def __init__(self, adj_mx, args):
    """Instantiates the GATRNN encoder model.

    Args:
      adj_mx: adjacency matrix, with shape (self.num_nodes, self.num_nodes).
      args: python argparse.ArgumentParser class, we only use model-related
        arguments here.
    """
    super().__init__()
    self._initialize_arguments(args)
    self.temperature = args.temperature
    self.adj_type = args.adj_type
    if args.adj_type == "fixed":
      self.adj_mx = adj_mx.to(device)
    elif args.adj_type == "empty":
      self.adj_mx = torch.zeros(
          size=(args.num_nodes, args.num_nodes, args.num_relation_types),
          device=device).float()

    self.global_embs = nn.Parameter(
        torch.empty((self.num_nodes, self.rnn_units), device=device))
    torch.nn.init.xavier_normal_(self.global_embs)
    self.fc_out = nn.Linear(self.rnn_units * 2, self.rnn_units)
    self.fc_cat = nn.Linear(self.rnn_units, self.num_relation_types)
    self.encoder = Encoder(args)
    self.decoder = Decoder(args)
    self.fc_graph_rec, self.fc_graph_send = self._get_fc_graph_rec_send()
    self.loss = nn.L1Loss()

  def _get_fc_graph_rec_send(self):
    """Gets all two-node receiver and sender node indexes.

    This returns one-hot vectors for each of the pairs.

    Returns:
      (receiver node one-hot indexs, sender node one-hot indexs).
    """

    def encode_onehot(labels):
      """One-hot encoding.

      Args:
        labels: input labels containing integer numbers.

      Returns:
        label_onehot: one-hot vectors of labels.
      """
      classes = set(labels)
      classes_dict = {
          c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
      }
      labels_onehot = np.array(
          list(map(classes_dict.get, labels)), dtype=np.int32)
      return labels_onehot

    off_diag = np.ones([self.num_nodes, self.num_nodes])
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    return torch.FloatTensor(rel_rec).to(device), torch.FloatTensor(
        rel_send).to(device)

  def forward(self, inputs):
    """GATRNN forward pass.

    Args:
      inputs: input time series sequence, with shape (batch_size,
        self.input_len, self.num_nodes, self.input_dim).

    Returns:
      outputs: output time series sequence, with shape (batch_size,
        self.output_len, self.num_nodes,  self.output_dim).
      edge_prob: predicted relation type probability for each two-node pair,
        with shape (self.num_nodes*self.num_nodes, self.num_relation_types).
    """
    if self.adj_type == "learned":
      adj, edge_prob = self.pred_adj()
    elif self.adj_type in ["fixed", "empty"]:
      edge_prob = None
      adj = self.adj_mx if self.adj_mx.dim() == 3 else self.adj_mx.unsqueeze(2)

    inputs = inputs.permute(1, 0, 2, 3)

    encoder_hidden_state = None
    for t in range(self.input_len):
      _, encoder_hidden_state = self.encoder(inputs[t], adj, self.global_embs,
                                             encoder_hidden_state)

    decoder_hidden_state = encoder_hidden_state
    decoder_input = torch.zeros((encoder_hidden_state.shape[1], self.num_nodes,
                                 self.decoder.output_dim),
                                device=device)
    outputs = []
    for t in range(self.output_len):
      decoder_output, decoder_hidden_state = self.decoder(
          decoder_input, adj, self.global_embs, decoder_hidden_state)
      outputs.append(decoder_output)
      decoder_input = decoder_output

    outputs = torch.stack(outputs)

    del encoder_hidden_state
    del decoder_hidden_state

    outputs = outputs.permute(1, 0, 2, 3)
    return outputs, edge_prob

  def pred_adj(self):
    """Predict relational graph.

    Returns:
      adj: predicted adjacency matrix of relational graph,
        with shape (self.num_nodes, self.num_nodes,
        self.num_relation_types-1).
      prob: predicted relation type probability for each two-node pair,
        with shape (self.num_nodes*self.num_nodes, self.num_relation_types).
    """
    receivers = torch.matmul(self.fc_graph_rec, self.global_embs)
    senders = torch.matmul(self.fc_graph_send, self.global_embs)
    x = torch.cat([senders, receivers], dim=1)
    x = torch.relu(self.fc_out(x))
    x = self.fc_cat(x)
    prob = F.softmax(x, dim=-1)

    if self.training:
      adj = gumbel_softmax(x, temperature=self.temperature, hard=True)
    else:
      adj = x.argmax(dim=1)
      adj = F.one_hot(adj, num_classes=self.num_relation_types)

    adj = adj[:, 1:].clone().reshape(self.num_nodes, self.num_nodes,
                                     self.num_relation_types - 1)

    mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
    mask = mask.unsqueeze(2).repeat_interleave(
        self.num_relation_types - 1, dim=2)
    adj.masked_fill_(mask, 0)

    return adj, prob
