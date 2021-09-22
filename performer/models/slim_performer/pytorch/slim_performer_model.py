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

# pylint: disable=invalid-name, unused-argument
"""Main SLiMPerformer model."""
import numpy as np
import torch

import performer.models.slim_performer.pytorch.numerator_and_denominator as num_and_den


def valid_feature_type(feature_type):
  bool1 = feature_type in ['relu', 'elu+1', 'sqr', 'favor+']
  bool2 = feature_type.startswith('favor+') and feature_type.split(
      '_')[1].isdigit()
  return bool1 or bool2


class SLiMPerformer(torch.nn.Module):
  """Full SLiMPerformer Transformer model."""

  def __init__(self, vocab_size, hidden_dim, n_layers, ffn_dim, n_heads,
               feature_type, compute_type, on_gptln):
    assert valid_feature_type(feature_type)
    assert compute_type in ['ps', 'iter', 'parallel_ps']
    super(SLiMPerformer, self).__init__()

    self._vocab_size = vocab_size
    self._hidden_dim = hidden_dim
    self._feature_type = feature_type

    self.input_map = torch.nn.Embedding(vocab_size, hidden_dim // 2)
    self.output_logit_map = torch.nn.Linear(hidden_dim, vocab_size)

    self.layers = torch.nn.ModuleList([
        SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads, feature_type,
                           compute_type, on_gptln) for _ in range(n_layers)
    ])

  def full_forward(self, x):
    """Naive full forward pass."""

    x = self.input_map(x)
    x = self._concat_pos_embs(x, 0)

    for layer in self.layers:
      x = layer.full_forward(x, layer.attention.sample_rfs(x.device))

    x = self.output_logit_map(x)

    return x

  def full_loss(self,
                inputs,
                with_grad=False,
                nonpad_mask=None,
                return_acc=False):
    """Naive full loss and grad."""

    if nonpad_mask is None:
      nonpad_mask = torch.ones_like(inputs, dtype=torch.bool)

    float_nonpad_mask = nonpad_mask.float()

    logits = self.full_forward(inputs[:, :-1])
    logits = logits.transpose(1, 2)
    neg_loglikes = torch.nn.functional.cross_entropy(
        logits, inputs[:, 1:], reduction='none')

    loss = (neg_loglikes * float_nonpad_mask[:, 1:]).sum()
    loss = loss / (1e-8 + float_nonpad_mask[:, 1:].sum())

    if with_grad:
      loss.backward()

    if return_acc:
      correct = nonpad_mask[:, 1:] & (inputs[:, 1:] == torch.argmax(logits, 1))
      acc = correct.float().sum() / float_nonpad_mask.sum()
      return loss, acc

    return loss

  def loss_with_grad(self,
                     inputs,
                     step_size,
                     nonpad_mask=None,
                     return_acc=False):
    """step_size: size of a parallel step, `1` corresponds to a fully-sequential mode."""

    if nonpad_mask is None:
      nonpad_mask = torch.ones_like(inputs, dtype=torch.bool)

    loss = 0.0
    sums = None
    x = inputs[:, :-1]

    if return_acc:
      acc = 0.0

    if self._feature_type == 'favor+':
      rfs = [layer.attention.sample_rfs(inputs.device) for layer in self.layers]
    else:
      rfs = [None] * len(self.layers)

    loss_normalization = nonpad_mask[:, 1:].float().sum() + 1e-8

    with torch.no_grad():

      for start_index, end_index, cur_x, cur_sums in self._forward_gen(
          step_size, x, rfs):

        sums = cur_sums

        cur_gt_preds = inputs[:, start_index + 1:end_index + 1]
        cur_nonpad_mask = nonpad_mask[:, start_index + 1:end_index + 1]
        float_cur_nonpad_mask = cur_nonpad_mask.float()

        c_x = cur_x.transpose(1, 2)

        neg_loglikes = torch.nn.functional.cross_entropy(
            c_x, cur_gt_preds, reduction='none')
        cur_loss = (neg_loglikes * float_cur_nonpad_mask).sum()

        loss = loss + cur_loss.detach().clone()

        if return_acc:
          correct = cur_nonpad_mask & (cur_gt_preds == torch.argmax(c_x, 1))
          acc = acc + correct.float().sum().detach().clone()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

      loss = loss / loss_normalization

      if return_acc:
        acc = acc / loss_normalization

    ps_gradients = None

    seq_len = int(inputs.shape[1]) - 1
    start_indices = np.arange(0, seq_len, step_size)[::-1]

    for start_index in start_indices:

      end_index = min(seq_len, start_index + step_size)

      cur_x = x[:, start_index:end_index]
      cur_gt_preds = inputs[:, start_index + 1:end_index + 1]
      float_cur_nonpad_mask = nonpad_mask[:, start_index + 1:end_index +
                                          1].float()

      cur_x = self.input_map(cur_x)
      cur_x = self._concat_pos_embs(cur_x, start_index)

      new_sums = []
      cur_sums = []

      for layer, (num_sums, den_sums), layer_rfs in zip(self.layers, sums, rfs):

        cur_x, new_num_sums, new_den_sums, cur_num_sums, cur_den_sums = layer.incr_step(
            cur_x, num_sums, den_sums, False, layer_rfs, start_index == 0)

        new_sums.append((new_num_sums, new_den_sums))
        cur_sums.append((cur_num_sums, cur_den_sums))

      sums = new_sums

      cur_x = self.output_logit_map(cur_x)

      cur_x = cur_x.transpose(1, 2)

      neg_loglikes = torch.nn.functional.cross_entropy(
          cur_x, cur_gt_preds, reduction='none')
      cur_loss = (neg_loglikes * float_cur_nonpad_mask).sum()

      cur_loss = cur_loss / loss_normalization

      if ps_gradients is not None:
        cur_loss = cur_loss + sum([(z[0] * y[0]).sum() + (z[1] * y[1]).sum()
                                   for z, y in zip(cur_sums, ps_gradients)
                                   if (y[0] is not None) and (y[1] is not None)
                                  ])

      cur_loss.backward()

      ps_gradients = [[(None if y.grad is None else y.grad.detach().clone())
                       for y in z]
                      for z in sums]

      torch.cuda.synchronize()
      torch.cuda.empty_cache()

    if return_acc:
      return loss, acc

    return loss

  def _forward_gen(self, step_size, x, rfs):

    sums = [layer.attention.init_sums(x.device) for layer in self.layers]

    seq_len = int(x.shape[1])

    for start_index in range(0, seq_len, step_size):

      end_index = min(seq_len, start_index + step_size)

      cur_x = self.input_map(x[:, start_index:end_index])
      cur_x = self._concat_pos_embs(cur_x, start_index)

      new_sums = []

      for layer, (num_sums, den_sums), layer_rfs in zip(self.layers, sums, rfs):
        cur_x, new_num_sums, new_den_sums = layer.incr_step(
            cur_x, num_sums, den_sums, True, layer_rfs, start_index == 0)
        new_sums.append((new_num_sums, new_den_sums))

      sums = new_sums

      cur_x = self.output_logit_map(cur_x)

      yield start_index, end_index, cur_x, sums

  def _concat_pos_embs(self, x, start_index):

    pos_emb_size = self._hidden_dim // 2

    positions = torch.arange(
        start_index, start_index + x.shape[1], dtype=x.dtype, device=x.device)
    freqs = torch.exp(
        torch.arange(0, pos_emb_size, 2, dtype=x.dtype, device=x.device) *
        (-np.log(10000) / pos_emb_size))
    args = positions[None, :, None] * freqs[None, None, :]
    sin_pos_embs = torch.sin(args) * torch.ones_like(x[:, :1, :1])
    cos_pos_embs = torch.cos(args) * torch.ones_like(x[:, :1, :1])

    return torch.cat([x, sin_pos_embs, cos_pos_embs], 2)


class SLiMPerformerLayer(torch.nn.Module):
  """Single SLiMPerformer layer (MLPs + Attention + LayerNorm)."""

  def __init__(self, hidden_dim, ffn_dim, n_heads, feature_type, compute_type,
               on_gptln):

    super(SLiMPerformerLayer, self).__init__()

    self.attention = MultiHeadAttention(feature_type, n_heads, hidden_dim,
                                        compute_type)

    self.U_map = torch.nn.Linear(hidden_dim, ffn_dim)
    self.V_map = torch.nn.Linear(ffn_dim, hidden_dim)
    self.layernorm1 = torch.nn.LayerNorm(hidden_dim)
    self.layernorm2 = torch.nn.LayerNorm(hidden_dim)

    self._on_gptln = on_gptln

  def full_forward(self, x, rfs):

    skip = x

    if not self._on_gptln:
      x = self.layernorm1(x)

    x = self.attention.full_forward(x, rfs)

    if self._on_gptln:
      x = self.layernorm1(x)

    x = skip + x

    x = self._ffn(x)

    return x

  def incr_step(self, x, num_sums, den_sums, on_forward, rfs, on_start):

    skip = x

    if not self._on_gptln:
      x = self.layernorm1(x)

    result = self.attention.incr_step(x, num_sums, den_sums, on_forward, rfs,
                                      on_start)

    if on_forward:
      x, num_sums, den_sums = result
    else:
      x, init_num_sums, init_den_sums, num_sums, den_sums = result  # pylint: disable=unbalanced-tuple-unpacking

    if self._on_gptln:
      x = self.layernorm1(x)

    x = skip + x
    x = self._ffn(x)

    if on_forward:
      return x, num_sums, den_sums

    return x, init_num_sums, init_den_sums, num_sums, den_sums

  def _ffn(self, x):

    skip = x

    if not self._on_gptln:
      x = self.layernorm2(x)

    x = self.U_map(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map(x)

    if self._on_gptln:
      x = self.layernorm2(x)

    x = skip + x

    return x


class MultiHeadAttention(torch.nn.Module):
  """Explicit multihead attention using prefix sum."""

  def __init__(self, feature_type, n_heads, hidden_dim, compute_type):

    super(MultiHeadAttention, self).__init__()

    self._feature_type = feature_type
    self._n_heads = n_heads
    self._hidden_dim = hidden_dim
    self._compute_type = compute_type

    self.q_map = torch.nn.Linear(hidden_dim, hidden_dim)
    self.k_map = torch.nn.Linear(hidden_dim, hidden_dim)
    self.v_map = torch.nn.Linear(hidden_dim, hidden_dim)

  def full_forward(self, x, rfs):

    queries, keys, values = self._get_queries_keys_values(x, rfs)

    num_sums, den_sums = self.init_sums(x.device)

    if self._compute_type == 'iter':
      num, _ = num_and_den.num_iter(queries, keys, values, num_sums)
      den, _ = num_and_den.den_iter(queries, keys, den_sums)
    elif self._compute_type == 'ps':
      num, _ = num_and_den.num_ps(queries, keys, values, num_sums, False)
      den, _ = num_and_den.den_ps(queries, keys, den_sums, False)
    else:
      num, _ = num_and_den.num_ps(queries, keys, values, num_sums, True)
      den, _ = num_and_den.den_ps(queries, keys, den_sums, True)

    num = torch.transpose(num, 0, 1)
    den = torch.transpose(den, 0, 1)

    outputs = num / (den[Ellipsis, None] + 1e-16)
    outputs = outputs.reshape(x.shape)

    return outputs

  def init_sums(self, device):

    head_dim = self._hidden_dim // self._n_heads

    if self._feature_type.startswith('favor+_'):
      splitted = self._feature_type.split('_')
      feature_dim = int(splitted[1]) * head_dim
    else:
      feature_dim = head_dim

    num_sums = torch.zeros([1, self._n_heads, feature_dim, head_dim],
                           device=device)
    den_sums = torch.zeros([1, self._n_heads, feature_dim], device=device)

    return num_sums, den_sums

  def incr_step(self, x, num_sums, den_sums, on_forward, rfs, on_start):

    queries, keys, values = self._get_queries_keys_values(x, rfs)

    if not on_forward:

      if on_start:
        num_sums = torch.zeros_like(num_sums)
        den_sums = torch.zeros_like(den_sums)
      elif self._compute_type == 'iter':
        num_sums = num_and_den.num_reverse_sums_iter(queries, keys, values,
                                                     num_sums)
        den_sums = num_and_den.den_reverse_sums_iter(queries, keys, den_sums)
      else:
        num_sums = num_and_den.num_reverse_sums_ps(queries, keys, values,
                                                   num_sums)
        den_sums = num_and_den.den_reverse_sums_ps(queries, keys, den_sums)

      num_sums = num_sums.detach().clone()
      num_sums.requires_grad = True
      den_sums = den_sums.detach().clone()
      den_sums.requires_grad = True

      init_num_sums = num_sums
      init_den_sums = den_sums

    if self._compute_type == 'iter':
      num, num_sums = num_and_den.num_iter(queries, keys, values, num_sums)
      den, den_sums = num_and_den.den_iter(queries, keys, den_sums)
    elif self._compute_type == 'ps':
      num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, False)
      den, den_sums = num_and_den.den_ps(queries, keys, den_sums, False)
    else:
      num, num_sums = num_and_den.num_ps(queries, keys, values, num_sums, True)
      den, den_sums = num_and_den.den_ps(queries, keys, den_sums, True)

    num = torch.transpose(num, 0, 1)
    den = torch.transpose(den, 0, 1)

    outputs = num / (den[Ellipsis, None] + 1e-16)
    outputs = outputs.reshape(x.shape)

    if on_forward:
      return outputs, num_sums, den_sums

    return outputs, init_num_sums, init_den_sums, num_sums, den_sums

  def _get_queries_keys_values(self, inputs, rfs):

    queries = self.q_map(inputs)
    keys = self.k_map(inputs)
    values = self.v_map(inputs)

    queries = queries.reshape(
        [queries.shape[0], queries.shape[1], self._n_heads, -1])
    keys = keys.reshape([keys.shape[0], keys.shape[1], self._n_heads, -1])
    values = values.reshape(
        [values.shape[0], values.shape[1], self._n_heads, -1])

    if self._feature_type == 'relu':
      queries = torch.nn.functional.relu(queries)
      keys = torch.nn.functional.relu(keys)
    elif self._feature_type == 'elu+1':
      queries = torch.nn.functional.elu(queries) + 1
      keys = torch.nn.functional.elu(keys) + 1
    elif self._feature_type == 'sqr':
      queries = queries**2
      keys = keys**2
    elif self._feature_type == 'abs':
      queries = torch.abs(queries)
      keys = torch.abs(keys)
    else:

      head_dim = self._hidden_dim // self._n_heads

      queries = queries * np.power(head_dim, -0.25)
      queries = torch.einsum('ijkl,klm->ijkm', queries, rfs) - (queries**2).sum(
          3, keepdim=True) / 2
      queries = torch.exp(queries)

      keys = keys * np.power(head_dim, -0.25)
      keys = torch.einsum('ijkl,klm->ijkm', keys, rfs) - (keys**2).sum(
          3, keepdim=True) / 2
      keys = torch.exp(keys)

    queries = queries.transpose(0, 1)
    keys = keys.transpose(0, 1)
    values = values.transpose(0, 1)

    return queries, keys, values

  def sample_rfs(self, device):

    if not self._feature_type.startswith('favor+'):
      return None

    if self._feature_type == 'favor+':
      factor = 1
    else:
      splitted = self._feature_type.split('_')
      factor = int(splitted[1])

    head_dim = self._hidden_dim // self._n_heads

    rfs = [[
        _sample_orth_matrix(head_dim, device)[None, Ellipsis] for _ in range(factor)
    ] for _ in range(self._n_heads)]
    rfs = [torch.cat(x, 2) for x in rfs]
    rfs = torch.cat(rfs, 0)
    rfs = rfs * np.sqrt(head_dim)

    return rfs


def _sample_orth_matrix(size, device):
  """Samples orthogonal matrix to reduce variance for random features."""
  subspace = torch.randn(size, size, device=device)
  subspace = torch.tril(subspace)
  subspace = subspace / torch.sqrt((subspace**2).sum(0, keepdim=True))

  S = torch.triu(subspace.T.mm(subspace)) - 0.5 * torch.eye(
      subspace.shape[1], device=device)

  result = torch.eye(
      subspace.shape[0], device=device) - subspace.mm(torch.inverse(S)).mm(
          subspace.T)

  return result
