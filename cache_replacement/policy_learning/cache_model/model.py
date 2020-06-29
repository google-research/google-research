# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# python3
"""Defines a models to predict eviction policies."""

import abc
import collections
import itertools
from absl import logging
import numpy as np
import torch
from torch import distributions as td
from torch import nn
from torch.nn import functional as F
from cache_replacement.policy_learning.cache_model import attention
from cache_replacement.policy_learning.cache_model import embed
from cache_replacement.policy_learning.cache_model import loss as L
from cache_replacement.policy_learning.cache_model import utils


class EvictionPolicyModel(nn.Module):
  """A model that approximates an eviction policy."""

  @classmethod
  def from_config(cls, config):
    """Creates a model from the config.

    Args:
      config (cfg.Config): specifies model architecture / hyperparams.

    Returns:
      EvictionPolicyModel: created based on the config.
    """
    address_embedder = embed.from_config(config.get("address_embedder"))
    pc_embedder = embed.from_config(config.get("pc_embedder"))
    if config.get("cache_line_embedder") == "address_embedder":
      cache_line_embedder = address_embedder
    else:
      cache_line_embedder = embed.from_config(config.get("cache_line_embedder"))

    cache_pc_embedder_type = config.get("cache_pc_embedder")
    if cache_pc_embedder_type == "none":  # default no pc embedding
      cache_pc_embedder = None
    elif config.get("cache_pc_embedder") == "pc_embedder":  # shared embedder
      cache_pc_embedder = pc_embedder
    else:
      cache_pc_embedder = embed.from_config(config.get("cache_pc_embedder"))

    positional_embedder = embed.from_config(config.get("positional_embedder"))

    supported = {
        "log_likelihood": LogProbLoss,
        "reuse_dist": ReuseDistanceLoss,
        "ndcg": ApproxNDCGLoss,
        "kl": KLLoss,
    }
    loss_fns = {loss_type: supported[loss_type]()
                for loss_type in config.get("loss")}

    return cls(pc_embedder, address_embedder, cache_line_embedder,
               positional_embedder, config.get("lstm_hidden_size"),
               config.get("max_attention_history"), loss_fns,
               cache_pc_embedder=cache_pc_embedder)

  def __init__(self, pc_embedder, address_embedder, cache_line_embedder,
               positional_embedder, lstm_hidden_size, max_attention_history,
               loss_fns=None, cache_pc_embedder=None):
    """Constructs a model to predict evictions from a EvictionEntries history.

    At each timestep t, receives:
      - pc_t: program counter of t-th memory access.
      - a_t: (cache-aligned) address of t-th memory access.
      - [l^0_t, ..., l^N_t]: the cache lines present in the cache set accessed
        by a_t. Each cache line consists of the cache-aligned address and the pc
        of the last access to that address.

    Computes:
      c_0, h_0 = zeros(lstm_hidden_size)
      c_{t + 1}, h_{t + 1} = LSTM([e(pc_t)]; e(a_t)], c_t, h_t)
      h^i = attention([h_{t - K}, ..., h_t], query=e(l^i_t)) for i = 1, ..., N
      eviction_score s^i = softmax(f(h^i))

    The line with the highest eviction score is evicted.

    Args:
      pc_embedder (embed.Embedder): embeds the program counter.
      address_embedder (embed.Embedder): embeds the address.
      cache_line_embedder (embed.Embedder): embed the cache line.
      positional_embedder (embed.Embedder): embeds positions of the access
        history.
      lstm_hidden_size (int): dimension of output of LSTM (h and c).
      max_attention_history (int): maximum number of past hidden states to
        attend over (K in the equation above).
      loss_fns (dict): maps a name (str) to a loss function (LossFunction).
        The name is used in the loss method. Defaults to top_1_log_likelihood.
      cache_pc_embedder (embed.Embedder | None): embeds the pc of each cache
        line, if provided. Otherwise cache line pcs are not embedded.
    """
    super(EvictionPolicyModel, self).__init__()
    self._pc_embedder = pc_embedder
    self._address_embedder = address_embedder
    self._cache_line_embedder = cache_line_embedder
    self._cache_pc_embedder = cache_pc_embedder
    self._lstm_cell = nn.LSTMCell(
        pc_embedder.embed_dim + address_embedder.embed_dim, lstm_hidden_size)
    self._positional_embedder = positional_embedder

    query_dim = cache_line_embedder.embed_dim
    if cache_pc_embedder is not None:
      query_dim += cache_pc_embedder.embed_dim
    self._history_attention = attention.MultiQueryAttention(
        attention.GeneralAttention(query_dim, lstm_hidden_size))
    # f(h, e(l))
    self._cache_line_scorer = nn.Linear(
        lstm_hidden_size + self._positional_embedder.embed_dim, 1)

    self._reuse_distance_estimator = nn.Linear(
        lstm_hidden_size + self._positional_embedder.embed_dim, 1)

    # Needs to be capped because of limited GPU memory
    self._max_attention_history = max_attention_history

    if loss_fns is None:
      loss_fns = {"log_likelihood": LogProbLoss()}
    self._loss_fns = loss_fns

  def forward(self, cache_accesses, prev_hidden_state=None, inference=False):
    """Computes cache line to evict.

      Each cache line in the cache access is scored (higher score ==> more
      likely to evict).

    Args:
      cache_accesses (list[CacheAccess]): batch of cache accesses to
        process and whose cache lines to choose from.
      prev_hidden_state (Object | None): the result from the
        previous call to this model on the previous cache access. Use None
        only on the first cache access from a trace.
      inference (bool): set to be True at inference time, when the outputs are
        not being trained on. If True, detaches the hidden state from the graph
        to save memory.

    Returns:
      scores (torch.FloatTensor): tensor of shape
        (batch_size, len(cache_access.cache_lines)). Each entry is the
        eviction score of the corresponding cache line. The candidate
        with the highest score should be chosen for eviction.
      predicted_reuse_distances (torch.FloatTensor): tensor of shape
        (batch_size, len(cache_access.cache_lines)). Each entry is the predicted
        reuse distance of the corresponding cache line.
      hidden_state (Object): hidden state to pass to the next call of
        this function. Must be called on consecutive EvictionEntries in a trace.
      access_attention (iterable[iterable[(torch.FloatTensor, CacheAccess)]]):
        batch (outer list) of attention weights. Each inner list element is the
        attention weights of each cache line in same order as
        cache_access.cache_lines (torch.FloatTensor of shape num_cache_lines)
        on a past CacheAccess arranged from earliest to most recent.
    """
    batch_size = len(cache_accesses)
    if prev_hidden_state is None:
      hidden_state, hidden_state_history, access_history = (
          self._initial_hidden_state(batch_size))
    else:
      hidden_state, hidden_state_history, access_history = prev_hidden_state

    pc_embedding = self._pc_embedder(
        [cache_access.pc for cache_access in cache_accesses])
    address_embedding = self._address_embedder(
        [cache_access.address for cache_access in cache_accesses])

    # Each (batch_size, hidden_size)
    next_c, next_h = self._lstm_cell(
        torch.cat((pc_embedding, address_embedding), -1), hidden_state)

    if inference:
      next_c = next_c.detach()
      next_h = next_h.detach()

    # Don't modify history in place
    hidden_state_history = hidden_state_history.copy()
    hidden_state_history.append(next_h)
    access_history = access_history.copy()
    access_history.append(cache_accesses)

    # Cache lines must be padded to at least length 1 for embedding layers.
    cache_lines, mask = utils.pad(
        [cache_access.cache_lines for cache_access in cache_accesses],
        min_len=1, pad_token=(0, 0))
    cache_lines = np.array(cache_lines)
    num_cache_lines = cache_lines.shape[1]

    # Flatten into single list
    cache_pcs = itertools.chain.from_iterable(cache_lines[:, :, 1])
    cache_addresses = itertools.chain.from_iterable(cache_lines[:, :, 0])

    # (batch_size, num_cache_lines, embed_dim)
    cache_line_embeddings = self._cache_line_embedder(cache_addresses).view(
        batch_size, num_cache_lines, -1)
    if self._cache_pc_embedder is not None:
      cache_pc_embeddings = self._cache_pc_embedder(cache_pcs).view(
          batch_size, num_cache_lines, -1)
      cache_line_embeddings = torch.cat(
          (cache_line_embeddings, cache_pc_embeddings), -1)

    # (batch_size, history_len, hidden_size)
    history_tensor = torch.stack(list(hidden_state_history), dim=1)

    # (batch_size, history_len, positional_embed_size)
    positional_embeds = self._positional_embedder(
        list(range(len(hidden_state_history)))).expand(batch_size, -1, -1)

    # attention_weights: (batch_size, num_cache_lines, history_len)
    # context: (batch_size, num_cache_lines, hidden_size + pos_embed_size)
    attention_weights, context = self._history_attention(
        history_tensor, torch.cat((history_tensor, positional_embeds), -1),
        cache_line_embeddings)

    # (batch_size, num_cache_lines)
    scores = F.softmax(self._cache_line_scorer(context).squeeze(-1), -1)
    probs = utils.mask_renormalize(scores, mask)

    pred_reuse_distances = self._reuse_distance_estimator(context).squeeze(-1)
    # Return reuse distances as scores if probs aren't being trained.
    if len(self._loss_fns) == 1 and "reuse_dist" in self._loss_fns:
      probs = torch.max(
          pred_reuse_distances, torch.ones_like(
              pred_reuse_distances) * 1e-5) * mask.float()

    # Transpose access_history to be (batch_size, history_len)
    unbatched_histories = zip(*access_history)
    # Nested zip of attention and access_history
    access_attention = (
        zip(weights.transpose(0, 1), history) for weights, history in
        zip(attention_weights, unbatched_histories))

    next_hidden_state = ((next_c, next_h), hidden_state_history, access_history)
    return probs, pred_reuse_distances, next_hidden_state, access_attention

  def loss(self, eviction_traces, warmup_period):
    """Computes the losses on a sequence of consecutive eviction entries.

    The model warms up its hidden state for the first warmup_period entries.
    Then it makes predictions on the remaining entries, and returns a loss over
    the predictions.

    See constructor for which loss functions are used.

    Args:
      eviction_traces (list[list[EvictionEntry]]): batch of subsequences of
        eviction traces each of same length (batch_size, sequence_length).
      warmup_period (int): number of eviction entries to warm up hidden state
        on.

    Returns:
      loss (dict{str: torch.FloatTensor}): maps each loss name to the mean loss
        (scalar). The total loss is sum(loss.values()).
    """
    def log(score):
      """Takes log(-score), handling infs."""
      upperbound = 5.
      if score == -np.inf:
        return upperbound
      return min(np.log10(-score), upperbound)

    if warmup_period >= len(eviction_traces[0]):
      raise ValueError(
          ("Warm up period ({}) is as long as the number of provided "
           "eviction entries ({}).").format(warmup_period,
                                            len(eviction_traces[0])))

    # Warm up hidden state
    batch_size = len(eviction_traces)
    hidden_state = self._initial_hidden_state(batch_size)
    for i in range(warmup_period):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      _, _, hidden_state, _ = self(cache_accesses, hidden_state)

    # Generate predictions
    losses = collections.defaultdict(list)
    for i in range(warmup_period, len(eviction_traces[0])):
      cache_accesses = [trace[i].cache_access for trace in eviction_traces]
      scores, pred_reuse_distances, hidden_state, _ = self(
          cache_accesses, hidden_state)

      # Assumes that the lines are being labeled with Belady's.
      # Shouldn't use a loss function with use_scores, otherwise.
      log_reuse_distances = []
      for trace in eviction_traces:
        log_reuse_distances.append(
            [log(trace[i].eviction_decision.cache_line_scores[line])
             for line, _ in trace[i].cache_access.cache_lines])
      log_reuse_distances, mask = utils.pad(log_reuse_distances)
      log_reuse_distances = torch.tensor(log_reuse_distances)

      for name, loss_fn in self._loss_fns.items():
        loss = loss_fn(scores, pred_reuse_distances, log_reuse_distances, mask)
        losses[name].append(loss)
    return {name: torch.cat(loss, -1).mean() for name, loss in losses.items()}

  def _initial_hidden_state(self, batch_size):
    """Returns the initial hidden state, used when no hidden state is provided.

    Args:
      batch_size (int): batch size of hidden state to return.

    Returns:
      initial_state (tuple(torch.FloatTensor)): tuple of initial cell state and
        initial LSTM hidden state.
      hidden_state_history (collections.deque[torch.FloatTensor]): list of past
        hidden states.
      access_history (collections.deque[list[CacheAccess]]): sequences of
        batches of cache accesses.
    """
    initial_cell_state = torch.zeros(batch_size, self._lstm_cell.hidden_size)
    initial_hidden_state = torch.zeros(batch_size, self._lstm_cell.hidden_size)
    initial_hidden_state_history = collections.deque(
        [], maxlen=self._max_attention_history)
    initial_access_history = collections.deque(
        [], maxlen=self._max_attention_history)
    return ((initial_cell_state, initial_hidden_state),
            initial_hidden_state_history, initial_access_history)


class LossFunction(abc.ABC):
  """The interface for loss functions that the EvictionPolicyModel uses."""

  @abc.abstractmethod
  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    """Computes the value of the loss.

    Args:
      probs (torch.FloatTensor): probability of each evicting line of shape
        (batch_size, num_lines).
      predicted_log_reuse_distances (torch.FloatTensor): log of the model
        predicted reuse distance of each line of shape (batch_size, num_lines).
      true_log_reuse_distances (torch.FloatTensor): log of the true reuse
        distance of each line of shape (batch_size, num_lines).
      mask (torch.ByteTensor): masks out elements if the value is 0 of shape
        (batch_size, num_lines).

    Returns:
      loss (torch.FloatTensor): loss for each batch of shape (batch_size,).
    """
    raise NotImplementedError


class LogProbLoss(LossFunction):
  """LossFunction wrapper around top_1_log_likelihood."""

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del predicted_log_reuse_distances
    del true_log_reuse_distances
    del mask

    return L.top_1_log_likelihood(probs)


class KLLoss(LossFunction):
  """Loss equal to D_KL(pi^opt || pi^learned).

  pi^opt is approximated by softmax(temperature * reuse distance).
  """

  def __init__(self, temperature=1):
    super().__init__()
    self._temperature = temperature

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    approx_oracle_policy = td.Categorical(
        logits=self._temperature * true_log_reuse_distances)
    learned_policy = td.Categorical(probs=probs)
    loss = td.kl.kl_divergence(approx_oracle_policy, learned_policy)
    return loss


class ApproxNDCGLoss(LossFunction):
  """LossFunction wrapper around plackett_luce."""

  def __init__(self):
    super().__init__()
    logging.warning("Expects that all calls to loss are labeled with Belady's")

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del predicted_log_reuse_distances

    return L.approx_ndcg(probs, true_log_reuse_distances, mask=mask)


class ReuseDistanceLoss(LossFunction):
  """Computes the MSE loss between predicted and true log reuse distances."""

  def __init__(self):
    super().__init__()
    logging.warning("Expects that all calls to loss are labeled with Belady's")

  def __call__(self, probs, predicted_log_reuse_distances,
               true_log_reuse_distances, mask):
    del probs

    return F.mse_loss(
        predicted_log_reuse_distances * mask.float(),
        true_log_reuse_distances * mask.float(), reduce=False).mean(-1)
