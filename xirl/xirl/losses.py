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

"""Loss functions."""

from typing import Tuple

import torch
import torch.nn.functional as F

TensorType = torch.Tensor


def one_hot(y, K, smooth_eps = 0):  # pylint: disable=invalid-name
  """One-hot encodes a tensor with optional label smoothing.

  Args:
    y: A tensor containing the ground-truth labels of shape (N,), i.e. one label
      for each element in the batch.
    K: The number of classes.
    smooth_eps: Label smoothing factor in [0, 1] range.

  Returns:
    A one-hot encoded tensor.
  """
  assert 0 <= smooth_eps <= 1
  assert y.ndim == 1, "Label tensor must be rank 1."
  y_hot = torch.eye(K)[y] * (1 - smooth_eps) + (smooth_eps / (K - 1))
  return y_hot.to(y.device)


def cross_entropy(
    logits,
    labels,
    smooth_eps = 0,
    reduction = "mean",
):
  """Cross-entropy loss with support for label smoothing.

  Args:
    logits: A `FloatTensor` containing the raw logits, i.e. no softmax has been
      applied to the model output. The tensor should be of shape `(N, K)` where
      K is the number of classes.
    labels: A rank-1 `LongTensor` containing the ground truth labels.
    smooth_eps: The label smoothing factor in [0, 1] range.
    reduction: The reduction strategy on the final loss tensor.

  Returns:
    If reduction is `none`, a 2D tensor.
    If reduction is `sum`, a 1D tensor.
    If reduction is `mean`, a scalar 1D tensor.
  """
  assert isinstance(logits, (torch.FloatTensor, torch.cuda.FloatTensor))
  assert isinstance(labels, (torch.LongTensor, torch.cuda.LongTensor))
  assert reduction in [
      "none",
      "mean",
      "sum",
  ], "reduction method is not supported"

  # Ensure logits are not 1-hot encoded.
  assert labels.ndim == 1, "[!] Labels are NOT expected to be 1-hot encoded."

  if smooth_eps == 0:
    return F.cross_entropy(logits, labels, reduction=reduction)

  # One-hot encode targets.
  labels = one_hot(labels, logits.shape[1], smooth_eps)

  # Convert logits to log probabilities.
  log_probs = F.log_softmax(logits, dim=-1)

  loss = (-labels * log_probs).sum(dim=-1)

  if reduction == "none":
    return loss
  elif reduction == "mean":
    return loss.mean()
  return loss.sum(dim=-1)  # reduction == "sum"


def huber_loss(
    input,  # pylint: disable=redefined-builtin
    target,
    delta,
    reduction = "mean",
):
  """Huber loss with tunable margin [1].

  This is a more general version of PyTorch's
  `torch.nn.functional.smooth_l1_loss` that allows the user to change the
  margin parameter.

  Args:
    input: A `FloatTensor` representing the model output.
    target: A `FloatTensor` representing the target values.
    delta: Given the tensor difference `diff`, delta is the value at which we
      incur a quadratic penalty if `diff` is at least delta and a linear penalty
      otherwise.
    reduction: The reduction strategy on the final loss tensor.

  Returns:
    If reduction is `none`, a 2D tensor.
    If reduction is `sum`, a 1D tensor.
    If reduction is `mean`, a scalar 1D tensor.

  References:
    [1]: Wikipedia Huber Loss,
    https://en.wikipedia.org/wiki/Huber_loss
  """
  assert isinstance(input, (torch.FloatTensor, torch.cuda.FloatTensor))
  assert isinstance(target, (torch.FloatTensor, torch.cuda.FloatTensor))
  assert reduction in [
      "none",
      "mean",
      "sum",
  ], "reduction method is not supported"

  diff = target - input
  diff_abs = torch.abs(diff)
  cond = diff_abs <= delta
  loss = torch.where(cond, 0.5 * diff**2, (delta * diff_abs) - (0.5 * delta**2))
  if reduction == "none":
    return loss
  elif reduction == "mean":
    return loss.mean()
  return loss.sum(dim=-1)  # reduction == "sum"


def compute_tcc_loss(
    embs,
    idxs,
    seq_lens,
    stochastic_matching = False,
    normalize_embeddings = False,
    loss_type = "classification",
    similarity_type = "l2",
    num_cycles = 20,
    cycle_length = 2,
    temperature = 0.1,
    label_smoothing = 0.1,
    variance_lambda = 0.001,
    huber_delta = 0.1,
    normalize_indices = True,
):
  """Computes TCC loss between sequences of embeddings."""
  msg = "Invalid similarity type."
  assert similarity_type in ["l2", "cosine"], msg
  msg = "Invalid loss type."
  assert loss_type in [
      "regression_mse_var",
      "regression_mse",
      "regression_huber",
      "classification",
  ], msg

  batch_size, num_cc = embs.shape[:2]

  if stochastic_matching:
    return stochastic_tcc_loss(
        embs=embs,
        idxs=idxs,
        seq_lens=seq_lens,
        num_cc=num_cc,
        batch_size=batch_size,
        loss_type=loss_type,
        similarity_type=similarity_type,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        temperature=temperature,
        label_smoothing=label_smoothing,
        variance_lambda=variance_lambda,
        huber_delta=huber_delta,
        normalize_indices=normalize_indices,
        normalize_dimension=(not normalize_embeddings),
    )

  return deterministic_tcc_loss(
      embs=embs,
      idxs=idxs,
      seq_lens=seq_lens,
      num_cc=num_cc,
      batch_size=batch_size,
      loss_type=loss_type,
      similarity_type=similarity_type,
      temperature=temperature,
      label_smoothing=label_smoothing,
      variance_lambda=variance_lambda,
      huber_delta=huber_delta,
      normalize_indices=normalize_indices,
      normalize_dimension=(not normalize_embeddings),
  )


def deterministic_tcc_loss(
    embs,
    idxs,
    seq_lens,
    num_cc,
    batch_size,
    loss_type,
    similarity_type,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
    normalize_dimension,
):
  """Deterministic alignment between all pairs of sequences in a batch."""

  batch_size = embs.shape[0]

  labels_list = []
  logits_list = []
  steps_list = []
  seq_lens_list = []

  for i in range(batch_size):
    for j in range(batch_size):
      if i != j:
        logits, labels = align_sequence_pair(
            embs[i],
            embs[j],
            similarity_type,
            temperature,
            normalize_dimension,
        )
        logits_list.append(logits)
        labels_list.append(labels)
        steps_list.append(idxs[i:i + 1].expand(num_cc, -1))
        seq_lens_list.append(seq_lens[i:i + 1].expand(num_cc))

  logits = torch.cat(logits_list, dim=0)
  labels = torch.cat(labels_list, dim=0)
  steps = torch.cat(steps_list, dim=0)
  seq_lens = torch.cat(seq_lens_list, dim=0)

  if loss_type == "classification":
    return classification_loss(logits, labels, label_smoothing)
  return regression_loss(
      logits,
      labels,
      steps,
      seq_lens,
      loss_type,
      normalize_indices,
      variance_lambda,
      huber_delta,
  )


def pairwise_l2_sq(
    x1,
    x2,
):
  """Compute pairwise squared Euclidean distances."""
  return torch.cdist(x1, x2).pow(2)


def get_scaled_similarity(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
  """Return pairwise similarity."""
  if similarity_type == "l2":
    similarity = -1.0 * pairwise_l2_sq(emb1, emb2)
    if normalize_dimension:
      similarity = similarity / emb1.shape[1]
  else:  # Cosine similarity.
    similarity = torch.mm(emb1, emb2.t())
  similarity = similarity / temperature
  return similarity


def align_sequence_pair(
    emb1,
    emb2,
    similarity_type,
    temperature,
    normalize_dimension,
):
  """Align a pair of sequences."""
  max_num_steps = emb1.shape[0]
  sim_12 = get_scaled_similarity(emb1, emb2, similarity_type, temperature,
                                 normalize_dimension)
  softmaxed_sim_12 = F.softmax(sim_12, dim=1)  # Row-wise softmax.
  nn_embs = torch.mm(softmaxed_sim_12, emb2)
  sim_21 = get_scaled_similarity(nn_embs, emb1, similarity_type, temperature,
                                 normalize_dimension)
  logits = sim_21
  labels = torch.arange(max_num_steps).to(logits.device)
  return logits, labels


def stochastic_tcc_loss(
    embs,
    idxs,
    seq_lens,
    num_cc,
    batch_size,
    loss_type,
    similarity_type,
    num_cycles,
    cycle_length,
    temperature,
    label_smoothing,
    variance_lambda,
    huber_delta,
    normalize_indices,
    normalize_dimension,
):
  """Stochastic alignment between randomly sampled cycles."""
  cycles = gen_cycles(num_cycles, batch_size, cycle_length)
  cycles = cycles.to(embs.device)

  logits, labels = align_find_cycles(
      cycles,
      embs,
      num_cc,
      num_cycles,
      cycle_length,
      similarity_type,
      temperature,
      normalize_dimension,
  )

  if loss_type == "classification":
    return classification_loss(logits, labels, label_smoothing)

  idxs = torch.index_select(idxs, 0, cycles[:, 0])
  seq_lens = torch.index_select(seq_lens, 0, cycles[:, 0])

  return regression_loss(
      logits,
      labels,
      idxs,
      seq_lens,
      loss_type,
      normalize_indices,
      variance_lambda,
      huber_delta,
  )


def gen_cycles(
    num_cycles,
    batch_size,
    cycle_length,
):
  """Generates cycles for alignment."""
  idxs = torch.arange(batch_size).unsqueeze(0).repeat(num_cycles, 1)
  rand_idxs = torch.rand(num_cycles, batch_size).argsort(dim=1)
  cycles = torch.gather(idxs, 1, rand_idxs)
  cycles = cycles[:, :cycle_length]
  cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)
  return cycles


def align_find_cycles(
    cycles,
    embs,
    num_cc,
    num_cycles,
    cycle_length,
    similarity_type,
    temperature,
    normalize_dimension,
):
  """Align cycles."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = align_single_cycle(
        cycles[i],
        embs,
        cycle_length,
        num_cc,
        similarity_type,
        temperature,
        normalize_dimension,
    )
    logits_list.append(logits)
    labels_list.append(labels)
  logits = torch.stack(logits_list)
  labels = torch.cat(labels_list)
  return logits, labels


def align_single_cycle(
    cycle,
    embs,
    cycle_length,
    num_steps,
    similarity_type,
    temperature,
    normalize_dimension,
):
  """Take a single cycle and returns logits and labels."""
  # Choose a random frame index and use it as the label.
  n_idx = torch.randint(num_steps, size=(1,))
  labels = n_idx.to(embs.device)

  # Query the features of the randomly sampled frame from the first video
  # sequence in the cycle.
  query_feats = embs[cycle[0], n_idx:n_idx + 1]

  num_channels = query_feats.shape[-1]
  for c in range(1, cycle_length + 1):
    candidate_feats = embs[cycle[c]]
    if similarity_type == "l2":
      # Note: Order matters here for correctly obtaining a tensor of shape
      # (num_steps, 1) and not (1, num_steps). This is because we are
      # replying on broadcasting since candidate_feats is of shape
      # (num_steps, D) and query_feats is of shape (1, D).
      similarity = -1.0 * pairwise_l2_sq(candidate_feats, query_feats)
      if normalize_dimension:
        similarity = similarity / num_channels
    else:  # Cosine similarity.
      # Again, the order matters here.
      similarity = torch.mm(candidate_feats, query_feats.t())

    similarity = similarity / temperature

    if c == cycle_length:
      break

    # Find weighted nearest neighbors.
    beta = F.softmax(similarity, dim=0)
    query_feats = (beta * candidate_feats).sum(dim=0, keepdim=True)

  similarity = similarity.squeeze()
  return similarity, labels


def classification_loss(
    logits,
    labels,
    label_smoothing,
):
  """Cycle-back classification loss."""
  return cross_entropy(logits, labels, label_smoothing, reduction="mean")


def regression_loss(
    logits,
    labels,
    steps,
    seq_lens,
    loss_type,
    normalize_indices,
    variance_lambda,
    huber_delta,
):
  """Cycle-back regression loss."""
  if normalize_indices:
    steps = steps.float() / seq_lens[:, None].float()
  else:
    steps = steps.float()
  labels = one_hot(labels, logits.shape[1])
  beta = F.softmax(logits, dim=1)
  time_true = (steps * labels).sum(dim=1)
  time_pred = (steps * beta).sum(dim=1)
  if loss_type in ["regression_mse", "regression_mse_var"]:
    if "var" in loss_type:  # Variance-aware regression.
      # Compute log of prediction variance.
      time_pred_var = (steps - time_pred.unsqueeze(1)).pow(2) * beta
      time_pred_var = torch.log(time_pred_var.sum(dim=1))
      err_sq = (time_true - time_pred).pow(2)
      loss = (
          torch.exp(-time_pred_var) * err_sq + variance_lambda * time_pred_var)
      return loss.mean()
    return F.mse_loss(time_pred, time_true)
  return huber_loss(time_pred, time_true, huber_delta)
