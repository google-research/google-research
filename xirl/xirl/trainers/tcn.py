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

"""TCN trainer."""

from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class TCNTrainer(Trainer):
  """A trainer that implements a single-view Time Contrastive Network [1].

  Should be used in conjunction with the WindowSampler frame sampler.

  References:
    [1]: https://arxiv.org/abs/1704.06888
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.temperature = config.loss.tcn.temperature
    self.num_pairs = config.loss.tcn.num_pairs
    self.pos_radius = config.loss.tcn.pos_radius
    self.neg_radius = config.loss.tcn.neg_radius

  def compute_loss(
      self,
      embs,
      batch,
  ):
    del batch

    batch_size, num_cc_frames, _ = embs.shape

    # A positive is centered at the current index and can extend pos_radius
    # forward or backward.
    # A negative can be sampled anywhere outside a negative radius centered
    # around the current index.
    batch_pos = []
    batch_neg = []
    idxs = np.arange(num_cc_frames)
    for i in range(batch_size):
      pos_delta = np.random.choice(
          [-self.pos_radius, self.pos_radius],
          size=(num_cc_frames, self.num_pairs),
      )
      pos_idxs = np.clip(idxs[:, None] + pos_delta, 0, num_cc_frames - 1)
      batch_pos.append(torch.LongTensor(pos_idxs))

      negatives = []
      for idx in idxs:
        allowed = (idxs > (idx + self.neg_radius)) | (
            idxs < (idx - self.neg_radius))
        neg_idxs = np.random.choice(idxs[allowed], size=self.num_pairs)
        negatives.append(neg_idxs)
      batch_neg.append(torch.LongTensor(np.vstack(negatives)))

    pos_losses = 0.0
    neg_losses = 0.0
    for i, (positives, negatives) in enumerate(zip(batch_pos, batch_neg)):
      row_idx = torch.arange(num_cc_frames).unsqueeze(1)

      # Compute pairwise squared L2 distances between the embeddings in a
      # sequence.
      emb_seq = embs[i]
      distances = torch.cdist(emb_seq, emb_seq).pow(2)
      distances = distances / self.temperature

      # For every embedding in the sequence, we need to minimize its
      # distance to every positive we sampled for it.
      pos_loss = distances[row_idx, positives]
      pos_losses += pos_loss.sum()

      # And for negatives, we need to ensure they are at least a distance
      # M apart. We use the squared hinge loss to express this constraint.
      neg_margin = 1 - distances[row_idx, negatives]
      neg_loss = torch.clamp(neg_margin, min=0).pow(2)
      neg_losses += neg_loss.sum()

    total_loss = (pos_losses + neg_losses) / (batch_size * num_cc_frames)
    return total_loss


class TCNCrossEntropyTrainer(Trainer):
  """Single-view TCN implemented with contrastive cross-entropy loss.

  Should be used in conjunction with the `UniformWithPositivesSampler` frame
  sampler.
  """

  def __init__(
      self,
      model,
      optimizer,
      device,
      config,
  ):
    super().__init__(model, optimizer, device, config)

    self.temperature = config.loss.tcn.temperature

  def compute_loss(
      self,
      embs,
      batch,
  ):
    del batch

    batch_size, num_cc_frames, _ = embs.shape
    pos_embs, curr_embs = torch.chunk(embs, 2, 1)
    loss = []
    for i in range(batch_size):
      curr_pos_sim = -1.0 * torch.cdist(pos_embs[i], curr_embs[i]).pow(2)
      curr_pos_sim = curr_pos_sim / self.temperature
      pos_labels = torch.arange(num_cc_frames // 2).to(embs.device)
      loss.append(F.cross_entropy(curr_pos_sim, pos_labels, reduction='none'))
    return torch.cat(loss, dim=0).mean()
