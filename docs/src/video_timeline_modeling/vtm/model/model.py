# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""The main model for video timeline modeling."""

from typing import Optional, Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from vtm.model.attention_head import AttentionHead
from vtm.model.encoder import Encoder
from vtm.model.encoder import PositionalEncoding


class TimelineModel(nn.Module):
  """Timeline model."""

  def __init__(self,
               max_num_cluster,
               max_num_video,
               num_emb,
               num_input_hidden_video,
               num_hidden,
               num_head,
               num_layers,
               video_pe=False,
               dropout=0.1,
               semantics_aware_head=False,
               semantics_aware_head_pos='pos1',
               remove_video_and_cluster_encoders=False,
               text_embedding_as_input=False,
               semantics_num_emb=256):
    super().__init__()
    self.max_num_cluster = max_num_cluster
    self.video_pe = video_pe
    self.semantics_aware_head = semantics_aware_head
    self.semantics_aware_head_pos = semantics_aware_head_pos
    self.remove_video_and_cluster_encoders = remove_video_and_cluster_encoders
    self.text_embedding_as_input = text_embedding_as_input
    self.cluster_emb = nn.Parameter(
        nn.init.xavier_uniform_(torch.empty(max_num_cluster, num_emb)))

    self.video_transform = nn.Linear(num_input_hidden_video, num_emb)
    self.cluster_video_encoder = Encoder(num_emb, num_hidden, num_head,
                                         num_layers, dropout)
    if not self.remove_video_and_cluster_encoders:
      self.cluster_encoder = Encoder(num_emb, num_hidden, num_head, num_layers,
                                     dropout)
      self.video_encoder = Encoder(num_emb, num_hidden, num_head, num_layers,
                                   dropout)
    if self.video_pe:
      self.pe_video = PositionalEncoding(num_emb, max_num_video)
    self.pe_cluster = PositionalEncoding(num_emb, max_num_cluster)
    self.attention_head = AttentionHead(num_emb)

  def forward(
      self, data_batch
  ):
    """Forward pass.

    Args:
      data_batch: input batched data, which is a dict with keys
        'video_features', 'cluster_text_features', 'video_cluster_label',
        'video_padding_mask', and 'cluster_non_padding_mask'. Each value is a
        tensor. The first dimension of each value is batch_size.
        'video_features': (batch_size, max_num_video_in_the_batch, feature_dim)
        'video_padding_mask': (batch_size, max_num_video_in_the_batch)
        'cluster_text_features': (batch_size, max_num_clusters). Note that
        max_num_clusters is 24, not the maximum number of clusters in the batch.
        'cluster_non_padding_mask': (batch_size, max_num_clusters)
        'video_cluster_label': (batch_size, max_num_video_in_the_batch)

    Returns:
      (1) The normalized attention scores (log_softmax) with shape (B,
      max_num_video_in_batch, max_num_cluster).
      (2) The intermediate cluster representations.
      (3) The intermediate video representations, if applicable. Otherwise,
      None.
    """
    batch_video_x = data_batch['video_features']
    batch_video_padding_mask = data_batch['video_padding_mask']
    video_x = self.video_transform(batch_video_x)
    if self.video_pe:
      video_x = self.pe_video(video_x)
    if self.text_embedding_as_input:
      cluster_x = self.cluster_emb + data_batch['cluster_text_features']
    else:
      cluster_x = self.cluster_emb
    # (B, max_num_cluster+max_num_video_in_batch, num_emb)
    cluster_video_x = torch.cat(
        (self.pe_cluster(cluster_x.expand(video_x.shape[0], -1, -1)), video_x),
        dim=1)
    # (B, max_num_cluster+max_num_video_in_batch, num_emb)
    cluster_video_h = self.cluster_video_encoder(
        cluster_video_x,
        torch.cat((torch.zeros(
            (video_x.shape[0], self.max_num_cluster),
            dtype=batch_video_padding_mask.dtype).to(
                batch_video_padding_mask.device), batch_video_padding_mask),
                  dim=-1))

    if self.remove_video_and_cluster_encoders:
      log_score = self.attention_head(
          cluster_video_h[:, self.max_num_cluster:, :],
          cluster_video_h[:, 0:self.max_num_cluster, :])
    else:
      # (B, max_num_cluster, num_emb)
      cluster_h = self.cluster_encoder(
          cluster_video_h[:, 0:self.max_num_cluster, :])
      # (B, max_num_video_in_batch, num_emb)
      video_h = self.video_encoder(cluster_video_h[:, self.max_num_cluster:, :],
                                   batch_video_padding_mask)
      # (B, max_num_video_in_batch, max_num_cluster)
      log_score = self.attention_head(video_h, cluster_h)
    # Semantics-aware head at pos 1 or 2
    if self.semantics_aware_head:
      if self.semantics_aware_head_pos == 'pos1':
        cluster_semantics_h = cluster_video_h[:, 0:self.max_num_cluster, :]
      elif self.semantics_aware_head_pos == 'pos2':
        cluster_semantics_h = cluster_h
      return log_score, cluster_semantics_h, None
    else:
      cluster_intermediate_h = cluster_video_h[:, 0:self.max_num_cluster, :]
      video_intermediate_h = cluster_video_h[:, self.max_num_cluster:, :]
      return log_score, cluster_intermediate_h, video_intermediate_h


class ClassifierModel(nn.Module):
  """The baseline classifier model."""

  def __init__(self,
               max_num_cluster,
               max_num_video,
               num_emb,
               num_input_hidden_video,
               num_hidden,
               num_head,
               num_layers,
               video_pe=False,
               dropout=0.1):
    super().__init__()
    self.max_num_cluster = max_num_cluster
    self.video_pe = video_pe

    self.video_transform = nn.Linear(num_input_hidden_video, num_emb)
    self.video_encoder = Encoder(num_emb, num_hidden, num_head, num_layers,
                                 dropout)
    if self.video_pe:
      self.pe_video = PositionalEncoding(num_emb, max_num_video)
    self.head = nn.Linear(num_emb, max_num_cluster)

  def forward(
      self, data_batch
  ):
    """Forward pass.

    Args:
      data_batch: input batched data, which is a dict with keys
        'video_features', 'cluster_text_features', 'video_cluster_label',
        'video_padding_mask', and 'cluster_non_padding_mask'. Each value is a
        tensor. The first dimension of each value is batch_size.
        'video_features': (batch_size, max_num_video_in_the_batch, feature_dim)
        'video_padding_mask': (batch_size, max_num_video_in_the_batch)
        'cluster_text_features': (batch_size, max_num_clusters). Note that
        max_num_clusters is 24, not the maximum number of clusters in the batch.
        'cluster_non_padding_mask': (batch_size, max_num_clusters)
        'video_cluster_label': (batch_size, max_num_video_in_the_batch) In the
        classifier model, we do not use 'cluster_text_features' and
        'cluster_non_padding_mask'.

    Returns:
      The normalized attention scores (log_softmax) with shape (B,
      max_num_video_in_batch, max_num_cluster).
    """
    batch_video_x = data_batch['video_features']
    batch_video_padding_mask = data_batch['video_padding_mask']
    video_x = self.video_transform(batch_video_x)
    if self.video_pe:
      video_x = self.pe_video(video_x)
    # (B, max_num_video_in_batch, num_emb)
    video_h = self.video_encoder(video_x, batch_video_padding_mask)
    # (B, max_num_video_in_batch, max_num_cluster)
    scores = self.head(video_h)
    return F.log_softmax(scores, dim=-1), None, None
