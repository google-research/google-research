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

"""TGN (Temporal Graph Network) model."""

import json
from typing import Sequence

import tensorflow.compat.v1 as tf
import torch
from torch_geometric import data as torch_geo_data

from fm4tlp.models import model_config_pb2
from fm4tlp.models import model_template
from fm4tlp.modules import neighbor_loader


_EDGE_TYPE = tuple[torch.Tensor, torch.Tensor]


def _convert_to_int_edge(edge):
  return (int(edge[0]), int(edge[1]))


class EdgeBank(model_template.TlpModel):
  """EdgeBank baseline model."""

  def __init__(
      self,
      *,
      model_config,
      total_num_nodes,
      raw_message_size,
      device,
      learning_rate,
      structural_feature_dim = 0,
      structural_feature_mean = (),
      structural_feature_std = (),
  ):  # pylint: disable=useless-super-delegation
    """Initializes the model."""
    super().__init__(
        model_config=model_config,
        total_num_nodes=total_num_nodes,
        raw_message_size=raw_message_size,
        device=device,
        learning_rate=learning_rate,
        structural_feature_dim=structural_feature_dim,
        structural_feature_mean=structural_feature_mean,
        structural_feature_std=structural_feature_std,
    )

  def save_model(self, model_path):
    with tf.io.gfile.GFile(model_path, 'w') as f:
      f.write(json.dumps(list(self._memory)))

  def load_model(self, model_path):
    with tf.io.gfile.GFile(model_path, 'r') as f:
      self._memory = set([(u, v) for u, v in json.loads(f.read())])

  def _initialize_model(self):
    self._memory: set[tuple[int, int]] = set()
    self._criterion = torch.nn.BCEWithLogitsLoss()

  def optimize(self, loss):
    pass

  def _add_to_memory(self, edge):
    self._memory.add(_convert_to_int_edge(edge))

  def _in_memory(self, edge):
    return _convert_to_int_edge(edge) in self._memory

  def update_memory(
      self,
      *,
      source_nodes,
      target_nodes_pos,
      target_nodes_neg,
      timestamps,
      messages,
      last_neighbor_loader,
      data,
  ):
    del (
        target_nodes_neg,
        timestamps,
        messages,
        last_neighbor_loader,
        data,
    )  # Unused.
    for src, dst in zip(source_nodes, target_nodes_pos):
      self._add_to_memory((src, dst))

  def reset_memory(self):
    self._memory.clear()

  @property
  def has_memory(self):
    return True

  def initialize_train(self):
    pass

  def initialize_test(self):
    """Initializes test evaluation."""

  def initialize_batch(self, batch):
    """Initializes batch processing."""

  def compute_loss(
      self,
      model_prediction,
      predicted_memory_emb,
      original_memory_emb,
  ):
    """Computes the loss from a model prediction."""
    model_loss = self._criterion(
        model_prediction.y_pred_pos,
        torch.ones_like(model_prediction.y_pred_pos),
    )
    if model_prediction.y_pred_neg is not None:
      model_loss += self._criterion(
          model_prediction.y_pred_neg,
          torch.zeros_like(model_prediction.y_pred_neg),
      )
    structmap_loss = torch.zeros(1)
    return model_loss, structmap_loss

  def get_memory_embeddings(self, nodes):
    """Gets memory embeddings for a set of nodes."""
    raise NotImplementedError('EdgeBank does not have memory embeddings.')

  def predict_on_edges(
      self,
      *,
      source_nodes,
      target_nodes_pos,
      target_nodes_neg = None,
      last_neighbor_loader,
      data,
  ):
    """Generates predictions from input edges.

    Args:
      source_nodes: Source nodes.
      target_nodes_pos: Target nodes for positive edges.
      target_nodes_neg: Target nodes for negative edges.
      last_neighbor_loader: Object to load recent node neighbors.
      data: The torch geo temporal dataset object.

    Returns:
      The model prediction. y_pred_neg is None if target_nodes_neg is None.
    """
    del data, last_neighbor_loader  # Unused.

    pos_pred = torch.zeros([len(source_nodes), 1])
    neg_pred = torch.zeros([len(source_nodes), 1])
    target_nodes_neg_local = torch.zeros_like(target_nodes_pos)
    if target_nodes_neg is not None:
      target_nodes_neg_local = target_nodes_neg

    idx = 0
    for src, pos_dst, neg_dst in zip(
        source_nodes, target_nodes_pos, target_nodes_neg_local
    ):
      if self._in_memory((src, pos_dst)):
        pos_pred[idx, 0] = 1.0
      if target_nodes_neg is not None and self._in_memory((src, neg_dst)):
        neg_pred[idx, 0] = 1.0
      idx += 1

    return model_template.ModelPrediction(
        y_pred_pos=pos_pred,
        y_pred_neg=neg_pred if target_nodes_neg is not None else None,
    )

  def predict_memory_embeddings(
      self, struct_feat
  ):
    """Predicts memory embeddings for a set of nodes from structural embeddings."""
    raise NotImplementedError('EdgeBank does not have memory embeddings.')

  @property
  def has_struct_mapper(self):
    return False
