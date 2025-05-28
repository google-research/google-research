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

"""Abstract model class for temporal link prediction."""

import abc
import dataclasses
from typing import Sequence

import torch
from torch_geometric import data as torch_geo_data

from fm4tlp.models import model_config_pb2
from fm4tlp.modules import neighbor_loader


@dataclasses.dataclass(frozen=True)
class ModelPrediction:
  y_pred_pos: torch.Tensor = dataclasses.field(default_factory=torch.Tensor)
  y_pred_neg: torch.Tensor | None = None


class TlpModel(abc.ABC):
  """Abstract model class for temporal link prediction."""

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
  ):
    """Initializes the model."""
    self._config = model_config
    self._raw_message_size = raw_message_size
    self._total_num_nodes = total_num_nodes
    self._device = device
    self._learning_rate = learning_rate
    self._structural_feature_dim = structural_feature_dim
    self._initialize_model()
    self._structural_feature_mean = structural_feature_mean
    self._structural_feature_std = structural_feature_std

  @property
  @abc.abstractmethod
  def has_memory(self):
    return False

  @property
  @abc.abstractmethod
  def has_struct_mapper(self):
    return False

  @property
  def model_name(self):
    return self._config.model_name

  @property
  def structural_feature_mean(self):
    return self._structural_feature_mean

  @property
  def structural_feature_std(self):
    return self._structural_feature_std

  @abc.abstractmethod
  def _initialize_model(self):
    """Initializes model parameters."""

  @abc.abstractmethod
  def optimize(self, loss):
    """Optimizes the model."""

  @abc.abstractmethod
  def save_model(self, model_path):
    """Saves the model."""

  @abc.abstractmethod
  def load_model(self, model_path):
    """Loads the model."""

  @abc.abstractmethod
  def initialize_train(self):
    """Initializes the training."""

  @abc.abstractmethod
  def initialize_test(self):
    """Initializes test evaluation."""

  @abc.abstractmethod
  def initialize_batch(self, batch):
    """Initializes batch processing."""

  @abc.abstractmethod
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

  @abc.abstractmethod
  def compute_loss(
      self,
      model_prediction,
      predicted_memory_emb,
      memory_emb,
  ):
    """Computes the loss from a model prediction."""

  @abc.abstractmethod
  def get_memory_embeddings(self, nodes):
    """Gets memory embeddings for a set of nodes."""

  @abc.abstractmethod
  def predict_memory_embeddings(
      self, structral_feat
  ):
    """Predicts memory embeddings for a set of nodes from structural embeddings."""

  def initialize_memory_embedding(
      self, nodes, memory_emb
  ):
    """Initializes memory embeddings for a set of nodes."""

  @abc.abstractmethod
  def reset_memory(self):
    """For models with memory modules, resets memory parameters."""

  @abc.abstractmethod
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
    """For models with memory modules, updates memory parameters."""
