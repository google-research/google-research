# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
from typing import Optional

import torch
from torch_geometric import data as torch_geo_data

from models import model_config_pb2
from modules import neighbor_loader


@dataclasses.dataclass(frozen=True)
class ModelPrediction:
  y_pred_pos: torch.Tensor = dataclasses.field(default_factory=torch.Tensor)
  y_pred_neg: Optional[torch.Tensor] = None


class TlpModel(abc.ABC):
  """Abstract model class for temporal link prediction."""

  def __init__(
      self,
      model_config,
      total_num_nodes,
      raw_message_size,
      device,
      structural_feature_dim = 0,
      structural_feature_mean = list(),
      structural_feature_std = list(),
  ):
    """Initializes the model."""
    self._config = model_config
    self._raw_message_size = raw_message_size
    self._total_num_nodes = total_num_nodes
    self._device = device
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
    pass

  @abc.abstractmethod
  def optimize(self, loss):
    """Optimizes the model."""
    pass

  @abc.abstractmethod
  def save_model(self, model_path):
    """Saves the model."""
    pass

  @abc.abstractmethod
  def load_model(self, model_path):
    """Loads the model."""
    pass

  @abc.abstractmethod
  def initialize_train(self):
    """Initializes the training."""
    pass

  @abc.abstractmethod
  def initialize_test(self):
    """Initializes test evaluation."""
    pass

  @abc.abstractmethod
  def initialize_batch(self, batch):
    """Initializes batch processing."""
    pass

  @abc.abstractmethod
  def predict_on_edges(
      self,
      source_nodes,
      target_nodes_pos,
      data,
      last_neighbor_loader,
      target_nodes_neg = None,
  ):
    """Generates predictions from input edges.

    Args:
      source_nodes: Source nodes.
      target_nodes_pos: Target nodes for positive edges.
      data: The torch geo temporal dataset object.
      last_neighbor_loader: Object to load recent node neighbors.
      target_nodes_neg: Target nodes for negative edges.

    Returns:
      The model prediction. y_pred_neg is None if target_nodes_neg is None.
    """
    pass

  @abc.abstractmethod
  def compute_loss(
      self,
      model_prediction,
      predicted_memory_emb,
      memory_emb,
  ):
    """Computes the loss from a model prediction."""
    pass

  @abc.abstractmethod
  def get_memory_embeddings(self, nodes):
    """Gets memory embeddings for a set of nodes."""
    pass

  @abc.abstractmethod
  def predict_memory_embeddings(self, structral_feat):
    """Predicts memory embeddings for a set of nodes from structural embeddings."""
    pass

  def initialize_memory_embedding(self, nodes, memory_emb):
    """Initializes memory embeddings for a set of nodes."""
    pass

  @abc.abstractmethod
  def reset_memory(self):
    """For models with memory modules, resets memory parameters."""
    pass

  @abc.abstractmethod
  def update_memory(
      self,
      source_nodes,
      target_nodes,
      timestamps,
      messages,
  ):
    """For models with memory modules, updates memory parameters."""
    pass
