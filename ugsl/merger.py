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

"""Merger layer of the GSL Layer.

This step merges an input graph with a generated graph and returns a
GraphTensor as the final output.
"""
import tensorflow as tf

from ugsl import datasets


@tf.keras.utils.register_keras_serializable(package="GSL")
class Merger(tf.keras.layers.Layer):

  def __init__(self, graph_data):
    super().__init__()
    self._graph_data = graph_data

  def get_config(self):
    return dict(graph_data=self._graph_data, **super().get_config())


class WeightedSum(Merger):
  """Sums a generated adjacency with a given adjacency into a GraphTensor."""

  def __init__(
      self,
      graph_data,
      dropout_rate,
      given_adjacency_weight = 1.0,
  ):
    super().__init__(graph_data)
    self._dropout_rate = dropout_rate
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    self._given_adjacency_weight = given_adjacency_weight

  def call(self, inputs):
    graph_structure = inputs[0]
    node_embeddings = inputs[1]
    noisy_gt = self._graph_data.get_input_graph_tensor()
    given_noisy_sources = noisy_gt.edge_sets["edges"].adjacency.source
    given_noisy_targets = noisy_gt.edge_sets["edges"].adjacency.target
    noisy_sources = tf.concat(
        (graph_structure.sources, given_noisy_sources), axis=0
    )
    noisy_targets = tf.concat(
        (graph_structure.targets, given_noisy_targets), axis=0
    )
    noisy_weights = tf.concat(
        (
            graph_structure.weights,
            self._given_adjacency_weight * tf.ones(given_noisy_sources.shape),
        ),
        axis=0,
    )
    graph_tensor = self._graph_data.as_graph_tensor_given_adjacency(
        [noisy_sources, noisy_targets],
        edge_weights=self._dropout_layer(noisy_weights),
        node_features=node_embeddings,
    )
    return graph_tensor

  def get_config(self):
    return dict(
        dropout_rate=self._dropout_rate,
        given_adjacency_weight=self._given_adjacency_weight,
        **super().get_config(),
    )


class ToGraphTensor(Merger):
  """ToGraphTensor converts an adjacency in the form of rows, columns, and weights into a GraphTensor."""

  def __init__(
      self,
      graph_data,
      dropout_rate,
      **kwargs,
  ):
    super().__init__(graph_data)
    self._dropout_rate = dropout_rate
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    graph_structure = inputs[0]
    node_embeddings = inputs[1]
    graph_tensor = self._graph_data.as_graph_tensor_given_adjacency(
        [graph_structure.sources, graph_structure.targets],
        edge_weights=self._dropout_layer(graph_structure.weights),
        node_features=node_embeddings,
    )
    return graph_tensor

  def get_config(self):
    return dict(dropout_rate=self._dropout_rate, **super().get_config())


class RandomGraphTensor(Merger):
  """Generates a random graph tensor to be tested as baseline in the framework."""

  def __init__(
      self,
      graph_data,
      dropout_rate,
      **kwargs,
  ):
    super().__init__(graph_data)
    self._graph_data = graph_data
    self._dropout_rate = dropout_rate
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    input_gt = self._graph_data.get_input_graph_tensor()
    number_of_edges = input_gt.edge_sets["edges"].adjacency.source.shape[0]
    number_of_nodes = input_gt.node_sets["nodes"].features["feat"].shape[0]
    self._random_sources = tf.random.uniform(
        shape=(number_of_edges,),
        minval=0,
        maxval=number_of_nodes,
        dtype=tf.int32,
    )
    self._random_targets = tf.random.uniform(
        shape=(number_of_edges,),
        minval=0,
        maxval=number_of_nodes,
        dtype=tf.int32,
    )
    self._random_weights = tf.random.uniform(
        shape=(number_of_edges,), minval=0, maxval=1.0, dtype=tf.float32
    )

  def call(self, inputs):
    node_embeddings = inputs[1]
    graph_tensor = self._graph_data.as_graph_tensor_given_adjacency(
        tf.stack([self._random_sources, self._random_targets], axis=0),
        edge_weights=self._dropout_layer(self._random_weights),
        node_features=node_embeddings,
    )
    return graph_tensor

  def get_config(self):
    return dict(
        dropout_rate=self._dropout_rate,
        **super().get_config(),
    )


class InputGraphTensor(Merger):
  """Sums a generated adjacency with a given adjacency into a GraphTensor."""

  def __init__(
      self,
      graph_data,
      dropout_rate,
      **kwargs,
  ):
    super().__init__(graph_data)
    self._dropout_rate = dropout_rate
    self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    node_embeddings = inputs[1]
    noisy_gt = self._graph_data.get_input_graph_tensor()
    noisy_sources = noisy_gt.edge_sets["edges"].adjacency.source
    noisy_targets = noisy_gt.edge_sets["edges"].adjacency.target
    noisy_weights = tf.ones(noisy_sources.shape)

    graph_tensor = self._graph_data.as_graph_tensor_given_adjacency(
        [noisy_sources, noisy_targets],
        edge_weights=self._dropout_layer(noisy_weights),
        node_features=node_embeddings,
    )
    return graph_tensor

  def get_config(self):
    return dict(
        dropout_rate=self._dropout_rate,
        **super().get_config(),
    )


def get_merger(
    graph_data, name, **kwargs
):
  """Return the corresponding merger based on the name provided.

  Args:
    graph_data: the GSL graph data.
    name: name of the merger to use in the gsl framework.
    **kwargs:

  Returns:
    Merger associated to the provided name.
  Raises:
    ValueError: if the merger name is not defined.
  """
  if name == "none":
    return ToGraphTensor(graph_data, **kwargs)
  elif name == "weighted-sum":
    return WeightedSum(graph_data, **kwargs)
  elif name == "random":
    return RandomGraphTensor(graph_data, **kwargs)
  elif name == "input":
    return InputGraphTensor(graph_data, **kwargs)
  else:
    raise ValueError(f"Merger {name} is not defined.")
