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

"""An encoder model for the GSL layer."""
from typing import Any, Optional, Callable

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn


def make_map_node_features_layer(
    layer,
    node_set = tfgnn.NODES,
    feature_name = tfgnn.HIDDEN_STATE):
  """Copied from tfgnn/experimental/in_memory/models.py."""

  target_node_set = node_set
  target_feature = feature_name
  def _map_node_features(node_set, *, node_set_name):
    """Map feature `target_feature` of `target_node_set` but copy others."""
    if target_node_set != node_set_name:
      return node_set
    return {feat_name: layer(tensor) if feat_name == target_feature else tensor
            for feat_name, tensor in node_set.features.items()}

  return tfgnn.keras.layers.MapFeatures(node_sets_fn=_map_node_features)


@tf.keras.utils.register_keras_serializable(package="GSL")
class GCNLayer(tf.keras.layers.Layer):
  """A GCN layer as an encoder in the GSL framework."""

  def __init__(
      self,
      layer_number,
      depth,
      output_size,
      hidden_units,
      activation,
      dropout_rate,
      **kwargs,
  ):
    super().__init__()
    self._layer_number = layer_number
    self._depth = depth
    self._output_size = output_size
    self._hidden_units = (
        hidden_units if layer_number < (depth - 1) else output_size
    )
    self._activation = activation if layer_number < (depth - 1) else None
    self._dropout_rate = dropout_rate

  def build(self, input_shape):
    layers = [
        gcn.GCNHomGraphUpdate(
            units=self._hidden_units,
            receiver_tag=tfgnn.SOURCE,
            name="gcn_layer_%i" % self._layer_number,
            activation=self._activation,
            edge_weight_feature_name="weights",
            degree_normalization="in_out",
        )
    ]

    if self._layer_number < self._depth - 1:
      layers.append(
          make_map_node_features_layer(
              tf.keras.layers.Dropout(self._dropout_rate)
          )
      )
    self._model = tf.keras.Sequential(layers)

  def call(self, inputs):
    # Returning node embeddings
    return self._model(inputs).node_sets["nodes"]["hidden_state"]

  def get_config(self):
    return dict(
        layer_number=self._layer_number,
        hidden_units=self._hidden_units,
        depth=self._depth,
        output_size=self._output_size,
        activation=self._activation,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class GINModel(tf.keras.layers.Layer):
  """A GIN layer as an encoder in the GSL framework."""

  def __init__(
      self,
      layer_number,
      depth,
      output_size,
      hidden_units,
      activation,
      dropout_rate,
      **kwargs,
  ):
    super().__init__()
    self._layer_number = layer_number
    self._depth = depth
    self._output_size = output_size
    self._hidden_units = (
        hidden_units if layer_number < (depth - 1) else output_size
    )
    self._activation = activation if layer_number < (depth - 1) else None
    self._dropout_rate = dropout_rate

  def build(self, input_shape):
    self._mlps = [
        tf.keras.layers.Dense(self._hidden_units, activation=self._activation)
        for _ in range(self._depth - 1)
    ]

    self._mlps.append(tf.keras.layers.Dense(self._output_size, activation=None))
    self._epsilons = [
        tf.Variable(0.0, trainable=True, name="epsilon")
        for _ in range(self._depth)
    ]

  def call(self, inputs):
    graph = inputs
    node_set_name = "nodes"
    edge_set_name = "edges"
    feature_name = "hidden_state"
    receiver, sender = tfgnn.TARGET, tfgnn.SOURCE
    # Initialize h with node features.
    h = graph.node_sets[node_set_name][feature_name]
    for i in range(self._depth):
      source_bcast = tfgnn.broadcast_node_to_edges(
          graph, edge_set_name, sender, feature_value=h
      )
      edge_weights = tf.expand_dims(
          graph.edge_sets[edge_set_name]["weights"], -1
      )
      source_bcast = source_bcast * edge_weights
      pooled = tfgnn.pool_edges_to_node(
          graph, edge_set_name, receiver, "mean", feature_value=source_bcast
      )
      h = h * tf.keras.activations.relu(1 + self._epsilons[i]) + pooled
      if i == self._depth - 1:
        node_embeddings = h
      h = self._mlps[i](h)
    return node_embeddings, h

  def get_config(self):
    return dict(
        layer_number=self._layer_number,
        hidden_units=self._hidden_units,
        depth=self._depth,
        output_size=self._output_size,
        activation=self._activation,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class GCNModel(tf.keras.layers.Layer):
  """A GCN model as an encoder in the GSL framework."""

  def __init__(
      self,
      layer_number,
      depth,
      output_size,
      hidden_units,
      activation,
      dropout_rate,
      **kwargs,
  ):
    super().__init__()
    self._layer_number = layer_number
    self._depth = depth
    self._output_size = output_size
    self._hidden_units = (
        hidden_units if layer_number < (depth - 1) else output_size
    )
    self._activation = activation if layer_number < (depth - 1) else None
    self._dropout_rate = dropout_rate

  def build(self, input_shape):
    layers = []
    for i in range(self._depth - 1):
      layers.append(
          gcn.GCNHomGraphUpdate(
              units=self._hidden_units,
              receiver_tag=tfgnn.SOURCE,
              name="gcn_layer_%i" % i,
              activation=self._activation,
              edge_weight_feature_name="weights",
              degree_normalization="in_out",
          )
      )
      layers.append(
          make_map_node_features_layer(
              tf.keras.layers.Dropout(self._dropout_rate)
          )
      )
    self._initial_layers = tf.keras.Sequential(layers)
    self._final_layer = gcn.GCNHomGraphUpdate(
        units=self._output_size,
        receiver_tag=tfgnn.SOURCE,
        name="gcn_layer_%i" % i,
        activation=self._activation if i < self._depth - 1 else None,
        edge_weight_feature_name="weights",
        degree_normalization="in_out",
    )

  def call(self, inputs):
    # Returning node embeddings for contrastive loss
    intermediate_graph_tensor = self._initial_layers(inputs)
    node_embedddings = intermediate_graph_tensor.node_sets["nodes"][
        "hidden_state"
    ]
    predictions = self._final_layer(intermediate_graph_tensor).node_sets[
        "nodes"
    ]["hidden_state"]
    return node_embedddings, predictions

  def get_config(self):
    return dict(
        layer_number=self._layer_number,
        hidden_units=self._hidden_units,
        depth=self._depth,
        output_size=self._output_size,
        activation=self._activation,
        dropout_rate=self._dropout_rate,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class MLPModel(tf.keras.layers.Layer):
  """An MLP model as an encoder in the GSL framework."""

  def __init__(
      self,
      depth,
      output_size,
      hidden_units,
      activation,
      dropout_rate,
      **kwargs,
  ):
    super().__init__()
    self._depth = depth
    self._output_size = output_size
    self._hidden_units = hidden_units
    self._activation = activation
    self._dropout_rate = dropout_rate

  def build(self, input_shape):
    initial_layers = []
    for _ in range(self._depth - 1):
      tf.keras.layers.Dense(
          units=self._hidden_units, activation=self._activation
      )
      initial_layers.append(tf.keras.layers.Dropout(rate=self._dropout_rate))

    self._initial_layers = tf.keras.Sequential(initial_layers)
    self._final_layer = tf.keras.layers.Dense(
        units=self._output_size, activation=None
    )

  def call(self, inputs):
    # input is a graph tensor
    node_embeddings = self._initial_layers(
        inputs.node_sets["nodes"]["hidden_state"]
    )
    return node_embeddings, self._final_layer(node_embeddings)

  def get_config(self):
    return dict(
        hidden_units=self._hidden_units,
        depth=self._depth,
        output_size=self._output_size,
        activation=self._activation,
        dropout_rate=self._dropout_rate,
        **super().get_config(),
    )


def get_encoder(
    name,
    adjacency_learning_mode,
    layer_number,
    depth,
    output_size,
    **kwargs,
):
  """Returns the encoder of the gsl layer depending on the given args.

  Args:
    name: name of the encoder layer type.
    adjacency_learning_mode: whether to learn an adjacency per layer or shared.
    layer_number: the gsl layer number.
    depth: depth of the encoder.
    output_size: the output size for the encoder.
    **kwargs: rest of the args.

  Returns:
    An encoder layer or an encoder model.
  Raises:
    ValueError: if the encoder type is not defined.
  """
  if name == "gcn":
    if adjacency_learning_mode == "per_layer_adjacency_matrix":
      return GCNLayer(
          layer_number=layer_number,
          depth=depth,
          output_size=output_size,
          **kwargs,
      )
    elif adjacency_learning_mode == "shared_adjacency_matrix":
      return GCNModel(
          layer_number=layer_number,
          depth=depth,
          output_size=output_size,
          **kwargs,
      )
  elif name == "gin":
    return GINModel(
        layer_number=layer_number,
        depth=depth,
        output_size=output_size,
        **kwargs,
    )
  elif name == "mlp":
    return MLPModel(depth=depth, output_size=output_size, **kwargs)
  else:
    raise ValueError(f"Encoder {name} is not defined.")
