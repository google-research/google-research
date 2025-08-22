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

"""EdgeScorer layer of the GSL Model.

The edge scorer contains multiple functions tried in existing models.
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="GSL")
class EdgeScorer(tf.keras.layers.Layer):
  """Wraps edge scorers to be used for graph structure learning."""

  def compute_cosine_similarities(self, node_embeddings):
    node_embeddings = tf.math.l2_normalize(node_embeddings, dim=1)
    return self.compute_dot_product_similarities(node_embeddings)

  def compute_dot_product_similarities(self, node_embeddings):
    similarities = tf.matmul(node_embeddings, node_embeddings, transpose_b=True)
    return similarities


@tf.keras.utils.register_keras_serializable(package="GSL")
class Attentive(EdgeScorer):
  """Generates a fully connected adjacency using an attentive approach.

  This edge scorer, first uses a vector to project the node features and then
  creates a fully connected graph of their similarities.
  """

  def __init__(
      self,
      initialization,
      nheads,
      seed = 133337,
      **kwargs,
  ):
    super().__init__()
    initialization_dict = {"method1": "ones", "method2": "random_uniform"}
    self._initialization = initialization_dict[initialization]
    if nheads <= 0:
      raise ValueError("Number of heads should be greater than zero.")
    self._nheads = nheads
    self._seed = seed

  def build(self, input_shape):
    node_embedding_dim = input_shape[-1]
    self._attention_vectors = []
    for _ in range(self._nheads):
      # Multiple instances of initializer should be created to give different
      # outputs.
      if self._initialization == "ones":
        cfg = {"class_name": self._initialization, "config": {}}
      else:
        cfg = {
            "class_name": self._initialization,
            "config": {"seed": self._seed},
        }
        self._seed += 1
      initializer = tf.keras.initializers.get(cfg)
      self._attention_vectors.append(
          tf.Variable(
              initial_value=initializer(shape=(node_embedding_dim,)),
              trainable=True,
          )
      )

  def compute_one_head(
      self, features, attention_vector
  ):
    node_embeddings = tf.multiply(attention_vector, features)
    return self.compute_cosine_similarities(node_embeddings)

  def call(self, inputs):
    similarities = 0
    for vector in self._attention_vectors:
      similarities += self.compute_one_head(inputs, vector)
    return similarities / self._nheads

  def get_config(self):
    return dict(
        initialization=self._initialization,
        nheads=self._nheads,
        seed=self._seed,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class FP(EdgeScorer):
  """Generates a fully connected adjacency with all values as parameters."""

  def __init__(
      self,
      node_features,
      initialization,
      **kwargs,
  ):
    super().__init__()
    initialization_dict = {"method1": "similarity", "method2": "glorot_uniform"}
    self._initialization = initialization_dict[initialization]
    self._node_features = node_features

  def build(self, input_shape=None):
    number_of_nodes = input_shape[-2]
    if self._initialization == "similarity":
      self._similarities = tf.Variable(
          initial_value=self.compute_cosine_similarities(self._node_features),
          trainable=True,
      )
    else:
      initializer = tf.keras.initializers.get(self._initialization)
      self._similarities = tf.Variable(
          initial_value=initializer(shape=(number_of_nodes, number_of_nodes)),
          trainable=True,
      )

  def call(self, inputs):
    # FP edge scorer only returns the adjacecy as is.
    return self._similarities

  def get_config(self):
    return dict(
        node_features=self._node_features,
        initialization=self._initialization,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class MLP(EdgeScorer):
  """Generates a fully connected adjacency using an MLP model.

  This edge scorer, first uses an MLP to project the node features and then
  creates a fully connected graph of their similarities.
  """

  def __init__(
      self,
      hidden_size,
      output_size,
      nlayers,
      activation,
      initialization,
      dropout_rate,
      **kwargs,
  ):
    super().__init__()
    initialization_dict = {"method1": "identity", "method2": "glorot_uniform"}
    self._initialization = initialization_dict[initialization]
    self._hidden_size = hidden_size
    self._output_size = output_size
    self._nlayers = nlayers
    self._activation = activation
    self._dropout_rate = dropout_rate

  def build(self, input_shape=None):
    layers = []
    for i in range(self._nlayers):
      layers.append(
          tf.keras.layers.Dense(
              units=self._hidden_size
              if i < (self._nlayers - 1)
              else self._output_size,
              activation=self._activation if i < (self._nlayers - 1) else None,
              kernel_initializer=tf.keras.initializers.get(
                  self._initialization
              ),
              use_bias=False,
          )
      )
      if i < (self._nlayers - 1):
        layers.append(tf.keras.layers.Dropout(rate=self._dropout_rate))
    self._model = tf.keras.Sequential(layers)

  def call(self, inputs):
    node_embeddings = self._model(inputs)
    similarities = self.compute_cosine_similarities(node_embeddings)
    return similarities

  def get_config(self):
    return dict(
        hidden_size=self._hidden_size,
        output_size=self._output_size,
        nlayers=self._nlayers,
        activation=self._activation,
        initialization=self._initialization,
        dropout_rate=self._dropout_rate,
        **super().get_config(),
    )


def get_edge_scorer(
    name, node_features, **kwargs
):
  if name == "mlp":
    return MLP(**kwargs)
  elif name == "attentive":
    return Attentive(**kwargs)
  elif name == "fp":
    return FP(node_features, **kwargs)
  else:
    raise ValueError(f"Edge scorer {name} is not defined.")
