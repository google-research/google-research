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

"""Unsupervised models to be used in the gsl framework."""
from typing import Callable

from ml_collections import config_dict
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gcn

from ugsl import datasets


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
class DenoisingModel(tf.keras.layers.Layer):
  """Creates a denosing autoencoder as propsoed in SLAPS.

  Attributes:
    node_features: initial features provided for the nodes.
  """

  def __init__(
      self,
      node_features,
      ones_ratio,
      negative_ratio,
      hidden_units,
      depth,
      dropout_rate,
      activation,
      **kwargs,
  ):
    super().__init__()
    self._node_features = node_features
    self._ones_ratio = ones_ratio
    self._negative_ratio = negative_ratio
    self._nfeatures = self._node_features.shape[1]
    self._hidden_units = hidden_units
    self._depth = depth
    self._dropout_rate = dropout_rate
    self._activation = activation

  def compute_probability_mask(self):
    nones = tf.math.count_nonzero(self._node_features).numpy()
    nzeros = self._node_features.shape[0] * self._node_features.shape[1] - nones
    probability_zeros = nones / nzeros / self._ones_ratio * self._negative_ratio
    probabilities = tf.math.scalar_mul(
        probability_zeros,
        tf.cast(tf.equal(self._node_features, 0), dtype=tf.float32),
    ) + tf.math.scalar_mul(
        1 / self._ones_ratio,
        tf.cast(tf.equal(self._node_features, 1), dtype=tf.float32),
    )
    return probabilities

  def get_random_mask(self):
    sample = tf.random.uniform(shape=self._node_features.shape, maxval=1)
    mask = tf.cast(
        tf.math.greater(self._probabilities, sample), dtype=tf.float32
    )
    return mask

  def build(self, input_shape=None):
    layers = []
    for i in range(self._depth):
      layers.append(
          gcn.GCNHomGraphUpdate(
              units=self._hidden_units
              if i < self._depth - 1
              else self._nfeatures,
              receiver_tag=tfgnn.SOURCE,
              name="gcn_layer_%i" % i,
              activation=self._activation if i < self._depth - 1 else None,
              edge_weight_feature_name="weights",
              degree_normalization="in_out",
          )
      )
      if i < self._depth - 1:
        layers.append(
            make_map_node_features_layer(
                tf.keras.layers.Dropout(self._dropout_rate)
            )
        )
    self._model = tf.keras.Sequential(layers)
    self._probabilities = self.compute_probability_mask()

  def call(self, inputs):
    random_mask = self.get_random_mask()
    masked_features = self._node_features * (1 - random_mask)
    graph = inputs.replace_features(
        {
            tfgnn.HIDDEN_STATE: tfgnn.NodeSet.from_fields(
                sizes=masked_features.shape,
                features={tfgnn.HIDDEN_STATE: masked_features},
            )
        }
    )
    predictions = self._model(graph).node_sets["nodes"]["hidden_state"][
        random_mask > 0.0
    ]
    labels = self._node_features[random_mask > 0.0]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=predictions
        )
    )

  def get_config(self):
    return dict(
        node_features=self._node_features,
        ones_ratio=self._ones_ratio,
        negative_ratio=self._negative_ratio,
        hidden_units=self._hidden_units,
        depth=self._depth,
        dropout_rate=self._dropout_rate,
        activation=self._activation,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class ContrastiveModel(tf.keras.layers.Layer):
  """Returns contrastive loss from GSL-Paper among 2 node embedding tensors.

  This function applies a softmax cross entropy loss based on Liu et al.: [
  Towards Unsupervised Deep Graph Structure Learning]
  (https://arxiv.org/abs/2201.06367), 2022. Utilizing the cosine similarity of
  the embeddings for each node(mapping to the first dimension of each input
  tensor), it applies a log softmax loss function, pushing apart different
  nodes in the embedding space.

  Attributes:
    node_embeddings: <float>[node_count, embedding_dim]
    augmented_node_embeddings: <float>[node_count, embedding_dim]
    temperature: Parameter to scale similarities.
  """

  def __init__(
      self,
      node_features,
      temperature,
      **kwargs,
  ):
    super().__init__()
    self._node_features = node_features
    self._temperature = temperature

  def _get_negative_log_softmax(
      self, node_similarities, axis
  ):
    """Returns reduced negative log softmax of the given matrix.


    Given node_similarities, positive node similarities are in the diagonal of
    the given matrix.

    Args:
      node_similarities: <float>[node_count,]
      axis: Reduce dimension to find sum of similarities.
    """
    # Get positive(same node) similarities.
    # <float>(node_count,)
    positive_similarities = tf.linalg.tensor_diag_part(node_similarities)
    # <float>(node_count,)
    node_similarities_sum = tf.math.reduce_sum(node_similarities, axis=axis)
    # <float>(node_count,)
    log_softmax = -tf.math.log(
        tf.math.divide_no_nan(
            positive_similarities,
            node_similarities_sum - positive_similarities,
        )
    )
    return tf.math.reduce_mean(log_softmax, axis=0)

  def _get_exp_cosine_similarities(
      self,
      node_embeddings,
      augmented_node_embeddings,
      temperature,
  ):
    """Returns exponent of normalized dot product between given node embeddings.

    Args:
      node_embeddings: <float>[node_count, embedding_dim]
      augmented_node_embeddings: <float>[node_count, embedding_dim]
      temperature: Parameter to scale similarities.
    """
    # L2 normalize embeddings.
    node_embeddings_normalized = tf.linalg.l2_normalize(
        node_embeddings, axis=-1
    )
    augmented_node_embeddings_normalized = tf.linalg.l2_normalize(
        augmented_node_embeddings, axis=-1
    )
    # Find cosine similarities across node-sets.
    # <float>[node_count, node_count]
    normalized_node_similarities = tf.linalg.matmul(
        node_embeddings_normalized,
        augmented_node_embeddings_normalized,
        transpose_b=True,
    )
    normalized_node_similarities = tf.math.divide_no_nan(
        normalized_node_similarities, temperature
    )
    normalized_node_similarities = tf.math.exp(normalized_node_similarities)
    return normalized_node_similarities

  def call(self, inputs):
    node_embeddings, augmented_node_embeddings = inputs
    validation_ops = [
        tf.debugging.assert_equal(
            tf.shape(node_embeddings),
            tf.shape(augmented_node_embeddings),
            message=(
                f"Tensor shapes for {node_embeddings} and "
                f"{augmented_node_embeddings} has to match."
            ),
        ),
        tf.debugging.assert_equal(
            tf.rank(node_embeddings),
            2,
            message="Input tensors need to have 2 dimensions.",
        ),
    ]
    with tf.control_dependencies(validation_ops):
      normalized_node_similarities = self._get_exp_cosine_similarities(
          node_embeddings, augmented_node_embeddings, self._temperature
      )
      # Calculate softmax log loss for both node embeddings.
      regular_to_annotated_log_softmax = self._get_negative_log_softmax(
          normalized_node_similarities, axis=1
      )
      annotated_to_regular_log_softmax = self._get_negative_log_softmax(
          normalized_node_similarities, axis=0
      )
      return (
          regular_to_annotated_log_softmax + annotated_to_regular_log_softmax
      ) / 2

  def get_config(self):
    return dict(
        node_features=self._node_features,
        temperature=self._temperature,
        **super().get_config(),
    )


class AnchorGraphTensor(tf.keras.layers.Layer):
  """Creates an anchor graph tensor for the contrastive loss."""

  def __init__(
      self,
      graph_data,
      feature_mask_rate,
      tau,
      **kwargs,
  ):
    super().__init__()
    self._graph_data = graph_data
    self._feature_mask_rate = feature_mask_rate
    self._tau = tau

  def call(self, inputs):
    graph_structure = inputs[0]
    node_embeddings = inputs[1]

    number_of_nodes, number_of_features = node_embeddings.shape
    probabilities = tf.repeat(self._feature_mask_rate, number_of_features)
    sample = tf.random.uniform(shape=(1, number_of_features), maxval=1.0)
    mask = tf.cast(tf.math.greater(probabilities, sample), dtype=tf.float32)
    mask = tf.repeat(mask, number_of_nodes, axis=0)
    anchor_node_features = node_embeddings * (1 - mask)
    sources = tf.range(number_of_nodes, dtype=tf.int32)
    targets = tf.range(number_of_nodes, dtype=tf.int32)
    edge_weights = tf.ones([number_of_nodes], dtype=tf.float32)
    if self._tau > 0.0:
      sources = tf.concat([sources, graph_structure.sources], axis=0)
      targets = tf.concat([targets, graph_structure.targets], axis=0)
      edge_weights = tf.concat(
          [edge_weights, self._tau * graph_structure.weights], axis=0
      )
    return self._graph_data.as_graph_tensor_given_adjacency(
        tf.concat(
            [tf.expand_dims(sources, 0), tf.expand_dims(targets, 0)], axis=0
        ),
        edge_weights=edge_weights,
        node_features=anchor_node_features,
    )

  def get_config(self):
    return dict(
        feature_mask_rate=self._feature_mask_rate,
        tau=self._tau,
        graph_data=self._graph_data,
        **super().get_config(),
    )


def add_denoising_loss(
    model,
    model_graph,
    node_features,
    cfg,
):
  denoising_model = DenoisingModel(node_features, **cfg.denoising_cfg)
  model.add_loss(cfg.denoising_cfg.w * denoising_model(model_graph))
  return model


def add_contrastive_loss(
    model,
    node_features,
    node_embeddings,
    augmented_node_embeddings,
    cfg,
):
  """Adding the contrastive loss to the keras model.

  Args:
    model: the keras model to add the loss for.
    node_features: the tensor of node features in the graph.
    node_embeddings: the node embeddings from the encoder.
    augmented_node_embeddings: the augmented node embeddings from the encoder.
    cfg: the config to pass the arguments.

  Returns:
    The model with the contrastive loss added.
  """
  contrastive_model = ContrastiveModel(
      node_features=node_features, **cfg.contrastive_cfg
  )
  model.add_loss(
      cfg.contrastive_cfg.w
      * contrastive_model((node_embeddings, augmented_node_embeddings))
  )
  return model
