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

"""Creating GSL model based on the given config."""

from ml_collections import config_dict
import tensorflow as tf
import tensorflow_gnn as tfgnn

from ugsl import edge_scorer
from ugsl import encoder
from ugsl import input_layer
from ugsl import merger
from ugsl import processor
from ugsl import regularizers
from ugsl import sparsifier
from ugsl import unsupervised_losses


def get_gsl_model(
    input_graph,
    cfg,
):
  """Returns a GSL model based on the given config values in cfg.

  Args:
    input_graph: dataset in the form of an input graph.
    cfg: config values for the GSL model.

  Returns:
    A GSL model based on the given configs in the form of keras sequential.

  Raises:
    ValueError: if the provided config values are not defined.
  """
  if cfg.adjacency_learning_mode == "shared_adjacency_matrix":
    gsl_layers = GSLLayer(
        input_graph=input_graph,
        layer_number=0,
        **cfg,
    )
  elif cfg.adjacency_learning_mode == "per_layer_adjacency_matrix":
    layers = []
    for i in range(cfg.depth):
      layers.append(
          GSLLayer(
              input_graph=input_graph,
              layer_number=i,
              **cfg,
          )
      )
    gsl_layers = tf.keras.Sequential(layers)
  else:
    raise ValueError(
        f"Adjacency type {cfg.adjacency_learning_mode} is not defined."
    )
  shape = input_graph.get_initial_node_features().shape
  inputs = tf.keras.layers.Input(shape=(shape[1],), batch_size=shape[0])
  split = tf.keras.Input((), dtype=tf.int32)
  gsl_outputs = gsl_layers(inputs)
  node_embeddings = gsl_outputs["embeddings"]
  predictions = gsl_outputs["predictions"]
  graph_tensor = gsl_outputs["graph_tensor"]
  split_node_embeddings = tf.gather(predictions, split)
  model = tf.keras.models.Model(
      inputs=(inputs, split), outputs=split_node_embeddings
  )
  model = regularizers.add_loss_regularizers(
      model,
      graph_tensor,
      input_graph.get_input_graph_tensor(),
      cfg.regularizer_cfg,
  )
  if cfg.unsupervised_cfg.denoising_cfg.enable:
    model = unsupervised_losses.add_denoising_loss(
        model,
        graph_tensor,
        input_graph.get_initial_node_features(),
        cfg.unsupervised_cfg,
    )
  if cfg.unsupervised_cfg.contrastive_cfg.enable:
    if "augmented_embeddings" in gsl_outputs:
      augmented_node_embeddings = gsl_outputs["augmented_embeddings"]
    else:
      raise ValueError("Output does not contain augmented embeddings.")
    model = unsupervised_losses.add_contrastive_loss(
        model,
        input_graph.get_initial_node_features(),
        node_embeddings,
        augmented_node_embeddings,
        cfg.unsupervised_cfg,
    )
  model.gt = graph_tensor
  return model


@tf.keras.utils.register_keras_serializable(package="GSL")
class GSLLayer(tf.keras.layers.Layer):
  """A GSL layer consists of different modules as defined in the config."""

  def __init__(
      self,
      input_graph,
      layer_number,
      adjacency_learning_mode,
      depth,
      edge_scorer_cfg,
      sparsifier_cfg,
      processor_cfg,
      merger_cfg,
      encoder_cfg,
      unsupervised_cfg,
      **kwargs,
  ):
    super().__init__()
    self._input_graph = input_graph
    self._layer_number = layer_number
    self._adjacency_learning_mode = adjacency_learning_mode
    self._depth = depth
    self._edge_scorer_cfg = edge_scorer_cfg
    self._sparsifier_cfg = sparsifier_cfg
    self._processor_cfg = processor_cfg
    self._merger_cfg = merger_cfg
    self._encoder_cfg = encoder_cfg
    self._unsupervised_cfg = unsupervised_cfg

  def build(self, input_shape=None):
    def node_sets_fn(node_set, node_set_name):
      del node_set_name
      return node_set["feat"]

    edge_scorer_l = edge_scorer.get_edge_scorer(
        node_features=self._input_graph.get_initial_node_features(),
        **self._edge_scorer_cfg,
    )
    sparsifier_l = sparsifier.get_sparsifier(
        self._input_graph.get_number_of_nodes(), **self._sparsifier_cfg
    )
    processor_l = processor.get_processor(**self._processor_cfg)
    merger_l = merger.get_merger(
        self._input_graph.get_graph_data(),
        **self._merger_cfg,
    )
    map_feature_l = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)
    anchor_graph_l = unsupervised_losses.AnchorGraphTensor(
        self._input_graph.get_graph_data(),
        **self._unsupervised_cfg.contrastive_cfg,
    )
    encoder_l = encoder.get_encoder(
        **self._encoder_cfg,
        adjacency_learning_mode=self._adjacency_learning_mode,
        layer_number=self._layer_number,
        depth=self._depth,
        output_size=self._input_graph.get_number_of_classes(),
    )
    inputs = node_embedding_l = tf.keras.Input(shape=input_shape)
    outputs = {}
    fully_connected_l = edge_scorer_l(node_embedding_l)
    graph_structure_l = sparsifier_l(fully_connected_l)
    graph_structure_l = processor_l(graph_structure_l)
    graph_tensor = merger_l(
        (graph_structure_l, tf.squeeze(node_embedding_l, axis=0))
    )
    graph_tensor = map_feature_l(graph_tensor)
    outputs["graph_tensor"] = graph_tensor
    outputs["embeddings"], outputs["predictions"] = encoder_l(graph_tensor)

    if self._unsupervised_cfg.contrastive_cfg.w > 0.0:
      anchor_graph_tensor = anchor_graph_l(
          (graph_structure_l, tf.squeeze(node_embedding_l, axis=0))
      )
      anchor_graph_tensor = map_feature_l(anchor_graph_tensor)
      node_embedding_augmented, _ = encoder_l(anchor_graph_tensor)
      outputs["augmented_embeddings"] = node_embedding_augmented

    self._gsl_layer = tf.keras.Model((inputs), outputs)
    print("BUILD is complete.")

  def call(self, inputs):
    inputs = tf.expand_dims(inputs, axis=0)
    return self._gsl_layer(inputs)

  def get_config(self):
    return dict(
        input_graph=self._input_graph,
        layer_number=self._layer_number,
        adjacency_learning_mode=self._adjacency_learning_mode,
        depth=self._depth,
        **super().get_config(),
    )
