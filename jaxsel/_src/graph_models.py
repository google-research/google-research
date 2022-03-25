# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Models on graphs.

These models take a graph as input, and output per-node or per-edge features.
"""

import abc
from typing import Any, Callable, List, Optional, Tuple

import flax
from flax import struct
import flax.linen as nn

import jax
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

from jaxsel._src import graph_api

################
# Graph models #
################


class GraphModel(abc.ABC):
  """Abstract class for all graph models.

  Graph models take a batch of problem specific features (node, task, edges)
  as input.
  Their output is task specific, e.g. usually some feature vector per node,
  which may be aggregated later, possibly class logits.
  """

  @abc.abstractmethod
  def __call__(self, node_features, adjacency_mat,
               qstar):
    """Performs a forward pass on the model.

    Args:
      node_features: features associated to the nodes on the extracted subgraph.
      adjacency_mat: Extracted adjacency matrix.
      qstar: Optimal weights on the nodes, given by our subgraph extraction
        scheme. If not using subgraph extraction, `qstar` should be a vector of
        ones.

    Returns:
      Output of the model, e.g. logprobs for a classification task...
    """
    Ellipsis


###########################
# Flax based Graph Models #
###########################

# Transformer models are adapted from
# https://github.com/google/flax/blob/main/examples/wmt/models.py


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  graph_parameters: graph_api.GraphParameters
  hidden_dim: int  # Used to standardize node feature and position embeddings.
  num_classes: int
  image_size: int
  dtype: Any = jnp.float32
  embedding_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  # Initializers take in (key, shape, dtype) and return arrays.
  kernel_init: Callable[[Any, Any, Any],
                        jnp.array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[Any, Any, Any],
                      jnp.array] = nn.initializers.normal(stddev=1e-6)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)
    x = nn.Dense(
        cfg.mlp_dim,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init)(
            inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init)(
            x)
    output = nn.Dropout(rate=cfg.dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               inputs,
               encoder_mask = None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 2
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=cfg.attention_dropout_rate,
        deterministic=cfg.deterministic)(x, encoder_mask)

    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y)

    return x + y


class SubgraphEmbedding(nn.Module):
  """Embeds a bag of nodes features and positions."""
  config: TransformerConfig

  def setup(self):
    cfg = self.config
    self.node_embedding = nn.Embed(cfg.graph_parameters.node_vocab_size,
                                   cfg.embedding_dim)
    # graph_embedding is for embedding the whole bag of nodes. Similar to the
    # CLS token in BERT.
    self.graph_embedding = nn.Embed(1, cfg.hidden_dim)
    # The +2 accounts for the -1 "out of bounds" node, and the "not a node"
    # index.
    # The "not a node" index stems from jax sparse: for an array of shape n,
    # if part of the `nse` elements of the array are actually 0,
    # they will be matched to the index `n`.
    # This happens in our setup when we use L1 penalties causing the actual
    # size of the subgraph to be stricly smaller than max_subgraph_size.
    # Because jax arrays clip out of bounds indices, we only need to
    # add 1 element in the embedding to account for this, and not mix the info
    # with a different node.
    self.position_embedding = nn.Embed(cfg.image_size + 2, cfg.embedding_dim)

    self.node_hidden_layer = nn.Dense(cfg.hidden_dim)
    self.position_hidden_layer = nn.Dense(cfg.hidden_dim)

  def __call__(self, node_features,
               node_ids):
    """Embeds nodes by features and node_id.

    Args:
      node_features: float or int tensor representing the current node's fixed
        features. These features are not learned.
      node_ids: id of the node in the image. Used in place of the position in
        the image.

    Returns:
      logits: float tensor of shape (num_classes,)
    """
    cfg = self.config

    num_nodes = len(node_ids)

    # Embed nodes
    node_embs = self.node_embedding(node_features)
    node_embs = node_embs.reshape(num_nodes, -1)
    node_hiddens = self.node_hidden_layer(node_embs)
    graph_hidden = self.graph_embedding(jnp.zeros(1, dtype=int))
    node_hiddens = jnp.row_stack((node_hiddens, graph_hidden))

    # Embed positions
    # TODO(gnegiar): We need to clip the "not a node" node to make sure it
    # propagates gradients correctly. jax.experimental.sparse uses an out of
    # bounds index to encode elements with 0 value.
    # See https://github.com/google/jax/issues/5760
    node_ids = jnp.clip(node_ids, a_max=cfg.image_size - 1)
    position_embs = self.position_embedding(node_ids + 1)
    position_hiddens = self.position_hidden_layer(position_embs)
    # The graph node has no position.
    position_hiddens = jnp.row_stack(
        (position_hiddens, jnp.zeros(position_hiddens.shape[-1])))

    return node_hiddens, position_hiddens


class TransformerGraphEncoder(nn.Module):
  """Encodes a bag of nodes into a subgraph representation.

  Adapted from https://github.com/google/flax/blob/main/examples/wmt/models.py
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, node_feature_embeddings,
               node_position_embeddings, adjacency_mat,
               qstar):
    """Applies the TransformerEncoder module.

    Args:
      node_feature_embeddings: Embeddings representing nodes.
      node_position_embeddings: Embeddings representing node positions.
      adjacency_mat: Adjacency matrix over the nodes. Not used for now.
      qstar: float tensor of shape (num_of_nodes,) The optimal q weighting over
        the nodes of the graph, from the subgraph selection module.

    Returns:
      encoded: Encoded nodes, with extra Graph node at the end.
    """
    cfg = self.config
    x = node_feature_embeddings + node_position_embeddings

    # Add average weight to graph node for scale
    qstar = jnp.append(qstar, jnp.mean(qstar))

    # Multiply embeddings by node weights. => learn the agent model.
    x = x * qstar[Ellipsis, None]
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)

    x = x.astype(cfg.dtype)
    # TODO(gnegiar): Plot x here to check
    # Keep nodes with positive weights
    mask1d = qstar != 0
    encoder_mask = nn.attention.make_attention_mask(mask1d, mask1d)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(
          config=cfg, name=f"encoderblock_{lyr}")(x, encoder_mask)
      x = x * mask1d[Ellipsis, None]
      # TODO(gnegiar): Also plot x here
      # Possibly plot gradient norms per encoder layer
      # Plot attention weights?
    encoded = nn.LayerNorm(dtype=cfg.dtype, name="encoder_norm")(x)
    return encoded


class ClassificationHead(nn.Module):
  """A 2 layer fully connected network for classification."""
  config: TransformerConfig

  def setup(self):
    cfg = self.config
    self.fc1 = nn.Dense(cfg.hidden_dim)
    self.fc2 = nn.Dense(cfg.num_classes if cfg.num_classes > 2 else 1)

  def __call__(self, x):
    x = nn.relu(self.fc1(x))
    logits = self.fc2(x)
    if self.config.num_classes > 2:
      logits = jax.nn.log_softmax(logits)
    return logits


class TransformerClassifier(nn.Module):
  """A transformer based graph classifier.

  Attributes:
    config: Configuration for the model.
  """
  config: TransformerConfig

  def setup(self):
    cfg = self.config
    self.embedder = SubgraphEmbedding(cfg)
    self.encoder = TransformerGraphEncoder(cfg)
    self.classifier = ClassificationHead(cfg)

  def encode(self, node_features, node_ids,
             adjacency_mat, qstar):
    node_feature_embeddings, node_position_embeddings = self.embedder(
        node_features, node_ids)
    return self.encoder(node_feature_embeddings, node_position_embeddings,
                        adjacency_mat, qstar)

  def decode(self, encoded_graph):
    graph_embedding = encoded_graph[-1]
    logits = self.classifier(graph_embedding)
    return logits

  def __call__(self, node_features, node_ids,
               adjacency_mat, qstar):
    adjacency_mat = adjacency_mat.squeeze(-1)
    encoded_graph = self.encode(node_features, node_ids, adjacency_mat, qstar)
    # The encoder encodes the whole graph in a special token in last position
    return self.decode(encoded_graph)


