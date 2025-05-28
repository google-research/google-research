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

"""Preprocess the dataset from graph tensors to dense tensors.

The script will do the following:
  1. One-hot encoding for categorical features
  2. Normalization for numerical features
  3. Remove graphs with zero cardinality
  4. Remove graphs with no predicate
  5. Remove unused features
  6. Remove NaNs for correlation coefficients and clip them between -1 and 1
  7. Add virtual node and edge to make the graph a connected graph
  8. Compute topological order, spatial encoding, and causal masks
  9. Convert the graph to a dense tensor and save it to a tfrecord file
"""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import tqdm

from CardBench_zero_shot_cardinality_training.graph_transformer import constants
import sparse_deferred.np as sdnp
from sparse_deferred.structs import graph_struct

_INPUT_DATASET_PATH = flags.DEFINE_string(
    "input_dataset_path",
    None,
    "Input dataset path.",
    required=True,
)

_DATASET_NAME = flags.DEFINE_string(
    "dataset_name", None, "Dataset name.", required=True
)

_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    None,
    ["binary_join", "single_table"],
    "Dataset type.",
    required=True,
)

_SCALING_STRATEGY_FILENAME = flags.DEFINE_string(
    "scaling_strategy_filename",
    None,
    "Scaling strategy filename.",
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Output path.",
    required=True,
)


def _onehot_encode(
    graph,
    onehot_encoders,
):
  """Helper function to do one-hot encoding for categorical features."""

  new_value = {}
  for node_name, features_dict in onehot_encoders.items():
    new_value[node_name] = {}
    for feature_name, onehot_encoder in features_dict.items():
      feature = graph.nodes[node_name][feature_name]
      if np.size(feature) == 0:
        continue
      feature = onehot_encoder.transform(np.array(feature).reshape(-1, 1))
      new_value[node_name][feature_name] = feature

  graph = graph.update(nodes=new_value)

  return graph


def _scale_features(
    graph,
    scaling_strategy,
):
  """Helper function to scale numerical features with log and standardization."""

  def _scale_feature_helper(
      value,
      feature_scaling_strategy,
  ):
    if feature_scaling_strategy["log_scale"]:
      log_mean = feature_scaling_strategy["log_mean"]
      log_std = feature_scaling_strategy["log_std"]
      return (
          np.log10(np.clip(value, a_min=1e-9, a_max=None)) - log_mean
      ) / log_std
    else:
      mean = feature_scaling_strategy["mean"]
      std = feature_scaling_strategy["std"]
      return (feature - mean) / std

  new_values = {}
  for node_name in graph.nodes:
    new_values[node_name] = {}
    for feature_name, feature in graph.nodes[node_name].items():
      if (
          node_name in scaling_strategy
          and feature_name in scaling_strategy[node_name]
      ):
        new_values[node_name][feature_name] = _scale_feature_helper(
            feature,
            feature_scaling_strategy=scaling_strategy[node_name][feature_name],
        )
      else:
        new_values[node_name][feature_name] = feature

  graph = graph.update(nodes=new_values)

  return graph


def _scale_histogram(
    graph,
):
  """Scale histograms with its relative value to the min & max value."""

  histogram = graph.nodes["attributes"]["percentiles_100_numeric"]
  histogram_min = np.min(histogram)
  histogram_max = np.max(histogram)
  relative_value = (histogram - histogram_min) / np.clip(
      histogram_max - histogram_min, a_min=1e-9, a_max=None
  )
  relative_value = np.nan_to_num(relative_value, nan=-1.0)
  graph = graph.update(
      nodes={"attributes": {"percentiles_100_numeric": relative_value}}
  )
  return graph


def _preprocess_correlation(graph):
  """Returns an updated graph with NaN correlation feature replaced with 0."""

  correlation = graph.nodes["correlations"]["correlation"]
  correlation = np.clip(correlation, a_min=-1.0, a_max=1.0)
  correlation = np.nan_to_num(correlation, nan=0.0)
  graph = graph.update(nodes={"correlations": {"correlation": correlation}})
  return graph


def shortest_distances(adj):
  """Compute the shortest distances between all pairs of nodes."""

  adj = tf.cast(
      tf.not_equal(tf.identity(adj, "adj"), 0), dtype=tf.float32
  ).numpy()

  adj[adj == 0] = np.inf

  dist = list(map(lambda p: list(map(lambda q: q, p)), adj))

  # Adding vertices individually
  n = len(adj)
  for r in range(n):
    for p in range(n):
      for q in range(n):
        dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])

  for i in range(len(dist)):
    for j in range(len(dist[i])):
      if dist[i][j] == np.inf:
        dist[i][j] = constants.MAX_NUM_NODES

  return tf.convert_to_tensor(dist, dtype=tf.float32)


def build_parent_mask(dist):
  mask = tf.cast(tf.equal(dist, tf.constant(1.0)), dtype=tf.float32)
  # Add self edge
  mask = tf.eye(mask.shape[0], dtype=tf.float32) + mask
  # Transpose to fit transformer mask definition
  mask = tf.transpose(mask, [1, 0])
  return mask


def build_ancestor_mask(dist):
  mask = tf.cast(tf.less(dist, constants.MAX_NUM_NODES), dtype=tf.float32)
  # Add self edge
  mask = tf.eye(mask.shape[0], dtype=tf.float32) + mask
  # Transpose to fit transformer mask definition
  mask = tf.transpose(mask, [1, 0])
  return mask


def build_topological_order(dist):
  """Find out the index of the root of the tree."""

  root_index = tf.reshape(
      tf.where(
          tf.equal(
              tf.reduce_sum(
                  tf.cast(
                      tf.math.greater_equal(dist, constants.MAX_NUM_NODES),
                      dtype=tf.int32,
                  ),
                  axis=1,
              ),
              tf.shape(dist)[0],
          )
      ),
      shape=-1,
  )

  # Distance to the root is the topological order
  # Plus 1 for the virtual node
  order = tf.reshape(
      tf.reduce_min(tf.gather(dist, root_index, axis=1) + 1, axis=1),
      shape=(-1, 1),
  )
  order = tf.where(tf.equal(order, constants.MAX_NUM_NODES + 1), 1, order)
  return order


def bidirectional_spatial_encoding(dist):
  """Bidirectional spatial encoding for DAGs.

  d_ii = MAX_NUM_NODES
  If there exists i->j:
    d_ij = shortest distance from node i to node j
    d_ji = negative shortest distance from node j to node i
  else:
    d_ij = MAX_NUM_NODES
    d_ji = MAX_NUM_NODES

  Args:
    dist: shortest distance from node i to node j

  Returns:
    bidirectional spatial encoding
  """

  dist_neg_transposed = -tf.transpose(dist, [1, 0])
  dist_neg_transposed = tf.where(
      tf.equal(dist_neg_transposed, -constants.MAX_NUM_NODES),
      constants.MAX_NUM_NODES,
      dist_neg_transposed,
  )

  dist = tf.where(
      tf.equal(dist, constants.MAX_NUM_NODES), dist_neg_transposed, dist
  )
  return dist


def convert_graph_to_dense_tensor(
    graph,
):
  """Encode the GraphTensor record to a Tensor dictionary.

  Args:
    graph: GraphTensor record of a graph

  Returns:
    Tensor dictionary
    {
      `node_padding`: 1 if padded, otherwise 0[MAX_NUM_NODES],
      `node`: node features, [n_node, NODE_FEATURE_DIM],
      `in_degree`: in degree, [n_node, 1],
      `out_degree`: out degree, [n_node, 1],
      `spatial_encoding`: d_ij = shortest distance from node i to node j,
      [n_node, n_node],
      `topological_order`: topological order, [n_node, 1],
      `parent_causal_mask`: m_ij = whether j is a parent of i, [n_node, n_node],
      `ancestor_causal_mask`: m_ij = whether j is an ancestor of i, [n_node,
      n_node],
      `edge_encoding`: Adj matrix with edge types [n_node, n_node],
    }
  """
  # One hot encoder for node types
  node_type_onehot_encoder = sklearn.preprocessing.OneHotEncoder(  # pytype: disable=wrong-keyword-args  # sklearn-update
      categories=[constants.NODE_TYPES], sparse=False, dtype=np.int64
  )
  node_type_onehot_encoder.fit(np.array(constants.NODE_TYPES).reshape(-1, 1))

  # compute graph level node index and compose tensors from graph tensors
  start_index = {}
  n_node = 0  # total number of nodes in a single graph
  node_tensors = []
  for node_name in graph.nodes:
    # Skip graph-level feature
    if node_name == "g":
      continue
    # Skip empty node
    if graph.get_num_nodes(sdnp.engine, node_name) == 0:
      continue

    start_index[node_name] = n_node
    dim = graph.get_num_nodes(sdnp.engine, node_name)
    n_node += dim
    feature_tensors = []
    for feature_name in sorted(graph.nodes[node_name]):
      feature_tensor = graph.nodes[node_name][feature_name]
      feature_tensors.append(np.reshape(feature_tensor, [dim, -1]))

    # Concatenate all features in the feature dimension
    all_features_tensor = np.concatenate(feature_tensors, axis=1)

    # Add node type as a new feature at the beginning
    node_type_tensor = node_type_onehot_encoder.transform(
        np.array(node_name).reshape(-1, 1)
    )
    node_type_tensor = np.repeat(node_type_tensor, dim, axis=0)

    # Pad feature dimension
    padding_dim = (
        constants.NODE_FEATURE_DIM
        - all_features_tensor.shape[1]
        - node_type_tensor.shape[1]
    )

    all_features_tensor = np.concatenate(
        [
            node_type_tensor,
            all_features_tensor,
            np.zeros(shape=[dim, padding_dim], dtype=np.float32),
        ],
        axis=1,
    )
    node_tensors.append(all_features_tensor)

  # Concatenate accross all nodes
  node_tensor = tf.cast(tf.concat(node_tensors, axis=0), dtype=tf.float32)

  edge_list = []
  adj_value = []

  # Compute out degree and in degree for each node
  for edge_name in graph.edges:
    # Skip empty edge
    if np.size(graph.edges[edge_name][0][0]) == 0:
      continue

    source_name = constants.EDGE_TYPES[edge_name][0]
    target_name = constants.EDGE_TYPES[edge_name][1]

    source = np.reshape(
        graph.edges[edge_name][0][0] + start_index[source_name], [-1, 1]
    )
    target = np.reshape(
        graph.edges[edge_name][0][1] + start_index[target_name], [-1, 1]
    )

    edge_list.append(np.concatenate([source, target], axis=1))

    # Virtual edge type
    adj_value.extend(
        [list(constants.EDGE_TYPES.keys()).index(edge_name)] * source.shape[0]
    )

  # Compute adjacency matrix
  adj_index = np.concatenate(edge_list, axis=0).astype(np.int64)
  adj_sparse = tf.sparse.reorder(
      tf.sparse.SparseTensor(
          indices=adj_index, values=adj_value, dense_shape=(n_node, n_node)
      )
  )

  adj = tf.sparse.to_dense(adj_sparse)

  # Compute spatial encoding
  # d_ij = shortest distance from node i to node j
  dist = shortest_distances(adj)

  # Compute bidirectional encoding
  bidi_dist = bidirectional_spatial_encoding(dist)

  # compute topological order
  topological_order = build_topological_order(dist)

  # Compute causal masks
  # m_ij = whether node j is considered when computing the attention weights
  # for node i
  parent_causal_mask = build_parent_mask(dist)
  ancestor_causal_mask = build_ancestor_mask(dist)

  # Compose input data tensor `x`
  x = {}
  # Add virtual node at the beginnning
  # v-node feature, all zeros
  node_tensor = tf.concat(
      [tf.zeros([1, node_tensor.shape[1]], dtype=tf.float32), node_tensor],
      axis=0,
  )
  # topological order
  # v-node topological order = 0
  topological_order = tf.concat(
      [tf.Variable([[0]], dtype=tf.float32), topological_order], axis=0
  )
  # adj matrix
  # v-node edge type = VIRTUAL_EDGE_TYPE = 0
  adj = tf.concat(
      [
          tf.repeat(
              tf.Variable(
                  [[list(constants.EDGE_TYPES.keys()).index("pseudo_edge")]],
                  dtype=tf.float32,
              ),
              repeats=n_node,
              axis=0,
          ),
          tf.cast(adj, dtype=tf.float32),
      ],
      axis=1,
  )
  adj = tf.concat([tf.zeros(shape=[1, n_node + 1]), adj], axis=0)

  # dist matrix
  # v-node distances:
  #   node i -> v-node = 0 (special distance), i.e., d_i0 = 0
  #   v-node -> node i = MAX_NUM_NODES, i.e, d_0i = MAX_NUM_NODES
  #
  dist = tf.concat(
      [tf.zeros(shape=[n_node, 1]), dist],
      axis=1,
  )
  dist = tf.concat(
      [constants.MAX_NUM_NODES * tf.ones(shape=[1, n_node + 1]), dist], axis=0
  )

  # bidi dist matrix
  # v-node distances:
  #   node i -> v-node = 0 (special distance), i.e., d_i0 = 0
  #   v-node -> node i = MAX_NUM_NODES, i.e, d_0i = MAX_NUM_NODES
  #
  bidi_dist = tf.concat(
      [tf.zeros(shape=[n_node, 1]), bidi_dist],
      axis=1,
  )
  bidi_dist = tf.concat(
      [constants.MAX_NUM_NODES * tf.ones(shape=[1, n_node + 1]), bidi_dist],
      axis=0,
  )

  # causal_masks
  # v-node parent causal masks: node i -> v-node = 1; v-node -> node i = 0
  parent_causal_mask = tf.concat(
      [tf.zeros(shape=[n_node, 1]), parent_causal_mask],
      axis=1,
  )
  parent_causal_mask = tf.concat(
      [tf.ones(shape=[1, n_node + 1]), parent_causal_mask], axis=0
  )

  # causal_masks
  # v-node ancestor causal masks: node i -> v-node = 1; v-node -> node i = 0
  ancestor_causal_mask = tf.concat(
      [tf.zeros(shape=[n_node, 1]), ancestor_causal_mask],
      axis=1,
  )
  ancestor_causal_mask = tf.concat(
      [tf.ones(shape=[1, n_node + 1]), ancestor_causal_mask], axis=0
  )

  x["node_padding"] = tf.concat(
      [
          tf.zeros((n_node + 1,), dtype=tf.float32),
          tf.ones((constants.MAX_NUM_NODES - n_node - 1,), dtype=tf.float32),
      ],
      axis=0,
  )
  x["node"] = tf.convert_to_tensor(node_tensor, dtype=tf.float32)
  x["spatial_encoding"] = tf.convert_to_tensor(dist, dtype=tf.float32)
  x["bidi_spatial_encoding"] = tf.convert_to_tensor(bidi_dist, dtype=tf.float32)
  x["topological_order"] = tf.convert_to_tensor(
      topological_order, dtype=tf.float32
  )
  x["parent_causal_mask"] = tf.convert_to_tensor(
      parent_causal_mask, dtype=tf.float32
  )
  x["ancestor_causal_mask"] = tf.convert_to_tensor(
      ancestor_causal_mask, dtype=tf.float32
  )
  x["edge_encoding"] = tf.convert_to_tensor(adj, dtype=tf.float32)

  x["cardinality"] = tf.convert_to_tensor(
      graph.nodes["g"]["cardinality"], dtype=tf.float32
  )
  x["exec_time"] = tf.convert_to_tensor(
      graph.nodes["g"]["exec_time"], dtype=tf.float32
  )
  x["query_id"] = tf.convert_to_tensor(
      graph.nodes["g"]["query_id"], dtype=tf.float32
  )

  return x


def write_graph(
    file_writer, x
):
  """Write a graph to a tfrecord file."""

  record_bytes = tf.train.Example(
      features=tf.train.Features(
          feature={
              "query_id": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["query_id"]).numpy()]
                  )
              ),
              "node": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["node"]).numpy()]
                  )
              ),
              "node_padding": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["node_padding"]).numpy()]
                  )
              ),
              "edge_encoding": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["edge_encoding"]).numpy()]
                  )
              ),
              "spatial_encoding": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[
                          tf.io.serialize_tensor(x["spatial_encoding"]).numpy()
                      ]
                  )
              ),
              "bidi_spatial_encoding": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[
                          tf.io.serialize_tensor(
                              x["bidi_spatial_encoding"]
                          ).numpy()
                      ]
                  )
              ),
              "topological_order": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[
                          tf.io.serialize_tensor(x["topological_order"]).numpy()
                      ]
                  )
              ),
              "parent_causal_mask": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[
                          tf.io.serialize_tensor(
                              x["parent_causal_mask"]
                          ).numpy()
                      ]
                  )
              ),
              "ancestor_causal_mask": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[
                          tf.io.serialize_tensor(
                              x["ancestor_causal_mask"]
                          ).numpy()
                      ]
                  )
              ),
              "cardinality": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["cardinality"]).numpy()]
                  )
              ),
              "exec_time": tf.train.Feature(
                  bytes_list=tf.train.BytesList(
                      value=[tf.io.serialize_tensor(x["exec_time"]).numpy()]
                  )
              ),
          }
      )
  ).SerializeToString()

  file_writer.write(record_bytes)


def prepare_dataset(
    db,
    scaling_strategy,
    file_writer,
    features_to_remove = None,
    remove_zero_card = True,
    remove_no_pred = True,
):
  """Preprocess the dataset from graph tensors and store it as a tfrecord file."""

  # Prepare onehot encoders
  onehot_encoders = {}
  for (
      node_name,
      features_dict,
  ) in constants.CATEGORICAL_FEATURE_UNIQUE_DICT.items():
    onehot_encoders[node_name] = {}
    for feature_name, (
        _,
        feature_unique_value,
    ) in features_dict.items():
      onehot_encoder = sklearn.preprocessing.OneHotEncoder(  # pytype: disable=wrong-keyword-args  # sklearn-update
          categories=[feature_unique_value], sparse=False, dtype=np.int64
      )
      onehot_encoder.fit(np.array(feature_unique_value).reshape(-1, 1))
      onehot_encoders[node_name][feature_name] = onehot_encoder

  # Iterate over all graphs in the dataset
  for i in tqdm.tqdm(range(db.size)):
    graph = db.get_item(i)
    # Remove graph with zero cardinality
    if remove_zero_card and graph.nodes["g"]["cardinality"][0] == 0:
      logging.warning(
          "Graph query_id=%s with zero cardinality is removed.",
          graph.nodes["g"]["query_id"][0],
      )
      continue

    # Remove graph with no predicate
    if (
        remove_no_pred
        and np.size(graph.nodes["predicates"]["predicate_operator"]) == 0
    ):
      logging.warning(
          "Graph query_id=%s with no predicate is removed.",
          graph.nodes["g"]["query_id"][0],
      )
      continue

    # Remove unused features
    if features_to_remove is not None:
      for node, features in features_to_remove.items():
        for feature in features:
          graph.nodes[node].pop(feature)

    # One-hot encoder for catagorical features
    graph = _onehot_encode(graph, onehot_encoders)

    # Normalization for numerical features
    graph = _scale_features(graph, scaling_strategy)

    # Normalize histograms if exist
    if (
        "attributes" in graph.nodes
        and "percentiles_100_numeric" in graph.nodes["attributes"]
    ):
      graph = _scale_histogram(graph)

    # Remove NaNs for correlatino coefficients and clip them between -1 and 1
    if (
        "correlations" in graph.nodes
        and "correlation" in graph.nodes["correlations"]
    ):
      graph = _preprocess_correlation(graph)

    # Convert graph to dense tensor and write to file
    graph_dense_tensor = convert_graph_to_dense_tensor(graph)
    write_graph(file_writer, graph_dense_tensor)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  db = graph_struct.InMemoryDB.from_sharded_files(
      os.path.join(
          _INPUT_DATASET_PATH.value,
          _DATASET_TYPE.value,
          f"{_DATASET_NAME.value}_{_DATASET_TYPE.value}.npz",
      )
  )
  with open(
      os.path.join(
          _INPUT_DATASET_PATH.value,
          _DATASET_TYPE.value,
          _SCALING_STRATEGY_FILENAME.value,
      )
  ) as f:
    scaling_strategy = json.load(f)

  file_writer = tf.io.TFRecordWriter(
      os.path.join(
          _OUTPUT_PATH.value,
          _DATASET_TYPE.value,
          f"{_DATASET_NAME.value}_{_DATASET_TYPE.value}.tfrecord",
      )
  )

  prepare_dataset(
      db,
      scaling_strategy,
      file_writer,
      features_to_remove=constants.REMOVE_FEATURE_DICT,
  )


if __name__ == "__main__":
  app.run(main)
