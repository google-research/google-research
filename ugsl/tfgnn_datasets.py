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

# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
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
# ==============================================================================
"""Infrastructure and implementation of in-memory graph data.

Instantiating an object will download a dataset, and cache it locally. The
datasets will be cached on ~/data/ogb (for "ogbn-" and "ogbl-" datasets), which
can be overridden by setting environment variable `OGB_CACHE_DIR`; and on
~/data/planetoid (for "cora", "citeseer", "pubmed"), which can be overridden by
environment variable `PLANETOID_CACHE_DIR`.

High-level Abstract Classes:

  * `InMemoryGraphData`: provides nodes, edges, and features, for a
     homogeneous or a heteregenous graph.
  * `NodeClassificationGraphData`: an `InMemoryGraphData` that also provides
    list of {train, test, validation} nodes, as well as their labels.
  * `LinkPredictionGraphData`: an `InMemoryGraphData` that also provides lists
    of edges in {train, test, validation} partitions.


`InMemoryGraphData` implementations can provide

  * a single GraphTensor for training on one big graph (e.g., for node
    classification with `tf_trainer.py` or `keras_trainer.py`),
  * a big graph from which in-memory sampling (e.g., `int_arithmetic_sampler`)
    can create dataset of sampled subgraphs (encoded as `tfgnn.GraphTensor`).

All `InMemoryGraphData` implementations automatically inherit abilities of:

  * `as_graph_tensor()` .
  * These methods can be plugged-into TF-GNN models and training loops, e.g.,
    for node classification (see `tf_trainer.py` and `keras_trainer.py`).
  * In addition, they can be plugged-into in-memory sampling (see
    `int_arithmetic_sampler.py`, and example trainer script,
    `keras_minibatch_trainer.py`).


Concrete implementations:

  * Node classification (inheriting `NodeClassificationGraphData`)

    * `OgbnData`: Wraps node classification graph data from OGB, i.e., with
      name prefix of "ogbn-", such as, "ogbn-arxiv".

    * `PlanetoidGraphData`: wraps graph data that are popularized by GCN paper
      (cora, citeseer, pubmed).

  * Link prediction (inheriting `LinkPredictionGraphData`)

    * `OgblData`: Wraps link prediction graph data from OGB, i.e., with name
      prefix of "ogbl-", such as, "ogbl-ddi".


# Usage Example.

```
graph_data = datasets.OgbnData('ogbn-arxiv')

# Optionally, make graph undirected.
graph_data = graph_data.with_self_loops(True)

# add self-loops:
graph_data = graph_data.with_undirected_edges(True)

# To get GraphTensor and GraphSchema at any graph data:
graph_tensor = graph_data.as_graph_tensor()
graph_schema = graph_data.graph_schema()

spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
# or optionally, by "relaxing" the batch dimension of `graph_tensor` (to None):
# spec = graph_tensor.spec.relax(num_nodes=True, num_edges=True)
```

The first line is equivalent to
`graph_data = datasets.get_in_memory_graph_data('ogbn-arxiv')`. Which is more
general, because it can load other data types:
  * ogbn-* are node-calssificiation datasets for OGB.
  * 'pubmed', 'cora', 'citeseer', correspond to transductive graphs used in
    Planetoid (Yang et al, ICML'16).


`graph_tensor` (type `GraphTensor`) contains all nodes, edges, and features.
If it is a node-classification dataset, the training labels are also populated.
**For nodes not in training set**, label feature will be `-1`. To also include
If you want to explicitly get all labels from all partitions, you may:

```
graph_data = graph_data.with_split(['train', 'test', 'validation'])
graph_tensor = graph_data.graph_tensor
```

Chaining `with_*` calls can reduce verbosity. For example,
```
graph_data = (
    datasets.OgbnData('ogbn-arxiv').with_undirected_edges(True)
    .with_self_loops(True))
graph_tensor = graph_data.as_graph_tensor()
```
"""
import abc
import copy
import io
import json
import os
import pickle
import sys
from typing import Any, List, Mapping, NamedTuple, Tuple, Union, Optional
import urllib.request

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_gnn as tfgnn


class InMemoryGraphData(abc.ABC):
  """Abstract class for hold a graph data in-memory (nodes, edges, features).

  Subclasses must implement methods `node_features_dicts()`, `node_counts()`,
  `edge_lists()`, `node_sets()`, and optionally, `context()`. They inherit
  methods `graph_schema()`, `edge_sets()`, and `as_graph_tensor()` based on
  those.
  """

  def __init__(self, make_undirected = False,
               add_self_loops = False):
    self._make_undirected = make_undirected
    self._add_self_loops = add_self_loops

  @property
  def name(self):
    """Returns name of dataset object. Can be overridden to return data name."""
    return self.__class__.__name__

  def with_undirected_edges(self, make_undirected):
    """Returns same graph data but with undirected edges added (or removed).

    Subsequent calls to `.graph_schema()` and to `.as_graph_tensor()` will be
    affected. Specifically, the generated output `tfgnn.GraphTensor` (by
    `.as_graph_tensor()`) will reverse all homogeneous edge sets (where its
    source node set equals its target node set). Suppose edge `(i, j)` is
    included in *homogeneous* edge set "MyEdgeSet", then output `GraphTensor`
    will also contain edge `(j, i)` on edge set "MyEdgeSet". If edge `(j, i)`
    already exists, then it will be duplicated.

    If make_undirected == True:

      * output of `.as_graph_tensor()` will contain only edge-set names that are
        returned by `.edge_sets()`, where each homogeneous edge-set with M edges
        will be expanded to M*2 edges with edge `M+k` reversing edge `k`.
      * output of `.graph_schema()` will contain only edge-sets returned by
        `edge_sets`.

    If make_undirected == False:

      * output of `.as_graph_tensor()` will contain, for each edge set "EdgeSet"
        (returned by `.edge_sets()`) a new edge-set "rev_EdgeSet" that reverses
        the "EdgeSet".
      * output of `.graph_schema()`. will have both "EdgeSet" and "rev_EdgeSet".
      * `with_reverse_edge_sets()` is an equivalent and a more explicit method
        to add reverse edge sets to the graph tensor and its schema.
    Args:
      make_undirected: If True, subsequent calls to `.graph_schema()` and
        `.as_graph_tensor()` will export an undirected graph. If False, a
        directed graph (with additional "rev_*" edges).
    """
    modified = copy.copy(self)
    modified._make_undirected = make_undirected  # pylint: disable=protected-access -- same class.
    return modified

  def with_reverse_edge_sets(self):
    """Returns same graph data but with reverse edge sets added."""

    # Calling `with_undirected_edges` with `False` input automatically makes the
    # output of `.as_graph_tensor()` to contain, for each edge set "EdgeSet"
    # (returned by `.edge_sets()`) a new edge-set "rev_EdgeSet" that reverses
    # the "EdgeSet". Similarly, output of `.graph_schema()`. will have both
    # "EdgeSet" and "rev_EdgeSet".
    return self.with_undirected_edges(False)

  def with_self_loops(self, add_self_loops):
    """Returns same graph data but with self-loops added (or removed).

    If add_self_loops == True, then subsequent calls to `.as_graph_tensor()`
    will contain edges `[(i, i) for i in range(N_j)]`, for each homogeneous edge
    set j, where `N_j` is the number of nodes in node set connected by edge set
    `j`.

    NOTE: self-loops will be added *regardless* if they already exist or not.
    If the datasets already has self-loops, calling this, will double the self-
    loop edges.

    Args:
      add_self_loops: If set, self-loops will be amended on subsequent calls to
      `.as_graph_tensor()`. If not, no self-loops will be automatically added.
    """
    modified = copy.copy(self)
    modified._add_self_loops = add_self_loops  # pylint: disable=protected-access -- same class.
    return modified

  @abc.abstractmethod
  def node_counts(self):
    """Returns total number of graph nodes per node set."""
    raise NotImplementedError()

  @abc.abstractmethod
  def node_features_dicts(self):
    """Returns 2-level dict: NodeSetName->FeatureName->Feature tensor.

    For every node set (`"x"`), feature tensor must have leading dimension equal
    to number of nodes in node set (`.node_counts()["x"]`). Other dimensions are
    dataset specific.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def edge_lists(self):
    """Returns dict from "edge type tuple" to int Tensor of shape (2, num_edges).

    "edge type tuple" string three-tuple:
      `(source node set name, edge set name, target node set name)`.
    where `edge set name` must be unique.
    """
    raise NotImplementedError()

  def node_sets(self):
    """Returns node sets of entire graph (dict: node set name -> NodeSet)."""
    node_counts = self.node_counts()
    features_dicts = self.node_features_dicts()
    node_set_names = set(node_counts.keys()).union(features_dicts.keys())
    return (
        {name: tfgnn.NodeSet.from_fields(sizes=as_tensor([node_counts[name]]),
                                         features=features_dicts.get(name, {}))
         for name in node_set_names})

  def context(self):
    return None

  def as_graph_tensor(self):
    """Returns `GraphTensor` holding the entire graph."""
    return tfgnn.GraphTensor.from_pieces(
        node_sets=self.node_sets(), edge_sets=self.edge_sets(),
        context=self.context())

  def graph_schema(self):
    """`tfgnn.GraphSchema` instance corresponding to `as_graph_tensor()`."""
    # Populate node features specs.
    schema = tfgnn.GraphSchema()
    for node_set_name, node_set in self.node_sets().items():
      node_features = schema.node_sets[node_set_name]
      for feat_name, feature in node_set.features.items():
        node_features.features[feat_name].dtype = feature.dtype.as_datatype_enum
        for dim in feature.shape[1:]:
          node_features.features[feat_name].shape.dim.add().size = dim

    # Populate edge specs.
    for edge_type in self.edge_lists().keys():
      src_node_set_name, edge_set_name, dst_node_set_name = edge_type
      # Populate edges with adjacency and it transpose.
      schema.edge_sets[edge_set_name].source = src_node_set_name
      schema.edge_sets[edge_set_name].target = dst_node_set_name
      if not self._make_undirected:
        schema.edge_sets['rev_' + edge_set_name].source = dst_node_set_name
        schema.edge_sets['rev_' + edge_set_name].target = src_node_set_name

    return schema

  def edge_sets(self):
    """Returns edge sets of entire graph (dict: edge set name -> EdgeSet)."""
    edge_sets = {}
    node_counts = self.node_counts() if self._add_self_loops else None
    for edge_type, edge_list in self.edge_lists().items():
      (source_node_set_name, edge_set_name, target_node_set_name) = edge_type

      if self._make_undirected and source_node_set_name == target_node_set_name:
        edge_list = tf.concat([edge_list, edge_list[::-1]], axis=-1)
      if self._add_self_loops and source_node_set_name == target_node_set_name:
        all_nodes = tf.range(node_counts[source_node_set_name],
                             dtype=edge_list.dtype)
        self_connections = tf.stack([all_nodes, all_nodes], axis=0)
        edge_list = tf.concat([edge_list, self_connections], axis=-1)
      edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
          sizes=tf.shape(edge_list)[1:2],
          adjacency=tfgnn.Adjacency.from_indices(
              source=(source_node_set_name, edge_list[0]),
              target=(target_node_set_name, edge_list[1])))
      if not self._make_undirected:
        edge_sets['rev_' + edge_set_name] = tfgnn.EdgeSet.from_fields(
            sizes=tf.shape(edge_list)[1:2],
            adjacency=tfgnn.Adjacency.from_indices(
                source=(target_node_set_name, edge_list[1]),
                target=(source_node_set_name, edge_list[0])))
    return edge_sets

  def save(self, filename):
    """Superclasses can save themselves to disk."""
    raise NotImplementedError()


class NodeSplit(NamedTuple):
  """Contains 1D int tensors holding positions of {train, valid, test} nodes.

  This is returned by `NodeClassificationGraphData.node_split()`
  """
  train: tf.Tensor
  validation: tf.Tensor
  test: tf.Tensor


class EdgeSplit(NamedTuple):
  """Contains positive and negative edges in {train, test, valid} partitions.

  Each `tf.Tensor` will be of shape `[2, num_edges]` with dtype int64.
  """
  # Only need positive edges for training. The (entire) graph compliment can be
  # used for negative edges.
  train_edges: tf.Tensor
  validation_edges: tf.Tensor
  test_edges: tf.Tensor
  negative_validation_edges: tf.Tensor
  negative_test_edges: tf.Tensor


class NodeClassificationGraphData(InMemoryGraphData):
  """Adapts `InMemoryGraphData` for node classification settings.

  Subclasses should information for node classification: (node labels, name of
  node set, and partitions train:validation:test nodes).
  """

  def __init__(self, split = 'train', use_labels_as_features=False):
    super().__init__()
    self._splits = [split]
    self._use_labels_as_features = use_labels_as_features

  def with_split(self, split = 'train'
                 ):
    """Returns same graph data but with specific partition.

    Args:
      split: must be one of {"train", "validation", "test"}.
    """
    splits = split if isinstance(split, (tuple, list)) else [split]
    for split in splits:
      if split not in ('train', 'validation', 'test'):
        raise ValueError(
            'split must be one of {"train", "validation", "test"}.')
    modified = copy.copy(self)
    modified._splits = splits  # pylint: disable=protected-access -- same class.
    return modified

  def with_labels_as_features(
      self, use_labels_as_features):
    """Returns same graph data with labels as an additional feature on nodes.

    The feature will be added to the node-set with name `self.labeled_nodeset`.

    Args:
      use_labels_as_features: Label feature will be added iff set to True.
    """
    modified = copy.copy(self)
    modified._use_labels_as_features = use_labels_as_features  # pylint: disable=protected-access -- same class.
    return modified

  @property
  def splits(self):
    return copy.copy(self._splits)

  @abc.abstractmethod
  def num_classes(self):
    """Number of node classes. Max of `labels` should be `< num_classes`."""
    raise NotImplementedError('num_classes')

  @abc.abstractmethod
  def node_split(self):
    """`NodeSplit` with attributes `train`, `validation`, `test` set.

    The attributes are set to indices of the `labeled_nodeset`. Specifically,
    they correspond to leading dimension of features of the node set.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def labels(self):
    """int vector containing labels for train & validation nodes.

    Size of vector is number of nodes in the labeled node set. In particular:
    `self.labels().shape[0] == self.node_counts()[self.labeled_nodeset]`.
    Specifically, the vector has as many entries as there are nodes belonging to
    the node set that this task aims to predict labels for.

    Entry `labels()[i]` will be -1 iff `i in self.node_split().test`. Otherwise,
    `labels()[i]` will be int in range [`0`, `self.num_classes() - 1`].
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def test_labels(self):
    """Like the above but contains no -1's.

    Every {train, valid, test} node will have its class label.
    """
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def labeled_nodeset(self):
    """Name of node set which `labels` and `node_splits` reference."""
    raise NotImplementedError()

  @abc.abstractmethod
  def node_features_dicts_without_labels(self):
    raise NotImplementedError()

  def node_features_dicts(self):
    """Implements a method required by the base class.

    This method combines the data from `labels()` or `test_labels()` with the
    data from `node_features_dicts_without_labels()` into a single features
    dict.

    Subclasses need to implement aforementioned methods and may inherit this.

    Returns:
      NodeSetName -> FeatureName -> Feature Tensor.
    """
    node_features_dicts = self.node_features_dicts_without_labels()
    node_features_dicts = {ns: dict(features)  # Shallow copy.
                           for ns, features in node_features_dicts.items()}
    if self._use_labels_as_features:
      if 'test' in self._splits:
        node_features_dicts[self.labeled_nodeset]['label'] = self.test_labels()
      else:
        node_features_dicts[self.labeled_nodeset]['label'] = self.labels()

    return node_features_dicts

  def context(self):
    node_split = self.node_split()
    seed_nodes = tf.concat(
        [getattr(node_split, split) for split in self._splits], axis=0)
    seed_nodes = tf.expand_dims(seed_nodes, axis=0)
    seed_feature_name = 'seed_nodes.' + self.labeled_nodeset

    return tfgnn.Context.from_fields(features={seed_feature_name: seed_nodes})

  def graph_schema(self):
    graph_schema = super().graph_schema()
    context_features = graph_schema.context.features
    context_features['seed_nodes.' + self.labeled_nodeset].dtype = (
        tf.int64.as_datatype_enum)
    return graph_schema

  def save(self, filename):
    """Saves the dataset on numpy compressed (.npz) file.

    The file runs once the functions,
    (labeled_nodeset, test_labels, labels, node_split, edge_lists, node_counts,
     node_features, num_classes),
    composes a flat dict (keys are json-encoded arrays), then writes as numpy
    file. Flat dict is needed as numpy only saves named arrays, not nested
    structures.

    Args:
      filename: file path to save onto. ".npz" extension is recommended. Parent
        directory must exist.
    """
    features_without_labels = self.node_features_dicts_without_labels()
    node_split = self.node_split()

    attribute_dict = {
        ('num_classes',): self.num_classes(),
        ('node_split', 'train'): node_split.train.numpy(),
        ('node_split', 'test'): node_split.test.numpy(),
        ('node_split', 'validation'): node_split.validation.numpy(),
        ('labels',): self.labels().numpy(),
        ('test_labels',): self.test_labels().numpy(),
        ('labeled_nodeset',): self.labeled_nodeset,
    }

    # Edge sets.
    for (src_name, es_name, tgt_name), es_indices in self.edge_lists().items():
      key = ('e', '#', src_name, es_name, tgt_name)
      attribute_dict[key] = es_indices.numpy()

    for ns_name, features in features_without_labels.items():
      for feature_name, feature_tensor in features.items():
        attribute_dict[('n', ns_name, feature_name)] = feature_tensor.numpy()

    for node_set_name, node_count in self.node_counts().items():
      attribute_dict[('nc', node_set_name)] = node_count

    bytes_io = io.BytesIO()
    attribute_dict = {json.dumps(k): v for k, v in attribute_dict.items()}
    np.savez_compressed(bytes_io, **attribute_dict)
    with tf.io.gfile.GFile(filename, 'wb') as f:
      f.write(bytes_io.getvalue())

  @staticmethod
  def load(filename):
    """Loads from disk `NodeClassificationGraphData` that was `save()`ed."""
    dataset_dict = dict(np.load(tf.io.gfile.GFile(filename, 'rb')))
    dataset_dict = {tuple(json.loads(k)): v for k, v in dataset_dict.items()}
    edge_lists = {}
    node_features = {}
    node_counts = {}
    for key, array in dataset_dict.items():
      # edge lists.
      if key[0] == 'e':
        if key[1] != '#':
          raise ValueError('Expecting ("e", "#", ...) but got %s' % str(key))
        src_name = key[2]
        es_name = key[3]
        tgt_name = key[4]
        indices = as_tensor(array)
        edge_lists[(src_name, es_name, tgt_name)] = indices
      # node features.
      if key[0] == 'n':
        node_set_name = key[1]
        feature_name = key[2]
        if node_set_name not in node_features:
          node_features[node_set_name] = {}
        node_features[node_set_name][feature_name] = as_tensor(array)
      if key[0] == 'nc':
        node_counts[key[1]] = int(array)

    return _PreloadedNodeClassificationGraphData(
        num_classes=dataset_dict[('num_classes',)],
        node_features_dicts_without_labels=node_features,
        node_counts=node_counts,
        edge_lists=edge_lists,
        node_split=NodeSplit(
            train=as_tensor(dataset_dict[('node_split', 'train')]),
            validation=as_tensor(dataset_dict[('node_split', 'validation')]),
            test=as_tensor(dataset_dict[('node_split', 'test')])),
        labels=as_tensor(dataset_dict[('labels',)]),
        test_labels=as_tensor(dataset_dict[('test_labels',)]),
        labeled_nodeset=str(dataset_dict[('labeled_nodeset',)]))


class _PreloadedNodeClassificationGraphData(NodeClassificationGraphData):
  """Dataset from pre-computed attributes."""

  def __init__(
      self, num_classes,
      node_features_dicts_without_labels,
      node_counts,
      edge_lists,
      node_split, labels, test_labels,
      labeled_nodeset):
    super().__init__()
    self._num_classes = num_classes
    self._node_features_dicts_without_labels = (
        node_features_dicts_without_labels)
    self._node_counts = node_counts
    self._edge_lists = edge_lists
    self._node_split = node_split
    self._labels = labels
    self._test_labels = test_labels
    self._labeled_nodeset = labeled_nodeset

  def num_classes(self):
    return self._num_classes

  def node_features_dicts_without_labels(self):
    return self._node_features_dicts_without_labels

  def node_counts(self):
    return self._node_counts

  def edge_lists(self):
    return self._edge_lists

  def node_split(self):
    return self._node_split

  def labels(self):
    return self._labels

  def test_labels(self):
    return self._test_labels

  @property
  def labeled_nodeset(self):
    return self._labeled_nodeset


class _OgbGraph:
  """Wraps data exposed by OGB graph objects, while enforcing heterogeneity.

  Attributes offered by this class are consistent with the APIs of GraphData.
  """

  def __init__(self, graph):
    """Reads dict OGB `graph` and into the attributes defined below.

    Args:
      graph: Dict, described in
        https://github.com/snap-stanford/ogb/blob/master/ogb/io/README.md#2-saving-graph-list
    """
    if 'edge_index_dict' in graph:  # Heterogeneous graph
      assert 'num_nodes_dict' in graph
      assert 'node_feat_dict' in graph

      # node set name -> feature name -> feature matrix (numNodes x featDim).
      node_set = {node_set_name: {'feat': as_tensor(feat)}
                  for node_set_name, feat in graph['node_feat_dict'].items()
                  if feat is not None}
      # Populate remaining features
      for key, node_set_name_to_feat in graph.items():
        if key.startswith('node_') and key != 'node_feat_dict':
          feat_name = key.split('node_', 1)[-1]
          for node_set_name, feat in node_set_name_to_feat.items():
            node_set[node_set_name][feat_name] = as_tensor(feat)
      self._num_nodes_dict = graph['num_nodes_dict']
      self._node_feat_dict = node_set
      self._edge_index_dict = tf.nest.map_structure(
          as_tensor, graph['edge_index_dict'])
    else:  # Homogenous graph. Make heterogeneous.
      if graph.get('node_feat', None) is not None:
        node_features = {
            tfgnn.NODES: {'feat': as_tensor(graph['node_feat'])}
        }
      else:
        node_features = {
            tfgnn.NODES: {
                'feat': tf.zeros([graph['num_nodes'], 0], dtype=tf.float32)
            }
        }

      self._edge_index_dict = {
          (tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): as_tensor(
              graph['edge_index']),
      }
      self._num_nodes_dict = {tfgnn.NODES: graph['num_nodes']}
      self._node_feat_dict = node_features

  @property
  def num_nodes_dict(self):
    """Maps "node set name" -> number of nodes."""
    return self._num_nodes_dict

  @property
  def node_feat_dict(self):
    """Maps "node set name" to dict of "feature name"->tf.Tensor."""
    return self._node_feat_dict

  @property
  def edge_index_dict(self):
    """Adjacency lists for all edge sets.

    Returns:
      Dict (source node set name, edge set name, target node set name) -> edges.
      Where `edges` is tf.Tensor of shape (2, num edges), with `edges[0]` and
      `edges[1]`, respectively, containing source and target node IDs (as 1D int
      tf.Tensor).
    """
    return self._edge_index_dict


def _get_ogbn_dataset(dataset_name, cache_dir = None):
  """Imports ogb and returns `NodePropPredDataset`."""
  # This is done on purpose: we only import ogb if an ogb dataset is requested.
  import ogb.nodeproppred  # pylint: disable=g-import-not-at-top
  return ogb.nodeproppred.NodePropPredDataset(dataset_name, root=cache_dir)


def _get_ogbl_dataset(dataset_name, cache_dir = None):
  """Imports ogb and returns `LinkPropPredDataset`."""
  # This is done on purpose: we only import ogb if an ogb dataset is requested.
  import ogb.linkproppred  # pylint: disable=g-import-not-at-top
  return ogb.linkproppred.LinkPropPredDataset(dataset_name, root=cache_dir)


class OgbnData(NodeClassificationGraphData):
  """Wraps node classification graph data of ogbn-* for in-memory learning."""

  def __init__(self, dataset_name, cache_dir=None):
    super().__init__()
    self._dataset_name = dataset_name
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self._ogb_dataset = _get_ogbn_dataset(dataset_name, cache_dir)
    self._graph, self._node_labels, self._node_split, self._labeled_nodeset = (
        OgbnData._to_heterogeneous(self._ogb_dataset))

    # rehape from [N, 1] to [N].
    self._node_labels = self._node_labels[:, 0]

    # train labels (test set to -1).
    self._train_labels = np.copy(self._node_labels)
    self._train_labels[self._node_split.test] = -1

    self._train_labels = as_tensor(self._train_labels)
    self._node_labels = as_tensor(self._node_labels)

  @property
  def name(self):
    return self._dataset_name

  @staticmethod
  def _to_heterogeneous(
      ogb_dataset):
    """Returns heterogeneous dicts from homogeneous or heterogeneous OGB dataset.

    Args:
      ogb_dataset: OGBN dataset. It can be homogeneous (single node set type,
        single edge set type), or heterogeneous (various node/edge set types),
        and returns data structure as-if the dataset is heterogeneous (i.e.,
        names each node/edge set). If input is a homogeneous graph, then the
        node set will be named "nodes" and the edge set will be named "edges".

    Returns:
      tuple: `(ogb_graph, node_labels, idx_split, labeled_nodeset)`, where:
        `ogb_graph` is instance of _OgbGraph.
        `node_labels`: np.array of labels, with .shape[0] equals number of nodes
          in node set with name `labeled_nodeset`.
        `idx_split`: instance of NodeSplit. Members `train`, `test` and `valid`,
          respectively, contain indices of nodes in node set with name
          `labeled_nodeset`.
        `labeled_nodeset`: name of node set that the node-classification task is
          designed over.
    """
    graph, node_labels = ogb_dataset[0]
    ogb_graph = _OgbGraph(graph)
    if 'edge_index_dict' in graph:  # Graph is heterogeneous
      assert 'num_nodes_dict' in graph
      assert 'node_feat_dict' in graph
      labeled_nodeset = list(node_labels.keys())
      if len(labeled_nodeset) != 1:
        raise ValueError('Expecting OGB dataset with *one* node set with '
                         'labels. Found: ' + ', '.join(labeled_nodeset))
      labeled_nodeset = labeled_nodeset[0]

      node_labels = node_labels[labeled_nodeset]
      # idx_split is dict: {'train': {labeled_nodeset: np.array}, 'test': ...}.
      idx_split = ogb_dataset.get_idx_split()
      # Change to {'train': Tensor, 'test': Tensor, 'valid': Tensor}
      idx_split = {split_name: as_tensor(split_dict[labeled_nodeset])
                   for split_name, split_dict in idx_split.items()}
      # third-party OGB class returns dict with key 'valid'. Make consistent
      # with TF nomenclature by renaming.
      idx_split['validation'] = idx_split.pop('valid')  # Rename
      idx_split = NodeSplit(**idx_split)

      return ogb_graph, node_labels, idx_split, labeled_nodeset

    # Copy other node information.
    for key, value in graph.items():
      if key != 'node_feat' and key.startswith('node_'):
        key = key.split('node_', 1)[-1]
        ogb_graph.node_feat_dict[tfgnn.NODES][key] = as_tensor(value)  # pytype: disable=unsupported-operands  # always-use-property-annotation
    idx_split = ogb_dataset.get_idx_split()
    idx_split['validation'] = idx_split.pop('valid')  # Rename
    idx_split = NodeSplit(**tf.nest.map_structure(
        tf.convert_to_tensor, idx_split))
    return ogb_graph, node_labels, idx_split, tfgnn.NODES

  def num_classes(self):
    return self._ogb_dataset.num_classes

  def node_features_dicts_without_labels(self):
    # Deep-copy dict (*but* without copying tf.Tensor objects).
    node_sets = self._graph.node_feat_dict
    node_sets = {node_set_name: dict(node_set.items())
                 for node_set_name, node_set in node_sets.items()}
    node_counts = self.node_counts()
    for node_set_name, count in node_counts.items():
      if node_set_name not in node_sets:
        node_sets[node_set_name] = {}
      feat_dict = node_sets[node_set_name]
      feat_dict['#id'] = tf.range(count, dtype=tf.int32)
    return node_sets

  @property
  def labeled_nodeset(self):
    return self._labeled_nodeset

  def node_counts(self):
    return self._graph.num_nodes_dict

  def edge_lists(self):
    return self._graph.edge_index_dict

  def node_split(self):
    return self._node_split

  def labels(self):
    return self._train_labels

  def test_labels(self):
    """int numpy array of length num_nodes containing train and test labels."""
    return self._node_labels


def _maybe_download_file(source_url, destination_path, make_dirs=True):
  """Downloads URL `source_url` onto file `destination_path` if not present."""
  if not tf.io.gfile.exists(destination_path):
    dir_name = os.path.dirname(destination_path)
    if make_dirs:
      try:
        tf.io.gfile.makedirs(dir_name)
      except FileExistsError:
        pass

    with urllib.request.urlopen(source_url) as fin:
      with tf.io.gfile.GFile(destination_path, 'wb') as fout:
        fout.write(fin.read())


class PlanetoidGraphData(NodeClassificationGraphData):
  """Wraps Planetoid node-classificaiton datasets.

  These datasets first appeared in the Planetoid [1] paper and popularized by
  the GCN paper [2].

  [1] Yang et al, ICML'16
  [2] Kipf & Welling, ICLR'17.
  """

  def __init__(self, dataset_name, cache_dir=None):
    super().__init__()
    self._dataset_name = dataset_name
    allowed_names = ('pubmed', 'citeseer', 'cora')

    url_template = (
        'https://github.com/kimiyoung/planetoid/blob/master/data/'
        'ind.%s.%s?raw=true')
    file_parts = ['ally', 'allx', 'graph', 'ty', 'tx', 'test.index']
    if dataset_name not in allowed_names:
      raise ValueError('Dataset must be one of: ' + ', '.join(allowed_names))
    if cache_dir is None:
      cache_dir = os.environ.get(
          'PLANETOID_CACHE_DIR', os.path.expanduser(
              os.path.join('~', 'data', 'planetoid')))
    base_path = os.path.join(cache_dir, 'ind.%s' % dataset_name)
    # Download all files.
    for file_part in file_parts:
      source_url = url_template % (dataset_name, file_part)
      destination_path = os.path.join(
          cache_dir, 'ind.%s.%s' % (dataset_name, file_part))
      _maybe_download_file(source_url, destination_path)

    # Load data files.
    edge_lists = pickle.load(tf.io.gfile.GFile(base_path + '.graph', 'rb'))
    allx = PlanetoidGraphData.load_x(base_path + '.allx')
    ally = np.load(tf.io.gfile.GFile(base_path + '.ally', 'rb'),
                   allow_pickle=True)

    testx = PlanetoidGraphData.load_x(base_path + '.tx')

    # Add test
    test_idx = list(
        map(int, tf.io.gfile.GFile(
            base_path + '.test.index').read().split('\n')[:-1]))

    num_test_examples = max(test_idx) - min(test_idx) + 1
    sparse_zeros = scipy.sparse.csr_matrix((num_test_examples, allx.shape[1]),
                                           dtype='float32')

    allx = scipy.sparse.vstack((allx, sparse_zeros))
    llallx = allx.tolil()
    llallx[test_idx] = testx
    self._allx = as_tensor(np.array(llallx.todense()))

    testy = np.load(tf.io.gfile.GFile(base_path + '.ty', 'rb'),
                    allow_pickle=True)
    ally = np.pad(ally, [(0, num_test_examples), (0, 0)], mode='constant')
    ally[test_idx] = testy

    self._num_nodes = len(edge_lists)
    self._num_classes = ally.shape[1]
    self._node_labels = np.argmax(ally, axis=1)
    self._train_labels = self._node_labels + 0  # Copy.
    self._train_labels[test_idx] = -1
    self._node_labels = as_tensor(self._node_labels)
    self._train_labels = as_tensor(self._train_labels)
    self._test_idx = as_tensor(np.array(test_idx, dtype='int32'))
    self._node_split = None  # Populated on `node_split()`

    # Will be used to construct (sparse) adjacency matrix.
    adj_src = []
    adj_target = []
    for node, neighbors in edge_lists.items():
      adj_src.extend([node] * len(neighbors))
      adj_target.extend(neighbors)

    self._edge_list = as_tensor(np.stack([adj_src, adj_target], axis=0))

  @property
  def name(self):
    return self._dataset_name

  @staticmethod
  def load_x(filename):
    if sys.version_info > (3, 0):
      return pickle.load(tf.io.gfile.GFile(filename, 'rb'), encoding='latin1')
    else:
      return np.load(tf.io.gfile.GFile(filename))

  def num_classes(self):
    return self._num_classes

  def node_features_dicts_without_labels(self):
    features = {'feat': self._allx}
    features['#id'] = tf.range(self._num_nodes, dtype=tf.int32)
    return {tfgnn.NODES: features}

  def node_counts(self):
    return {tfgnn.NODES: self._num_nodes}

  def edge_lists(self):
    return {(tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): self._edge_list}

  def node_split(self):
    if self._node_split is None:
      # By default, we mimic Planetoid & GCN setup -- i.e., 20 labels per class.
      labels_per_class = int(os.environ.get('PLANETOID_LABELS_PER_CLASS', '20'))
      num_train_nodes = labels_per_class * self.num_classes()
      num_validation_nodes = 500
      train_ids = tf.range(num_train_nodes, dtype=tf.int32)
      validation_ids = tf.range(
          num_train_nodes,
          num_train_nodes + num_validation_nodes, dtype=tf.int32)
      self._node_split = NodeSplit(train=train_ids, validation=validation_ids,
                                   test=self._test_idx)
    return self._node_split

  @property
  def labeled_nodeset(self):
    return tfgnn.NODES

  def labels(self):
    return self._train_labels

  def test_labels(self):
    """int numpy array of length num_nodes containing train and test labels."""
    return self._node_labels


class LinkPredictionGraphData(InMemoryGraphData):
  """Superclasses must wrap dataset of graph(s) for link-prediction tasks."""

  @abc.abstractmethod
  def edge_split(self):
    """Returns edge endpoints for {train, test, valid} partitions."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_edgeset(self):
    """Name of edge set over which link prediction is defined."""
    raise NotImplementedError()

  @property
  def source_node_set_name(self):
    """Node set name of source node of (task) target_edgeset."""
    return self.graph_schema().edge_sets[self.target_edgeset].source

  @property
  def target_node_set_name(self):
    """Node set name of target node of (task) target_edgeset."""
    return self.graph_schema().edge_sets[self.target_edgeset].target

  @property
  def num_source_nodes(self):
    """Number of nodes in the source endpoint of (task) target_edgeset."""
    return self.node_counts()[self.source_node_set_name]

  @property
  def num_target_nodes(self):
    """Number of nodes in the target endpoint of (task) target_edgeset."""
    return self.node_counts()[self.target_node_set_name]


class OgblData(LinkPredictionGraphData):
  """Wraps link prediction datasets of ogbl-* for in-memory learning."""

  def __init__(self, dataset_name, cache_dir = None):
    super().__init__()
    self._dataset_name = dataset_name
    if cache_dir is None:
      cache_dir = os.environ.get(
          'OGB_CACHE_DIR', os.path.expanduser(os.path.join('~', 'data', 'ogb')))

    self._ogb_dataset = _get_ogbl_dataset(dataset_name, cache_dir)

    ogb_edge_dict = self._ogb_dataset.get_edge_split()
    self._edge_split = EdgeSplit(
        train_edges=as_tensor(ogb_edge_dict['train']['edge']),
        validation_edges=as_tensor(ogb_edge_dict['train']['edge']),
        test_edges=as_tensor(ogb_edge_dict['test']['edge']),
        negative_validation_edges=as_tensor(ogb_edge_dict['valid']['edge_neg']),
        negative_test_edges=as_tensor(ogb_edge_dict['test']['edge_neg']))

    self._ogb_graph = _OgbGraph(self._ogb_dataset.graph)

  @property
  def name(self):
    return self._dataset_name

  def node_features_dicts(self, add_id = True):
    features = self._ogb_graph.node_feat_dict
    # 2-level dict shallow copy. Inner value stores reference to tf.Tensor,
    features = {node_set_name: copy.copy(features)
                for node_set_name, features in features.items()}
    if add_id:
      counts = self.node_counts()
      for node_set_name, feats in features.items():
        feats['#id'] = tf.range(counts[node_set_name], dtype=tf.int32)  # pytype: disable=unsupported-operands  # always-use-property-annotation
    return features

  def node_counts(self):
    return dict(self._ogb_graph.num_nodes_dict)  # Return copy.

  def edge_lists(self):
    return dict(self._ogb_graph.edge_index_dict)  # Return shallow copy.

  def edge_split(self):
    return self._edge_split

  @property
  def target_edgeset(self):
    return tfgnn.EDGES


def get_in_memory_graph_data(dataset_name):
  if dataset_name.startswith('ogbn-'):
    return OgbnData(dataset_name)
  elif dataset_name.startswith('ogbl-'):
    return OgblData(dataset_name)
  elif dataset_name in ('cora', 'citeseer', 'pubmed'):
    return PlanetoidGraphData(dataset_name)
  else:
    raise ValueError('Unknown Dataset name: ' + dataset_name)


# Shorthand. Can be replaced with: `as_tensor = tf.convert_to_tensor`.
def as_tensor(obj):
  """short-hand for tf.convert_to_tensor."""
  return tf.convert_to_tensor(obj)


def load_ogbn_graph_tensor(
    dataset_path, *, add_reverse_edge_sets = False
):
  """Load OGBN graph data as a graph tensor from numpy compressed (.npz) files.

  To generate the .npz files from the original OGB dataset, please refer to
  tensorflow_gnn/converters/ogb/convert_ogb_to_npz.py

  Args:
    dataset_path: Path to the saved OGBN numpy compressed (.npz) files.
    add_reverse_edge_sets: Flag to determine whether to add reversed edge sets.

  Returns:
    A tfgnn.GraphTensor comprising of the full OGBN graph loaded in-memory.
  """
  graph_data = NodeClassificationGraphData.load(dataset_path)
  graph_data = graph_data.with_labels_as_features(True)
  if add_reverse_edge_sets:
    graph_data = graph_data.with_reverse_edge_sets()
  graph_tensor = graph_data.as_graph_tensor()
  return graph_tensor
