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

"""Dataset definitions for in-memory graph structure learning."""
import copy
import io
import math
import os
import random
from typing import List, Mapping, MutableMapping, Tuple

import numpy as np
import scipy.sparse
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_hub as tfhub

from ugsl import tfgnn_datasets


class GSLGraphData:
  """Wraps graph datasets to be used for graph structure learning.

  GSLGraphData can take a given tensor as a generated adjacency and incorporate
  it in the graph tensow.
  """

  def __init__(
      self,
      remove_noise_ratio = 0.0,
      add_noise_ratio = 0.0,
  ):
    super().__init__()
    # Saving the generated noisy adjacency to reuse.
    self._cached_noisy_adjacency = None
    self._input_gt = self.as_graph_tensor_noisy_adjacency(
        remove_noise_ratio=remove_noise_ratio, add_noise_ratio=add_noise_ratio
    )

  def node_sets(self):
    raise NotImplementedError

  def splits(self):
    return copy.copy(self._splits)

  def num_classes(self):
    raise NotImplementedError('num_classes')

  def node_split(self):
    raise NotImplementedError()

  def labels(self):
    raise NotImplementedError()

  def test_labels(self):
    raise NotImplementedError()

  @property
  def labeled_nodeset(self):
    raise NotImplementedError()

  def node_features_dicts_without_labels(
      self,
  ):
    raise NotImplementedError()

  def edge_lists(
      self,
  ):
    raise NotImplementedError()

  def as_graph_tensor(self):
    raise NotImplementedError()

  def node_features_dicts(
      self,
  ):
    raise NotImplementedError()

  def get_input_graph_tensor(self):
    return self._input_gt

  def as_graph_tensor_given_adjacency(
      self,
      adjacency_tensor,
      edge_weights,
      node_features,
      make_undirected = False,
      add_self_loops = False,
  ):
    """Returns `GraphTensor` holding the entire graph."""
    return tfgnn.GraphTensor.from_pieces(
        node_sets=self.node_sets_given_features(node_features),
        edge_sets=self.edge_sets_given_adjacency(
            adjacency_tensor,
            edge_weights,
            make_undirected,
            add_self_loops,
        ),
        context=self.context(),
    )

  def node_sets_given_features(
      self, node_features
  ):
    """Returns node sets of entire graph (dict: node set name -> NodeSet)."""
    node_counts = self.node_counts()
    features_dicts = self.node_features_dicts()
    node_set_names = set(node_counts.keys()).union(features_dicts.keys())
    return {
        name: tfgnn.NodeSet.from_fields(
            sizes=tf.convert_to_tensor([node_counts[name]]),
            features={'feat': node_features})
        for name in node_set_names
    }

  def edge_sets_given_adjacency(
      self,
      edge_list,
      edge_weights,
      make_undirected = False,
      add_self_loops = False,
  ):
    """Returns edge sets of entire graph (dict: edge set name -> EdgeSet)."""
    if make_undirected:
      edge_list = tf.concat([edge_list, edge_list[::-1]], axis=-1)
      edge_weights = tf.concat([edge_weights, edge_weights[::-1]], axis=-1)
    if add_self_loops:
      node_counts = self.node_counts()
      all_nodes = tf.range(node_counts[tfgnn.NODES], dtype=edge_list.dtype)
      self_connections = tf.stack([all_nodes, all_nodes], axis=0)
      # The following line adds self_connections to the existing edges.
      # It is possible for an edge to be both available in the edge_list and
      # also in the self_connections.
      edge_list = tf.concat([edge_list, self_connections], axis=-1)
      edge_weights = tf.concat(
          [edge_weights, tf.ones(node_counts[tfgnn.NODES])], axis=-1
      )
    return {
        tfgnn.EDGES: tfgnn.EdgeSet.from_fields(
            sizes=tf.shape(edge_list)[1:2],
            adjacency=tfgnn.Adjacency.from_indices(
                source=(tfgnn.NODES, edge_list[0]),
                target=(tfgnn.NODES, edge_list[1]),
            ),
            features={'weights': edge_weights},
        )
    }

  def as_graph_tensor_noisy_adjacency(
      self,
      remove_noise_ratio,
      add_noise_ratio,
      make_undirected = False,
      add_self_loops = False,
  ):
    """Returns `GraphTensor` holding the entire graph."""
    return tfgnn.GraphTensor.from_pieces(
        node_sets=self.node_sets(),
        edge_sets=self.edge_sets_noisy_adjacency(
            add_noise_ratio=add_noise_ratio,
            remove_noise_ratio=remove_noise_ratio,
            make_undirected=make_undirected,
            add_self_loops=add_self_loops,
        ),
        context=self.context(),
    )

  def edge_sets_noisy_adjacency(
      self,
      add_noise_ratio,
      remove_noise_ratio,
      make_undirected = False,
      add_self_loops = False,
  ):
    """Returns noisy edge sets of entire graph (dict: edge set name -> EdgeSet)."""
    if self._cached_noisy_adjacency:
      return self._cached_noisy_adjacency
    edge_sets = {}
    node_counts = self.node_counts()
    for edge_type, edge_list in self.edge_lists().items():
      (source_node_set_name, edge_set_name, target_node_set_name) = edge_type
      number_of_nodes = node_counts[source_node_set_name]
      sources = edge_list[0].numpy()
      targets = edge_list[1].numpy()
      number_of_edges = len(sources)
      if add_noise_ratio:
        number_of_edges_to_add = math.floor(
            ((number_of_nodes * number_of_nodes) / 2 - number_of_edges)
            * add_noise_ratio
        )
        sources_to_add = np.array(
            random.choices(range(number_of_nodes), k=number_of_edges_to_add)
        )
        targets_to_add = np.array(
            random.choices(range(number_of_nodes), k=number_of_edges_to_add)
        )
      else:
        sources_to_add, targets_to_add = np.array([]), np.array([])
      if remove_noise_ratio:
        number_of_edges_to_remove = math.floor(
            number_of_edges * remove_noise_ratio
        )
        edge_indices_to_remove = random.sample(
            range(0, number_of_edges), number_of_edges_to_remove
        )
        noisy_sources = np.delete(sources, edge_indices_to_remove)
        noisy_targets = np.delete(targets, edge_indices_to_remove)
      else:
        noisy_sources, noisy_targets = sources, targets
      noisy_sources = tf.constant(
          np.concatenate((noisy_sources, sources_to_add)), dtype=tf.int32
      )
      noisy_targets = tf.constant(
          np.concatenate((noisy_targets, targets_to_add)), dtype=tf.int32
      )
      edge_list = tf.stack([noisy_sources, noisy_targets])
      if make_undirected:
        edge_list = tf.concat([edge_list, edge_list[::-1]], axis=-1)
      if add_self_loops:
        all_nodes = tf.range(number_of_nodes, dtype=edge_list.dtype)
        self_connections = tf.stack([all_nodes, all_nodes], axis=0)
        edge_list = tf.concat([edge_list, self_connections], axis=-1)
      edge_sets[edge_set_name] = tfgnn.EdgeSet.from_fields(
          sizes=tf.shape(edge_list)[1:2],
          adjacency=tfgnn.Adjacency.from_indices(
              source=(source_node_set_name, edge_list[0]),
              target=(target_node_set_name, edge_list[1]),
          ),
      )
    self._cached_noisy_adjacency = edge_sets
    return edge_sets


class GSLPlanetoidGraphData(tfgnn_datasets.PlanetoidGraphData, GSLGraphData):
  """Wraps Planetoid graph datasets to be used for graph structure learning.

  Besides the initial input adjacency matrix, GSLGraphData can take a given
  tensor as a generated adjacency and incorporate it in the graph tensow.
  """

  def __init__(
      self,
      dataset_name,
      remove_noise_ratio,
      add_noise_ratio,
  ):
    tfgnn_datasets.PlanetoidGraphData.__init__(self, dataset_name)
    GSLGraphData.__init__(
        self,
        remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio,
    )


class GcnBenchmarkFileGraphData(tfgnn_datasets.NodeClassificationGraphData):
  """Adapt npz with format of github.com/shchur/gnn-benchmark into TF-GNN.

  NOTE: This can be moved to TF-GNN (tfgnn/experimental/in_memory/datasets.py).
  """

  def __init__(self, dataset_path):
    """Loads .npz file following shchur's format."""
    if not tf.io.gfile.exists(dataset_path):
      raise ValueError('Dataset file not found: ' + dataset_path)

    adj_matrix, attr_matrix, labels, label_mask = _load_npz_to_sparse_graph(
        dataset_path)
    del label_mask

    edge_indices = tf.convert_to_tensor(adj_matrix.nonzero())
    self._edge_lists = {(tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): edge_indices}

    num_nodes = attr_matrix.shape[0]
    self._node_features_dicts = {
        tfgnn.NODES: {
            'feat': tf.convert_to_tensor(attr_matrix),
            '#id': tf.range(num_nodes),
        }
    }
    self._node_counts = {tfgnn.NODES: num_nodes}
    self._num_classes = labels.max() + 1
    self._test_labels = tf.convert_to_tensor(labels)

    permutation = np.random.default_rng(seed=1234).permutation(num_nodes)
    num_train_examples = num_nodes // 10
    num_validate_examples = num_nodes // 10
    train_indices = permutation[:num_train_examples]
    num_validate_plus_train = num_validate_examples + num_train_examples
    validate_indices = permutation[num_train_examples:num_validate_plus_train]
    test_indices = permutation[num_validate_plus_train:]

    self._node_split = tfgnn_datasets.NodeSplit(
        tf.convert_to_tensor(train_indices),
        tf.convert_to_tensor(validate_indices),
        tf.convert_to_tensor(test_indices))

    self._train_labels = labels + 0  # Make a copy.
    self._train_labels[test_indices] = -1
    self._train_labels = tf.convert_to_tensor(self._train_labels)
    super().__init__()

  def node_counts(self):
    return self._node_counts

  def edge_lists(self):
    return self._edge_lists

  def num_classes(self):
    return self._num_classes

  def node_split(self):
    return self._node_split

  def labels(self):
    return self._train_labels

  def test_labels(self):
    return self._test_labels

  @property
  def labeled_nodeset(self):
    return tfgnn.NODES

  def node_features_dicts_without_labels(self):
    return self._node_features_dicts


_maybe_download_file = tfgnn_datasets._maybe_download_file  # pylint: disable=protected-access


class GcnBenchmarkUrlGraphData(GcnBenchmarkFileGraphData):

  def __init__(
      self, npz_url,
      cache_dir = os.path.expanduser(
          os.path.join('~', 'data', 'gnn-benchmark'))):
    destination_url = os.path.join(cache_dir, os.path.basename(npz_url))
    _maybe_download_file(npz_url, destination_url)
    super().__init__(destination_url)


def _load_npz_to_sparse_graph(file_name):
  """Copied from experimental/users/tsitsulin/gcns/cgcn/utilities/graph.py."""
  file_bytes = tf.io.gfile.GFile(file_name, 'rb').read()
  bytes_io = io.BytesIO(file_bytes)
  with np.load(bytes_io, allow_pickle=True) as fin:
    loader = dict(fin)
    adj_matrix = scipy.sparse.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    if 'attr_data' in loader:
      # Attributes are stored as a sparse CSR matrix
      attr_matrix = scipy.sparse.csr_matrix(
          (loader['attr_data'], loader['attr_indices'],
           loader['attr_indptr']),
          shape=loader['attr_shape']).todense()
    elif 'attr_matrix' in loader:
      # Attributes are stored as a (dense) np.ndarray
      attr_matrix = loader['attr_matrix']
    else:
      raise ValueError('No attributes in the data file: ' + file_name)

    if 'labels_data' in loader:
      # Labels are stored as a CSR matrix
      labels = scipy.sparse.csr_matrix(
          (loader['labels_data'], loader['labels_indices'],
           loader['labels_indptr']),
          shape=loader['labels_shape'])
      label_mask = labels.nonzero()[0]
      labels = labels.nonzero()[1]
    elif 'labels' in loader:
      # Labels are stored as a numpy array
      labels = loader['labels']
      label_mask = np.ones(labels.shape, dtype=np.bool_)
    else:
      raise ValueError('No labels in the data file: ' + file_name)

  return adj_matrix, attr_matrix, labels, label_mask


class GSLAmazonPhotosGraphData(GcnBenchmarkUrlGraphData, GSLGraphData):
  """Wraps GCN Benchmark datasets to be used for graph structure learning."""

  def __init__(
      self,
      dataset_name,
      remove_noise_ratio,
      add_noise_ratio,
  ):
    GcnBenchmarkUrlGraphData.__init__(
        self,
        'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'
        'amazon_electronics_photo.npz')
    GSLGraphData.__init__(
        self,
        remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio,
    )


class StackOverflowGraphlessData(tfgnn_datasets.NodeClassificationGraphData):
  """Stackoverflow dataset contains node features and labels (but no edges)."""

  def __init__(
      self, cache_dir = os.path.expanduser(
          os.path.join('~', 'data', 'stackoverflow-bert'))):
    labels_path = os.path.join(cache_dir, 'labels.npy')
    embeddings_path = os.path.join(cache_dir, 'embeddings.npy')

    if (not tf.io.gfile.exists(labels_path) or
        not tf.io.gfile.exists(embeddings_path)):
      if not tf.io.gfile.exists(cache_dir):
        tf.io.gfile.makedirs(cache_dir)
      # Download.
      self._download_dataset_extract_features(labels_path, embeddings_path)

    node_features = np.load(tf.io.gfile.GFile(embeddings_path, 'rb'))
    node_labels = np.load(tf.io.gfile.GFile(labels_path, 'rb'))
    num_nodes = node_features.shape[0]
    self._node_counts = {tfgnn.NODES: num_nodes}
    self._num_classes = node_labels.max() + 1
    self._test_labels = tf.convert_to_tensor(node_labels)
    self._edge_lists = {
        (tfgnn.NODES, tfgnn.EDGES, tfgnn.NODES): (
            tf.zeros(shape=[2, 0], dtype=tf.int32))}
    self._node_features_dicts = {
        tfgnn.NODES: {
            'feat': tf.convert_to_tensor(node_features, dtype=tf.float32),
            '#id': tf.range(num_nodes),
        }
    }
    permutation = np.random.default_rng(seed=1234).permutation(num_nodes)
    num_train_examples = num_nodes // 10
    num_validate_examples = num_nodes // 10
    train_indices = permutation[:num_train_examples]
    num_validate_plus_train = num_validate_examples + num_train_examples
    validate_indices = permutation[num_train_examples:num_validate_plus_train]
    test_indices = permutation[num_validate_plus_train:]

    self._node_split = tfgnn_datasets.NodeSplit(
        tf.convert_to_tensor(train_indices),
        tf.convert_to_tensor(validate_indices),
        tf.convert_to_tensor(test_indices))

    self._train_labels = node_labels + 0  # Make a copy.
    self._train_labels[test_indices] = -1
    self._train_labels = tf.convert_to_tensor(self._train_labels)
    super().__init__()

  def node_counts(self):
    return self._node_counts

  def edge_lists(self):
    return self._edge_lists

  def num_classes(self):
    return self._num_classes

  def node_split(self):
    return self._node_split

  def labels(self):
    return self._train_labels

  def test_labels(self):
    return self._test_labels

  @property
  def labeled_nodeset(self):
    return tfgnn.NODES

  def node_features_dicts_without_labels(self):
    return self._node_features_dicts

  def _download_dataset_extract_features(
      self, labels_path, embeddings_path):
    cache_dir = os.path.dirname(labels_path)
    url = ('https://raw.githubusercontent.com/rashadulrakib/'
           'short-text-clustering-enhancement/master/data/stackoverflow/'
           'traintest')
    tab_separated_filepath = os.path.join(cache_dir, 'traintest.tsv')
    _maybe_download_file(url, tab_separated_filepath)

    data_cluster = {}
    with tf.io.gfile.GFile(tab_separated_filepath, 'r') as f:
      for line in f:
        l1, l2, text = line.strip().split('\t')
        data_cluster[text] = (int(l1), int(l2))

    def remove_cls_sep(masks):
      last_1s = np.sum(masks, axis=1) - 1
      for i in range(masks.shape[0]):
        masks[i][0] = 0
        masks[i][last_1s[i]] = 0
      return masks

    def bert_embs(texts):
      text_preprocessed = bert_preprocess_model(texts)
      bert_results = bert_model(text_preprocessed)
      masks = np.expand_dims(
          remove_cls_sep(text_preprocessed['input_mask'].numpy()), axis=2)
      emb = (np.sum(bert_results['sequence_output'].numpy() * masks, axis=1)
             / np.sum(masks, axis=1))
      return emb

    # Instantiate BERT model.
    bert_preprocess_model = tfhub.KerasLayer(
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_model = tfhub.KerasLayer(
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')

    # Map keys of `cluster` through `bert_model``
    data_cluster_keys = list(data_cluster.keys())
    embeddings = []
    for i in range(0, len(data_cluster_keys), 100):
      embeddings.append(bert_embs(data_cluster_keys[i:i+100]))
    embeddings = np.vstack(embeddings)
    labels = np.array([data_cluster[t][1] for t in data_cluster_keys])

    with tf.io.gfile.GFile(labels_path, 'wb') as fout:
      np.save(fout, labels)

    with tf.io.gfile.GFile(embeddings_path, 'wb') as fout:
      np.save(fout, embeddings)


class GSLStackOverflowGraphlessData(StackOverflowGraphlessData, GSLGraphData):
  """Wraps Stackoverflow datasets to be used for graph structure learning."""

  def __init__(
      self,
      remove_noise_ratio,
      add_noise_ratio,
      cache_dir = os.path.expanduser(
          os.path.join('~', 'data', 'stackoverflow-bert'))):
    StackOverflowGraphlessData.__init__(self, cache_dir=cache_dir)
    GSLGraphData.__init__(
        self, remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio)


def get_in_memory_graph_data(
    dataset_name,
    remove_noise_ratio,
    add_noise_ratio,
):
  """Getting the dataset based on the name.

  Args:
    dataset_name: the name of the dataset to prepare.
    remove_noise_ratio: ratio of the existing edge to remove.
    add_noise_ratio: ratio of the non-existing edges to add.

  Returns:
    The graph data to be used in training.
  Raises:
    ValueError: if the name of the dataset is not defined.
  """
  if dataset_name in ('cora', 'citeseer', 'pubmed'):
    return GSLPlanetoidGraphData(
        dataset_name,
        remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio,
    )
  elif dataset_name == 'amazon_photos':
    return GSLAmazonPhotosGraphData(
        dataset_name,
        remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio,
    )
  elif dataset_name == 'stackoverflow':
    return GSLStackOverflowGraphlessData(
        remove_noise_ratio=remove_noise_ratio,
        add_noise_ratio=add_noise_ratio,
    )
  else:
    raise ValueError('Unknown Dataset name: ' + dataset_name)
