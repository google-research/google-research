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

"""Interfaces for reading raw graph datasets."""

import abc
import json
import os
from typing import Set

from absl import logging
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing
import tensorflow as tf



class Dataset(abc.ABC):
  """Abstract base class for datasets."""

  senders: np.ndarray
  receivers: np.ndarray
  node_features: np.ndarray
  node_labels: np.ndarray
  train_nodes: np.ndarray
  validation_nodes: np.ndarray
  test_nodes: np.ndarray

  def num_nodes(self):
    """Returns the number of nodes in the dataset."""
    return len(self.node_labels)

  def num_edges(self):
    """Returns the number of edges in the dataset."""
    return len(self.senders)


class DummyDataset(Dataset):
  """A dummy dataset for testing."""

  NUM_DUMMY_TRAINING_SAMPLES: int = 3
  NUM_DUMMY_VALIDATION_SAMPLES: int = 3
  NUM_DUMMY_TEST_SAMPLES: int = 3
  NUM_DUMMY_FEATURES: int = 5
  NUM_DUMMY_CLASSES: int = 3

  def __init__(self):
    num_samples = (
        DummyDataset.NUM_DUMMY_TRAINING_SAMPLES +
        DummyDataset.NUM_DUMMY_VALIDATION_SAMPLES +
        DummyDataset.NUM_DUMMY_TEST_SAMPLES)
    self.senders = np.arange(num_samples)
    self.receivers = np.roll(np.arange(num_samples), -1)
    self.node_features = np.repeat(
        np.arange(num_samples), DummyDataset.NUM_DUMMY_FEATURES)
    self.node_features = self.node_features.reshape(
        (num_samples, DummyDataset.NUM_DUMMY_FEATURES))
    self.node_labels = np.zeros(num_samples)
    self.train_nodes = np.arange(
        DummyDataset.NUM_DUMMY_TRAINING_SAMPLES)
    self.validation_nodes = np.arange(
        DummyDataset.NUM_DUMMY_TRAINING_SAMPLES,
        DummyDataset.NUM_DUMMY_TRAINING_SAMPLES +
        DummyDataset.NUM_DUMMY_VALIDATION_SAMPLES)
    self.test_nodes = np.arange(
        DummyDataset.NUM_DUMMY_TRAINING_SAMPLES +
        DummyDataset.NUM_DUMMY_VALIDATION_SAMPLES, num_samples)


class OGBTransductiveDataset(Dataset):
  """Reads Open Graph Benchmark (OGB) datasets."""

  def __init__(self, dataset_name, dataset_path):
    super(OGBTransductiveDataset, self).__init__()
    self.name = dataset_name.replace('-disjoint', '').replace('-', '_')
    base_path = os.path.join(dataset_path, self.name)

    if self.name == 'ogbn_arxiv':
      split_property = 'split/time/'
    elif self.name == 'ogbn_mag':
      split_property = 'split/time/paper/'
    elif self.name == 'ogbn_products':
      split_property = 'split/sales_ranking/'
    elif self.name == 'ogbn_proteins':
      split_property = 'split/species/'
    else:
      raise ValueError('Unsupported dataset.')

    train_split_file = os.path.join(
        base_path, split_property, 'train.csv.gz')
    validation_split_file = os.path.join(
        base_path, split_property, 'valid.csv.gz')
    test_split_file = os.path.join(
        base_path, split_property, 'test.csv.gz')

    if self.name == 'ogbn_mag':
      node_feature_file = os.path.join(base_path,
                                       'raw/node-feat/paper/node-feat.csv.gz')
      node_label_file = os.path.join(base_path,
                                     'raw/node-label/paper/node-label.csv.gz')
    else:
      node_feature_file = os.path.join(base_path, 'raw/node-feat.csv.gz')
      node_label_file = os.path.join(base_path, 'raw/node-label.csv.gz')

    logging.info('Reading node features...')
    self.node_features = pd.read_csv(
        node_feature_file, header=None).values.astype(np.float32)
    logging.info('Node features loaded.')

    logging.info('Reading node labels...')
    self.node_labels = pd.read_csv(
        node_label_file, header=None).values.astype(np.int64).squeeze()
    logging.info('Node labels loaded.')

    if self.name == 'ogbn_mag':
      edge_file = os.path.join(
          base_path, 'raw/relations/paper___cites___paper/edge.csv.gz')
    else:
      edge_file = os.path.join(base_path, 'raw/edge.csv.gz')

    logging.info('Reading edges...')
    senders_receivers = pd.read_csv(
        edge_file, header=None).values.T.astype(np.int64)
    self.senders, self.receivers = senders_receivers
    logging.info('Edges loaded.')

    logging.info('Reading train, validation and test splits...')
    self.train_nodes = pd.read_csv(
        train_split_file, header=None).values.T.astype(np.int64).squeeze()
    self.validation_nodes = pd.read_csv(
        validation_split_file, header=None).values.T.astype(np.int64).squeeze()
    self.test_nodes = pd.read_csv(
        test_split_file, header=None).values.T.astype(np.int64).squeeze()
    logging.info('Loaded train, test and validation splits.')


class OGBDisjointDataset(OGBTransductiveDataset):
  """A disjoint version of a OGB dataset, with no inter-split edges."""

  def __init__(self, dataset_name, dataset_path):
    super(OGBDisjointDataset, self).__init__(dataset_name, dataset_path)
    self.name = dataset_name

    train_split = set(self.train_nodes.flat)
    validation_split = set(self.validation_nodes.flat)
    test_split = set(self.test_nodes.flat)
    splits = [train_split, validation_split, test_split]

    def _compute_split_index(elem):
      elem_index = None
      for index, split in enumerate(splits):
        if elem in split:
          if elem_index is not None:
            raise ValueError(f'Node {elem} present in multiple splits.')
          elem_index = index
      if elem_index is None:
        raise ValueError(f'Node {elem} present in none of the splits.')
      return elem_index

    senders_split_indices = np.vectorize(_compute_split_index)(self.senders)
    receivers_split_indices = np.vectorize(_compute_split_index)(self.receivers)
    in_same_split = (senders_split_indices == receivers_split_indices)

    self.senders = self.senders[in_same_split]
    self.receivers = self.receivers[in_same_split]




class GraphSAINTTransductiveDataset(Dataset):
  """Reads a GraphSAINT-format transductive dataset."""

  def __init__(self, dataset_name, dataset_path):
    super(GraphSAINTTransductiveDataset, self).__init__()

    self.name = dataset_name
    base_name = dataset_name.replace('-disjoint', '')
    base_name = base_name.replace('-transductive', '')


    self.base_name = base_name
    base_path = os.path.join(dataset_path, base_name)

    logging.info('Reading graph data...')
    self.adj_full = sp.load_npz(
        tf.io.gfile.GFile(os.path.join(base_path, 'adj_full.npz'), 'rb'))
    graph = nx.from_scipy_sparse_matrix(self.adj_full)
    graph_data = nx.readwrite.node_link_data(graph)
    logging.info('Graph data loaded.')

    self.senders = [e[0] for e in graph.edges]
    self.receivers = [e[1] for e in graph.edges]

    train_nodes = []
    validation_nodes = []
    test_nodes = []

    splits = json.load(
        tf.io.gfile.GFile(os.path.join(base_path, 'role.json'), 'r'))
    train_split = set(splits['tr'])
    validation_split = set(splits['va'])
    test_split = set(splits['te'])

    for node in graph_data['nodes']:
      node_id = node['id']
      if node_id in validation_split:
        validation_nodes.append(node_id)
      elif node_id in test_split:
        test_nodes.append(node_id)
      elif node_id in train_split:
        train_nodes.append(node_id)
      else:
        raise ValueError(f'Node {node_id} not present in any split.')

    self.train_nodes = np.asarray(train_nodes)
    self.validation_nodes = np.asarray(validation_nodes)
    self.test_nodes = np.asarray(test_nodes)

    logging.info('Reading node features...')
    node_features = np.load(
        tf.io.gfile.GFile(os.path.join(base_path, 'feats.npy'), 'rb'))
    logging.info('Node features loaded.')

    logging.info('Preprocessing node features...')
    train_node_features = node_features[self.train_nodes]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_node_features)
    self.node_features = scaler.transform(node_features)
    logging.info('Node features preprocessed.')

    logging.info('Reading node labels...')
    class_map = json.load(
        tf.io.gfile.GFile(os.path.join(base_path, 'class_map.json'), 'r'))
    labels = [class_map[node_id] for node_id in sorted(class_map)]
    self.node_labels = np.asarray(labels).squeeze()
    logging.info('Node labels loaded.')


class GraphSAINTDisjointDataset(GraphSAINTTransductiveDataset):
  """Reads a GraphSAINT-format disjoint dataset."""

  def __init__(self, dataset_name, dataset_path):
    super(GraphSAINTDisjointDataset, self).__init__(dataset_name, dataset_path)

    self.name = dataset_name

    train_split = set(self.train_nodes)
    validation_split = set(self.validation_nodes)
    test_split = set(self.test_nodes)

    graph_train = _get_graph_for_split(self.adj_full, train_split)
    graph_validation = _get_graph_for_split(self.adj_full, validation_split)
    graph_test = _get_graph_for_split(self.adj_full, test_split)
    graph = nx.union_all((graph_train, graph_validation, graph_test))

    self.senders = [e[0] for e in graph.edges]
    self.receivers = [e[1] for e in graph.edges]


def _get_graph_for_split(adj_full,
                         split_set):
  """Returns the induced subgraph for the required split."""
  def edge_generator():
    senders, receivers = adj_full.nonzero()
    for sender, receiver in zip(senders, receivers):
      if sender in split_set and receiver in split_set:
        yield sender, receiver

  graph_split = nx.Graph()
  graph_split.add_nodes_from(split_set)
  graph_split.add_edges_from(edge_generator())
  return graph_split


def get_dataset(dataset_name, dataset_path):
  """Returns a graph dataset."""
  special_dataset_fns = {
      'dummy': DummyDataset,
  }
  if dataset_name in special_dataset_fns:
    return special_dataset_fns[dataset_name]()

  if dataset_name.startswith('ogb'):
    if dataset_name.endswith('disjoint'):
      return OGBDisjointDataset(dataset_name, dataset_path)
    return OGBTransductiveDataset(dataset_name, dataset_path)

  graphsaint_datasets = ['reddit', 'yelp', 'flickr']
  if any(dataset_name.startswith(name) for name in graphsaint_datasets):
    if dataset_name.endswith('disjoint'):
      return GraphSAINTDisjointDataset(dataset_name, dataset_path)
    if dataset_name.endswith('transductive'):
      return GraphSAINTTransductiveDataset(dataset_name, dataset_path)
    raise ValueError(
        'Please prefix dataset_name with `transductive` or `disjoint`.')

  raise ValueError(f'Unsupported dataset: {dataset_name}.')
