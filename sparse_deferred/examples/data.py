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

"""Functions to read Planetoid dataset.

The top level call `get_planetoid_dataset` allows for a selection between
`pubmed`, `citeseer`, and `cora`. This code is adapted from
`learning/gnn/in_memory/datasets.py`
"""

import os
import pickle
import tarfile
from typing import IO, Union
import urllib.request
import zipfile

import numpy as np
import scipy.sparse

from sparse_deferred.implicit import matrix
from sparse_deferred.structs import graph_struct


ComputeEngine = matrix.ComputeEngine
GraphStruct = graph_struct.GraphStruct
InMemoryDB = graph_struct.InMemoryDB


def _file_exists(path):
  return os.path.exists(path)


def _make_dirs(path):
  return os.makedirs(path)


def open_file(path, mode = 'r'):
  return open(path, mode)


def _maybe_download_file(source_url, destination_path, make_dirs=True):
  """Downloads URL `source_url` onto file `destination_path` if not present."""
  if not _file_exists(destination_path):
    dir_name = os.path.dirname(destination_path)
    if make_dirs:
      try:
        _make_dirs(dir_name)
      except FileExistsError:
        pass

    with urllib.request.urlopen(source_url) as fin:
      with open_file(destination_path, 'wb') as fout:
        fout.write(fin.read())


def get_planetoid_dataset(dataset_name, cache_dir=None):
  """Returns allx, edge_list, node_labels, test_idx."""
  allowed_names = ('pubmed', 'citeseer', 'cora')

  url_template = (
      'https://github.com/kimiyoung/planetoid/blob/master/data/'
      'ind.%s.%s?raw=true'
  )
  file_parts = ['ally', 'allx', 'graph', 'ty', 'tx', 'test.index']
  if dataset_name not in allowed_names:
    raise ValueError('Dataset must be one of: ' + ', '.join(allowed_names))
  if cache_dir is None:
    cache_dir = os.environ.get(
        'PLANETOID_CACHE_DIR',
        os.path.expanduser(os.path.join('~', 'data', 'planetoid')),
    )
  base_path = os.path.join(cache_dir, 'ind.%s' % dataset_name)
  # Download all files.
  for file_part in file_parts:
    source_url = url_template % (dataset_name, file_part)
    destination_path = os.path.join(
        cache_dir, 'ind.%s.%s' % (dataset_name, file_part)
    )
    _maybe_download_file(source_url, destination_path)

  # Load data files.
  edge_lists = pickle.load(open_file(base_path + '.graph', 'rb'))
  allx = pickle.load(open_file(base_path + '.allx', 'rb'), encoding='latin1')
  ally = np.load(open_file(base_path + '.ally', 'rb'), allow_pickle=True)

  testx = pickle.load(open_file(base_path + '.tx', 'rb'), encoding='latin1')

  # Add test
  test_idx = list(
      map(
          int,
          open_file(base_path + '.test.index').read().split('\n')[:-1],
      )
  )

  num_test_examples = max(test_idx) - min(test_idx) + 1
  sparse_zeros = scipy.sparse.csr_matrix(
      (num_test_examples, allx.shape[1]), dtype='float32'
  )

  allx = scipy.sparse.vstack((allx, sparse_zeros))
  llallx = allx.tolil()
  llallx[test_idx] = testx
  allx = np.array(llallx.todense())

  testy = np.load(open_file(base_path + '.ty', 'rb'), allow_pickle=True)
  ally = np.pad(ally, [(0, num_test_examples), (0, 0)], mode='constant')
  ally[test_idx] = testy

  node_labels = np.argmax(ally, axis=1)

  test_idx = np.array(test_idx, dtype='int32')

  # Will be used to construct (sparse) adjacency matrix.
  adj_src = []
  adj_target = []
  for node, neighbors in edge_lists.items():
    adj_src.extend([node] * len(neighbors))
    adj_target.extend(neighbors)

  edge_list = np.stack([adj_src, adj_target], axis=0)

  return allx, edge_list, node_labels, test_idx


def get_malnet_tiny_dataset(
    engine,
    cache_dir = None,
    add_features = None,
    constant_value = 1,
):
  """Returns train, val, and test InMemoryDBs for the MalNet Tiny dataset."""
  download_url = (
      'http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz'
  )
  split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'

  if cache_dir is None:
    cache_dir = os.environ.get(
        'MALNET_TINY_CACHE_DIR',
        os.path.expanduser(os.path.join('~', 'data', 'malnet_tiny')),
    )

  splits = ['train', 'val', 'test']
  db_paths = [os.path.join(cache_dir, f'{s}_db.npz') for s in splits]
  if all(map(_file_exists, db_paths)):  # All database files already exist.
    return tuple([InMemoryDB.from_file(db_path) for db_path in db_paths])

  splits_dir = os.path.join(cache_dir, 'split_info_tiny', 'type')
  data_dir = os.path.join(cache_dir, 'malnet-graphs-tiny')

  if not _file_exists(data_dir):
    destination_tar = os.path.join(cache_dir, 'malnet-graphs-tiny.tar.gz')
    _maybe_download_file(download_url, destination_tar)
    with tarfile.open(fileobj=open_file(destination_tar, 'rb')) as f:
      f.extractall(cache_dir)

  if not _file_exists(splits_dir):
    split_zip = os.path.join(cache_dir, 'split_info_tiny.zip')
    _maybe_download_file(split_url, split_zip)
    with zipfile.ZipFile(split_zip, 'r') as f:
      f.extractall(cache_dir)

  train_graphs = []
  val_graphs = []
  test_graphs = []
  max_degree = 0
  labels = {}
  for split in splits:
    with open_file(os.path.join(splits_dir, f'{split}.txt'), 'r') as f:
      filenames = f.read().split('\n')[:-1]

      for filename in filenames:
        label = filename.split('/')[0]
        if label not in labels and not labels:
          labels[label] = 0
        elif label not in labels:
          labels[label] = max(labels.values()) + 1
        y = labels[label]
        path = os.path.join(data_dir, f'{filename}.edgelist')
        with open(path, 'r') as f:
          s = f.read().split('\n')
          edges = s[5:-1]

        srcs = []
        tgts = []
        for edge in edges:
          src, tgt = edge.split()
          srcs.append(int(src))
          tgts.append(int(tgt))
        srcs = engine.cast(srcs, 'int32')
        tgts = engine.cast(tgts, 'int32')
        edges = {'my_edges': ((srcs, tgts), {})}
        num_nodes = max(engine.maximum(srcs, tgts)) + 1

        label = engine.cast([y], 'int32')
        nodes = {
            'my_nodes': {'ids': engine.range(num_nodes)},
            'g': {'y': label},
        }

        g = GraphStruct.new(
            nodes=nodes,
            edges=edges,
            schema={'my_edges': ('my_nodes', 'my_nodes')},
        )

        if add_features == 'constant':
          g = add_constant_features(engine, g, constant_value)
        elif add_features == 'one_hot_degree':
          adj = g.adj(engine, 'my_edges')
          degrees = engine.cast(adj.rowsums(), 'int32')
          max_degree = max(int(max(degrees)), max_degree)
        elif add_features == 'local_degree_profile':
          pass

        if split == 'train':
          train_graphs.append(g)
        elif split == 'val':
          val_graphs.append(g)
        elif split == 'test':
          test_graphs.append(g)

  train_db = InMemoryDB()
  val_db = InMemoryDB()
  test_db = InMemoryDB()

  for g in train_graphs:
    if add_features == 'one_hot_degree':
      g = add_one_hot_degree_features(engine, g, max_degree)
    train_db.add(g)
  for g in val_graphs:
    if add_features == 'one_hot_degree':
      g = add_one_hot_degree_features(engine, g, max_degree)
    val_db.add(g)
  for g in test_graphs:
    if add_features == 'one_hot_degree':
      g = add_one_hot_degree_features(engine, g, max_degree)
    test_db.add(g)

  train_db.finalize()
  val_db.finalize()
  test_db.finalize()

  train_db.save(db_paths[0])
  val_db.save(db_paths[1])
  test_db.save(db_paths[2])

  return train_db, val_db, test_db


def add_constant_features(
    engine, g, value = 1
):
  """Adds constant (default value of 1) features to a MalNet Tiny graph."""
  num_nodes = g.nodes['my_nodes']['ids'].shape[0]
  features = engine.cast(engine.ones([num_nodes, 1]) * value, 'float32')
  return g.update(nodes={'my_nodes': {'x': features}})


def add_one_hot_degree_features(
    engine, g, max_degree
):
  """Adds one-hot degree features to a MalNet Tiny graph."""
  adj = g.adj(engine, 'my_edges')
  degrees = engine.cast(adj.rowsums(), 'int32')
  features = engine.one_hot(degrees, max_degree + 1)
  return g.update(nodes={'my_nodes': {'x': features}})
