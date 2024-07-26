# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
import urllib.request
import numpy as np
import scipy.sparse


def _maybe_download_file(source_url, destination_path, make_dirs=True):
  """Downloads URL `source_url` onto file `destination_path` if not present."""
  if not os.path.exists(destination_path):
    dir_name = os.path.dirname(destination_path)
    if make_dirs:
      try:
        os.makedirs(dir_name)
      except FileExistsError:
        pass

    with urllib.request.urlopen(source_url) as fin:
      with open(destination_path, 'wb') as fout:
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
  edge_lists = pickle.load(open(base_path + '.graph', 'rb'))
  allx = pickle.load(open(base_path + '.allx', 'rb'), encoding='latin1')
  ally = np.load(open(base_path + '.ally', 'rb'), allow_pickle=True)

  testx = pickle.load(open(base_path + '.tx', 'rb'), encoding='latin1')

  # Add test
  test_idx = list(
      map(
          int,
          open(base_path + '.test.index').read().split('\n')[:-1],
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

  testy = np.load(open(base_path + '.ty', 'rb'), allow_pickle=True)
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
