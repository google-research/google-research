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

r"""tgbl-coin community cluster data preparation and negative sampling.

The purpose of this binary is to prepare transfer learning sub-graphs for the
tgbl-coin dataset experiments. We sample two sub-graphs, one intended for
the train/val periods, and one intended for the test period:

    |                  subgraph 1                   |     subgraph 2     |
  min_t [ train period ] val_time [val period] test_time [test period] max_t

val_time and test_time are already known prior to this binary being run, and
are loaded from files that result from `tgbl_coin_dataprep.py`.

The sampling is performed from an on-disk community-map file, which maps
community IDs to the nodes in that community. The binary assumes the presence
of the following files in _ROOT_DIR.value + 'datasets/tgbl_coin':

1. tgbl-coin_edgelist_v2.csv: the edgelist of the entire graph.
2. tgbl_coin_node_index_map.pkl: a mapping from node ID to its index in the
   edgelist.
3. tgbl_coin_node_community_map.pkl: a mapping from node ID to its community
   ID.
4. tgbl_coin_timesplit.csv: a file with the val_time and test_time columns,
   each with one entry specifying the boundaries of the val and test periods.

If any of these files are missing, run `tgbl_coin_dataprep.py` first.

The sampling process is:

1. Train subgraph: sample communities without replacement with probability
   proportional to the community sizes. Stop when the cc size surpasses
   `graph_fraction`. Re-sample if condition (1) above is not met.
2. Test subgraph: sample communities (from the remainder after sampling train
   sugraph) without replacement with probability proportional to the community
   sizes. Stop when the cc size surpasses `graph_fraction`. Re-sample if
   condition (2) above is not met.

The subgraphs will be sampled so that they are approximately a `graph_fraction`
proportion of the entire graph. Furthermore, they are sampled so that they are
not too edge-density imbalanced across the train / val / test splits.
Specifically, define the edge-density of a subgraph C on a split (time interval)
S as D_C[S] = |E_C[S]| / (|S| * (|C| * (|C| - 1))). This binary samples
C1 and C2 such that:

  D_C1[train] / time_interval_balance_factor <= D_C1[val] <= D_C1[train] * time_interval_balance_factor    (1)
and
  D_C1[train] / time_interval_balance_factor <= D_C2[test] <= D_C1[train] * time_interval_balance_factor   (2)

...effectively ensuring that the edge-density of the val and test graphs are
within a `time_interval_balance_factor` of the edge-density of the test graph.

>> Val / Test temporal density balancing: this script also checks that the
first k sub-intervals of the val and test intervals have above some threshold
of edge density. In particular, this scipt accepts a vector of temporal
quantiles (e.g. q = [0.05, 0.10, 0.15, ...]) and ensures that for every i,
the edge density on the range [q[i], q[i+1]] is above some delta% of the edge
density on the full interval. This check is important to ensure that warm-start
techniques on the resulting datasets are successful.

Example run command:

python google_research/fm4tlp/tgbl_coin_subgraph_prep -- \
    --root_dir=./data
"""

import collections
import os
import pickle
import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tqdm

from fm4tlp.utils import communities
from fm4tlp.utils import negative_sampler


_GRAPH_FRACTION = flags.DEFINE_float(
    'graph_fraction',
    0.3,
    'Relative size of the community clusters w.r.t. the entire graph. Must be'
    ' less than 0.5. If not specified, no community clusters are sampled. If'
    ' specified, the explicit community choice is ignored.',
)


_TIME_INTERVAL_BALANCE_FACTOR = flags.DEFINE_float(
    'time_interval_balance_factor',
    5.0,
    'This number controls the level of (multiplicative) imbalance between the'
    ' train and val community clusters and also between the val and test'
    ' community clusters. Must be > 1.0.',
)


_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)

_WARMSTART_QUANTILES_TO_CHECK = flags.DEFINE_list(
    'warmstart_quantiles_to_check',
    ['0.10', '0.25', '0.50'],
    'Temporal quantiles to check for edge density in the val and test sets.',
)

_WARMSTART_QUANTILES_DELTA = flags.DEFINE_float(
    'warmstart_quantiles_delta',
    0.50,
    (
        'All temporal ranges defined by warmstart quantiles will have at least'
        ' this fraction of the edge density of the full interval.'
    ),
)

_SEED = flags.DEFINE_integer('seed', 12345, 'Seed for random number generator.')


def main(_):

  random.seed(_SEED.value)
  np.random.seed(_SEED.value)

  if _GRAPH_FRACTION.value >= 0.5 or _GRAPH_FRACTION.value <= 0.0:
    raise ValueError('graph_fraction must be in (0.0, 0.5).')

  if _TIME_INTERVAL_BALANCE_FACTOR.value <= 1.0:
    raise ValueError('time_interval_balance_factor must be greater than 1.0.')

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets/tgbl_coin')

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl-coin_edgelist_v2.csv'), 'r'
  ) as f:
    tgbl_coin_edgelist = pd.read_csv(f)
  tgbl_coin_edgelist.rename(
      columns={'src': 'source', 'dst': 'target', 'day': 'ts'}, inplace=True
  )

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_coin_address_index_map.pkl'), 'rb'
  ) as f:
    node_index_dict = pickle.load(f)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_coin_address_community_map.pkl'), 'rb'
  ) as f:
    community_node_map = pickle.load(f)

  communities.reindex_communities(community_node_map, node_index_dict)

  with tf.io.gfile.GFile(dataset_root + '/tgbl_coin_timesplit.csv', 'r') as f:
    timesplit = pd.read_csv(f)

  source_mapped_l = []
  target_mapped_l = []

  for _, row in tqdm.tqdm(tgbl_coin_edgelist.iterrows()):
    source_mapped_l.append(node_index_dict[row['source']])
    target_mapped_l.append(node_index_dict[row['target']])

  tgbl_coin_edgelist['source'] = source_mapped_l
  tgbl_coin_edgelist['target'] = target_mapped_l

  val_time = timesplit['val_time'][0]
  test_time = timesplit['test_time'][0]

  train_val_subgraph, test_subgraph = communities.get_community_cluster_nodes(
      edgelist_df=tgbl_coin_edgelist,
      community_node_map=community_node_map,
      time_interval_balance_factor=_TIME_INTERVAL_BALANCE_FACTOR.value,
      val_time=val_time,
      test_time=test_time,
      target_community_cluster_size=int(
          _GRAPH_FRACTION.value * len(node_index_dict)
      ),
      warmstart_quantiles_to_check=[
          float(s) for s in _WARMSTART_QUANTILES_TO_CHECK.value
      ],
      warmstart_quantiles_delta=_WARMSTART_QUANTILES_DELTA.value,
  )

  # Prepare train/val edgelist and negative samples.
  train_val_dataset_name = 'cc-subgraph'

  train_val_edgelist = tgbl_coin_edgelist[
      tgbl_coin_edgelist['source'].isin(train_val_subgraph)
      & tgbl_coin_edgelist['target'].isin(train_val_subgraph)
  ]

  train_val_edgelist = train_val_edgelist.sort_values('ts')

  with tf.io.gfile.GFile(
      os.path.join(
          dataset_root,
          'tgbl_coin_' + train_val_dataset_name + '_edgelist.csv',
      ),
      'w',
  ) as f:
    train_val_edgelist.to_csv(f, index=False)

  community_edgelist_train = train_val_edgelist[
      train_val_edgelist['ts'] <= val_time
  ]
  community_edgelist_val = train_val_edgelist[
      (train_val_edgelist['ts'] > val_time)
      & (train_val_edgelist['ts'] <= test_time)
  ]

  filename = 'tgbl_coin_' + train_val_dataset_name + '_train_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    community_edgelist_train.to_csv(f, index=False)

  filename = 'tgbl_coin_' + train_val_dataset_name + '_val_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    community_edgelist_val.to_csv(f, index=False)

  print(
      'Edges in train: ',
      len(community_edgelist_train),
  )
  print('Edges in val: ', len(community_edgelist_val))

  train_val_nodes = (
      set(community_edgelist_train['source'])
      .union(set(community_edgelist_train['target']))
      .union(set(community_edgelist_val['source']))
      .union(set(community_edgelist_val['target']))
  )

  val_ns = dict()

  val_historical_neighbor_sets = collections.defaultdict(set)
  train_val_nodes_list = list(train_val_nodes)
  community_edgelist_val.sort_values(by='ts', inplace=True)
  for ts, source, target in tqdm.tqdm(
      zip(
          community_edgelist_val['ts'],
          community_edgelist_val['source'],
          community_edgelist_val['target'],
      )
  ):
    val_ns[(source, target, ts)] = negative_sampler.get_negatives(
        val_historical_neighbor_sets[source],
        train_val_nodes_list,
        20,
    )
    val_historical_neighbor_sets[source].add(target)

  filename = 'tgbl_coin_' + train_val_dataset_name + '_val_ns.pkl'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'wb') as f:
    pickle.dump(val_ns, f)

  # Prepare test edgelist and negative samples.
  test_dataset_name = 'cc-subgraph'

  test_edgelist = tgbl_coin_edgelist[
      tgbl_coin_edgelist['source'].isin(test_subgraph)
      & tgbl_coin_edgelist['target'].isin(test_subgraph)
  ]

  filename = 'tgbl_coin_' + test_dataset_name + '_test_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    test_edgelist.to_csv(f, index=False)

  print(
      'Edges in test: ',
      len(test_edgelist),
  )

  test_nodes = set(test_edgelist['source']).union(set(test_edgelist['target']))

  test_ns = dict()
  test_historical_neighbor_sets = collections.defaultdict(set)
  test_nodes_list = list(test_nodes)
  test_edgelist.sort_values(by='ts', inplace=True)
  for ts, source, target in tqdm.tqdm(
      zip(test_edgelist['ts'], test_edgelist['source'], test_edgelist['target'])
  ):
    test_ns[(source, target, ts)] = negative_sampler.get_negatives(
        test_historical_neighbor_sets[source],
        test_nodes_list,
        20,
    )
    test_historical_neighbor_sets[source].add(target)

  filename = 'tgbl_coin_' + test_dataset_name + '_test_ns.pkl'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'wb') as f:
    pickle.dump(test_ns, f)


if __name__ == '__main__':
  app.run(main)
