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

r"""Continent-wise data prep and negative smapling for validation and test.

Example command:

python google_research/fm4tlp/tgbl_flight_negatives -- \
    --root_dir=./data \
    --continent=EU
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

from fm4tlp.utils import negative_sampler


_CONTINENT = flags.DEFINE_string(
    'continent',
    None,
    'Continent Code',
    required=True,
)

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)

_SEED = flags.DEFINE_integer('seed', 12345, 'Seed for random number generator.')


def main(_):

  random.seed(_SEED.value)
  np.random.seed(_SEED.value)

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets/tgbl_flight')

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl-flight_edgelist_v2.csv'), 'r'
  ) as f:
    tgbl_flight_edgelist = pd.read_csv(f)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'airport_node_feat_v2.csv'), 'r'
  ) as f:
    airport_feat = pd.read_csv(f, keep_default_na=False)

  airport_feat_valid_continent = airport_feat[~airport_feat.continent.isna()]

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_flight_airport_index_map.pkl'), 'rb'
  ) as f:
    airpot_code_index_dict = pickle.load(f)

  continent_airport = airport_feat_valid_continent[
      airport_feat_valid_continent['continent'] == _CONTINENT.value
  ]['airport_code'].tolist()

  continent_airport_tgbl_flight_edgelist = tgbl_flight_edgelist[
      tgbl_flight_edgelist['src'].isin(continent_airport)
      & tgbl_flight_edgelist['dst'].isin(continent_airport)
  ]

  time_l = []
  src_l = []
  dst_l = []
  callsign_l = []
  typecode_l = []

  for unused_index, row in tqdm.tqdm(
      continent_airport_tgbl_flight_edgelist.iterrows()
  ):
    time_l.append(row['timestamp'])
    src_l.append(airpot_code_index_dict[row['src']])
    dst_l.append(airpot_code_index_dict[row['dst']])
    callsign_l.append(row['callsign'])
    typecode_l.append(row['typecode'])

  continent_airport_tgbl_flight_edgelist_indexed = pd.DataFrame({
      'timestamp': time_l,
      'src': src_l,
      'dst': dst_l,
      'callsign': callsign_l,
      'typecode': typecode_l,
  }).sort_values('timestamp')

  filename = 'tgbl_flight_' + _CONTINENT.value + '_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    continent_airport_tgbl_flight_edgelist_indexed.to_csv(f, index=False)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_flight_timesplit.csv'), 'r'
  ) as f:
    timesplit = pd.read_csv(f)

  val_time = timesplit['val_time'][0]
  test_time = timesplit['test_time'][0]

  continent_airport_tgbl_flight_edgelist_indexed_train = (
      continent_airport_tgbl_flight_edgelist_indexed[
          continent_airport_tgbl_flight_edgelist_indexed['timestamp']
          <= val_time
      ]
  )
  continent_airport_tgbl_flight_edgelist_indexed_val = (
      continent_airport_tgbl_flight_edgelist_indexed[
          (
              continent_airport_tgbl_flight_edgelist_indexed['timestamp']
              > val_time
          )
          & (
              continent_airport_tgbl_flight_edgelist_indexed['timestamp']
              <= test_time
          )
      ]
  )
  continent_airport_tgbl_flight_edgelist_indexed_test = (
      continent_airport_tgbl_flight_edgelist_indexed[
          continent_airport_tgbl_flight_edgelist_indexed['timestamp']
          > test_time
      ]
  )

  filename = 'tgbl_flight_' + _CONTINENT.value + '_train_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    continent_airport_tgbl_flight_edgelist_indexed_train.to_csv(f, index=False)

  filename = 'tgbl_flight_' + _CONTINENT.value + '_val_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    continent_airport_tgbl_flight_edgelist_indexed_val.to_csv(f, index=False)

  filename = 'tgbl_flight_' + _CONTINENT.value + '_test_edgelist.csv'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'w') as f:
    continent_airport_tgbl_flight_edgelist_indexed_test.to_csv(f, index=False)

  print(
      'Edges in train: ',
      len(continent_airport_tgbl_flight_edgelist_indexed_train),
  )
  print(
      'Edges in val: ', len(continent_airport_tgbl_flight_edgelist_indexed_val)
  )
  print(
      'Edges in test: ',
      len(continent_airport_tgbl_flight_edgelist_indexed_test),
  )

  all_nodes = set(continent_airport_tgbl_flight_edgelist_indexed['src']).union(
      set(continent_airport_tgbl_flight_edgelist_indexed['dst'])
  )

  val_ns = dict()

  val_historical_neighbor_sets = collections.defaultdict(set)
  all_nodes_list = list(all_nodes)
  continent_airport_tgbl_flight_edgelist_indexed_val.sort_values(
      by='timestamp', inplace=True
  )
  for ts, source, target in tqdm.tqdm(
      zip(
          continent_airport_tgbl_flight_edgelist_indexed_val['timestamp'],
          continent_airport_tgbl_flight_edgelist_indexed_val['src'],
          continent_airport_tgbl_flight_edgelist_indexed_val['dst'],
      )
  ):
    val_ns[(source, target, ts)] = negative_sampler.get_negatives(
        val_historical_neighbor_sets[source],
        all_nodes_list,
        20,
    )
    val_historical_neighbor_sets[source].add(target)

  filename = 'tgbl_flight_' + _CONTINENT.value + '_val_ns.pkl'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'wb') as f:
    pickle.dump(val_ns, f)

  test_ns = dict()
  test_historical_neighbor_sets = collections.defaultdict(set)
  continent_airport_tgbl_flight_edgelist_indexed_test.sort_values(
      by='timestamp', inplace=True
  )
  for ts, source, target in tqdm.tqdm(
      zip(
          continent_airport_tgbl_flight_edgelist_indexed_test['timestamp'],
          continent_airport_tgbl_flight_edgelist_indexed_test['src'],
          continent_airport_tgbl_flight_edgelist_indexed_test['dst'],
      )
  ):
    test_ns[(source, target, ts)] = negative_sampler.get_negatives(
        test_historical_neighbor_sets[source],
        all_nodes_list,
        20,
    )
    test_historical_neighbor_sets[source].add(target)

  filename = 'tgbl_flight_' + _CONTINENT.value + '_test_ns.pkl'
  with tf.io.gfile.GFile(os.path.join(dataset_root, filename), 'wb') as f:
    pickle.dump(test_ns, f)


if __name__ == '__main__':
  app.run(main)
