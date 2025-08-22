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

r"""Stores subgraphs for downstream structural feature computation.

Command for an example run:

python google_research/fm4tlp/generate_subgraphs  -- \
  --data=tgbl_wiki \
  --root_dir=./data_clone \
  --community=cc-subgraph \
  --split=train \
  --bs=200 \
  --aggregation_window_frac=0.01 \
  --gfs_user=moments-research-cns-storage-owner \
  --alsologtostderr
"""

import concurrent.futures
import os
import sys

from absl import app
from absl import flags
from absl import logging
import networkx as nx
import pandas as pd
import tensorflow.compat.v1 as tf
import tqdm

from fm4tlp.utils import structural_feature_helper


_DATA = flags.DEFINE_string(
    'data',
    None,
    help='tgbl_flight, tgbl_comment, tgbl_wiki, tgbl_coin, or tgbl_review',
    required=True,
)

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)

_COMMUNITY = flags.DEFINE_string(
    'community',
    None,
    help='Continent or community',
    required=True,
)

_SPLIT = flags.DEFINE_string(
    'split',
    None,
    help='train, val, or test',
    required=True,
)

_BATCH_SIZE = flags.DEFINE_integer('bs', 200, 'Batch size.')

_AGGREGATION_WINDOW_FRAC = flags.DEFINE_float(
    'aggregation_window_frac',
    0.01,
    'Fraction of the total time interval to use for aggregation.',
)

_OVERWRITE_GRAPHS = flags.DEFINE_bool(
    'overwrite_graphs', False, 'Whether to overwrite graphs.'
)

_NUM_WORKERS = flags.DEFINE_integer('num_workers', 10, 'Number of workers.')


def main(_):

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets', _DATA.value)
  batches_root = os.path.join(dataset_root, 'structural_features_by_batch')
  if not tf.io.gfile.exists(batches_root):
    tf.io.gfile.makedirs(batches_root)

  filename_prefix = _DATA.value + '_' + _COMMUNITY.value + '_' + _SPLIT.value

  edgelist_filename = filename_prefix + '_edgelist.csv'
  logging.info('Loading edgelist from %s', edgelist_filename)
  with tf.io.gfile.GFile(
      os.path.join(dataset_root, edgelist_filename), 'r'
  ) as f:
    edgelist = pd.read_csv(f)
  logging.info('Loaded edgelist.')

  if _DATA.value in ['tgbl_wiki', 'tgbl_review', 'tgbl_comment', 'tgbl_coin']:
    time_identifier = 'ts'
    source_identifier = 'source'
    target_identifier = 'target'
  elif _DATA.value in ['tgbl_flight']:
    time_identifier = 'timestamp'
    source_identifier = 'src'
    target_identifier = 'dst'
  else:
    raise ValueError('Unsupported data.')

  edgelist = edgelist.sort_values(by=time_identifier, ascending=True)
  batches = structural_feature_helper.chunker(edgelist, _BATCH_SIZE.value)

  G = nx.MultiGraph()

  time_max, time_min = max(edgelist[time_identifier]), min(
      edgelist[time_identifier]
  )

  # Lookup which subgraph indices need to be processed.
  existing_subgraph_filenames = tf.io.gfile.glob(
      os.path.join(batches_root, filename_prefix + '_subgraph_*')
  )
  processed_subgraph_indices = set()
  if _OVERWRITE_GRAPHS.value:
    for filename in existing_subgraph_filenames:
      tf.io.gfile.remove(filename)
  else:
    for filename in existing_subgraph_filenames:
      processed_subgraph_indices.add(int(filename.split('_')[-1].split('.')[0]))
  logging.info('Found %d processed subgraphs.', len(processed_subgraph_indices))

  min_edge_index = 0
  max_edge_index = 0
  G_subs = []
  for batch_index in tqdm.tqdm(range(len(batches))):
    # Get batch data.
    batch = batches[batch_index]
    src_l = batch[source_identifier].tolist()
    pos_dst_l = batch[target_identifier].tolist()
    ts = batch[time_identifier].tolist()
    time_window = _AGGREGATION_WINDOW_FRAC.value * (time_max - time_min)
    batch_t_max = ts[-1]
    batch_t_min = batch_t_max - time_window
    all_nodes = set(src_l).union(set(pos_dst_l))
    max_edge_index += len(src_l)

    # Advance lowest edge index.
    while edgelist[time_identifier][min_edge_index] < batch_t_min:
      min_edge_index += 1

    # Skip graph if it's done.
    if (
        batch_index in processed_subgraph_indices
        and not _OVERWRITE_GRAPHS.value
    ):
      G_subs.append(None)
      continue

    # Add edges to a new subgraph.
    G_sub = nx.Graph()
    edges = []
    for u, v in zip(
        edgelist[source_identifier][min_edge_index:max_edge_index],
        edgelist[target_identifier][min_edge_index:max_edge_index],
    ):
      if u in all_nodes or v in all_nodes:
        edges.append((u, v))
    G_sub.add_edges_from(edges)
    G_subs.append(G_sub)

  def _write_subgraph(G_sub, batch_index):
    if G_sub is None:
      return
    filename = filename_prefix + '_subgraph_' + str(batch_index) + '.graphml'
    with tf.io.gfile.GFile(
        os.path.join(batches_root, filename),
        'wb',
    ) as f:
      nx.write_graphml(G_sub, f)

  # Keep this bool here. Needed for routing to external code.
  run_concurrent = True
  if run_concurrent:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_NUM_WORKERS.value
    ) as executor:
      executor.map(
          lambda p: _write_subgraph(*p),
          [(graph, batch_index) for batch_index, graph in enumerate(G_subs)]
      )


if __name__ == '__main__':
  app.run(main)
