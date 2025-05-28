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

r"""TGBL comment dataprep.

Reddit user ID mapping for tgbl-comment. Community detection. Time split for train, validation, and test.

Example cmmand:

python google_research/fm4tlp/tgbl_comment_dataprep -- \
  --root_dir=./data
"""

import os
import pickle
from absl import app
from absl import flags
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tqdm

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    None,
    'Root directory with processed files for transfer learning',
    required=True,
)


def main(_):

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets/tgbl_comment')
  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl-comment_edgelist.csv'), 'r'
  ) as f:
    tgbl_comment_edgelist = pd.read_csv(f)

  node_mapping = dict()

  for unused_index, row in tqdm.tqdm(tgbl_comment_edgelist.iterrows()):
    src = row['src']
    dst = row['dst']
    if src not in node_mapping:
      node_mapping[src] = len(node_mapping)
    if dst not in node_mapping:
      node_mapping[dst] = len(node_mapping)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_comment_user_index_map.pkl'), 'wb'
  ) as f:
    pickle.dump(node_mapping, f)

  user_count = pd.DataFrame()
  user_count['num_nodes'] = [len(node_mapping)]

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_comment_total_count.csv'), 'w'
  ) as f:
    user_count.to_csv(f, index=False)

  print('User index map created.')

  G_user = nx.Graph()
  edgelist = zip(tgbl_comment_edgelist.src, tgbl_comment_edgelist.dst)
  G_user.add_edges_from(edgelist)

  louvain_communities_user = nx.community.louvain_communities(
      G_user, resolution=1, threshold=1e-07, seed=123
  )

  louvain_communities_user = dict(enumerate(louvain_communities_user))

  # Community index determines the size of communities.
  community_index_len_dict = {}

  for key in tqdm.tqdm(louvain_communities_user.keys()):
    community_index_len_dict[key] = len(louvain_communities_user[key])

  sorted_community_indices = [
      k
      for k, v in sorted(
          community_index_len_dict.items(),
          reverse=True,
          key=lambda item: item[1],
      )
  ]

  top_100_communities = sorted_community_indices[:100]

  community_user_map = dict()
  community_index = 0

  for comm in top_100_communities:
    community_user_map['community' + str(community_index)] = (
        louvain_communities_user[comm]
    )
    community_index += 1

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_comment_user_community_map.pkl'), 'wb'
  ) as f:
    pickle.dump(community_user_map, f)

  print('Communities created.')

  val_ratio = 0.15
  test_ratio = 0.15

  val_time, test_time = list(
      np.quantile(
          tgbl_comment_edgelist['ts'].tolist(),
          [(1 - val_ratio - test_ratio), (1 - test_ratio)],
      )
  )

  timesplit = pd.DataFrame(
      {'val_time': [int(val_time)], 'test_time': [int(test_time)]}
  )

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_comment_timesplit.csv'), 'w'
  ) as f:
    timesplit.to_csv(f, index=False)

  print('Time split created.')


if __name__ == '__main__':
  app.run(main)
