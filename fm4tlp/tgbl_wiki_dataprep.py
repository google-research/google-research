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

r"""TGBL wiki dataprep.

Wiki node ID mapping for tgbl-wiki. Community detection. Time split for train, validation, and test.

Note that the user_id and the item_id form a bipartite graph. The user_id and
item_id both have integer IDs. Their value might collide but they are different
nodes. To make them unique, we make user_id even numbers and item_id odd. In
particular, the user_id is multiplied by 2, and item_id is multiplied by 2 and
then add 1.

Note: we can consider using Barber modularity clustering instead:
 * https://doi.org/10.1103/PhysRevE.76.066102
 * https://medium.com/eni-digitalks/uncovering-hidden-communities-in-bipartite-graphs-8a1fc518a04a


Example cmmand:

python google_research/fm4tlp/tgbl_wiki_dataprep -- \
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

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets/tgbl_wiki')
  tgbl_wiki_edgelist = pd.DataFrame(
      columns=[
          'timestamp',
          'user_id',
          'item_id',
          'state_label',
          'comma_separated_list_of_features',
      ]
  )
  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl-wiki_edgelist_v2.csv'), 'r'
  ) as f:
    # Because each row has more than one commas and only 5 columns, regular
    # read_csv() does not work.
    first = True
    for line in f:
      # Skip the header.
      if first:
        first = False
        continue
      line = line.replace('\n', '')
      comma_separated_list = line.split(',')
      # The last column is comma_separated_list_of_features, which has many
      # commas.
      row = comma_separated_list[0:4] + [
          ','.join(comma_separated_list[4 : len(comma_separated_list)])
      ]
      tgbl_wiki_edgelist.loc[len(tgbl_wiki_edgelist)] = [
          int(float(row[2])),
          int(row[0]),
          int(row[1]),
          row[3],
          row[4],
      ]

  # The user_id and the item_id form a bipartite graph. The user_id and item_id
  # both have integer IDs. Their value might collide but they are different
  # nodes. To make them unique, we make user_id even numbers and item_id odd.
  tgbl_wiki_edgelist['user_id'] = tgbl_wiki_edgelist['user_id'] * 2
  tgbl_wiki_edgelist['item_id'] = tgbl_wiki_edgelist['item_id'] * 2 + 1
  node_mapping = dict()

  for _, row in tqdm.tqdm(tgbl_wiki_edgelist.iterrows()):
    user_id = row['user_id']
    item_id = row['item_id']
    if user_id not in node_mapping:
      node_mapping[user_id] = len(node_mapping)
    if item_id not in node_mapping:
      node_mapping[item_id] = len(node_mapping)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_wiki_node_index_map.pkl'), 'wb'
  ) as f:
    pickle.dump(node_mapping, f)

  node_count = pd.DataFrame()
  node_count['num_nodes'] = [len(node_mapping)]

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_wiki_total_count.csv'), 'w'
  ) as f:
    node_count.to_csv(f, index=False)

  print('node index map created.')

  node_graph = nx.Graph()
  edgelist = zip(tgbl_wiki_edgelist.user_id, tgbl_wiki_edgelist.item_id)
  node_graph.add_edges_from(edgelist)

  louvain_communities_node = nx.community.louvain_communities(
      node_graph, resolution=1, threshold=1e-07, seed=123
  )

  louvain_communities_node = dict(enumerate(louvain_communities_node))
  # Community index determines the size of communities.
  community_index_len_dict = {}

  for key in tqdm.tqdm(louvain_communities_node.keys()):
    community_index_len_dict[key] = len(louvain_communities_node[key])

  sorted_community_indices = [
      k
      for k, _ in sorted(
          community_index_len_dict.items(),
          reverse=True,
          key=lambda item: item[1],
      )
  ]

  top_100_communities = sorted_community_indices[:100]

  community_node_map = dict()
  community_index = 0

  for comm in top_100_communities:
    community_node_map['community' + str(community_index)] = (
        louvain_communities_node[comm]
    )
    community_index += 1

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_wiki_node_community_map.pkl'), 'wb'
  ) as f:
    pickle.dump(community_node_map, f)

  print('Communities created.')

  val_ratio = 0.15
  test_ratio = 0.15

  val_time, test_time = list(
      np.quantile(
          tgbl_wiki_edgelist['timestamp'].tolist(),
          [(1 - val_ratio - test_ratio), (1 - test_ratio)],
      )
  )

  timesplit = pd.DataFrame(
      {'val_time': [int(val_time)], 'test_time': [int(test_time)]}
  )

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_wiki_timesplit.csv'), 'w'
  ) as f:
    timesplit.to_csv(f, index=False)

  print('Time split created.')


if __name__ == '__main__':
  app.run(main)
