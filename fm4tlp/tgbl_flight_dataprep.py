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

r"""TGBL flight dataprep.

Airpot ID mapping for tgbl-flight. Time split for train, validation, and test.

Example cmmand:

python google_research/fm4tlp/tgbl_flight_dataprep -- \
  --root_dir=./data
"""

import os
import pickle
from absl import app
from absl import flags
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

  dataset_root = os.path.join(_ROOT_DIR.value, 'datasets/tgbl_flight')

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl-flight_edgelist_v2.csv'), 'r'
  ) as f:
    tgbl_flight_edgelist = pd.read_csv(f)

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'airport_node_feat_v2.csv'), 'r'
  ) as f:
    airport_feat = pd.read_csv(f, keep_default_na=False)

  print('Different continents: ', len(set(airport_feat.continent)))

  airport_feat_valid_continent = airport_feat[~airport_feat.continent.isna()]

  for cont in tqdm.tqdm(set(airport_feat_valid_continent.continent)):
    print(
        'Continent and number of airports: ',
        cont,
        len(
            airport_feat_valid_continent[
                airport_feat_valid_continent['continent'] == cont
            ].airport_code
        ),
    )

  ## Assign unique index to airports
  airport_code_index_dict = dict()
  index = 0
  for airport_code in airport_feat_valid_continent.airport_code:
    airport_code_index_dict[airport_code] = index
    index += 1

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_flight_airport_index_map.pkl'), 'wb'
  ) as f:
    pickle.dump(airport_code_index_dict, f)

  count_airports = index + 1
  print('Total airports: ', count_airports)

  airport_count = pd.DataFrame()
  airport_count['num_nodes'] = [count_airports]

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_flight_total_count.csv'), 'w'
  ) as f:
    airport_count.to_csv(f, index=False)

  val_ratio = 0.15
  test_ratio = 0.15

  val_time, test_time = list(
      np.quantile(
          tgbl_flight_edgelist['timestamp'].tolist(),
          [(1 - val_ratio - test_ratio), (1 - test_ratio)],
      )
  )

  timesplit = pd.DataFrame(
      {'val_time': [int(val_time)], 'test_time': [int(test_time)]}
  )

  with tf.io.gfile.GFile(
      os.path.join(dataset_root, 'tgbl_flight_timesplit.csv'), 'w'
  ) as f:
    timesplit.to_csv(f, index=False)


if __name__ == '__main__':
  app.run(main)
