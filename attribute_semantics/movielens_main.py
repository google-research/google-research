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

"""Runner file for using WALS movie embeddings and MovieLens tags."""

import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from tensorflow.io import gfile

from attribute_semantics import cavs

_INPUT_DATA = flags.DEFINE_string(
    'input_data',
    '',
    'Path to MovieLens data.',
)
_SAVE_DIR = flags.DEFINE_string(
    'save_dir',
    '',
    'Save directory.',
)
_CAV_MODES = flags.DEFINE_list(
    'cav_modes',
    [
        'binary',
        'em_binary',
        'em_ranking_simple',
        'em_ranking_lambda',
    ],
    'List of mode options for CAV training.',
)
_NUM_NEGATIVE_MULTIPLIER = flags.DEFINE_integer(
    'num_negative_multiplier',
    3,
    'Number of negative samples for each positive sample',
)
_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    '',
    'Path to WALS model embeddings used to train CAVs',
)


def main(_):
  np.random.seed(0)

  with gfile.GFile(_MODEL_PATH.value, 'rb') as f:
    with np.load(f) as npz:
      item_embs = npz['embeddings']
      movie_ids = npz['movie_ids']
      user_ids = npz['user_ids']

  with gfile.GFile(_INPUT_DATA.value, 'r') as f:
    tags = pd.read_csv(f)

  # Remove rows not in movie_ids or user_ids
  tags = tags[tags['movieId'].isin(movie_ids) & tags['userId'].isin(user_ids)]
  # Then set item_index to be the index in movie_ids matching each movieId
  tags['item_index'] = tags['movieId'].apply(
      lambda x: np.where(movie_ids == x)[0].item()
  )
  tags['user_index'] = tags['userId'].apply(
      lambda x: np.where(user_ids == x)[0].item()
  )

  # train_tags.csv already joins ratings with tagged movies.
  tags = tags.drop(['userId', 'movieId'], axis=1)

  respath = os.path.join(_SAVE_DIR.value, 'le_False')

  mlp_model_item = None

  # concepts = tags['tag'].value_counts()[: _N_CONCEPTS.value].index.to_list()
  concepts = [
      'animated',
      'artsy',
      'believable',
      'big budget',
      'bizarre',
      'boring',
      'cheesy',
      'complicated',
      'confusing',
      'dramatic',
      'entertaining',
      'factual',
      'funny',
      'gory',
      'harsh',
      'incomprehensible',
      'intense',
      'interesting',
      'long',
      'original',
      'over the top',
      'overrated',
      'pointless',
      'predictable',
      'realistic',
      'romantic',
      'scary',
      'unrealistic',
      'violent',
      # Less than 5 examples in each class.
      'cartoonish',
      'exaggerated',
      'light-hearted',
      'mainstream',
      'mindless',
      'terrifying',
      'sappy',
      # Only one class.
      # 'budget',
      # 'dense',
      # 'documentary style',
      # 'far-fetched',
      # 'heartfelt',
      # 'juvenile humor',
      # 'playful',
      # 'serious',
      # 'slow-paced',
      # 'tearful',
      # 'twisty',
      # 'typical',
      # 'unique story',
  ]

  for cav_mode in _CAV_MODES.value:
    if 'em_' in cav_mode:
      cav_dict = cavs.train_subjective_cavs(
          item_embs=item_embs,
          cav_mode=cav_mode,
          df=tags,
          concepts=concepts,
          mlp_model_item=mlp_model_item,
          num_negative_multiplier=_NUM_NEGATIVE_MULTIPLIER.value,
      )
    else:
      cav_dict = cavs.train_cavs(
          item_embs=item_embs,
          cav_mode=cav_mode,
          df=tags,
          concepts=concepts,
          mlp_model_item=mlp_model_item,
          num_negative_multiplier=_NUM_NEGATIVE_MULTIPLIER.value,
      )

    # save cav model
    cavs.save_model_pickle(
        cav_dict,
        os.path.join(respath, 'CAVs'),
        cav_mode + '_neg_multiplier:' + str(_NUM_NEGATIVE_MULTIPLIER.value),
    )


if __name__ == '__main__':
  app.run(main)
