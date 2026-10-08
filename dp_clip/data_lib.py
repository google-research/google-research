# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Dataset loaders for loading experimental data."""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_steam_dataset(path):
  """Loads the Steam dataset and constructs a sparse interaction matrix.

  Source: https://www.kaggle.com/datasets/tamber/steam-video-games

  Args:
    path: String path to the Steam dataset.

  Returns:
    A scipy.sparse CSR matrix of user-game play interactions.
  """
  with open(path, 'r') as f:
    df_steam = pd.read_csv(f).dropna()
  play_data = df_steam[df_steam.iloc[:, 2] == 'play'].copy()
  row_indices = pd.factorize(play_data.iloc[:, 0])[0]
  col_indices = pd.factorize(play_data.iloc[:, 1])[0]
  data = play_data.iloc[:, 3].to_numpy()
  return sp.csr_matrix((data, (row_indices, col_indices)))


def load_reddit_dataset(path):
  """Loads the AskReddit dataset and constructs a sparse word count matrix.

  Source:
  https://github.com/heyyjudes/differentially-private-set-union/tree/master

  Args:
    path: String path to the AskReddit dataset.

  Returns:
    A scipy.sparse CSR matrix of author-word counts.
  """
  with open(path, 'r') as f:
    df_reddit = pd.read_csv(f, index_col=0).dropna()
  df_words = df_reddit[['author', 'clean_text']].copy()
  df_words['word'] = df_words['clean_text'].str.split(' ')
  df_exploded = df_words.explode('word')
  df_exploded = df_exploded[df_exploded['word'].ne('')]
  row_indices = pd.factorize(df_exploded['author'])[0]
  col_indices = pd.factorize(df_exploded['word'])[0]
  data = np.ones(len(df_exploded), dtype=np.int32)
  return sp.csr_matrix((data, (row_indices, col_indices)))


def load_lastfm_dataset(path, num_rows = 500000):
  """Loads the LastFM dataset and constructs a sparse play count matrix.

  Source: https://zenodo.org/records/6090214

  Args:
    path: String path to the LastFM dataset.
    num_rows: Integer limit on number of rows to load.

  Returns:
    A scipy.sparse CSR matrix of user-artist plays.
  """
  with open(path, 'r') as f:
    df_lastfm = pd.read_csv(
        f,
        header=None,
        names=['user', 'artist_id', 'artist_name', 'plays'],
        sep='\t',
        nrows=num_rows,
    )
  df_clean = df_lastfm.dropna(subset=['user', 'artist_id'])
  user_indices = df_clean['user'].astype('category').cat.codes
  artist_indices = df_clean['artist_id'].astype('category').cat.codes
  plays = df_clean['plays'].values
  return sp.csr_matrix((plays, (user_indices, artist_indices)))


def load_hn_dataset(path):
  """Loads the HackerNews dataset into a sparse comment indicator matrix.

  Source: https://huggingface.co/datasets/open-index/hacker-news

  Args:
    path: String path to the HackerNews dataset.

  Returns:
    A scipy.sparse CSR matrix of user-story comment interactions.
  """
  with open(path, 'rb') as f:
    df_hn = pd.read_parquet(f)
  story_ids = set(df_hn.loc[df_hn['type'] == 1, 'id'])
  parent_map = df_hn.loc[df_hn['type'] == 2].set_index('id')['parent'].to_dict()
  story_cache = {}

  def find_root_story(cid):
    path = []
    curr = cid
    while curr is not None:
      if curr in story_ids:
        root = curr
        break
      if curr in story_cache:
        root = story_cache[curr]
        break
      path.append(curr)
      curr = parent_map.get(curr)
    else:
      root = None
    for node in path:
      story_cache[node] = root
    return root

  is_comment = df_hn['type'] == 2
  is_target_week = (df_hn['time'] >= '2026-03-01') & (
      df_hn['time'] <= '2026-03-31'
  )
  comments_df = df_hn[is_comment & is_target_week].copy()
  comments_df['parent_story'] = comments_df['id'].apply(find_root_story)
  final_df = comments_df.dropna(subset=['parent_story'])[
      ['id', 'by', 'parent_story']
  ]
  final_df['parent_story'] = final_df['parent_story'].astype(int)

  row_indices = pd.factorize(final_df['by'])[0]
  col_indices = pd.factorize(final_df['parent_story'])[0]
  data = np.ones(len(final_df), dtype=np.int32)
  return sp.csr_matrix((data, (row_indices, col_indices)))


def load_movielens_dataset(path):
  """Loads the MovieLens dataset into a sparse review indicator matrix.

  Source: https://huggingface.co/datasets/reczoo/Movielens1M_m1/tree/main

  Args:
    path: String path to the MovieLens dataset.

  Returns:
    A scipy.sparse CSR matrix of user-item binary ratings.
  """
  with open(path, 'rb') as f:
    df_movielens = pd.read_json(f)
  stacked = df_movielens.stack()
  row_indices = pd.factorize(stacked.index.get_level_values(0))[0]
  col_indices = pd.factorize(stacked.values)[0]
  data = np.ones(len(stacked), dtype=np.int8)
  return sp.csr_matrix((data, (row_indices, col_indices)))


def load_pantry_dataset(path):
  """Loads the Amazon Pantry dataset into a sparse review indicator matrix.

  Source: https://nijianmo.github.io/amazon/index.html

  Args:
    path: String path to the Amazon Pantry dataset.

  Returns:
    A scipy.sparse CSR matrix of user-item reviews.
  """
  with open(path, 'r') as f:
    df_pantry = pd.read_csv(
        f, header=None, names=['item', 'user', 'rating', 'timestamp']
    )
  edges = df_pantry[['user', 'item']].drop_duplicates()
  user_indices = edges['user'].astype('category').cat.codes
  item_indices = edges['item'].astype('category').cat.codes
  data = np.ones(len(edges), dtype=np.int8)
  return sp.csr_matrix((data, (user_indices, item_indices)))


def load_twitch_dataset(path):
  """Loads the Twitch dataset and constructs a sparse stream indicator matrix.

  Source: https://cseweb.ucsd.edu/~jmcauley/datasets.html#twitch

  Args:
    path: String path to the Twitch dataset.

  Returns:
    A scipy.sparse CSR matrix of user-stream interactions.
  """
  with open(path, 'r') as f:
    df_twitch = pd.read_csv(
        f, header=None, names=['user', 'stream', 'streamer', 'start', 'stop']
    )
  edges = df_twitch[['user', 'stream']].drop_duplicates()
  user_indices = edges['user'].astype('category').cat.codes
  stream_indices = edges['stream'].astype('category').cat.codes
  data = np.ones(len(edges), dtype=np.int8)
  return sp.csr_matrix((data, (user_indices, stream_indices)))


def load_foursquare_dataset(path):
  """Loads the Foursquare Tokyo dataset into a sparse visit count matrix.

  Source: https://sites.google.com/site/yangdingqi/home/foursquare-dataset

  Args:
    path: String path to the Foursquare dataset.

  Returns:
    A scipy.sparse CSR matrix of user-location visits.
  """
  with open(path, 'rb') as f:
    df_foursquare = pd.read_table(f, encoding='latin1')
  rows = pd.factorize(df_foursquare.iloc[:, 0])[0]
  cols = pd.factorize(df_foursquare.iloc[:, 1])[0]
  data = np.ones(len(df_foursquare), dtype=int)
  return sp.coo_matrix((data, (rows, cols))).tocsr()


def load_jester_dataset(path):
  """Loads the Jester joke ratings dataset (with negative entries in [-10, 10]).

  Source: https://goldberg.berkeley.edu/jester-data/

  Args:
    path: String path to the Jester .xls dataset.

  Returns:
    A scipy.sparse CSR matrix of user-joke ratings.
  """
  with open(path, 'rb') as f:
    df_jester = pd.read_excel(f, header=None).iloc[:, 1:].dropna()
  df_jester = df_jester.replace(99, 0)
  jester_sparse_matrix = sp.csr_matrix(df_jester.to_numpy(dtype=np.float64))
  del df_jester
  return jester_sparse_matrix


def load_slashdot_dataset(path):
  """Loads the Slashdot Zoo signed social network (+1 / -1 edges).

  Source: https://snap.stanford.edu/data/soc-sign-Slashdot090221.html

  Args:
    path: String path to the Slashdot edge list (.txt).

  Returns:
    A scipy.sparse CSR matrix where rows are active raters, columns are rated
    users, and entries are +1 or -1.
  """
  with open(path, 'r') as f:
    df_slashdot = pd.read_csv(
        f,
        sep='\t',
        header=None,
        names=['rater_id', 'ratee_id', 'rating'],
        comment='#',
        dtype={'rater_id': int, 'ratee_id': int, 'rating': int},
    )
  rows = pd.factorize(df_slashdot['rater_id'])[0]
  cols = pd.factorize(df_slashdot['ratee_id'])[0]
  data = df_slashdot['rating'].to_numpy(dtype=np.float64)
  slashdot_sparse_matrix = sp.csr_matrix((data, (rows, cols)))
  del df_slashdot
  return slashdot_sparse_matrix


def load_ansur_dataset(path):
  """Loads the ANSUR dataset and constructs a sparse physical attributes matrix.

  Source: https://www.openlab.psu.edu/datasets/ansur/

  Args:
    path: String path to the ANSUR dataset.

  Returns:
    A scipy.sparse CSR matrix of physical attributes.
  """
  with open(path, 'r') as f:
    df_ansur = pd.read_csv(f)
  df_ansur_cleaned = df_ansur.dropna()
  df_ansur_numeric = df_ansur_cleaned.select_dtypes(include=['number'])
  return sp.csr_matrix(df_ansur_numeric.values)


def load_atus_dataset(path, num_rows = 100000):
  """Loads the ATUS dataset and constructs a sparse time usage matrix.

  Source: https://www.kaggle.com/datasets/bls/american-time-use-survey/data

  Args:
    path: String path to the ATUS dataset.
    num_rows: Integer limit on number of rows to load.

  Returns:
    A scipy.sparse CSR matrix of user-activity time.
  """
  with open(path, 'r') as f:
    df_atus = pd.read_csv(f, nrows=num_rows)
  df_grouped = df_atus.groupby(['tucaseid', 'tuactivity_n'], as_index=False)[
      'tuactdur24'
  ].sum()
  row_idx, _ = pd.factorize(df_grouped['tucaseid'])
  col_idx, _ = pd.factorize(df_grouped['tuactivity_n'])
  data_vals = df_grouped['tuactdur24'].values
  num_users = row_idx.max() + 1
  num_activities = col_idx.max() + 1
  return sp.coo_matrix(
      (data_vals, (row_idx, col_idx)), shape=(num_users, num_activities)
  ).tocsr()


def load_nfl_dataset(path):
  """Loads the NFL Combine dataset and constructs a sparse metrics matrix.

  Source:
  https://www.kaggle.com/datasets/thomassshaw/nfl-combine-performance-dataset

  Args:
    path: String path to the NFL Combine dataset.

  Returns:
    A scipy.sparse CSR matrix of player metrics.
  """
  with open(path, 'r') as f:
    df_nfl = pd.read_csv(f)
  cols_to_keep = ['Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump']
  df_subset = df_nfl[cols_to_keep].copy()

  def height_to_inches(ht):
    if isinstance(ht, str) and '-' in ht:
      try:
        feet, inches = ht.split('-')
        return int(feet) * 12 + int(inches)
      except ValueError:
        return np.nan
    return ht

  df_subset['Height'] = df_subset['Height'].apply(height_to_inches)
  for col in df_subset.columns:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
  df_cleaned = df_subset.dropna()
  return sp.csr_matrix(df_cleaned.values)


def load_powerlifting_dataset(
    path, num_rows = 100000
):
  """Loads the Powerlifting dataset and constructs a sparse metrics matrix.

  Source:
  https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database?select=openpowerlifting.csv

  Args:
    path: String path to the Powerlifting dataset.
    num_rows: Integer limit on number of rows to load.

  Returns:
    A scipy.sparse CSR matrix of powerlifting metrics.
  """
  with open(path, 'r') as f:
    df_powerlifting = pd.read_csv(f, nrows=num_rows)
  cols_to_keep = [
      'Age',
      'BodyweightKg',
      'Best3SquatKg',
      'Best3BenchKg',
      'Best3DeadliftKg',
  ]
  df_subset = df_powerlifting[cols_to_keep].copy()

  for col in df_subset.columns:
    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

  df_cleaned = df_subset.dropna()
  return sp.csr_matrix(df_cleaned.values)
