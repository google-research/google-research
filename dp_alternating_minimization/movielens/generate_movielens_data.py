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

"""Generates data for the MovieLens benchmarks.

Movielens 10M: follows the procedure used by Lee et al.
(https://dl.acm.org/doi/10.5555/2946645.2946660) to partition the data.
Movielens 20M: follows the procedure used by Liang et al.
(https://dl.acm.org/doi/10.1145/3178876.3186150) to partition the data. The code
is branched from their implementation (https://github.com/dawenl/vae_cf) adapted
to PY3.
"""

import os
from typing import Sequence
import urllib.request
import zipfile

from absl import app
import numpy as np
import pandas as pd


def generate_ml10m_data(
    ml10m_dir,
    val_frac,
    test_frac,
    min_movie_count = 0):
  """Downloads and processes MovieLens 10M data.

  Args:
    ml10m_dir: directory to write the processed data to.
    val_frac: fraction of the validation split.
    test_frac: fraction of the test split.
    min_movie_count: filter out movies with fewer than min_movie_count ratings.
  """
  if not os.path.exists(ml10m_dir):
    os.makedirs(ml10m_dir)
  print('Downloading and extracting Movielens 10M data')
  ml10m_zip = os.path.join(ml10m_dir, 'ml10m.zip')
  urllib.request.urlretrieve(
      'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
      ml10m_zip)
  zip_ref = zipfile.ZipFile(ml10m_zip, 'r')
  zip_ref.extractall(ml10m_dir)
  os.remove(ml10m_zip)
  print('Processing Movielens 10M data')
  ratings_file = os.path.join(ml10m_dir, 'ml-10M100K/ratings.dat')
  movies_file = os.path.join(ml10m_dir, 'ml-10M100K/movies.dat')
  ratings = pd.read_csv(
      open(ratings_file),
      names=['uid', 'sid', 'rating', 'timestamp'],
      delimiter='::',
      engine='python')
  del ratings['timestamp']
  # Index users from 0.
  ratings['uid'] = ratings['uid'] - 1
  ratings['rating'] = ratings['rating'].astype(np.float32)
  # Remove movies with no ratings
  def remap_ids(key, min_count):
    counts = ratings.groupby(key)[key].count()
    ids_to_keep = counts[counts > min_count].index.values
    max_id = max(ratings[key])
    new_ids = -np.ones(max_id+1, dtype=np.int32)
    for new, old in enumerate(ids_to_keep):
      new_ids[old] = new
    ratings[key] = [new_ids[old] for old in ratings[key]]
    new_ratings = ratings[ratings[key] > -1]
    return new_ratings, new_ids
  ratings, sid_map = remap_ids('sid', min_movie_count)
  ratings, _ = remap_ids('uid', 0)
  num_users = max(ratings['uid']) + 1
  num_movies = max(ratings['sid']) + 1
  num_ratings = len(ratings)
  sparsity = num_ratings/(num_users * num_movies)
  print(f'After filtering, there are {num_ratings} ratings from {num_users} '
        f'users and {num_movies} movies (sparsity: {100*sparsity:.3f}%)')

  # Split the data
  train_frac = 1 - (val_frac + test_frac)
  train = ratings.sample(frac=train_frac, replace=False, random_state=1234)
  vt = ratings[~ratings.index.isin(train.index)]
  validation = vt.sample(
      frac=val_frac/(val_frac + test_frac), replace=False, random_state=1234)
  test = vt[~vt.index.isin(validation.index)]

  print('Extracting movie metadata')
  # Write movies metadata
  movies = []
  titles = []
  genres = []
  with open(movies_file, 'rb') as f:
    for line in f:
      movie, title, genre = line.decode('utf-8').strip().split('::')
      sid = sid_map[int(movie)]
      if sid > -1:
        movies.append(sid)
        titles.append(title)
        genres.append(genre)
  movies_df = pd.DataFrame.from_dict(
      {'sid': movies, 'title': titles, 'genres': genres})

  def write(df, filename):
    with open(os.path.join(ml10m_dir, filename), 'w') as f:
      df.to_csv(f, index=False, header=True)
  write(train, 'train.csv')
  write(validation, 'validation.csv')
  write(test, 'test.csv')
  write(movies_df, 'features.csv')
  print('Done.')


def _get_count(tp, key):
  playcount_groupbyid = tp[[key]].groupby(key, as_index=True)
  count = playcount_groupbyid.size()
  return count


def _filter_triplets(tp, min_uc, min_sc):
  """Filters a DataFrame.

  Args:
    tp: a DataFrame of (movieId, userId, rating) triplets.
    min_uc: filter out users with fewer than min_uc ratings.
    min_sc: filter out items with fewer than min_sc ratings.
  Returns:
    A DataFrame tuple of the filtered data, the user counts and the item counts.
  """
  # Only keep the triplets for items which were clicked on by at least min_sc
  # users.
  if min_sc > 0:
    itemcount = _get_count(tp, 'movieId')
    tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

  # Only keep the triplets for users who clicked on at least min_uc items
  # After doing this, some of the items will have less than min_uc users, but
  # should only be a small proportion
  if min_uc > 0:
    usercount = _get_count(tp, 'userId')
    tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

  # Update both usercount and itemcount after filtering
  usercount, itemcount = _get_count(tp, 'userId'), _get_count(tp, 'movieId')
  return tp, usercount, itemcount


def _split_train_test_proportion(data, test_prop=0.2):
  """Splits a DataFrame into train and test sets.

  Args:
    data: a DataFrame of (userId, itemId, rating).
    test_prop: the proportion of test ratings.
  Returns:
    Two DataFrames of the train and test sets. The data is grouped by user, then
    each user (with 5 ratings or more) is randomly split into train and test
    ratings.
  """
  data_grouped_by_user = data.groupby('userId')
  tr_list, te_list = list(), list()

  np.random.seed(98765)

  for _, group in data_grouped_by_user:
    n_items_u = len(group)

    if n_items_u >= 5:
      idx = np.zeros(n_items_u, dtype='bool')
      idx[np.random.choice(
          n_items_u, size=int(test_prop * n_items_u), replace=False)
          .astype('int64')] = True

      tr_list.append(group[np.logical_not(idx)])
      te_list.append(group[idx])
    else:
      tr_list.append(group)

  data_tr = pd.concat(tr_list)
  data_te = pd.concat(te_list)

  return data_tr, data_te


def _generate_data(raw_data, output_dir, n_heldout_users, min_uc, min_sc):
  """Generates and writes train, validation and test data.

  The raw_data is first split into train, validation and test by user. For the
  validation set, each user's ratings are randomly partitioned into two subsets
  following a (80, 20) split (see split_train_test_proportion), and written to
  validation_tr.csv and validation_te.csv. A similar split is applied to the
  test set.

  Args:
    raw_data: a DataFrame of (userId, movieId, rating).
    output_dir: path to the output directory.
    n_heldout_users: this many users are held out for each of the validation and
      test sets.
    min_uc: filter out users with fewer than min_uc ratings.
    min_sc: filter out items with fewer than min_sc ratings.
  """
  raw_data, user_activity, item_popularity = _filter_triplets(
      raw_data, min_uc, min_sc)
  sparsity = 1. * raw_data.shape[0] / (
      user_activity.shape[0] * item_popularity.shape[0])
  print('After filtering, there are %d watching events from %d users and %d '
        'movies (sparsity: %.3f%%)' %
        (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0],
         sparsity * 100))
  unique_uid = user_activity.index
  np.random.seed(98765)
  idx_perm = np.random.permutation(unique_uid.size)
  unique_uid = unique_uid[idx_perm]
  n_users = unique_uid.size
  tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
  vd_users = unique_uid[(n_users - n_heldout_users * 2):
                        (n_users - n_heldout_users)]
  te_users = unique_uid[(n_users - n_heldout_users):]
  train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
  unique_sid = pd.unique(train_plays['movieId'])
  show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
  profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
  def numerize(tp):
    uid = [profile2id[x] for x in tp['userId']]
    sid = [show2id[x] for x in tp['movieId']]
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

  pro_dir = output_dir
  if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)
  vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
  vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
  vad_plays_tr, vad_plays_te = _split_train_test_proportion(vad_plays)
  test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
  test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
  test_plays_tr, test_plays_te = _split_train_test_proportion(test_plays)

  train_data = numerize(train_plays)
  train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

  vad_data_tr = numerize(vad_plays_tr)
  vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

  vad_data_te = numerize(vad_plays_te)
  vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

  test_data_tr = numerize(test_plays_tr)
  test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

  test_data_te = numerize(test_plays_te)
  test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

  sid_mapping = pd.DataFrame(
      [{'new': i, 'old': sid} for (i, sid) in enumerate(unique_sid)])
  sid_mapping.to_csv(os.path.join(pro_dir, 'sid_mapping.csv'), index=False)


def _generate_metadata(movie_metadata_file, sid_mapping_file, output_file):
  """Generate MovieLens 20M metadata.

  Args:
    movie_metadata_file: movies.csv file downloaded from the MovieLens website.
    sid_mapping_file: csv file containing two columns: old and new.
    output_file: file to write to (this will contain the mapped sids).
  """
  md = pd.read_csv(movie_metadata_file)
  mapping = pd.read_csv(sid_mapping_file, delimiter=',')
  old_to_new = {}
  for new, old in zip(mapping.new, mapping.old):
    old_to_new[old] = new
  items = []
  id_col = 'movieId'
  metadata_cols = ['title', 'genres']
  for sid, metadata in zip(md[id_col].values, md[metadata_cols].values):
    if sid in old_to_new:
      d = {'sid': old_to_new[sid]}
      d.update({k: v for k, v in zip(metadata_cols, metadata)})
      items.append(d)
  items_df = pd.DataFrame.from_records(items)
  items_df.to_csv(output_file, index=False, header=True)


def generate_ml20m_data(ml20m_dir):
  """Downloads and processes MovieLens 20M data."""
  ratings_file = os.path.join(ml20m_dir, 'ml-20m/ratings.csv')
  if not os.path.exists(ml20m_dir):
    os.makedirs(ml20m_dir)
  print('Downloading and extracting Movielens 20M data')
  ml20m_zip = os.path.join(ml20m_dir, 'ml20m.zip')
  urllib.request.urlretrieve(
      'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
      ml20m_zip)
  with zipfile.ZipFile(ml20m_zip, 'r') as zipref:
    zipref.extractall(ml20m_dir)
  os.remove(ml20m_zip)
  raw_data = pd.read_csv(ratings_file, header=0)
  # binarize the data (only keep ratings >= 4)
  raw_data = raw_data[raw_data['rating'] > 3.5]
  print('Processing Movielens 20M data')
  _generate_data(
      raw_data, output_dir=ml20m_dir, n_heldout_users=10000, min_uc=5, min_sc=0)
  print('Extracting movie metadata')
  _generate_metadata(
      movie_metadata_file=os.path.join(ml20m_dir, 'ml-20m/movies.csv'),
      sid_mapping_file=os.path.join(ml20m_dir, 'sid_mapping.csv'),
      output_file=os.path.join(ml20m_dir, 'features.csv'))
  print('Done.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  generate_ml10m_data('ml10m/', 0.1, 0.1)
  generate_ml20m_data('ml20m/')


if __name__ == '__main__':
  app.run(main)
