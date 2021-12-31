# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Generates data for the VAE benchmarks.

Branched from https://github.com/dawenl/vae_cf with modification.
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd


def get_count(tp, id):
  playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
  count = playcount_groupbyid.size()
  return count


def filter_triplets(tp, min_uc, min_sc):
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
    itemcount = get_count(tp, 'movieId')
    tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

  # Only keep the triplets for users who clicked on at least min_uc items
  # After doing this, some of the items will have less than min_uc users, but
  # should only be a small proportion
  if min_uc > 0:
    usercount = get_count(tp, 'userId')
    tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

  # Update both usercount and itemcount after filtering
  usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
  return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
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

  for i, (_, group) in enumerate(data_grouped_by_user):
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

    if i % 1000 == 0:
      print('%d users sampled' % i)
      sys.stdout.flush()

  data_tr = pd.concat(tr_list)
  data_te = pd.concat(te_list)

  return data_tr, data_te


def generate_data(raw_data, output_dir, n_heldout_users, min_uc, min_sc):
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
  raw_data, user_activity, item_popularity = filter_triplets(
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
  with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
      f.write('%s\n' % sid)
  vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
  vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]
  vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
  test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
  test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]
  test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str, default='',
                      help='Path where to save the datasets.')
  args = parser.parse_args()

  # MovieLens 20M
  ml20m_zip = os.path.join(args.output_dir, 'ml20m.zip')
  ml20m_dir = os.path.join(args.output_dir, 'ml-20m/')
  ml20m_file = os.path.join(args.output_dir, 'ml-20m/ratings.csv')
  print('Downloading and extracting Movielens 20M data')
  urllib.request.urlretrieve(
      'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
      ml20m_zip)
  with zipfile.ZipFile(ml20m_zip, 'r') as zipref:
    zipref.extract('ml-20m/ratings.csv', args.output_dir)
  os.remove(ml20m_zip)
  raw_data = pd.read_csv(ml20m_file, header=0)
  os.remove(ml20m_file)
  # binarize the data (only keep ratings >= 4)
  raw_data = raw_data[raw_data['rating'] > 3.5]
  generate_data(
      raw_data, output_dir=ml20m_dir, n_heldout_users=10000, min_uc=5, min_sc=0)
  print('Done processing Movielens 20M.')

  # Million Song Data
  print('Downloading and extracting Million Song Data')
  msd_zip = os.path.join(args.output_dir, 'msd.zip')
  msd_dir = os.path.join(args.output_dir, 'msd/')
  msd_file = os.path.join(args.output_dir, 'msd/train_triplets.txt')
  urllib.request.urlretrieve(
      'http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip',
      msd_zip)
  with zipfile.ZipFile(msd_zip, 'r') as zipref:
    zipref.extractall(msd_dir)
  os.remove(msd_zip)
  raw_data = pd.read_csv(
      msd_file, sep='\t', header=None, names=['userId', 'movieId', 'count'])
  os.remove(msd_file)
  generate_data(
      raw_data, output_dir=msd_dir, n_heldout_users=50000, min_uc=20,
      min_sc=200)
  print('Done processing Million Song Data.')


if __name__ == '__main__':
  main()
