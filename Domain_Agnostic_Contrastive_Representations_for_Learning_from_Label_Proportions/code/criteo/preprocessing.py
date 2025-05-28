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

"""Preprocessing code to prepare dataset."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CriteoPreprocess:
  """Preparing dataset for LLP and supervised training."""

  def __init__(self, data_path, split_frac, store_dir):
    self.data = pd.read_csv(data_path, delimiter=',')
    self.add_col_names()
    self.process_infreq()
    self.process_feats()
    data_size = self.data.shape[0]
    # supervised data
    self.store_dir = store_dir
    self.store_files(self.store_dir + 'train/', 'train_', 0,
                     int(split_frac[0] * data_size))
    self.store_files(self.store_dir + 'test/', 'test_',
                     int(split_frac[0] * data_size),
                     int(split_frac[1] * data_size))
    self.store_files(self.store_dir + 'val/', 'val',
                     int(split_frac[1] * data_size), data_size)
    # separating training split for creating training bags for LLP
    self.train_split = self.data[0:int(split_frac[0]*data_size)]

  def create_bags(self,
                  id1,
                  id2,
                  min_bag_threshold,
                  bag_per_file,
                  sample_per_file,
                  max_bag_size):
    """Creating LLP bags by combining 'id1' and 'id2' of categorical columns."""
    id1_name = self.train_split.columns.values[12 + id1]
    id2_name = self.train_split.columns.values[12 + id2]
    # creating new bag-id
    self.train_split['bag_id'] = self.train_split[id1_name].astype(
        str) + ':' + self.train_split[id2_name].astype(str)
    self.merge_bags(min_bag_threshold)
    self.store_bags(self.store_dir + 'bags' + str(id1) + '_' + str(id2) + '/',
                    bag_per_file, sample_per_file, max_bag_size)

  def merge_bags(self, min_bag_threshold):
    """Combine all |bags| <='min_bag_threshold' & divide into 'v' bags."""
    print('Init. unique bag_ids: ' + str(self.train_split['bag_id'].nunique()))
    for i in range(1, min_bag_threshold):
      print('\n' + str(i) + ' frequent items.', end=' ')
      value_counts = self.train_split['bag_id'].value_counts(
      )  # Specific column
      to_remove = value_counts[value_counts == i].index
      print('removing: ' + str(len(to_remove)) + ' ids.')

      s, e, v = 0, 0, 25
      part_len = int(len(to_remove) / v)

      if part_len <= 1:
        break
      while True:
        e = min(s + part_len, len(to_remove))
        print('bag:' + str(i) + ':' + str(s), end=' ')
        self.train_split['bag_id'].loc[self.train_split['bag_id'].isin(
            to_remove[s:e])] = 'bag:' + str(i) + ':' + str(s)
        s = e
        if s >= len(to_remove):
          break
    print('Frequent unique bag_ids  :  ' +
          str(self.train_split['bag_id'].nunique()))

  def store_bags(self, store_dir, bag_per_file, sample_per_file, max_bag_size):
    """Storing LLP training bags to 'store_dir'."""
    vc = self.train_split['bag_id'].value_counts()
    idx = vc[vc < max_bag_size].index
    idx = idx.tolist()
    print('Total no. of bags: '+ str(len(idx)))
    if not os.path.exists(store_dir):
      os.makedirs(store_dir)
    start_df, my_df = True, None
    bag_cnt = 0
    s = 0
    while s < len(idx):
      e = min(s+bag_per_file, len(idx))
      bag_df = self.train_split[self.train_split['bag_id'].isin(idx[s:e])]
      print(str(s)+ ' : '+  str(e) +  ' === '+ str(bag_df.shape))
      s = e
      if start_df:
        my_df = bag_df
        start_df = False
      else:
        my_df = pd.concat([my_df, bag_df], axis=0)
      if my_df.shape[0] >= sample_per_file:
        start_df = True
        f_name = store_dir+'/bag_'+str(bag_cnt)+'.csv'
        print('writing at ' + f_name + ' : ' + str(my_df.shape))
        my_df.to_csv(f_name, sep=',', index=True, header=True)
        # reset
        my_df = None
        start_df = True
        bag_cnt += 1
      if not start_df:
        f_name = store_dir+'/bag_'+str(bag_cnt)+'.csv'
        my_df.to_csv(f_name, sep=',', index=False, header=True)

  def add_col_names(self):
    column_names = ['label']
    for i in range(13):
      column_names.append('I'+str(i+1))
    for i in range(26):
      column_names.append('C'+str(i+1))
    self.data.columns = column_names
    cols = self.data.columns.values
    self.dense_feats = [f for f in cols if f[0] == 'I']
    self.sparse_feats = [f for f in cols if f[0] == 'C']

  def process_infreq(self, threshold=15):
    """For sparse_feats, replace infrequent entries (<= threshold) to None."""
    total_unique = 0
    for col in self.sparse_feats:
      print(col + ':' + str(self.data[col].nunique()))
      value_counts = self.data[col].value_counts()
      to_remove = value_counts[value_counts < threshold].index
      to_keep = self.data[col].nunique()-len(to_remove)
      total_unique += to_keep
      print(col + ':' + str(to_keep) + ':' + str(len(to_remove)))
      self.data[col].loc[self.data[col].isin(to_remove)] = np.nan
    print('Total unique: ' + str(total_unique))

  def rename(self, data_sparse):
    for col in self.sparse_feats:
      oldname = col
      newname = col + ' ' + str(max(data_sparse[col]))
      data_sparse.rename(columns={oldname: newname}, inplace=True)
    return data_sparse

  def process_feats(self):
    data_dense = self.data[self.dense_feats].fillna(0.0)
    data_sparse = self.data[self.sparse_feats].fillna('-1')
    for f in self.sparse_feats:
      label_encoder = LabelEncoder()
      data_sparse[f] = label_encoder.fit_transform(data_sparse[f])
    total_data = pd.concat([data_dense, self.rename(data_sparse)], axis=1)
    total_data['label'] = self.data['label']
    self.data = total_data

  def store_files(self,
                  store_dir,
                  file_prefix,
                  f_start,
                  f_end,
                  chunk_size=5000):
    """Given the data splits, divide and store them at 'store_dir'."""
    cnt = 0
    start = f_start
    if not os.path.exists(store_dir):
      os.makedirs(store_dir)
    while True:
      end = start + chunk_size
      if end >= f_end:
        end = f_end
      f_name = store_dir + '/' +file_prefix+str(cnt)+'.csv'
      self.data[start:end].to_csv(f_name, sep=',', index=False, header=True)
      print(f_name + '   Ranges: ' + str(start) + '  : ' + str(end))
      cnt = cnt+1
      start = end
      if end >= f_end:
        break

if __name__ == '__main__':
  processor = CriteoPreprocess('train_small.txt', [0.75, 0.95], './data/')
  _id1, _id2 = 5, 7
  _min_bag_threshold, _bag_per_file = 5, 25
  _sample_per_file, _max_bag_size = 15000, 2500
  processor.create_bags(_id1, _id2, _min_bag_threshold, _bag_per_file,
                        _sample_per_file, _max_bag_size)
