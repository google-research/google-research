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

"""Batch Generator Code."""

import os
import random
import threading

import numpy as np
import pandas as pd


class MiniBatchGenerator:
  """Generate instance level batches."""

  def __init__(self,
               data_dir,
               batch_size,
               dense_feats,
               sparse_feats):
    self.start = 0
    self.end = batch_size
    self.batch_size = batch_size
    self.data = None
    self.next_file = True
    self.file_id = 0
    self.data_dir = data_dir
    self.file_list = os.listdir(data_dir)
    self.lock = threading.Lock()
    self.dense_feats = dense_feats
    self.sparse_feats = sparse_feats

  def __iter__(self):
    return self

  def __next__(self):
    with self.lock:
      if self.next_file:
        f_name = self.data_dir + '/'+ self.file_list[self.file_id]
        self.data = pd.read_csv(f_name)
        self.process_dense_feats(self.data)
        self.start = 0
        self.file_id = (self.file_id + 1) % len(self.file_list)
      self.end = min(self.start + self.batch_size, self.data.shape[0])
      dense_x = [
          self.data[self.start:self.end][f].values for f in self.dense_feats
      ]
      sparse_x = [
          self.data[self.start:self.end][f].values for f in self.sparse_feats
      ]
      y = [self.data[self.start:self.end]['label'].values]
      self.start = self.end
      self.next_file = False
      if self.end >= self.data.shape[0]:
        self.next_file = True
      return dense_x + sparse_x, np.array(y).transpose().astype('float32')

  def process_dense_feats(self, data):
    for f in self.dense_feats:
      data[f] = data[f].apply(lambda x: np.log(x+1) if x > -1 else -1)


class BagGenerator:
  """Generate bags."""

  def __init__(self,
               data_dir,
               nb_bag,
               dense_feats,
               sparse_feats,
               model_type='selfclr_llp'):
    self.nb_bag = nb_bag
    self.max_size = 2500
    self.data = None
    self.next_file = True
    self.file_id = 0
    self.data_dir = data_dir
    self.file_list = os.listdir(data_dir)
    self.lock = threading.Lock()
    self.dense_feats = dense_feats
    self.sparse_feats = sparse_feats
    self.name_lst = None
    self.name_cnt = 0
    self.model_type = model_type

  def __iter__(self):
    return self

  def load_next(self):
    """Randomly select next bag."""
    f_name = self.data_dir + '/'+ self.file_list[self.file_id]
    self.data = pd.read_csv(f_name)
    self.process_dense_feats(self.data)
    self.data.set_index(keys=['bag_id'], drop=False, inplace=True)
    self.name_lst = self.data['bag_id'].unique().tolist()
    random.shuffle(self.name_lst)
    self.start = 0
    self.name_cnt = 0
    self.next_file = False
    self.file_id = (self.file_id + 1) % len(self.file_list)
    return

  def process_dense_feats(self, data):
    for f in self.dense_feats:
      data[f] = data[f].apply(lambda x: np.log(x+1) if x > -1 else -1)
    return

  def update_label(self, bag_df):
    total_pos = sum(bag_df['label'])
    total = bag_df.shape[0]
    bag_df[0:1]['label'] = total
    bag_df[1:2]['label'] = total_pos
    return

  def process_selfclrllp(self, y):
    s = 0
    y2 = np.concatenate([y, y], axis=1)
    while s < y.shape[0]:
      y2[s, 1] = 0
      y2[s + 1, 0] = y[s + 1, 0] / y[s, 0]
      y2[s + 1, 1] = 1 - y2[s + 1, 0]
      s = s + int(y[s])
    return y2

  def process_dllp(self, y):
    s = 0
    while s < y.shape[0]:
      y[s + 1] = y[s + 1] / y[s]
      s = s + int(y[s])
    return y

  def __next__(self):
    with self.lock:
      start_df = True
      my_df = None
      bag_cnt = 0

      while True:
        if self.next_file:
          self.load_next()
        bag_df = self.data.loc[self.data.bag_id == self.name_lst[self.name_cnt]]
        if bag_df.shape[0] > self.max_size:
          self.name_cnt += 1
        elif start_df:
          self.update_label(bag_df)
          my_df = bag_df
          self.name_cnt += 1
          bag_cnt += 1
          start_df = False
        else:
          self.update_label(bag_df)
          my_df = pd.concat([my_df, bag_df], axis=0)
          self.name_cnt += 1
          bag_cnt += 1
        # update conditions.
        if self.name_cnt >= len(self.name_lst):
          self.next_file = True
        if bag_cnt >= self.nb_bag:
          break
      # process the batch first
      dense_x = [my_df[f].values for f in self.dense_feats]
      sparse_x = [my_df[f].values for f in self.sparse_feats]
      bag_y = np.array([my_df['label'].values]).transpose().astype('float32')
      if self.model_type == 'selfclr_llp':
        return dense_x + sparse_x, self.process_selfclrllp(bag_y)
      else:
        return dense_x + sparse_x, self.process_dllp(bag_y)
