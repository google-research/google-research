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

"""Instance level batch generation."""

import random
import threading

import numpy as np
import pandas as pd


class MiniBatchGenerator:
  """Generate instance level batches."""

  def __init__(self, data, batch_size, sparse_feats, bool_shuffel=True):
    self.start = 0
    self.batch_size = batch_size
    self.end = batch_size
    self.data = data
    self.bool_shuffel = bool_shuffel
    self.lock = threading.Lock()
    self.restart = False
    self.sparse_feats = sparse_feats

  def __iter__(self):
    return self

  def __next__(self):
    with self.lock:
      if self.restart:
        self.start = 0
      self.end = min(self.start + self.batch_size, self.data.shape[0])
      x_batch = [
          self.data[self.start:self.end][f].values for f in self.sparse_feats
      ]
      y_batch = [self.data[self.start:self.end]['rating 2'].values]

      self.start = self.end
      self.restart = False
      if self.end >= self.data.shape[0]:
        self.restart = True
      return x_batch, np.array(y_batch).transpose().astype('float32')


class BagGenerator:
  """Generate bags."""

  def __init__(self, data, nb_bags, sparse_feats, model_type='selfclr_llp'):
    self.nb_bags = nb_bags
    self.data = data
    self.lock = threading.Lock()
    self.sparse_feats = sparse_feats
    self.name_lst = None
    self.first_process()
    self.model_type = model_type

  def __iter__(self):
    return self

  def first_process(self):
    self.data['bag_id'] = self.data['zip_code 3439'].astype(str)
    self.data.set_index(keys=['bag_id'], drop=False, inplace=True)
    self.name_lst = self.data['bag_id'].unique().tolist()
    return

  def update_label(self, bag_df):
    total_pos = sum(bag_df['rating 2'])
    total = bag_df.shape[0]
    bag_df[0:1]['rating 2'] = total
    bag_df[1:2]['rating 2'] = total_pos
    return

  def process_selfclrllp(self, y):
    s_id = 0
    y2 = np.concatenate([y, y], axis=1)
    while s_id < y.shape[0]:
      y2[s_id, 1] = 0
      y2[s_id + 1, 0] = y[s_id + 1, 0] / y[s_id, 0]
      y2[s_id + 1, 1] = 1 - y2[s_id + 1, 0]
      s_id = s_id + int(y[s_id])
    return y2

  def process_dllp(self, y):
    s_id = 0
    while s_id < y.shape[0]:
      y[s_id+1] = y[s_id+1]/ y[s_id]
      s_id = s_id + int(y[s_id])
    return y

  def __next__(self):
    with self.lock:
      start_df = True
      my_df = None
      bag_cnt = 0
      while True:
        name_cnt = random.randint(0, len(self.name_lst) - 1)
        bag_df = self.data.loc[self.data.bag_id == self.name_lst[name_cnt]]
        if start_df:
          self.update_label(bag_df)
          my_df = bag_df
          bag_cnt += 1
          start_df = False
        else:
          self.update_label(bag_df)
          my_df = pd.concat([my_df, bag_df], axis=0)
          bag_cnt += 1
        if bag_cnt >= self.nb_bags:
          break
      x_batch = [my_df[f].values for f in self.sparse_feats]
      y_batch = [my_df['rating 2'].values]
      bag_y = np.array(y_batch).transpose().astype('float32')
      if self.model_type == 'selfclr_llp':
        return x_batch, self.process_selfclrllp(bag_y)
      else:
        return x_batch, self.process_dllp(bag_y)
