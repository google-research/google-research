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

"""Preprocessing code to prepare dataset.

We download and keep the original Movielens-1M dataset in the same directory
as this code. We randomly split the dataset as training:test :: 80:20 ratio.
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MovieLensDataset():
  """Preprocessing code for MovieLens-1M dataset."""

  def __init__(self, data_path):
    self.label_encoder = LabelEncoder()
    self.load_data(data_path)
    self.rename()

  def load_data(self, data_path):
    """Main function to combine different files for ML-1M dataset."""

    # loading from users.dat
    col_names = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    df_user = pd.read_csv(
        data_path + 'users.dat', sep='::', names=col_names, engine='python')
    for col in df_user.columns[1:]:
      df_user[col] = self.label_encoder.fit_transform(df_user[col])

    # loading from movies.dat
    col_names = ['item_id', 'title', 'genres']
    df_item = pd.read_csv(
        data_path + 'movies.dat',
        sep='::',
        names=col_names,
        encoding='ISO-8859-1',
        engine='python')
    year = df_item.title.str[-5:-1].astype('int')
    df_item['years'] = self.label_encoder.fit_transform(year)

    df_genres = df_item.genres.str.split('|').str.join('|').str.get_dummies()
    df_genres['genre'] = ''
    for col in df_genres.columns:
      df_genres['genre'] += df_genres[col].map(str)
      if col != 'genre':
        del df_genres[col]
    df_genres['genre'] = self.label_encoder.fit_transform(df_genres['genre'])
    df_item.drop(['title', 'genres'], axis=1, inplace=True)
    df_item = pd.concat([df_item, df_genres], axis=1)

    # loading from rating.dat
    col_names = ['user_id', 'item_id', 'rating', 'timestamp']
    self.df_data = pd.read_csv(
        data_path + 'ratings.dat', sep='::', names=col_names, engine='python')
    self.df_data['timestamp'] = self.df_data['timestamp'] - self.df_data[
        'timestamp'].min()
    self.df_data['timestamp'] = self.df_data['timestamp'].apply(
        self.process_timestamp)
    self.df_data['timestamp'] = self.label_encoder.fit_transform(
        self.df_data['timestamp'])

    self.df_data = self.df_data[(
        ~self.df_data['rating'].isin([3]))].reset_index(drop=True)
    self.df_data['rating'] = np.where(self.df_data['rating'] > 3, 1, 0)
    self.df_data = self.df_data.merge(df_user, on='user_id', how='left')
    self.df_data = self.df_data.merge(df_item, on='item_id', how='left')

  def process_timestamp(self, val):
    v = int(val)
    return str(int(math.log(v) ** 2)) if v > 2 else str(v - 2)

  def rename(self):
    for col in self.df_data.columns:
      newname = col + ' ' + str(self.df_data[col].nunique())
      self.df_data.rename(columns={col: newname}, inplace=True)


if __name__ == '__main__':
  movie_lens = MovieLensDataset(data_path='')
  df_data = movie_lens.df_data.sample(frac=1)
  train_size = int(movie_lens.df_data.shape[0]*0.8)
  df_data.iloc[:train_size, 2:].to_csv(
      'movie_train.csv', sep=',', index=False, header=True)
  df_data.iloc[train_size:, 2:].to_csv(
      'movie_test.csv', sep=',', index=False, header=True)
