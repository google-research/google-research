# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Data loading and preprocessing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import urllib
import zipfile
import numpy as np
import pandas
from sklearn import preprocessing


def load_synthetic_data(data_name, dict_no, seed=0):
  """Generates synthetic datasets.

  This module generates 3 synthetic datasets and ground truth local dynamics.

  Args:
    data_name: Syn1, Syn2, Syn3
    dict_no: number of training, probe and testing sets
    seed: random seed

  Returns:
    x_train: training features
    y_train: training labels
    x_probe: probe features
    y_probe: probe labels
    x_test: testing features
    y_test: testing labels
    c_test: ground truth weights
  """

  np.random.seed(seed)

  # Parameters
  train_no = dict_no['train']
  probe_no = dict_no['probe']
  test_no = dict_no['test']
  dim_no = dict_no['dim']

  data_no = train_no + probe_no + test_no

  # Generates X (X ~ N(0,I))
  data_x = np.random.normal(0, 1, [data_no, dim_no])

  # Initializes labels (Y) and ground truth local dynamics (C)
  data_y = np.zeros([data_no,])
  data_c = np.zeros([data_no, 11])

  # Defines boundary
  if data_name == 'Syn1':
    idx0 = np.where(data_x[:, 9] < 0)[0]
    idx1 = np.where(data_x[:, 9] >= 0)[0]

  elif data_name == 'Syn2':
    idx0 = np.where(data_x[:, 9] + np.exp(data_x[:, 10]) < 1)[0]
    idx1 = np.where(data_x[:, 9] + np.exp(data_x[:, 10]) >= 1)[0]

  elif data_name == 'Syn3':
    idx0 = np.where(data_x[:, 9] + np.power(data_x[:, 10], 3) < 0)[0]
    idx1 = np.where(data_x[:, 9] + np.power(data_x[:, 10], 3) >= 0)[0]

  # Generates label (Y)
  data_y[idx0] = data_x[idx0, 0] + 2 * data_x[idx0, 1]
  data_y[idx1] = data_x[idx1, 2] + 2 * data_x[idx1, 3]

  # Generates ground truth local dynamics (C)
  data_c[idx0, 0] = 1.0
  data_c[idx0, 1] = 2.0

  data_c[idx1, 2] = 1.0
  data_c[idx1, 3] = 2.0

  # Splits training / probe / testing sets
  x_train = data_x[:train_no, :]
  x_probe = data_x[train_no:(train_no + probe_no), :]
  x_test = data_x[(train_no + probe_no):, :]

  y_train = data_y[:train_no]
  y_probe = data_y[train_no:(train_no + probe_no)]
  y_test = data_y[(train_no + probe_no):]

  # Defines ground truth local dynamics (C) for testing set
  c_test = data_c[(train_no + probe_no):, :]

  return x_train, y_train, x_probe, y_probe, x_test, y_test, c_test


def load_facebook_data(dict_rate, seed=0):
  """Loads Facebook Comment Volume dataset.

  This module loads load facebook comment volume dataset. Data link is
  https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset.
  It divides the entire data into training, probe, and testing sets and saves
  them as train.csv, prob.csv and test.csv in data_files folder.

  It divides the entire data into training, probe, and testing sets and saves
  them as train.csv, prob.csv and test.csv in data_files folder.

  Args:
    dict_rate: ratio between train and probe datasets
    seed: random seed
  """

  # Downloads zip file
  base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
  resp = urllib.request.urlopen('%s00363/Dataset.zip' % (base_url))
  zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

  # Loads train data from zip file
  for i in range(5):

    file_name = 'Dataset/Training/Features_Variant_%s.csv' % (i + 1)
    temp_data = pandas.read_csv(zip_file.open(file_name), header=None)

    if i == 0:
      data_train = temp_data
    else:
      data_train = pandas.concat((data_train, temp_data), axis=0)

  # Loads test data from zip file
  for i in range(9):

    file_name = 'Dataset/Testing/TestSet/Test_Case_%s.csv' % (i + 1)
    temp_data = pandas.read_csv(zip_file.open(file_name), header=None)

    if i == 0:
      data_test = temp_data
    else:
      data_test = pandas.concat((data_test, temp_data), axis=0)

  # Concatenates train and test
  df = pandas.concat((data_train, data_test), axis=0)

  # Removes the rows with missing data
  df = df.dropna()

  # Column names
  df.columns = ('Page Popularity/likes', 'Page Check', 'Page talk',
                'Page Category',
                'min # of comments', 'min # of comments in last 24 hours',
                'min # of comments in last 48 hours',
                'min # of comments in the first 24 hours',
                'min # of comments in last 48 to last 24 hours',
                'max # of comments', 'max # of comments in last 24 hours',
                'max # of comments in last 48 hours',
                'max # of comments in the first 24 hours',
                'max # of comments in last 48 to last 24 hours',
                'avg # of comments', 'avg # of comments in last 24 hours',
                'avg # of comments in last 48 hours',
                'avg # of comments in the first 24 hours',
                'avg # of comments in last 48 to last 24 hours',
                'median # of comments',
                'median # of comments in last 24 hours',
                'median # of comments in last 48 hours',
                'median # of comments in the first 24 hours',
                'median # of comments in last 48 to last 24 hours',
                'std # of comments', 'std # of comments in last 24 hours',
                'std # of comments in last 48 hours',
                'std # of comments in the first 24 hours',
                'std # of comments in last 48 to last 24 hours',
                '# of comments', '# of comments in last 24 hours',
                '# of comments in last 48 hours',
                '# of comments in the first 24 hours',
                '# of comments in last 48 to last 24 hours',
                'Base time', 'Post length', 'Post Share Count',
                'Post Promotion Status', 'H Local',
                'post was published on Sunday', 'post was published on Monday',
                'post was published on Tuesday',
                'post was published on Wednesday',
                'post was published on Thursday',
                'post was published on Friday',
                'post was published on Saturday',
                'basetime (Sunday)', 'basetime (Monday)', 'basetime (Tuesday)',
                'basetime (Wednesday)', 'basetime (Thursday)',
                'basetime (Friday)', 'basetime (Saturday)',
                'Y')

  # Splits train / probe / test sets
  # Resets index
  df = df.reset_index()
  df = df.drop(['index'], axis=1)

  # Parameters
  dict_no = dict()
  dict_no['train'] = int(len(df) * dict_rate['train'])
  dict_no['probe'] = int(len(df) * dict_rate['probe'])

  np.random.seed(seed)
  idx = np.random.permutation(len(data_train))
  train_idx = idx[:dict_no['train']]
  probe_idx = idx[dict_no['train']:(dict_no['train'] + dict_no['probe'])]
  test_idx = np.random.permutation(len(data_test)) + len(data_train)

  # Assigns train / probe / test set
  train = df.loc[train_idx]
  probe = df.loc[probe_idx]
  test = df.loc[test_idx]

  # Saves data
  if not os.path.exists('data_files'):
    os.makedirs('data_files')

  train.to_csv('./data_files/train.csv', index=False)
  probe.to_csv('./data_files/probe.csv', index=False)
  test.to_csv('./data_files/test.csv', index=False)


def preprocess_data(normalization,
                    train_file_name, probe_file_name, test_file_name):
  """Loads train / probe / test datasets and preprocess.

  This module loads datasets from data_files folder.
  Then, it normalizes the features and divides features and labels.

  Args:
    normalization: 'minmax' or 'standard'
    train_file_name: file name of training datasets
    probe_file_name: file name of probe datasets
    test_file_name: file name of testing datasets

  Returns:
    x_train: training features
    y_train: training labels
    x_probe: probe features
    y_probe: probe labels
    x_test: testing features
    y_test: testing labels
    col_names: column names
  """

  # Loads datasets
  train = pandas.read_csv('./data_files/%s' % (train_file_name))
  probe = pandas.read_csv('./data_files/%s' % (probe_file_name))
  test = pandas.read_csv('./data_files/%s' % (test_file_name))

  # Extracts label
  y_train = np.asarray(train['Y'])
  y_probe = np.asarray(probe['Y'])
  y_test = np.asarray(test['Y'])

  # Drops labels
  train = train.drop(columns=['Y'])
  probe = probe.drop(columns=['Y'])
  test = test.drop(columns=['Y'])

  # Sets column names
  col_names = train.columns.values.astype(str)

  # Combines train, probe, test for normalization
  df = pandas.concat((train, probe, test), axis=0)

  # Normalizes
  if normalization == 'minmax':
    scaler = preprocessing.MinMaxScaler()
  elif normalization == 'standard':
    scaler = preprocessing.StandardScaler()

  scaler.fit(df)
  df = scaler.transform(df)

  # Divides into train / probe / test sets
  train_no = len(train)
  probe_no = len(probe)
  test_no = len(test)

  x_train = df[range(train_no), :]
  x_probe = df[range(train_no, train_no + probe_no), :]
  x_test = df[range(train_no + probe_no, train_no + probe_no + test_no), :]

  return x_train, y_train, x_probe, y_probe, x_test, y_test, col_names
