# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
import pandas as pd

from sklearn import preprocessing
from dvrl import dvrl_utils


def load_tabular_data(data_name, dict_no, noise_rate):
  """Loads Adult Income and Blog Feedback datasets.

  This module loads the two tabular datasets and saves train.csv, valid.csv and
  test.csv files under data_files directory.

  UCI Adult data link: https://archive.ics.uci.edu/ml/datasets/Adult
  UCI Blog data link: https://archive.ics.uci.edu/ml/datasets/BlogFeedback

  If noise_rate > 0.0, adds noise on the datasets.
  Then, saves train.csv, valid.csv, test.csv on './data_files/' directory

  Args:
    data_name: 'adult' or 'blog'
    dict_no: training and validation set numbers
    noise_rate: label corruption ratio

  Returns:
    noise_idx: indices of noisy samples
  """

  # Loads datasets from links
  uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

  # Adult Income dataset
  if data_name == 'adult':

    train_url = uci_base_url + 'adult/adult.data'
    test_url = uci_base_url + 'adult/adult.test'

    data_train = pd.read_csv(train_url, header=None)
    data_test = pd.read_csv(test_url, skiprows=1, header=None)

    df = pd.concat((data_train, data_test), axis=0)

    # Column names
    df.columns = ['Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
                  'MaritalStatus', 'Occupation', 'Relationship', 'Race',
                  'Gender', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek',
                  'NativeCountry', 'Income']

    # Creates binary labels
    df['Income'] = df['Income'].map({' <=50K': 0, ' >50K': 1,
                                     ' <=50K.': 0, ' >50K.': 1})

    # Changes string to float
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.CapitalGain = df.CapitalGain.astype(float)
    df.CapitalLoss = df.CapitalLoss.astype(float)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['WorkClass', 'Education', 'MaritalStatus',
                                     'Occupation', 'Relationship',
                                     'Race', 'Gender', 'NativeCountry'])

    # Sets label name as Y
    df = df.rename(columns={'Income': 'Y'})
    df['Y'] = df['Y'].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=['index'])

  # Blog Feedback dataset
  elif data_name == 'blog':

    resp = urllib.request.urlopen(uci_base_url + '00304/BlogFeedback.zip')
    zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

    # Loads train dataset
    train_file_name = 'blogData_train.csv'
    data_train = pd.read_csv(zip_file.open(train_file_name), header=None)

    # Loads test dataset
    for i in range(29):
      if i < 9:
        file_name = 'blogData_test-2012.02.0'+ str(i+1) + '.00_00.csv'
      else:
        file_name = 'blogData_test-2012.02.'+ str(i+1) + '.00_00.csv'

      temp_data = pd.read_csv(zip_file.open(file_name), header=None)

      if i == 0:
        data_test = temp_data
      else:
        data_test = pd.concat((data_test, temp_data), axis=0)

    for i in range(31):
      if i < 9:
        file_name = 'blogData_test-2012.03.0'+ str(i+1) + '.00_00.csv'
      elif i < 25:
        file_name = 'blogData_test-2012.03.'+ str(i+1) + '.00_00.csv'
      else:
        file_name = 'blogData_test-2012.03.'+ str(i+1) + '.01_00.csv'

      temp_data = pd.read_csv(zip_file.open(file_name), header=None)

      data_test = pd.concat((data_test, temp_data), axis=0)

    df = pd.concat((data_train, data_test), axis=0)

    # Removes rows with missing data
    df = df.dropna()

    # Sets label and named as Y
    df.columns = df.columns.astype(str)

    df['280'] = 1*(df['280'] > 0)
    df = df.rename(columns={'280': 'Y'})
    df['Y'] = df['Y'].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=['index'])

  # Splits train, valid and test sets
  train_idx = range(len(data_train))
  train = df.loc[train_idx]

  test_idx = range(len(data_train), len(df))
  test = df.loc[test_idx]

  train_idx_final = np.random.permutation(len(train))[:dict_no['train']]

  temp_idx = np.random.permutation(len(test))
  valid_idx_final = temp_idx[:dict_no['valid']] + len(data_train)
  test_idx_final = temp_idx[dict_no['valid']:] + len(data_train)

  train = train.loc[train_idx_final]
  valid = test.loc[valid_idx_final]
  test = test.loc[test_idx_final]

  # Adds noise on labels
  y_train = np.asarray(train['Y'])
  y_train, noise_idx = dvrl_utils.corrupt_label(y_train, noise_rate)
  train['Y'] = y_train

  # Saves data
  if not os.path.exists('data_files'):
    os.makedirs('data_files')

  train.to_csv('./data_files/train.csv', index=False)
  valid.to_csv('./data_files/valid.csv', index=False)
  test.to_csv('./data_files/test.csv', index=False)

  # Returns indices of noisy samples
  return noise_idx


def load_rossmann_data(dict_no, setting, test_store_type):
  """Loads Rossmann data.

  This module loads Rossmann data for a domain adaptation application.
  Rossmann data link: https://www.kaggle.com/c/rossmann-store-sales

  The users should download 'rossmann-store-sales.zip' from the above link and
  save it in './data_files/' directory

  Args:
    dict_no: the number of source and valid samples
    setting: 'train-on-all', 'train-on-rest', or 'train-on-specific'
    test_store_type: 'A', 'B', 'C', or 'D'
  """

  # Loads datasets
  zip_file = zipfile.ZipFile('./data_files/rossmann-store-sales.zip')
  train_data = pd.read_csv(zip_file.open('train.csv'))
  store_data = pd.read_csv(zip_file.open('store.csv'))

  # Extracts features
  train_data = train_data[['Store', 'Sales', 'DayOfWeek', 'Customers', 'Open',
                           'Promo', 'StateHoliday', 'SchoolHoliday']]
  store_data = store_data[['Store', 'StoreType', 'Assortment',
                           'CompetitionDistance', 'CompetitionOpenSinceMonth',
                           'Promo2', 'Promo2SinceWeek']]

  # Data preprocessing
  # Fill na to 0
  store_data = store_data.fillna(0)
  # Converts string to int
  train_data['StateHoliday'] = train_data['StateHoliday'].replace(['a', 'b',
                                                                   'c'], 1)

  # One-hot encoding
  store_data = pd.get_dummies(store_data)

  # Combines store data and train data
  data_x = pd.merge(train_data, store_data, on='Store')

  # Removes the samples when close
  remove_idx = data_x.index[data_x['Sales'] == 0].tolist()
  data_x = data_x.drop(remove_idx, axis=0)

  # Renames target variable to 'Y'
  data_x = data_x.rename(columns={'Sales': 'Y'})

  # Defines store types
  data_c = data_x[['StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d']]
  data_c = data_c.rename(columns={'StoreType_a': 'A', 'StoreType_b': 'B',
                                  'StoreType_c': 'C', 'StoreType_d': 'D'})

  # Defines features
  data_x = data_x.drop(['StoreType_a', 'StoreType_b',
                        'StoreType_c', 'StoreType_d'], axis=1)

  # Resets index
  data_x = data_x.reset_index()
  data_c = data_c.reset_index()

  data_x = data_x.drop(['index'], axis=1)
  data_c = data_c.drop(['index'], axis=1)

  # Splits source, valid, and target sets
  # Random partitioning
  idx = np.random.permutation(len(data_x))

  source_idx = idx[:dict_no['source']]
  valid_idx = idx[dict_no['source']:(dict_no['source']+dict_no['valid'])]
  target_idx = idx[(dict_no['source']+dict_no['valid']):]

  x_source = data_x.loc[source_idx]
  c_source = data_c.loc[source_idx]

  x_valid = data_x.loc[valid_idx]
  c_valid = data_c.loc[valid_idx]

  x_target = data_x.loc[target_idx]
  c_target = data_c.loc[target_idx]

  # Selects source dataset based on the setting and test_store_type
  if setting == 'train-on-all':
    source_sub_idx = c_source.index[c_source[test_store_type] >= 0].tolist()
  elif setting == 'train-on-rest':
    source_sub_idx = c_source.index[c_source[test_store_type] == 0].tolist()
  elif setting == 'train-on-specific':
    source_sub_idx = c_source.index[c_source[test_store_type] == 1].tolist()

  # Selects valid and target datasets based on test_store_type
  valid_sub_idx = c_valid.index[c_valid[test_store_type] == 1].tolist()
  target_sub_idx = c_target.index[c_target[test_store_type] == 1].tolist()

  # Divides source, valid, and target datasets
  source = x_source.loc[source_sub_idx]
  valid = x_valid.loc[valid_sub_idx]
  target = x_target.loc[target_sub_idx]

  source.to_csv('./data_files/source.csv', index=False)
  valid.to_csv('./data_files/valid.csv', index=False)
  target.to_csv('./data_files/target.csv', index=False)

  return


def preprocess_data(normalization,
                    train_file_name, valid_file_name, test_file_name):
  """Loads datasets, divides features and labels, and normalizes features.

  Args:
    normalization: 'minmax' or 'standard'
    train_file_name: file name of training set
    valid_file_name: file name of validation set
    test_file_name: file name of testing set

  Returns:
    x_train: training features
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    x_test: testing features
    y_test: testing labels
    col_names: column names
  """

  # Loads datasets
  train = pd.read_csv('./data_files/'+train_file_name)
  valid = pd.read_csv('./data_files/'+valid_file_name)
  test = pd.read_csv('./data_files/'+test_file_name)

  # Extracts label
  y_train = np.asarray(train['Y'])
  y_valid = np.asarray(valid['Y'])
  y_test = np.asarray(test['Y'])

  # Drops label
  train = train.drop(columns=['Y'])
  valid = valid.drop(columns=['Y'])
  test = test.drop(columns=['Y'])

  # Column names
  col_names = train.columns.values.astype(str)

  # Concatenates train, valid, test for normalization
  df = pd.concat((train, valid, test), axis=0)

  # Normalization
  if normalization == 'minmax':
    scaler = preprocessing.MinMaxScaler()
  elif normalization == 'standard':
    scaler = preprocessing.StandardScaler()

  scaler.fit(df)
  df = scaler.transform(df)

  # Divides df into train / valid / test sets
  train_no = len(train)
  valid_no = len(valid)
  test_no = len(test)

  x_train = df[range(train_no), :]
  x_valid = df[range(train_no, train_no + valid_no), :]
  x_test = df[range(train_no+valid_no, train_no+valid_no+test_no), :]

  return x_train, y_train, x_valid, y_valid, x_test, y_test, col_names
