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

"""Loads tabular datasets (UCI Adult Income and UCI Blog Feedback datasets).
"""

#%% Necessary packages and function call
# Zip file reading
import io
from urllib.request import urlopen
import zipfile

import numpy as np
import pandas as pd

# Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def tabular_data_loading(data_name, dict_no, normalization='minmax'):
  """Loads Adult Income and Blog Feedback datasets.

  Args:
    data_name: 'adult' or 'blog'
    dict_no: training and validation set numbers
    normalization: 'minmax' or 'standard'

  Returns:
    x_train: training features
    y_train: training labels
    x_valid: validation features
    y_valid: validation labels
    x_test: testing features
    y_test: testing labels
  """

  # Loads train & test datasets
  if data_name == 'adult':

    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
    train_url = base_url + 'adult.data'
    test_url = base_url + 'adult.test'

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
    data_y = np.asarray(df['Income'])

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

    # Exclude label information to make data_x
    df = df.drop('Income', axis=1)
    data_x = np.asarray(df)

  elif data_name == 'blog':

    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    resp = urlopen(base_url + '00304/BlogFeedback.zip')
    zip_file = zipfile.ZipFile(io.BytesIO(resp.read()))

    # Loads train data
    train_file_name = 'blogData_train.csv'
    data_train = pd.read_csv(zip_file.open(train_file_name), header=None)

    # Loads test data
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

    # Makes features and binary labels
    data_array = np.asarray(df)
    feature_no = len(data_array[0, :]) - 1

    data_x = data_array[:, 1:feature_no]
    data_y = 1*(data_array[:, feature_no] > 0)

  # Normalization
  if normalization == 'minmax':
    scaler = MinMaxScaler()
  elif normalization == 'standard':
    scaler = StandardScaler()

  scaler.fit(data_x)
  data_x = scaler.transform(data_x)

  # Splits train/test sets
  train_no = len(data_train)

  x_train = data_x[:train_no, :]
  y_train = data_y[:train_no]

  x_test = data_x[train_no:, :]
  y_test = data_y[train_no:]

  # Random partitioning
  train_idx = np.random.permutation(len(y_train))[:dict_no['train']]

  temp_idx = np.random.permutation(len(y_test))
  valid_idx = temp_idx[:dict_no['valid']]
  test_idx = temp_idx[dict_no['valid']:]

  y_train = y_train.astype(int)
  y_test = y_test.astype(int)

  x_train = x_train[train_idx, :]
  y_train = y_train[train_idx]

  x_valid = x_test[valid_idx, :]
  y_valid = y_test[valid_idx]

  x_test = x_test[test_idx, :]
  y_test = y_test[test_idx]

  return x_train, y_train, x_valid, y_valid, x_test, y_test
