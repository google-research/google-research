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

# Lint as: python3
"""Data readers for regression/ binary classification datasets."""

import gzip
import os.path as osp
import tarfile
from typing import Tuple, Dict, Union, Iterator, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow.compat.v1 as tf
gfile = tf.gfile

DATA_PATH = 'gs://nam_datasets/data'
DatasetType = Tuple[np.ndarray, np.ndarray]


def save_array_to_disk(filename,
                       np_arr,
                       allow_pickle = False):
  """Saves a np.ndarray to a specified file on disk."""
  with gfile.Open(filename, 'wb') as f:
    with gzip.GzipFile(fileobj=f) as outfile:
      np.save(outfile, np_arr, allow_pickle=allow_pickle)


def read_dataset(dataset_name,
                 header = 'infer',
                 names = None,
                 delim_whitespace = False):
  dataset_path = osp.join(DATA_PATH, dataset_name)
  with gfile.Open(dataset_path, 'r') as f:
    df = pd.read_csv(
        f, header=header, names=names, delim_whitespace=delim_whitespace)
  return df


def load_breast_data():
  """Load and return the Breast Cancer Wisconsin dataset (classification)."""

  breast = load_breast_cancer()
  feature_names = list(breast.feature_names)
  return {
      'problem': 'classification',
      'X': pd.DataFrame(breast.data, columns=feature_names),
      'y': breast.target,
  }


def load_adult_data():
  """Loads the Adult Income dataset.

  Predict whether income exceeds $50K/yr based on census data. Also known as
  "Census Income" dataset. For more info, see
  https://archive.ics.uci.edu/ml/datasets/Adult.

  Returns:
    A dict containing the `problem` type (regression or classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('adult.data', header=None)
  df.columns = [
      'Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
      'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Gender',
      'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income'
  ]
  train_cols = df.columns[0:-1]
  label = df.columns[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {'problem': 'classification', 'X': x_df, 'y': y_df}


def load_heart_data():
  """Loads the Heart Disease dataset.

  The Cleveland Heart Disease Data found in the UCI machine learning repository
  consists of 14 variables measured on 303 individuals who have heart disease.
  See https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci for more
  info.

  Returns:
    A dict containing the `problem` type (regression or classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('HeartDisease.csv')
  train_cols = df.columns[0:-2]
  label = df.columns[-2]
  x_df = df[train_cols]
  y_df = df[label]
  # Replace NaN values with the mode value in the column.
  for col_name in x_df.columns:
    x_df[col_name].fillna(x_df[col_name].mode()[0], inplace=True)
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_credit_data():
  """Loads the Credit Fraud Detection dataset.

  This dataset contains transactions made by credit cards in September 2013 by
  european cardholders. It presents transactions that occurred in 2 days, where
  we have 492 frauds out of 284,807 transactions. It is highly unbalanced, the
  positive class (frauds) account for 0.172% of all transactions.
  See https://www.kaggle.com/mlg-ulb/creditcardfraud for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('creditcard.csv')
  df = df.dropna()
  train_cols = df.columns[0:-1]
  label = df.columns[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_telco_churn_data():
  """Loads Telco Customer Churn dataset.

  Predict behavior to retain customers based on relevant customer data.
  See https://www.kaggle.com/blastchar/telco-customer-churn/ for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """
  df = read_dataset('WA_Fn-UseC_-Telco-Customer-Churn.csv')
  train_cols = df.columns[1:-1]  # First column is an ID
  label = df.columns[-1]
  x_df = df[train_cols]
  # Impute missing values
  x_df['TotalCharges'] = x_df['TotalCharges'].replace(' ', 0).astype('float64')
  y_df = df[label]  # 'Yes', 'No'.
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_mimic2_data():
  """Loads the preprocessed Mimic-II ICU Mortality prediction dataset.

  The task is to predict mortality rate in Intensive Care Units (ICUs) based on
  using data from the first 48 hours of the ICU stay. See
  https://mimic.physionet.org/ for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """

  # Create column names
  attr_dict_path = osp.join(DATA_PATH, 'mimic2/mimic2.dict')
  attributes = gfile.Open(attr_dict_path, 'r').readlines()
  column_names = [x.split(' ,')[0] for x in attributes]

  df = read_dataset(
      'mimic2/mimic2.data',
      header=None,
      names=column_names,
      delim_whitespace=True)
  train_cols = column_names[:-1]
  label = column_names[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_recidivism_data():
  """Loads the ProPublica COMPAS recidivism dataset.

  COMPAS is a proprietary score developed to predict re-cidivism risk, which is
  used to inform bail, sentencing and parole decisions. In 2016, ProPublica
  released recidivism data on defendants in Broward County, Florida. See
  https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis
  for more info.

  Returns:
    A dict containing the `problem` type (i.e. classification) and the
    input features `X` as a pandas.Dataframe and the labels `y` as a pd.Series.
  """

  # Create column names
  attr_dict_path = osp.join(DATA_PATH, 'recidivism/recid.attr')
  attributes = gfile.Open(attr_dict_path, 'r').readlines()
  column_names = [x.split(':')[0] for x in attributes]

  df = read_dataset(
      'recidivism/recid.data',
      header=None,
      names=column_names,
      delim_whitespace=True)
  train_cols = column_names[:-1]
  label = column_names[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'classification',
      'X': x_df,
      'y': y_df,
  }


def load_fico_score_data():
  """Loads the FICO Score dataset.

  The FICO score is a widely used proprietary credit score todetermine credit
  worthiness for loans in the United States. The FICO dataset is comprised of
  real-world anonymized credit applications made by customers and their assigned
  FICO Score, based on their credit report information. For more info, refer to
  https://community.fico.com/s/explainable-machine-learning-challenge.

  Returns:
    A dict containing the `problem` type (i.e. regression) and the
    input features `X` as a pandas.Dataframe and the FICO scores `y` as
    np.ndarray.
  """

  # Create column names
  attr_dict_path = osp.join(DATA_PATH, 'fico/fico_score.attr')
  attributes = gfile.Open(attr_dict_path, 'r').readlines()
  column_names = [x.split(':')[0] for x in attributes]

  df = read_dataset(
      'fico/fico_score.data',
      header=None,
      names=column_names,
      delim_whitespace=True)
  train_cols = column_names[:-1]
  label = column_names[-1]
  x_df = df[train_cols]
  y_df = df[label]
  return {
      'problem': 'regression',
      'X': x_df,
      'y': y_df.values,
  }


def load_california_housing_data(
):
  """Loads the California Housing dataset.

  California  Housing  dataset is a canonical machine learning dataset derived
  from the 1990 U.S. census to understand the influence of community
  characteristics on housing prices. The task is regression to predict the
  median price of houses (in million dollars) in each district in California.
  For more info, refer to
  https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset.

  Returns:
    A dict containing the `problem` type (i.e. regression) and the
    input features `X` as a pandas.Dataframe and the regression targets `y` as
    np.ndarray.
  """
  feature_names = [
      'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
      'Latitude', 'Longitude'
  ]

  archive_path = osp.join(DATA_PATH, 'cal_housing.tgz')
  with gfile.Open(archive_path, 'rb') as fileobj:
    with tarfile.open(fileobj=fileobj, mode='r:gz') as f:
      cal_housing = np.loadtxt(
          f.extractfile('CaliforniaHousing/cal_housing.data'), delimiter=',')
      # Columns are not in the same order compared to the previous
      # URL resource on lib.stat.cmu.edu
      columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
      cal_housing = cal_housing[:, columns_index]

  target, data = cal_housing[:, 0], cal_housing[:, 1:]

  # avg rooms = total rooms / households
  data[:, 2] /= data[:, 5]

  # avg bed rooms = total bed rooms / households
  data[:, 3] /= data[:, 5]

  # avg occupancy = population / households
  data[:, 5] = data[:, 4] / data[:, 5]

  # target in units of 100,000
  target = target / 100000.0

  return {
      'problem': 'regression',
      'X': pd.DataFrame(data, columns=feature_names),
      'y': target,
  }


class CustomPipeline(Pipeline):
  """Custom sklearn Pipeline to transform data."""

  def apply_transformation(self, x):
    """Applies all transforms to the data, without applying last estimator.

    Args:
      x: Iterable data to predict on. Must fulfill input requirements of first
        step of the pipeline.

    Returns:
      xt: Transformed data.
    """
    xt = x
    for _, transform in self.steps[:-1]:
      xt = transform.fit_transform(xt)
    return xt


def transform_data(df):
  """Apply a fixed set of transformations to the pd.Dataframe `df`.

  Args:
    df: Input dataframe containing features.

  Returns:
    Transformed dataframe and corresponding column names. The transformations
    include (1) encoding categorical features as a one-hot numeric array, (2)
    identity `FunctionTransformer` for numerical variables. This is followed by
    scaling all features to the range (-1, 1) using min-max scaling.
  """
  column_names = df.columns
  new_column_names = []
  is_categorical = np.array([dt.kind == 'O' for dt in df.dtypes])
  categorical_cols = df.columns.values[is_categorical]
  numerical_cols = df.columns.values[~is_categorical]
  for index, is_cat in enumerate(is_categorical):
    col_name = column_names[index]
    if is_cat:
      new_column_names += [
          '{}: {}'.format(col_name, val) for val in set(df[col_name])
      ]
    else:
      new_column_names.append(col_name)
  cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

  cat_pipe = Pipeline([cat_ohe_step])
  num_pipe = Pipeline([('identity', FunctionTransformer(validate=True))])
  transformers = [('cat', cat_pipe, categorical_cols),
                  ('num', num_pipe, numerical_cols)]
  column_transform = ColumnTransformer(transformers=transformers)

  pipe = CustomPipeline([('column_transform', column_transform),
                         ('min_max', MinMaxScaler((-1, 1))), ('dummy', None)])
  df = pipe.apply_transformation(df)
  return df, new_column_names


def load_dataset(dataset_name):
  """Loads the dataset according to the `dataset_name` passed.

  Args:
    dataset_name: Name of the dataset to be loaded.

  Returns:
    data_x: np.ndarray of size (n_examples, n_features) containining the
      features per input data point where n_examples is the number of examples
      and n_features is the number of features.
    data_y: np.ndarray of size (n_examples, ) containing the label/target
      for each example where n_examples is the number of examples.
    column_names: A list containing the feature names.

  Raises:
    ValueError: If the `dataset_name` is not in ('Telco', 'BreastCancer',
    'Adult', 'Credit', 'Heart', 'Mimic2', 'Recidivism', 'Fico', Housing').
  """
  if dataset_name == 'Telco':
    dataset = load_telco_churn_data()
  elif dataset_name == 'BreastCancer':
    dataset = load_breast_data()
  elif dataset_name == 'Adult':
    dataset = load_adult_data()
  elif dataset_name == 'Credit':
    dataset = load_credit_data()
  elif dataset_name == 'Heart':
    dataset = load_heart_data()
  elif dataset_name == 'Mimic2':
    dataset = load_mimic2_data()
  elif dataset_name == 'Recidivism':
    dataset = load_recidivism_data()
  elif dataset_name == 'Fico':
    dataset = load_fico_score_data()
  elif dataset_name == 'Housing':
    dataset = load_california_housing_data()
  else:
    raise ValueError('{} not found!'.format(dataset_name))

  data_x, data_y = dataset['X'].copy(), dataset['y'].copy()
  problem_type = dataset['problem']
  data_x, column_names = transform_data(data_x)
  data_x = data_x.astype('float32')
  if (problem_type == 'classification') and \
      (not isinstance(data_y, np.ndarray)):
    data_y = pd.get_dummies(data_y).values
    data_y = np.argmax(data_y, axis=-1)
  data_y = data_y.astype('float32')
  return data_x, data_y, column_names


def get_train_test_fold(
    data_x,
    data_y,
    fold_num,
    num_folds,
    stratified = True,
    random_state = 42):
  """Returns a specific fold split for K-Fold cross validation.

  Randomly split dataset into `num_folds` consecutive folds and returns the fold
  with index `fold_index` for testing while the `num_folds` - 1 remaining folds
  form the training set.

  Args:
    data_x: Training data, with shape (n_samples, n_features), where n_samples
      is the number of samples and n_features is the number of features.
    data_y: The target variable, with shape (n_samples), for supervised learning
      problems.  Stratification is done based on the y labels.
    fold_num: Index of fold used for testing.
    num_folds: Number of folds.
    stratified: Whether to preserve the percentage of samples for each class in
      the different folds (only applicable for classification).
    random_state: Seed used by the random number generator.

  Returns:
    (x_train, y_train): Training folds containing 1 - (1/`num_folds`) fraction
      of entire data.
    (x_test, y_test): Test fold containing 1/`num_folds` fraction of data.
  """
  if stratified:
    stratified_k_fold = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=random_state)
  else:
    stratified_k_fold = KFold(
        n_splits=num_folds, shuffle=True, random_state=random_state)
  assert fold_num <= num_folds and fold_num > 0, 'Pass a valid fold number.'
  for train_index, test_index in stratified_k_fold.split(data_x, data_y):
    if fold_num == 1:
      x_train, x_test = data_x[train_index], data_x[test_index]
      y_train, y_test = data_y[train_index], data_y[test_index]
      return (x_train, y_train), (x_test, y_test)
    else:
      fold_num -= 1


def split_training_dataset(
    data_x,
    data_y,
    n_splits,
    stratified = True,
    test_size = 0.125,
    random_state = 1337):
  """Yields a generator that randomly splits data into (train, validation) set.

  The train set is used for fitting the DNNs/NAMs while the validation set is
  used for early stopping.

  Args:
    data_x: Training data, with shape (n_samples, n_features), where n_samples
      is the number of samples and n_features is the number of features.
    data_y: The target variable, with shape (n_samples), for supervised learning
      problems.  Stratification is done based on the y labels.
    n_splits: Number of re-shuffling & splitting iterations.
    stratified: Whether to preserve the percentage of samples for each class in
      the (train, validation) splits. (only applicable for classification).
    test_size: The proportion of the dataset to include in the validation split.
    random_state: Seed used by the random number generator.

  Yields:
    (x_train, y_train): The training data split.
    (x_validation, y_validation): The validation data split.
  """
  if stratified:
    stratified_shuffle_split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)
  else:
    stratified_shuffle_split = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state)
  split_gen = stratified_shuffle_split.split(data_x, data_y)

  for train_index, validation_index in split_gen:
    x_train, x_validation = data_x[train_index], data_x[validation_index]
    y_train, y_validation = data_y[train_index], data_y[validation_index]
    assert x_train.shape[0] == y_train.shape[0]
    yield (x_train, y_train), (x_validation, y_validation)
