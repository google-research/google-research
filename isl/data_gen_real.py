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

"""Functions used for loading, preprocessing and saving real-world datasets."""

import argparse
import glob
import os

import metrics
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import utils


# pylint: disable=invalid-name
# pylint: disable=f-string-without-interpolation
# pylint: disable=unused-variable
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
def read_sachs_all(folder_path):
  """Reads all the sachs data specified in the folder_path.

  Args:
    folder_path: str specifying the folder containing the sachs data

  Returns:
    An np.array containing all the sachs data
  """

  sachs_data = list()

  # Divides the Sachs dataset into environments.
  for _, file in enumerate(glob.glob(f'{folder_path}*.xls')):
    sachs_df = pd.read_excel(file)
    sachs_array = sachs_df.to_numpy()
    sachs_data.append(sachs_array)

  sachs_data_envs = np.vstack(sachs_data)

  return sachs_data_envs


def preprocess_sachs(folder_path,
                     save_path,
                     num_envs,
                     normalization='standard'):
  """Preprocesses Sachs.

  Args:
    folder_path: Read sachs from the path
    save_path: Save data in this path
    num_envs: The number of environments to cluster the data into
    normalization: Normalization option

  Returns:
    The sachs dataset with different envs [num_envs, number of sample in
    each envs, num of features]

  """

  np.set_printoptions(precision=3)
  X = read_sachs_all(folder_path)
  _, d = X.shape
  kmeans = KMeans(n_clusters=num_envs, max_iter=1000).fit(X)
  labeled_X = kmeans.fit_predict(X)
  X_envs = utils.classify_x(X, labeled_X)  # X_cluster is a dict
  X, Y = utils.preprocess_labeled_data(X, labeled_X, normalization)
  X_res, Y_res = utils.over_sampling(X, Y)
  X_envs = utils.classify_x(X_res, Y_res)

  os.makedirs(save_path, exist_ok=True)
  for i, X in X_envs.items():
    exp = save_path + f'sachs_env_{i+1}_{num_envs}.csv'
    if X.shape[0] > 1:
      np.savetxt(exp, X, delimiter=',')

  return utils.distribute_to_envs(X_envs)


def read_sachs_to_envs(folder_path, num_envs, normalization):
  """Loads Sachs data and return sachs data divided into different environments.

  Args:
    folder_path:
    num_envs: the number of envs to read the data
    normalization: normalization type

  Returns:
    A tensor with shape [num_envs, number of sample in each envs, num of
    features]
  """
  sachs_data = list()

  if num_envs == 14:

    y_label = []
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()
      sachs_array = utils.preprocess(sachs_array, normalization)
      sachs_data.append(sachs_array)
      y_label.append(np.ones(sachs_array.shape[0]) * i)

    sachs_data_envs = np.vstack(sachs_data)
    sachs_data_labels = np.hstack(y_label)
    X_res, Y_res = utils.over_sampling(sachs_data_envs, sachs_data_labels)
    X_cluster = utils.classify_x(X_res, Y_res)
    X_envs = utils.distribute_to_envs(X_cluster)

  elif num_envs == 2:

    X_envs = [None] * num_envs
    y_label = [None] * num_envs
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      start_index = file.index('sachs_data/') + 11
      end_index = file.index(' ') - 1
      file_index = int(file[start_index:end_index])
      label = 0 if file_index <= 9 else 1

      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()
      if X_envs[label] is None:
        X_envs[label] = sachs_array
        y_label[label] = np.ones(sachs_array.shape[0]) * label
      else:
        X_envs[label] = np.concatenate((X_envs[label], sachs_array), axis=0)
        y_label[label] = np.concatenate(
            (y_label[label], (np.ones(sachs_array.shape[0]) * label)), axis=0)

    for i in range(num_envs):
      X_envs[i], y_label[i] = utils.preprocess_labeled_data(
          X_envs[i], y_label[i], normalization)

    X = np.vstack(X_envs)
    Y = np.hstack(y_label)
    X_res, Y_res = utils.over_sampling(X, Y)
    X_cluster = utils.classify_x(X_res, Y_res)
    X_envs = utils.distribute_to_envs(X_cluster)

  elif num_envs == 3:

    if not os.path.exists(f'./data/cluster/*3.csv'):
      X_envs = preprocess_sachs(folder_path, f'./data/cluster/', 3,
                                normalization)
    else:
      for file in glob.glob(f'./data/cluster/*3.csv'):
        sachs_array = np.loadtxt(file, delimiter=',')
        # sachs_array = preprocess(sachs_array, args=args)
        sachs_data.append(sachs_array)
        X_envs = np.stack(sachs_data)

  elif num_envs == 6:

    if not os.path.exists(f'./data/cluster/*3.csv'):
      X_envs = preprocess_sachs(folder_path, f'./data/cluster/', 6,
                                normalization)
    else:
      for file in glob.glob(f'./data/cluster/*6.csv'):
        sachs_array = np.loadtxt(file, delimiter=',')
        sachs_array = utils.preprocess(sachs_array, normalization)
        sachs_data.append(sachs_array)
        X_envs = np.stack(sachs_data)

  elif num_envs == 7:

    X_envs = [None] * num_envs
    y_label = [None] * num_envs
    for i, file in enumerate(glob.glob(f'{folder_path}*.xls')):
      start_index = file.index('sachs_data/') + 11
      end_index = file.index(' ') - 1
      file_index = int(file[start_index:end_index])
      label = file_index % num_envs

      sachs_df = pd.read_excel(file)
      sachs_array = sachs_df.to_numpy()

      if X_envs[label] is None:
        X_envs[label] = sachs_array
        y_label[label] = np.ones(sachs_array.shape[0]) * label
      else:
        X_envs[label] = np.concatenate((X_envs[label], sachs_array), axis=0)
        y_label[label] = np.concatenate(
            (y_label[label], (np.ones(sachs_array.shape[0]) * label)), axis=0)

    for i in range(num_envs):
      X_envs[i], y_label[i] = utils.preprocess_labeled_data(
          X_envs[i], y_label[i], normalization)

    X = np.vstack(X_envs)
    Y = np.hstack(y_label)
    X_res, Y_res = utils.over_sampling(X, Y)
    X_cluster = utils.classify_x(X_res, Y_res)
    X_envs = utils.distribute_to_envs(X_cluster)

  return X_envs


def find_opt_k(x_raw):
  """Plots the wss and silhouse score for the optimal k for clustering X_raw."""

  label = np.ones(len(x_raw))
  x, x_test, _, _ = train_test_split(
      x_raw, label, test_size=0.1, random_state=42)

  # X_train_val, X_test, _, _ = train_test_split(
  #  X_raw, label, test_size=0.1, random_state=42)
  # X, X_val, _, _ = train_test_split(
  #  X_train_val, np.ones(len(X_train_val)), test_size=0.1, random_state=1)
  n, d = x.shape
  wss = metrics.calculate_wss(x, d)
  sil = metrics.compute_silhouette_score(x, d)
  metrics.visulize_tsne(x)
  metrics.plot_metrics(x, d)
  metrics.visulize_tsne(x)


def preprocess_BH(save_path, cluster):
  """Clusters the data and prerpocess it.

  Args:
    save_path: the path to save the proprocessed data
    cluster: number of clusters to cluster the dataset

  Returns:
    A tensor of [num_envs, number of sample in each envs, num of features]
  """
  np.set_printoptions(precision=3)

  x_raw = utils.load_BH()
  label = np.ones(len(x_raw))

  # only train and test
  x, x_test, _, _ = train_test_split(
      x_raw, label, test_size=0.1, random_state=42)

  # train val and test
  # X_train_val, X_test, _, _ = train_test_split(
  #  X_raw, label, test_size=0.1, random_state=42)
  # X, X_val, _, _ = train_test_split(
  # X_train_val, np.ones(len(X_train_val)), test_size=0.1, random_state=1)

  n, d = x.shape
  kmeans = KMeans(n_clusters=cluster, max_iter=1000).fit(x)
  labeled_x = kmeans.fit_predict(x)
  x_cluster = utils.classify_x(x, labeled_x)  # X_cluster is a dict

  os.makedirs(save_path, exist_ok=True)

  x_upsampled, y_upsampled = utils.over_sampling(x, labeled_x)
  x_envs = utils.classify_x(x_upsampled, y_upsampled)
  standard_scaler = preprocessing.StandardScaler()

  x_train = list()
  i = 1

  # Save the data for different envs
  for x_env in x_envs.items():
    standardx = standard_scaler.fit_transform(x_env[1])
    exp = save_path + f'standard_BH_env_{i}_{cluster}.csv'
    np.savetxt(exp, standardx, fmt='%.3f', delimiter=',')
    i += 1
    x_train.append(standardx)

  # Standard the train test dataset using the mean and std of the train dataset
  standard_train_param = standard_scaler.fit(x)
  standardxtrain = standard_scaler.transform(x)
  x_trainexp = save_path + f'standard_BH_train.csv'
  np.savetxt(x_trainexp, standardxtrain, fmt='%.3f', delimiter=',')

  standardxtest = standard_scaler.transform(x_test)
  x_testexp = save_path + f'standard_BH_test.csv'
  np.savetxt(x_testexp, standardxtest, fmt='%.3f', delimiter=',')

  return np.stack(x_train), x_testexp


def cluster_Insurance(x, num_cluster):
  """Clusters and returns the data clustered into different environments.

  Args:
   x: the data to be clustered
   num_cluster: the number of clusters

  Returns:
    np.array: the data clustered into different environments
  """
  _, d = x.shape
  kmeans = KMeans(n_clusters=num_cluster, max_iter=1000).fit(x)
  labeled_x = kmeans.fit_predict(x)
  x_cluster = utils.classify_x(x, labeled_x)  # X_cluster is a dict
  x_upsampled, y_upsampled = utils.over_sampling(x, labeled_x)
  x_envs = utils.classify_x(x_upsampled, y_upsampled)

  return x_envs


def load_insurance(file_path, model_type='ISL'):
  """Loads Insurance dataset based on the model_type.

  Args:
    file_path: the file path of insurance dataset
    model_type: the model that utilizes the loaded data ISL requires train data
      from different envs other models require train and test data

  Returns:
    clustered data.
  """
  x_raw = pd.read_csv(file_path, delimiter=',').to_numpy()
  label = np.ones(len(x_raw))

  # only train and test

  x, x_test, _, _ = train_test_split(
      x_raw, label, test_size=0.1, random_state=42)

  if model_type != 'ISL':
    return x, x_test
  else:
    return cluster_Insurance(x, 3), x_test


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(description='cluster dataset algorithm')
  parser.add_argument(
      '--dataset',
      type=str,
      default='BH',
      help='the name of the dataset to be loaded')

  parser.add_argument('--normalization', type=str, default='standard', help='')
  parser.add_argument(
      '--folder_path',
      type=str,
      default='sachs',
      help='which data set to cluster')
  parser.add_argument(
      '--num_cluster', type=int, default=3, help='number of clusters')

  args = parser.parse_args()
  return args


def load_data(dataset_name, preprocess, num_envs):
  """Loads the dataset according to the instructions.

  Args:
    dataset_name: name of the dataset to be loaded
    preprocess: preprocess method: standard or minimax
    num_envs: numbe of envs to create

  Returns:
    x: training data for ISL
    x_test: test data for ISL
  """
  if dataset_name == 'sachs':
    return read_sachs_to_envs(
        './data/sachs_data', num_envs=num_envs, normalization=preprocess)

  elif dataset_name == 'BH':
    x, x_test = preprocess_BH('./data/BH/', num_envs)
    return

  elif dataset_name == 'Insurance':
    x, x_test = load_insurance('./data/insurance_numeric.csv')
    return x, x_test

  else:
    ValueError('Invalid dataset name')

  return


if __name__ == '__main__':
  args = parse_args()
  x, x_test = load_data(args.dataset, args.normalization, args.num_cluster)
  sachs_envs = read_sachs_to_envs(args.folder_path, args.num_cluster,
                                  args.normalization)
