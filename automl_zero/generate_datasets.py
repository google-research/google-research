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

"""Script for generating the binary classification datasets from CIFAR10/MNIST.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow_datasets as tfds

import task_pb2

flags.DEFINE_string(
    'data_dir', '/tmp/binary_cifar10_data/',
    'Path of the folder to save the datasets.')

flags.DEFINE_string(
    'tfds_data_dir', '/tmp/',
    'Path for tensorflow_datasets to cache downloaded datasets, '
    'only used in local runs.')

flags.DEFINE_integer('num_train_examples', 8000,
                     'Number of training examples in each dataset.')

flags.DEFINE_integer('num_valid_examples', 1000,
                     'Number of validation examples in each dataset.')

flags.DEFINE_integer('num_test_examples', 1000,
                     'Number of test examples in each dataset.')

flags.DEFINE_integer('projected_dim', 16,
                     'The dimensionality to project the data into.')

flags.DEFINE_string('dataset_name', 'cifar10',
                    'Name of the dataset to generatee '
                    'more binary classification datasets.')

flags.DEFINE_integer('min_data_seed', 0,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_integer('max_data_seed', 100,
                     'Generate one dataset for each seed in '
                     '[min_data_seed, max_data_seed).')

flags.DEFINE_list('class_ids', '0,1,2,3,4,5,6,7,8,9',
                  'Classes included to generate binary'
                  ' classification datasets.')

FLAGS = flags.FLAGS


def create_projected_binary_dataset(
    dataset_name, positive_class, negative_class,
    num_train_examples, num_valid_examples, num_test_examples,
    projected_dim, seed, load_fn):
  """Create a projected binary dataset from the given spec and seed."""
  num_samples = (
      num_train_examples +
      num_valid_examples +
      num_test_examples)
  pos = positive_class
  neg = negative_class
  # Only support training data from MNIST and CIFAR10 for experiments.
  data, labels, _, _ = get_dataset(
      dataset_name,
      int(num_samples / 2), [pos, neg], load_fn=load_fn)
  labels[np.where(labels == pos)] = -1
  labels[np.where(labels == neg)] = 0
  labels[np.where(labels == -1)] = 1

  (train_data, train_labels, valid_data, valid_labels,
   test_data, test_labels) = train_valid_test_split(
       data, labels,
       num_train_examples,
       num_valid_examples,
       num_test_examples,
       seed)

  np.random.seed(seed)
  random_mat = np.random.randn(
      train_data.shape[-1], projected_dim)
  train_data = np.dot(train_data, random_mat)
  valid_data = np.dot(valid_data, random_mat)
  if test_data is not None:
    test_data = np.dot(test_data, random_mat)

  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(train_data)
  train_data = scaler.transform(train_data)
  valid_data = scaler.transform(valid_data)
  if test_data is not None:
    test_data = scaler.transform(test_data)

  dataset = task_pb2.ScalarLabelDataset()
  for i in range(train_data.shape[0]):
    train_feature = dataset.train_features.add()
    train_feature.features.extend(list(train_data[i]))
    dataset.train_labels.append(train_labels[i])
  for i in range(valid_data.shape[0]):
    valid_feature = dataset.valid_features.add()
    valid_feature.features.extend(list(valid_data[i]))
    dataset.valid_labels.append(valid_labels[i])
  if test_data is not None:
    for i in range(test_data.shape[0]):
      test_feature = dataset.test_features.add()
      test_feature.features.extend(list(test_data[i]))
      dataset.test_labels.append(test_labels[i])
  return dataset


def load_projected_binary_dataset(saved_dataset):
  """Load the binary dataset saved in a ScalarLabelDataset proto."""
  num_train = len(saved_dataset.train_labels)
  assert len(saved_dataset.train_labels) == len(saved_dataset.train_features)
  num_valid = len(saved_dataset.valid_labels)
  assert len(saved_dataset.valid_labels) == len(saved_dataset.valid_features)
  num_test = len(saved_dataset.test_labels)
  assert len(saved_dataset.test_labels) == len(saved_dataset.test_features)
  if num_train == 0 or num_valid == 0:
    raise ValueError('Number of train/valid examples'
                     ' must be more than zero.')
  feature_size = len(saved_dataset.train_features[0].features)

  train_data = np.zeros((num_train, feature_size))
  train_labels = np.zeros(num_train)
  for i in range(num_train):
    train_labels[i] = saved_dataset.train_labels[i]
    for j in range(feature_size):
      train_data[i][j] = saved_dataset.train_features[i].features[j]

  valid_data = np.zeros((num_valid, feature_size))
  valid_labels = np.zeros(num_valid)
  for i in range(num_valid):
    valid_labels[i] = saved_dataset.valid_labels[i]
    for j in range(feature_size):
      valid_data[i][j] = saved_dataset.valid_features[i].features[j]

  if num_test > 0:
    test_data = np.zeros((num_test, feature_size))
    test_labels = np.zeros(num_test)
    for i in range(num_test):
      test_labels[i] = saved_dataset.test_labels[i]
      for j in range(feature_size):
        test_data[i][j] = saved_dataset.test_features[i].features[j]
  else:
    test_data = None
    test_labels = None

  return (train_data, train_labels, valid_data, valid_labels,
          test_data, test_labels)


def get_dataset(
    name, num_samples_per_class=None, class_ids=None, load_fn=tfds.load,
    data_dir=None):
  """Get the subset of the MNIST dataset containing the selected digits.

  Args:
    name: name of the dataset. Currently support mnist and cifar10.
    num_samples_per_class: number of samples for each class.
    class_ids: a list of class ids that will be included. Set to None to
      include all the classes.
    load_fn: function to load datasets, used for unit test.
    data_dir: the folder to load data from if it is already there, otherwise
      download data to this folder.

  Returns:
    train_data: a matrix of all the flattened training images.
    train_labels: a vector of all the training labels.
    test_data: a matrix of all the flattened test images.
    test_labels: a vector of all the test labels.
  """
  # Load datasets.
  dataset_dict = load_fn(
      name, data_dir=data_dir, batch_size=-1)
  # Whether the dataset is from tfds or given in unit test.
  if load_fn == tfds.load:
    train_set = tfds.as_numpy(dataset_dict[tfds.Split.TRAIN])
    test_set = tfds.as_numpy(dataset_dict[tfds.Split.TEST])
  else:
    train_set = dataset_dict[tfds.Split.TRAIN]
    test_set = dataset_dict[tfds.Split.TEST]
  train_data, train_labels = train_set['image'], train_set['label']
  test_data, test_labels = test_set['image'], test_set['label']

  train_data = train_data.astype(np.float)
  test_data = test_data.astype(np.float)
  assert train_data.shape[0] == train_labels.shape[0]
  assert test_data.shape[0] == test_labels.shape[0]

  if name == 'mnist':
    width = 28
    height = 28
    channel = 1
  elif name == 'cifar10':
    width = 32
    height = 32
    channel = 3
  else:
    raise ValueError('Dataset {} not supported!'.format(name))

  dim = width * height * channel
  train_data = train_data.reshape([-1, dim])
  test_data = test_data.reshape([-1, dim])

  if class_ids is not None:
    def select_classes(data, labels):
      data_list = [
          data[labels == class_id][:num_samples_per_class]
          for class_id in class_ids]
      labels_list = [
          labels[labels == class_id][:num_samples_per_class]
          for class_id in class_ids]
      selected_data = np.concatenate(data_list, axis=0)
      selected_labels = np.concatenate(labels_list, axis=0)
      return selected_data, selected_labels
    train_data, train_labels = select_classes(train_data, train_labels)
    test_data, test_labels = select_classes(test_data, test_labels)

  assert train_data.shape[0] == train_labels.shape[0]
  assert test_data.shape[0] == test_labels.shape[0]

  return (train_data, train_labels, test_data, test_labels)


def train_valid_test_split(
    data, labels,
    num_train_examples, num_valid_examples, num_test_examples,
    seed, use_stratify=True):
  """Split data into train, valid and test with given seed."""
  if num_test_examples > 0:
    if use_stratify:
      stratify = labels
    else:
      stratify = None
    train_data, test_data, train_labels, test_labels = (
        sklearn.model_selection.train_test_split(
            data, labels,
            train_size=(
                num_train_examples +
                num_valid_examples),
            test_size=num_test_examples,
            random_state=seed, stratify=stratify))
  else:
    train_data, train_labels = data, labels
    test_data = None
    test_labels = None
  if use_stratify:
    stratify = train_labels
  else:
    stratify = None
  train_data, valid_data, train_labels, valid_labels = (
      sklearn.model_selection.train_test_split(
          train_data, train_labels,
          train_size=num_train_examples,
          test_size=num_valid_examples,
          random_state=seed, stratify=stratify))
  return (
      train_data, train_labels,
      valid_data, valid_labels,
      test_data, test_labels)


def main(unused_argv):
  """Create and save the datasets."""
  del unused_argv
  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  tfds_cached_dict = {}
  data_dir = FLAGS.tfds_data_dir if FLAGS.tfds_data_dir else None
  name = FLAGS.dataset_name
  tfds_cached_dict[name] = tfds.load(name, batch_size=-1, data_dir=data_dir)
  dataset_dict = tfds_cached_dict[name]
  dataset_dict[tfds.Split.TRAIN] = tfds.as_numpy(
      dataset_dict[tfds.Split.TRAIN])
  dataset_dict[tfds.Split.TEST] = tfds.as_numpy(
      dataset_dict[tfds.Split.TEST])
  # To mock the API of tfds.load to cache the downloaded datasets.
  # Used as an argument to `get_dataset`.
  def load_fn(name, data_dir=None, batch_size=-1):
    # This function will always return the whole dataset.
    assert batch_size == -1
    del data_dir
    del batch_size
    return tfds_cached_dict[name]
  class_ids = sorted([int(x) for x in FLAGS.class_ids])
  num_classes = len(class_ids)
  for i in range(num_classes):
    for j in range(i+1, num_classes):
      print('Generating pos {} neg {}'.format(i, j))
      positive_class = class_ids[i]
      negative_class = class_ids[j]
      random_seeds = range(FLAGS.min_data_seed, FLAGS.max_data_seed)
      for seed in random_seeds:
        dataset = create_projected_binary_dataset(
            FLAGS.dataset_name, positive_class, negative_class,
            FLAGS.num_train_examples, FLAGS.num_valid_examples,
            FLAGS.num_test_examples, FLAGS.projected_dim, seed, load_fn)
        filename = 'binary_{}-pos_{}-neg_{}-dim_{}-seed_{}'.format(
            FLAGS.dataset_name, positive_class, negative_class,
            FLAGS.projected_dim, seed)
        serialized_dataset = dataset.SerializeToString()

        with open(os.path.join(FLAGS.data_dir, filename), 'wb') as f:
          f.write(serialized_dataset)

if __name__ == '__main__':
  app.run(main)
