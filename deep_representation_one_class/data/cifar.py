# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""CIFAR data."""

import os

import numpy as np
from six.moves import cPickle
import tensorflow as tf

from deep_representation_one_class.data.augment import retrieve_augment
from deep_representation_one_class.data.data_util import BasicImageProcess


def rotate_array_deterministic(data):
  """Rotate numpy array into 4 rotation angles.

  Args:
    data: data numpy array, B x H x W x C

  Returns:
    A concatenation of the original and 3 rotations.
  """
  return np.concatenate(
      [data] + [np.rot90(data, k=k, axes=(1, 2)) for k in range(1, 4)], axis=0)


def hflip_array_deterministic(data):
  """Flips numpy array into 2 dierction.

  Args:
    data: data numpy array, B x H x W x C

  Returns:
    A concatenation of the original data and its horizontally flipped one.
  """
  return np.concatenate([data] + [np.flip(data, axis=2)], axis=0)


def vflip_array_deterministic(data):
  """Flips numpy array into 2 dierction.

  Args:
    data: data numpy array, B x H x W x C

  Returns:
    A concatenation of the original data and its vertically flipped one.
  """
  return np.concatenate([data] + [np.flip(data, axis=1)], axis=0)


def geotrans_array_deterministic(data, num_aug):
  """Transforms numpy array with geometric transformations.

  Original
   + Horizontal flip
  rotation 90
   + Horizontal flip
  rotation 180
   + Horizontal flip
  rotation 270
   + Horizontal flip

  Args:
    data: data numpy array, B x H x W x C
    num_aug: number of distribution augmentations.

  Returns:
    A list of distortions.
  """
  rot_list = [data] + [np.rot90(data, k=k, axes=(1, 2)) for k in range(1, 4)]
  rot_flip_list = [[rdata] + [np.flip(rdata, axis=2)] for rdata in rot_list]

  return_list = []
  for sublist in rot_flip_list:
    for item in sublist:
      return_list.append(item)

  return np.concatenate(return_list[:num_aug], axis=0)


class CIFAR(object):
  """CIFAR data loader."""

  def __init__(self, root, dataset='cifar10', input_shape=(32, 32, 3)):
    self.root = root
    label_mode = 'coarse' if dataset in ['cifar20', 'cifar20ood'] else 'fine'
    if dataset in ['cifar10', 'cifar10ood']:
      dataset_raw = 'cifar10'
    else:
      dataset_raw = 'cifar100'
    (x_train, y_train), (x_test, y_test) = self.load_data(
        dataset=dataset_raw, label_mode=label_mode)
    self.trainval_data = [
        x_train, y_train,
        np.expand_dims(np.arange(len(y_train)), axis=1)
    ]
    self.test_data = [
        x_test, y_test,
        np.expand_dims(np.arange(len(y_test)), axis=1)
    ]
    self.dataset = dataset
    self.input_shape = input_shape

  def load_from_path(self, paths, label_key='labels'):
    """Loads CIFAR data from the path."""
    data = []
    labels = []
    for path in paths:
      with tf.io.gfile.GFile(path, 'rb') as f:
        d = {
            k.decode('utf8'): v
            for k, v in cPickle.load(f, encoding='bytes').items()
        }
      data.append(d['data'])
      labels.append(d[label_key])
    data = np.concatenate(data, axis=0)
    data = data.reshape((data.shape[0], 3, 32, 32))
    labels = np.concatenate(labels, axis=0)
    labels = np.reshape(labels, (len(labels), 1))

    if tf.keras.backend.image_data_format() == 'channels_last':
      data = data.transpose(0, 2, 3, 1)

    return data, labels

  def load_data(self, dataset='cifar10', label_mode='fine'):
    """Loads CIFAR dataset from root if specified or using tf.keras.dataset."""
    if dataset == 'cifar10':
      if self.root:
        x_train, y_train = self.load_from_path(
            [os.path.join(self.root, f'data_batch_{i}') for i in range(1, 6)])
        x_test, y_test = self.load_from_path(
            [os.path.join(self.root, 'test_batch')])
        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)
        return (x_train, y_train), (x_test, y_test)
      else:
        return tf.keras.datasets.cifar10.load_data()
    elif dataset in ['cifar20', 'cifar100']:
      if self.root:
        x_train, y_train = self.load_from_path(
            [os.path.join(self.root, 'train')], label_key=label_mode)
        x_test, y_test = self.load_from_path([os.path.join(self.root, 'test')])
        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)
        return (x_train, y_train), (x_test, y_test)
      else:
        return tf.keras.datasets.cifar100.load_data(label_mode=label_mode)

  def process_for_ood(self, category=0):
    """Process data for OOD experiment."""
    assert category in np.unique(
        self.trainval_data[1]), 'category is not in a label set'
    train_neg_idx = np.where(self.trainval_data[1] == category)[0]
    train_pos_idx = np.where(self.trainval_data[1] != category)[0]
    test_neg_idx = np.where(self.test_data[1] == category)[0]
    test_pos_idx = np.where(self.test_data[1] != category)[0]
    self.trainval_data_pos = [
        self.trainval_data[0][train_pos_idx],
        self.trainval_data[1][train_pos_idx],
        self.trainval_data[2][train_pos_idx]
    ]
    self.trainval_data[0] = self.trainval_data[0][train_neg_idx]
    self.trainval_data[1] = self.trainval_data[1][train_neg_idx]
    self.trainval_data[2] = self.trainval_data[2][train_neg_idx]
    self.test_data[1][test_neg_idx] = np.zeros_like(
        self.test_data[1][test_neg_idx])
    self.test_data[1][test_pos_idx] = np.ones_like(
        self.test_data[1][test_pos_idx])

  def get_prefix(self, aug_list):
    """Gets naming prefix."""
    shape_str = 'x'.join(('%d' % s for s in self.input_shape))
    self.fname = f'{self.dataset}_c{self.category}_s{shape_str}'
    self.fname = os.path.join(self.fname, '_'.join(s for s in aug_list))

  def load_dataset(self,
                   is_validation=False,
                   aug_list=None,
                   aug_list_for_test=None,
                   batch_size=64,
                   num_batch_per_epoch=None,
                   distaug_type=''):
    """Load dataset."""

    # Constructs dataset for validation or test.
    if is_validation:
      np.random.seed(1)
      idx = np.random.permutation(len(self.trainval_data[0]))
      idx_train = idx[:int(len(idx) * 0.9)]
      idx_val_neg = idx[int(len(idx) * 0.9):]
      idx_val_pos = np.random.permutation(len(
          self.trainval_data_pos[0]))[:int(len(idx_val_pos))]
      train_data = [
          self.trainval_data[0][idx_train], self.trainval_data[1][idx_train],
          np.arange(len(idx_train))[:, None]
      ]
      test_data = [
          np.concatenate((self.trainval_data[0][idx_val_neg],
                          self.trainval_data_pos[0][idx_val_pos]),
                         axis=0),
          np.concatenate((np.zeros_like(self.trainval_data[1][idx_val_neg]),
                          np.ones_like(self.trainval_data_pos[1][idx_val_pos])),
                         axis=0),
          np.arange(len(idx_val_pos) + len(idx_val_neg))[:, None]
      ]
    else:
      train_data = self.trainval_data
      test_data = self.test_data

    # Sets aside unaugmented training data to construct a classifier.
    # We limit the number by 20000 for efficiency of learning classifier.
    indices = np.random.permutation(len(
        train_data[0]))[:20000] if len(train_data[0]) > 20000 else np.arange(
            len(train_data[0]))
    train_data_for_cls = [data[indices] for data in train_data]

    if distaug_type:
      # Applies offline distribution augmentation on train data.
      # Type of augmentation: Rotation (0, 90, 180, 270), horizontal or
      # vertical flip, combination of rotation and horizontal flips.
      assert distaug_type in ['rot', 'hflip', 'vflip'] + [
          1, 2, 3, 4, 5, 6, 7, 8
      ], f'{distaug_type} is not supported distribution augmentation type.'
      if distaug_type == 'rot':
        aug_data = rotate_array_deterministic(train_data[0])
        lab_data = np.concatenate([train_data[1] for _ in range(4)], axis=0)
      elif distaug_type == 'hflip':
        aug_data = hflip_array_deterministic(train_data[0])
        lab_data = np.concatenate([train_data[1] for _ in range(2)], axis=0)
      elif distaug_type == 'vflip':
        aug_data = vflip_array_deterministic(train_data[0])
        lab_data = np.concatenate([train_data[1] for _ in range(2)], axis=0)
      elif distaug_type in [1, 2, 3, 4, 5, 6, 7, 8]:
        aug_data = geotrans_array_deterministic(train_data[0], distaug_type)
        lab_data = np.concatenate([train_data[1] for _ in range(distaug_type)],
                                  axis=0)
      train_data = [aug_data, lab_data, np.arange(len(aug_data))[:, None]]
    train_set = BasicImageProcess(
        data=tuple(train_data), input_shape=self.input_shape)
    train_set_for_cls = BasicImageProcess(
        data=tuple(train_data_for_cls), input_shape=self.input_shape)
    test_set = BasicImageProcess(
        data=tuple(test_data), input_shape=self.input_shape)

    aug_args = {'size': self.input_shape[0]}
    augs = retrieve_augment(aug_list, **aug_args)
    train_aug, test_aug = augs[:-1], augs[-1]
    if aug_list_for_test is not None:
      test_aug = retrieve_augment(aug_list_for_test, **aug_args)
      test_aug = test_aug[:-1]

    train_loader = train_set.input_fn(
        is_training=True,
        batch_size=batch_size,
        aug_list=train_aug,
        num_batch_per_epoch=num_batch_per_epoch,
        training_dataset_cache=True)
    train_loader_for_cls = train_set_for_cls.input_fn(
        is_training=False,
        batch_size=batch_size if aug_list_for_test is None else batch_size //
        len(aug_list_for_test),
        aug_list=test_aug,
        force_augment=aug_list_for_test is not None,
        training_dataset_cache=True)
    test_loader = test_set.input_fn(
        is_training=False,
        batch_size=batch_size if aug_list_for_test is None else batch_size //
        len(aug_list_for_test),
        aug_list=test_aug,
        force_augment=aug_list_for_test is not None,
        training_dataset_cache=True)

    self.get_prefix(aug_list)
    return [train_loader, train_loader_for_cls, test_loader]


class CIFAROOD(CIFAR):
  """CIFAR for OOD."""

  def __init__(self,
               root,
               dataset='cifar10',
               input_shape=(32, 32, 3),
               category=0):
    super(CIFAROOD, self).__init__(
        root=root, dataset=dataset, input_shape=input_shape)
    if isinstance(category, str):
      try:
        category = int(float(category))
      except ValueError:
        msg = f'category {category} must be integer convertible.'
        raise ValueError(msg)
    self.category = category
    self.process_for_ood(category=self.category)
