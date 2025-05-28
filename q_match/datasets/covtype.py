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

"""CoverType dataset.

Available here: https://archive.ics.uci.edu/ml/datasets/covertype
"""
import os

import tensorflow as tf

from q_match.datasets.dataset import Dataset

NUMERIC_FEATURES = ['Elevation', 'Aspect', 'Slope',
                    'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']

# The categorical features are all binary
CATEGORICAL_FEATURES = ([f'Wilderness_Area_{x}' for x in range(4)]
                        +[f'Soil_Type_{x}' for x in range(40)])

TARGET = 'Cover_Type'


def decode_fn(record_bytes):
  features = dict()
  for num_feature in NUMERIC_FEATURES+CATEGORICAL_FEATURES:
    features[num_feature] = tf.io.FixedLenFeature([], dtype=tf.float32)
  features[TARGET] = tf.io.FixedLenFeature([], dtype=tf.int64)
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      features
  )


class CoverTypeDataset(Dataset):
  """Forest Cover Dataset.

  By defualt, uses the train ds as the pretext ds.
  """

  def __init__(self, dataset_path, batch_size=32, num_parallel_calls=60):
    self.train_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype',
                     'trainval.tfrecord')).map(decode_fn)

    self.validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype',
                     'val.tfrecord')).map(decode_fn)

    self.test_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype',
                     'test.tfrecord')).map(decode_fn)

    self.pretext_validation_ds = None

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

  def _format_data(self, original_data):
    return {
        'features': [
            original_data[feature]
            for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ],
        'target': original_data[TARGET] - 1
    }  # index at 1

  def get_pretext_ds(self, cache=True, shuffle=True):
    train_ds = self.train_ds.map(self._format_data,
                                 num_parallel_calls=self.num_parallel_calls)
    if shuffle:
      train_ds = train_ds.shuffle(10000)
    if cache:
      train_ds = train_ds.cache()
    return train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_train_ds(self, cache=True, shuffle=True):
    train_ds = self.train_ds.map(self._format_data,
                                 num_parallel_calls=self.num_parallel_calls)
    if shuffle:
      train_ds = train_ds.shuffle(10000)
    if cache:
      train_ds = train_ds.cache()
    return train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_validation_ds(self,):
    validation_ds = self.validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return validation_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_test_epoch_iterator(self):
    test_ds = self.test_ds.map(self._format_data,
                               num_parallel_calls=self.num_parallel_calls)
    return test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_example_features(self):
    train_iterator = iter(self.get_train_ds())
    example_features = train_iterator.get_next()['features']
    del train_iterator
    return example_features

  def get_pretext_validation_ds(self):
    if self.pretext_validation_ds is None:
      return None
    validation_ds = self.pretext_validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return validation_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_num_classes(self):
    return 7


class CoverTypeImixDataset(CoverTypeDataset):
  """Forest Cover Dataset."""

  def __init__(self, dataset_path, batch_size=32, num_parallel_calls=60):
    super(CoverTypeImixDataset, self).__init__(dataset_path)
    folder = 'covtype15k'

    self.train_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, folder,
                     'trainval.tfrecord')).map(decode_fn)

    self.validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, folder,
                     'val.tfrecord')).map(decode_fn)

    self.pretext_validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, folder,
                     'val.tfrecord')).map(decode_fn)

    self.trainval_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, folder,
                     'trainval.tfrecord')).map(decode_fn)

    self.test_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, folder,
                     'test.tfrecord')).map(decode_fn)

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

  def get_train_val_ds(self):
    trainval_ds = self.trainval_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return trainval_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


class CoverTypeNew1PDataset(CoverTypeDataset):
  """Forest Cover Dataset with 1% Labels."""

  def __init__(self, dataset_path, batch_size=32, num_parallel_calls=60):
    super(CoverTypeNew1PDataset, self).__init__(dataset_path)

    self.pretext_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_new_1p',
                     'pretext.tfrecord')).map(decode_fn)

    self.train_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_new_1p',
                     'train.tfrecord')).map(decode_fn)

    self.validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_new_1p',
                     'val.tfrecord')).map(decode_fn)

    self.test_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_new_1p',
                     'test.tfrecord')).map(decode_fn)

    self.pretext_validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_new_1p',
                     'val.tfrecord')).map(decode_fn)

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

  def get_pretext_ds(self, cache=True, shuffle=True):
    pretext_ds = self.pretext_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if shuffle:
      pretext_ds = pretext_ds.shuffle(10000)
    if cache:
      pretext_ds = pretext_ds.cache()
    return pretext_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_train_val_ds(self):
    trainval_ds = self.train_ds.concatenate(self.validation_ds)
    trainval_ds = trainval_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return trainval_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


class CoverType10PDataset(CoverTypeDataset):
  """Forest Cover Dataset with 10% Labels as defined in constrastiv mixup."""

  def __init__(self, dataset_path, batch_size=32, num_parallel_calls=60):
    super(CoverType10PDataset, self).__init__(dataset_path)

    self.pretext_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_10p',
                     'pretext.tfrecord')).map(decode_fn)

    self.train_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_10p',
                     'train.tfrecord')).map(decode_fn)

    self.validation_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_10p',
                     'val.tfrecord')).map(decode_fn)

    self.test_ds = tf.data.TFRecordDataset(
        os.path.join(dataset_path, 'covtype_10p',
                     'test.tfrecord')).map(decode_fn)

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

  def get_pretext_ds(self, cache=True, shuffle=True):
    pretext_ds = self.pretext_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if shuffle:
      pretext_ds = pretext_ds.shuffle(10000)
    if cache:
      pretext_ds = pretext_ds.cache()
    return pretext_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
