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

"""Adult Classification dataset."""

import abc
import os

import tensorflow as tf
from q_match.datasets.dataset import Dataset


NUMERIC_FEATURES = ['age',
                    'fnlwgt',
                    'education-num',
                    'capital-gain',
                    'capital-loss',
                    'hours-per-week']

CATEGORICAL_FEATURES = ['workclass',
                        'education',
                        'marital-status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'native-country']

ONE_HOT_CATEGORICAL_FEATURES = [
    'workclass_Private', 'workclass_Self-emp-not-inc', 'workclass_Self-emp-inc',
    'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_State-gov',
    'workclass_Without-pay', 'workclass_Never-worked', 'workclass_?',
    'education_Bachelors', 'education_Some-college', 'education_11th',
    'education_HS-grad', 'education_Prof-school', 'education_Assoc-acdm',
    'education_Assoc-voc', 'education_9th', 'education_7th-8th',
    'education_12th', 'education_Masters', 'education_1st-4th',
    'education_10th', 'education_Doctorate', 'education_5th-6th',
    'education_Preschool', 'education_?', 'marital-status_Married-civ-spouse',
    'marital-status_Divorced', 'marital-status_Never-married',
    'marital-status_Separated', 'marital-status_Widowed',
    'marital-status_Married-spouse-absent', 'marital-status_Married-AF-spouse',
    'marital-status_?', 'occupation_Tech-support', 'occupation_Craft-repair',
    'occupation_Other-service', 'occupation_Sales',
    'occupation_Exec-managerial', 'occupation_Prof-specialty',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Adm-clerical', 'occupation_Farming-fishing',
    'occupation_Transport-moving', 'occupation_Priv-house-serv',
    'occupation_Protective-serv', 'occupation_Armed-Forces', 'occupation_?',
    'relationship_Wife', 'relationship_Own-child', 'relationship_Husband',
    'relationship_Not-in-family', 'relationship_Other-relative',
    'relationship_Unmarried', 'relationship_?', 'race_White',
    'race_Asian-Pac-Islander', 'race_Amer-Indian-Eskimo', 'race_Other',
    'race_Black', 'race_?', 'sex_Female', 'sex_Male', 'sex_?',
    'native-country_United-States', 'native-country_Cambodia',
    'native-country_England', 'native-country_Puerto-Rico',
    'native-country_Canada', 'native-country_Germany',
    'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_India',
    'native-country_Japan', 'native-country_Greece', 'native-country_South',
    'native-country_China', 'native-country_Cuba', 'native-country_Iran',
    'native-country_Honduras', 'native-country_Philippines',
    'native-country_Italy', 'native-country_Poland', 'native-country_Jamaica',
    'native-country_Vietnam', 'native-country_Mexico',
    'native-country_Portugal', 'native-country_Ireland',
    'native-country_France', 'native-country_Dominican-Republic',
    'native-country_Laos', 'native-country_Ecuador', 'native-country_Taiwan',
    'native-country_Haiti', 'native-country_Columbia', 'native-country_Hungary',
    'native-country_Guatemala', 'native-country_Nicaragua',
    'native-country_Scotland', 'native-country_Thailand',
    'native-country_Yugoslavia', 'native-country_El-Salvador',
    'native-country_Trinadad&Tobago', 'native-country_Peru',
    'native-country_Hong', 'native-country_Holand-Netherlands',
    'native-country_?'
]

TARGET = 'target'


def decode_fn(record_bytes):
  features = dict()
  for num_feature in NUMERIC_FEATURES+ONE_HOT_CATEGORICAL_FEATURES:
    features[num_feature] = tf.io.FixedLenFeature([], dtype=tf.float32)
  features[TARGET] = tf.io.FixedLenFeature([], dtype=tf.int64)
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      features
  )


class AdultDatasetBase(Dataset):
  """Base class for Adult datasets."""

  def __init__(self, dataset_path, batch_size=32, num_parallel_calls=60):
    self.dataset_path = dataset_path
    ds_name = self.get_ds_name()
    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

    self.pretext_ds = self.read_ds(ds_name, 'pretext.tfrecord')
    self.pretext_val_ds = self.read_ds(ds_name, 'pretext_val.tfrecord')
    self.train_ds = self.read_ds(ds_name, 'train.tfrecord')
    self.validation_ds = self.read_ds(ds_name, 'val.tfrecord')
    self.test_ds = self.read_ds(ds_name, 'test.tfrecord')

  @abc.abstractmethod
  def get_ds_name(self):
    """The dataset name."""

  def read_ds(self, ds_name, filename):
    return tf.data.TFRecordDataset(
        os.path.join(self.dataset_path, ds_name, filename)).map(decode_fn)

  def _format_data(self, original_data):
    return {
        'features': [
            original_data[feature]
            for feature in NUMERIC_FEATURES + ONE_HOT_CATEGORICAL_FEATURES
        ],
        'target': original_data[TARGET]
    }

  def _chain_ops(self, ds, cache=False, shuffle=False):
    ds = ds.map(self._format_data, num_parallel_calls=self.num_parallel_calls)

    if shuffle:
      ds = ds.shuffle(10000)

    if cache:
      ds = ds.cache()

    return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_pretext_ds(self):
    return self._chain_ops(self.pretext_ds, shuffle=True, cache=True)

  def get_pretext_validation_ds(self):
    return self._chain_ops(self.pretext_val_ds, cache=True)

  def get_train_ds(self):
    return self._chain_ops(self.train_ds, shuffle=True, cache=True)

  def get_validation_ds(self):
    return self._chain_ops(self.validation_ds, cache=True)

  def get_test_epoch_iterator(self):
    return self._chain_ops(self.test_ds)

  def get_example_features(self):
    train_iterator = iter(self.get_train_ds())
    example_features = train_iterator.get_next()['features']
    del train_iterator
    return example_features

  def get_num_classes(self):
    return 2


class Adult1PDataset(AdultDatasetBase):

  def get_ds_name(self):
    return 'adult1p'
