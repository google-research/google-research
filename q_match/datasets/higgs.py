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

"""Higgs dataset."""

import abc

import tensorflow as tf
import tensorflow_datasets as tfds

from q_match.datasets import dataset_splits
from q_match.datasets.dataset import Dataset

NUMERIC_FEATURES = ['jet_1_b-tag', 'jet_1_eta', 'jet_1_phi', 'jet_1_pt',
                    'jet_2_b-tag', 'jet_2_eta', 'jet_2_phi', 'jet_2_pt',
                    'jet_3_b-tag', 'jet_3_eta', 'jet_3_phi', 'jet_3_pt',
                    'jet_4_b-tag', 'jet_4_eta', 'jet_4_phi', 'jet_4_pt',
                    'lepton_eta', 'lepton_pT', 'lepton_phi', 'm_bb', 'm_jj',
                    'm_jjj', 'm_jlv', 'm_lv', 'm_wbb', 'm_wwbb',
                    'missing_energy_magnitude', 'missing_energy_phi',]

TARGET = 'class_label'


class HiggsBase(Dataset):
  """Parent class for a Higgs dataset."""

  def __init__(self, dataset_path, batch_size=512, num_parallel_calls=60):
    self.pretext_ds = tfds.load(
        'higgs',
        data_dir=dataset_path,
        split=self.get_pretext_split())

    self.train_ds = tfds.load(
        'higgs',
        data_dir=dataset_path,
        split=self.get_train_split())

    self.validation_ds = tfds.load(
        'higgs',
        data_dir=dataset_path,
        split=self.get_validation_split())

    self.test_ds = tfds.load(
        'higgs',
        data_dir=dataset_path,
        split=self.get_test_split())

    # By default, there is no pretext val dataset, and it's not required.
    if self.get_pretext_validation_split() is not None:
      self.pretext_validation_ds = tfds.load(
          'higgs',
          data_dir=dataset_path,
          split=self.get_pretext_validation_split())
    else:
      self.pretext_validation_ds = None

    # By default, there is no imputation val dataset, and it's not required.
    if self.get_imputation_validation_split() is not None:
      self.imputation_validation_ds = tfds.load(
          'higgs',
          data_dir=dataset_path,
          split=self.get_imputation_validation_split())
    else:
      self.imputation_validation_ds = None

    # By default, there is no imputation train dataset, and it's not required.
    if self.get_imputation_train_split() is not None:
      self.imputation_train_ds = tfds.load(
          'higgs',
          data_dir=dataset_path,
          split=self.get_imputation_train_split())
    else:
      self.imputation_train_ds = None

    self.batch_size = batch_size
    self.num_parallel_calls = num_parallel_calls

    self.num_classes = dataset_splits.DATASET_TO_NUM_CLASSES['higgs']

    self.example_features = None

  @abc.abstractmethod
  def get_pretext_split(self):
    """Pretext split."""

  @abc.abstractmethod
  def get_train_split(self):
    """Train split."""

  @abc.abstractmethod
  def get_validation_split(self):
    """Validation split."""

  @abc.abstractmethod
  def get_test_split(self):
    """Test split."""

  def get_pretext_validation_split(self):
    """Pretext validation split."""
    return None

  def get_imputation_validation_split(self):
    """Imputation validation split."""
    return None

  def get_imputation_train_split(self):
    """Imputation train split."""
    return None

  def _format_data(self, original_data):
    return {'features': [original_data[feature]
                         for feature in NUMERIC_FEATURES],
            'target': tf.cast(original_data[TARGET], tf.int32)}

  def get_train_ds(self, cache=True, shuffle=True):
    train_ds = self.train_ds.map(self._format_data,
                                 num_parallel_calls=self.num_parallel_calls)
    if cache:
      train_ds = train_ds.cache()
    if shuffle:
      train_ds = train_ds.shuffle(10000)
    return train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_validation_ds(self):
    validation_ds = self.validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    return validation_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_test_epoch_iterator(self):
    test_ds = self.test_ds.map(self._format_data,
                               num_parallel_calls=self.num_parallel_calls)
    return test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_example_features(self):
    if self.example_features is None:
      train_iterator = iter(self.get_train_ds())
      example_features = train_iterator.get_next()['features']
      del train_iterator
      self.example_features = example_features
    return self.example_features

  def get_pretext_ds(self, cache=True, shuffle=True):
    pretext_ds = self.pretext_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if cache:
      pretext_ds = pretext_ds.cache()
    if shuffle:
      pretext_ds = pretext_ds.shuffle(10000)
    return pretext_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_pretext_validation_ds(self, cache=True, shuffle=True):
    if self.pretext_validation_ds is None:
      return None
    pretext_ds = self.pretext_validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if cache:
      pretext_ds = pretext_ds.cache()
    if shuffle:
      pretext_ds = pretext_ds.shuffle(10000)
    return pretext_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_imputation_validation_ds(self, cache=True):
    if self.imputation_validation_ds is None:
      return None
    ds = self.imputation_validation_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if cache:
      ds = ds.cache()
    return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_imputation_train_ds(self, cache=True, shuffle=True):
    if self.imputation_train_ds is None:
      return None
    train_ds = self.imputation_train_ds.map(
        self._format_data, num_parallel_calls=self.num_parallel_calls)
    if cache:
      train_ds = train_ds.cache()
    if shuffle:
      train_ds = train_ds.shuffle(10000)
    return train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

  def get_num_classes(self):
    return 2


class HiggsDataset(HiggsBase):
  """Full Higgs dataset."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs']['train']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs']['test']


class Higgs50kDataset(HiggsBase):
  """Higgs dataset with 5k training examples, 50k pretext examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs50k']['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs50k']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs50k']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs50k']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs50k']['test']


class Higgs10kDataset(HiggsBase):
  """Tabnet Higgs dataset with 10k train, 500k val, 10M pretext, 500k test."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs10k']['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs10k']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs10k']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs10k']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs10k']['test']


class Higgs100kDataset(HiggsBase):
  """Higgs dataset with 100k training examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs100k']['train']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs100k']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs100k']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs100k']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs100k']['test']

  def get_imputation_train_split(self):
    return dataset_splits.DATASETS['higgs100k']['imputation_train']

  def get_imputation_validation_split(self):
    return dataset_splits.DATASETS['higgs100k']['imputation_val']


class Higgs100k20pDataset(HiggsBase):
  """Higgs dataset with 100k pretext examples and 20% are training examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs100k20p']['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs100k20p']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs100k20p']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs100k20p']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs100k20p']['test']


class Higgs100k10pDataset(HiggsBase):
  """Higgs dataset with 100k pretext examples and 10% are training examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs100k10p']['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs100k10p']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs100k10p']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs100k10p']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs100k10p']['test']


class Higgs100k1pDataset(HiggsBase):
  """Higgs dataset with 100k pretext examples and 1% are training examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs100k1p']['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS['higgs100k1p']['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs100k1p']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs100k1p']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs100k1p']['test']


class Higgs1MDataset(HiggsBase):
  """Higgs dataset with 1M training examples."""

  def get_pretext_split(self):
    return dataset_splits.DATASETS['higgs1M']['train']

  def get_train_split(self):
    return dataset_splits.DATASETS['higgs1M']['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS['higgs1M']['val']

  def get_test_split(self):
    return dataset_splits.DATASETS['higgs1M']['test']


# Changing the pretext size
class HiggsVariablePretextDataset(HiggsBase):
  """Higgs dataset with X pretext size examples.

  Has 10k labeled, 10k labeled val, 10k imputation, 10k imputation val,
  10k pretext val, and 500k test.
  """

  def get_key_name(self,):
    return ''

  def get_pretext_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['val']

  def get_test_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['test']

  def get_imputation_train_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['imputation_train']

  def get_imputation_validation_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['imputation_val']


class HiggsPre10kDataset(HiggsVariablePretextDataset):
  """10k pretext size."""

  def get_key_name(self):
    return 'higgspre10k'


class HiggsPre40kDataset(HiggsVariablePretextDataset):
  """40k pretext size."""

  def get_key_name(self):
    return 'higgspre40k'


class HiggsPre160kDataset(HiggsVariablePretextDataset):
  """160k pretext size."""

  def get_key_name(self):
    return 'higgspre160k'


class HiggsPre640kDataset(HiggsVariablePretextDataset):
  """640k pretext size."""

  def get_key_name(self):
    return 'higgspre640k'


class HiggsPre2560kDataset(HiggsVariablePretextDataset):
  """2560k pretext size."""

  def get_key_name(self):
    return 'higgspre2560k'


# Changing the Label size
class HiggsVariableLabelDataset(HiggsBase):
  """Higgs dataset with X labeled size examples.

  Has X labeled, 10k labeled val, 10k pretext val, and 500k test.
  """

  def get_key_name(self,):
    return ''

  def get_pretext_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['pretext']

  def get_pretext_validation_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['pretext_validation']

  def get_train_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['train']

  def get_validation_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['val']

  def get_test_split(self):
    return dataset_splits.DATASETS[self.get_key_name()]['test']


class HiggsPre640kSup100Dataset(HiggsVariableLabelDataset):
  """640k pretext size, 100 labeled."""

  def get_key_name(self):
    return 'higgspre640ksup100'


class HiggsPre640kSup500Dataset(HiggsVariableLabelDataset):
  """640k pretext size, 500 labeled."""

  def get_key_name(self):
    return 'higgspre640ksup500'


class HiggsPre640kSup1kDataset(HiggsVariableLabelDataset):
  """640k pretext size, 1k labeled."""

  def get_key_name(self):
    return 'higgspre640ksup1k'
