# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Data loader for Camelyon17 dataset."""

from gift.data import base_dataset
from gift.data.builders import camelyon17_builder


class Camelyon17(base_dataset.MutliEnvironmentImageDataset):
  """Data loader for Camelyon17."""

  # Environments are hospital Ids.
  _ALL_ENVIRONMENTS = ['0', '1', '2', '3', '4']

  @property
  def name(self):
    return 'camelyon17'

  def get_builder(self, name):
    return camelyon17_builder.Camelyon17(data_dir='PATH_TO_DATA')

  def set_static_dataset_configs(self):
    self._channels = 3
    train_splits = {}
    for env in self.train_environments:
      train_splits[env] = f'{env}'

    test_splits = {}
    valid_splits = {}
    for env in self.eval_environments:
      valid_splits[env] = f'{env}'
      test_splits[env] = f'{env}'

    self._splits_dict = {
        'train': train_splits,
        'test': test_splits,
        'validation': valid_splits
    }
    self._crop_padding = 32
    self._mean_rgb = [0.485, 0.456, 0.406]
    self._stddev_rgb = [0.229, 0.224, 0.225]
    self.resolution = self.resolution or 224
    self.resize_mode = 'resize'
    self.data_augmentations = self.data_augmentations or ['center_crop']
    self.teacher_data_augmentations = self.teacher_data_augmentations or [
        'center_crop'
    ]
    self.eval_augmentations = ['center_crop']
    self.if_cache = True

  def get_tfds_env_name(self, name):
    return name

  def get_tfds_ds_and_info(self, name, data_range):
    del name
    ds = self.builder.as_dataset(split=data_range)

    return ds, self.builder.info

  def get_num_classes(self):
    return self.builder.info.features['label'].num_classes
