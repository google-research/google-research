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

"""Data loader for FMoW dataset."""

from gift.data import base_dataset
from gift.data.builders import fmow_builder


class Fmow(base_dataset.MutliEnvironmentImageDataset):
  """Data loader for FMoW."""

  _ALL_ENVIRONMENTS = ['train', 'val_id', 'val_ood', 'test_id', 'test_ood']

  @property
  def name(self):
    return 'fmow'

  def get_builder(self, name):
    return fmow_builder.Fmow(data_dir='PATH_TO_DATA')

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
