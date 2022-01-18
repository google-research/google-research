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

"""Mapping of all defined dataset in the project.

 dataset name --> dataset class.
"""

from gift.data import camelyon17
from gift.data import cifar10
from gift.data import fmow

ALL_DATASETS = {
    'cifar10': cifar10.Cifar10,
    'translated_cifar10': cifar10.TranslatedCifar10,
    'scaled_translated_cifar10': cifar10.ScaledTranslatedCifar10,
}

ALL_MULTI_ENV_DATASETS = {
    'multi_cifar10': cifar10.MultiCifar10,
    'multi_cifar10_rotated': cifar10.MultiCifar10Rotated,
    'multi_cifar10_scaled': cifar10.MultiCifar10Scaled,
    'camelyon17': camelyon17.Camelyon17,
    'fmow': fmow.Fmow
}

ALL = {}
ALL.update(ALL_DATASETS)
ALL.update(ALL_MULTI_ENV_DATASETS)


def get_dataset(dataset_name):
  """Maps dataset name to a dataset_builder.

  Args:
    dataset_name: string; Name of the dataset.

  Returns:
    A dataset builder.
  """

  if dataset_name not in ALL.keys():
    raise ValueError('Unrecognized dataset: {}'.format(dataset_name))
  return ALL[dataset_name]
