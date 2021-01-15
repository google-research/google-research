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

"""Generalized Linear Modeling module for fitting calibration functions."""

import enum
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from caltrain.glm_modeling.dataset import Dataset
from caltrain.glm_modeling.transformpair import TransformPair
from caltrain.utils import Enum


class Transforms(Enum):
  """Class to define available GLM transformation functions."""
  log = TransformPair(f=np.log, finv=np.exp, str_formatter=r'$\log({var})$')
  logflip = TransformPair(
      f=lambda x: np.log(1 - np.array(x)),
      finv=lambda x: 1 - np.exp(x),
      str_formatter=r'$\log(1-{var})$')
  logit = TransformPair(
      f=lambda x: np.log(np.array(x) / (1 - np.array(x))),
      finv=lambda x: 1. / (1 + np.exp(-np.array(x))),
      str_formatter=r'$\log \frac{var}{1-{var}}$')


class Folds(Enum):
  """Class define available dataset folds."""

  val = enum.auto()
  test = enum.auto()


def get_datasets(data_dir=None):
  """Function to return Dataset container from data_dir."""

  class Datasets(Enum):
    """Class to contain empirical datasets for analysis."""

    resnet110_c10 = {
        fold:
        Dataset.get_dataset('resnet110_c10', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet110_SD_c10 = {  # pylint: disable=invalid-name
        fold:
        Dataset.get_dataset('resnet110_SD_c10', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet_wide32_c10 = {
        fold:
        Dataset.get_dataset('resnet_wide32_c10', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    densenet40_c10 = {
        fold:
        Dataset.get_dataset('densenet40_c10', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    lenet5_c10 = {
        fold: Dataset.get_dataset('lenet5_c10', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet110_c100 = {
        fold:
        Dataset.get_dataset('resnet110_c100', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet110_SD_c100 = {  # pylint: disable=invalid-name
        fold:
        Dataset.get_dataset('resnet110_SD_c100', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet_wide32_c100 = {
        fold:
        Dataset.get_dataset('resnet_wide32_c100', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    densenet40_c100 = {
        fold:
        Dataset.get_dataset('densenet40_c100', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    lenet5_c100 = {
        fold: Dataset.get_dataset('lenet5_c100', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet152_imgnet = {
        fold:
        Dataset.get_dataset('resnet152_imgnet', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    densenet161_imgnet = {
        fold:
        Dataset.get_dataset('densenet161_imgnet', fold=fold, data_dir=data_dir)
        for fold in Folds
    }
    resnet50_birds = {
        fold:
        Dataset.get_dataset('resnet50_birds', fold=fold, data_dir=data_dir)
        for fold in Folds
    }

  return Datasets


def get_guo_et_al_data(data_dir):
  """Function to get data from Guo et al figure."""

  datasets = get_datasets(data_dir=data_dir)
  guo_et_al_data = {'ECE': {}}
  guo_et_al_data['ECE'][datasets.resnet50_birds] = .0919
  guo_et_al_data['ECE'][datasets.resnet110_c10] = .046
  guo_et_al_data['ECE'][datasets.resnet110_SD_c10] = .0412
  guo_et_al_data['ECE'][datasets.resnet_wide32_c10] = .0452
  guo_et_al_data['ECE'][datasets.densenet40_c10] = .0328
  guo_et_al_data['ECE'][datasets.lenet5_c10] = .0302
  guo_et_al_data['ECE'][datasets.resnet110_c100] = .1653
  guo_et_al_data['ECE'][datasets.resnet110_SD_c100] = .1267
  guo_et_al_data['ECE'][datasets.resnet_wide32_c100] = .1500
  guo_et_al_data['ECE'][datasets.densenet40_c100] = .1037
  guo_et_al_data['ECE'][datasets.lenet5_c100] = .0485
  guo_et_al_data['ECE'][datasets.densenet161_imgnet] = .0628
  guo_et_al_data['ECE'][datasets.resnet152_imgnet] = .0548

  guo_et_al_data['ECE_ts'] = {}
  guo_et_al_data['ECE_ts'][datasets.resnet50_birds] = .0185
  guo_et_al_data['ECE_ts'][datasets.resnet110_c10] = .0083
  guo_et_al_data['ECE_ts'][datasets.resnet110_SD_c10] = .0060
  guo_et_al_data['ECE_ts'][datasets.resnet_wide32_c10] = .0054
  guo_et_al_data['ECE_ts'][datasets.densenet40_c10] = .0033
  guo_et_al_data['ECE_ts'][datasets.lenet5_c10] = .0093
  guo_et_al_data['ECE_ts'][datasets.resnet110_c100] = .0126
  guo_et_al_data['ECE_ts'][datasets.resnet110_SD_c100] = .0096
  guo_et_al_data['ECE_ts'][datasets.resnet_wide32_c100] = .0232
  guo_et_al_data['ECE_ts'][datasets.densenet40_c100] = .0118
  guo_et_al_data['ECE_ts'][datasets.lenet5_c100] = .0202
  guo_et_al_data['ECE_ts'][datasets.densenet161_imgnet] = .0199
  guo_et_al_data['ECE_ts'][datasets.resnet152_imgnet] = .0186

  guo_et_al_data['ECE_ir'] = {}
  guo_et_al_data['ECE_ir'][datasets.resnet50_birds] = .0522
  guo_et_al_data['ECE_ir'][datasets.resnet110_c10] = .0081
  guo_et_al_data['ECE_ir'][datasets.resnet110_SD_c10] = .0111
  guo_et_al_data['ECE_ir'][datasets.resnet_wide32_c10] = .0108
  guo_et_al_data['ECE_ir'][datasets.densenet40_c10] = .0061
  guo_et_al_data['ECE_ir'][datasets.lenet5_c10] = .0185
  guo_et_al_data['ECE_ir'][datasets.resnet110_c100] = .0499
  guo_et_al_data['ECE_ir'][datasets.resnet110_SD_c100] = .0416
  guo_et_al_data['ECE_ir'][datasets.resnet_wide32_c100] = .0585
  guo_et_al_data['ECE_ir'][datasets.densenet40_c100] = .0451
  guo_et_al_data['ECE_ir'][datasets.lenet5_c100] = .0235
  guo_et_al_data['ECE_ir'][datasets.densenet161_imgnet] = .0518
  guo_et_al_data['ECE_ir'][datasets.resnet152_imgnet] = .0477

  guo_et_al_data['ECE_hb'] = {}
  guo_et_al_data['ECE_hb'][datasets.resnet50_birds] = .0434
  guo_et_al_data['ECE_hb'][datasets.resnet110_c10] = .0058
  guo_et_al_data['ECE_hb'][datasets.resnet110_SD_c10] = .0067
  guo_et_al_data['ECE_hb'][datasets.resnet_wide32_c10] = .0072
  guo_et_al_data['ECE_hb'][datasets.densenet40_c10] = .0044
  guo_et_al_data['ECE_hb'][datasets.lenet5_c10] = .0156
  guo_et_al_data['ECE_hb'][datasets.resnet110_c100] = .0266
  guo_et_al_data['ECE_hb'][datasets.resnet110_SD_c100] = .0246
  guo_et_al_data['ECE_hb'][datasets.resnet_wide32_c100] = .0301
  guo_et_al_data['ECE_hb'][datasets.densenet40_c100] = .0268
  guo_et_al_data['ECE_hb'][datasets.lenet5_c100] = .0648
  guo_et_al_data['ECE_hb'][datasets.densenet161_imgnet] = .0452
  guo_et_al_data['ECE_hb'][datasets.resnet152_imgnet] = .0436

  return guo_et_al_data


class CachedCloudData:
  """Container to lazily access cached data."""

  def __init__(self, uri):
    self.uri = uri
    self._data = None

  def __getitem__(self, key):
    if self._data is None:
      with tf.io.gfile.GFile(self.uri, 'rb') as f:
        self._data = pickle.load(f)

    return self._data[key]


def get_glm_fit_data(data_dir):
  return CachedCloudData(os.path.join(data_dir, 'glm_fit_data.p'))


def get_beta_fit_data(data_dir):
  return CachedCloudData(os.path.join(data_dir, 'beta_fit_data.p'))


def get_eece_sece_data(data_dir):
  return CachedCloudData(os.path.join(data_dir, 'eece_sece_data.p'))
