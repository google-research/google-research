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

"""Config used for approxNN project."""

import datetime
import os
import sys

from absl import logging
import numpy as np


RUNNING_INTERNALLY = False

# Create folders to save results, models, and plots.
PLOTS_SUBPATH = '_plots'
MODELS_SUBPATH = '_models'

# Define constants.
ALLOWABLE_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped']
ALLOWABLE_EXPLANATION_METHODS = [
    'ig',
    'grad',
    'gradcam',
    'guided_ig',
    'smooth_ig',
    'smooth_grad',
    'smooth_gradcam',
]

# Specify the column names and types to be read from metrics.csv for base models
CAT_HPARAMS = [
    'config.b_init',
    'config.w_init',
    'config.optimizer',
    'config.activation',
]
NUM_HPARAMS = [
    'config.l2reg',
    'config.dropout',
    'config.init_std',
    'config.learning_rate',
    'config.train_fraction'
]
ALL_HPARAMS = [*CAT_HPARAMS, *NUM_HPARAMS]
ALL_HPARAM_RANGES = {
    'config.b_init': ['zeros'],
    'config.w_init': [
        'he_normal',
        'orthogonal',
        'glorot_normal',
        'RandomNormal',
        'TruncatedNormal',
    ],
    'config.optimizer': ['rmsprop', 'sgd', 'adam'],
    'config.activation': ['relu', 'tanh'],
    'config.l2reg': [1e-8, 1e-6, 1e-4, 1e-2],
    'config.dropout': [0, 0.2, 0.45, 0.7],
    'config.init_std': [1e-3, 1e-2, 1e-1, 0.5],
    'config.learning_rate': [5e-4, 5e-3, 5e-2],
    'config.train_fraction': [0.1, 0.25, 0.5, 1.0],
}
assert not np.intersect1d(NUM_HPARAMS, CAT_HPARAMS)  # if len(.) == 0
assert set(np.union1d(CAT_HPARAMS, NUM_HPARAMS)) == set(ALL_HPARAMS)
ALL_METRICS = [
    'test_accuracy',
    'test_loss',
    'train_accuracy',
    'train_loss'
]


class Config(object):
  """A class to allow for overwritable and assertable config attributes."""

  def __init__(self):

    # Use __dict__ so as to not invoke __setattr__ below as the variable
    # is not yet set.
    self.__dict__['initialized'] = False

    # pylint: disable=invalid-name

    # Default values may be updated through absl.FLAGS.
    self.RANDOM_SEED = 42
    self.DATASET = 'cifar10'
    self.EXPLANATION_TYPE = 'ig'
    self.RUN_ON_TEST_DATA = False
    self.NUM_BASE_MODELS = 30000
    self.NUM_SAMPLES_PER_BASE_MODEL = 8
    self.NUM_SAMPLES_TO_PLOT_TE_FOR = 8
    self.KEEP_MODELS_ABOVE_TEST_ACCURACY = 0.55
    self.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS = True

    # Not currently updated through absl.FLAGS.
    self.BASE_MODEL_BATCH_SIZE = 32
    self.META_MODEL_BATCH_SIZE = 32
    self.TRAIN_FRACTIONS = [0.01, 0.03, 0.1]
    self.META_MODEL_EPOCHS = 50
    self.COVARIATES_SETTINGS = [
        # {'chkpt': 0},
        # {'chkpt': 1},
        # {'chkpt': 2},
        # {'chkpt': 20},
        # {'chkpt': 40},
        # {'chkpt': 60},
        {'chkpt': 86},
    ]

    # The attributes below will only be set after calling initialize_config().
    self.SETUP_NAME = None
    self.EXPERIMENT_DIR = None
    self.DATA_DIR = None
    self.EXP_DIR_PATH = None
    self.DATA_DIR_PATH = None
    self.PLOTS_DIR_PATH = None
    self.MODELS_DIR_PATH = None

    # pylint: enable=invalid-name

    self.paths_set = False
    self.initialized = True
    logging.info('Do not forget to call set_config_paths() to set all params.')

  def __setattr__(self, name, value):
    """Central method for asserting new attribute values before updating.

    Args:
      name: name of the class attribute to be set (updated).
      value: value of the class attribute to be set (updated).
    """

    # This function is meant only for updating attributes, but not for __init__.
    if not self.initialized:
      super(Config, self).__setattr__(name, value)
      return

    if name not in self.__dict__:
      raise ValueError('Cannot set unrecognized attribute with name %s.' % name)

    if name == 'DATASET' and value not in ALLOWABLE_DATASETS:
      raise ValueError('DATASET not recognized.')

    if (
        name == 'EXPLANATION_TYPE' and
        value not in ALLOWABLE_EXPLANATION_METHODS
    ):
      raise ValueError('EXPLANATION_TYPE not recognized.')

    if name == 'RUN_ON_TEST_DATA' and value and self.DATASET != 'mnist':
      raise ValueError('Invoke test data only when dataset is MNIST.')

    if (
        name == 'NUM_SAMPLES_PER_BASE_MODEL' and
        value < self.NUM_SAMPLES_TO_PLOT_TE_FOR
    ) or (
        name == 'NUM_SAMPLES_TO_PLOT_TE_FOR' and
        self.NUM_SAMPLES_PER_BASE_MODEL < value
    ):
      raise ValueError(
          'Num samples to plot treatment effects for '
          'must be less than num samples per base model.'
      )

    super(Config, self).__setattr__(name, value)

  def set_config_paths(self, attr_dict):
    """Update default config attributes with args passed in through cmd line.

    Args:
      attr_dict: a dictionary of keyword arguments for attributes to be updated.
    """

    if self.paths_set:
      raise ValueError('Paths should only be set once per execution.')

    # Confirm that we are setting already defined attributes.
    for key, value in attr_dict.items():
      self.__setattr__(key, value)

    # Create timestamp'ed experiment folder and subfolders.
    self.SETUP_NAME = (
        f'dataset_{self.DATASET}_'
        f'explainer_{self.EXPLANATION_TYPE}_'
        f'num_base_models_{self.NUM_BASE_MODELS}_'
        f'min_test_accuracy_{self.KEEP_MODELS_ABOVE_TEST_ACCURACY}_'
        f'num_image_samples_{self.NUM_SAMPLES_PER_BASE_MODEL}_'
        f'identical_samples_{self.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS}'
    )
    self.EXPERIMENT_DIR = (
        f'_experiments/{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}__'
        f'{self.SETUP_NAME}'
    )

    # Create folders to save results, models, and plots.
    if self.RUN_ON_TEST_DATA:
      self.DATA_DIR = 'test_data/'
    else:
      self.DATA_DIR = self.DATASET + '/'

    if RUNNING_INTERNALLY:
      self.EXP_DIR_PATH = os.path.join(CNS_PATH, self.EXPERIMENT_DIR)
      self.DATA_DIR_PATH = READAHEAD + os.path.join(CNS_PATH, self.DATA_DIR)
    else:
      self.EXP_DIR_PATH = os.path.join(
          os.path.dirname(__file__),
          self.EXPERIMENT_DIR,
      )
      self.DATA_DIR_PATH = os.path.join(
          os.path.dirname(__file__),
          self.DATA_DIR,
      )
    self.PLOTS_DIR_PATH = os.path.join(self.EXP_DIR_PATH, PLOTS_SUBPATH)
    self.MODELS_DIR_PATH = os.path.join(self.EXP_DIR_PATH, MODELS_SUBPATH)

    self.paths_set = True


cfg = Config()
