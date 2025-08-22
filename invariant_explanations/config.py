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

"""Config used for approxNN project."""
# pylint: skip-file

import datetime
import os
import sys

from absl import logging
import numpy as np


RUNNING_INTERNALLY = False

# Create folders to save results, models, and plots.
PAPER_BU_PATH = '_paper_bu'
MEDIATION_TYPE = 'mediated'
# MEDIATION_TYPE = 'unmediated'
MERGED_DATA_PATH = '_merged_' + MEDIATION_TYPE
PLOTS_SUBPATH = '_plots'
MODELS_SUBPATH = '_models'


MARKER_COLORS = ['blue', 'orange', 'green', 'red', 'magenta']
MARKER_SHAPES = ['o', 'X', 's', 'P', 'D']

# Define constants.
ALLOWABLE_DATA_FILES = [
    'samples',
    'y_preds',
    'y_trues',
    'w_chkpt',
    'w_final',
    'explans',
    'hparams',
    'metrics',
]
ALLOWABLE_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped']
ALLOWABLE_EXPLANATION_METHODS = [
    'grad',
    'smooth_grad',
    'ig',
    'gradcam',
    # 'guided_ig',
    # 'smooth_ig',
    # 'smooth_gradcam',
]
ALLOWABLE_EXPLAN_NORM_METHODS = [
    '01',
    '-11',
    'l21',
    # '-11_clip_01',
    # '01_no_percentile',
]
ALLOWABLE_TREATMENT_KERNELS = [
    'linear',
    'poly',
    'rbf',
    'cosine',
]

RANGE_ACCURACY_CONVERTER = {
  # cifar10
  '0.056_0.154': '05 - 15\%', # '0 - 20 pctl.',
  '0.155_0.253': '15 - 25\%', # '20 - 40 pctl.',
  '0.253_0.33': '25 - 33\%', # '40 - 60 pctl.',
  '0.33_0.385': '33 - 38\%', # '60 - 80 pctl.',
  '0.385_0.461': '38 - 46\%', # '80 - 90 pctl.',
  '0.461_0.501': '46 - 50\%', # '90 - 95 pctl.',
  '0.501_0.521': '50 - 52\%', # '95 - 99 pctl.',
  '0.521_0.575': '52 - 57\%', # '99 - 100 pctl.',
  # svhn_cropped
  '0.07_0.179': '07 - 17\%', # '0 - 20 pctl.',
  '0.179_0.195': '17 - 19.5\%', # '20 - 40 pctl.',
  '0.195_0.196': '19.5 - 19.6\%', # '40 - 60 pctl.',
  '0.196_0.333': '19.6 - 33\%', # '60 - 80 pctl.',
  '0.333_0.516': '33 - 51\%', # '80 - 90 pctl.',
  '0.516_0.595': '51 - 59\%', # '90 - 95 pctl.',
  '0.595_0.653': '59 - 65\%', # '95 - 99 pctl.',
  '0.653_0.781': '65 - 78\%', # '99 - 100 pctl.',
  # mnist
  '0.047_0.113': '4 - 11\%', # '0 - 20 pctl.',
  '0.113_0.359': '11 - 35\%', # '20 - 40 pctl.',
  '0.359_0.739': '35 - 73\%', # '40 - 60 pctl.',
  '0.739_0.898': '73 - 89\%', # '60 - 80 pctl.',
  '0.898_0.955': '89 - 95\%', # '80 - 90 pctl.',
  '0.955_0.969': '95 - 96\%', # '90 - 95 pctl.',
  '0.969_0.974': '96 - 97\%', # '95 - 99 pctl.',
  '0.974_0.986': '97 - 98\%', # '99 - 100 pctl.',
  # fashion_mnist
  '0.016_0.118': '1 - 11\%', # '0 - 20 pctl.',
  '0.118_0.474': '11 - 47\%', # '20 - 40 pctl.',
  '0.474_0.686': '47 - 68\%', # '40 - 60 pctl.',
  '0.686_0.762': '68 - 76\%', # '60 - 80 pctl.',
  '0.762_0.826': '76 - 82\%', # '80 - 90 pctl.',
  '0.826_0.846': '82 - 84\%', # '90 - 95 pctl.',
  '0.846_0.857': '84 - 85\%', # '95 - 99 pctl.',
  '0.857_0.887': '85 - 88\%', # '99 - 100 pctl.',
}
EXPLAN_NAME_CONVERTER = {
  'grad': 'Grad',
  'smooth_grad': 'SG',
  'ig': 'IG',
  'gradcam': 'Grad-CAM',
}
KERNEL_NAME_CONVERTER = {
  'linear': 'Linear Kernel',
  'poly': 'Polynomial Kernel',
  'rbf': 'RBF Kernel',
  'cosine': 'Cosine Kernel',
}
HPARAM_NAME_CONVERTER = {
  'config.b_init': r"$b_{init}$",
  'config.w_init': r"$w_{init}$",
  'config.optimizer': r"Optimizer",
  'config.activation': r"Activation",
  'config.l2reg': r"$\ell_2$",
  'config.dropout': r"Dropout",
  'config.init_std': r"$\sigma^2_{init}$",
  'config.learning_rate': r"Learning Rate",
  'config.train_fraction': r"Training Fraction",
}

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

NUM_MODELS_PER_DATASET = {
  'mnist': 269973,
  'fashion_mnist': 270000,
  'cifar10': 270000,
  'svhn_cropped': 269892,
}

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
    self.EXPLAN_NORM_TYPE = '01'
    self.TREATMENT_KERNEL = 'rbf'
    self.RUN_ON_TEST_DATA = False
    self.RUN_ON_PRECOMPUTED_GCP_DATA = False
    self.NUM_BASE_MODELS = 30000
    self.NUM_SAMPLES_PER_BASE_MODEL = 8
    self.NUM_SAMPLES_TO_PLOT_TE_FOR = 8
    self.NUM_BASE_MODELS_FOR_KERNEL = 100
    self.MIN_BASE_MODEL_ACCURACY = 0.55
    self.MAX_BASE_MODEL_ACCURACY = 1.00
    self.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS = True
    self.MODEL_BATCH_IDX = 0
    self.MODEL_BATCH_COUNT = 1

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

    if (
        name == 'EXPLAN_NORM_TYPE' and
        value not in ALLOWABLE_EXPLAN_NORM_METHODS
    ):
      raise ValueError('EXPLAN_NORM_TYPE not recognized.')

    if (
        name == 'TREATMENT_KERNEL' and
        value not in ALLOWABLE_TREATMENT_KERNELS
    ):
      raise ValueError('TREATMENT_KERNEL not recognized.')

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

    if self.MODEL_BATCH_IDX >= self.MODEL_BATCH_COUNT:
      raise ValueError(
          'The value of model_batch_idx must be smaller than model_batch_count.'
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
        f'explanation_type_{self.EXPLANATION_TYPE}_'
        f'explan_norm_type_{self.EXPLAN_NORM_TYPE}_'
        f'num_base_models_{self.NUM_BASE_MODELS}_'
        f'min_test_accuracy_{self.MIN_BASE_MODEL_ACCURACY}_'
        f'max_test_accuracy_{self.MAX_BASE_MODEL_ACCURACY}_'
        f'num_image_samples_{self.NUM_SAMPLES_PER_BASE_MODEL}_'
        f'identical_samples_{self.USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS}'
        f'batch_{self.MODEL_BATCH_IDX}_of_{self.MODEL_BATCH_COUNT}'
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
