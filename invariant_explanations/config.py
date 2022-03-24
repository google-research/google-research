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
import numpy as np


RANDOM_SEED = 42

RUN_ON_TEST_DATA = False  # Set this variable to True to run on sample test data
                         # otherwise, download the metrics/weights for a desired
                         # dataset (asoutlined in the README), and set this
                         # variable to False.

ALLOWABLE_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'svhn_cropped']
DATASET = 'cifar10'
assert DATASET in ALLOWABLE_DATASETS
if RUN_ON_TEST_DATA:  # The test data is a subset of the MNIST dataset.
  assert DATASET == 'mnist'

if RUN_ON_TEST_DATA:
  DATA_DIR = 'test_data/'
else:
  DATA_DIR = DATASET + '/'

RUNNING_INTERNALLY = False

PLOTS_SUBPATH = '_plots'
MODELS_SUBPATH = '_models'

NUM_BASE_MODELS = 30000
NUM_SAMPLES_PER_BASE_MODEL = 8
NUM_SAMPLES_TO_PLOT_TE_FOR = 8
KEEP_MODELS_ABOVE_TEST_ACCURACY = 0.55
USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS = True
assert NUM_SAMPLES_TO_PLOT_TE_FOR <= NUM_SAMPLES_PER_BASE_MODEL

ALLOWABLE_EXPLANATION_METHODS = [
    'ig',
    'grad',
    'gradcam',
    'guided_ig',
    'smooth_ig',
    'smooth_grad',
    'smooth_gradcam',
]
EXPLANATION_TYPE = 'guided_ig'
assert EXPLANATION_TYPE in ALLOWABLE_EXPLANATION_METHODS

BASE_MODEL_BATCH_SIZE = 32
META_MODEL_BATCH_SIZE = 32
TRAIN_FRACTIONS = [0.01, 0.03, 0.1]
if RUNNING_INTERNALLY:
  META_MODEL_EPOCHS = 50
else:
  META_MODEL_EPOCHS = 10

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

COVARIATES_SETTINGS = [
    # {'chkpt': 0},
    # {'chkpt': 1},
    # {'chkpt': 2},
    # {'chkpt': 20},
    # {'chkpt': 40},
    # {'chkpt': 60},
    {'chkpt': 86},
]

# Create timestamp'ed experiment folder and subfolders.
SETUP_NAME = (
    f'dataset_{DATASET}_'
    f'num_base_model_{NUM_BASE_MODELS}_'
    f'min_test_accuracy_{KEEP_MODELS_ABOVE_TEST_ACCURACY}_'
    f'num_image_samples_{NUM_SAMPLES_PER_BASE_MODEL}_'
    f'identical_samples_{USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS}'
)
EXPERIMENT_DIR = (
    f'_experiments/{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}__'
    f'{SETUP_NAME}'
)

# Create folders to save results, models, and plots.
if RUNNING_INTERNALLY:
  EXP_DIR_PATH = os.path.join(CNS_PATH, EXPERIMENT_DIR)
  DATA_DIR_PATH = READAHEAD + os.path.join(CNS_PATH, DATA_DIR)
else:
  EXP_DIR_PATH = os.path.join(os.path.dirname(__file__), EXPERIMENT_DIR)
  DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), DATA_DIR)
PLOTS_DIR_PATH = os.path.join(EXP_DIR_PATH, PLOTS_SUBPATH)
MODELS_DIR_PATH = os.path.join(EXP_DIR_PATH, MODELS_SUBPATH)
