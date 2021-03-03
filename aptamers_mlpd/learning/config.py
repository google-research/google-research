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

# Lint as: python3
"""This file holds constants we reuse across the train and eval scripts."""

# The order of these bases is crucial in converting to/from the one hot array.
# The order here must match the order for 'bases' in custom_ops.cc.
ORDERED_BASES = 'ATGC'

# Configuration for standard filtering of aptamer reads.
EXPECTED_APTAMER_LENGTH = 40
MIN_BASE_QUALITY = 20
MAX_BAD_BASES = 5
MIN_AVG_READ_QUALITY = 30.0
MAX_PAIR_DISSIMILARITY = 5

tr_report_name = 'tr_report.json'
eval_report_name = 'eval_report.json'

wetlab_experiment_train_name = 'wetlab_experiment_train.pbtxt'
wetlab_experiment_val_name = 'wetlab_experiment_val.pbtxt'

hparams_name = 'hparams.pbtxt'

val_fold = 0
num_folds = 5

# TensorFlow experiment-related constants
experiment_training_dir = 'eval-training'
experiment_validation_dir = 'eval-validation'
experiment_report_name = 'report.nc'
experiment_best_report_name = 'best_model.nc'


def get_wetlab_experiment_train_pbtxt_path(fold):
  """Returns the basename of the training proto.

  Args:
    fold: The integer fold of interest.

  Returns:
    The basename of the training proto.

  Raises:
    ValueError: The requested fold is invalid.
  """
  if not 0 <= fold < num_folds:
    raise ValueError('Invalid fold: %i' % fold)
  return 'experiment_fold_%i_train.pbtxt' % fold


def get_wetlab_experiment_val_pbtxt_path(fold, template=''):
  """Returns the basename of the validation proto.

  Args:
    fold: The integer fold of interest.
    template: An optional string to change the name of the proto used.

  Returns:
    The basename of the validation proto.

  Raises:
    ValueError: The requested fold is invalid.
  """
  if not 0 <= fold < num_folds:
    raise ValueError('Invalid fold: %i' % fold)
  return 'experiment_%sfold_%i_test.pbtxt' % (template, fold)


def get_example_sstable_path(fold, template=''):
  """Returns the basename of the SSTable fold.

  Args:
    fold: The integer fold of interest.
    template: An optional string to change the name of the SSTable used.

  Returns:
    The basename of the SSTable.

  Raises:
    ValueError: The requested fold is invalid.
  """
  if not 0 <= fold < num_folds:
    raise ValueError('Invalid fold: %i' % fold)
  return 'examples_%sfold_%i.sstable' % (template, fold)


def get_hdf5_path(fold, template=''):
  """Returns the basename of the HDF5 representation of the fold data.

  Args:
    fold: The integer fold of interest.
    template: An optional string to change the name of the HDF5 file used.

  Returns:
    The basename of the HDF5 file.

  Raises:
    ValueError: The requested fold is invalid.
  """
  if not 0 <= fold < num_folds:
    raise ValueError('Invalid fold: %i' % fold)
  return 'table_%sfold_%i.h5' % (template, fold)


wetlab_experiment_train_pbtxt_path = [
    get_wetlab_experiment_train_pbtxt_path(n) for n in range(num_folds)
]

wetlab_experiment_val_pbtxt_path = [
    get_wetlab_experiment_val_pbtxt_path(n) for n in range(num_folds)
]

example_sstable_paths = [get_example_sstable_path(n) for n in range(num_folds)]

hdf5_paths = [get_hdf5_path(n) for n in range(num_folds)]

# Directories where commonly-used input data reside.
_BASEDIR = 'xxx'
INPUT_DATA_DIRS = {
    'xxx':
        _BASEDIR + 'xxx/paired/low_quality/folds',
    'aptitude':
        _BASEDIR + 'aptitude/r=3/fastq/processed4/folds',
}

# Target counts used to compute affinity in the fully observed model for
# each dataset (see predict_affinity in FullyObserved output_layers).
# These maps aren't used in training but are used in inference when an affinity
# is desired.
#
# For each dataset, we define a dictionary where each key is a selection
# affinity molecule (e.g. the protein used in selection) and each value is a
# tuple of predicted target output names to use when calculating affinity.
DEFAULT_AFFINITY_TARGET_MAPS = {
    'xxx': {
        'xxx': ['round5_murine'],
        'xxx': ['round5'],
        'xxx': ['round5_igg']
    },
    'aptitude': {
        'target': [
            'round2_high_no_serum_positive', 'round2_medium_no_serum_positive'
        ],
        'serum': [
            'round2_high_with_serum_positive',
            'round2_medium_with_serum_positive'
        ]
    },
    'aptitude_binned': {
        'target': [
            'low_3bins',
            'med_3bins',
            'high_3bins',
        ],
    },
    'aptitude_super_binned': {
        'target': ['super_bin',],
    },
    'aptitudecluster': {
        'target': [
            'round2_high_no_serum_positive', 'round2_medium_no_serum_positive'
        ],
        'serum': [
            'round2_high_with_serum_positive',
            'round2_medium_with_serum_positive'
        ]
    },
    'for_testing': {
        'proteinA': ['round2_A_count'],
        'proteinB': ['round1_B_count', 'round2_B_count']
    },
}
