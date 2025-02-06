# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Main file used for approxNN project."""

from typing import Sequence
import warnings

from absl import app
from absl import flags

import config
import other
import plotting
import utils


FLAGS = flags.FLAGS

_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    42,
    'The seed used for all numpy random operations.',
)
_DATASET = flags.DEFINE_string(
    'dataset',
    'cifar10',
    'The dataset, chosen from config.ALLOWABLE_DATASETS.',
)
_EXPLANATION_TYPE = flags.DEFINE_string(
    'explanation_type',
    'ig',
    'The explanation method, chosen from config.ALLOWABLE_EXPLANATION_METHODS.',
)
_EXPLAN_NORM_TYPE = flags.DEFINE_string(
    'explan_norm_type',
    '01',
    'The method used to normalize explanations.'
)
_TREATMENT_KERNEL = flags.DEFINE_string(
    'treatment_kernel',
    'rbf',
    'The kernel to use for measuring the (extended notion of) treatment effect.'
)
_RUN_ON_TEST_DATA = flags.DEFINE_boolean(
    'run_on_test_data',
    False,
    'The flag used to specify whether or not to run on sample test images.',
)
_RUN_ON_PRECOMPUTED_GCP_DATA = flags.DEFINE_boolean(
    'run_on_precomputed_gcp_data',
    False,
    'The flag used to specify whether or not to analyze precomputed GCP data',
)
_NUM_BASE_MODELS = flags.DEFINE_integer(
    'num_base_models',
    30000,
    'The number of base models to load from the CNN Zoo.',
)
_MIN_BASE_MODEL_ACCURACY = flags.DEFINE_float(
    'min_base_model_accuracy',
    0.55,
    'The threshold to use when select models from the CNN Zoo.',
)
_MAX_BASE_MODEL_ACCURACY = flags.DEFINE_float(
    'max_base_model_accuracy',
    1.00,
    'The threshold to use when select models from the CNN Zoo.',
)
_NUM_SAMPLES_PER_BASE_MODEL = flags.DEFINE_integer(
    'num_samples_per_base_model',
    32,
    'The number of sample images to use per base model.',
)
_NUM_SAMPLES_TO_PLOT_TE_FOR = flags.DEFINE_integer(
    'num_samples_to_plot_te_for',
    8,
    'The number of samples for which to plot treatment effects.',
)
_NUM_BASE_MODELS_FOR_KERNEL = flags.DEFINE_integer(
    'num_base_models_for_kernel',
    100,
    'The number of base models to split into h_i vs h_not_i to compute ITE.',
)
_USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS = flags.DEFINE_boolean(
    'use_identical_samples_over_base_models',
    True,
    'A flag indicating whether or not to use identical samples on base models.',
)
_MODEL_BATCH_COUNT = flags.DEFINE_integer(
    'model_batch_count',
    1,
    'A total number of model batches to use (from a total of num_base_models).',
)
_MODEL_BATCH_IDX = flags.DEFINE_integer(
    'model_batch_idx',
    0,
    'The index of the batch of models to use for analysis.',
)


warnings.simplefilter('ignore')


def main(argv):

  # Update config file defaults if the arguments are passed in via the cmd line.
  config.cfg.set_config_paths({
      'RANDOM_SEED': _RANDOM_SEED.value,
      'DATASET': _DATASET.value,
      'EXPLANATION_TYPE': _EXPLANATION_TYPE.value,
      'EXPLAN_NORM_TYPE': _EXPLAN_NORM_TYPE.value,
      'TREATMENT_KERNEL': _TREATMENT_KERNEL.value,
      'RUN_ON_TEST_DATA': _RUN_ON_TEST_DATA.value,
      'RUN_ON_PRECOMPUTED_GCP_DATA': _RUN_ON_PRECOMPUTED_GCP_DATA.value,
      'NUM_BASE_MODELS': _NUM_BASE_MODELS.value,
      'MIN_BASE_MODEL_ACCURACY': _MIN_BASE_MODEL_ACCURACY.value,
      'MAX_BASE_MODEL_ACCURACY': _MAX_BASE_MODEL_ACCURACY.value,
      'NUM_SAMPLES_PER_BASE_MODEL': _NUM_SAMPLES_PER_BASE_MODEL.value,
      'NUM_SAMPLES_TO_PLOT_TE_FOR': _NUM_SAMPLES_TO_PLOT_TE_FOR.value,
      'NUM_BASE_MODELS_FOR_KERNEL': _NUM_BASE_MODELS_FOR_KERNEL.value,
      'USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS': (
          _USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS.value
      ),
      'MODEL_BATCH_COUNT': _MODEL_BATCH_COUNT.value,
      'MODEL_BATCH_IDX': _MODEL_BATCH_IDX.value,
  })

  utils.create_experimental_folders()

  # utils.analyze_accuracies_of_base_models()

  # utils.process_and_resave_cnn_zoo_data(
  #     config.cfg.RANDOM_SEED,
  #     other.get_model_wireframe(),
  #     config.cfg.COVARIATES_SETTINGS,
  # )

  # utils.train_meta_model_over_different_setups(config.cfg.RANDOM_SEED)

  # utils.save_heat_map_of_meta_model_results()

  # utils.process_per_class_explanations(config.cfg.RANDOM_SEED)

  # utils.measure_prediction_explanation_variance(config.cfg.RANDOM_SEED)
  # utils.measure_prediction_explanation_variance_all(config.cfg.RANDOM_SEED)

  # utils.measure_equivalence_class_of_explanations(config.cfg.RANDOM_SEED)

  plotting.plot_paper_figures()

if __name__ == '__main__':
  app.run(main)
