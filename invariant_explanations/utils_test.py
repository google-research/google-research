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

"""Unittest for methods in utils.py."""

import os
import pickle

from absl import flags
from absl.testing import absltest

from invariant_explanations import config
from invariant_explanations import utils

FLAGS = flags.FLAGS


class UtilsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    """This method will be run once for all test methods in the class."""
    super().setUpClass()

    # Override some configuration parameters for reproducible tests.
    config.cfg.set_config_paths({
        'RANDOM_SEED': 42,
        'DATASET': 'mnist',
        'DATA_DIR': 'test_data/',
        'EXPLANATION_TYPE': 'ig',
        'RUN_ON_TEST_DATA': True,
        'NUM_BASE_MODELS': 30000,
        'NUM_SAMPLES_PER_BASE_MODEL': 8,
        'NUM_SAMPLES_TO_PLOT_TE_FOR': 8,
        'KEEP_MODELS_ABOVE_TEST_ACCURACY': 0.98,
        'USE_IDENTICAL_SAMPLES_OVER_BASE_MODELS': True,
        'COVARIATES_SETTINGS': [{'chkpt': 86}],
    })

    if not config.RUNNING_INTERNALLY:
      config.cfg.DATA_DIR_PATH = os.path.join(
          os.path.dirname(__file__),
          config.cfg.DATA_DIR,
      )

    utils.create_experimental_folders()

  def test_analyze_accuracies_of_base_models(self):

    # Lookup the accuracies of saved base models and save resulting analysis.
    utils.analyze_accuracies_of_base_models()

    # Load the saved results.
    with utils.file_handler(
        config.cfg.EXP_DIR_PATH,
        'accuracy_tracker.npy',
        'rb',
    ) as f:
      acc_tracker = pickle.load(f)

    # Assert that hand-crafted test data is correct by checking its properties.
    self.assertLen(acc_tracker, 50)
    self.assertLen(acc_tracker[acc_tracker['accuracy_type'] == 'train'], 25)
    self.assertLen(acc_tracker[acc_tracker['accuracy_type'] == 'test'], 25)
    self.assertLen(acc_tracker[acc_tracker['chkpt'] == 20], 2)
    self.assertLen(acc_tracker[acc_tracker['chkpt'] == 40], 4)
    self.assertLen(acc_tracker[acc_tracker['chkpt'] == 60], 12)
    self.assertLen(acc_tracker[acc_tracker['chkpt'] == 80], 14)
    self.assertLen(acc_tracker[acc_tracker['chkpt'] == 86], 18)
    self.assertAlmostEqual(max(acc_tracker['accuracy']), 0.994, delta=1e-3)
    self.assertAlmostEqual(min(acc_tracker['accuracy']), 0.979, delta=1e-3)


if __name__ == '__main__':
  absltest.main()
