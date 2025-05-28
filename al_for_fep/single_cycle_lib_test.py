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

"""Tests for experiment_lib."""
import os
import tempfile
import unittest

import ml_collections
import pandas as pd

from al_for_fep import single_cycle_lib

_TEST_CONFIG = ml_collections.ConfigDict({
    'model_config':
        ml_collections.ConfigDict({
            'model_type':
                'rf',
            'hyperparameters':
                ml_collections.ConfigDict({
                    'criterion': 'squared_error',
                    'max_depth': 10,
                    'min_samples_split': 0.01,
                    'min_samples_leaf': 0.01,
                    'min_weight_fraction_leaf': 0.0,
                    'min_impurity_decrease': 0.0,
                    'bootstrap': True,
                    'oob_score': False,
                    'random_state': 142753869,
                    'verbose': 0,
                    'warm_start': True,
                    'ccp_alpha': 0.0,
                }),
            'tuning_hyperparameters': [{
                'n_estimators': [30]
            }],
            'data_splitting':
                ml_collections.ConfigDict({
                    'strategy': 'noop',
                    'params': {}
                }),
            'features':
                ml_collections.ConfigDict({
                    'feature_type': 'fingerprint',
                    'params': {
                        'feature_column': 'SMILES',
                        'fingerprint_size': 256,
                        'fingerprint_radius': 3
                    }
                }),
            'targets':
                ml_collections.ConfigDict({
                    'feature_type': 'number',
                    'params': {
                        'feature_column': 'lig_b_perses_dg',
                    }
                }),
            'replication':
                ml_collections.ConfigDict({
                    'replicas': 1,
                    'replicator': 'naive'
                }),
            'compile_params': {},
            'fit_params': {},
            'predict_params': {}
        }),
    'selection_config':
        ml_collections.ConfigDict({
            'selection_type': 'greedy',
            'hyperparameters': ml_collections.ConfigDict({}),
            'num_elements': 1,
            'selection_columns': ['SMILES', 'lig_b_perses_dg']
        }),
    'metadata': ('Test cycle. No one should be seeing this in a file unless '
                 'debugging.')
})


class ExperimentLibTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

    _TEST_CONFIG.cycle_dir = tempfile.mkdtemp()
    training_handle, _TEST_CONFIG.training_pool = tempfile.mkstemp()

    pd.DataFrame({
        'SMILES': ['C', 'O', 'CCON', 'CCN', 'CCCC']
    }).to_csv(
        _TEST_CONFIG.training_pool, index=False)

    lib_handle, _TEST_CONFIG.virtual_library = tempfile.mkstemp()

    pd.DataFrame({
        'SMILES': ['CCC', 'ONO', 'OCCO', 'C', 'O', 'CCON', 'CCN', 'CCCC'],
        'lig_b_perses_dg': [1.4, 5.2, 7, 12, 10, 7, 2, 3]
    }).to_csv(
        _TEST_CONFIG.virtual_library, index=False)

    os.close(training_handle)
    os.close(lib_handle)

  def test_cycle_success(self):
    test_cycle = single_cycle_lib.MakitaCycle(_TEST_CONFIG)
    test_cycle.run_cycle()

    self._verify_cycle_result()

  def test_cycle_idempotency(self):
    test_cycle = single_cycle_lib.MakitaCycle(_TEST_CONFIG)
    test_cycle.run_cycle()
    test_cycle.run_cycle()

    self._verify_cycle_result()

  def test_missing_metadata_throws(self):
    _TEST_CONFIG.metadata = ''
    test_cycle = single_cycle_lib.MakitaCycle(_TEST_CONFIG)

    with self.assertRaisesRegex(
        ValueError,
        'Expected non-empty "metadata" key with description of cycle.'):
      test_cycle.run_cycle()

  def _verify_cycle_result(self):
    inference_results = pd.read_csv(
        os.path.join(_TEST_CONFIG.cycle_dir,
                     'virtual_library_with_predictions.csv'))
    inference_results.loc[:, 'regression'] = inference_results[
        'regression'].astype(str)

    selection_results = pd.read_csv(
        os.path.join(_TEST_CONFIG.cycle_dir, 'selection.csv'))

    pd.testing.assert_frame_equal(
        inference_results.sort_index(axis=1),
        pd.DataFrame({
            'SMILES': ['CCC', 'ONO', 'OCCO', 'C', 'O', 'CCON', 'CCN', 'CCCC'],
            'regression': [
                '5.26', '9.74', '8.02', '10.65', '9.61', '5.89', '3.08', '5.11'
            ],
            'Training Example': [
                False, False, False, True, True, True, True, True
            ],
            'lig_b_perses_dg': [1.4, 5.2, 7, 12, 10, 7, 2, 3]
        }).sort_index(axis=1))

    pd.testing.assert_frame_equal(
        selection_results.sort_index(axis=1),
        pd.DataFrame({
            'SMILES': ['CCC'],
            'lig_b_perses_dg': [1.4]
        }).sort_index(axis=1))


if __name__ == '__main__':
  unittest.main()
