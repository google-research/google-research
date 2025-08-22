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

"""Tests for data utils library."""
import unittest

from absl.testing import parameterized
import numpy as np
import pandas as pd

from al_for_fep.data import utils

_FEATURE_COLUMN = 'features'


class UtilsTest(parameterized.TestCase):

  def test_parse_vector_success(self):
    vector_one = [1, 0, 3]
    vector_two = [14.4, 15.1, -1.2]

    test_data = pd.DataFrame({
        _FEATURE_COLUMN: [
            '[' + ','.join(map(str, vector_values)) + ']'
            for vector_values in [vector_one, vector_two]
        ]
    })

    result = utils.parse_feature_vectors(
        test_data, feature_column=_FEATURE_COLUMN)

    np.testing.assert_array_equal(result, np.array([vector_one, vector_two]))

  @parameterized.named_parameters(
      ('bad_vector_format_throws',
       pd.DataFrame({_FEATURE_COLUMN: ['not_a_vector']}), ValueError),
      ('bad_value_in_vector_throws',
       pd.DataFrame({_FEATURE_COLUMN: ['[3, |, 4]']}), SyntaxError))
  def test_parse_vector(self, test_data, expected_error):
    with self.assertRaises(expected_error):
      utils.parse_feature_vectors(test_data, feature_column=_FEATURE_COLUMN)

  @parameterized.named_parameters(
      ('success', pd.DataFrame({_FEATURE_COLUMN: [-4.73, 2.56, -36]
                               }), np.array([-4.73, 2.56, -36])),
      ('string_cast_success',
       pd.DataFrame({_FEATURE_COLUMN: ['1', '-1.2', '15.7']
                    }), np.array([1, -1.2, 15.7])))
  def test_parse_number(self, test_data, expected_result):
    result = utils.parse_feature_numbers(
        test_data, feature_column=_FEATURE_COLUMN)

    np.testing.assert_array_equal(result, expected_result)

  def test_parse_number_invalid_value_throws(self):
    values = pd.DataFrame({_FEATURE_COLUMN: [1, 'not_a_float', 6]})

    with self.assertRaises(ValueError):
      utils.parse_feature_numbers(values, feature_column=_FEATURE_COLUMN)

  def test_fingerprint_parse_success(self):
    test_data = pd.DataFrame({'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'C']})

    results = utils.parse_feature_smiles_morgan_fingerprint(
        feature_dataframe=test_data,
        feature_column='smiles',
        fingerprint_radius=2,
        fingerprint_size=8)

    np.testing.assert_array_equal(
        results, np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0]]))

  def test_rdkit_descriptors_success(self):
    test_data = pd.DataFrame({'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'C']})

    results = utils.parse_feature_smiles_rdkit_properties(
        feature_dataframe=test_data, feature_column='smiles')

    np.testing.assert_array_almost_equal(
        results[:, :12],
        np.array([[
            10.61195, -1.114028, 10.61195, .01601852, 0.5501218, 180.1590,
            172.0950, 180.0423, 68.00000, 0.00, .3390038, -0.4775395
        ],
                  [
                      0., 0., 0., 0., 0.359785, 16.043, 12.011, 16.0313, 8., 0.,
                      -0.077558, -0.077558
                  ]]),
        decimal=4)

  def test_fingerprint_and_descriptors_success(self):
    test_data = pd.DataFrame({'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'C']})

    results = utils.parse_feature_smiles_morgan_fingerprint_with_descriptors(
        feature_dataframe=test_data,
        feature_column='smiles',
        fingerprint_radius=2,
        fingerprint_size=8)

    np.testing.assert_array_almost_equal(
        results[:, :20],
        np.array([[
            1., 1., 1., 1., 1., 1., 1., 1., 10.61195, -1.114028, 10.61195,
            .01601852, 0.5501218, 180.1590, 172.0950, 180.0423, 68.00000, 0.00,
            .3390038, -0.4775395
        ],
                  [
                      1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.359785,
                      16.043, 12.011, 16.0313, 8., 0., -0.077558, -0.077558
                  ]]),
        decimal=4)


if __name__ == '__main__':
  unittest.main()
