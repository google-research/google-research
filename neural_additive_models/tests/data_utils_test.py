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

# Lint as: python3
"""Tests functionality of loading the different datasets."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from neural_additive_models import data_utils


class LoadDataTest(parameterized.TestCase):
  """Data Loading Tests."""

  @parameterized.named_parameters(
      ('breast_cancer', 'BreastCancer', 569),
      ('fico', 'Fico', 9861),
      ('housing', 'Housing', 20640),
      ('recidivism', 'Recidivism', 6172),
      ('credit', 'Credit', 284807),
      ('adult', 'Adult', 32561),
      ('telco', 'Telco', 7043))
  def test_data(self, dataset_name, dataset_size):
    """Test whether a dataset is loaded properly with specified size."""
    x, y, _ = data_utils.load_dataset(dataset_name)
    self.assertIsInstance(x, np.ndarray)
    self.assertIsInstance(y, np.ndarray)
    self.assertLen(x, dataset_size)

if __name__ == '__main__':
  absltest.main()
