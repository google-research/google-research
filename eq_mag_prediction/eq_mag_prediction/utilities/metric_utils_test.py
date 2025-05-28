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

"""Tests for utilities.metric_utils."""

import numpy as np
import sklearn
import xarray as xr
from absl.testing import absltest
from eq_mag_prediction.utilities import metric_utils


class LogLossTest(absltest.TestCase):

  def test_values_and_dimensions(self):
    predicted = xr.DataArray(
        np.linspace(0, 1, 200).reshape(10, 20), dims=['x', 'y']
    )
    true_labels = predicted > 0.5
    xr_log_losses = metric_utils.xr_log_loss(true_labels, predicted, 'x')

    # Asserts that x is no longer a dimension since it was averaged on.
    self.assertNotIn('x', xr_log_losses)

    # Asserts numerical value of logloss for each y.
    for i, xr_log_loss in enumerate(xr_log_losses):
      sk_log_loss = sklearn.metrics.log_loss(true_labels[:, i], predicted[:, i])
      self.assertTrue(np.allclose(sk_log_loss, xr_log_loss))

  def test_scalar_input(self):
    random_state = np.random.RandomState(123)
    true_labels = xr.DataArray(
        random_state.choice([True, False], [10, 20]), dims=['x', 'y']
    )
    predicted_scalar = 0.6
    predicted_array = xr.full_like(true_labels, predicted_scalar, dtype='f')
    self.assertTrue(
        np.allclose(
            metric_utils.xr_log_loss(true_labels, predicted_scalar, 'y'),
            metric_utils.xr_log_loss(true_labels, predicted_array, 'y'),
        )
    )


if __name__ == '__main__':
  absltest.main()
