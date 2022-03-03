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

"""Tests for loss."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sklearn import metrics

from symbolic_functionals.syfes import loss


class DatasetTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_weighted_root_mean_square_deviation(self, use_jax):
    y_pred = np.random.rand(10)
    y_true = np.random.rand(10)
    weights = np.random.rand(10)

    np.testing.assert_almost_equal(
        loss.weighted_root_mean_square_deviation(
            y_pred, y_true, weights, use_jax=use_jax),
        np.sqrt(
            metrics.mean_squared_error(y_pred, y_true, sample_weight=weights) *
            np.sum(weights) / 10))

  def test_combine_wrmsd(self):
    num_samples_1 = 10
    num_samples_2 = 5
    y_pred_1 = np.random.rand(num_samples_1)
    y_true_1 = np.random.rand(num_samples_1)
    weights_1 = np.random.rand(num_samples_1)
    y_pred_2 = np.random.rand(num_samples_2)
    y_true_2 = np.random.rand(num_samples_2)
    weights_2 = np.random.rand(num_samples_2)

    np.testing.assert_almost_equal(
        loss.combine_wrmsd(
            coeff_1=num_samples_1 / (num_samples_1 + num_samples_2),
            coeff_2=num_samples_2 / (num_samples_1 + num_samples_2),
            wrmsd_1=loss.weighted_root_mean_square_deviation(
                y_pred_1, y_true_1, weights_1, use_jax=False),
            wrmsd_2=loss.weighted_root_mean_square_deviation(
                y_pred_2, y_true_2, weights_2, use_jax=False),
            ),
        loss.weighted_root_mean_square_deviation(
            np.concatenate([y_pred_1, y_pred_2]),
            np.concatenate([y_true_1, y_true_2]),
            np.concatenate([weights_1, weights_2]),
            use_jax=False)
        )


if __name__ == '__main__':
  absltest.main()
