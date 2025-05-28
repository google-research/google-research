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

"""Tests for statistics_utils."""

import timeit

import numpy as np
import tensorflow_probability as tfp

from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import statistics_utils


class StatisticsTest(parameterized.TestCase):
  normal_inst_0_1 = tfp.distributions.Normal(0, 1)
  normal_inst_3_1 = tfp.distributions.Normal(3, 1)
  lognormal_inst_0_1 = tfp.distributions.LogNormal(0, 1)
  weibull_inst_2_3 = tfp.distributions.Weibull(2, 3)

  @parameterized.named_parameters(
      ('normal_acc_1e-2', normal_inst_0_1, 0.3, [-np.inf, np.inf], 1e-2),
      ('normal_acc_1e-3', normal_inst_0_1, 0.3, [-np.inf, np.inf], 1e-3),
      ('normal_mu=3_acc_1e-2', normal_inst_3_1, 0.3, [-np.inf, np.inf], 1e-2),
      ('lognormal_acc_1e-2', lognormal_inst_0_1, 0.3, [0, np.inf], 1e-2),
      ('weibull_acc_1e-2', weibull_inst_2_3, 0.3, [0, np.inf], 1e-2),
      ('weibull_acc_1e-3', weibull_inst_2_3, 0.3, [0, np.inf], 1e-3),
  )
  def test_quantile_for_p_val(self, distribution_inst, p_value, support, acc):
    shift = 0
    numerical_x = statistics_utils.quantile_for_p_val(
        distribution_inst, p_value, support, shift, acc
    )
    p_val_tf = 1 - distribution_inst.cdf(numerical_x)
    np.testing.assert_allclose(p_value, p_val_tf, rtol=0, atol=acc)

  def test_quantile_computation_verify_operation_different_batch_shape(self):
    p_value, support, shift, acc = (0.3, [0, np.inf], 0, 1e-2)

    weibull_inst_2_3_scalar_input = tfp.distributions.Weibull(2, 3)
    quantile_scalar_input = statistics_utils.quantile_for_p_val(
        weibull_inst_2_3_scalar_input, p_value, support, shift, acc
    )

    weibull_inst_2_3_list_len_1 = tfp.distributions.Weibull([2], [3])
    quantile_list_len_1 = statistics_utils.quantile_for_p_val(
        weibull_inst_2_3_list_len_1, p_value, support, shift, acc
    )

    n = 4
    weibull_inst_2_3_list_len_n = tfp.distributions.Weibull([2] * n, [3] * n)
    quantile_list_len_n = statistics_utils.quantile_for_p_val(
        weibull_inst_2_3_list_len_n, p_value, support, shift, acc
    )

    np.testing.assert_allclose(
        quantile_scalar_input, quantile_list_len_1, atol=acc, rtol=0
    )
    np.testing.assert_allclose(
        np.repeat(quantile_list_len_1, n), quantile_list_len_n, atol=acc, rtol=0
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='low_acc_will_converge',
          acc=1e-0,
          max_iterations=1_000,
          will_converge=True,
      ),
      dict(
          testcase_name='high_acc_will_not_converge',
          acc=1e-40,
          max_iterations=1_000,
          will_converge=False,
      ),
      dict(
          testcase_name='med_acc_few_iterations_will_not_converge',
          acc=1e-5,
          max_iterations=1,
          will_converge=False,
      ),
      dict(
          testcase_name='med_acc_lots_iterations_will_not_converge',
          acc=1e-5,
          max_iterations=1_000,
          will_converge=True,
      ),
  )
  def test_quantile_computation_convergence(
      self, acc, will_converge, max_iterations
  ):
    p_value, support, shift = (0.3, [0, np.inf], 0)

    k_vec = [1, 4, 3]
    l_vec = [3, 2, 1]
    weibull_inst = tfp.distributions.Weibull(k_vec, l_vec)
    numerical_quantile = statistics_utils.quantile_for_p_val(
        weibull_inst, p_value, support, shift, acc, max_iterations
    )
    self.assertTrue(all(np.isnan(numerical_quantile) == (not will_converge)))
    self.assertTrue(all(np.isfinite(numerical_quantile) == (will_converge)))

  def test_quantile_verify_break_when_converged(self):
    n_reruns = 1000
    time_max_iter_is_10 = (
        timeit.timeit(
            lambda: self._compute_quantile_specify_max_iter(10), number=n_reruns
        )
        / n_reruns
    )
    time_max_iter_is_20000 = (
        timeit.timeit(
            lambda: self._compute_quantile_specify_max_iter(1000),
            number=n_reruns,
        )
        / n_reruns
    )
    timing_ratio = time_max_iter_is_20000 / time_max_iter_is_10
    np.testing.assert_almost_equal(timing_ratio, 1, 0)  # Should be of order 1

  def _compute_quantile_specify_max_iter(self, max_iter):
    p_value, support, shift = (0.3, [0, np.inf], 0)
    k_vec = [1, 4, 3]
    l_vec = [3, 2, 1]
    weibull_inst = tfp.distributions.Weibull(k_vec, l_vec)
    acc = 1  # Low accuracy to ensure quick convergence
    statistics_utils.quantile_for_p_val(
        weibull_inst, p_value, support, shift, acc, max_iter
    )


class MovingWindowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.rng = np.random.RandomState(seed=1905)
    self.time_series_length = 10000
    self.local_std = 0.05
    self.sample_values = np.ones(self.time_series_length, dtype=np.float32)
    self.sample_values[: int(self.time_series_length / 2)] = 0
    self.sample_values += self.rng.normal(
        0, scale=self.local_std, size=self.time_series_length
    )
    self.sample_timestamps = np.arange(
        self.time_series_length, dtype=np.float32
    )
    self.sample_timestamps -= self.sample_timestamps[
        int(self.time_series_length / 2)
    ]
    self.sample_timestamps += self.rng.uniform(
        low=-0.01, high=0.01, size=self.time_series_length
    )

    self.estimate_times = (
        self.time_series_length * np.array([0.25, 0.5, 0.75])
        - self.time_series_length / 2
    )
    self.expected_avg = np.array([0, 0.5, 1])
    self.expected_std = np.array(
        [self.local_std, np.std(self.sample_values), self.local_std]
    )

  @parameterized.named_parameters(
      ('window_size_100', 100),
      ('window_size_1000', 1000),
  )
  def test_moving_avg_and_std_by_time_window(self, window_size):
    weight_on_past = 0.5
    measured_avg, measured_std = (
        statistics_utils.moving_avg_and_std_by_time_window(
            estimate_times=self.estimate_times,
            timestamps=self.sample_timestamps,
            values=self.sample_values,
            window_size=window_size,
            weight_on_past=weight_on_past,
        )
    )
    np.testing.assert_allclose(
        measured_avg, self.expected_avg, atol=self.local_std
    )
    np.testing.assert_allclose(
        measured_std, self.expected_std, atol=self.local_std
    )

    measured_avg, measured_std = (
        statistics_utils.moving_avg_and_std_by_sample_length(
            estimate_times=self.estimate_times,
            timestamps=self.sample_timestamps,
            values=self.sample_values,
            window_size=window_size,
            weight_on_past=weight_on_past,
        )
    )
    np.testing.assert_allclose(
        measured_avg, self.expected_avg, atol=self.local_std
    )
    np.testing.assert_allclose(
        measured_std, self.expected_std, atol=self.local_std
    )


if __name__ == '__main__':
  absltest.main()
