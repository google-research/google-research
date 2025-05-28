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

"""Tests for metrics."""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.forecasting import metrics
from eq_mag_prediction.utilities import catalog_analysis

_ABS_NUMERICAL_ERROR = 1e-10  # A threshold for an asbsolute numerical error
_REL_NUMERICAL_ERROR = 1e-4  # A threshold for a relative numerical error


class MetricsTest(parameterized.TestCase):

  def test_l_test(self):
    actual = np.array([[0, 0, 1], [2, 1, 0]])

    all_zeros_score = metrics.l_test(actual, np.zeros((2, 3)))
    perfect_score = metrics.l_test(actual, actual)
    mediocre_score = metrics.l_test(
        actual, np.array([[0.1, 0.1, 0.2], [0.5, 0.2, 0.1]])
    )
    better_mediocre_score = metrics.l_test(
        actual, np.array([[0.1, 0.1, 0.3], [0.5, 0.2, 0.1]])
    )
    self.assertLess(all_zeros_score, -50)
    self.assertLess(all_zeros_score, mediocre_score)
    self.assertLess(mediocre_score, better_mediocre_score)
    self.assertLess(better_mediocre_score, perfect_score)

  def test_l_test_tf(self):
    rs = np.random.RandomState(42)
    shape = (10, 5, 3)
    for _ in range(10):
      labels = rs.randint(0, 5, size=shape)
      forecasts = rs.random(shape)
      np.testing.assert_almost_equal(
          metrics.l_test(labels, forecasts),
          metrics.l_test_tf(labels, forecasts).numpy(),
      )

  def test_s_test(self):
    actual = np.array([[[0, 1, 1], [3, 1, 0]], [[0, 0, 0], [1, 0, 1]]])

    all_zeros_score = metrics.s_test(actual, np.zeros((2, 2, 3)))
    perfect_score = metrics.s_test(actual, actual)
    better_prediction = np.array(
        [[[1, 1, 1], [3, 2, 1]], [[0, 0, 0], [0, 0, 0]]], dtype='float64'
    )
    mediocre_score = metrics.s_test(actual, better_prediction)
    self.assertLess(all_zeros_score, mediocre_score)
    self.assertLess(mediocre_score, perfect_score)

    # Should ignore temporal changes, as long as the spatial distribution is
    # the same.
    same_marginal_distribution = better_prediction
    diff = np.array([[0, 0.1, -0.1], [0, 0, 0.2]])
    same_marginal_distribution[0, :, :] -= diff
    same_marginal_distribution[1, :, :] += diff
    np.testing.assert_allclose(
        mediocre_score,
        metrics.s_test(actual, same_marginal_distribution),
        rtol=0.01,
    )

  def test_s_test_tf(self):
    rs = np.random.RandomState(42)
    shape = (10, 5, 3)
    for _ in range(10):
      labels = rs.randint(0, 5, size=shape).astype('float64')
      forecasts = rs.random(shape)
      np.testing.assert_almost_equal(
          metrics.s_test(labels, forecasts),
          metrics.s_test_tf(labels, forecasts).numpy(),
      )

  @parameterized.named_parameters(
      ('under-prediction', 50, 100, (0, 1)),
      ('over-prediction', 150, 100, (1, 0)),
      ('exact prediction', 1000, 1000, (0.5, 0.5)),
      ('reasonable prediction', 88, 100, (0.1, 0.9)),
  )
  def test_n_test(self, forecasted, observed, expected_probabilities):
    prob_at_least, prob_at_most = metrics.n_test(forecasted, observed)

    self.assertAlmostEqual(prob_at_least, expected_probabilities[0], places=1)
    self.assertAlmostEqual(prob_at_most, expected_probabilities[1], places=1)

  def test_weighted_mse_soft_inverse_gr_tf(self):
    gr_func = lambda m, m_c, beta: beta * np.exp(beta * (m_c - m))

    # test with a vector all differences are =1
    test_1 = (np.arange(1, 6).astype(float), np.arange(2, 7).astype(float))
    output_tensor = metrics.weighted_mse_soft_inverse_gr_tf(
        test_1[0], test_1[1], weighting_exponent=0
    )
    self.assertAlmostEqual(output_tensor.numpy(), 1.0, places=4)

    output_tensor = metrics.weighted_mse_soft_inverse_gr_tf(
        test_1[0],
        test_1[1],
        weighting_exponent=-1,
        magnitude_threshold=2,
        gr_beta=1.5,
    )
    manual_result = (1 / gr_func(test_1[0], 2, 1.5)).sum() / test_1[0].size
    self.assertAlmostEqual(output_tensor.numpy(), manual_result, places=4)

    # test with a vector all differences are =3
    test_2 = (np.arange(1, 6).astype(float), np.arange(4, 9).astype(float))
    output_tensor = metrics.weighted_mse_soft_inverse_gr_tf(
        test_2[0], test_2[1], weighting_exponent=0
    )
    self.assertAlmostEqual(output_tensor.numpy(), 9.0, places=4)

    output_tensor = metrics.weighted_mse_soft_inverse_gr_tf(
        test_2[0],
        test_2[1],
        weighting_exponent=-1,
        magnitude_threshold=2,
        gr_beta=1.5,
    )
    manual_result = (9 / gr_func(test_2[0], 2, 1.5)).sum() / test_2[0].size
    self.assertAlmostEqual(output_tensor.numpy(), manual_result, places=4)

  def test_mean_pow_err_energy_tf(self):
    self.assertAlmostEqual(
        metrics.mean_pow_err_energy_tf(1.0, 1.0, 1.0).numpy(), 0.0, places=4
    )

    self.assertAlmostEqual(
        metrics.mean_pow_err_energy_tf(1.0, 1.0, 0.2).numpy(), 0.0, places=4
    )

    test_2 = (np.arange(1, 6).astype(float), np.arange(4, 9).astype(float))
    self.assertAlmostEqual(
        metrics.mean_pow_err_energy_tf(test_2[0], test_2[1], 0).numpy(),
        1,
        places=4,
    )

    manual_result = np.mean(
        np.abs(np.exp(test_2[0]) - np.exp(test_2[1])) ** 0.2
    )
    self.assertAlmostEqual(
        metrics.mean_pow_err_energy_tf(test_2[0], test_2[1], 0.2).numpy(),
        manual_result,
        places=4,
    )

    manual_result = np.mean(
        np.abs(np.exp(test_2[0]) - np.exp(test_2[1])) ** (-0.2)
    )
    self.assertAlmostEqual(
        metrics.mean_pow_err_energy_tf(test_2[0], test_2[1], -0.2).numpy(),
        manual_result,
        places=4,
    )


class WeibullTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('lambda=1e-2', 1e-2),
      ('lambda=1', 1),
      ('lambda=1e2', 1e2),
  )
  def test_weibull_likelihood(self, lambda_):
    x = np.linspace(0, 6 / lambda_, 100)
    # verify convergence to exponential distribution
    exponent_result = tfp.distributions.Exponential(
        rate=1 / lambda_, force_probs_to_zero_outside_support=True
    ).prob(x)
    weibull_result = metrics.weibull_likelihood(x, k=1, l=lambda_)
    np.testing.assert_allclose(
        exponent_result, weibull_result, rtol=0, atol=_ABS_NUMERICAL_ERROR
    )

  def test_shift_in_weibull_likelihood(self):
    lambda_ = 1
    x = np.linspace(-6, 6, 100)
    expected_diff = x[1] - x[0]

    for k in (0.5, 1, 1.5, 5):
      weibull_result = metrics.weibull_likelihood(x, k=k, l=lambda_, shift=0)
      max_location = x[np.nanargmax(weibull_result)]
      for shift in (-3, 3):
        # Does the mode shift as an indicator for the entire distribution shift?
        shifted_weibull_result = metrics.weibull_likelihood(
            x, k=k, l=lambda_, shift=shift
        )
        shifted_max_location = x[np.nanargmax(shifted_weibull_result)]
        assert_message = (
            f'With k={k}, l={lambda_}, shift={shift} '
            'Diff in max location is '
            f'{max_location + shift - shifted_max_location} '
            f'whereas expected diff is {expected_diff}'
            f'{max_location + shift=}, {shifted_max_location=}'
        )
        self.assertTrue(
            np.isclose(
                max_location + shift,
                shifted_max_location,
                rtol=0,
                atol=expected_diff,
            ),
            assert_message,
        )

  def test_weibull_loglikelihood_loss(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 100
    weibull_parameters = rnd_seed.rand(n, 2) * 10
    # scale domain to be proportional to typical length to avoid infs and zeros:
    x = rnd_seed.rand(n) * 5 / weibull_parameters[:, 1] + 1e-3
    weibull_minus_ll = metrics.weibull_loglikelihood_loss(x, weibull_parameters)
    result = np.exp(-weibull_minus_ll)
    expected = metrics.weibull_likelihood(
        x, k=weibull_parameters[:, 0], l=weibull_parameters[:, 1]
    )
    np.testing.assert_allclose(
        expected, result, rtol=0, atol=_ABS_NUMERICAL_ERROR
    )

  def test_conditional_loglikelihood_for_weibull(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 100
    weibull_parameters = rnd_seed.rand(n, 2) * 10
    # scale domain to be proportional to typical length to avoid infs and zeros:
    x = rnd_seed.rand(n) * 5 / weibull_parameters[:, 1] + 1e-3
    conditional_minus_loglikelihood = metrics.conditional_minus_loglikelihood(
        x, weibull_parameters
    )
    conditional_prob_from_ll = np.exp(-conditional_minus_loglikelihood)
    conditional_likelihood = metrics.conditional_likelihood(
        x, weibull_parameters
    )
    finite_values = np.isfinite(conditional_prob_from_ll)
    # verify that finite values are in identicla places, loglikelihood is
    # expected to return nans where likelihood is epxected to return +/-inf:
    np.testing.assert_array_equal(
        finite_values, np.isfinite(conditional_likelihood)
    )
    # verify identicality among finite values only
    np.testing.assert_allclose(
        conditional_prob_from_ll[finite_values],
        conditional_likelihood[finite_values],
        rtol=0,
        atol=_ABS_NUMERICAL_ERROR,
    )

  @parameterized.named_parameters(
      ('weibull_loglikelihood_loss', 'weibull_loglikelihood_loss'),
      ('conditional_likelihood', 'conditional_likelihood'),
      ('conditional_minus_loglikelihood', 'conditional_minus_loglikelihood'),
  )
  def test_tensor_returning_functions(self, function_name):
    numpy_function = getattr(metrics, function_name)
    tensor_function = getattr(metrics, function_name + '_tf')
    rnd_seed = np.random.RandomState(seed=1905)
    n = 100
    weibull_parameters = rnd_seed.rand(n, 2) * 10
    # scale domain to be proportional to typical length to avoid infs and zeros:
    x = rnd_seed.rand(n) * 5 / weibull_parameters[:, 1] + 1e-3
    tensor_result = tensor_function(x, weibull_parameters)
    # numpy versions resturn elements whereas tf's return mean:
    numpy_result = numpy_function(x, weibull_parameters).mean()
    np.testing.assert_allclose(
        numpy_result, tensor_result.numpy(), rtol=_REL_NUMERICAL_ERROR, atol=0
    )
    pass

  def test_weibull_mixture_instance_coincides_with_single_weibull(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 500
    weibull_parameters = _stretch_01_to_range(rnd_seed.rand(n, 2), (0.1, 6))
    single_weibull_instance = _return_weibull_random_variable(
        weibull_parameters
    )
    weibull_mixture_parameters = np.concatenate(
        (weibull_parameters, np.ones((n, 1))), axis=1
    )
    weibull_mixture_instance = metrics.weibull_mixture_instance(
        weibull_mixture_parameters
    )
    labels = _stretch_01_to_range(rnd_seed.rand(n), (1e-5, 5))
    np.testing.assert_allclose(
        single_weibull_instance.prob(labels),
        weibull_mixture_instance.prob(labels),
        rtol=0,
        atol=_ABS_NUMERICAL_ERROR,
    )


class GutenbergRictherTest(absltest.TestCase):

  def test_magnitude_threshold_gr_likelihood(self):
    x = np.linspace(-6, 6, 100)
    expected_diff = x[1] - x[0]
    for b in (0.5, 1, 1.5):
      for magnitude_thresh in (-3, 0, 3):
        # Does the zeros tail on the left shift?
        gr_result = metrics.gr_likelihood(
            x, gr_beta=b, magnitude_threshold=magnitude_thresh
        )
        self.assertTrue(
            np.all(gr_result[x < magnitude_thresh] == 0)
            & np.all(gr_result[x >= magnitude_thresh] != 0)
        )
        # Does the mode shift as an indicator for the entire distribution shift?
        max_location = x[np.argmax(gr_result)]
        self.assertTrue(
            np.isclose(
                max_location, magnitude_thresh, rtol=0, atol=expected_diff
            )
        )

  def test_normalization_of_gr_likelihood(self):
    for beta in (0.8, 1, 1.5):
      rnd = np.random.RandomState(seed=1905)
      random_locations = np.sort(rnd.rand(100, 2) * 10, axis=1)
      delta = np.diff(random_locations, axis=1)
      expected_quotient_in_range = np.exp(-beta * delta)
      gr_likelihood = metrics.gr_likelihood(
          random_locations, gr_beta=beta, magnitude_threshold=0
      )
      resulting_quotient = gr_likelihood[:, 1] / gr_likelihood[:, 0]
      np.testing.assert_allclose(
          expected_quotient_in_range.ravel(),
          resulting_quotient.ravel(),
          rtol=0,
          atol=_ABS_NUMERICAL_ERROR,
      )

  def test_gr_moving_window_likelihood(self):
    beta_early = 2.4
    beta_late = 0.4
    completeness_early = 2.5
    completeness_late = 1
    len_early = 90000
    len_late = 500

    rnd_state = np.random.RandomState(seed=1905)
    magnitudes_early = (
        rnd_state.exponential(1 / beta_early, len_early) + completeness_early
    )
    magnitudes_late = (
        rnd_state.exponential(1 / beta_late, len_late) + completeness_late
    )

    combined_magnitudes = np.concatenate((magnitudes_early, magnitudes_late))
    completeness_combined, beta_combined = (
        catalog_analysis.estimate_completeness_and_beta(combined_magnitudes)
    )

    earthquake_times = np.arange(len(combined_magnitudes))
    test_catalog = pd.DataFrame({
        'magnitude': combined_magnitudes,
        'time': earthquake_times,
        'longitude': np.random.rand(len(combined_magnitudes)),
        'latitude': np.random.rand(len(combined_magnitudes)),
    })

    lookbacks = np.array([len_early, len_late]).astype(int)
    magnitude_labels = np.array([3, 5, 6])
    for mag_label in magnitude_labels:
      data = np.array([mag_label])

      # Test at the timestamp between the two distributions
      resulting_likelihood_early = metrics.gr_moving_window_likelihood(
          data,
          np.array([len_early]),
          lookbacks,
          test_catalog,
      )
      expected_likelihood = np.array([
          metrics.gr_likelihood(
              data,
              gr_beta=beta_early,
              magnitude_threshold=completeness_early,
          ),
          metrics.gr_likelihood(
              data,
              gr_beta=beta_early,
              magnitude_threshold=completeness_early,
          ),
      ]).T

      np.testing.assert_allclose(
          expected_likelihood,
          resulting_likelihood_early,
          rtol=0,
          atol=1e-1,
      )

      # Test at the timestamp after both distributions
      resulting_likelihood_late = metrics.gr_moving_window_likelihood(
          data,
          np.array([len_early + len_late]),
          lookbacks,
          test_catalog,
      )
      expected_likelihood = np.array([
          metrics.gr_likelihood(
              data,
              gr_beta=beta_combined,
              magnitude_threshold=completeness_combined,
          ),
          metrics.gr_likelihood(
              data,
              gr_beta=beta_late,
              magnitude_threshold=completeness_late,
          ),
      ]).T

      np.testing.assert_allclose(
          expected_likelihood,
          resulting_likelihood_late,
          rtol=0,
          atol=1e-1,
      )

  def test_gr_moving_window_likelihood_constant_m_c(self):
    completeness_1 = 2.5
    completeness_2 = 1
    beta_1 = 2.4
    beta_2 = 0.4
    len_1 = 50000
    len_2 = 500

    rnd_state = np.random.RandomState(seed=1905)
    magnitudes_1 = rnd_state.exponential(1 / beta_1, len_1) + completeness_1
    magnitudes_2 = rnd_state.exponential(1 / beta_2, len_2) + completeness_2

    combined_magnitudes = np.concatenate((magnitudes_1, magnitudes_2))
    rnd_state.shuffle(combined_magnitudes)

    beta_combined = catalog_analysis.estimate_beta(
        combined_magnitudes, completeness_2
    )
    earthquake_times = np.arange(len(combined_magnitudes))
    test_catalog = pd.DataFrame({
        'magnitude': combined_magnitudes,
        'time': earthquake_times,
        'longitude': np.random.rand(len(combined_magnitudes)),
        'latitude': np.random.rand(len(combined_magnitudes)),
    })

    timestamps = np.array([len_1 + len_2] * 4)
    lookbacks = np.array([len_1, len_1 + len_2])
    # test for higher completeness
    test_magnitudes = np.linspace(completeness_1, completeness_1 + 3, 4)
    expected_likelihood = metrics.gr_likelihood(
        data=test_magnitudes,
        gr_beta=beta_1,
        magnitude_threshold=completeness_1,
    )
    expected_likelihood = np.repeat(
        expected_likelihood.reshape((-1, 1)), 2, axis=1
    )
    resulting_likelihoods = (
        metrics.gr_moving_window_likelihood_constant_completeness(
            data=test_magnitudes,
            timestamps=timestamps,
            lookbacks=lookbacks,
            catalog=test_catalog,
            completeness_magnitude=completeness_1,
        )
    )
    np.testing.assert_allclose(
        expected_likelihood, resulting_likelihoods, rtol=0, atol=1e-1
    )

    # test for lower completeness
    test_magnitudes = np.linspace(completeness_2, completeness_2 + 3, 4)
    expected_likelihood = metrics.gr_likelihood(
        data=test_magnitudes,
        gr_beta=beta_combined,
        magnitude_threshold=completeness_2,
    )
    expected_likelihood = np.repeat(
        expected_likelihood.reshape((-1, 1)), 2, axis=1
    )
    resulting_likelihoods = (
        metrics.gr_moving_window_likelihood_constant_completeness(
            data=test_magnitudes,
            timestamps=timestamps,
            lookbacks=lookbacks,
            catalog=test_catalog,
            completeness_magnitude=completeness_2,
        )
    )
    np.testing.assert_allclose(
        expected_likelihood, resulting_likelihoods, rtol=0, atol=1e-1
    )

  def test_vector_behavior(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 100
    gr_parameters = _stretch_01_to_range(rnd_seed.rand(n, 2), (0.5, 3))
    magnitudes = _stretch_01_to_range(rnd_seed.rand(n), (0.5, 9))
    likelihood_vector = metrics.gr_likelihood(
        magnitudes, gr_parameters[:, 0], gr_parameters[:, 1]
    )
    # pylint: disable=[g-complex-comprehension]
    likelihood_scaler = np.array(
        [
            metrics.gr_likelihood(
                magnitudes[i : (i + 1)],
                gr_parameters[i, 0],
                gr_parameters[i, 1],
            )
            for i in range(n)
        ]
    )
    # pylint: enable=[g-complex-comprehension]
    np.testing.assert_allclose(
        likelihood_vector.ravel(),
        likelihood_scaler.ravel(),
        rtol=_REL_NUMERICAL_ERROR,
        atol=_ABS_NUMERICAL_ERROR,
    )

  def test_conditioned_gr(self):
    beta = 2.0
    completeness_magnitude = 2.4
    magnitude_cutoff = 3.0
    # When evaluation is equal to cutoff should return the decay coefficient:
    estimation_magnitude = magnitude_cutoff
    result = metrics.gr_conditioned_likelihood(
        np.array([estimation_magnitude]),
        beta,
        completeness_magnitude,
        magnitude_cutoff,
    ).numpy()[0]
    np.testing.assert_allclose(
        result, beta, rtol=_REL_NUMERICAL_ERROR, atol=_ABS_NUMERICAL_ERROR
    )
    # self.assertEqual(result, beta)
    # When evaluation is larger than cutoff should return value samller than
    # decay coefficient:
    estimation_magnitude = magnitude_cutoff + 1
    result = metrics.gr_conditioned_likelihood(
        np.array([estimation_magnitude]),
        beta,
        completeness_magnitude,
        magnitude_cutoff,
    ).numpy()[0]
    self.assertLess(result, beta)
    # When evaluation is below cutoff of completeness should not return values:
    estimation_magnitudes = np.array(
        [completeness_magnitude - 0.5, magnitude_cutoff - 0.5]
    )
    result = metrics.gr_conditioned_likelihood(
        estimation_magnitudes,
        beta,
        completeness_magnitude,
        magnitude_cutoff,
    ).numpy()
    self.assertEqual(result.size, 0)
    # When cutoff is below completeness, the values between them are returned
    # with a score =0:
    result = metrics.gr_conditioned_likelihood(
        estimation_magnitudes,
        beta,
        magnitude_cutoff,
        completeness_magnitude,
    ).numpy()
    self.assertEqual(result.size, 1)
    np.testing.assert_allclose(
        result, 0, rtol=_REL_NUMERICAL_ERROR, atol=_ABS_NUMERICAL_ERROR
    )


class RandomVariableMetricsTest(absltest.TestCase):

  def test_result_is_finite(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 500
    weibull_parameters = _stretch_01_to_range(rnd_seed.rand(n, 2), (0.1, 6))
    x = np.array(list(range(n - 4)) + [np.inf, -np.inf, np.nan, 0])
    loss_instance = metrics.MinusLoglikelihoodLoss(
        _return_weibull_random_variable
    )
    returned_minus_ll = loss_instance.call(x, weibull_parameters)
    self.assertTrue(np.all(returned_minus_ll.numpy()))

  def test_loss_coincides_with_single_weibull(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 500
    weibull_parameters = _stretch_01_to_range(rnd_seed.rand(n, 2), (0.1, 6))
    # scale domain to be proportional to typical length to avoid infs and zeros:
    x = _stretch_01_to_range(rnd_seed.rand(n), (1e-5, 5))

    # Test loss
    ll_per_example = _return_weibull_random_variable(
        weibull_parameters
    ).log_prob(x)
    finite_logical = ll_per_example > metrics._LOG_EPSILON
    expected_minus_ll = -tf.math.reduce_mean(ll_per_example[finite_logical])
    conditional_instance = metrics.MinusLoglikelihoodLoss(
        _return_weibull_random_variable
    )
    returned_minus_ll = conditional_instance.call(
        x[finite_logical], weibull_parameters[finite_logical]
    )
    np.testing.assert_allclose(
        expected_minus_ll, returned_minus_ll, rtol=0, atol=_ABS_NUMERICAL_ERROR
    )

    # Test conditional probability loss
    mag_threshold = metrics._DEFAULT_MAG_THRESH
    # Calculate result per example to avoid infs:
    minus_ll_per_example = metrics.conditional_minus_loglikelihood(
        x, weibull_parameters, mag_threshold
    )
    finite_logical = np.isfinite(minus_ll_per_example)
    expected_conditional_minus_ll = metrics.conditional_minus_loglikelihood_tf(
        x[finite_logical], weibull_parameters[finite_logical], mag_threshold
    )
    conditional_instance = metrics.ConditionalMinusLoglikelihoodLoss(
        _return_weibull_random_variable, mag_threshold
    )
    returned_conditional_minus_ll = conditional_instance.call(
        x[finite_logical], weibull_parameters[finite_logical]
    )
    np.testing.assert_allclose(
        expected_conditional_minus_ll,
        returned_conditional_minus_ll,
        rtol=0,
        atol=_ABS_NUMERICAL_ERROR,
    )

  def test_gaussian_mixture_instance_coincides_with_single_gaussian(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 500
    gaussian_parameters = _stretch_01_to_range(rnd_seed.rand(n, 2), (-5, 5))
    gaussian_parameters[:, 1] = gaussian_parameters[:, 1] * 10
    single_gaussian_instance = _return_gaussian_random_variable(
        gaussian_parameters
    )
    gaussian_mixture_parameters = np.concatenate(
        (gaussian_parameters, np.ones((n, 1))), axis=1
    )
    gaussian_mixture_instance = metrics.gaussian_mixture_instance(
        gaussian_mixture_parameters
    )
    labels = _stretch_01_to_range(rnd_seed.rand(n), (-10, 10))
    np.testing.assert_allclose(
        single_gaussian_instance.prob(labels),
        gaussian_mixture_instance.prob(labels),
        rtol=0,
        atol=_ABS_NUMERICAL_ERROR,
    )


class MixtureKwargsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('validate_args_false', False),
      ('validate_args_true', True),
  )
  def test_mixture_kwargs(self, validate_args):
    mixture_parameters = np.array(
        [[1.0547e00, 8.9112e-01, 1.6968e01, 3.1244e00, 2.5064e00, 1.6695e-02]]
    )

    pdf_instance = metrics.kumaraswamy_mixture_instance(
        mixture_parameters, validate_args=validate_args
    )
    self.assertEqual(pdf_instance.validate_args, validate_args)


class ShiftStretchLossTest(absltest.TestCase):

  def test_shift_stretch_coincides_with_regular(self):
    rnd_seed = np.random.RandomState(seed=1905)
    n = 100
    kumar_parameters = rnd_seed.rand(n, 3) * 20
    mock_labels = _stretch_01_to_range(rnd_seed.rand(n), (1e-3, 1 - 1e-3))
    simple_loss_finction = metrics.MinusLoglikelihoodLoss(
        metrics.kumaraswamy_mixture_instance
    )
    stretch_shift_loss_function = (
        metrics.MinusLoglikelihoodConstShiftStretchLoss(
            metrics.kumaraswamy_mixture_instance
        )
    )
    stretched_shifted_loss = stretch_shift_loss_function.call(
        mock_labels, kumar_parameters
    )
    simple_loss = simple_loss_finction.call(mock_labels, kumar_parameters)
    self.assertAlmostEqual(stretched_shifted_loss, simple_loss, places=4)

    random_var_shift = 3
    random_var_stretch = 7
    stretch_shift_loss_function_modified = (
        metrics.MinusLoglikelihoodConstShiftStretchLoss(
            metrics.kumaraswamy_mixture_instance,
            random_var_shift,
            random_var_stretch,
        )
    )
    stretched_shifted_modified_loss = stretch_shift_loss_function_modified.call(
        mock_labels * random_var_stretch + random_var_shift, kumar_parameters
    )

    self.assertAlmostEqual(
        np.array(stretched_shifted_loss) + np.log(random_var_stretch),
        np.array(stretched_shifted_modified_loss),
        places=4,
    )


def _stretch_01_to_range(numbers, target_range):
  delta = np.max(target_range) - np.min(target_range)
  return (numbers * delta) + np.min(target_range)


def _return_weibull_random_variable(forecasts):
  ks = forecasts[:, 0]
  ls = forecasts[:, 1]
  return tfp.distributions.Weibull(concentration=ks, scale=ls)


def _return_gaussian_random_variable(forecasts):
  locs = forecasts[:, 0]
  scales = forecasts[:, 1]
  return tfp.distributions.Normal(loc=locs, scale=scales)


class HistogramBasedMtericsTest(absltest.TestCase):

  def test_histogram_likelihood(self):
    fake_data = np.arange(0.5, 9.5, 1)
    histogram_bins = np.arange(10)
    n_bins = len(histogram_bins) - 1
    expected_likelihood = 1 / n_bins

    # data in between bin edges
    labels = np.arange(3, 8)
    resulting_likelihood = metrics.histogram_likelihood(
        labels=labels,
        data_pool=fake_data,
        histogram_bins=histogram_bins,
    )
    np.testing.assert_allclose(expected_likelihood, resulting_likelihood)

    # data on edges to the right
    resulting_likelihood = metrics.histogram_likelihood(
        labels=labels + 0.5,
        data_pool=fake_data,
        histogram_bins=histogram_bins,
    )
    np.testing.assert_allclose(expected_likelihood, resulting_likelihood)

    # data on edges to the left
    resulting_likelihood = metrics.histogram_likelihood(
        labels=labels - 0.5,
        data_pool=fake_data,
        histogram_bins=histogram_bins,
    )
    np.testing.assert_allclose(expected_likelihood, resulting_likelihood)

    # data outside distribution
    resulting_likelihood = metrics.histogram_likelihood(
        labels=np.array([histogram_bins[0] - 0.5, histogram_bins[-1] + 0.5]),
        data_pool=fake_data,
        histogram_bins=histogram_bins,
    )
    np.testing.assert_allclose(0, resulting_likelihood)


class BinnedDistributionTest(absltest.TestCase):

  def test_gr_bins_sum_to_1(self):
    for bin_width in [0.1, 0.5, 1, 3]:
      bins = np.arange(-10, 100, bin_width)
      for beta in (0.5, 1, 1.5):
        for mc in (-3, 0, 3):
          data = bins[:-1] + (bins[1] - bins[0]) / 2
          probs = metrics.gr_probability_of_bin(data, bins, beta, mc)
          np.testing.assert_allclose(
              probs.sum(), 1, rtol=0, atol=_ABS_NUMERICAL_ERROR
          )

  def test_random_variable_bins_sum_to_1(self):
    test_random_variables_creators = [
        lambda a, b: tfp.distributions.Kumaraswamy(a, b),  # pylint: disable=[unnecessary-lambda]
        lambda a, b: tfp.distributions.TruncatedNormal(a, b, 0, 1),
    ]
    rnd_seed = np.random.RandomState(seed=1905)
    n = 10
    variable_parameters = (rnd_seed.rand(n, 2) * 3) + 1e-3
    for bin_width in np.logspace(-4, 0, 5):
      bins = np.arange(0, 1 + bin_width, bin_width)
      for creator in test_random_variables_creators:
        for params in variable_parameters:
          random_variable = creator(params[0], params[1])
          data = bins[:-1] + (bins[1] - bins[0]) / 2
          probs = metrics.random_variable_probability_of_bin(
              data, bins, random_variable
          )
          np.testing.assert_allclose(
              probs.sum(), 1, rtol=bin_width * 10, atol=bin_width * 10
          )

  def test_random_variable_coincides_with_gr(self):
    for bin_width in [0.1, 0.5, 1, 3]:
      bins = np.arange(0, 100, bin_width)
      for beta in (0.5, 1, 1.5):
        data = bins[:-1] + (bins[1] - bins[0]) / 2
        probs_gr = metrics.gr_probability_of_bin(data, bins, beta, 0)
        probs_random_var = metrics.random_variable_probability_of_bin(
            data,
            bins,
            tfp.distributions.Exponential(
                beta, force_probs_to_zero_outside_support=True
            ),
        )

        np.testing.assert_allclose(
            probs_gr, probs_random_var, rtol=bin_width * 10, atol=bin_width * 10
        )


if __name__ == '__main__':
  absltest.main()
