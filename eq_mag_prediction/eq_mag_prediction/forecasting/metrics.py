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

"""Common metrics for earthquake forecasting models.

Source for most metrics:
http://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/zechar.pdf_2063069299.pdf
"""

import numbers
from typing import Callable, Iterable, Optional, Sequence, Type, Union

import gin
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
import sklearn.metrics
import tensorflow as tf
import tensorflow_probability as tfp

from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import geometry


_EPSILON = 1e-10
_BIGGER_EPSILON = 1e-3  # Useful when the small value is involved in division.
_LOG_EPSILON = np.log(_EPSILON)
_DEFAULT_BETA = 1.5  # Common value of beta
_DEFAULT_MAG_THRESH = 1.8  # Common completeness mag in large and modern catalog
DEFAULT_DISTRIBUTION_KWARGS = {'validate_args': True, 'allow_nan_stats': False}
DistributionType = Type[tfp.distributions.Distribution]


def precision_at_recall(
    labels, forecasts, delta
):
  """Returns precision-at-recall for thresholds in [0, 1], with delta jumps."""
  thresholds = np.arange(0, 1, delta)
  precision, recall, _ = sklearn.metrics.precision_recall_curve(
      labels.ravel(), forecasts.ravel()
  )
  return np.array([np.max(precision[np.where(recall > t)]) for t in thresholds])


@gin.configurable(allowlist=['delta'])
def average_precision_at_recall(
    labels, forecasts, delta = 0.001
):
  """Returns the area under the precision-at-recall curve."""
  thresholds = np.arange(0, 1, delta)
  precision_at_recall_array = precision_at_recall(labels, forecasts, delta)
  return sklearn.metrics.auc(thresholds, precision_at_recall_array)


def _aftershock_days_mask(
    labels, days_after_earthquake
):
  """Returns a mask of which days are potential 'aftershock' days."""
  result = np.zeros(labels.shape[0], dtype='bool')
  for i in range(len(labels) - 1):
    if np.any(labels[i]):
      result[i + 1 : i + days_after_earthquake + 1] = True
  return result


def precision_at_recall_for_aftershocks(
    *,
    labels,
    days_after_earthquake,
    forecasts,
    delta,
):
  """Returns precision-at-recall only for times that are in aftershock times."""
  aftershocks_mask = _aftershock_days_mask(labels, days_after_earthquake)
  sub_labels = labels[np.where(aftershocks_mask)]
  sub_forecasts = forecasts[np.where(aftershocks_mask)]
  return precision_at_recall(sub_labels, sub_forecasts, delta)


def average_precision_at_recall_for_aftershocks(
    *,
    labels,
    days_after_earthquake,
    forecasts,
    delta,
):
  """Returns the area under the precision-at-recall curve for aftershocks."""
  thresholds = np.arange(0, 1, delta)
  precision_at_recall_array = precision_at_recall_for_aftershocks(
      labels=labels,
      days_after_earthquake=days_after_earthquake,
      forecasts=forecasts,
      delta=delta,
  )
  return sklearn.metrics.auc(thresholds, precision_at_recall_array)


def _mainshock_days_mask(
    labels, days_without_earthquakes
):
  """Returns a mask of which days are potential 'mainshock' days."""
  result = np.zeros(labels.shape[0], dtype='bool')
  for i in range(days_without_earthquakes, len(labels)):
    if not np.any(labels[i - days_without_earthquakes : i]):
      result[i] = True
  return result


def precision_at_recall_for_mainshocks(
    *,
    labels,
    days_without_earthquakes,
    forecasts,
    delta,
):
  """Returns precision-at-recall only for times that are in mainshock times."""
  mainshocks_mask = _mainshock_days_mask(labels, days_without_earthquakes)
  sub_labels = labels[np.where(mainshocks_mask)]
  sub_forecasts = forecasts[np.where(mainshocks_mask)]
  return precision_at_recall(sub_labels, sub_forecasts, delta)


def average_precision_at_recall_for_mainshocks(
    *,
    labels,
    days_without_earthquakes,
    forecasts,
    delta,
):
  """Returns the area under the precision-at-recall curve for mainshocks."""
  thresholds = np.arange(0, 1, delta)
  precision_at_recall_array = precision_at_recall_for_mainshocks(
      labels=labels,
      days_without_earthquakes=days_without_earthquakes,
      forecasts=forecasts,
      delta=delta,
  )
  return sklearn.metrics.auc(thresholds, precision_at_recall_array)


def l_test(labels, forecasts):
  """The L-test score. This is Poisson loss, up to a linear transformation."""
  fixed_forecasts = np.maximum(forecasts, _EPSILON)
  return np.sum(labels * np.log(fixed_forecasts) - fixed_forecasts)


def l_test_tf(labels, forecasts):
  """A tensorflow implementation of the L-test, can be used as a metric."""
  fixed_forecasts = tf.maximum(forecasts, _EPSILON)
  return tf.reduce_sum(labels * tf.math.log(fixed_forecasts) - fixed_forecasts)


def s_test(labels, forecasts):
  """The S-test score. Scores the marginal spatial distribution of forecasts."""
  fixed_forecasts = np.maximum(forecasts, _EPSILON)
  n_forecast = np.sum(fixed_forecasts)
  n_observed = np.sum(labels)
  if n_observed == 0 or n_forecast == 0:
    spatial_forecasts = np.sum(fixed_forecasts, axis=0)
  else:
    spatial_forecasts = (n_observed / n_forecast) * np.sum(
        fixed_forecasts, axis=0
    )
  spatial_labels = np.sum(labels, axis=0)
  return np.sum(spatial_labels * np.log(spatial_forecasts) - spatial_forecasts)


def s_test_tf(labels, forecasts):
  """A tensorflow implementation of the S-test, can be used as a metric."""
  fixed_forecasts = tf.maximum(forecasts, _EPSILON)
  n_forecast = tf.reduce_sum(fixed_forecasts)
  n_observed = tf.reduce_sum(labels)
  spatial_forecasts = tf.reduce_sum(fixed_forecasts, axis=0)
  if n_observed != 0 and n_forecast != 0:
    spatial_forecasts = (n_observed / n_forecast) * spatial_forecasts
  spatial_labels = tf.reduce_sum(labels, axis=0)

  return tf.reduce_sum(
      spatial_labels * tf.math.log(spatial_forecasts) - spatial_forecasts
  )


def n_test(num_forecasted, num_observed):
  """The N-test score.

  Scores the marginal distribution of the number of earthquakes. Returns two
  probabilities, both should be non-trivial (say, above 5%).

  Args:
    num_forecasted: The number of forecasted earthquakes. Can be a float.
    num_observed: The actual number of observed earthquakes.

  Returns:
    The probability of observing at least N earthquakes, and the probability of
    observing at most N earthquakes (given the forecast), where N is the actual
    number of observed earthquakes.
  """
  return (
      1 - scipy.stats.poisson.cdf(num_observed - 1, num_forecasted),
      scipy.stats.poisson.cdf(num_observed, num_forecasted),
  )


def weighted_mse_soft_inverse_gr_tf(
    labels,
    forecasts,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
    weighting_exponent = -0.5,
):
  """A metric computing weighted mse.

  Weights are the inverse values of a given Gutenberg- Richter distribution,
  raised to a power:
  score = sum( w(labels) * (labels - forecasts)**2 )/N
  where w(labels) = (GR(m))**weighting_exponent.

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts.
    gr_beta: gr_beta of the GR distribution (see Aki (1965)).
    magnitude_threshold: Completeness magnitude of the GR distribution.
    weighting_exponent: Power raising the weights to soften the penalty (see
      description). Will typically be <0.

  Returns:
    A scalar tf.Tensor containing measure's score.
  """
  weight_factor = tf.math.pow(
      gr_beta * tf.math.exp(gr_beta * (magnitude_threshold - labels)),
      weighting_exponent,
  )
  mse = tf.square(labels - forecasts)
  weighted_values = tf.math.multiply(weight_factor, mse)
  return tf.math.reduce_mean(weighted_values)


def weighted_mse_inverse_gr_tf(
    labels,
    forecasts,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """A non-softened version of the weighted_mse_soft_inverse_gr_tf metric."""

  return weighted_mse_soft_inverse_gr_tf(
      labels, forecasts, gr_beta, magnitude_threshold, weighting_exponent=1
  )


def mean_pow_err_energy_tf(
    labels,
    forecasts,
    p_norm,
):
  """Mean abs error of exp(value) to some power p_norm.

  The calculation done here goes by:
  score = sum( abs(exp(labels) - exp(forecasts))**p_norm )/N

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts.
    p_norm: The power by which to raise the error.

  Returns:
    A scalar tf.Tensor containing measure's score.
  """
  mean_powered_error = tf.pow(
      tf.math.abs(tf.math.exp(labels) - tf.math.exp(forecasts)), p_norm
  )
  return tf.math.reduce_mean(mean_powered_error)


def me_energy_tf(
    labels,
    forecasts,
):
  """Mean absolute error calculated using mean_pow_err_energy with p_norm=1."""
  return mean_pow_err_energy_tf(labels, forecasts, p_norm=1)


def mse_energy_tf(
    labels,
    forecasts,
):
  """Mean squared error calculated using mean_pow_err_energy with p_norm=2."""
  return mean_pow_err_energy_tf(labels, forecasts, p_norm=2)


def histogram_likelihood(
    labels,
    data_pool,
    histogram_bins = 10,
):
  """Estimates label's likelihood from the distribution of the binned data."""
  hist_values, bins = np.histogram(data_pool, bins=histogram_bins, density=True)
  bin_idx = np.minimum(
      np.digitize(labels, bins, right=True), len(hist_values) - 1
  )
  likelihood = hist_values[bin_idx]
  likelihood[labels < data_pool.min()] = 0
  likelihood[labels > data_pool.max()] = 0
  return likelihood


def gr_likelihood(
    data,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """Probability of getting given data from a Gutenberg-Richter distribution."""
  gr_likelihood_result = gr_beta * np.exp(
      -gr_beta * (data - magnitude_threshold)
  )
  # gr_likelihood_result[data < magnitude_threshold] = 0
  np.put_along_axis(
      gr_likelihood_result,
      np.expand_dims(
          np.where(data < magnitude_threshold)[0], (tuple(range(1, data.ndim)))
      ),
      0,
      0,
  )
  return gr_likelihood_result


def gr_moving_window_likelihood(
    data,
    timestamps,
    lookbacks,
    catalog,
):
  """Probability of getting the data from a distribution fitted on near past.

  Args:
    data: Magnitude labels to predict.
    timestamps: Corresponding timestamps of the data.
    lookbacks: A list window sizes to define the data range used to calculate
      the instanteneous Gutenberg-Richter (GR) distribution.
    catalog: The catalog from which to use for the GR fit.

  Returns:
    An [m,n] shaped ndarray containing the likelihood to draw each of the n data
    points for every m lookback (time window).
  """
  data = data.ravel()[:, None]
  # Create examples
  catalog_center = geometry.Point(
      catalog.longitude.mean(), catalog.latitude.mean()
  )
  examples = {t: [[catalog_center]] for t in timestamps}
  m_c_beta_feature = catalog_analysis.compute_property_in_time_and_space(
      catalog,
      catalog_analysis.completeness_and_beta_in_square,
      examples,
      grid_side_deg=180,
      lookback_seconds=lookbacks,
      magnitudes=[-100],
  )
  # Select only axes of timestamps, lookbacks and m_c/beta:
  m_c, beta = np.moveaxis(m_c_beta_feature[:, 0, 0, :, 0, :].T, 2, 1)
  return gr_likelihood(data, gr_beta=beta, magnitude_threshold=m_c)


def gr_moving_window_likelihood_constant_completeness(
    data,
    timestamps,
    lookbacks,
    catalog,
    completeness_magnitude = None,
):
  """Returns the likelihood of data by fitting Gutenberg-Richter to the near past.

  Will compute the Gutenberg Richter of the past (per lookback, per timestamp)
  by finding the best exponential decay with a given and constant completeness
  magnitude.

  Args:
    data: Magnitude labels to predict.
    timestamps: Corresponding timestamps of the data.
    lookbacks: A list window sizes to define the data range used to calculate
      the instanteneous Gutenberg-Richter (GR) distribution.
    catalog: The catalog from which to use for the GR fit.
    completeness_magnitude: The completeness magnitude to be used for the
      computation. If None is given, will use the computed completeness of the
      entire catalog.

  Returns:
    An [m,n] shaped ndarray containing the likelihood to draw each of the n data
    points for every m lookback (time window).
  """
  data = data.ravel()[:, None]
  # Create examples
  catalog_center = geometry.Point(
      catalog.longitude.mean(), catalog.latitude.mean()
  )
  examples = {t: [[catalog_center]] for t in timestamps}

  if completeness_magnitude is None:
    completeness_magnitude = catalog_analysis.estimate_completeness(
        catalog.magnitude.values
    )

  def beta_of_constant_completeness_in_square(
      catalog, time_slice, centers, grid_side_degrees
  ):
    return catalog_analysis.beta_of_constant_completeness_in_square(
        catalog, time_slice, centers, grid_side_degrees, completeness_magnitude
    )

  beta_feature = catalog_analysis.compute_property_in_time_and_space(
      catalog,
      beta_of_constant_completeness_in_square,
      examples,
      grid_side_deg=180,
      lookback_seconds=lookbacks,
      magnitudes=[-100],
  )
  beta = beta_feature[:, 0, 0, :, 0, 0]
  return gr_likelihood(
      data, gr_beta=beta, magnitude_threshold=completeness_magnitude
  )


def gr_likelihood_above_cutoff(
    data,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
    magnitude_cutoff = _DEFAULT_MAG_THRESH,
):
  """Returns likelihoods of data points above magnitude_cutoff only."""
  return gr_likelihood(data, gr_beta, magnitude_threshold)[
      data >= magnitude_cutoff
  ]


def gr_conditioned_likelihood(
    data,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
    magnitude_cutoff = _DEFAULT_MAG_THRESH,
):
  """Returns the likelihood of data to be observed from a conditioned exponent.

  gr_conditioned_likelihood will compute the probability that data is drawn from
  A Gutenberg Richter (GR) distribution conditioned on the data being above a
  specific magntiude magnitude_cutoff.

  Args:
    data: Magnitude labels to predict.
    gr_beta: The beta value of the intended GR distribution.
    magnitude_threshold: The completeness magnitude of the GR distribution.
    magnitude_cutoff: The magnitude above which the result will be conditioned.

  Returns:
    An array of the likelihoods of all data points conditioned on
    data>=magnitude_cutoff. Array size will be (data>=magnitude_cutoff).sum()
  """

  def num_to_vec(num):
    if isinstance(num, numbers.Number):
      return np.full_like(data, num)
    return num

  gr_random_variable = tfp.distributions.Exponential(
      num_to_vec(gr_beta), force_probs_to_zero_outside_support=True
  )
  exponent_survival = gr_random_variable.survival_function(
      num_to_vec(magnitude_cutoff - magnitude_threshold)
  )
  return (
      gr_likelihood_above_cutoff(
          data, gr_beta, magnitude_threshold, magnitude_cutoff
      )
      / exponent_survival[data >= magnitude_cutoff]
  )


def gr_loglikelihood_loss(
    labels,
    forecasts,
):
  """Returns a simplified expression of -log(f(x; m_thresh, beta)).

  Here f is the Gutenberg-Ricghter (GR) distribution, x are the input points and
  m_thresh and beta are the two parameters of the GR function.

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter m_thresh. Column 1 will be considered the parameter beta.

  Returns:
    A np.ndarray containing measure's score.
  """
  m_thresh = forecasts[:, 0]
  beta = forecasts[:, 1]
  return (
      -tfp.distributions.Exponential(
          beta, force_probs_to_zero_outside_support=1
      )
      .log_prob(labels - m_thresh)
      .numpy()
  )


def gr_loglikelihood_loss_tf(
    labels,
    forecasts,
):
  """Returns a simplified expression of -log(f(x; m_thresh, beta)).

  Here f is the Gutenberg-Ricghter (GR) distribution, x are the input points and
  m_thresh and beta are the two parameters of the GR function.

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts. A (N,2) tf tensor. Column 0 will be
      considered the parameter m_thresh. Column 1 will be considered the
      parameter beta.

  Returns:
    A tf.Tensor containing measure's score.
  """
  m_thresh = forecasts[:, 0]
  beta = forecasts[:, 1]
  return -tfp.distributions.Exponential(
      beta, force_probs_to_zero_outside_support=1
  ).log_prob(labels - m_thresh)


def gr_mean(
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """Mean of Gutenberg-Richter distribution given beta, magnitude threshold."""
  return magnitude_threshold + 1 / gr_beta


def gr_probability_of_bin(
    data,
    bins,
    gr_beta = _DEFAULT_BETA,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """Probability of a getting the relevant bin from a GR distribution.

  Given data, a bin partition of the magnitude scale and parameters of the
  Gutenberg-Ricter distribution, the function will return the probability of the
  bin where each data point is located in.
  E.g. if a data point is m=3.5, located by the bin partiotn in the bin [3,4),
  the returned value will be the integral between 3 to 4 of the
  Gutenberg-Richter distribution given by the input parameters.

  Args:
    data: The data indicating on the bins to be computed for probability.
    bins: A vector indicating bin esges partiotioning the magnitude scale.
    gr_beta: gr_beta of the GR distribution (see Aki (1965)).
    magnitude_threshold: Completeness magnitude of the GR distribution.

  Returns:
    An np.ndarray same size data, containing the probabilities of the relevant
    bins.
  """
  trimmed_data = np.maximum(bins[0], np.minimum(data, bins[-2]))
  small_edges, large_edges = _data_to_bin_edges(trimmed_data, bins)
  return _bin_probability(
      gr_beta, magnitude_threshold, small_edges, large_edges
  )


def _data_to_bin_edges(data, bins):
  bin_idxs = np.digitize(data, bins)
  return (np.take(bins, bin_idxs - 1), np.take(bins, bin_idxs))


def _bin_probability(gr_beta, magnitude_threshold, edge1, edge2):
  # This is an analytical simplification of the difference between two CDFs at
  # edge1 and edge2.
  probs = np.exp(gr_beta * magnitude_threshold) * (
      np.exp(-gr_beta * np.maximum(edge1, magnitude_threshold))
      - np.exp(-gr_beta * edge2)
  )
  if isinstance(probs, Iterable):
    probs[edge2 <= magnitude_threshold] = 0
  return probs


def random_variable_probability_of_bin(
    data,
    bins,
    random_variable,
    force_nans_to_zeros = True,
):
  """Probability of a getting the relevant bin from a given random variable."""
  trimmed_data = np.maximum(bins[0], np.minimum(data, bins[-2]))
  small_edges, large_edges = _data_to_bin_edges(trimmed_data, bins)
  large_cdf = random_variable.cdf(large_edges)
  small_cdf = random_variable.cdf(small_edges)
  if force_nans_to_zeros:
    large_cdf = np.where(np.isnan(large_cdf), 0, large_cdf)
    small_cdf = np.where(np.isnan(small_cdf), 0, small_cdf)
  return large_cdf - small_cdf


def weibull_likelihood(
    data,
    k,
    l,
    shift = 0,
):
  """Probability of getting given data from a Weibull distribution.

  Args:
    data: The points to which the likelihood should be computed.
    k: The k aprameter of the Weibull function
    l: The lambda aprameter of the Weibull function
    shift: A shift to the Weibull domain. Will perform the transformation data
      -> data-shift.

  Returns:
    An np.ndarray same size data, containing the Weibull likelihood.
  """

  val = data - shift
  probs = tfp.distributions.Weibull(k, l).prob(val)
  return tf.where(val >= 0, probs, 0.0).numpy()


def weibull_loglikelihood_loss(
    labels,
    forecasts,
    shift = 0,
):
  """numpy version for the function weibull_loglikelihood_loss_tf.

  Returns a simplified expression of -log(f(x; k, l)) where f is the Weibull
  function, x are the input points and k and l are the two parameters of the
  Weibull function.
  Weibull's parameters are forced to the range [_EPSILON, inf)

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter k. Column 1 will be considered the parameter l.
    shift: Shift the domain of the Weibull function from (0,inf) to (shift,
      inf).

  Returns:
    A np.ndarray containing measure's score.
  """

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  k[k <= _EPSILON] = _EPSILON
  l[l <= _EPSILON] = _EPSILON
  return -tfp.distributions.Weibull(k, l).log_prob(labels - shift).numpy()


def return_shifted_weibull_loglikelihood_loss(
    shift = 0,
):
  """Return a Weibull LL function to be used as a loss, given shift."""

  def loss_function(labels, forecasts):
    return weibull_loglikelihood_loss(labels, forecasts, shift)

  return loss_function


def weibull_loglikelihood_loss_tf(
    labels,
    forecasts,
    shift = 0,
):
  """Weibull log likelihood loss for trainig a model with Weibull pdf as output.

  Returns a simplified expression of -log(f(x; k, l)) where f is the Weibull
  function, x are the input points and k and l are the two parameters of the
  Weibull function. Mean is taken over x.
  Weibull's parameters are forced to the range [_EPSILON, inf)

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter k. Column 1 will be considered the parameter l.
    shift: Shift the domain of the Weibull function from (0,inf) to (shift, inf)

  Returns:
    A tf.Tensor containing measure's score.
  """
  k = tf.where((forecasts[:, 0] < _EPSILON), _EPSILON, forecasts[:, 0])
  l = tf.where((forecasts[:, 1] < _EPSILON), _EPSILON, forecasts[:, 1])
  shifted_truncated_labels = labels - shift
  minus_loglikelihood = -tf.math.reduce_mean(
      tfp.distributions.Weibull(k, l).log_prob(shifted_truncated_labels)
  )
  return minus_loglikelihood


def return_shifted_weibull_loglikelihood_loss_tf(
    shift = 0,
):
  """Return a Weibull LL function to be used as a loss, given shift."""

  def loss_function(labels, forecasts):
    return weibull_loglikelihood_loss_tf(labels, forecasts, shift)

  return loss_function


def conditional_likelihood(
    labels,
    forecasts,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """The probability of receiving a value above the magnitude threshold.

  We seek for the probability of returning a magnitude given
  the labels are above the magnitude threshold m_c: P(m|m>mc).
  Following Bayes' rule: P(m|m>mc)~P(m)/P(m>mc) (true up to a constant), the RHS
  is implemented with a Weibull distribution as P.

  Args:
    labels: The labels used for constructing the model.
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter k. Column 1 will be considered the parameter l.
    magnitude_threshold: Completeness magnitude of the GR distribution.

  Returns:
    A np.ndarray containing measure's score.
  """

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  likelihood = weibull_likelihood(labels, k, l)
  weibull_cdf = (
      1 - tfp.distributions.Weibull(k, l).cdf(magnitude_threshold).numpy()
  )
  return likelihood / weibull_cdf


def conditional_likelihood_tf(
    labels,
    forecasts,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """Same as conditional_likelihood, returning mean over x as a tf.Tensor."""

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  likelihood = tfp.distributions.Weibull(k, l).prob(labels)
  weibull_cdf = 1 - tfp.distributions.Weibull(k, l).cdf(magnitude_threshold)
  return tf.math.reduce_mean(likelihood / weibull_cdf)


def conditional_minus_loglikelihood(
    labels,
    forecasts,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """-loglikelihood of conditional_likelihood."""

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  log_likelihood = tfp.distributions.Weibull(k, l).log_prob(labels).numpy()
  weibull_log_survivlal = (
      tfp.distributions.Weibull(k, l)
      .log_survival_function(magnitude_threshold)
      .numpy()
  )
  return weibull_log_survivlal - log_likelihood


def conditional_minus_loglikelihood_tf(
    labels,
    forecasts,
    magnitude_threshold = _DEFAULT_MAG_THRESH,
):
  """Same as conditional_minus_loglikelihood, return the mean as a tf.Tensor."""

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  log_likelihood = tfp.distributions.Weibull(k, l).log_prob(labels)
  weibull_log_survival = tfp.distributions.Weibull(k, l).log_survival_function(
      magnitude_threshold
  )
  return tf.math.reduce_mean(weibull_log_survival - log_likelihood)


def weibull_mean(
    forecasts, shifts = 0
):
  """Mean of the weibull distribition.

  Args:
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter k. Column 1 will be considered the parameter l.
    shifts: A vector or (scalar if constant) of shift of the Weibull domain.

  Returns:
    A np.ndarray of length N containing the Weibull mean for each pair of
      parameters (k,l).
  """

  k = forecasts[:, 0]
  l = forecasts[:, 1]
  # Implement the formula for the mean:
  # https://en.wikipedia.org/wiki/Weibull_distribution
  return l * scipy.special.gamma(1 + 1 / k) + shifts


def weibull_mode(
    forecasts, shifts = 0
):
  """Mode of the weibull distribition.

  Args:
    forecasts: The model's forecasts. A (N,2) array. Column 0 will be considered
      the parameter k. Column 1 will be considered the parameter l.
    shifts: A vector or (scalar if constant) of shift of the Weibull domain.

  Returns:
    A np.ndarray of length N containing the Weibull mode for each pair of
      parameters (k,l).
  """
  k = forecasts[:, 0]
  l = forecasts[:, 1]
  # Implement the formula for the mode:
  # https://en.wikipedia.org/wiki/Weibull_distribution
  large_k = k > 1

  mode = np.zeros_like(k)  # for k<=1 the mode=0 for the unshifted distribution
  mode[large_k] = l[large_k] * ((k[large_k] - 1) / k[large_k]) ** (
      1 / k[large_k]
  )
  return mode + shifts


@gin.configurable
def weibull_mixture_instance(
    forecasts
):
  """Creates a distribution instance of a weibull mixture.

  Args:
    forecasts: The model's forecasts. A (M,3*n) array. n is be the number of
      weibulls in the mixture. 1st n columns are the k's parameter, next n
      columns are l parameter, last n columns are the weights between weibulls.
      M is the number of random variables in the probability instance.

  Returns:
    A probability instance of type tfp.distributions.MixtureSameFamily with
    batch_size of M.
  """
  # Expect forecasts to have a 2nd dimension 3*n where n is the number of
  # weibulls in the mixture:
  assert (
      forecasts.shape[1] % 3
  ) == 0, 'forecasts 2nd dimension does not have a size of 3*n'

  n = int(forecasts.shape[1] / 3)
  # The first n columns are the k parameters
  ks = forecasts[:, :n]
  # The second n columns are the l parameters
  ls = forecasts[:, n : (2 * n)]
  # The last n columns are the weights, phi, for the Weibulls in the mixture
  phis_not_norm = forecasts[:, (-n):]
  phis_sum = tf.reshape(tf.reduce_sum(phis_not_norm, axis=1), (-1, 1))
  phis = phis_not_norm / phis_sum

  mixture_instance = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(probs=phis),
      components_distribution=tfp.distributions.Weibull(
          concentration=ks, scale=ls, validate_args=True
      ),
  )

  return mixture_instance


def gaussian_mixture_instance(
    forecasts
):
  """Creates a distribution instance of a Gaussian mixture.

  Args:
    forecasts: The model's forecasts. A (M,3*n) array. n is the number of
      gaussians in the mixture. 1st n columns are the location parameters, next
      n columns are scale parameters, last n columns are the weights between
      gaussians. M is the number of random variables in the probability instance
      (batch size).

  Returns:
    A probability instance of type tfp.distributions.MixtureSameFamily with
    batch_size of M.
  """
  # Expect forecasts to have a 2nd dimension 3*n where n is the number of
  # Gaussians in the mixture:
  assert (
      forecasts.shape[1] % 3
  ) == 0, 'forecasts 2nd dimension does not have a size of 3*n'

  forecasts_tensor = tf.convert_to_tensor(forecasts)
  n = int(forecasts_tensor.shape[1] / 3)
  # The first n columns are the k parameters
  locs = forecasts_tensor[:, :n]
  # The second n columns are the l parameters
  scales = forecasts_tensor[:, n : 2 * n]
  # The last n columns are the weights, phi, for the Gaussians in the mixture
  phis_not_norm = forecasts_tensor[:, -n:]
  phis_sum = tf.reshape(tf.reduce_sum(phis_not_norm, axis=1), (-1, 1))
  phis = phis_not_norm / phis_sum

  mixture_instance = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(probs=phis),
      components_distribution=tfp.distributions.Normal(
          loc=locs,
          scale=scales,
          # validate_args=True
      ),
  )

  return mixture_instance


def kumaraswamy_mixture_instance(
    forecasts,
    **distribution_kwargs,
):
  """Creates a distribution instance of a Kumaraswamy mixture.

  Args:
    forecasts: The model's forecasts. A (m,3*n) array. n is the number of
      Kumaraswamy random variables in the mixture. 1st n columns are the 'a'
      parameters, next n columns are the 'b' parameters, last n columns are the
      weights between distributions. M is the batch size. More about the
      Kumaraswamy distribution:
      https://en.wikipedia.org/wiki/Kumaraswamy_distribution
      https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Kumaraswamy
    **distribution_kwargs: kwargs for tfp.distributions.Kumaraswamy and
      tfp.distributions.MixtureSameFamily. Defaults to
      DEFAULT_DISTRIBUTION_KWARGS.

  Returns:
    A probability instance of type tfp.distributions.MixtureSameFamily with
    batch_size of M.
  """
  # Expect forecasts to have a 2nd dimension 3*n where n is the number of
  # Kumaraswamy in the mixture:
  assert (
      forecasts.shape[1] % 3
  ) == 0, 'forecasts 2nd dimension does not have a size of 3*n'
  distribution_kwargs = {**DEFAULT_DISTRIBUTION_KWARGS, **distribution_kwargs}

  forecasts_tensor = tf.convert_to_tensor(forecasts)
  n = forecasts_tensor.shape[1] // 3
  # The first n columns are the a parameters
  a = forecasts_tensor[:, :n]
  # The second n columns are the b parameters
  b = forecasts_tensor[:, n : 2 * n]
  # The last n columns are the weights, phi, for the Kumaraswamy in the mixture
  phis_not_norm = forecasts_tensor[:, -n:]
  phis_sum = tf.reshape(tf.reduce_sum(phis_not_norm, axis=1), (-1, 1))
  phis = phis_not_norm / phis_sum

  mixture_instance = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(probs=phis),
      components_distribution=tfp.distributions.Kumaraswamy(
          a,
          b,
          **distribution_kwargs,
      ),
      **distribution_kwargs,
  )

  return mixture_instance


@gin.configurable
def gaussian2d_mixture_instance(
    forecasts
):
  """Creates a random varialbe that is a mixture of 2d Gaussians."""
  assert (
      forecasts.shape[1] % 7
  ) == 0, 'forecasts 2nd dimension does not have a size of 7*n'

  forecasts_tensor = tf.convert_to_tensor(forecasts)
  loc_x = forecasts_tensor[:, ::7]
  loc_y = forecasts_tensor[:, 1::7]
  # Constructing the covariance matrix from a Cholesky decomposition. It
  # requires 2 non-negative and 1 possibly negative value.
  a = forecasts_tensor[:, 2::7]
  b = forecasts_tensor[:, 3::7] - forecasts_tensor[:, 4::7]
  c = forecasts_tensor[:, 5::7]
  cov_xx = a**2 + _BIGGER_EPSILON
  cov_yy = b**2 + c**2 + _BIGGER_EPSILON
  cov_xy = a * b
  weights = forecasts_tensor[:, 6::7] + _BIGGER_EPSILON

  mixture_probs = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
  location = tf.stack([loc_x, loc_y], axis=2)
  covariance = (
      tf.stack(
          [
              tf.stack([cov_xx, cov_xy], axis=2),
              tf.stack([cov_xy, cov_yy], axis=2),
          ],
          axis=3,
      ),
  )

  distribution = tfp.distributions.MultivariateNormalFullCovariance(
      location, covariance, validate_args=True, allow_nan_stats=False
  )

  mixture_instance = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(probs=mixture_probs),
      components_distribution=distribution,
      validate_args=True,
      allow_nan_stats=False,
  )

  return mixture_instance


class MinusLoglikelihoodConstShiftStretchLoss(tf.keras.losses.Loss):
  """A loss class to compute -log(probability) of a random variable.

  Attributes:
    random_variable_generator: a function that returns a
      tfp.distributions.Distribution instance given 'forecasts'.
    shift: a shift factor for the distribution. This shifts the entire domain of
      the distribution, e.g. may be utilized to create an exponential
      distribution that starts at 'shift' rather than at 0.
    stretch: a stretch factor for the distribution. This stretches the entire
      domain of the distribution, e.g. may be utilized to create a beta
      distribution that spans on [0, stretch] instead of [0,1].
  """

  def __init__(
      self,
      random_variable_generator,
      shift = 0,
      stretch = 1,
  ):
    super().__init__()
    self.random_variable_generator = random_variable_generator
    self.shift = shift
    self.stretch = stretch

  def call(
      self,
      labels,
      forecasts,
  ):
    random_variable = self.random_variable_generator(forecasts)
    labels_tensor = tf.reshape(tf.convert_to_tensor(labels), (-1,))
    shifted_labels = tf.maximum(
        self.shift_strech_input(labels_tensor), _EPSILON
    )
    loglikelihood = random_variable.log_prob(
        tf.cast(shifted_labels, random_variable.dtype)
    ) - np.log(self.stretch)
    return -tf.math.reduce_mean(loglikelihood)

  def shift_strech_input(
      self, x
  ):
    """Transform the input value to the random variable."""
    return (x - self.shift) / self.stretch


class MinusLoglikelihoodLoss(tf.keras.losses.Loss):
  """A loss class to compute -log(probability) of a random variable.

  Attributes:
    random_variable_generator: A function that returns a
      tfp.distributions.Distribution instance given forecasts.
    shift: A shift factor for the distribution. This shifts the entire domain of
      the distribution, e.g. may be utilized to create an exponential
      distribution that starts at `shift` rather than at 0.
    labels_shape: The expected shape of the labels. If not present, assumed to
      be flat.
  """

  def __init__(
      self,
      random_variable_generator,
      shift = 0,
      labels_shape = None,
  ):
    super().__init__()
    self.random_variable_generator = random_variable_generator
    self.shift = shift
    self.labels_shape = labels_shape

  def call(self, labels, forecasts):
    """Calculates the loss."""
    random_variable = self.random_variable_generator(forecasts)
    shape = (-1,) if self.labels_shape is None else (-1, *self.labels_shape)
    labels_tensor = tf.reshape(tf.convert_to_tensor(labels), shape)
    shifted_labels = tf.maximum(labels_tensor - self.shift, _EPSILON)
    loglikelihood = random_variable.log_prob(shifted_labels)
    return -tf.math.reduce_mean(loglikelihood)


class ConditionalMinusLoglikelihoodLoss(tf.keras.losses.Loss):
  """A loss class to compute -log(conditional probability) of a random variable.

  A loss class to compute minus loglokilihood of a given random variable.
  Likelihood is computed as the conditional probability:
  Following Baye's law: P(m|m>m_c)~P(m)/P(m>m_c) where P(m) is the probability
  to return a certain magnitude, P(m>m_c) is the probability to return a
  magnitude higher than the magnitude threshold (typically the completeness
  magnitude).

  Attributes:
    random_variable_generator: a function that returns a
      tfp.distributions.Distribution instance given 'forecasts'.
    magnitude_threshold: the cutoff magnitude of the distribution. Typically the
      complentess magnitude.
    shift: a shift factor for the distribution. This shifts the entire domain of
      the distribution, e.g. may be utilized to create an exponential
      distribution that starts ans 'shift' rather than at 0.
  """

  def __init__(
      self,
      random_variable_generator,
      magnitude_threshold,
      shift = 0,
  ):
    super().__init__()
    self.random_variable_generator = random_variable_generator
    self.magnitude_threshold = magnitude_threshold
    self.shift = shift

  def call(
      self,
      labels,
      forecasts,
  ):
    # Create a tfp.distributions.Distribution instance:
    random_variable = self.random_variable_generator(forecasts)
    labels_tensor = tf.convert_to_tensor(labels)
    shifted_labels = labels_tensor - self.shift
    log_not_finite_logical = shifted_labels <= 0
    log_survival = random_variable.log_survival_function(
        self.magnitude_threshold
    )
    loglikelihood = random_variable.log_prob(shifted_labels)
    log_diff = log_survival - loglikelihood
    # values may be -inf or nan if either survival or likelihood are very low
    log_epsilon = tf.cast(_LOG_EPSILON, log_diff.dtype)
    log_diff = tf.where(log_not_finite_logical, -log_epsilon, log_diff)
    return tf.math.reduce_mean(log_diff)
