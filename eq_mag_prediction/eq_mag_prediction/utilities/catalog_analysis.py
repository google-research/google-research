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

"""Common statistical seismology analyses on catalogs."""

from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

import gin
import numpy as np
import pandas as pd
from scipy import stats
from shapely import geometry as shg
import tqdm
import xarray as xr

from eq_mag_prediction.utilities import geometry

BINS_TYPE = (
    int
    | Sequence[float]
    | Sequence[int]
    | tuple[Sequence[float], Sequence[float]]
    | None
)
DAYS_TO_SECONDS = 60 * 60 * 24
KM_TO_DEG = 1 / 111.11
MARGIN_EPSILON = 1e-6
KM_TO_DEG = 1 / 111.11
MARGIN_EPSILON = 1e-6


@gin.configurable
def estimate_beta(
    magnitudes,
    magnitude_threshold,
    method = 'MLE',
):
  """Estimate beta of the GR distribution by method of choice.

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    magnitude_threshold: The minimum threshold of magnitudes.
    method: string indicating the method by which to calculate beta.

  Returns:
    Estimate of beta.
  """
  if method == 'MLE':
    return estimate_beta_by_maximum_likelihood_estimation(
        magnitudes, magnitude_threshold
    )
  if method == 'BPOS':
    return estimate_beta_by_b_positive(magnitudes, None)
  return np.nan


def estimate_beta_by_maximum_likelihood_estimation(
    magnitudes, magnitude_threshold
):
  """Estimate the beta of the Gutenberg-Richter law with MLE.

  Calculation follows Aki (1965) and Marzocchi and Sandri (2003).

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    magnitude_threshold: The minimum threshold of magnitudes.

  Returns:
    The MLE estimate of beta.
  """
  magnitudes = np.array([m for m in magnitudes if m >= magnitude_threshold])
  return 1 / np.average(magnitudes - magnitude_threshold)


def estimate_beta_by_b_positive(
    magnitudes, minimal_magnitude = None
):
  """Estimate the beta of the Gutenberg-Richter using the b-positive method.

  b-positive method uses the differences between following events in a catalog
  in order to estimate the beta value of the GR distribution. It is considered
  less sucepstible to changes in temporal incompleteness and catalog size.
  The method is described in van der Elst 2021 JGR Solid Earth,
  https://doi.org/10.1029/2020JB021027

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    minimal_magnitude: A shift added to the beta calculation. If not given, the
      magnitude binning will be estimed if possible, otherwise the minimal
      magnitude difference will be used (see equation 6 and 8 in reference
      above).

  Returns:
    The estimate of beta.
  """
  # pylint: disable=g-explicit-length-test
  if len(magnitudes) == 0:
    return np.nan
  # pylint: enable=g-explicit-length-test
  if len(magnitudes) < 4:
    return estimate_beta_by_maximum_likelihood_estimation(
        magnitudes, np.max(np.array(magnitudes))
    )
  magnitude_diff = np.diff(magnitudes)
  if minimal_magnitude is None:
    # estimate if data is binned
    unique_magnitudes = np.unique(magnitudes)
    if (len(unique_magnitudes) / len(magnitudes) < 0.1) | (
        len(unique_magnitudes) < 10
    ):
      minimal_magnitude = (
          2 * np.diff(unique_magnitudes).min()
      )  # Mc in eq. 8 in ref.
    else:
      minimal_magnitude = np.abs(magnitude_diff).min()  # Mc in eq. 6 in ref.

  return 1 / (
      magnitude_diff[magnitude_diff >= minimal_magnitude].mean()
      - minimal_magnitude
  )  # Eq. 6 in ref.


def estimate_beta_given_mc(
    timestamps,
    mc,
    catalog,
    n_events = 250,
    weight_on_past = 1.0,
):
  """Estimates beta parameter on a window with specified completeness magnitude.

  Args:
    timestamps: Timestamps for which to estimate beta.
    mc: Magnitude completeness value corresponding to timestamps.
    catalog: A catalog to use for the calculation.
    n_events: Number of past events to use to compute the current distribution.
      Should be a positive integer, Defaults to 250 as proposed in Gulia &
      Wiemer (Nature, 2019).
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      all n_events are prior to estimation time, whereas 0 considers only
      present and future events.

  Returns:
    Array of beta values, length of the timestamps.
  """
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  window_past = int(np.round(n_events * weight_on_past))
  window_future = n_events - window_past
  timestamps = np.array(timestamps)
  cat_time = catalog.time.values
  cat_mag = catalog.magnitude.values

  beta = np.empty_like(timestamps)
  for i, t in enumerate(timestamps):
    insert_index = np.searchsorted(cat_time, t, side='left')
    magnitudes = cat_mag[
        (insert_index - window_past) : (insert_index + window_future)
    ]
    beta[i] = estimate_beta(magnitudes, mc[i])
  return beta


def estimate_completeness_and_beta(
    magnitudes,
    method = 'MBS',
    **kwargs,
):
  """Estimates the completeness magnitude and beta of a catalog.

  A wrapper function for algorithms to find completeness and beta using an
  algorithm of choice.

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    method: All caps abbreviation of the algorithm for computing the
      completeness magnitude: 'MBS' - Default. Computes using the Mc by b-value
      stability method. See estimate_completeness_and_beta_by_b_stability
      documentation for references. 'MAXC' - Maximum curvature method. See
      estimate_completeness_and_beta_by_maximal_curvature documentation for
      references.
    **kwargs: Additional optional parameters, method specific.

  Returns:
    An estimate for the completeness magnitude.
    An estimate of the beta value of magnitudes above the estimated
      completeness.
  """
  if method == 'MBS':
    mc, beta = estimate_completeness_and_beta_by_b_stability(
        magnitudes, **kwargs
    )
  elif method == 'MAXC':
    mc, beta = estimate_completeness_and_beta_by_maximal_curvature(
        magnitudes, **kwargs
    )
  elif method == 'CONST':
    mc, beta = estimate_completeness_and_beta_with_constant_mc(
        magnitudes, **kwargs
    )
  else:
    mc, beta = np.nan, np.nan
  return mc, beta


def estimate_completeness_and_beta_by_b_stability(
    magnitudes,
    min_mc = 1,
    max_mc = 5,
    n_thresholds_for_mean = 5,
):
  """Estimates the completeness magnitude and beta of a catalog.

  The calculation follows 'Mc by b-value stability (MBS) (Cao and Gao, 2002)'
  Introduced in Woessner and Wiemer (2005) as method number 4. A simple
  explanation can be found on CORSSA.

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    min_mc: An underestimate for the magnitude completeness.
    max_mc: An overestimate for the magnitude completeness.
    n_thresholds_for_mean: A hyperparameter for calculating the stability of the
      b-value. Set to 5 in the original paper.

  Returns:
    An estimate for the completeness magnitude, between `min_mc` and `max_mc`,
      in increments of 0.1.
    An estimate of the beta value of magnitudes above the estimated
      completeness.
  """
  mc_to_test = np.arange(min_mc, max_mc + 0.01, 0.1)
  best_mc, best_b, least_change = np.nan, np.nan, 100
  b_values, b_uncertainties = [], []

  for mc in mc_to_test:
    above_mc = [m for m in magnitudes if m >= mc]
    b_values.append(estimate_beta(above_mc, mc, 'MLE') * np.log10(np.e))

    # Formula due to Shi and Bolt (1982), used in Woessner and Wiemer (2005).
    n_events = len(above_mc)
    mean_magnitude = np.mean(above_mc)
    b_value_uncertainty = 2.3 * (b_values[-1] ** 2)
    b_value_uncertainty *= (
        np.sum((above_mc - mean_magnitude) ** 2) / (n_events * (n_events - 1))
    ) ** 0.5
    b_uncertainties.append(b_value_uncertainty)

  average_bs = np.convolve(
      b_values,
      np.ones(n_thresholds_for_mean) / n_thresholds_for_mean,
      mode='valid',
  )
  shift = n_thresholds_for_mean - 1
  b_values, b_uncertainties, mc_to_test = (
      b_values[:-shift],
      b_uncertainties[:-shift],
      mc_to_test[:-shift],
  )

  for i, average_b in enumerate(average_bs):
    mc, b_value, uncertainty = mc_to_test[i], b_values[i], b_uncertainties[i]
    change = abs(average_b - b_value)
    if change <= min(uncertainty, 0.03):
      return mc, np.log(10) * b_value

    if change <= least_change:
      # We should find a threshold where the change in b-value is below the
      # uncertainty. If we don't, return the threshold with the lowest change.
      best_mc = mc
      best_b = b_value
      least_change = change

  return best_mc, np.log(10) * best_b


def estimate_completeness_and_beta_by_maximal_curvature(
    magnitudes,
    mc_addition_factor = 0.2,
):
  mc = estimate_completeness_by_maximal_curvature(
      magnitudes, mc_addition_factor=mc_addition_factor
  )
  beta = estimate_beta(magnitudes, mc)
  return mc, beta


def estimate_completeness_and_beta_with_constant_mc(
    magnitudes,
    mc_value = 2.4,
):
  beta = estimate_beta(magnitudes, mc_value)
  return mc_value, beta


@gin.configurable
def estimate_completeness(
    magnitudes,
    method = 'MBS',
    **kwargs,
):
  """Estimates the completeness magnitude of a catalog."""
  if method == 'MBS':
    mc = estimate_completeness_by_b_stability(magnitudes, **kwargs)
  elif method == 'MAXC':
    mc = estimate_completeness_by_maximal_curvature(magnitudes, **kwargs)
  else:
    mc = np.nan
  return mc


def estimate_completeness_by_b_stability(
    magnitudes,
    min_mc = 1,
    max_mc = 5,
    n_thresholds_for_mean = 5,
):
  """Estimates the completeness magnitude of a catalog."""
  m_c, _ = estimate_completeness_and_beta_by_b_stability(
      magnitudes, min_mc, max_mc, n_thresholds_for_mean
  )
  return m_c


def estimate_completeness_by_maximal_curvature(
    magnitudes,
    mc_addition_factor = 0.2,
    bin_width = 0.1,
):
  """Estimates the completeness magnitude and beta of a catalog.

  The calculation follows 'Maximum curvature-method (MAXC) (Wiemer and Wyss,
  2000)'. Introduced in Woessner and Wiemer (2005) as method number 2.

  Args:
    magnitudes: A list of magnitudes of earthquakes.
    mc_addition_factor: It has been empirically verified adding a factor of 0.2
      to the maximal curvature result is recommended (see Woessner & Wiemer,
      2005).
    bin_width: Bin width of the histogram upon which the maximum curvature is
      searched for.

  Returns:
    An estimate for the completeness magnitude.
    An estimate of the beta value of magnitudes above the estimated
      completeness.
  """

  # pylint: disable=g-explicit-length-test
  if len(magnitudes) == 0:
    return np.nan
  if len(magnitudes) == 1:
    return magnitudes[0]
  # pylint: enable=g-explicit-length-test
  magnitudes = np.array(magnitudes)
  min_hist_value = np.minimum(magnitudes.min(), 0)
  magnitude_bins = np.arange(
      min_hist_value - bin_width / 2,
      magnitudes.max() + bin_width / 2,
      bin_width,
  )
  hist, _ = np.histogram(magnitudes, bins=magnitude_bins)
  max_bin = (
      len(hist) - 1 - np.argmax(np.flip(hist))
  )  # Take the largest index maximal bin.
  mc = magnitude_bins[max_bin : (max_bin + 2)].mean() + mc_addition_factor
  return mc


def return_constant_completeness(
    unused_magnitudes,
    return_mc = 2.0,
):
  """Utility function returning a constant value regardless of input magnitudes."""
  return return_mc


def estimate_completeness_by_analytical_formula(
    timestamps,
    catalog,
    mainshock_mag = 6,
    lower_bound = 1.67,
):
  """Estimates aftershock temporal incompleteness following Helmstetter etal.

  Helmstetter et. al. https://doi.org/10.1785/0120050067 presents a method for
  approximating the temporal incompleteness after large earthquakes by fitting
  a decaying logarithm bound by a minimal value.

  Args:
    timestamps: Timestamps for which to estimate the completness and beta.
    catalog: A catalog to use for the calculation.
    mainshock_mag: The magnitude above which an earthquake is considered a
      mainshock.
    lower_bound: The minimal value the temporal incompleteness can reach.

  Returns:
    An numpy array indicating the temporal incompleteness per timestamp.
  """

  magnitudes = catalog.magnitude.values
  timestamps = np.array(timestamps)

  mainshock_idxs = np.where(magnitudes >= mainshock_mag)[0]
  large_event_mags = magnitudes[mainshock_idxs]
  large_event_time = catalog.time.iloc[mainshock_idxs].values
  insert_indices = np.searchsorted(large_event_time, timestamps, side='left')
  mc = np.full_like(timestamps, lower_bound)
  for i in np.unique(insert_indices):
    if i == 0:
      continue
    else:
      last_reference_time = large_event_time[i - 1]
      time_since_reference = (
          timestamps[insert_indices == i] - last_reference_time
      )
      mc[insert_indices == i] = _helmstetter_mc_formula(
          large_event_mags[i - 1], time_since_reference, lower_bound
      )
  return mc


def _helmstetter_mc_formula(
    m,
    dt,
    lower_bound = 1.67,
):
  """Follows Helmstetter et. al. https://doi.org/10.1785/0120050067.

  The constant for the formula, including default value for 'lower_bound',
  were fitted for the Hauksson catalog.

  Args:
    m: Mainshock's magnitude for which to calculate the temporal incompleteness.
    dt: Time (seconds) since the mainshock at which to estimate incompleteness.
    lower_bound: Minimal value temporal incompleteness is bound at.

  Returns:
    An array, same size as dt, with values of the temporal incompletenes
    magnitudes.
  """
  return np.maximum(
      (m - 4.06 - 0.44 * np.log10(dt.ravel() / DAYS_TO_SECONDS)), lower_bound
  )


def kde_moving_window_n_events(
    estimate_times,
    catalog,
    n_events = 250,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    completeness_calculator = None,
):
  """Computes a KDE for each timestamp, based on the last n_events.

  Args:
    estimate_times: Timestamps for which to estimate the completness and beta.
    catalog: A catalog to use for the calculation.
    n_events: Number of past events to use to compute the current distribution.
      Should be a positive integer, Defaults to 250 as proposed in Gulia &
      Wiemer (Nature, 2019).
    m_minimal: A threshold determining the minimal magnitude to include in the
      last n_events.
    n_above_complete: A minimal number of events above the computed completenes
      magnitude. If computation results in less than n_above_complete, will
      recompute by enlarging n_events for the current timestamp. Defaults to 1,
      should be a positive integer, Gulia & Wiemer (Nature, 2019) set to 50.
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      all n_events are prior to estimation time, whereas 0 considers only
      present and future events.
    completeness_calculator: A callable that takes a sequence of magnitudes and
      returns a float, completeness. E.g.
      estimate_completeness_by_maximal_curvature (default)

  Returns:
    Array of magnitude completeness values, length of the estimate_timestamps
    List of scipy.stats.gaussian_kde instances, same length.
  """

  assert n_events > 0
  assert n_above_complete > 0
  assert n_above_complete < n_events
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  if completeness_calculator is None:
    completeness_calculator = estimate_completeness_by_maximal_curvature

  def mc_and_kde(magnitudes):
    mc_inner = completeness_calculator(magnitudes)
    if len(magnitudes) > 1:
      kde_inner = stats.gaussian_kde(magnitudes)
    # force prediction in case of single data point
    elif len(magnitudes) == 1:
      kde_inner = stats.gaussian_kde(
          [magnitudes[0], magnitudes[0] + MARGIN_EPSILON]
      )
    # force ignorant prediction in case no data is available
    else:
      mu = catalog.magnitude.values.mean()
      kde_inner = stats.gaussian_kde([mu, mu + MARGIN_EPSILON])
    return mc_inner, kde_inner

  completeness_vec = []
  kde_list = []
  for t in estimate_times:
    mc, kde = _return_completeness_and_beta_for_timestamp(
        t,
        catalog,
        n_events,
        m_minimal,
        n_above_complete,
        weight_on_past,
        mc_and_kde,
    )
    completeness_vec.append(mc)
    kde_list.append(kde)
  completeness_vec = np.array(completeness_vec)
  kde_list = np.array(kde_list)
  return completeness_vec, kde_list


def kde_moving_window_n_events(
    estimate_times,
    catalog,
    n_events = 250,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    completeness_calculator = None,
):
  """Computes a KDE for each timestamp, based on the last n_events.

  Args:
    estimate_times: Timestamps for which to estimate the completness and beta.
    catalog: A catalog to use for the calculation.
    n_events: Number of past events to use to compute the current distribution.
      Should be a positive integer, Defaults to 250 as proposed in Gulia &
      Wiemer (Nature, 2019).
    m_minimal: A threshold determining the minimal magnitude to include in the
      last n_events.
    n_above_complete: A minimal number of events above the computed completenes
      magnitude. If computation results in less than n_above_complete, will
      recompute by enlarging n_events for the current timestamp. Defaults to 1,
      should be a positive integer, Gulia & Wiemer (Nature, 2019) set to 50.
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      all n_events are prior to estimation time, whereas 0 considers only
      present and future events.
    completeness_calculator: A callable that takes a sequence of magnitudes and
      returns a float, completeness. E.g.
      estimate_completeness_by_maximal_curvature (default)

  Returns:
    Array of magnitude completeness values, length of the estimate_timestamps
    List of scipy.stats.gaussian_kde instances, same length.
  """

  assert n_events > 0
  assert n_above_complete > 0
  assert n_above_complete < n_events
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  if completeness_calculator is None:
    completeness_calculator = estimate_completeness_by_maximal_curvature

  def mc_and_kde(magnitudes):
    mc_inner = completeness_calculator(magnitudes)
    if len(magnitudes) > 1:
      kde_inner = stats.gaussian_kde(magnitudes)
    # force prediction in case of single data point
    elif len(magnitudes) == 1:
      kde_inner = stats.gaussian_kde(
          [magnitudes[0], magnitudes[0] + MARGIN_EPSILON]
      )
    # force ignorant prediction in case no data is available
    else:
      mu = catalog.magnitude.values.mean()
      kde_inner = stats.gaussian_kde([mu, mu + MARGIN_EPSILON])
    return mc_inner, kde_inner

  completeness_vec = []
  kde_list = []
  for t in estimate_times:
    mc, kde = _return_completeness_and_beta_for_timestamp(
        t,
        catalog,
        n_events,
        m_minimal,
        n_above_complete,
        weight_on_past,
        mc_and_kde,
    )
    completeness_vec.append(mc)
    kde_list.append(kde)
  completeness_vec = np.array(completeness_vec)
  kde_list = np.array(kde_list)
  return completeness_vec, kde_list


def gr_moving_window_n_events(
    estimate_times,
    catalog,
    n_events = 250,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    completeness_and_beta_calculator = None,
):
  """Computes a GR distribution for each timestamp, based on the last n_events.

  Args:
    estimate_times: Timestamps for which to estimate the completness and beta.
    catalog: A catalog to use for the calculation.
    n_events: Number of past events to use to compute the current distribution.
      Should be a positive integer, Defaults to 250 as proposed in Gulia &
      Wiemer (Nature, 2019).
    m_minimal: A threshold determining the minimal magnitude to include in the
      last n_events.
    n_above_complete: A minimal number of events above the computed completenes
      magnitude. If computation results in less than n_above_complete, will
      recompute by enlarging n_events for the current timestamp. Defaults to 1,
      should be a positive integer, Gulia & Wiemer (Nature, 2019) set to 50.
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      all n_events are prior to estimation time, whereas 0 considers only
      present and future events.
    completeness_and_beta_calculator: A callable that takes a sequence of
      magnitudes and returns a tuple of (completeness, beta). E.g.
      estimate_completeness_and_beta_by_maximal_curvature or
      estimate_completeness_and_beta_by_b_stability (default).

  Returns:
    Array of magnitude completeness values, length of the estimate_timestamps
    Array of beta values, length of the estimate_timestamps
  """

  assert n_events > 0
  assert n_above_complete > 0
  assert n_above_complete < n_events
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  if completeness_and_beta_calculator is None:
    completeness_and_beta_calculator = (
        estimate_completeness_and_beta_by_b_stability
    )

  completeness_vec = []
  beta_vec = []
  for t in estimate_times:
    mc, beta = _return_completeness_and_beta_for_timestamp(
        t,
        catalog,
        n_events,
        m_minimal,
        n_above_complete,
        weight_on_past,
        completeness_and_beta_calculator,
    )
    completeness_vec.append(mc)
    beta_vec.append(beta)
  completeness_vec = np.array(completeness_vec)
  beta_vec = np.array(beta_vec)
  return completeness_vec, beta_vec


def gr_moving_window_constant_time(
    estimate_times,
    catalog,
    window_time = 10 * DAYS_TO_SECONDS,
    m_minimal = -100,
    n_above_complete = 1,
    weight_on_past = 1,
    default_beta = np.nan,
    default_mc = None,
    completeness_calculator = None,
):
  """Computes a GR distribution for each timestamp, based on current time window.

  Args:
    estimate_times: Timestamps for which to estimate the completness and beta.
    catalog: A catalog to use for the calculation.
    window_time: The length of the window in seconds from which to consider the
      magnitudes for the calculation.
    m_minimal: A threshold determining the minimal magnitude to include in the
      last n_events.
    n_above_complete: A minimal number of events above the computed completenes
      magnitude. If computation results in less than n_above_complete, assign
      default_beta for the current timestamp.
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      entire window is prior to estimation time, whereas 0 considers only
      present and future events.
    default_beta: The default value returned in case where not enough samples
      are present in time window.
    default_mc: A constant value of magnitude completeness to consider for the
      calculation of beta. If not given, will recalculate for each timestamp
      using completeness_calculator.
    completeness_calculator: A callable that takes a sequence of magnitudes and
      returns a float (completeness magnitude). E.g.
      estimate_completeness_by_b_stability (default) or
      estimate_completeness_by_maximal_curvature. Will be used to reevaluate the
      completeness magnitude in each timestamp, if a default_mc is not given.

  Returns:
    Two arrays, one for magnitude completeness values, and one for beta values
    (each has the length of estimate_timestamps).
  """

  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  if default_mc is not None:
    completeness_calculator = lambda magnitudes: default_mc
  if completeness_calculator is None:
    completeness_calculator = estimate_completeness_by_b_stability

  completeness_vec = []
  beta_vec = []
  for t in estimate_times:
    last_index = np.searchsorted(
        catalog.time.values,
        t + ((1 - weight_on_past) * window_time),
        side='left',
    )
    first_index = np.searchsorted(
        catalog.time.values, t - (weight_on_past * window_time), side='left'
    )
    time_slice = slice(first_index, last_index)
    magnitudes_time_sliced = catalog.magnitude.values[time_slice]
    magnitude_logical = magnitudes_time_sliced >= m_minimal
    magnitudes = magnitudes_time_sliced[magnitude_logical]
    completeness_vec.append(completeness_calculator(magnitudes))
    if (magnitudes >= completeness_vec[-1]).sum() < n_above_complete:
      beta_vec.append(default_beta)
    else:
      beta_vec.append(estimate_beta(magnitudes, completeness_vec[-1]))
  completeness_vec = np.array(completeness_vec)
  beta_vec = np.array(beta_vec)
  return completeness_vec, beta_vec


def gr_spatial_beta_const_mc(
    estimate_coors,
    catalog,
    completeness_magnitude = None,
    mc_calc_method = 'MAXC',
    grid_spacing = 0.1,
    smoothing_distance = 30 * KM_TO_DEG,
    discard_few_event_locations = 200,
    estimate_by_vicinity = False,
    display_progress = False,
):
  """Computes a local GR distribution for each coordinate.

  Essentially a wrapper for SpatialBetaCalculator.__call__.
  This is an implementation of the method resented in
  Taroni, Zhuang, Marzocchi, SRL 2021, https://doi.org/10.1785/0220210017
  Where the completness magnitude is computed across all coordinates jointly,
  and beta is computed per coordinate, then smoothed.

  Args:
    estimate_coors: Sequence of (lon, lat) pairs where beta will be estimated.
    catalog: A catalog to use for the calculation.
    completeness_magnitude: A constant value of magnitude completeness to
      consider for the calculation of beta. If not given, will recalculate for
      the entire catalog using mc_calc_method.
    mc_calc_method: The method by which to compute mc is completeness_magnitude
      is not given. Defaults to 'MAXC', see
      catalog_analysis.estimate_completeness for options.
    grid_spacing: Spacing of the grid that defines the spatial binning of the
      events in catalog, for which beta will be calculated.
    smoothing_distance: The radius of the circular smoothing kernel, in degrees.
      Defaults to the equivalent of 30km.
    discard_few_event_locations: Set beta as np.nan for coordinates which use
      calculations of discard_few_event_locations or fewer. None is equivalent
      to inf.
    estimate_by_vicinity: bool. Whether or not to use the
      estimate_beta_at_location_by_vicinity_to_sample method which is cheaper to
      compute. Default to False.
    display_progress: Whether or not to display a progress bar.

  Returns:
    Two arrays, one for magnitude completeness values, and one for beta values
    (each has the length of estimate_coors).
  """
  spatial_beta_inst = SpatialBetaCalculator(
      catalog=catalog,
      completeness_magnitude=completeness_magnitude,
      mc_calc_method=mc_calc_method,
      grid_spacing=grid_spacing,
      smoothing_distance=smoothing_distance,
      display_progress=display_progress,
  )

  lons, lats = list(zip(*estimate_coors))
  if estimate_by_vicinity:
    spatial_betas_result = (
        spatial_beta_inst.estimate_beta_at_location_by_vicinity_to_sample(
            lons, lats, discard_few_event_locations
        )
    )
  else:
    spatial_betas_result = spatial_beta_inst(
        lons, lats, discard_few_event_locations
    )
  mc_vec = np.full_like(
      spatial_betas_result, spatial_beta_inst.completeness_magnitude
  )
  return mc_vec, spatial_betas_result


def gr_spatial_beta_const_mc(
    estimate_coors,
    catalog,
    completeness_magnitude = None,
    mc_calc_method = 'MAXC',
    grid_spacing = 0.1,
    smoothing_distance = 30 * KM_TO_DEG,
    discard_few_event_locations = 200,
    estimate_by_vicinity = False,
    display_progress = False,
):
  """Computes a local GR distribution for each coordinate.

  Essentially a wrapper for SpatialBetaCalculator.__call__.
  This is an implementation of the method resented in
  Taroni, Zhuang, Marzocchi, SRL 2021, https://doi.org/10.1785/0220210017
  Where the completness magnitude is computed across all coordinates jointly,
  and beta is computed per coordinate, then smoothed.

  Args:
    estimate_coors: Sequence of (lon, lat) pairs where beta will be estimated.
    catalog: A catalog to use for the calculation.
    completeness_magnitude: A constant value of magnitude completeness to
      consider for the calculation of beta. If not given, will recalculate for
      the entire catalog using mc_calc_method.
    mc_calc_method: The method by which to compute mc is completeness_magnitude
      is not given. Defaults to 'MAXC', see
      catalog_analysis.estimate_completeness for options.
    grid_spacing: Spacing of the grid that defines the spatial binning of the
      events in catalog, for which beta will be calculated.
    smoothing_distance: The radius of the circular smoothing kernel, in degrees.
      Defaults to the equivalent of 30km.
    discard_few_event_locations: Set beta as np.nan for coordinates which use
      calculations of discard_few_event_locations or fewer. None is equivalent
      to inf.
    estimate_by_vicinity: bool. Whether or not to use the
      estimate_beta_at_location_by_vicinity_to_sample method which is cheaper to
      compute. Default to False.
    display_progress: Whether or not to display a progress bar.

  Returns:
    Two arrays, one for magnitude completeness values, and one for beta values
    (each has the length of estimate_coors).
  """
  spatial_beta_inst = SpatialBetaCalculator(
      catalog=catalog,
      completeness_magnitude=completeness_magnitude,
      mc_calc_method=mc_calc_method,
      grid_spacing=grid_spacing,
      smoothing_distance=smoothing_distance,
      display_progress=display_progress,
  )

  lons, lats = list(zip(*estimate_coors))
  if estimate_by_vicinity:
    spatial_betas_result = (
        spatial_beta_inst.estimate_beta_at_location_by_vicinity_to_sample(
            lons, lats, discard_few_event_locations
        )
    )
  else:
    spatial_betas_result = spatial_beta_inst(
        lons, lats, discard_few_event_locations
    )
  mc_vec = np.full_like(
      spatial_betas_result, spatial_beta_inst.completeness_magnitude
  )
  return mc_vec, spatial_betas_result


def _return_completeness_and_beta_for_timestamp(
    timestamp,
    catalog,
    n_events,
    m_minimal,
    n_above_complete,
    weight_on_past,
    completeness_and_beta_calculator,
):
  """Returns completeness and beta for a given timestamp, computed on n events."""
  return _return_properties_for_timestamp(
      timestamp,
      catalog,
      n_events,
      m_minimal,
      n_above_complete,
      weight_on_past,
      completeness_and_beta_calculator,
  )


def _return_properties_for_timestamp(
    timestamp,
    catalog,
    n_events,
    m_minimal,
    n_above_complete,
    weight_on_past,
    property_function,
):
  """Returns properties for a given timestamp, computed on n events.

  Will return mc and a specified property derived from the last n events'
  magnitudes. mc is required as it is used to determine if enough events have
  been considered.
  Args:
    timestamp: The timestamp at which to estimate the window.
    catalog:  Catalog to estimate for.
    n_events: Number of events, i.e. window length.
    m_minimal: Minimal mc to consider for calculation.
    n_above_complete: A minimal number of events above the computed completenes
      magnitude. If computation results in less than n_above_complete, will
      recompute by enlarging n_events for the current timestamp. Defaults to 1,
      should be a positive integer, Gulia & Wiemer (Nature, 2019) set to 50.
    weight_on_past: Float in range [0, 1]. 1 considers only past events, i.e.
      all n_events are prior to estimation time, whereas 0 considers only
      present and future events.
    property_function: A function returning mc and another calculation given a
      sequence of magnitudes.

  Returns:
    mc and the property given by property_function.
  """
  n_events_past = int(np.round(n_events * weight_on_past))
  n_events_future = n_events - n_events_past

  past_time_logical = catalog.time < timestamp
  future_time_logical = catalog.time >= timestamp
  magnitude_logical = catalog.magnitude >= m_minimal
  past_time_magnitude_logical = past_time_logical & magnitude_logical
  future_time_magnitude_logical = future_time_logical & magnitude_logical
  mc, return_value = np.nan, np.nan
  above_complete_counter = 0
  additional_events_past = 0
  additional_events_future = 0
  past_idxs = np.where(past_time_magnitude_logical)[0]
  future_idxs = np.where(future_time_magnitude_logical)[0]
  while above_complete_counter < n_above_complete:
    last_past_idxs = past_idxs[
        len(past_idxs)
        - (n_events_past + additional_events_past) : len(past_idxs)
    ]
    magnitudes_to_use_past = catalog.magnitude.values[last_past_idxs]

    first_future_idxs = future_idxs[
        : (n_events_future + additional_events_future)
    ]
    magnitudes_to_use_future = catalog.magnitude.values[first_future_idxs]
    magnitudes_to_use = np.concatenate(
        (magnitudes_to_use_past, magnitudes_to_use_future)
    )

    mc, return_value = property_function(magnitudes_to_use)
    above_complete_counter = (magnitudes_to_use >= mc).sum()

    # If timestamp is too close to either edge, break and use result as is:
    required_past_events_exceeds_length = (
        n_events_past + additional_events_past
        > past_time_magnitude_logical.sum()
    )
    required_future_events_exceeds_length = (
        n_events_future + additional_events_future
        > future_time_magnitude_logical.sum()
    )
    if (
        required_past_events_exceeds_length
        or required_future_events_exceeds_length
    ):
      break

    additional_events = additional_events_past + additional_events_future
    additional_events += np.maximum(
        n_above_complete - above_complete_counter, 0
    )
    additional_events_past = int(np.round(additional_events * weight_on_past))
    additional_events_future = additional_events - additional_events_past
  return mc, return_value


def compute_property_in_time_and_space(
    catalog,
    property_function,
    examples,
    grid_side_deg,
    lookback_seconds,
    magnitudes,
):
  """Computes a function of magnitudes for a set of examples.

  Will return the products of the property_function for all earthquakes above a
  certain magnitude, in a defined area around the example, in a defined
  time-window preceding the example.

  Args:
    catalog: Catalog of earthquakes.
    property_function: A function that takes a catalog, time_slice, centers,
      grid_side_degrees to return n scalars computed in the cells defined by the
      grid size on the time slice of the catalog. Should return a np.ndarray of
      shape [j,k,n] ((j, k)-th cell, n-th return of computation).
    examples: Times and locations at which to calculate the features.
    grid_side_deg: The size (in degrees) of every grid cell.
    lookback_seconds: Time intervals (in the past from the evaluation times) for
      calculation of seismicity rates.
    magnitudes: Magnitude thresholds to calculate seismicity.

  Returns:
    A 6-dimensional array, that holds at index [i,j,k,l,m,n] the n-th quantity
    computed, in the (j, k)-th cell, in the `lookback_seconds[l]` seconds before
    the i-th example time, above magnitude `magnitudes[m]`.
  """
  magnitudes = sorted(magnitudes, reverse=True)
  lookback_seconds = np.array(sorted(lookback_seconds, reverse=True)).astype(
      'float64'
  )
  first_locations = next(iter(examples.values()))
  computation_output = property_function(
      catalog.iloc[:1], slice(0, 1), list(examples.values())[0], grid_side_deg
  )
  function_output_size = computation_output.shape[-1]

  features = np.zeros((
      len(examples),
      len(first_locations),
      len(first_locations[0]),
      len(lookback_seconds),
      len(magnitudes),
      function_output_size,
  ))
  for mag_i, magnitude in enumerate(magnitudes):
    subcatalog = catalog[catalog.magnitude >= magnitude]
    times_array = subcatalog.time.values

    for time_i, timestamp in enumerate(examples.keys()):
      last_index = np.searchsorted(times_array, timestamp, side='left')
      centers = examples[timestamp]

      for lookback_i, lookback in enumerate(lookback_seconds):
        first_index = np.searchsorted(
            times_array, timestamp - lookback, side='left'
        )
        time_slice = slice(first_index, last_index)
        computation_output = property_function(
            subcatalog, time_slice, centers, grid_side_deg
        )
        features[time_i, :, :, lookback_i, mag_i, :] = computation_output
  return features


def function_in_square(
    catalog,
    time_slice,
    centers,
    grid_side_degrees,
    function,
):
  """Applies a function to the magnitudes of earthquakes in given squares."""
  catalog_lngs = catalog.longitude.values[time_slice]
  catalog_lats = catalog.latitude.values[time_slice]
  magnitudes = catalog.magnitude.values[time_slice]
  center_lngs = np.hstack([[center.lng for center in row] for row in centers])
  center_lats = np.hstack([[center.lat for center in row] for row in centers])

  in_longitude_range = np.greater_equal.outer(
      catalog_lngs, center_lngs - grid_side_degrees / 2
  ) & np.less.outer(catalog_lngs, center_lngs + grid_side_degrees / 2)
  in_latitude_range = np.greater_equal.outer(
      catalog_lats, center_lats - grid_side_degrees / 2
  ) & np.less.outer(catalog_lats, center_lats + grid_side_degrees / 2)
  index = in_longitude_range & in_latitude_range

  function_output = function(magnitudes[:1])
  sequence_output = isinstance(function_output, Sequence)
  output_length = len(function_output) if sequence_output else 1
  res = np.zeros((len(centers) * len(centers[0]), output_length))

  for column in range(index.shape[1]):
    function_output = function(magnitudes[np.where(index[:, column])])
    if sequence_output:
      for out_i, _ in enumerate(function_output):
        res[column, out_i] = function_output[out_i]
    else:
      res[column, 0] = function_output

  return res.reshape((len(centers), -1, output_length))


def energy_in_square(
    catalog,
    time_slice,
    centers,
    grid_side_degrees,
):
  """Calculates the energy of earthquakes from given earthquake locations."""
  return function_in_square(
      catalog,
      time_slice,
      centers,
      grid_side_degrees,
      lambda magnitudes: np.sum(np.exp(magnitudes)),
  )


def counts_in_square(
    catalog,
    time_slice,
    centers,
    grid_side_degrees,
):
  """Counts the number of earthquakes from given earthquake locations."""
  return function_in_square(
      catalog,
      time_slice,
      centers,
      grid_side_degrees,
      lambda magnitudes: magnitudes.size,
  )


def completeness_and_beta_in_square(
    catalog,
    time_slice,
    centers,
    grid_side_degrees,
    method = 'MBS',
):
  """Returns m_c and beta of earthquakes from given earthquake locations."""

  def estimate_completeness_and_beta_with_method(magnitudes):
    return estimate_completeness_and_beta(magnitudes, method=method)

  return function_in_square(
      catalog,
      time_slice,
      centers,
      grid_side_degrees,
      estimate_completeness_and_beta_with_method,
  )


@gin.configurable
def beta_of_constant_completeness_in_square(
    catalog,
    time_slice,
    centers,
    grid_side_degrees,
    completeness_magnitude = None,
):
  """Returns m_c and beta of earthquakes from given earthquake locations."""
  if completeness_magnitude is None:
    completeness_magnitude = estimate_completeness(catalog.magnitude.values)

  def estimate_beta_for_m_c(magnitudes):
    return estimate_beta(magnitudes, completeness_magnitude)

  return function_in_square(
      catalog,
      time_slice,
      centers,
      grid_side_degrees,
      estimate_beta_for_m_c,
  )


def estimate_beta_on_grid(
    catalog,
    bins = None,
):
  """Estimates beta values from binned seismicity map."""

  beta_grid, _, _, binnumber = stats.binned_statistic_2d(
      catalog.longitude.values,
      catalog.latitude.values,
      catalog.magnitude.values,
      statistic=estimate_beta_by_b_positive,
      bins=bins,
      expand_binnumbers=True,
  )
  counts_grid, _, _, _ = stats.binned_statistic_2d(
      catalog.longitude.values,
      catalog.latitude.values,
      catalog.magnitude.values,
      statistic='count',
      bins=bins,
  )
  return beta_grid, counts_grid, binnumber


class SpatialBetaCalculator:
  """Class to calculate spatialy varying beta value.

  Default argument values follow:
  Taroni, Zhuang, Marzocchi, SRL 2021, https://doi.org/10.1785/0220210017
  """

  def __init__(
      self,
      catalog,
      completeness_magnitude = None,
      mc_calc_method = 'MAXC',
      grid_spacing = 0.1,
      smoothing_distance = 30 * KM_TO_DEG,
      display_progress = False,
  ):
    self.catalog = catalog
    self._display_progress = display_progress
    if completeness_magnitude is None:
      self.completeness_magnitude = estimate_completeness(
          self.catalog.magnitude.values, method=mc_calc_method
      )
    else:
      self.completeness_magnitude = completeness_magnitude
    self.grid_spacing = grid_spacing
    self.smoothing_distance = smoothing_distance
    self.calc_beta_on_grid()

  def __call__(
      self,
      longitude,
      latitude,
      discard_few_event_locations = None,
  ):
    """Calculates beta at given locations, with smoothing.

    Args:
      longitude: Sequence of longitudes to estimate the smoothed beta at.
      latitude: Sequence of latitudes to estimate the smoothed beta at.
      discard_few_event_locations: If given, will discard locations with less
        than this number of events.

    Returns:
      Array of beta values, length of the input sequences.
    """
    return self._sample_smoothed_beta_value_for_query_coors(
        longitude, latitude, discard_few_event_locations
    )

  def calc_beta_on_grid(self):
    """Calculates beta on a grid of points, store result in instance."""
    lon_vec = np.arange(
        self.catalog.longitude.min(),
        self.catalog.longitude.max(),
        self.grid_spacing,
    )
    lat_vec = np.arange(
        self.catalog.latitude.min(),
        self.catalog.latitude.max(),
        self.grid_spacing,
    )
    bin_edges = (
        np.append(
            lon_vec - self.grid_spacing / 2, lon_vec[-1] + self.grid_spacing / 2
        ),
        np.append(
            lat_vec - self.grid_spacing / 2, lat_vec[-1] + self.grid_spacing / 2
        ),
    )
    beta_grid, counts_grid, binnumber = estimate_beta_on_grid(
        self.catalog, bin_edges
    )
    self.counts_binnumber = binnumber
    self.beta_xr = xr.DataArray(
        beta_grid.T,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat_vec, 'longitude': lon_vec},
    )
    self.counts_xr = xr.DataArray(
        counts_grid.T,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat_vec, 'longitude': lon_vec},
    )

  def estimate_beta_at_location_by_vicinity_to_sample(
      self,
      longitude,
      latitude,
      discard_few_event_locations = None,
  ):
    """Calculates an approximate beta for the query coordinates.

    Will calculate beta values on a rectangular grid covering the area of
    interest. Returns the value of the nearest grid point as the result.
    A quicker, less accurate, way to calculate the same as
    self._sample_smoothed_beta_value_for_query_coors
    Args:
      longitude: Sequence of longitudes to estimate the smoothed beta at.
      latitude: Sequence of latitudes to estimate the smoothed beta at.
      discard_few_event_locations: If given, will discard locations with less
        than this number of events.

    Returns:
      Array of beta values, length of the input sequences.
    """
    # Create sampling grid
    longitude_coords, latitude_coords = np.meshgrid(
        np.arange(
            np.min(longitude),
            np.max(longitude) + self.smoothing_distance / 2,
            self.smoothing_distance / 2,
        ),
        np.arange(
            np.min(latitude),
            np.max(latitude) + self.smoothing_distance / 2,
            self.smoothing_distance / 2,
        ),
    )
    longitude_coords, latitude_coords = (
        longitude_coords.ravel(),
        latitude_coords.ravel(),
    )

    # Map between query coordinates to grid sample coordinates
    ordered_grid_coords = []
    unique_grid_coords = []
    for estimate_lon, estimate_lat in tqdm.tqdm(
        zip(longitude, latitude),
        total=len(longitude),
        desc='Smoothing beta values',
        disable=(not self._display_progress),
    ):
      nearest_idx = np.argmin(
          (estimate_lon - longitude_coords) ** 2
          + (estimate_lat - latitude_coords) ** 2
      )
      grid_coord = (longitude_coords[nearest_idx], latitude_coords[nearest_idx])
      ordered_grid_coords.append(grid_coord)
      if grid_coord not in unique_grid_coords:
        unique_grid_coords.append(grid_coord)

    # Calculate relevant grid coordinates
    unique_lon, unique_lat = list(zip(*unique_grid_coords))
    beta_vals = self._sample_smoothed_beta_value_for_query_coors(
        unique_lon,
        unique_lat,
        discard_few_event_locations,
    )
    unique_coords_mapping = dict(zip(unique_grid_coords, beta_vals))
    return np.array([unique_coords_mapping[v] for v in ordered_grid_coords])

  def _sample_smoothed_beta_value_for_query_coors(
      self,
      longitude,
      latitude,
      discard_few_event_locations = None,
  ):
    """Calculates beta at given locations, with smoothing."""
    self._long_vec, self._lat_vec = np.meshgrid(
        self.beta_xr.longitude.values, self.beta_xr.latitude.values
    )
    discard_few_event_locations = (
        0
        if discard_few_event_locations is None
        else discard_few_event_locations
    )
    self._enough_samples_mask = (
        self.counts_xr.values >= discard_few_event_locations
    )
    smoothed_beta_values = []
    for estimate_lon, estimate_lat in tqdm.tqdm(
        zip(longitude, latitude),
        total=len(longitude),
        desc='Smoothing beta values',
        disable=(not self._display_progress),
    ):
      self._set_bins_logical_array(
          estimate_lat, estimate_lon, discard_few_event_locations
      )
      if not self._dist_logical.any():
        smoothed_beta_values.append(np.nan)
        continue
      intersections, betas = self._intersections_betas_for_estimate_coor(
          estimate_lon, estimate_lat
      )
      smoothed_beta_values.append(
          self._weighted_beta_value(betas, intersections)
      )
    return np.array(smoothed_beta_values)

  def _set_bins_logical_array(
      self, estimate_lat, estimate_lon, discard_few_event_locations
  ):
    """Sets the logical array of bins to use for smoothing."""
    dist_mat = np.sqrt(
        (self.beta_xr.latitude - estimate_lat) ** 2
        + (self.beta_xr.longitude - estimate_lon) ** 2
    )  # Array with distances of bin centers to querried cdoordinate
    self._dist_logical = (
        dist_mat <= ((self.grid_spacing / np.sqrt(2)) + self.smoothing_distance)
    ).values
    # Use only bins that beta was calculated with minimal number of events:
    if discard_few_event_locations is not None:
      self._dist_logical = self._dist_logical & self._enough_samples_mask

  def _weighted_beta_value(self, betas, intersections):
    return np.nansum(
        (np.array(intersections) * np.array(betas)) / np.nansum(intersections)
    )

  def _intersections_betas_for_estimate_coor(
      self, estimate_lon, estimate_lat
  ):
    """Returns the intersection area and beta values of relevant bins."""
    relevant_longs = self._long_vec[
        self._dist_logical & self._enough_samples_mask
    ]
    relevant_lats = self._lat_vec[
        self._dist_logical & self._enough_samples_mask
    ]
    relevant_betas = self.beta_xr.values[
        self._dist_logical & self._enough_samples_mask
    ]
    intersections = [np.nan]
    betas = [np.nan]
    # For each coordinate iterate over all adjacent bins with beta value:
    for grid_ln, grid_lt, beta in zip(
        relevant_longs, relevant_lats, relevant_betas
    ):
      intersection_area = circle_square_overlap(
          circle_center=(estimate_lon, estimate_lat),
          circle_radius=self.smoothing_distance,
          square_center=(grid_ln, grid_lt),
          square_side=self.grid_spacing,
      )
      if intersection_area == 0 or ~np.isfinite(beta):
        intersections.append(np.nan)
        betas.append(np.nan)
      else:
        intersections.append(intersection_area)
        betas.append(beta)
    return intersections, betas

  def count_events_in_radius(
      self, longitude, latitude
  ):
    distances_from_center = np.sqrt(
        (self.catalog.longitude.values - longitude) ** 2
        + (self.catalog.latitude.values - latitude) ** 2
    )
    return (distances_from_center <= self.smoothing_distance).sum()


def circle_square_overlap(
    circle_center,
    circle_radius,
    square_center,
    square_side,
    circle_resolution = 50,
):
  circle = shg.Point(*circle_center).buffer(circle_radius, circle_resolution)
  circle_scrap = shg.Point(*square_center).buffer(square_side)
  square = circle_scrap.from_bounds(*circle_scrap.bounds)
  return circle.intersection(square).area


def _calc_mc_at_coordinates(
    longitudes,
    latitudes,
    catalog,
    minimal_radius = 0.1,
    minimal_events = 100,
    completeness_calculator = None,
):
  """Returns completeness at points, with minimal surrounding events per point.

  Args:
    longitudes: Sequence of longitudes to estimate the completeness at.
    latitudes: Sequence of latitudes to estimate the completeness at.
    catalog: A catalog to use for the calculation.
    minimal_radius: The minimal radius to consider for the calculation.
    minimal_events: The minimal number of events to consider for the
      calculation.
    completeness_calculator: A callable that takes a sequence of magnitudes and
      returns a float (completeness magnitude). E.g.
      estimate_completeness_by_maximal_curvature (default).

  Returns:
    Three arrays, one for the number of events used in the calculation, one for
    the radius of the calculation, and one for the completeness magnitude.
  """

  if completeness_calculator is None:
    completeness_calculator = estimate_completeness_by_maximal_curvature
  reset_catalog = catalog.reset_index(inplace=False)

  number_of_events_list = []
  radius_of_calculation_list = []
  m_c_list = []
  for lon, lat in zip(longitudes, latitudes):
    radii = np.sqrt(
        (reset_catalog.longitude.values - lon) ** 2
        + (reset_catalog.latitude.values - lat) ** 2
    )
    min_radii_idxs = set(np.where(radii <= minimal_radius)[0])
    partitioned_radi = np.argpartition(radii, minimal_events)
    selected_idxs = list(
        set(partitioned_radi[:minimal_events]).union(min_radii_idxs)
    )
    near_catalog = reset_catalog.loc[selected_idxs]
    number_of_events = len(near_catalog)
    radius_of_calculation = radii[selected_idxs].max()
    m_c = completeness_calculator(near_catalog.magnitude.values)
    number_of_events_list.append(number_of_events)
    radius_of_calculation_list.append(radius_of_calculation)
    m_c_list.append(m_c)
  return (
      np.array(number_of_events_list),
      np.array(radius_of_calculation_list),
      np.array(m_c_list),
  )


def compute_grid_of_local_completeness(
    catalog,
    grid_spacing = 0.1,
    minimal_radius = 0.1,
    maximal_radius = 2,
    minimal_events = 100,
    completeness_calculator = None,
):
  """Computes a grid of local completeness."""
  if completeness_calculator is None:
    completeness_calculator = estimate_completeness_by_maximal_curvature
  longitudes = np.arange(
      catalog.longitude.values.min() + grid_spacing / 2,
      catalog.longitude.values.max() + grid_spacing,
      grid_spacing,
  )
  latitudes = np.arange(
      catalog.latitude.values.min() + grid_spacing / 2,
      catalog.latitude.values.max() + grid_spacing,
      grid_spacing,
  )
  longitudes_arr, latitudes_arr = np.meshgrid(longitudes, latitudes)
  number_of_events, radius_of_calculation, m_c = _calc_mc_at_coordinates(
      longitudes_arr.ravel(),
      latitudes_arr.ravel(),
      catalog,
      minimal_radius,
      minimal_events,
      completeness_calculator,
  )
  m_c[radius_of_calculation > maximal_radius] = np.nan
  mc_xr = xr.DataArray(
      m_c.reshape(longitudes_arr.shape),
      dims=['latitude', 'longitude'],
      coords={'latitude': latitudes, 'longitude': longitudes},
  )
  nevents_xr = xr.DataArray(
      number_of_events.reshape(longitudes_arr.shape),
      dims=['latitude', 'longitude'],
      coords={'latitude': latitudes, 'longitude': longitudes},
  )
  radius_xr = xr.DataArray(
      radius_of_calculation.reshape(longitudes_arr.shape),
      dims=['latitude', 'longitude'],
      coords={'latitude': latitudes, 'longitude': longitudes},
  )
  return mc_xr, nevents_xr, radius_xr
