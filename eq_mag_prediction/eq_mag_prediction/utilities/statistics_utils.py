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

"""Statistics and probability utility functions."""

# from collections.abc import Sequence, Type
from typing import Sequence, Type
import numpy as np
from scipy import stats
import tensorflow_probability as tfp
from eq_mag_prediction.forecasting import training_examples


DistributionType = Type[tfp.distributions.Distribution]
_SEARCH_RESOLUTION = 100


def probability_above_x(
    random_variable, x, shift = 0
):
  """The probability of an observation above x, int_x_inf(p(x))dx."""
  return 1 - random_variable.cdf(x - shift).numpy()


def probability_between_x1_x2(
    random_variable, x, shift = 0
):
  """The probability of an observation between x1 and x2, int_x1_x2(p(x))dx."""
  x_low = np.min(x)
  x_high = np.max(x)
  return (
      random_variable.cdf(x_high - shift).numpy()
      - random_variable.cdf(x_low - shift).numpy()
  )


def quantile_for_p_val(
    random_variable,
    p_value,
    support,
    shift = 0,
    acc = 1e-4,
    max_iterations = 1000,
):
  """Numerically compute x so that p_value=int_{x}^{inf} pdf(x') dx'.

  Args:
    random_variable: A tensorflow probability instance, child of
      tfp.distributions.Distribution
    p_value: The tails' requested integral. p_value=int_{x}^{inf} pdf(x') dx' .
    support:  Support of the non shifted given pdf.
    shift:  Shift of the pdf along the x-axis.
    acc:  Accuracy of x location required by user.
    max_iterations: maximal number of iterations for numerical process. If
      exceeded and no result has yet to be obtained np.nan is returned.

  Returns:
    np.ndarray of x that satisfies the tail weight by the given p_value, so that
    p_value=int_{x}^{inf} pdf(x') dx' .
  """

  # Number of separate distributions given in the distribution instance
  if not random_variable.batch_shape.as_list():
    random_variable = tfp.distributions.BatchReshape(random_variable, (1,))
  n_samples = random_variable.batch_shape[0]
  x_final = np.empty(n_samples)

  # set search margin in an initial reasonable guess:
  x_i = random_variable.mean().numpy()
  p_tail_x_i = probability_above_x(random_variable, x_i)
  has_converged = np.isclose(p_tail_x_i, p_value, atol=acc, rtol=0)
  x_final[has_converged] = x_i[has_converged]
  # NOMUTANTS the initial margin isn't critical and just a reasonable guess
  initial_margin = 15 * random_variable.stddev().numpy()
  x_boundaries = np.vstack([x_i - initial_margin, x_i + initial_margin])

  counter = 0
  # Iteretivaley find the required locations:
  while counter < max_iterations:
    counter += 1
    x_boundaries = np.clip(x_boundaries, np.min(support), np.max(support))
    # Set locations upon which to search for the relevant x:
    x_4_search = np.linspace(
        np.min(x_boundaries, axis=0),
        np.max(x_boundaries, axis=0),
        _SEARCH_RESOLUTION,
    )

    p_at_xs = probability_above_x(random_variable, x_4_search)
    p_idx = np.argmin(np.abs(p_at_xs - p_value), axis=0)
    x_i = x_4_search[p_idx, np.arange(n_samples)]

    p_tail_x_i = probability_above_x(random_variable, x_i)
    has_converged_now = np.isclose(p_tail_x_i, p_value, atol=acc, rtol=0)
    newly_converged = has_converged_now & (~has_converged)
    # Write the results for the distribuition which have converged:
    x_final[newly_converged] = x_i[newly_converged]
    has_converged = has_converged | newly_converged
    if np.all(has_converged):
      break

    # Reset x search locations
    x_start = x_4_search[np.maximum(p_idx - 1, 0), np.arange(n_samples)]
    x_start[p_idx == 0] = np.min(support)
    # NOMUTANTS margin update isn't critical and just a reasonable guess
    x_end = x_4_search[
        np.minimum(p_idx + 1, _SEARCH_RESOLUTION - 1), np.arange(n_samples)
    ]
    x_end[p_idx == _SEARCH_RESOLUTION - 1] = np.max(support)
    x_boundaries = [x_start, x_end]

  x_final[~has_converged] = np.nan
  return x_final + shift


def spatiotemporal_event_binning(
    examples,
    values,
    longitude_bins,
    latitude_bins,
    temporal_bins,
    statistic = 'mean',
    **histdd_kwargs,
):
  """Divides values associated with examples into spatiotemporal bins.

  Convention is longitude, latitude, temporal in that order.

  Args:
    examples: Events' examples as defined in training_examples module.
    values: The values associated with the events. E.g. energy of events,
      likelihood of events according to some model, etc.
    longitude_bins: Bin edges for longitudal division.
    latitude_bins: Bin edges for latitudal division.
    temporal_bins: Bin edges for temporal division.
    statistic:  statistic to compute for the output. See
      scipy.stats.binned_statistic_dd documentation for details.
    **histdd_kwargs: kwargs for the binned_statistic_dd function.

  Returns:
    Outputs of the binned_statistic_dd function.
  """

  if values is not None:
    assert len(examples) == len(values)
  else:
    values = np.ones(len(examples))
  examples_array = np.array(
      [(v[0][0].lng, v[0][0].lat, k) for k, v in examples.items()]
  )
  statistic, edges, bin_number = stats.binned_statistic_dd(
      sample=examples_array,
      values=values,
      bins=(longitude_bins, latitude_bins, temporal_bins),
      statistic=statistic,
      expand_binnumbers=True,
      **histdd_kwargs,
  )
  return statistic, edges, bin_number


def moving_avg_and_std_by_time_window(
    estimate_times,
    timestamps,
    values,
    window_size,
    weight_on_past,
    omit_nans = True,
):
  """Computes moving average and std by time window."""
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)
  moving_avg = []
  moving_std = []
  for t in estimate_times:
    last_index = np.searchsorted(
        timestamps,
        t + ((1 - weight_on_past) * window_size),
        side='left',
    )
    first_index = np.searchsorted(
        timestamps, t - (weight_on_past * window_size), side='left'
    )
    time_slice = slice(first_index, last_index)
    if first_index == last_index:
      moving_avg.append(np.nan)
      moving_std.append(np.nan)
      continue

    values_time_sliced = values[time_slice]
    if omit_nans:
      moving_avg.append(np.nanmean(values_time_sliced))
      moving_std.append(np.nanstd(values_time_sliced))
    else:
      moving_avg.append(values_time_sliced.mean())
      moving_std.append(values_time_sliced.std())
  return np.array(moving_avg), np.array(moving_std)


def moving_avg_and_std_by_sample_length(
    estimate_times,
    timestamps,
    values,
    window_size,
    weight_on_past,
    omit_nans = True,
):
  """Computes moving average and std by sample length."""
  assert window_size > 0
  weight_on_past = np.maximum(np.minimum(weight_on_past, 1.0), 0.0)

  moving_avg = []
  moving_std = []
  for t in estimate_times:
    n_events_past = int(np.round(window_size * weight_on_past))
    n_events_future = window_size - n_events_past

    past_time_logical = timestamps < t
    future_time_logical = timestamps >= t
    if (not np.any(past_time_logical)) and (not np.any(future_time_logical)):
      moving_avg.append(np.nan)
      moving_std.append(np.nan)
      continue

    past_idxs = np.where(past_time_logical)[0]
    future_idxs = np.where(future_time_logical)[0]

    last_past_idxs = past_idxs[len(past_idxs) - n_events_past : len(past_idxs)]
    first_future_idxs = future_idxs[:n_events_future]

    values_to_use_past = values[last_past_idxs]
    values_to_use_future = values[first_future_idxs]
    values_to_use = np.concatenate((values_to_use_past, values_to_use_future))
    if omit_nans:
      moving_avg.append(np.nanmean(values_to_use))
      moving_std.append(np.nanstd(values_to_use))
    else:
      moving_avg.append(values_to_use.mean())
      moving_std.append(values_to_use.std())
  return np.array(moving_avg), np.array(moving_std)
