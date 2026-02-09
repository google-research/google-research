# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name

import datetime

from absl import app
import pandas as pd

from constant_defs import HORIZONS
from constant_defs import QUANTILES
from constant_defs import REQUIRED_CDC_LOCATIONS
from constant_defs import TARGET_STR
from epi_utils import compute_rolling_evaluation
from epi_utils import format_for_cdc
from epi_utils import get_most_recent_saturday_date_str
from epi_utils import get_next_saturday_date_str
from epi_utils import get_saturdays_between_dates
from plotting_utils import plot_season_forecasts

timedelta = datetime.timedelta


INPUT_DIR = ''
MODEL_NAME = 'Google_SAI-Hybrid_3'
TARGET_STR = ''

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm

# QUANTILE_COLUMNS relies on the global QUANTILES defined in the preamble.
QUANTILE_COLUMNS = [f'quantile_{q}' for q in QUANTILES]


def lowpass_filter_data(
    data, cutoff_freq, order, fs
):
  """Applies a Butterworth low-pass filter to a 1D array to smooth time series data.

  Ensures numerical stability for small datasets based on scipy.signal.filtfilt
  requirements. This function is a core component inspired by Code 1's signal
  processing for robust R_t estimation.

  Args:
      data (np.ndarray): The time series data.
      cutoff_freq (float): The cutoff frequency of the filter (normalized to
        Nyquist).
      order (int): The order of the filter.
      fs (float): The sampling frequency of the data (e.g., 1.0 for weekly
        data).

  Returns:
      np.ndarray: The filtered data.
  """
  # If data is too short for meaningful filtering or filter order, return original data
  # filtfilt typically requires len(x) > 3*order for stable results with padding.
  # At minimum, len(data) must be > order for butter to not raise warnings/errors.
  if (
      len(data) < order + 2
  ):  # At least order + 2 points for a meaningful filter of order 'order'
    return data.astype(float)

  nyq = 0.5 * fs
  # Ensure cutoff is within (0, Nyquist) and normalized to (0, 1) for butter.
  normal_cutoff = min(0.99, max(0.01, cutoff_freq / nyq))

  # If the normalized cutoff is not strictly between 0 and 1, filter design might fail or be meaningless.
  if not (0 < normal_cutoff < 1):
    return data.astype(float)  # Return original data if cutoff is problematic

  try:
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # filtfilt requires padlen < len(x) - 1. A common heuristic for padlen is 3*order.
    # Ensure padlen does not exceed len(data) - 2 and is non-negative.
    padlen_actual = min(order * 3, len(data) - 2)
    padlen_actual = max(0, padlen_actual)  # Ensure non-negative

    filtered_data = filtfilt(b, a, data, padtype='odd', padlen=padlen_actual)
  except (
      ValueError
  ):  # Catch potential errors from butter or filtfilt (e.g., invalid filter design for extreme parameters)
    filtered_data = data.astype(float)
  return filtered_data


def weighted_percentile(
    data, weights, quantiles
):
  """Calculates weighted percentiles for a given dataset.

  This implements the principle of direct quantile estimation from Code 2,
  enhanced with weighting to give more importance to relevant historical data
  points.

  Args:
      data (np.ndarray): The input data.
      weights (np.ndarray): Corresponding weights for each data point.
      quantiles (list[float]): A list of quantile levels (e.0.01, 0.5, 0.99).

  Returns:
      np.ndarray: An array of calculated quantiles.
  """
  data = np.asarray(data)
  weights = np.asarray(weights)

  # Filter out NaN values from data and weights, and ensure weights are positive
  valid_indices = (
      ~np.isnan(data) & ~np.isnan(weights) & (weights > 1e-9)
  )  # Use 1e-9 for numerical stability
  data = data[valid_indices]
  weights = weights[valid_indices]

  if (
      len(data) == 0 or np.sum(weights) <= 1e-9
  ):  # Use 1e-9 for numerical stability
    # Return 0.0 for all quantiles if no valid data or weights
    return np.full(len(quantiles), 0.0)

  # Sort data and weights by data values
  sorted_indices = np.argsort(data)
  sorted_data = data[sorted_indices]
  sorted_weights = weights[sorted_indices]

  cumulative_weights = np.cumsum(sorted_weights)
  total_weight = cumulative_weights[-1]

  if (
      total_weight <= 1e-9
  ):  # Additional check for total weight being effectively zero
    return np.full(len(quantiles), 0.0)

  normalized_cumulative_weights = cumulative_weights / total_weight

  # Use interpolation to find the quantile values
  results = np.interp(quantiles, normalized_cumulative_weights, sorted_data)
  return results


def _weighted_average(values, weights):
  """Calculates the weighted average of an array."""
  # Filter out NaN values from values and corresponding weights, and ensure weights are positive
  valid_indices = (
      ~np.isnan(values) & ~np.isnan(weights) & (weights > 1e-9)
  )  # Use 1e-9 for numerical stability
  values = values[valid_indices]
  weights = weights[valid_indices]

  if (
      len(values) == 0 or np.sum(weights) <= 1e-9
  ):  # Use 1e-9 for numerical stability
    return 0.0  # Default to zero if no valid values or weights
  return np.sum(values * weights) / np.sum(weights)


def _weighted_std_dev(values, weights):
  """Calculates the weighted standard deviation of an array."""
  # Filter out NaN values from values and corresponding weights, and ensure weights are positive
  valid_indices = (
      ~np.isnan(values) & ~np.isnan(weights) & (weights > 1e-9)
  )  # Use 1e-9 for numerical stability
  values = values[valid_indices]
  weights = weights[valid_indices]

  if (
      len(values) == 0 or np.sum(weights) <= 1e-9
  ):  # Use 1e-9 for numerical stability
    return 0.0  # Default to zero if no valid values or weights

  mean = _weighted_average(values, weights)
  # Weighted variance, ensure it's non-negative before sqrt
  variance = _weighted_average((values - mean) ** 2, weights)
  return np.sqrt(max(0.0, variance))


# --- Core Principles of the Hybrid Strategy ---
# 1. Trend-Aware Mechanistic Component (Bayesian-inspired R_t & Log-Growth Forecasting with Monte Carlo): This component
#    estimates an effective log-growth rate (log(R_t_effective)) from recent smoothed historical hospitalization data
#    using weighted linear regression. Crucially, this data-driven estimate is then adaptively blended with a prior
#    expectation of stable disease transmission (log(R_t_effective) = 0, i.e., R_t_effective = 1) based on the reliability
#    of the recent observations. Future cases are projected through Monte Carlo simulations, where the blended
#    log(R_t_effective) gradually decays towards the endemic prior mean and incorporates stochastic process noise
#    (both data-driven and prior-informed) at each step. This process simulates the dynamic evolution of the epidemic,
#    conceptually driven by an R_t parameter and its associated uncertainty, akin to a simplified renewal equation
#    I_t = R_t * I_{t-1} for weekly steps.
# 2. Climatological/Seasonal Component (Weighted Historical Averaging): This component leverages recurring
#    seasonal patterns by identifying analogous historical periods (same week of year, surrounding weeks).
#    It directly computes empirical quantiles from past observations, robustly combining location-specific
#    historical data with broader geo-aggregated case rates. Weights are applied to prioritize more recent years
#    and closer seasonal weeks, along with dynamic blending based on data availability, ensuring the climatological
#    forecast is relevant and robust.
# 3. Adaptive Log-Additive Blending Strategy: The model intelligently combines the probabilistic outputs of the
#    mechanistic and climatological components. A dynamic blending weight, which considers both the forecast
#    horizon and the reliability (magnitude and volatility) of the estimated recent trend, is applied. This blending
#    occurs on log1p-transformed quantiles, enabling a combination that is approximately additive for low counts
#    and multiplicative for high counts, effectively leveraging the strengths of each component across different
#    epidemic states.
# 4. Robust Quantile Generation and Post-processing: Ensures that all generated probabilistic forecasts are
#    numerically stable, adhere to non-negativity constraints, and strictly maintain monotonic increasing
#    order across quantile levels (e.g., 0.01 <= 0.025 <= ... <= 0.99), even after aggregation and blending,
#    thereby providing a well-formed and valid set of predictions.


def fit_and_predict_fn(
    train_x, train_y, test_x
):
  """Makes probabilistic forecasts of COVID-19 hospital admissions using a robust hybrid strategy.

  This model intelligently blends two fundamentally different approaches: 1.
  **Trend-Aware Mechanistic Component (Monte Carlo Simulation):** Estimates a
  recent log-growth factor

      (a proxy for the effective reproduction number, R_t) from smoothed
      historical data. It projects
      future cases by running multiple Monte Carlo simulations, where the
      log-growth rate decays
      towards an an endemic level and incorporates stochastic process noise at
      each step, generating
      a distribution of outcomes. This component now uses weighted linear
      regression for more robust
      trend and noise estimation, with enhanced uncertainty for low counts.
  2.  **Climatological/Seasonal Component:** Leverages historical seasonal
  patterns by identifying
      analogous weeks from past years. It calculates empirical quantiles from
      these observations,
      incorporating both location-specific and geo-aggregated data, weighted by
      recency and
      seasonal proximity.

  The key innovation is a dynamic log-additive blending strategy that combines
  these two components.
  The blending weight adjusts based on the forecast horizon and the reliability
  (magnitude and volatility)
  of the recent trend, ensuring the model leverages the strengths of each
  approach in different scenarios.

  Args:
      train_x (pd.DataFrame): Training features (e.g., location, population,
        dates).
      train_y (pd.Series): Training target values (Total COVID-19 Admissions).
      test_x (pd.DataFrame): Test features for future time periods to predict.

  Returns:
      pd.DataFrame: A DataFrame with quantile predictions for each row in
      test_x,
                    with columns named 'quantile_0.01', ..., 'quantile_0.99'.
  """
  # Define default configuration parameters, based on insights from previous trials and minor adjustments.
  # These defaults are chosen for robust performance and as a solid starting point for tuning.
  conf = {
      # --- Data Preprocessing & Smoothing ---
      'smoothing_cutoff_freq': (
          0.20
      ),  # Butterworth filter cutoff frequency (normalized to Nyquist, 0-0.5).
      # Higher value means less smoothing, lower means more.
      'smoothing_order': (
          2
      ),  # Order of Butterworth filter. Higher order means sharper cutoff.
      # --- Mechanistic Component (Log-Growth Factor / R_t & Trend) ---
      'trend_window_weeks': (
          4
      ),  # Number of recent weeks to consider for log-growth trend estimation.
      'log_growth_trend_decay_factor': (
          0.8
      ),  # Decay factor for weighting recent log-growth rates in regression (1.0 for no decay, <1.0 for more recent data emphasis).
      'min_admissions_for_growth_calc': (
          10
      ),  # Minimum smoothed admissions (sum over window) to calculate a meaningful growth trend. Below this, trend is assumed flat.
      'log_growth_proxy_min': (
          -0.7
      ),  # Min clip for log-growth factor (corresponds to weekly change of ~-50%). Prevents extreme drops.
      'log_growth_proxy_max': (
          0.9
      ),  # Max clip for log-growth factor (corresponds to weekly change of ~145%). Prevents extreme surges.
      'min_std_log_growth': (
          0.04
      ),  # Minimum standard deviation for log growth rates (used as process noise). Prevents overconfidence.
      'low_count_noise_factor': (
          0.5
      ),  # Factor to add extra noise for very low admission counts. Higher for more uncertainty.
      'max_low_count_noise_contribution': (
          0.2
      ),  # Maximum value the additional noise from low counts can contribute. Prevents explosive noise.
      'num_simulations_mechanistic': (
          1000
      ),  # Number of Monte Carlo simulations for the mechanistic component.
      'log_growth_decay_horizon': (
          4
      ),  # Weeks over which log-growth rate decays significantly towards the endemic rate.
      'endemic_log_growth_rate': (
          0.0
      ),  # Log-growth rate the model tends towards (e.g., 0.0 for R_t=1, stability).
      # --- Bayesian-inspired Prior for R_t ---
      'prior_log_growth_mean': (
          0.0
      ),  # Prior mean for log(R_t_effective), representing R_t=1 (stability).
      'prior_log_growth_std': (
          0.15
      ),  # Prior standard deviation for log(R_t_effective).
      # Wider for more uncertainty, narrower for stronger belief in R_t=1.
      # --- Climatological Component ---
      'min_year_for_climatology': (
          2020
      ),  # Earliest year to include in historical climatology.
      'climatology_window_size_weeks': (
          3
      ),  # Number of weeks on each side of target week for climatology.
      'climatology_smoothing_factor': (
          0.50
      ),  # Factor to blend climatological quantiles towards zero (0 to 1). Higher value pulls towards zero.
      'climatology_year_weight_decay': (
          0.60
      ),  # Decay factor for older years in climatology (1.0 for no decay, 0.0 for extreme decay).
      'climatology_week_spread_std': (
          2.0
      ),  # Std dev for Gaussian weighting of weeks in climatology window. Smaller value means tighter week focus.
      'min_data_points_for_climatology': (
          2
      ),  # Minimum data points for valid percentile calculation in climatology.
      'climatology_geo_blend_factor': (
          0.5
      ),  # Weight for blending geo-specific vs geo-aggregated climatology (0-1, 0.5 for equal).
      'min_geo_specific_data_for_full_blend': (
          10
      ),  # Min data points for full geo-specific climatology weight, below which it reduces.
      # --- Hybrid Blending ---
      'damping_factor': (
          0.85
      ),  # Factor for blending weight decay over forecast horizon. Higher value maintains trend influence longer.
      'trend_reliability_threshold_multiplier': (
          1.0
      ),  # Multiplier for min_admissions_for_growth_calc to set threshold for trend reliability based on magnitude.
      'max_acceptable_std_log_growth': (
          0.15
      ),  # Max std dev of log growth for full trend reliability (below this, no penalty).
      'std_log_growth_penalty_factor': (
          1.2
      ),  # Factor to penalize trend reliability when std_log_growth exceeds max_acceptable_std_log_growth.
  }

  # 1. Data Preparation and Smoothing (Leveraging Code 1 principles: Low-pass filtering)
  train_df = train_x.copy()
  train_df['Total COVID-19 Admissions'] = train_y

  # Ensure date columns are in datetime format
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  train_df = train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  train_df['week_of_year'] = (
      train_df['target_end_date'].dt.isocalendar().week.astype(int)
  )
  train_df['year'] = train_df['target_end_date'].dt.year.astype(int)

  # Robustly handle population for case rate calculation: replace 0 population with NaN for division, then fillna with 0 for case_rate
  train_df.loc[:, 'population_effective'] = (
      train_df['population'].replace(0, np.nan).astype(float)
  )
  # Calculate case rate per 100,000 population, handling potential NaN from zero population
  train_df.loc[:, 'case_rate'] = train_df['Total COVID-19 Admissions'] / (
      train_df['population_effective'] / 100000
  )
  train_df.loc[:, 'case_rate'] = train_df['case_rate'].fillna(
      0
  )  # Fill NaN case rates (from zero population) with 0

  # Apply smoothing to raw admissions, handling potential errors from lowpass_filter_data
  # Convert to float and clamp at 0 to ensure non-negative smoothed values
  train_df['smoothed_admissions'] = (
      train_df.groupby('location')['Total COVID-19 Admissions']
      .transform(
          lambda x: lowpass_filter_data(
              x.to_numpy(),
              conf['smoothing_cutoff_freq'],
              conf['smoothing_order'],
              fs=1.0,
          )
      )
      .astype(float)
  )
  train_df['smoothed_admissions'] = np.maximum(
      0, train_df['smoothed_admissions']
  )

  # Apply smoothing to case rates for geo-aggregated climatology
  # Convert to float and clamp at 0 to ensure non-negative smoothed values
  train_df['smoothed_case_rate'] = (
      train_df.groupby('location')['case_rate']
      .transform(
          lambda x: lowpass_filter_data(
              x.to_numpy(),
              conf['smoothing_cutoff_freq'],
              conf['smoothing_order'],
              fs=1.0,
          )
      )
      .astype(float)
  )
  train_df['smoothed_case_rate'] = np.maximum(0, train_df['smoothed_case_rate'])

  # Pre-calculate year weights for climatology (optimization moved outside the test_x loop)
  climatology_data_all_years = train_df[
      train_df['year'] >= conf['min_year_for_climatology']
  ].copy()
  if not climatology_data_all_years.empty:
    max_year_in_climatology = climatology_data_all_years['year'].max()
    climatology_data_all_years.loc[:, 'year_weight'] = conf[
        'climatology_year_weight_decay'
    ] ** (max_year_in_climatology - climatology_data_all_years['year'])
  else:
    # If no data available for climatology years, set year_weight to 0 for all rows
    climatology_data_all_years.loc[:, 'year_weight'] = 0.0

  # 2. Mechanistic Trend and Uncertainty Calculation (Bayesian-inspired Log-Growth Factor / R_t for Monte Carlo)
  location_trend_data = {}

  for loc_id in train_df['location'].unique():
    loc_data = train_df[train_df['location'] == loc_id].copy()
    loc_data = loc_data.sort_values(by='target_end_date').reset_index(drop=True)

    last_known_smoothed_admissions = 0.0
    last_known_date_for_location = pd.NaT
    # Initialize with prior values for robustness
    estimated_log_R_t_effective = conf['prior_log_growth_mean']
    std_log_growth_process_noise = conf[
        'prior_log_growth_std'
    ]  # Start with prior std

    if not loc_data.empty:
      last_known_smoothed_admissions = loc_data['smoothed_admissions'].iloc[-1]
      last_known_date_for_location = loc_data['target_end_date'].iloc[-1]

      # Calculate reliability factor based on last known admissions, similar to blending weight
      # This 'current_reliability_factor' now serves to blend data-driven trend with prior.
      trend_reliability_threshold_for_prior = (
          conf['min_admissions_for_growth_calc']
          * conf['trend_reliability_threshold_multiplier']
      )
      current_reliability_factor_for_prior = 1.0
      if last_known_smoothed_admissions < trend_reliability_threshold_for_prior:
        current_reliability_factor_for_prior = (
            last_known_smoothed_admissions
            / trend_reliability_threshold_for_prior
        )
        current_reliability_factor_for_prior = max(
            0.0, min(1.0, current_reliability_factor_for_prior)
        )

      # Use a window of `trend_window_weeks` for robust growth rate estimation
      trend_data_window_smoothed = (
          loc_data['smoothed_admissions']
          .tail(conf['trend_window_weeks'] + 1)
          .to_numpy()
      )

      data_driven_log_growth = conf[
          'prior_log_growth_mean'
      ]  # Default to prior if no data
      data_driven_std_log_growth = conf[
          'prior_log_growth_std'
      ]  # Default to prior std

      if (
          len(trend_data_window_smoothed) >= 2
          and np.sum(trend_data_window_smoothed)
          >= conf['min_admissions_for_growth_calc']
      ):

        ys = np.log1p(trend_data_window_smoothed)
        xs = np.arange(len(ys))  # Time indices

        # Assign weights: more recent data gets higher weight
        weights_lr = conf['log_growth_trend_decay_factor'] ** (xs[::-1])
        weights_lr = np.maximum(
            1e-9, weights_lr
        )  # Ensure small positive weight for polyfit

        if (
            np.sum(weights_lr) > 1e-9
            and len(np.unique(xs[weights_lr > 1e-9])) >= 2
        ):
          try:
            coeffs = np.polyfit(xs, ys, 1, w=weights_lr)
            data_driven_log_growth = coeffs[
                0
            ]  # The slope is the log-growth rate

            predicted_ys = np.polyval(coeffs, xs)
            residuals = ys - predicted_ys
            data_driven_std_log_growth = _weighted_std_dev(
                residuals, weights_lr
            )
            data_driven_std_log_growth = max(
                data_driven_std_log_growth, conf['min_std_log_growth']
            )

          except (np.linalg.LinAlgError, ValueError):
            # Fallback if regression encounters issues, use prior-influenced defaults
            pass  # data_driven_log_growth and data_driven_std_log_growth remain their prior-influenced defaults
      # else: defaults to prior-influenced values already set.

      # Bayesian-inspired blending: Blend data-driven estimate with prior based on data reliability
      estimated_log_R_t_effective = (
          current_reliability_factor_for_prior * data_driven_log_growth
          + (1 - current_reliability_factor_for_prior)
          * conf['prior_log_growth_mean']
      )

      # Combine std devs - use a weighted average or simply take the max for robustness.
      # A more principled way might combine variances (1/var_post = 1/var_data + 1/var_prior)
      # but for simplicity, use a weighted blend, ensuring a minimum.
      std_log_growth_process_noise = (
          current_reliability_factor_for_prior * data_driven_std_log_growth
          + (1 - current_reliability_factor_for_prior)
          * conf['prior_log_growth_std']
      )
      std_log_growth_process_noise = max(
          std_log_growth_process_noise, conf['min_std_log_growth']
      )

    # Apply clipping to mean_log_growth (R_t) to prevent extreme values, even after blending
    estimated_log_R_t_effective = np.clip(
        estimated_log_R_t_effective,
        conf['log_growth_proxy_min'],
        conf['log_growth_proxy_max'],
    )

    # --- Enhancement: Increase std_log_growth_process_noise for very low counts ---
    # This is an additional noise factor, applied on top of blended std_log_growth.
    if (
        last_known_smoothed_admissions
        < conf['min_admissions_for_growth_calc'] * 2
    ):
      additional_noise_from_low_counts = conf[
          'low_count_noise_factor'
      ] / np.sqrt(last_known_smoothed_admissions + 1e-6)
      additional_noise_from_low_counts = min(
          additional_noise_from_low_counts,
          conf['max_low_count_noise_contribution'],
      )
      std_log_growth_process_noise = max(
          std_log_growth_process_noise, additional_noise_from_low_counts
      )
    # --- End Enhancement ---

    location_trend_data[loc_id] = {
        'last_known_smoothed_admissions': last_known_smoothed_admissions,
        'last_known_date_for_location': last_known_date_for_location,
        'estimated_log_R_t_effective': (
            estimated_log_R_t_effective
        ),  # Renamed for conceptual clarity
        'std_log_growth_process_noise': std_log_growth_process_noise,
    }

  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=QUANTILE_COLUMNS
  )

  def get_cyclical_week_distance(week1, week2):
    """Calculates the shortest cyclical distance between two week numbers."""
    max_weeks_in_year = 53  # Standard maximum for ISO week calendar
    diff = abs(week1 - week2)
    return min(diff, max_weeks_in_year - diff)

  # 3. Hybrid Blending for Forecasting
  for idx, row in test_x.iterrows():
    target_date = row['target_end_date']
    location_id = row['location']
    target_population = row['population']
    target_week_of_year = target_date.isocalendar().week

    loc_trend = location_trend_data.get(
        location_id,
        {
            'last_known_smoothed_admissions': 0.0,
            'last_known_date_for_location': pd.NaT,
            'estimated_log_R_t_effective': conf[
                'prior_log_growth_mean'
            ],  # Default to prior if no data
            'std_log_growth_process_noise': conf[
                'prior_log_growth_std'
            ],  # Default to prior std
        },
    )
    last_known_smoothed_admissions = loc_trend['last_known_smoothed_admissions']
    last_known_date_for_location = loc_trend['last_known_date_for_location']
    estimated_log_R_t_effective_initial = loc_trend[
        'estimated_log_R_t_effective'
    ]
    std_log_growth_process_noise = loc_trend['std_log_growth_process_noise']

    # --- Mechanistic Projection (Monte Carlo Simulation) ---
    # Start log admissions from the last known smoothed value, using log1p
    base_log1p_admissions = np.log1p(last_known_smoothed_admissions)

    steps_to_project = 0
    if pd.notna(last_known_date_for_location):
      # Calculate steps based on weeks between last known and target, rounding to nearest week
      steps_to_project = round(
          (target_date - last_known_date_for_location).days / 7
      )
    steps_to_project = max(0, steps_to_project)  # Ensure non-negative steps

    # Initialize Monte Carlo trajectories
    simulated_log1p_admissions_trajectories = np.full(
        conf['num_simulations_mechanistic'], base_log1p_admissions
    )

    # For each step in the forecast horizon
    for k_step in range(1, steps_to_project + 1):
      # Calculate the target mean log growth rate for this step, decaying towards endemic rate (log(R_t=1)=0)
      current_step_target_log_R_t_effective = conf[
          'endemic_log_growth_rate'
      ] + (
          estimated_log_R_t_effective_initial - conf['endemic_log_growth_rate']
      ) * np.exp(
          -k_step / conf['log_growth_decay_horizon']
      )

      # Add process noise to the log growth rate for each simulation in this step
      noise_for_step = norm.rvs(
          loc=0,
          scale=std_log_growth_process_noise,
          size=conf['num_simulations_mechanistic'],
      )

      sim_log_R_t_effective_at_step = (
          current_step_target_log_R_t_effective + noise_for_step
      )

      # Clip the individual simulation's log growth rates to reasonable bounds
      clipped_sim_log_R_t_effective = np.clip(
          sim_log_R_t_effective_at_step,
          conf['log_growth_proxy_min'],
          conf['log_growth_proxy_max'],
      )

      # Apply the growth to the simulated log1p admissions (akin to I_t = I_{t-1} * R_t_effective)
      simulated_log1p_admissions_trajectories += clipped_sim_log_R_t_effective

    # Convert simulated log1p admissions back to admissions using expm1 and ensure non-negativity
    simulated_admissions_trajectories = np.expm1(
        simulated_log1p_admissions_trajectories
    )
    simulated_admissions_trajectories = np.maximum(
        0, simulated_admissions_trajectories
    )

    # Calculate empirical quantiles from the simulated trajectories
    mechanistic_q_values = np.percentile(
        simulated_admissions_trajectories, [q * 100 for q in QUANTILES]
    )
    # Convert to log1p-quantiles for log-additive blending, handling zero values
    mechanistic_log_q_values = np.log1p(mechanistic_q_values)

    # --- Climatological Quantile Estimation (Optimized) ---
    climatology_weeks = []
    for i in range(
        -conf['climatology_window_size_weeks'],
        conf['climatology_window_size_weeks'] + 1,
    ):
      adjusted_week = target_week_of_year + i
      # Handle cyclical nature of week numbers (e.g., week 53 to week 1, week 1 to week 53)
      if adjusted_week > 53:
        adjusted_week -= 53
      elif adjusted_week < 1:
        adjusted_week += 53
      climatology_weeks.append(adjusted_week)
    climatology_weeks = list(
        set([w for w in climatology_weeks if 1 <= w <= 53])
    )  # Ensure unique and valid weeks

    # Filter the pre-calculated climatology data for the relevant weeks (optimization)
    climatology_data_windowed_for_target_week = climatology_data_all_years[
        climatology_data_all_years['week_of_year'].isin(climatology_weeks)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning during weight assignments

    # Initialize base climatological quantiles to 0
    base_climatological_quantiles = np.full(len(QUANTILES), 0.0)

    if not climatology_data_windowed_for_target_week.empty:
      # Apply week-of-year Gaussian weighting: weeks closer to target week get higher weight
      climatology_data_windowed_for_target_week.loc[:, 'week_distance'] = (
          climatology_data_windowed_for_target_week['week_of_year'].apply(
              lambda x: get_cyclical_week_distance(x, target_week_of_year)
          )
      )
      effective_week_spread_std = max(
          1e-9, conf['climatology_week_spread_std']
      )  # Ensure positive std
      climatology_data_windowed_for_target_week.loc[
          :, 'week_of_year_weight'
      ] = np.exp(
          -0.5
          * (
              climatology_data_windowed_for_target_week['week_distance']
              / effective_week_spread_std
          )
          ** 2
      )

      # Combine pre-calculated year_weight with current week_of_year_weight
      climatology_data_windowed_for_target_week.loc[:, 'total_weight'] = (
          climatology_data_windowed_for_target_week['year_weight']
          * climatology_data_windowed_for_target_week['week_of_year_weight']
      )

      climatology_data_windowed_for_target_week.loc[:, 'total_weight'] = (
          np.maximum(
              0, climatology_data_windowed_for_target_week['total_weight']
          )
      )  # Ensure non-negative weights

      geo_specific_data_filtered = climatology_data_windowed_for_target_week[
          climatology_data_windowed_for_target_week['location'] == location_id
      ]
      geo_specific_values = geo_specific_data_filtered[
          'smoothed_admissions'
      ].to_numpy()
      geo_specific_weights = geo_specific_data_filtered[
          'total_weight'
      ].to_numpy()
      geo_specific_q = weighted_percentile(
          geo_specific_values, geo_specific_weights, QUANTILES
      )

      geo_aggregated_values = climatology_data_windowed_for_target_week[
          'smoothed_case_rate'
      ].to_numpy()
      geo_aggregated_weights = climatology_data_windowed_for_target_week[
          'total_weight'
      ].to_numpy()
      geo_aggregated_q_rate = weighted_percentile(
          geo_aggregated_values, geo_aggregated_weights, QUANTILES
      )

      # Robust population handling for scaling: Ensure target_population is at least 1 to avoid division/multiplication by zero
      effective_target_pop_for_scaling = max(1, target_population)
      geo_aggregated_q = geo_aggregated_q_rate * (
          effective_target_pop_for_scaling / 100000
      )

      # Blend geo-specific and geo-aggregated quantiles based on data availability (dynamic blending)
      is_geo_specific_valid = (
          np.sum(geo_specific_weights) > 1e-9
          and len(geo_specific_values)
          >= conf['min_data_points_for_climatology']
      )
      is_geo_aggregated_valid = (
          np.sum(geo_aggregated_weights) > 1e-9
          and len(geo_aggregated_values)
          >= conf['min_data_points_for_climatology']
      )

      effective_climatology_geo_blend_factor = conf[
          'climatology_geo_blend_factor'
      ]
      if (
          is_geo_specific_valid
          and len(geo_specific_values)
          < conf['min_geo_specific_data_for_full_blend']
      ):
        # Linearly reduce geo-specific weight if its data is sparse
        blend_reduction_factor = (
            len(geo_specific_values)
            / conf['min_geo_specific_data_for_full_blend']
        )
        effective_climatology_geo_blend_factor *= blend_reduction_factor
        effective_climatology_geo_blend_factor = max(
            0.0, min(1.0, effective_climatology_geo_blend_factor)
        )

      if is_geo_specific_valid and is_geo_aggregated_valid:
        base_climatological_quantiles = (
            effective_climatology_geo_blend_factor * geo_specific_q
            + (1 - effective_climatology_geo_blend_factor) * geo_aggregated_q
        )
      elif is_geo_specific_valid:
        base_climatological_quantiles = geo_specific_q
      elif is_geo_aggregated_valid:
        base_climatological_quantiles = geo_aggregated_q
      # else: remains np.full(len(QUANTILES), 0.0) from initialization if neither is valid

    # Apply smoothing factor for climatological quantiles, pulling them towards zero
    base_climatological_quantiles = base_climatological_quantiles * (
        1 - conf['climatology_smoothing_factor']
    )
    base_climatological_quantiles = np.maximum(
        0, base_climatological_quantiles
    )  # Ensure non-negativity

    # --- Hybrid Blending ---
    # Calculate reliability factors for the mechanistic trend component
    # Reliability based on the magnitude of recent admissions
    trend_reliability_threshold = (
        conf['min_admissions_for_growth_calc']
        * conf['trend_reliability_threshold_multiplier']
    )
    current_reliability_factor_for_blending = 1.0
    if last_known_smoothed_admissions < trend_reliability_threshold:
      current_reliability_factor_for_blending = (
          last_known_smoothed_admissions / trend_reliability_threshold
      )
      current_reliability_factor_for_blending = max(
          0.0, min(1.0, current_reliability_factor_for_blending)
      )  # Clamp between 0 and 1

    # Reliability based on the volatility of the estimated growth rate
    volatility_reliability_factor = 1.0
    if std_log_growth_process_noise > conf['max_acceptable_std_log_growth']:
      # Penalize reliability if log-growth standard deviation is too high (indicating noisy/unreliable trend)
      penalty_exponent = conf['std_log_growth_penalty_factor'] * (
          std_log_growth_process_noise - conf['max_acceptable_std_log_growth']
      )
      volatility_reliability_factor = np.exp(-penalty_exponent)
      volatility_reliability_factor = max(
          0.0, min(1.0, volatility_reliability_factor)
      )
    else:
      volatility_reliability_factor = 1.0

    # Blending weight for the mechanistic model: decays with horizon and affected by reliability
    blending_weight = (
        (conf['damping_factor'] ** steps_to_project)
        * current_reliability_factor_for_blending
        * volatility_reliability_factor
    )
    blending_weight = max(
        0.0, min(1.0, blending_weight)
    )  # Clamp blending weight

    predicted_quantiles_for_row = []
    for i in range(len(QUANTILES)):
      C_q = base_climatological_quantiles[i]
      # Use log1p for climatological quantiles for consistency
      log1p_C_q = np.log1p(C_q)

      log1p_M_q = mechanistic_log_q_values[
          i
      ]  # This is already log1p-transformed from earlier step

      # Log-additive blending of the two components (on log1p scale)
      final_log1p_q = (
          1 - blending_weight
      ) * log1p_C_q + blending_weight * log1p_M_q

      final_q = np.expm1(final_log1p_q)  # Reverse with expm1
      predicted_quantiles_for_row.append(final_q)

    predicted_quantiles_for_row = np.array(predicted_quantiles_for_row)
    predicted_quantiles_for_row = np.maximum(
        0, predicted_quantiles_for_row
    )  # Ensure non-negativity

    # Round to nearest integer and convert to int
    predicted_quantiles_for_row_int = np.round(
        predicted_quantiles_for_row
    ).astype(int)

    # Ensure monotonicity explicitly after rounding and converting to int
    # This is crucial to satisfy the output format requirement.
    predicted_quantiles_for_row_int = np.maximum.accumulate(
        predicted_quantiles_for_row_int
    )

    test_y_hat_quantiles.loc[idx, QUANTILE_COLUMNS] = (
        predicted_quantiles_for_row_int
    )

  return test_y_hat_quantiles


def main(argv):
  del argv  # Unused.
  locations = locations[locations['location'].isin(REQUIRED_CDC_LOCATIONS)]
  locations['location'] = locations['location'].astype(int)
  location_codes = locations['location'].unique()

  print('Locations sample:')
  print(locations.head())

  dataset = pd.read_csv(f'{INPUT_DIR}/dataset.csv')
  dataset['target_end_date'] = pd.to_datetime(
      dataset['target_end_date']
  ).dt.date

  print('Dataset sample (check for existence of most recent data):')
  print(dataset.sort_values(by=['target_end_date'], ascending=False).head())

  dataset['Total Influenza Admissions'] = (
      pd.to_numeric(dataset['Total Influenza Admissions'], errors='coerce')
      .replace({np.nan: np.nan})
      .astype('Int64')
  )

  # --- Execute Validation Run ---
  print('--- Starting Validation Run ---')
  # Define validation and test periods

  validation_date_end = get_most_recent_saturday_date_str()
  validation_date_start = pd.to_datetime(validation_date_end) - pd.Timedelta(
      weeks=3
  )

  validation_reference_dates = get_saturdays_between_dates(
      validation_date_start, validation_date_end
  )
  print('validation_reference_dates:', validation_reference_dates)
  validation_forecasts, validation_score = compute_rolling_evaluation(
      observed_values=dataset.copy(),
      reference_dates=validation_reference_dates,
      fit_and_predict_fn=fit_and_predict_fn,
      horizons=HORIZONS,
      location_codes=location_codes,
      locations_df=locations,
  )

  print(f'\nValidation Score: {validation_score}')
  if not validation_forecasts.empty:
    validation_forecasts.to_csv('/tmp/validation_forecasts.csv', index=False)
    print("Validation forecasts saved to '/tmp/validation_forecasts.csv'")

  # Plot forecast and predictions on validation dates against observed data

  validation_forecasts['target_end_date'] = pd.to_datetime(
      validation_forecasts['target_end_date']
  )
  validation_forecasts['reference_date'] = pd.to_datetime(
      validation_forecasts['reference_date']
  )

  # Prepare the observed data
  national_observed_all = (
      dataset.groupby('target_end_date')['Total Influenza Admissions']
      .sum()
      .reset_index()
  )
  national_observed_all['target_end_date'] = pd.to_datetime(
      national_observed_all['target_end_date']
  )

  dates_to_plot_validation = [
      {
          'start': pd.to_datetime(validation_date_start) - timedelta(weeks=2),
          'end': pd.to_datetime(validation_date_end) + timedelta(weeks=5),
          'name': 'validation',
      },
  ]

  for season in dates_to_plot_validation:
    print(f"--- Generating plot for {season['name']} dates ---")
    plot_season_forecasts(
        season_start=season['start'],
        season_end=season['end'],
        season_name=season['name'],
        all_forecasts_df=validation_forecasts,
        national_observed_df=national_observed_all,
        step_size=1,
    )

  submission_date_str = get_next_saturday_date_str()
  submission_date = pd.to_datetime(submission_date_str).date()

  test_forecasts, _ = compute_rolling_evaluation(
      observed_values=dataset.copy(),
      reference_dates=[submission_date],
      fit_and_predict_fn=fit_and_predict_fn,
      horizons=HORIZONS,
      location_codes=location_codes,
      locations_df=locations,
  )

  print('\n--- Creating the submission file ---')

  if not test_forecasts.empty:
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc flu hosp')
    cdc_submission.to_csv(
        f'/tmp/{submission_date_str}_{MODEL_NAME}.csv', index=False
    )
    print(
        'Submission forecasts saved to'
        f" '/tmp/{submission_date_str}_{MODEL_NAME}.csv'"
    )

    print('Verify final submission file:')
    print(cdc_submission)

    # Convert dates in test_forecasts to Timestamp
    test_forecasts['target_end_date'] = pd.to_datetime(
        test_forecasts['target_end_date']
    )
    test_forecasts['reference_date'] = pd.to_datetime(
        test_forecasts['reference_date']
    )

    # Plot forecasts for submission (all horizons)
    cdc_submission['target_end_date'] = pd.to_datetime(
        cdc_submission['target_end_date']
    )
    cdc_submission['reference_date'] = pd.to_datetime(
        cdc_submission['reference_date']
    )

    dates_to_plot_submission = [
        {
            'start': pd.to_datetime(submission_date) - timedelta(weeks=1),
            'end': pd.to_datetime(submission_date) + timedelta(weeks=3),
            'name': f'{submission_date} forecast',
        },
    ]

    for season in dates_to_plot_submission:
      print(f"--- Generating plot for {season['name']} dates ---")
      plot_season_forecasts(
          season_start=season['start'],
          season_end=season['end'],
          season_name=season['name'],
          all_forecasts_df=test_forecasts,
          national_observed_df=None,
          step_size=1,
      )


if __name__ == '__main__':
  app.run(main)
