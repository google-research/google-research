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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation

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
MODEL_NAME = 'Google_SAI-Adapted_3'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# The 4 core principles of the method described:
# 1. Ingestion: Generate diverse long-term projection trajectories using scenario archetypes, dynamically informed by local historical trends, seasonality, and volatility.
# 2. Validation: Calculate the likelihood of recent observed ground-truth data given the initial forecast steps of each scenario trajectory, using a robust, overdispersed count data model.
# 3. Filtering/Pruning: Apply an adaptive rejection scheme to discard scenario runs with likelihood scores below a defined threshold (or keep the top matches).
# 4. Aggregation: Construct the final forecast distribution by aggregating the remaining, empirically validated scenario trajectories.

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # For robust growth factor
from scipy.ndimage import gaussian_filter1d  # For smoothing seasonal factors
from typing import Any  # Kept for context

TARGET_STR = 'Total COVID-19 Admissions'

QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]
N_SCENARIOS = (
    5000  # Increased number of scenario trajectories for more robust quantiles
)
PERCENTILE_TO_KEEP = 0.4  # Proportion of top scenarios to keep after validation

# Default parameters for volatility if historical data is insufficient or too stable
DEFAULT_GROWTH_FACTOR_STD_BASE = 0.05  # default std for multiplicative growth factor (e.g., 5% weekly change variability)
DEFAULT_NOISE_REL_STD_FACTOR = (
    0.1  # default std for additive noise, relative to current value (e.g., 10%)
)
MIN_ABSOLUTE_NOISE_STD = 2.0  # Minimum absolute noise standard deviation to ensure some variability and avoid division by zero.
MIN_GROWTH_FACTOR_STD = (
    0.01  # Minimum std for growth factor to ensure some variability
)
MIN_HISTORICAL_POINTS_FOR_TREND = (
    3  # Minimum points needed to calculate a robust recent trend
)
RECENT_VOLATILITY_WEEKS = (
    52  # Look back 52 weeks for volatility calculations (1 year)
)
RECENT_TREND_WEEKS = 4  # Look back 4 weeks for robust observed trend estimation
EPSILON_FOR_LOG = 1e-6  # Small value to replace zeros before log transformation
VALIDATION_WINDOW_WEEKS = 3  # Number of initial forecast steps (h=0, h=1, h=2) to use for likelihood validation

# IMPROVED: Redefined MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR to be more robust.
MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR = (
    MIN_ABSOLUTE_NOISE_STD**2
)  # Use square of min absolute noise as floor for variance (e.g., 4.0 if MIN_ABSOLUTE_NOISE_STD = 2.0)


# IMPROVED: Consolidated definition of scenario archetypes with added volatility multipliers (mimics diverse Hub projections)
# Each archetype has a base growth factor, a damping factor, a proportion, and now volatility multipliers.
ARCHETYPE_DEFINITIONS = {
    'rapid_decline': {
        'base_growth_factor': 0.85,
        'damping_factor': 0.90,
        'proportion': 0.10,
        'growth_std_multiplier': 1.2,
        'noise_std_multiplier': 1.2,
    },
    'slow_decline': {
        'base_growth_factor': 0.95,
        'damping_factor': 0.95,
        'proportion': 0.20,
        'growth_std_multiplier': 1.0,
        'noise_std_multiplier': 1.0,
    },
    'stable': {
        'base_growth_factor': 1.0,
        'damping_factor': 0.98,
        'proportion': 0.40,
        'growth_std_multiplier': 0.8,
        'noise_std_multiplier': 0.8,
    },
    'slow_growth': {
        'base_growth_factor': 1.05,
        'damping_factor': 0.95,
        'proportion': 0.20,
        'growth_std_multiplier': 1.0,
        'noise_std_multiplier': 1.0,
    },
    'rapid_growth': {
        'base_growth_factor': 1.15,
        'damping_factor': 0.90,
        'proportion': 0.10,
        'growth_std_multiplier': 1.5,
        'noise_std_multiplier': 1.5,
    },
}
JITTER_STD_FOR_ARCHETYPE_GROWTH = 0.01  # Small std for jittering archetype base growth factors for more continuous diversity
INITIAL_SCENARIO_PERTURBATION_RELATIVE_STD = 0.02  # Relative std for initial scenario perturbation (e.g., 2% for immediate spread)

# Multiplicative increase to growth factor std per horizon (models increasing uncertainty)
DAMPING_FACTOR_STD_GROWTH_INCREASE_FACTOR = 1.015
INITIAL_SCENARIO_SMOOTHING_WEEKS = (
    3  # Weeks to average for scenario starting value for robustness
)
VALIDATION_GROWTH_WEEKS = (
    3  # Weeks for robust growth factor in validation's expected path
)

# Robustness constants for handling zero values
MIN_SCENARIO_PROJECTION_FLOOR = (
    1.0  # Minimum value for scenario projections during simulation
)
MIN_VALIDATION_BENCHMARK_VALUE = (
    1.0  # Minimum expected value for validation when actual observation is zero
)

# Seasonality constants
SEASONALITY_SMOOTH_WINDOW = 3  # Window for smoothing seasonal factors
MIN_WEEKS_FOR_SEASONALITY = (
    2 * 52 + SEASONALITY_SMOOTH_WINDOW
)  # Require at least two full years of data plus smoothing window


# Helper to calculate robust standard deviation from MAD (IMPROVEMENT 3)
def robust_std_from_mad(data):
  """Estimates standard deviation using Median Absolute Deviation (MAD)."""
  if len(data) < 2:
    return 0.0
  mad = np.median(np.abs(data - np.median(data)))
  return mad * 1.4826  # Conversion factor for normal distribution


# Helper to calculate robust growth factor (IMPROVEMENT 1: using Linear Regression)
def get_robust_growth_factor(
    series, num_weeks, min_periods = 2
):
  """Calculates a robust weekly growth factor from recent data using linear regression on log-transformed values."""
  series = series.sort_index()
  recent_data = series.tail(num_weeks)
  if len(recent_data) < min_periods:
    return 1.0  # Default to no growth if not enough data

  recent_data_positive = recent_data.replace(0, EPSILON_FOR_LOG)
  if (recent_data_positive == EPSILON_FOR_LOG).all():
    return 1.0  # If all are effectively zero, no growth

  X = np.arange(len(recent_data_positive)).reshape(-1, 1)
  y_log = np.log(recent_data_positive.values)

  model = LinearRegression()
  with warnings.catch_warnings():
    warnings.simplefilter(
        'ignore', category=UserWarning
    )  # Suppress convergence warnings from LinearRegression
    if len(X) > 1:  # Need at least 2 points for regression
      model.fit(X, y_log)
      log_growth_rate = model.coef_[0]
      return np.exp(log_growth_rate)
    else:
      return 1.0  # Fallback if not enough data for regression


# Helper function to estimate seasonal factors (IMPROVEMENT 2: with Gaussian smoothing and normalization and clipping)
def estimate_seasonal_factors(
    admissions_series, smooth_window = SEASONALITY_SMOOTH_WINDOW
):
  """Estimates weekly seasonal multiplicative factors from historical data, smoothed and normalized.

  The factors represent week-over-week growth rate adjustments and are
  normalized to have a geometric mean of 1.0 over a year, preventing them from
  introducing a long-term trend.
  """
  # Ensure index is datetime and sorted
  admissions_series = admissions_series.sort_index()

  # Need enough data for meaningful seasonality estimation
  if len(admissions_series) < MIN_WEEKS_FOR_SEASONALITY:
    return pd.Series(1.0, index=range(1, 54))

  df = admissions_series.to_frame(name='admissions')
  df['weekofyear'] = df.index.isocalendar().week.astype(int)  # ensure int type

  # Calculate weekly multiplicative changes (Current Week / Previous Week)
  df['prev_admissions'] = df['admissions'].shift(1)
  # Replace 0s with EPSILON_FOR_LOG to avoid division by zero or log(-inf)
  df['weekly_ratio'] = df['admissions'].replace(0, EPSILON_FOR_LOG) / df[
      'prev_admissions'
  ].replace(0, EPSILON_FOR_LOG)

  # IMPROVEMENT: Clip weekly_ratio to prevent extreme values from distorting median calculation.
  # This guards against cases where one of the values is very close to zero, leading to an
  # unrepresentatively large or small ratio, even with EPSILON_FOR_LOG.
  # Chosen range (0.1 to 10.0) assumes weekly changes are rarely less than 10% or more than 10x.
  df['weekly_ratio'] = np.clip(df['weekly_ratio'], 0.1, 10.0)

  df.dropna(inplace=True)

  if df.empty:
    return pd.Series(1.0, index=range(1, 54))

  # Calculate median weekly ratio for each week of the year
  median_weekly_ratios = df.groupby('weekofyear')['weekly_ratio'].median()

  # Create a full 53-week series, filling missing weeks with 1.0 (no change)
  full_seasonal_factors = pd.Series(1.0, index=range(1, 54))
  full_seasonal_factors.update(median_weekly_ratios)

  # Convert to log-scale for smoothing
  log_factors = np.log(full_seasonal_factors.values)

  # Apply cyclic Gaussian smoothing using scipy.ndimage.gaussian_filter1d
  sigma = smooth_window / 2.0  # Standard deviation for the Gaussian kernel.
  smoothed_log_factors = gaussian_filter1d(
      log_factors, sigma=sigma, mode='wrap'
  )

  # IMPROVEMENT: Normalize smoothed_log_factors to ensure geometric mean of 1.0 over a year.
  # Subtract the mean to ensure sum of log factors is 0, which means exp(sum) = 1 (geometric mean = 1).
  if len(smoothed_log_factors) > 0:
    smoothed_log_factors -= np.mean(smoothed_log_factors)

  # Convert back to multiplicative factors
  smoothed_multipliers = np.exp(smoothed_log_factors)

  # Re-index to weeks 1-53
  seasonal_multipliers = pd.Series(smoothed_multipliers, index=range(1, 54))

  return seasonal_multipliers


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make probabilistic predictions for test_x using a meta-ensemble approach

  that filters and aggregates simulated scenario trajectories.

  Args:
      train_x: DataFrame of historical features.
      train_y: Series of historical target values.
      test_x: DataFrame of future features for which to predict.

  Returns:
      A pandas DataFrame with quantile predictions for test_x.
  """
  # Ensure target_end_date is datetime for consistency
  train_x_processed = train_x.copy()
  test_x_processed = test_x.copy()
  train_x_processed['target_end_date'] = pd.to_datetime(
      train_x_processed['target_end_date']
  )
  test_x_processed['target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  )

  # Rename train_y Series for easier merging/access
  train_y_renamed = train_y.rename(TARGET_STR)

  # Combine train_x and train_y to easily access historical data for each location
  train_df = pd.merge(
      train_x_processed,
      train_y_renamed,
      left_index=True,
      right_index=True,
      how='inner',
  )

  # Determine the effective 'train_end_date' for the current forecast period.
  # This is the last date for which we have observed data *before* the first forecast point.
  reference_date = test_x_processed.loc[
      test_x_processed['horizon'] == 0, 'target_end_date'
  ].min()
  train_end_date = reference_date - pd.Timedelta(weeks=1)

  # Initialize DataFrame for storing final quantile predictions
  test_y_hat_quantiles = pd.DataFrame(index=test_x_processed.index)
  for q in QUANTILES:
    test_y_hat_quantiles[f'quantile_{q}'] = np.nan

  all_location_forecasts = []

  # Get unique locations that require forecasts from test_x
  unique_locations = test_x_processed['location'].unique()

  # Get the most recent observed value for each location from train_df
  last_actual_observed_data_series = train_df[
      (train_df['target_end_date'] == train_end_date)
      & (
          train_df['location'].isin(unique_locations)
      )  # Filter to only relevant locations
  ].set_index('location')[TARGET_STR]

  # Calculate a global median from available historical data to handle locations with no recent data.
  # Ensure a minimum floor of 1.0 for robustness.
  global_last_observed_median = (
      last_actual_observed_data_series.median()
      if not last_actual_observed_data_series.empty
      else 100.0
  )
  global_last_observed_median = max(
      MIN_VALIDATION_BENCHMARK_VALUE, float(global_last_observed_median)
  )

  # --- IMPROVED: Assign each scenario an archetype with its base growth factor, damping, and volatility multipliers (Ingestion Proxy) ---
  archetype_names = list(ARCHETYPE_DEFINITIONS.keys())
  archetype_proportions = [
      ARCHETYPE_DEFINITIONS[name]['proportion'] for name in archetype_names
  ]

  # Randomly assign archetypes to scenarios based on proportions
  scenario_archetypes = np.random.choice(
      archetype_names, size=N_SCENARIOS, p=archetype_proportions
  )

  # Map archetype names to their respective base growth factors, damping factors, and volatility multipliers
  scenario_base_growth_factors = np.array([
      ARCHETYPE_DEFINITIONS[arch]['base_growth_factor']
      for arch in scenario_archetypes
  ])
  # Add jitter to the archetype base growth factors for more continuous diversity
  scenario_base_growth_factors += np.random.normal(
      0, JITTER_STD_FOR_ARCHETYPE_GROWTH, N_SCENARIOS
  )

  scenario_damping_factors = np.array([
      ARCHETYPE_DEFINITIONS[arch]['damping_factor']
      for arch in scenario_archetypes
  ])

  # New: Archetype-specific volatility multipliers
  scenario_growth_std_multipliers = np.array([
      ARCHETYPE_DEFINITIONS[arch]['growth_std_multiplier']
      for arch in scenario_archetypes
  ])
  scenario_noise_std_multipliers = np.array([
      ARCHETYPE_DEFINITIONS[arch]['noise_std_multiplier']
      for arch in scenario_archetypes
  ])

  # --- Iterate through locations for forecasting ---
  for location_id in unique_locations:
    location_test_x = test_x_processed[
        test_x_processed['location'] == location_id
    ].copy()
    location_horizons = sorted(location_test_x['horizon'].unique())

    # actual_last_y_val: The true last observed value for this location at train_end_date.
    actual_last_y_val = last_actual_observed_data_series.get(
        location_id, global_last_observed_median
    )
    actual_last_y_val = max(
        0.0, float(actual_last_y_val)
    )  # Ensure non-negative, but allow 0 for historical values.

    # --- Dynamic Calculation of Volatility Parameters (Ingestion Proxy) ---
    location_history = train_df[
        (train_df['location'] == location_id)
        & (train_df['target_end_date'] <= train_end_date)
    ].sort_values('target_end_date')

    admissions_series = location_history.set_index('target_end_date')[
        TARGET_STR
    ].astype(float)

    admissions_for_volatility = admissions_series[
        admissions_series.index
        >= (train_end_date - pd.Timedelta(weeks=RECENT_VOLATILITY_WEEKS))
    ]
    if len(admissions_for_volatility) < MIN_HISTORICAL_POINTS_FOR_TREND:
      admissions_for_volatility = admissions_series.tail(
          max(MIN_HISTORICAL_POINTS_FOR_TREND, len(admissions_series))
      )

    # Initialize with default parameters
    growth_std_param = DEFAULT_GROWTH_FACTOR_STD_BASE
    noise_abs_base_param = MIN_ABSOLUTE_NOISE_STD
    noise_rel_factor_param = DEFAULT_NOISE_REL_STD_FACTOR

    # --- IMPROVEMENT: Dynamic Volatility Estimation using robust_std_from_mad ---
    if len(admissions_for_volatility) > 1:
      log_returns = (
          np.log(admissions_for_volatility.replace(0, EPSILON_FOR_LOG))
          .diff()
          .dropna()
      )
      if not log_returns.empty:
        # Use robust_std_from_mad for growth_std_param
        growth_std_param = robust_std_from_mad(log_returns.values)

      absolute_changes = admissions_for_volatility.diff().dropna()
      if not absolute_changes.empty:
        # Use robust_std_from_mad for noise_abs_base_param
        noise_abs_base_param = robust_std_from_mad(absolute_changes.values)

      avg_admissions_for_volatility = admissions_for_volatility.mean()
      if avg_admissions_for_volatility > EPSILON_FOR_LOG:
        robust_avg_admissions = max(1.0, avg_admissions_for_volatility)
        noise_rel_factor_param = noise_abs_base_param / robust_avg_admissions
        # IMPROVEMENT: Clip noise_rel_factor_param to a reasonable range
        noise_rel_factor_param = np.clip(noise_rel_factor_param, 0.005, 0.25)

    growth_std_param = max(MIN_GROWTH_FACTOR_STD, growth_std_param)
    noise_abs_base_param = max(MIN_ABSOLUTE_NOISE_STD, noise_abs_base_param)

    # Robust starting point for scenario generation
    initial_scenario_start_history = admissions_series.tail(
        INITIAL_SCENARIO_SMOOTHING_WEEKS
    )
    if (
        not initial_scenario_start_history.empty
        and initial_scenario_start_history.mean() > 0
    ):
      seeded_last_y_val_for_scenarios = initial_scenario_start_history.mean()
    else:
      seeded_last_y_val_for_scenarios = MIN_SCENARIO_PROJECTION_FLOOR
    seeded_last_y_val_for_scenarios = max(
        MIN_SCENARIO_PROJECTION_FLOOR, float(seeded_last_y_val_for_scenarios)
    )

    # Estimate seasonal factors for the current location (IMPROVEMENT 2: with Gaussian smoothing and normalization)
    seasonal_multipliers = estimate_seasonal_factors(
        admissions_series, smooth_window=SEASONALITY_SMOOTH_WINDOW
    )
    if seasonal_multipliers.empty or seasonal_multipliers.isnull().all():
      seasonal_multipliers = pd.Series(1.0, index=range(1, 54))

    # --- IMPROVEMENT: Estimate dispersion parameter (phi) for Negative Binomial-like variance for validation ---
    validation_phi_dispersion = np.inf  # Default to Poisson-like (infinite phi)
    min_phi_value_for_estimation = (
        0.5  # A reasonable lower bound for phi to prevent extreme dispersion
    )
    rolling_window_for_phi = (
        4  # Use a 4-week rolling window for mean/var estimation
    )

    if (
        len(admissions_series) >= RECENT_VOLATILITY_WEEKS
    ):  # Need enough data for robust phi
      # Ensure enough data points for rolling window calculation
      if len(admissions_series) >= rolling_window_for_phi:
        rolling_mean = (
            admissions_series.rolling(
                window=rolling_window_for_phi,
                min_periods=rolling_window_for_phi - 1,
            )
            .mean()
            .dropna()
        )
        rolling_var = (
            admissions_series.rolling(
                window=rolling_window_for_phi,
                min_periods=rolling_window_for_phi - 1,
            )
            .var()
            .dropna()
        )

        common_idx = rolling_mean.index.intersection(rolling_var.index)
        rolling_mean_filtered = rolling_mean.loc[common_idx].values
        rolling_var_filtered = rolling_var.loc[common_idx].values

        # Calculate phi = mean^2 / (var - mean) where var > mean
        # Use a vectorized operation for efficiency
        valid_idx = (
            rolling_var_filtered > rolling_mean_filtered + EPSILON_FOR_LOG
        )  # ensure var > mean, accounting for near-zero
        if np.any(valid_idx):
          phis = (rolling_mean_filtered[valid_idx] ** 2) / (
              rolling_var_filtered[valid_idx] - rolling_mean_filtered[valid_idx]
          )
          phis = phis[
              phis > 0.1
          ]  # Filter out near-zero or negative phis (unrealistic or under-dispersed)
          if len(phis) > 0:
            phi_estimate = np.median(phis)  # Use median for robustness
            validation_phi_dispersion = max(
                min_phi_value_for_estimation, float(phi_estimate)
            )

    # --- Scenario Generation (Dynamically applying "Hub" projections with local context) ---
    all_trajectories = np.zeros((N_SCENARIOS, len(location_horizons)))

    # Introduce initial perturbation to current_projected_vals
    initial_perturbation = np.random.normal(
        0,
        seeded_last_y_val_for_scenarios
        * INITIAL_SCENARIO_PERTURBATION_RELATIVE_STD,
        N_SCENARIOS,
    )
    current_projected_vals = np.maximum(
        MIN_SCENARIO_PROJECTION_FLOOR,
        seeded_last_y_val_for_scenarios + initial_perturbation,
    )

    for h_idx, h in enumerate(location_horizons):
      # IMPROVEMENT: Ensure base for multiplicative growth is floored to avoid issues with very low values
      base_for_multiplicative_growth = np.maximum(
          MIN_SCENARIO_PROJECTION_FLOOR, current_projected_vals
      )

      # IMPROVED: Apply differentiated damping to the archetype's base growth factor
      # The base growth factor represents the scenario's "belief" about the long-term trend.
      # Damping means this belief (e.g., strong growth or decline) might get closer to 1.0 (no change) over time.
      damped_archetype_growth_factor = 1.0 + (
          scenario_base_growth_factors - 1.0
      ) * (scenario_damping_factors**h_idx)

      # Dynamically adjust the growth standard deviation with horizon for this location,
      # AND apply archetype-specific growth_std_multiplier.
      location_growth_std_scaled = (
          growth_std_param
          * (DAMPING_FACTOR_STD_GROWTH_INCREASE_FACTOR**h_idx)
          * scenario_growth_std_multipliers
      )

      # Calculate the mean of the log-normal distribution for the growth factor.
      # This incorporates the damped archetype growth as the central tendency for the multiplicative step.
      log_growth_means_for_scenarios = np.log(
          damped_archetype_growth_factor
      ) - 0.5 * (location_growth_std_scaled**2)

      # Sample random log growth factors from this distribution.
      multiplicative_factor_from_scenario = np.exp(
          np.random.normal(
              log_growth_means_for_scenarios,
              location_growth_std_scaled,  # Use the scaled std
              N_SCENARIOS,
          )
      )

      # Get current target week of year for seasonality
      forecast_target_date = reference_date + pd.Timedelta(weeks=h)
      current_week_of_year = forecast_target_date.isocalendar().week
      seasonal_adjustment = seasonal_multipliers.get(current_week_of_year, 1.0)

      # Apply the scenario's projection, scaled by the current value and adjusted for local seasonality
      current_projected_vals_after_growth = (
          base_for_multiplicative_growth
          * multiplicative_factor_from_scenario
          * seasonal_adjustment
      )

      # Dynamically calculate std for additive noise for this step, based on value *after* growth,
      # AND apply archetype-specific noise_std_multiplier.
      noise_std_for_this_step = np.maximum(
          MIN_ABSOLUTE_NOISE_STD,
          np.maximum(
              noise_abs_base_param
              * scenario_noise_std_multipliers,  # Apply multiplier to base absolute noise
              noise_rel_factor_param
              * scenario_noise_std_multipliers
              * current_projected_vals_after_growth,  # Apply multiplier to relative noise
          ),
      )
      noise = np.random.normal(0, noise_std_for_this_step, N_SCENARIOS)
      current_projected_vals = current_projected_vals_after_growth + noise

      # Ensure non-negative values throughout the trajectory.
      current_projected_vals = np.maximum(0, current_projected_vals)
      all_trajectories[:, h_idx] = current_projected_vals

    # --- 2. Validation: Calculate likelihood of recent observed data given each scenario trajectory ---

    # Robustly calculate the observed immediate multiplicative growth factor for the expected path
    # --- IMPROVED: Using Linear Regression for robust_growth_factor over RECENT_TREND_WEEKS ---
    observed_growth_factor = get_robust_growth_factor(
        admissions_series, num_weeks=RECENT_TREND_WEEKS
    )

    # actual_last_y_val is the ground truth for validation.
    base_val_for_validation_path = max(
        MIN_VALIDATION_BENCHMARK_VALUE, actual_last_y_val
    )

    # Initialize total negative log-likelihood for each scenario
    # We are accumulating negative log-likelihoods, so initialize to 0 and add positive terms.
    total_negative_log_likelihood = np.zeros(N_SCENARIOS)

    num_validation_steps = min(VALIDATION_WINDOW_WEEKS, len(location_horizons))

    for k in range(num_validation_steps):
      expected_value_at_k = base_val_for_validation_path * (
          observed_growth_factor ** (k + 1)
      )

      validation_target_date_k = reference_date + pd.Timedelta(weeks=k)
      validation_week_of_year_k = validation_target_date_k.isocalendar().week
      seasonal_adjustment_for_validation = seasonal_multipliers.get(
          validation_week_of_year_k, 1.0
      )
      expected_value_at_k *= seasonal_adjustment_for_validation

      expected_value_at_k = np.maximum(
          0.0, expected_value_at_k
      )  # Ensure non-negative expectation for variance calculation

      scenario_value_at_k = all_trajectories[:, k]
      deviation_at_k = scenario_value_at_k - expected_value_at_k

      # --- IMPROVEMENT: Refined total_variance_for_likelihood_per_scenario for better conceptual clarity & numerical stability ---
      # This calculation represents Var(Scenario_k - Expected_k) = Var(Scenario_k) + Var(Expected_k)

      # Component 1: Base variance for count data (Poisson + Negative Binomial overdispersion)
      # This reflects the inherent variability of the observed data around its true mean.
      base_variance_for_expected_value = (
          expected_value_at_k  # Poisson component (mean=variance)
      )
      if (
          validation_phi_dispersion is not np.inf
          and expected_value_at_k > EPSILON_FOR_LOG
      ):
        base_variance_for_expected_value += (
            expected_value_at_k**2
        ) / validation_phi_dispersion  # Negative Binomial overdispersion
      # Ensure a minimum floor for this base variance
      base_variance_for_expected_value = np.maximum(
          base_variance_for_expected_value, MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR
      )

      # Component 2: Variance due to uncertainty in the *observed benchmark's* projection.
      # This accounts for the uncertainty in extrapolating `expected_value_at_k` itself from historical trends.
      # `growth_std_param` is the standard deviation of historical log returns.
      # Variance contribution grows quadratically with horizon (k+1) and with the mean squared.
      variance_of_observed_projection_error = (expected_value_at_k**2) * (
          (k + 1) * growth_std_param
      ) ** 2
      # Ensure a minimum floor for this component to prevent numerical issues
      variance_of_observed_projection_error = np.maximum(
          MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR / 4,
          variance_of_observed_projection_error,
      )  # Use a fraction of total floor

      # Component 3: Variance due to uncertainty in the *scenario's* projected mean itself for *each scenario*
      # This accounts for the accumulating uncertainty of the scenario's *own growth factor*.
      scenario_specific_growth_std = (
          growth_std_param * scenario_growth_std_multipliers
      )
      # Approximated as (mean^2) * (std_dev_log_growth * k)^2 (using delta method and assuming independence).
      variance_of_scenario_mean_uncertainty = (expected_value_at_k**2) * (
          (k + 1) * scenario_specific_growth_std
      ) ** 2
      # Ensure a minimum floor for this component
      variance_of_scenario_mean_uncertainty = np.maximum(
          MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR / 4,
          variance_of_scenario_mean_uncertainty,
      )  # Use a fraction of total floor

      # Component 4: Intrinsic noise expected from *each scenario's step-wise process*, accumulated over steps.
      # This represents the variance from the additive noise *accumulated up to horizon k*.
      scenario_specific_abs_noise_std = (
          noise_abs_base_param * scenario_noise_std_multipliers
      )
      scenario_specific_rel_noise_std = (
          noise_rel_factor_param * scenario_noise_std_multipliers
      )

      # Variance of a single step's additive noise, per scenario.
      single_step_additive_variance_per_scenario = (
          scenario_specific_abs_noise_std**2
      ) + (scenario_specific_rel_noise_std * expected_value_at_k) ** 2
      # Ensure a minimum floor for a single step's additive variance
      single_step_additive_variance_per_scenario = np.maximum(
          MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR / 4,
          single_step_additive_variance_per_scenario,
      )

      # Accumulate this additive variance over (k+1) validation steps.
      accumulated_additive_variance_per_scenario = (
          k + 1
      ) * single_step_additive_variance_per_scenario

      # Total variance for likelihood: Sum of all uncertainty components for *each scenario*.
      # This represents the total expected squared difference between a scenario value and the benchmark.
      total_variance_for_likelihood_per_scenario = (
          base_variance_for_expected_value
          + variance_of_observed_projection_error
          + variance_of_scenario_mean_uncertainty
          + accumulated_additive_variance_per_scenario
      )

      # Ensure total_variance_for_likelihood_per_scenario is never too small
      total_variance_for_likelihood_per_scenario = np.maximum(
          total_variance_for_likelihood_per_scenario,
          MIN_VARIANCE_FOR_LIKELIHOOD_FLOOR,
      )

      # --- CRITICAL IMPROVEMENT: Add the missing variance normalization term to the negative log-likelihood calculation. ---
      # For a Gaussian log-likelihood, the term -0.5 * log(sigma^2) is also needed.
      # Since total_negative_log_likelihood is accumulated by adding positive terms (representing negative log-likelihood),
      # we must add 0.5 * log(total_variance_for_likelihood_per_scenario).
      # The constant 0.5 * log(2*pi) is omitted as it does not affect the ranking of scenarios.
      total_negative_log_likelihood += (
          0.5 * (deviation_at_k**2) / total_variance_for_likelihood_per_scenario
      ) + (0.5 * np.log(total_variance_for_likelihood_per_scenario))

    # Convert total_negative_log_likelihood back to likelihood_scores.
    # Smaller negative log-likelihood (more likely) means larger exp value.
    likelihood_scores = np.exp(-total_negative_log_likelihood)
    likelihood_scores = np.nan_to_num(
        likelihood_scores, nan=0.0, posinf=0.0, neginf=0.0
    )

    # 3. Filtering/Pruning: Apply adaptive rejection scheme
    sorted_indices = np.argsort(likelihood_scores)[::-1]
    num_to_keep = int(N_SCENARIOS * PERCENTILE_TO_KEEP)
    num_to_keep = max(
        1, num_to_keep
    )  # Ensure at least one scenario is always kept

    filtered_trajectories = all_trajectories[sorted_indices[:num_to_keep]]

    # 4. Aggregation: Construct the final forecast distribution
    quantiles_for_location = pd.DataFrame(index=location_test_x.index)

    for h_idx, h_val in enumerate(location_horizons):
      horizon_scenario_values = filtered_trajectories[:, h_idx]
      q_forecasts = np.quantile(horizon_scenario_values, QUANTILES)

      target_idx_for_horizon = location_test_x[
          location_test_x['horizon'] == h_val
      ].index
      for q_idx, q_level in enumerate(QUANTILES):
        quantiles_for_location.loc[
            target_idx_for_horizon, f'quantile_{q_level}'
        ] = q_forecasts[q_idx]

    all_location_forecasts.append(quantiles_for_location)

  if all_location_forecasts:
    test_y_hat_quantiles = pd.concat(all_location_forecasts).sort_index()
  else:
    # If no forecasts were generated at all (e.g., test_x was empty), return an empty DataFrame
    return pd.DataFrame(
        index=test_x_processed.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )

  # Post-processing: Ensure non-negativity and strict monotonicity
  quantile_cols = [f'quantile_{q}' for q in QUANTILES]

  # 1. Ensure all predictions are non-negative (on float values)
  test_y_hat_quantiles[quantile_cols] = test_y_hat_quantiles[
      quantile_cols
  ].clip(lower=0)

  # 2. Enforce monotonicity: Each quantile must be greater than or equal to the preceding one.
  # Apply cummax operation row-wise across the quantile columns *before* rounding to maintain consistency in float values.
  test_y_hat_quantiles[quantile_cols] = test_y_hat_quantiles[
      quantile_cols
  ].cummax(axis=1)

  # 3. Round to nearest integer and convert to int type.
  test_y_hat_quantiles[quantile_cols] = (
      test_y_hat_quantiles[quantile_cols].round().astype(int)
  )

  # IMPROVED: Re-apply cummax after rounding to strictly guarantee monotonicity for integer values,
  # as rounding might occasionally make distinct float quantiles equal or, in rare edge cases, slightly
  # violate non-decreasing order if not re-enforced for integer types.
  test_y_hat_quantiles[quantile_cols] = test_y_hat_quantiles[
      quantile_cols
  ].cummax(axis=1)

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

  dataset[TARGET_STR] = (
      pd.to_numeric(dataset[TARGET_STR], errors='coerce')
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
      dataset.groupby('target_end_date')[TARGET_STR].sum().reset_index()
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
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc covid hosp')
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
