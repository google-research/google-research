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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,missing-function-docstring,g-doc-return-or-yield,g-doc-args,broad-exception-caught

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
MODEL_NAME = 'Google_SAI-Adapted_8'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Forecasting Method (as instructed by the expert):
# 1. Aggregation of Multiple Probabilistic Forecasts: The model combines predictions from several distinct (simulated) forecaster "sources".
# 2. Probabilistic Outputs from Individual Sources: Each simulated forecaster provides a full probability distribution, which is then represented by a set of quantiles derived from appropriate count data distributions (Negative Binomial or Poisson).
# 3. Consensus Algorithm for Combination: A specific weighting algorithm (weighted median) is applied to synthesize the individual distributions into a single consensus distribution.
# 4. Quantile Extraction from Consensus: The final reported quantiles are directly derived from the aggregated consensus distribution.

import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from typing import Any, Callable, Dict, List, Tuple

# Global constants QUANTILES and TARGET_STR are assumed to be available from the notebook's preamble.
# 'locations' DataFrame is assumed to be accessible in the global scope for _create_fold_scaffold if needed,
# though fit_and_predict_fn is designed to work with features pre-merged into train_x and test_x.


# --- Helper function for weighted median (unchanged as it is robust) ---
def _weighted_median(values, weights):
  values = np.asarray(values)
  weights = np.asarray(weights)

  if len(values) == 0:
    return 0.0

  sorted_indices = np.argsort(values)
  sorted_values = values[sorted_indices]
  sorted_weights = weights[sorted_indices]

  sum_weights = np.sum(sorted_weights)
  if sum_weights == 0:
    # Fallback for zero weights, return simple median or a default
    # If all weights are zero, this implies no valid forecasters contributed.
    # Returning the median of values if available, else 0.0
    return (
        sorted_values[len(sorted_values) // 2]
        if len(sorted_values) > 0
        else 0.0
    )

  normalized_weights = sorted_weights / sum_weights
  cumulative_weights = np.cumsum(normalized_weights)

  median_idx = np.searchsorted(cumulative_weights, 0.5, side='left')
  return sorted_values[min(median_idx, len(sorted_values) - 1)]


# --- Helper to get quantiles from distribution (Revised to prioritize Negative Binomial/Poisson with improved robustness) ---
def generate_quantiles_from_simulation(
    mean_values,
    overdispersion_param_phi_values,
    quantiles_list,
):
  """Generates a matrix of integer quantiles for multiple rows from Negative Binomial or Poisson distribution.

  This function explicitly chooses between Negative Binomial (for overdispersed
  data) and Poisson (for equi-dispersed or very low-mean data), which are
  appropriate for count data.

  Improvements:
  - Standardized Negative Binomial parameterization (Var = mu + phi*mu^2).
  - Improved numerical stability for distribution parameters.
  - Explicit handling of very low means and minimum dispersion.
  - Enhanced fallback strategy for ppf failures to ensure non-zero width
  intervals.
  """
  num_rows = len(mean_values)
  num_quantiles = len(quantiles_list)
  result_quantiles_matrix = np.zeros((num_rows, num_quantiles), dtype=int)

  # Convert quantiles_list to a numpy array for broadcasting with ppf
  q_levels = np.asarray(quantiles_list)

  # --- Define a robust fallback for when ppf fails to yield a distribution ---
  def _get_fallback_quantiles(
      mu_single, q_levels_arr
  ):
    # Create a simple spread around the mean.
    # Spread factor: scales with sqrt(mu) but has a minimum base width.
    # For mu=0, std=1. For mu=100, std=sqrt(100)*0.5+1 = 6. This creates a reasonable interval.
    spread_factor = np.sqrt(mu_single) * 0.5 + 1.0
    fallback_quantiles = mu_single + (q_levels_arr - 0.5) * spread_factor
    return np.maximum(0.0, fallback_quantiles)  # Ensure non-negative float

  # Identify rows where mean is effectively zero or negative
  # These will have all quantiles as 0, as there's no admission to predict.
  is_mean_zero_or_negative = mean_values < 1e-9
  result_quantiles_matrix[is_mean_zero_or_negative, :] = 0

  # Process rows with non-zero positive mean_values
  active_indices = np.where(~is_mean_zero_or_negative)[0]

  if active_indices.size > 0:
    mu_active = mean_values[active_indices]
    phi_active_raw = overdispersion_param_phi_values[active_indices]

    # Ensure phi is above a floor for numerical stability and minimum dispersion.
    # This prevents division by zero for r_nbinom = 1/phi_nbinom and ensures a minimum spread.
    min_overdispersion_phi_floor = 1e-4  # Enforce a minimal overdispersion
    phi_active = np.maximum(min_overdispersion_phi_floor, phi_active_raw)

    # Calculate variance using the standard overdispersion model: Var = mu + phi * mu^2
    variance_active = mu_active + phi_active * (mu_active**2)

    # Enforce a minimum standard deviation (and thus variance) for small means.
    # This ensures that even for low predicted counts, the intervals have some reasonable width.
    min_std_dev_for_low_mean = 0.5
    variance_active = np.maximum(variance_active, min_std_dev_for_low_mean**2)

    # Initialize an array for float quantiles for all active rows
    quantiles_float_active = np.zeros((active_indices.size, num_quantiles))

    # Determine which rows are Poisson-like vs. NBinom-like.
    # Poisson distribution is chosen if:
    # 1. The mean is very small (Poisson is often more stable and appropriate for very low counts).
    # 2. Variance is not significantly greater than the mean (equi-dispersed, i.e., phi is negligible).
    poisson_threshold_mean = (
        1.0  # Means below this threshold are often better modeled by Poisson
    )
    is_low_mean_poisson = mu_active < poisson_threshold_mean
    is_equidispersed_poisson = (
        variance_active <= mu_active * 1.001
    )  # Allows for slight deviation from perfect equidispersion

    is_poisson_like = is_low_mean_poisson | is_equidispersed_poisson

    # --- Process Poisson-like rows ---
    poisson_like_indices_in_active = np.where(is_poisson_like)[0]
    if poisson_like_indices_in_active.size > 0:
      mu_poisson = mu_active[poisson_like_indices_in_active]
      try:
        # Ensure mu is strictly positive for poisson.ppf to avoid domain errors.
        mu_poisson_clipped = np.maximum(mu_poisson, 1e-9)
        poisson_quantiles = poisson.ppf(
            q_levels[np.newaxis, :], mu=mu_poisson_clipped[:, np.newaxis]
        )
        quantiles_float_active[poisson_like_indices_in_active, :] = (
            poisson_quantiles
        )
      except Exception as e:
        warnings.warn(
            'Vectorized poisson.ppf failed for'
            f' {poisson_like_indices_in_active.size} Poisson-like rows. Falling'
            f' back to minimal spread quantiles. Error: {e}',
            RuntimeWarning,
        )
        # Generate fallback quantiles for the failing rows
        fallback_quantiles_for_failing_rows = np.array(
            [_get_fallback_quantiles(m, q_levels) for m in mu_poisson]
        )
        quantiles_float_active[poisson_like_indices_in_active, :] = (
            fallback_quantiles_for_failing_rows
        )

    # --- Process Negative Binomial-like (overdispersed) rows ---
    nbinom_like_indices_in_active = np.where(~is_poisson_like)[0]
    if nbinom_like_indices_in_active.size > 0:
      mu_nbinom = mu_active[nbinom_like_indices_in_active]
      phi_nbinom = phi_active[nbinom_like_indices_in_active]

      # Parameters for scipy.stats.nbinom(n, p) where mean = n*(1-p)/p, var = n*(1-p)/p^2.
      # Using the standard parameterization for Var = mu + phi*mu^2:
      r_nbinom = 1 / phi_nbinom  # 'n' parameter in scipy.stats.nbinom
      p_nbinom = 1 / (
          1 + phi_nbinom * mu_nbinom
      )  # 'p' parameter in scipy.stats.nbinom

      # Numerical stability: ensure p is strictly between 0 and 1, and r is positive.
      p_nbinom = np.clip(p_nbinom, 1e-9, 1.0 - 1e-9)
      r_nbinom = np.maximum(
          r_nbinom, 1e-9
      )  # Ensures r is never zero or negative (phi is never infinite)

      try:
        nbinom_quantiles = nbinom.ppf(
            q_levels[np.newaxis, :],
            n=r_nbinom[:, np.newaxis],
            p=p_nbinom[:, np.newaxis],
        )
        quantiles_float_active[nbinom_like_indices_in_active, :] = (
            nbinom_quantiles
        )
      except Exception as e:
        # Fallback from NBinom to Poisson for problematic rows
        warnings.warn(
            'Vectorized nbinom.ppf failed for'
            f' {nbinom_like_indices_in_active.size} overdispersed rows.'
            f' Attempting fallback to Poisson. Error: {e}',
            RuntimeWarning,
        )
        mu_for_poisson_fallback = mu_nbinom
        try:
          mu_for_poisson_fallback_clipped = np.maximum(
              mu_for_poisson_fallback, 1e-9
          )
          poisson_fallback_quantiles = poisson.ppf(
              q_levels[np.newaxis, :],
              mu=mu_for_poisson_fallback_clipped[:, np.newaxis],
          )
          quantiles_float_active[nbinom_like_indices_in_active, :] = (
              poisson_fallback_quantiles
          )
        except Exception as e_inner:
          # Fallback from Poisson to minimal spread quantiles if Poisson also fails
          warnings.warn(
              'Vectorized poisson.ppf failed as secondary fallback for'
              f' {nbinom_like_indices_in_active.size} rows. Falling back to'
              f' minimal spread quantiles. Error: {e_inner}',
              RuntimeWarning,
          )
          # Generate fallback quantiles for the failing rows
          fallback_quantiles_for_failing_rows = np.array([
              _get_fallback_quantiles(m, q_levels)
              for m in mu_for_poisson_fallback
          ])
          # Fixed typo: nbinom_like_like_indices_in_active -> nbinom_like_indices_in_active
          quantiles_float_active[nbinom_like_indices_in_active, :] = (
              fallback_quantiles_for_failing_rows
          )

    # Assign the calculated float quantiles back to the result matrix, ensuring non-negativity and integer type.
    result_quantiles_matrix[active_indices, :] = np.maximum(
        0, np.round(quantiles_float_active)
    ).astype(int)

  return result_quantiles_matrix


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """

  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y

  # Process test_x on a copy to avoid modifying the original input
  test_x_processed = test_x.copy()

  # Ensure date columns are datetime objects consistently at the start for both train and test
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  test_x_processed['reference_date'] = pd.to_datetime(
      test_x_processed['reference_date']
  )
  test_x_processed['target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  )

  # Get the last training date for weeks_ahead calculation.
  train_end_date = train_df['target_end_date'].max()

  # --- Preprocessing: Prepare historical data for each location ---
  # Ensure population is at least 1 to prevent division by zero for per-capita calculations.
  train_df['population_safe'] = train_df['population'].apply(
      lambda x: max(1, x)
  )
  test_x_processed['population_safe'] = test_x_processed['population'].apply(
      lambda x: max(1, x)
  )

  # Calculate per-capita admissions for consistent modeling
  train_df['per_capita_admissions'] = (
      train_df[TARGET_STR] / train_df['population_safe']
  )

  # Sort by target_end_date to ensure 'last' picks the most recent and consistent order.
  train_df_sorted = train_df.sort_values(['location', 'target_end_date']).copy()

  # --- Optimized Historical Feature Calculation ---
  # 1. Last observed per-capita admissions
  last_per_capita_admissions = train_df_sorted.groupby('location')[
      'per_capita_admissions'
  ].last()

  # 2. Last observed EWMA per-capita admissions
  ewma_span = 12  # Adjusted: Increased from 8 to 12 for smoother signal and better stability
  ewma_per_capita_series = train_df_sorted.groupby('location')[
      'per_capita_admissions'
  ].transform(lambda x: x.ewm(span=ewma_span, adjust=False).mean())
  last_ewma_per_capita = ewma_per_capita_series.groupby(
      train_df_sorted['location']
  ).last()

  # 3. Per-capita trend per week (IMPROVED: using median of differences for robustness with better clipping)
  trend_window_size = 4  # Hyperparameter for trend calculation window

  def calculate_trend(group_series):
    # Take the most recent `trend_window_size` values and drop NaNs
    recent_values = group_series.iloc[-trend_window_size:].dropna().values

    if len(recent_values) < 2:
      return 0.0  # Not enough data for a trend

    # Calculate weekly changes and take the median, more robust than linear fit
    weekly_changes = np.diff(recent_values)
    if len(weekly_changes) == 0:
      return 0.0

    trend_val = np.median(weekly_changes)

    # New clipping strategy: combine absolute and relative caps for robustness.
    # This prevents unrealistically large trends, especially for very small per-capita values.

    # Absolute cap: Max allowed per-capita change per week (e.g., 750 cases per 1M population per week)
    MAX_ABS_TREND_PER_CAPITA_WEEKLY = 0.00075  # Adjusted: Increased from 0.0005 to allow slightly faster absolute changes
    # Relative cap: Max allowed percentage change per week relative to the current value (e.g., 30%)
    MAX_RELATIVE_TREND_PER_CAPITA_WEEKLY = 0.3

    # Apply absolute cap first to bound the trend value within reasonable absolute limits.
    trend_val = np.clip(
        trend_val,
        -MAX_ABS_TREND_PER_CAPITA_WEEKLY,
        MAX_ABS_TREND_PER_CAPITA_WEEKLY,
    )

    # Apply relative cap based on the most recent observed value.
    current_value = recent_values[-1] if len(recent_values) > 0 else 0.0
    # Ensure current_value is non-negative for percentage calculations.
    current_value_safe = np.maximum(current_value, 0.0)

    # Calculate the maximum allowed absolute change dictated by the relative cap.
    max_change_from_relative_cap = (
        MAX_RELATIVE_TREND_PER_CAPITA_WEEKLY * current_value_safe
    )

    # Further clip the trend using the relative cap. This ensures that even if the absolute cap
    # was wide, the trend won't imply an absurd percentage jump/drop from the current base.
    # For very small current_value_safe, this will become the dominant and stricter clip.
    trend_val = np.clip(
        trend_val, -max_change_from_relative_cap, max_change_from_relative_cap
    )

    # Ensure trend_val is finite (e.g., not NaN or Inf from any intermediate calculation)
    if not np.isfinite(trend_val):
      return 0.0
    return trend_val

  per_capita_trend_per_week = train_df_sorted.groupby('location')[
      'per_capita_admissions'
  ].apply(calculate_trend)

  # Combine all historical features into a single DataFrame
  historical_features = pd.DataFrame({
      'train_last_per_capita_admissions': last_per_capita_admissions,
      'train_per_capita_trend_per_week': per_capita_trend_per_week,
      'train_last_ewma_per_capita': last_ewma_per_capita,
  }).fillna(
      0.0
  )  # Fill NaN from locations that might not have enough historical data

  # --- Enrich test_x_processed with pre-computed historical data and temporal features ---
  test_x_enriched = test_x_processed.merge(
      historical_features, left_on='location', right_index=True, how='left'
  ).fillna(
      0.0
  )  # Fill NaN from merge if a location was not in train_df

  # Calculate weeks_ahead (forecast horizon from the last known data point in training)
  test_x_enriched['weeks_ahead'] = (
      (test_x_enriched['target_end_date'] - train_end_date).dt.days / 7
  ).astype(int)

  # --- Streamlined Seasonal Naive (F3) Feature Pre-calculation ---
  historical_for_seasonal = train_df_sorted[
      ['target_end_date', 'location', 'per_capita_admissions']
  ].copy()

  # Create mapping Series for 1-year and 2-year prior
  seasonal_1yr_map = historical_for_seasonal.set_index([
      'location',
      historical_for_seasonal['target_end_date'] + pd.Timedelta(weeks=52),
  ])['per_capita_admissions'].rename('seasonal_1yr_pc_admissions')

  seasonal_2yr_map = historical_for_seasonal.set_index([
      'location',
      historical_for_seasonal['target_end_date'] + pd.Timedelta(weeks=104),
  ])['per_capita_admissions'].rename('seasonal_2yr_pc_admissions')

  # Merge seasonal data into test_x_enriched using the new index for lookup
  test_x_enriched = test_x_enriched.merge(
      seasonal_1yr_map,
      left_on=['location', 'target_end_date'],
      right_index=True,
      how='left',
  )
  test_x_enriched = test_x_enriched.merge(
      seasonal_2yr_map,
      left_on=['location', 'target_end_date'],
      right_index=True,
      how='left',
  )

  # --- Modularized Forecaster Definitions and Quantile Generation ---
  # These forecasters simulate individual "expert" predictions, each with its own
  # logic for mean prediction and an associated overdispersion parameter (phi)
  # that increases with horizon. The weights assigned reflect an assumed relative
  # accuracy/reliability of each forecaster, simulating a diverse "crowd" of expertise.
  # The phi functions model Var = mu + phi * mu^2, where phi increases with horizon
  # to account for growing uncertainty further into the future.
  forecaster_configs: Dict[str, Dict[str, Any]] = {
      'forecaster_1_naive': {
          'weight': 0.15,
          'mean_func': (
              lambda df: np.maximum(0.0, df['train_last_per_capita_admissions'])
              * df['population_safe']
          ),
          # Naive forecasts generally have higher uncertainty, hence higher base phi and horizon sensitivity.
          'overdispersion_param_phi_func': (
              lambda df: (0.25 + 0.10 * df['horizon']) ** 2
          ),
      },
      'forecaster_2_trend': {
          'weight': 0.35,
          'mean_func': (
              lambda df: (
                  np.maximum(
                      0.0,
                      df['train_last_per_capita_admissions']
                      + df['train_per_capita_trend_per_week']
                      * df['weeks_ahead'],
                  )
              )
              * df['population_safe']
          ),
          # Trend forecasts are more reliable than naive but less than seasonal, with moderate phi growth.
          'overdispersion_param_phi_func': (
              lambda df: (0.18 + 0.06 * df['horizon']) ** 2
          ),
      },
      'forecaster_3_seasonal': {
          'weight': 0.25,
          'mean_func': (
              lambda df: (
                  np.maximum(
                      0.0,
                      df['seasonal_1yr_pc_admissions']
                      .fillna(df['seasonal_2yr_pc_admissions'])
                      .fillna(df['train_last_ewma_per_capita']),
                  )
              )
              * df['population_safe']
          ),
          # Seasonal forecasts often leverage strong patterns, leading to lower uncertainty (smaller phi).
          'overdispersion_param_phi_func': (
              lambda df: (0.10 + 0.04 * df['horizon']) ** 2
          ),
      },
      'forecaster_4_ewma': {
          'weight': 0.25,
          'mean_func': (
              lambda df: np.maximum(0.0, df['train_last_ewma_per_capita'])
              * df['population_safe']
          ),
          # EWMA offers good smoothing, providing a robust baseline with moderate uncertainty.
          'overdispersion_param_phi_func': (
              lambda df: (0.20 + 0.07 * df['horizon']) ** 2
          ),
      },
  }

  all_forecaster_quantiles = []
  forecaster_weights_list = []

  # Calculate means and generate quantiles for each forecaster in a loop
  for forecaster_name, config in forecaster_configs.items():
    mean_prediction_values = config['mean_func'](test_x_enriched).values
    overdispersion_param_phi_values = config['overdispersion_param_phi_func'](
        test_x_enriched
    ).values

    quantiles_matrix = generate_quantiles_from_simulation(
        mean_prediction_values, overdispersion_param_phi_values, QUANTILES
    )
    quantiles_df = pd.DataFrame(
        quantiles_matrix,
        index=test_x.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    all_forecaster_quantiles.append(quantiles_df)
    forecaster_weights_list.append(config['weight'])

  # --- Consensus Algorithm: Optimized Weighted Median of Quantiles ---
  # Stack all forecaster quantile predictions into a 3D numpy array
  # Shape: (num_forecasters, num_predictions, num_quantiles)
  stacked_quantiles_values = np.stack(
      [df.values for df in all_forecaster_quantiles], axis=0
  )
  weights_array = np.asarray(forecaster_weights_list)

  num_predictions, num_quantiles = (
      stacked_quantiles_values.shape[1],
      stacked_quantiles_values.shape[2],
  )

  # Reshape for efficient iteration: (num_forecasters, num_predictions * num_quantiles)
  # The current approach flattens the (num_predictions, num_quantiles) dimensions
  # and then applies weighted median to each flattened point across forecasters.
  # This is efficient and correct.
  reshaped_quantiles = stacked_quantiles_values.reshape(
      len(forecaster_configs), -1
  )

  # Apply weighted_median in a single loop over the flattened dimension
  consensus_quantiles_flat = np.array([
      _weighted_median(reshaped_quantiles[:, i], weights_array)
      for i in range(reshaped_quantiles.shape[1])
  ])

  # Reshape back to (num_predictions, num_quantiles)
  consensus_quantiles_values = consensus_quantiles_flat.reshape(
      num_predictions, num_quantiles
  )

  test_y_hat_quantiles = pd.DataFrame(
      consensus_quantiles_values,
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
  )

  # --- Post-processing: Enforce non-negativity and final monotonicity ---
  # Convert to integer counts and ensure all predictions are non-negative.
  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(int)
  test_y_hat_quantiles = np.maximum(test_y_hat_quantiles, 0)

  # Enforce monotonic increase across quantiles for each row.
  # This is crucial for valid probabilistic forecasts, ensuring that a higher quantile
  # always corresponds to a value greater than or equal to a lower quantile.
  quantiles_array = test_y_hat_quantiles.values
  quantiles_array = np.maximum.accumulate(quantiles_array, axis=1)
  test_y_hat_quantiles.loc[:, :] = quantiles_array

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
