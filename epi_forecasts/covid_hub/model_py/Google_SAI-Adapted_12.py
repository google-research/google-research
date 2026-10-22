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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-multiple-import,g-doc-return-or-yield,g-doc-args

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
MODEL_NAME = 'Google_SAI-Adapted_12'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
import warnings

TARGET_STR = 'Total COVID-19 Admissions'

# QUANTILES is defined globally in the notebook's preamble.

# Core principles of the climatological forecasting method:
# - 1. Temporal Alignment & Population-Normalized Profiling: Aggregate historical hospitalization data and align observations by a seasonal index (Epidemiological Week). To account for population differences, admissions are normalized by population (per 100k) before statistical profiling. Robust handling of zero or near-zero populations is included.
# - 2. Statistical Profiling with Hierarchical Fallback and Time-Weighting: For each location and future Epi Week, compute time-weighted central tendency (mean) and dispersion (variance) of historical population-normalized admission rates. Stronger importance is given to more recent observations. A robust hierarchical fallback strategy is employed: preference is given to location-specific seasonal statistics, then general seasonal statistics, and finally global statistics, especially when more specific historical data is sparse or unreliable.
# - 3. Projection with Anchored Short-Term Trend Blending: Populate the forecast trajectory by blending a short-term projection (extrapolated from the latest observed rate using recent weekly trend) with the climatological mean for the corresponding future dates. This blend smoothly transitions from favoring recent momentum for short horizons to relying on the climatological average for longer horizons, ensuring forecasts are anchored to the latest data and regress towards the long-term seasonal pattern.
# - 4. Uncertainty Construction with Dynamic Distribution Selection: Generate prediction intervals by applying historical dispersion parameters (adjusted for forecast horizon) to the point forecast. Dynamically select a suitable discrete probability distribution (Poisson for equidispersion/underdispersion, Negative Binomial for overdispersion) for generating predictions. Robustness adjustments include ensuring non-negative means and variances, explicit handling for zero-mean forecasts, and applying a minimum variance floor to prevent overly confident forecasts for positive means.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """

  # Configuration for robust parameter estimation
  MIN_OBSERVATIONS_FOR_LOC_EPIWEEK = 5
  MIN_OBSERVATIONS_FOR_EPIWEEK = 15
  MIN_OBSERVATIONS_FOR_GLOBAL = 1
  MIN_WEIGHT_SUM_FOR_STATS = 1.0  # Minimum sum of weights required for mean/variance to be considered valid
  MAX_NBINOM_N_APPROX_POISSON = 1e4
  MIN_POPULATION_FOR_RATE_CALC = (
      1.0  # Minimum population to calculate a rate; avoids division by zero
  )

  # Constants for numerical stability and parameter validation
  NUMERICAL_STABILITY_EPSILON = (
      1e-6  # General epsilon for small positive numbers
  )
  WEIGHT_SUM_THRESHOLD_EPSILON = 1e-9  # Epsilon for very small sums of weights
  NBINOM_N_LOWER_CLIP = (
      0.1  # Minimum value for the 'n' parameter of Negative Binomial
  )
  NBINOM_P_LOWER_CLIP = (
      NUMERICAL_STABILITY_EPSILON  # Minimum value for the 'p' parameter
  )
  NBINOM_P_UPPER_CLIP = (
      1.0 - NUMERICAL_STABILITY_EPSILON
  )  # Maximum value for the 'p' parameter

  # Revised factors for improved responsiveness and uncertainty capture
  DECAY_FACTOR = 0.1  # For exponential time weighting
  HORIZON_VARIANCE_GROWTH_FACTOR = 0.75

  # NEW: Minimum variance multiplier to ensure reasonable dispersion for non-zero means.
  MIN_VARIANCE_MULTIPLIER_IF_MU_POSITIVE = 1.05

  # NEW: Fixed number of weeks over which the short-term trend blends into climatology
  TREND_DECAY_WEEKS = 6  # Trend influence decays over 6 weeks.

  quantile_cols = [f'quantile_{q}' for q in QUANTILES]

  TARGET_RATE_STR = 'admissions_per_100k_population'

  # Nested helper function for calculating weighted stats (mean, variance)
  def _calculate_weighted_stats(
      group_df,
      data_col,
      weight_col,
      min_effective_weight_sum,
      min_unweighted_obs,
  ):
    data_series = group_df[data_col]
    weight_series = group_df[weight_col]

    # Filter out NaNs in either series before calculation
    valid_data_mask = data_series.notna() & weight_series.notna()
    valid_data = data_series[valid_data_mask]
    valid_weights = weight_series[valid_data_mask]

    W_sum = valid_weights.sum()

    # If effective weight sum is too low (strictly less than threshold) or near zero,
    # or unweighted observation count is too low, return NaN for stats.
    if (
        len(valid_data) < min_unweighted_obs
        or W_sum < min_effective_weight_sum
        or W_sum <= WEIGHT_SUM_THRESHOLD_EPSILON
    ):
      return pd.Series({
          'mean_rate': np.nan,
          'var_rate': np.nan,
      })

    # Calculate weighted mean
    weighted_mean_val = (valid_data * valid_weights).sum() / W_sum

    # Calculate weighted variance (using population-like denominator W_sum)
    # Using ddof=0 for population variance, consistent with earlier code, or could use ddof=1 for sample var
    M2 = (valid_weights * (valid_data - weighted_mean_val) ** 2).sum()
    variance = M2 / W_sum

    return pd.Series({
        'mean_rate': weighted_mean_val,
        'var_rate': np.maximum(
            0.0, variance
        ),  # Ensure variance is non-negative
    })

  # --- Data Preparation ---
  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y

  # Explicitly filter out rows with NaN population in training data.
  # This ensures rate calculations are based on valid population figures.
  original_train_rows = len(train_df)
  train_df.dropna(subset=['population'], inplace=True)
  if len(train_df) < original_train_rows:
    warnings.warn(
        f'Filtered {original_train_rows - len(train_df)} training rows due to'
        ' NaN population.',
        UserWarning,
    )

  # Ensure target_end_date is datetime type
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  train_df['epiweek'] = (
      train_df['target_end_date'].dt.isocalendar().week.astype(int)
  )

  # Robustly handle population divisor for rate calculation
  train_df['population_for_rate'] = np.maximum(
      train_df['population'], MIN_POPULATION_FOR_RATE_CALC
  )
  train_df['population_divisor'] = train_df['population_for_rate'] / 100_000

  train_df[TARGET_RATE_STR] = (
      train_df[TARGET_STR] / train_df['population_divisor']
  )

  # CRITICAL: If actual population is 0, then admissions must be 0, and rate is 0.
  # This overrides any potential division by a 'minimum' population.
  train_df.loc[train_df['population'] == 0, TARGET_RATE_STR] = 0.0

  test_df_processed = test_x.copy()
  # Ensure target_end_date is datetime type
  test_df_processed['target_end_date'] = pd.to_datetime(
      test_df_processed['target_end_date']
  )
  test_df_processed['epiweek'] = (
      test_df_processed['target_end_date'].dt.isocalendar().week.astype(int)
  )

  # Store original population for final zero-population override
  test_df_processed['original_population'] = test_df_processed['population']
  # Robustly handle population divisor for test data
  test_df_processed['population_for_rate'] = np.maximum(
      test_df_processed['population'], MIN_POPULATION_FOR_RATE_CALC
  )
  test_df_processed['population_divisor'] = (
      test_df_processed['population_for_rate'] / 100_000
  )

  # Explicitly ensure that for locations with 0 original population, the divisor for rates implies zero activity.
  test_df_processed.loc[
      test_df_processed['original_population'] == 0, 'population_divisor'
  ] = np.inf

  # --- Time-Weighted Statistical Profiling ---
  # Define weighting scheme: exponential decay based on days from the latest training date
  train_end_date_max = train_df['target_end_date'].max()
  if pd.isna(train_end_date_max):  # Handle empty train_df
    train_df['time_weight'] = NUMERICAL_STABILITY_EPSILON
  else:
    train_df['time_weight'] = np.exp(
        -DECAY_FACTOR
        * (train_end_date_max - train_df['target_end_date']).dt.days
    )
    train_df['time_weight'] = np.maximum(
        train_df['time_weight'], NUMERICAL_STABILITY_EPSILON
    )  # Ensure weights are not extremely small/zero

  # Calculate statistics only if train_df is not empty after NaN filtering
  if train_df.empty:
    global_stats = pd.Series({'mean_rate': np.nan, 'var_rate': np.nan})
    epiweek_stats = pd.DataFrame()
    loc_epiweek_stats = pd.DataFrame()
  else:
    # 1. Global Statistics (mean_rate, var_rate)
    global_stats = _calculate_weighted_stats(
        train_df,
        TARGET_RATE_STR,
        'time_weight',
        MIN_WEIGHT_SUM_FOR_STATS,
        MIN_OBSERVATIONS_FOR_GLOBAL,
    )

    # 2. Epiweek-Specific Statistics (mean_rate, var_rate)
    epiweek_stats = train_df.groupby('epiweek').apply(
        lambda x: _calculate_weighted_stats(
            x,
            TARGET_RATE_STR,
            'time_weight',
            MIN_WEIGHT_SUM_FOR_STATS,
            MIN_OBSERVATIONS_FOR_EPIWEEK,
        )
    )

    # 3. Location-Epiweek-Specific Statistics (mean_rate, var_rate)
    loc_epiweek_stats = train_df.groupby(['location', 'epiweek']).apply(
        lambda x: _calculate_weighted_stats(
            x,
            TARGET_RATE_STR,
            'time_weight',
            MIN_WEIGHT_SUM_FOR_STATS,
            MIN_OBSERVATIONS_FOR_LOC_EPIWEEK,
        )
    )

  # --- Hierarchical Fallback for Rate Parameters ---
  # Prioritize (1) location-epiweek, then (2) epiweek, then (3) global statistics.
  final_params = test_df_processed.copy()

  # Merge location-epiweek specific stats (most specific)
  final_params = final_params.merge(
      loc_epiweek_stats.reset_index(),
      on=['location', 'epiweek'],
      how='left',
      suffixes=('', '_loc_epi'),
  ).rename(
      columns={'mean_rate': 'mean_rate_loc_epi', 'var_rate': 'var_rate_loc_epi'}
  )

  # Merge epiweek specific stats
  final_params = final_params.merge(
      epiweek_stats.reset_index(),
      on='epiweek',
      how='left',
      suffixes=('', '_epi'),
  ).rename(columns={'mean_rate': 'mean_rate_epi', 'var_rate': 'var_rate_epi'})

  # Apply hierarchical filling: fill from most specific to least specific
  final_params['mean_rate_climatological'] = (
      final_params['mean_rate_loc_epi']
      .fillna(final_params['mean_rate_epi'])
      .fillna(global_stats['mean_rate'])
  )

  final_params['var_rate_climatological'] = (
      final_params['var_rate_loc_epi']
      .fillna(final_params['var_rate_epi'])
      .fillna(global_stats['var_rate'])
  )

  # Drop the intermediate columns used for hierarchical merging
  final_params = final_params.drop(
      columns=[
          col
          for col in final_params.columns
          if '_loc_epi' in str(col) or '_epi' in str(col)
      ]
  )

  # --- Calculate Short-Term Trend Adjustment (on Rate Scale) ---
  recent_data_for_trend = pd.DataFrame()
  if not train_df.empty:
    train_sorted = train_df.sort_values(by=['location', 'target_end_date'])

    # Get the latest observation for each location
    # This is the anchor point for trend projections
    latest_obs = (
        train_sorted.groupby('location')
        .tail(1)
        .set_index('location')[[TARGET_RATE_STR]]
    )
    latest_obs.rename(columns={TARGET_RATE_STR: 'latest_rate'}, inplace=True)

    # Get the second-to-latest observation for each location to calculate trend
    # Use .apply(lambda x: x.iloc[-2] if len(x) >= 2 else pd.Series()) for robustness against groups with < 2 items
    second_latest_obs_raw = train_sorted.groupby('location').apply(
        lambda x: x.iloc[-2] if len(x) >= 2 else pd.Series(dtype=float)
    )
    if not second_latest_obs_raw.empty:
      second_latest_obs = second_latest_obs_raw.set_index('location')[
          TARGET_RATE_STR
      ].rename('prev_week_rate')
      recent_data_for_trend = latest_obs.join(second_latest_obs, how='left')
    else:  # If no location has at least two observations
      recent_data_for_trend = latest_obs
      recent_data_for_trend['prev_week_rate'] = np.nan

    # Calculate the trend for each location (change in rate)
    # Trend is 0 if data is missing or insufficient (NaN after pivot)
    trend_mask = (
        recent_data_for_trend['latest_rate'].notna()
        & recent_data_for_trend['prev_week_rate'].notna()
    )
    recent_data_for_trend['weekly_rate_trend'] = 0.0  # Default to 0
    recent_data_for_trend.loc[trend_mask, 'weekly_rate_trend'] = (
        recent_data_for_trend.loc[trend_mask, 'latest_rate']
        - recent_data_for_trend.loc[trend_mask, 'prev_week_rate']
    )
  else:
    # If train_df is empty, recent_data_for_trend will be empty and subsequent merges will produce NaNs.
    pass  # Handled by fillna below

  # Merge latest_rate and weekly_rate_trend into final_params
  final_params = final_params.merge(
      recent_data_for_trend[['latest_rate', 'weekly_rate_trend']],
      on='location',
      how='left',
  )

  # Fill NaNs for locations with no recent data (e.g., location in test_x but not train_x)
  # Default to 0 for rate and trend, which makes sense if there's no historical basis.
  final_params['latest_rate'] = final_params['latest_rate'].fillna(0.0)
  final_params['weekly_rate_trend'] = final_params['weekly_rate_trend'].fillna(
      0.0
  )

  # Calculate a blend factor: 0 at horizon 0, 1 at TREND_DECAY_WEEKS or more.
  # This factor determines how much to weigh climatology vs. recent trend.
  final_params['climatology_blend_factor'] = np.clip(
      final_params['horizon'] / TREND_DECAY_WEEKS, 0.0, 1.0
  )

  # Calculate the rate projected from the latest observed value using the weekly trend
  # We use (horizon + 1) because horizon=0 corresponds to the first week *after* the latest observation.
  final_params['projected_from_latest_rate'] = (
      final_params['latest_rate']
      + (final_params['horizon'] + 1) * final_params['weekly_rate_trend']
  )

  # Blend the projected rate with the climatological mean.
  # For horizon=0, climatology_blend_factor will be 0, so mean_rate_adjusted will be projected_from_latest_rate.
  # As horizon increases, climatology_blend_factor increases, blending towards mean_rate_climatological.
  final_params['mean_rate_adjusted'] = (
      final_params['projected_from_latest_rate']
      * (1 - final_params['climatology_blend_factor'])
      + final_params['mean_rate_climatological']
      * final_params['climatology_blend_factor']
  )

  # Ensure adjusted mean rate is non-negative
  final_params['mean_rate_adjusted'] = np.maximum(
      0.0, final_params['mean_rate_adjusted']
  )

  # --- Convert Rate Parameters to Count Parameters and Derive Distribution Params (Vectorized) ---

  # Calculate climatological count-scale mean from the adjusted rate
  final_params['mu_count_adjusted'] = (
      final_params['mean_rate_adjusted'] * final_params['population_divisor']
  )

  # Calculate base variance from climatological rate variance
  final_params['base_var_count_climatological'] = final_params[
      'var_rate_climatological'
  ] * (final_params['population_divisor'] ** 2)

  # Adjust variance based on forecast horizon
  final_params['var_count_adjusted'] = final_params[
      'base_var_count_climatological'
  ] * (1 + HORIZON_VARIANCE_GROWTH_FACTOR * final_params['horizon'])

  # CRITICAL FIX: Override for locations with 0 original population OR where mu_count_adjusted is 0.
  # For these locations, the true admissions must be 0, regardless of calculated rates.
  # Also, if mean is 0, variance must be 0 for count data.
  zero_mu_or_pop_mask = (final_params['mu_count_adjusted'] == 0) | (
      final_params['original_population'] == 0
  )
  final_params.loc[zero_mu_or_pop_mask, 'var_count_adjusted'] = 0.0
  final_params.loc[zero_mu_or_pop_mask, 'mu_count_adjusted'] = (
      0.0  # Ensure mean is also 0 if pop is 0
  )

  # NEW IMPROVEMENT: Ensure a minimum variance if mu_count_adjusted is positive but var_count_adjusted is pathologically low.
  # This avoids overly confident (deterministic) non-zero forecasts.
  low_variance_and_positive_mu_mask = (
      final_params['mu_count_adjusted'] > 0
  ) & (
      final_params['var_count_adjusted']
      < final_params['mu_count_adjusted']
      * MIN_VARIANCE_MULTIPLIER_IF_MU_POSITIVE
  )

  final_params.loc[low_variance_and_positive_mu_mask, 'var_count_adjusted'] = (
      np.maximum(
          final_params.loc[
              low_variance_and_positive_mu_mask, 'var_count_adjusted'
          ],
          final_params.loc[
              low_variance_and_positive_mu_mask, 'mu_count_adjusted'
          ]
          * MIN_VARIANCE_MULTIPLIER_IF_MU_POSITIVE,
      )
  )

  # Initialize columns for distribution parameters (default to NaN for unmodeled/zero cases)
  final_params['dist_type'] = (
      'unknown'  # Placeholder, will be 'poisson' or 'nbinom'
  )
  final_params['n'] = np.nan
  final_params['p'] = np.nan
  final_params['poisson_lambda'] = (
      np.nan
  )  # Renamed for clarity and to avoid keyword collision

  # Identify rows where parameters are valid and non-zero for modeling
  rows_to_model_mask = (
      (~final_params['mu_count_adjusted'].isna())
      & (~final_params['var_count_adjusted'].isna())
      & (~zero_mu_or_pop_mask)
  )

  if rows_to_model_mask.any():
    mu = final_params.loc[rows_to_model_mask, 'mu_count_adjusted']
    var = final_params.loc[rows_to_model_mask, 'var_count_adjusted']

    # --- Determine Poisson candidates (var <= mu) ---
    poisson_candidates_mask_local = var <= mu
    final_params.loc[
        rows_to_model_mask & poisson_candidates_mask_local, 'dist_type'
    ] = 'poisson'
    final_params.loc[
        rows_to_model_mask & poisson_candidates_mask_local, 'poisson_lambda'
    ] = mu[poisson_candidates_mask_local]

    # --- Determine Negative Binomial candidates (var > mu) ---
    nbinom_candidates_mask_local = var > mu
    nbinom_rows_global_idx = final_params.index[
        rows_to_model_mask & nbinom_candidates_mask_local
    ]

    if not nbinom_rows_global_idx.empty:
      mu_nbinom = mu[nbinom_candidates_mask_local]
      var_nbinom = var[nbinom_candidates_mask_local]

      # Robustness: Ensure denominator for 'n' parameter calculation is strictly positive.
      denom_for_n = np.maximum(
          NUMERICAL_STABILITY_EPSILON, var_nbinom - mu_nbinom
      )
      n_calculated = mu_nbinom**2 / denom_for_n
      # Robustness: Ensure denominator for 'p' parameter calculation is strictly positive.
      p_calculated = mu_nbinom / np.maximum(
          NUMERICAL_STABILITY_EPSILON, var_nbinom
      )

      # --- Handle cases where nbinom 'n' parameter is extremely large, approximating Poisson ---
      reclassify_to_poisson_nbinom_mask_local = (
          n_calculated >= MAX_NBINOM_N_APPROX_POISSON
      )

      # Reclassify to Poisson for these specific NBinom candidates
      reclassified_poisson_global_idx = nbinom_rows_global_idx[
          reclassify_to_poisson_nbinom_mask_local
      ]
      final_params.loc[reclassified_poisson_global_idx, 'dist_type'] = 'poisson'
      final_params.loc[reclassified_poisson_global_idx, 'poisson_lambda'] = (
          mu_nbinom[reclassify_to_poisson_nbinom_mask_local]
      )  # Use their mu

      # Assign NBinom parameters for rows that *remain* Negative Binomial
      remaining_nbinom_global_idx = nbinom_rows_global_idx[
          ~reclassify_to_poisson_nbinom_mask_local
      ]

      if not remaining_nbinom_global_idx.empty:
        # Clip n parameter to ensure it's positive and not excessively large for numerical stability.
        n_param_final = np.clip(
            n_calculated[~reclassify_to_poisson_nbinom_mask_local],
            NBINOM_N_LOWER_CLIP,
            MAX_NBINOM_N_APPROX_POISSON,
        )
        # Clip p parameter to ensure it's strictly between 0 and 1, vital for scipy.stats.nbinom.
        p_param_final = np.clip(
            p_calculated[~reclassify_to_poisson_nbinom_mask_local],
            NBINOM_P_LOWER_CLIP,
            NBINOM_P_UPPER_CLIP,
        )

        final_params.loc[remaining_nbinom_global_idx, 'dist_type'] = 'nbinom'
        final_params.loc[remaining_nbinom_global_idx, 'n'] = n_param_final
        final_params.loc[remaining_nbinom_global_idx, 'p'] = p_param_final

  # --- Generate Quantiles (Vectorized) ---
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=quantile_cols, dtype=float
  )

  # Explicitly handle cases where mu_count_adjusted is 0 or parameters couldn't be derived.
  # All quantiles should be 0 for these.
  zero_or_unmodeled_mask = (final_params['mu_count_adjusted'] == 0) | (
      final_params['dist_type'] == 'unknown'
  )
  test_y_hat_quantiles.loc[zero_or_unmodeled_mask, quantile_cols] = 0.0

  # Only process rows where a distribution type was successfully assigned and mean > 0
  rows_to_process_ppf_mask = ~zero_or_unmodeled_mask

  if rows_to_process_ppf_mask.any():
    # Suppress RuntimeWarnings from scipy.stats.ppf due to parameter clipping, as parameters are now validated.
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)

      df_to_process = final_params[rows_to_process_ppf_mask]

      # Pre-extract all necessary parameters as NumPy arrays for efficiency
      all_dist_type = df_to_process['dist_type'].values
      all_n_param = df_to_process['n'].values
      all_p_param = df_to_process['p'].values
      all_poisson_lambda_param = df_to_process[
          'poisson_lambda'
      ].values  # Use renamed parameter

      # Identify rows belonging to each distribution type for vectorized operations
      poisson_rows_idx_relative = np.where(all_dist_type == 'poisson')[0]
      nbinom_rows_idx_relative = np.where(all_dist_type == 'nbinom')[0]

      poisson_lambda_params = all_poisson_lambda_param[
          poisson_rows_idx_relative
      ]
      nbinom_n_params = all_n_param[nbinom_rows_idx_relative]
      nbinom_p_params = all_p_param[nbinom_rows_idx_relative]

      # --- Robust Parameter Validation before PPF calls ---
      # Poisson: Ensure lambda is non-negative and finite
      poisson_lambda_params = np.nan_to_num(
          poisson_lambda_params, nan=0.0, posinf=0.0, neginf=0.0
      )
      poisson_lambda_params = np.maximum(0.0, poisson_lambda_params)

      # Negative Binomial: Ensure n > 0, p in (0,1) and are finite
      nbinom_n_params = np.nan_to_num(
          nbinom_n_params,
          nan=NBINOM_N_LOWER_CLIP,
          posinf=MAX_NBINOM_N_APPROX_POISSON,
          neginf=NBINOM_N_LOWER_CLIP,
      )
      nbinom_n_params = np.maximum(
          NBINOM_N_LOWER_CLIP, nbinom_n_params
      )  # Ensure > 0

      nbinom_p_params = np.nan_to_num(
          nbinom_p_params,
          nan=0.5,
          posinf=NBINOM_P_UPPER_CLIP,
          neginf=NBINOM_P_LOWER_CLIP,
      )
      nbinom_p_params = np.clip(
          nbinom_p_params, NBINOM_P_LOWER_CLIP, NBINOM_P_UPPER_CLIP
      )  # Ensure (0,1)

      for q_idx, q_val in enumerate(QUANTILES):
        q_col = quantile_cols[q_idx]

        current_quantile_preds_for_processed_rows = np.full(
            len(df_to_process.index), np.nan
        )

        # Apply Poisson ppf to relevant rows
        if len(poisson_rows_idx_relative) > 0:
          current_quantile_preds_for_processed_rows[
              poisson_rows_idx_relative
          ] = poisson.ppf(q_val, poisson_lambda_params)

        # Apply Negative Binomial ppf to relevant rows
        if len(nbinom_rows_idx_relative) > 0:
          current_quantile_preds_for_processed_rows[
              nbinom_rows_idx_relative
          ] = nbinom.ppf(q_val, nbinom_n_params, nbinom_p_params)

        test_y_hat_quantiles.loc[rows_to_process_ppf_mask, q_col] = (
            current_quantile_preds_for_processed_rows
        )

  # --- Post-processing: Non-negative, Integer Conversion, and Monotonicity (VECTORIZED) ---

  # Ensure any remaining NaNs are filled with 0 before conversion to int.
  predicted_quantiles_array = test_y_hat_quantiles.fillna(0).values

  # Ensure non-negative predictions by clipping at 0 (vectorized).
  predicted_quantiles_array = np.maximum(0, predicted_quantiles_array)

  # Convert to integers (flooring) as hospital admissions are count data (vectorized).
  predicted_quantiles_array = predicted_quantiles_array.astype(int)

  # Enforce monotonicity of quantiles row-wise (vectorized).
  for i in range(1, len(QUANTILES)):
    predicted_quantiles_array[:, i] = np.maximum(
        predicted_quantiles_array[:, i], predicted_quantiles_array[:, i - 1]
    )

  # Convert the processed NumPy array back to a DataFrame with the original index and columns.
  predicted_quantiles_df = pd.DataFrame(
      predicted_quantiles_array, index=test_x.index, columns=quantile_cols
  )

  return predicted_quantiles_df


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
