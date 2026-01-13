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
MODEL_NAME = 'Google_SAI-Adapted_10'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# --- Core Principles of the PROF-like Method (Interpreted for Statistical Model) ---
#
# 1. Statistical Approximation of Mechanistic Dynamics with Arc-Tangent Trends:
#    A LightGBM model statistically approximates epidemiological system dynamics,
#    utilizing engineered time-series features including multiple arc-tangent terms
#    to represent a time-varying transmission rate. This implementation includes
#    ten distinct arc-tangent terms (with varying shifts and spreads from weeks to years)
#    to capture non-linear trends across various timescales, **with optimized parameters for
#    better capture of diverse epidemic phases.**
#
# 2. Augmented Training with Weighted Synthetic History:
#    Historical ILINet data is transformed to generate synthetic hospitalization history,
#    which augments the primary training set and receives a reduced fitting weight (0.5).
#    Lagged ILI features within the model serve as a statistical proxy for 'future'
#    insights typically gained from a Method Of Analogs (MOA).
#    **The ILI transformation learning is made more robust for low signal values.**
#
# 3. Direct Quantile Forecasting with Signal-Dependent Baseline:
#    LightGBM's quantile regression objective directly produces probabilistic forecasts
#    for each required quantile, serving as a statistical proxy for MCMC inference and
#    stochastic simulations. For locations or time periods with low epidemiological signal,
#    predictions are robustly substituted or dynamically scaled by a data-driven
#    statistical baseline, ensuring calibrated forecasts even in low-incidence scenarios.
#    **The statistical baselines for low-signal cases are re-calibrated for improved
#    robustness and consistency with per-capita scaling.**

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from typing import Any, Dict, List, Optional

TARGET_PER_CAPITA_STR = f'{TARGET_STR}_per_capita'
TARGET_MODEL_STR = (  # The actual target for LGBM
    f'{TARGET_PER_CAPITA_STR}_log1p'
)

# --- Configuration Constants (Assumed to be globally available) ---
# QUANTILES, horizons, ilinet_hhs, ilinet, ilinet_state, locations,
# sample_submission_df, example_train_x, example_train_y, example_test_x,
# example_reference_date are assumed to be accessible in the global scope.


def _engineer_features(
    df, min_overall_date = None
):
  """Extracts time-based features and adds population, including new season and trend features."""
  df = df.copy()

  # Ensure target_end_date is datetime
  if 'target_end_date' in df.columns:
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df['year'] = df['target_end_date'].dt.year
    df['month'] = df['target_end_date'].dt.month
    df['week_of_year'] = df['target_end_date'].dt.isocalendar().week.astype(int)

    # Cyclic features for week of year
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # NEW FEATURE: season_year (e.g., '2023/2024')
    df['season_year'] = df['target_end_date'].apply(
        lambda d: f'{d.year}/{d.year + 1}'
        if d.month >= 8
        else f'{d.year - 1}/{d.year}'
    )

    # NEW FEATURE: weeks_since_start
    if min_overall_date:
      df['weeks_since_start'] = (
          df['target_end_date'] - min_overall_date
      ).dt.days // 7
      df['weeks_since_start_squared'] = df['weeks_since_start'] ** 2

      # ADDED ARC-TANGENT TREND FEATURES (Principle 1)
      # These ten arc-tangent terms capture non-linear trends across distinct timescales
      # (e.g., very short-term, monthly, seasonal, multi-seasonal, and multi-year).
      # They serve as a statistical approximation of the time-dependent 'Beta' transmission-rate coefficient.
      # IMPROVED: More diverse shifts and spreads for better capture of epidemic dynamics.
      ws = df['weeks_since_start']
      df['arctan_trend_1'] = np.arctan(
          (ws - 26) / 4
      )  # Mid-year shift (e.g., peak of flu season), very sharp onset/decline
      df['arctan_trend_2'] = np.arctan(
          (ws - 52) / 13
      )  # Annual cycle, quarterly spread
      df['arctan_trend_3'] = np.arctan(
          (ws - 0) / 26
      )  # Overall trend from start, half-year spread
      df['arctan_trend_4'] = np.arctan(
          (ws - 104) / 52
      )  # Bi-annual cycle, annual spread
      df['arctan_trend_5'] = np.arctan(
          (ws - 78) / 26
      )  # 1.5 year cycle, half-year spread
      df['arctan_trend_6'] = np.arctan(
          (ws - 156) / 52
      )  # 3 year cycle, annual spread
      df['arctan_trend_7'] = np.arctan(
          (ws - 208) / 104
      )  # 4 year cycle, bi-annual spread
      df['arctan_trend_8'] = np.arctan(
          (ws - 312) / 104
      )  # 6 year cycle, bi-annual spread
      df['arctan_trend_9'] = np.arctan(
          (ws - 4) / 2
      )  # Very early phase, very sharp (kept one such term for rapid changes)
      df['arctan_trend_10'] = np.arctan(
          (ws - 416) / 208
      )  # 8 year cycle, 4-year spread

      # NEW FEATURE: season_progress - weeks into the flu season
      # Flu season often starts around week 40 (October). Adjust week_of_year to start from this point.
      season_start_week_offset = 40
      df['season_progress'] = (
          df['week_of_year'] - season_start_week_offset + 52
      ) % 52 + 1
    else:
      df['weeks_since_start'] = 0
      df['weeks_since_start_squared'] = 0
      # Initialize all arctan features
      for i in range(1, 11):
        df[f'arctan_trend_{i}'] = 0.0
      df['season_progress'] = 0
  return df


def _add_lagged_features_from_tracker(
    df_to_add_lags_to,
    lags,
    location_col,
    target_col_for_lags,
    prediction_tracker_lookup_series,
):
  """Adds lagged features to df_to_add_lags_to by looking up values in prediction_tracker_lookup_series."""
  df_with_lags = df_to_add_lags_to.copy()

  for lag in lags:
    lagged_date = df_with_lags['target_end_date'] - pd.Timedelta(weeks=lag)

    # Create a MultiIndex for mapping
    lookup_keys = pd.MultiIndex.from_arrays(
        [df_with_lags[location_col], lagged_date],
        names=[location_col, 'target_end_date'],
    )

    df_with_lags[f'{target_col_for_lags}_lag_{lag}'] = lookup_keys.map(
        prediction_tracker_lookup_series
    )

  return df_with_lags


def _add_ili_lagged_features(
    df,
    ili_lags,
    location_col,
    ili_signal_history_lookup,
    target_col_for_lags,  # Added to specify the name of the ILI column for lags
    fill_value = 0.0,
):
  """Adds lagged ILI signal features to a DataFrame using a historical lookup series."""
  df_with_ili_lags = df.copy()

  for lag in ili_lags:
    lagged_date = df_with_ili_lags['target_end_date'] - pd.Timedelta(weeks=lag)

    lookup_keys = pd.MultiIndex.from_arrays(
        [df_with_ili_lags[location_col], lagged_date],
        names=[location_col, 'target_end_date'],
    )

    df_with_ili_lags[f'{target_col_for_lags}_lag_{lag}'] = lookup_keys.map(
        ili_signal_history_lookup
    )

  # Fill any NaNs from lags (e.g., for early dates or future forecasts beyond ILI history)
  ili_lag_cols = [f'{target_col_for_lags}_lag_{lag}' for lag in ili_lags]
  df_with_ili_lags[ili_lag_cols] = df_with_ili_lags[ili_lag_cols].fillna(
      fill_value
  )

  return df_with_ili_lags


def _learn_ili_transformation(
    full_train_data,
    ili_data_for_learning,
    target_col,
    ili_col,  # This ili_col should now be per-capita
    min_ili_threshold = 1e-7,  # CHANGED: from int=1 to float=1e-7 for per-capita consistency
    min_overlap_points = 20,  # Increased for robustness
):
  """Learns a statistical transformation from ILI to target for each location.

  Uses log1p transformation internally for LinearRegression to handle skewed
  count data.
  """
  transformations = {}

  full_train_data = full_train_data.copy()
  ili_data_for_learning = ili_data_for_learning.copy()
  full_train_data['target_end_date'] = pd.to_datetime(
      full_train_data['target_end_date']
  )
  ili_data_for_learning['target_end_date'] = pd.to_datetime(
      ili_data_for_learning['target_end_date']
  )

  overlap_df = pd.merge(
      full_train_data[
          ['target_end_date', 'location', target_col, 'population']
      ],
      ili_data_for_learning[
          ['target_end_date', 'location', ili_col]
      ],  # Uses the passed ili_col
      on=['target_end_date', 'location'],
      how='inner',
  )

  # IMPROVEMENT: Ensure effective population is non-zero for per-capita calculation
  overlap_df['effective_population'] = np.where(
      overlap_df['population'] == 0, 1.0, overlap_df['population']
  )
  overlap_df[TARGET_PER_CAPITA_STR] = (
      overlap_df[target_col] / overlap_df['effective_population']
  )
  # Ensure per-capita target is non-negative after division
  overlap_df[TARGET_PER_CAPITA_STR] = np.maximum(
      0, overlap_df[TARGET_PER_CAPITA_STR]
  )

  # Calculate a global mean per-capita target for a robust fallback
  # Ensure global mean is at least a very small positive number if data exists, to avoid exact zero predictions
  global_mean_target_per_capita = (
      overlap_df[TARGET_PER_CAPITA_STR].mean() if not overlap_df.empty else 0.0
  )
  global_mean_target_per_capita = np.maximum(
      global_mean_target_per_capita, 1e-9
  )  # Ensure minimum positive value

  # MODIFIED: Define minimum points for robust ratio calculation (increased for stability)
  min_points_for_robust_ratio = (
      20  # Require at least 20 non-zero ILI points for robust median ratio
  )

  for loc in overlap_df['location'].unique():
    loc_overlap = overlap_df[overlap_df['location'] == loc].copy()

    # Filter for non-zero/sufficient ILI data to learn a meaningful relationship
    # Filter on the actual ili_col, which is now expected to be per-capita, using the improved threshold
    loc_overlap_filtered = loc_overlap[
        loc_overlap[ili_col] >= min_ili_threshold
    ]

    # Prepare local per-capita target data for fallback mean calculation
    loc_target_per_capita_data_for_mean = full_train_data[
        full_train_data['location'] == loc
    ].copy()
    loc_target_per_capita_data_for_mean['effective_population'] = np.where(
        loc_target_per_capita_data_for_mean['population'] == 0,
        1.0,
        loc_target_per_capita_data_for_mean['population'],
    )
    loc_target_per_capita_data_for_mean[TARGET_PER_CAPITA_STR] = (
        loc_target_per_capita_data_for_mean[target_col]
        / loc_target_per_capita_data_for_mean['effective_population']
    )
    # IMPROVEMENT 1: Ensure per-capita target is non-negative for local mean fallback
    loc_target_per_capita_data_for_mean[TARGET_PER_CAPITA_STR] = np.maximum(
        0, loc_target_per_capita_data_for_mean[TARGET_PER_CAPITA_STR]
    )

    # Conditions for log-linear model: sufficient overlap, variation in both target and ILI
    ili_std_log_check = False
    if (
        len(loc_overlap_filtered) >= min_overlap_points
        and loc_overlap_filtered[TARGET_PER_CAPITA_STR].std() > 0
        and loc_overlap_filtered[ili_col].std() > 0
    ):

      # Check std after log1p to ensure non-singular matrix for LinearRegression
      X_log_temp = np.log1p(loc_overlap_filtered[[ili_col]])
      y_log_temp = np.log1p(
          loc_overlap_filtered[TARGET_PER_CAPITA_STR]
      )  # Calculate y_log_temp here

      # IMPROVEMENT: Add check for sufficient variance in log-transformed target (y_log_temp)
      if X_log_temp.iloc[:, 0].std() > 1e-6 and y_log_temp.std() > 1e-6:
        ili_std_log_check = True

    if ili_std_log_check:
      model = LinearRegression()

      # Apply log1p transformation for LinearRegression
      X_log = X_log_temp
      y_log = y_log_temp  # Use the already calculated y_log_temp

      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(X_log, y_log)
      transformations[loc] = {'type': 'log_linear', 'model': model}
    else:  # Consolidated fallback logic - MODIFIED: Added minimum points check for robust ratio
      # Attempt ratio fallback on original scale, using median for robustness if ILI signal exists
      relevant_for_ratio = loc_overlap_filtered[
          loc_overlap_filtered[ili_col] > 0
      ].copy()

      # Only use ratio if there are enough points for a robust median
      if len(relevant_for_ratio) >= min_points_for_robust_ratio:
        # Calculate ratio of target_per_capita to ili_signal (both now per-capita)
        ratios_per_capita = relevant_for_ratio[TARGET_PER_CAPITA_STR] / (
            relevant_for_ratio[ili_col] + 1e-9
        )  # Added 1e-9 robustness
        avg_ratio_per_capita = (
            np.nanmedian(ratios_per_capita)
            if not np.all(np.isnan(ratios_per_capita))
            else 0.0
        )
        transformations[loc] = {
            'type': 'ratio_per_capita',
            'value': np.maximum(avg_ratio_per_capita, 1e-9),
        }  # Ensure min positive ratio
      else:
        # Fallback to mean per-capita target for this location, or global mean if local is empty
        # IMPROVEMENT: Ensure the local mean is calculated only if there's sufficient local data.
        # Otherwise, rely more heavily on the global mean for robustness.
        mean_val_per_capita = (
            global_mean_target_per_capita  # Already ensured min positive
        )
        if not loc_target_per_capita_data_for_mean.empty:
          local_mean = loc_target_per_capita_data_for_mean[
              TARGET_PER_CAPITA_STR
          ].mean()
          if not pd.isna(local_mean):
            mean_val_per_capita = np.maximum(
                local_mean, 1e-9
            )  # Ensure local mean is valid and positive
        transformations[loc] = {
            'type': 'mean_target_per_capita',
            'value': mean_val_per_capita,
        }  # Ensure min positive value

  return transformations


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # --- Core Principles of the PROF-like Method (Interpreted for Statistical Model) ---
  #
  # 1. Statistical Approximation of Mechanistic Dynamics with Arc-Tangent Trends:
  #    A LightGBM model statistically approximates epidemiological system dynamics,
  #    utilizing engineered time-series features including multiple arc-tangent terms
  #    to represent a time-varying transmission rate. This implementation includes
  #    ten distinct arc-tangent terms (with varying shifts and spreads from weeks to years)
  #    to capture non-linear trends across various timescales, **with optimized parameters for
  #    better capture of diverse epidemic phases.**
  #
  # 2. Augmented Training with Weighted Synthetic History:
  #    Historical ILINet data is transformed to generate synthetic hospitalization history,
  #    which augments the primary training set and receives a reduced fitting weight (0.5).
  #    Lagged ILI features within the model serve as a statistical proxy for 'future'
  #    insights typically gained from a Method Of Analogs (MOA).
  #    **The ILI transformation learning is made more robust for low signal values.**
  #
  # 3. Direct Quantile Forecasting with Signal-Dependent Baseline:
  #    LightGBM's quantile regression objective directly produces probabilistic forecasts
  #    for each required quantile, serving as a statistical proxy for MCMC inference and
  #    stochastic simulations. For locations or time periods with low epidemiological signal,
  #    predictions are robustly substituted or dynamically scaled by a data-driven
  #    statistical baseline, ensuring calibrated forecasts even in low-incidence scenarios.
  #    **The statistical baselines for low-signal cases are re-calibrated for improved
  #    robustness and consistency with per-capita scaling.**

  warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
  warnings.filterwarnings('ignore', category=FutureWarning)

  # --- 0. Initial Data Preparation & Population Merge ---
  train_x = train_x.copy()
  test_x = test_x.copy()
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  # Ensure population is available in train_x and test_x by merging if not already present
  if 'population' not in train_x.columns:
    train_x = train_x.merge(
        locations[['location', 'population', 'location_name']],
        on='location',
        how='left',
    )
  if 'population' not in test_x.columns:
    test_x = test_x.merge(
        locations[['location', 'population', 'location_name']],
        on='location',
        how='left',
    )

  full_train_df = train_x.copy()
  full_train_df[TARGET_STR] = train_y

  # Store original test_x index for output
  original_test_x_index = test_x.index

  # --- 1. Data Augmentation with ILINet (Principle 2) ---

  # 1.1 Process ILINet dataframes and create unified ILI history

  # Process State-level ILI data
  ili_state_proc = ilinet_state[ilinet_state['region_type'] == 'States'].copy()
  ili_state_proc = ili_state_proc.rename(
      columns={'region': 'location_name', 'week_start': 'target_end_date'}
  )
  ili_state_proc['target_end_date'] = pd.to_datetime(
      ili_state_proc['target_end_date']
  )
  ili_state_proc = pd.merge(
      ili_state_proc,
      locations[['location', 'location_name']],
      on='location_name',
      how='left',
  )
  ili_state_proc = ili_state_proc.dropna(subset=['location']).copy()
  ili_state_proc['location'] = ili_state_proc['location'].astype(int)
  ili_state_proc = ili_state_proc[
      ili_state_proc['target_end_date'] < '2022-10-15'
  ].copy()

  # Keep relevant ILI columns for state-level signal construction.
  ili_state_proc = ili_state_proc[[
      'location',
      'target_end_date',
      'weighted_ili',
      'unweighted_ili',
      'ilitotal',
      'total_patients',
  ]]

  # Process National-level ILI data
  ili_national_proc = ilinet[ilinet['region_type'] == 'National'].copy()
  ili_national_proc = ili_national_proc.rename(
      columns={'week_start': 'target_end_date'}
  )
  ili_national_proc['target_end_date'] = pd.to_datetime(
      ili_national_proc['target_end_date']
  )
  ili_national_proc = ili_national_proc[
      ili_national_proc['target_end_date'] < '2022-10-15'
  ].copy()
  ili_national_proc = ili_national_proc[
      ['target_end_date', 'weighted_ili']
  ].rename(columns={'weighted_ili': 'national_ili_weighted'})

  # Determine the global minimum date across relevant historical data for 'weeks_since_start'
  min_overall_date = pd.to_datetime(
      min(
          full_train_df['target_end_date'].min(),
          ili_state_proc['target_end_date'].min(),
      )
  )

  # 1.2 Create a composite ILI signal for each state for transformation learning and as features

  # Get all unique locations that we care about (from real train/test data)
  all_known_locations = pd.concat(
      [full_train_df['location'], test_x['location']]
  ).unique()

  # Get all unique historical dates from processed ILI data (already filtered < '2022-10-15')
  all_ili_dates = pd.to_datetime(ili_state_proc['target_end_date'].unique())

  # Create a scaffold for all location-date combinations for ILI history
  ili_scaffold = pd.MultiIndex.from_product(
      [all_known_locations, all_ili_dates],
      names=['location', 'target_end_date'],
  ).to_frame(index=False)

  # Merge state-level ILI onto the scaffold
  ili_history_df = pd.merge(
      ili_scaffold,
      ili_state_proc,
      on=['location', 'target_end_date'],
      how='left',
  )

  # Calculate ili_prop (ilitotal / total_patients) for state level
  # Add a small epsilon to denominator to prevent division by zero for robustness
  ili_history_df['state_ili_prop'] = ili_history_df['ilitotal'] / (
      ili_history_df['total_patients'] + 1e-9
  )  # Increased robustness
  # Explicitly fill any NaNs or Infs that may result from division
  ili_history_df['state_ili_prop'] = (
      ili_history_df['state_ili_prop']
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0.0)
  )

  # Merge national-level ILI (broadcasted to all locations for fallback)
  ili_history_df = pd.merge(
      ili_history_df, ili_national_proc, on='target_end_date', how='left'
  )

  # REFINED ILI SIGNAL (Minor Improvement): Prioritize unweighted_ili if available at state level due to better coverage,
  # then weighted_ili, then state_ili_prop, then national_ili_weighted.
  # This logic already exists and is a sound cascading fill.
  ili_history_df['ili_signal'] = ili_history_df['unweighted_ili'].fillna(
      ili_history_df['weighted_ili']
  )  # Prioritize unweighted
  ili_history_df['ili_signal'] = ili_history_df['ili_signal'].fillna(
      ili_history_df['state_ili_prop']
  )
  ili_history_df['ili_signal'] = ili_history_df['ili_signal'].fillna(
      ili_history_df['national_ili_weighted']
  )
  ili_history_df['ili_signal'] = ili_history_df['ili_signal'].fillna(0.0)

  # Ensure ili_signal is non-negative
  ili_history_df['ili_signal'] = np.maximum(0, ili_history_df['ili_signal'])

  # --- NEW: Scale ILI signal by population for consistency with target ---
  # Merge locations to ili_history_df to get population data
  ili_history_df = pd.merge(
      ili_history_df,
      locations[['location', 'population']],
      on='location',
      how='left',
  )
  ili_history_df['effective_population'] = np.where(
      ili_history_df['population'] == 0, 1.0, ili_history_df['population']
  )
  ili_history_df['ili_signal_per_capita'] = (
      ili_history_df['ili_signal'] / ili_history_df['effective_population']
  )
  ili_history_df['ili_signal_per_capita'] = np.maximum(
      0, ili_history_df['ili_signal_per_capita']
  )
  # ----------------------------------------------------------------------

  # Create a lookup series for historical ILI signal, limited to before 2022-10-15
  ili_signal_history_lookup_series = ili_history_df.set_index(
      ['location', 'target_end_date']
  )['ili_signal_per_capita']

  # 1.3 Learn transformation from ILINet to target (min_overlap_points increased for robustness)
  transformation_models = _learn_ili_transformation(
      full_train_df,
      ili_history_df,
      TARGET_STR,
      'ili_signal_per_capita',  # Now pass the per-capita ILI signal
      min_ili_threshold=1e-7,  # Pass the improved threshold
  )

  # 1.4 Generate synthetic history
  synthetic_data_raw_list = []

  # IMPROVEMENT: Use .to_dict() for faster population lookup, and defensive access
  location_population_map = locations.set_index('location')[
      'population'
  ].to_dict()

  for loc in all_known_locations:
    loc_ili_df = ili_history_df[ili_history_df['location'] == loc].copy()

    # IMPROVEMENT: Defensive population lookup, default to 1.0 to prevent division by zero in per-capita conversion
    loc_population = location_population_map.get(loc, 1.0)
    # Ensure effective_population is non-zero for calculations
    loc_population_effective = 1.0 if loc_population == 0 else loc_population

    if loc in transformation_models:
      transform_info = transformation_models[loc]
      synthetic_values_per_capita = None

      if transform_info['type'] == 'log_linear':
        # Apply log1p to ili_signal_per_capita for prediction, then expm1
        X_pred_log = np.log1p(loc_ili_df[['ili_signal_per_capita']])
        predicted_log_per_capita = transform_info['model'].predict(X_pred_log)
        synthetic_values_per_capita = np.expm1(predicted_log_per_capita)
      elif transform_info['type'] == 'ratio_per_capita':
        synthetic_values_per_capita = (
            loc_ili_df['ili_signal_per_capita'] * transform_info['value']
        )
      elif transform_info['type'] == 'mean_target_per_capita':
        synthetic_values_per_capita = np.full(
            len(loc_ili_df), transform_info['value']
        )

      if synthetic_values_per_capita is not None:
        # Convert back to total admissions using effective population
        synthetic_values_total = (
            synthetic_values_per_capita * loc_population_effective
        )
        # Ensure synthetic values are non-negative and integer-like
        loc_ili_df['synthetic_target'] = np.round(
            np.maximum(0, synthetic_values_total)
        ).astype(int)

        # Filter out synthetic data for dates that are already in the real training data
        dates_in_real_train = full_train_df[full_train_df['location'] == loc][
            'target_end_date'
        ].unique()
        loc_ili_df_filtered = loc_ili_df[
            ~loc_ili_df['target_end_date'].isin(dates_in_real_train)
        ].copy()

        if not loc_ili_df_filtered.empty:
          # Collect raw synthetic data, will merge with locations later
          synthetic_df_loc_raw = loc_ili_df_filtered[
              ['target_end_date', 'location']
          ].copy()
          synthetic_df_loc_raw[TARGET_STR] = loc_ili_df_filtered[
              'synthetic_target'
          ]
          synthetic_df_loc_raw['is_synthetic'] = True
          synthetic_data_raw_list.append(synthetic_df_loc_raw)

  if synthetic_data_raw_list:
    synthetic_df_combined_raw = pd.concat(
        synthetic_data_raw_list, ignore_index=True
    )
    # Perform single merge with locations DataFrame to get population and location_name
    synthetic_df = pd.merge(
        synthetic_df_combined_raw,
        locations[['location', 'population', 'location_name']],
        on='location',
        how='left',
    )

    full_train_df['is_synthetic'] = False
    augmented_train_df = pd.concat(
        [full_train_df, synthetic_df], ignore_index=True
    )
  else:
    augmented_train_df = full_train_df.copy()
    augmented_train_df['is_synthetic'] = False

  augmented_train_df = augmented_train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Convert target to per-capita and log1p scale for augmented_train_df
  # IMPROVEMENT: Ensure effective population is used and per-capita target is non-negative
  augmented_train_df['effective_population'] = np.where(
      augmented_train_df['population'] == 0,
      1.0,
      augmented_train_df['population'],
  )
  augmented_train_df[TARGET_PER_CAPITA_STR] = (
      augmented_train_df[TARGET_STR]
      / augmented_train_df['effective_population']
  )
  augmented_train_df[TARGET_PER_CAPITA_STR] = np.maximum(
      0, augmented_train_df[TARGET_PER_CAPITA_STR]
  )
  augmented_train_df[TARGET_MODEL_STR] = np.log1p(
      augmented_train_df[TARGET_PER_CAPITA_STR]
  )

  # --- 2. Feature Engineering for Augmented Training Data (Principles 1 & 2) ---
  lags = [1, 2, 3, 4, 8, 12, 52, 53, 54, 55]
  ili_lags = [
      1,
      2,
      3,
      4,
      8,
      12,
  ]  # Extended ili_lags for more historical context

  # Apply generic feature engineering to augmented training data, including new season/trend features
  augmented_train_df = _engineer_features(
      augmented_train_df, min_overall_date=min_overall_date
  )

  # Initialize a combined tracker with historical actuals and synthetic data for consistent lag generation
  combined_tracker_initial_history = augmented_train_df[
      ['location', 'target_end_date', TARGET_MODEL_STR]
  ].copy()
  combined_tracker_initial_history.set_index(
      ['location', 'target_end_date'], inplace=True
  )
  combined_tracker_initial_history.sort_index(inplace=True)

  # Create the lookup series for _add_lagged_features_from_tracker, now includes all historical (real+synthetic)
  combined_tracker_lookup_series = combined_tracker_initial_history[
      TARGET_MODEL_STR
  ]

  # Add TARGET_MODEL_STR lagged features to augmented training data using the consistent tracker approach
  augmented_train_df = _add_lagged_features_from_tracker(
      augmented_train_df,
      lags,
      'location',
      TARGET_MODEL_STR,
      combined_tracker_lookup_series,
  )

  # Add ILI lagged features to augmented training data - now using ili_signal_per_capita
  augmented_train_df = _add_ili_lagged_features(
      augmented_train_df,
      ili_lags,
      'location',
      ili_signal_history_lookup_series,
      'ili_signal_per_capita',
  )

  # Drop initial rows with NaN lags for target or ILI (these are unavoidable at the start of the time series)
  lag_cols_to_check = [f'{TARGET_MODEL_STR}_lag_{l}' for l in lags] + [
      f'ili_signal_per_capita_lag_{l}' for l in ili_lags
  ]
  augmented_train_df = augmented_train_df.dropna(subset=lag_cols_to_check)

  # --- 3. Model Training (Principle 1 & 3) ---

  # Features to use for training - UPDATED: added new arctan trends and ili_signal_per_capita lags
  features = (
      [
          'month',
          'week_of_year',
          'week_sin',
          'week_cos',
          'season_progress',
          'weeks_since_start',
          'weeks_since_start_squared',
          'arctan_trend_1',
          'arctan_trend_2',
          'arctan_trend_3',
          'arctan_trend_4',
          'arctan_trend_5',
          'arctan_trend_6',
          'arctan_trend_7',
          'arctan_trend_8',
          'arctan_trend_9',
          'arctan_trend_10',  # Added new arctan features
      ]
      + [f'{TARGET_MODEL_STR}_lag_{l}' for l in lags]
      + [f'ili_signal_per_capita_lag_{l}' for l in ili_lags]
  )

  # Handle categorical features (FIPS codes for location, and NEW season_year)
  categorical_features = ['location', 'season_year']
  for col in categorical_features:
    augmented_train_df[col] = augmented_train_df[col].astype('category')

  # Store train categories to align test set categorical features later
  train_location_categories = augmented_train_df['location'].cat.categories
  train_season_year_categories = augmented_train_df[
      'season_year'
  ].cat.categories

  models = {}
  for q in QUANTILES:
    # Assign sample weights: lower weight for synthetic data (0.5) vs real data (1.0)
    sample_weights = np.where(augmented_train_df['is_synthetic'], 0.5, 1.0)

    # Ensure only features present in augmented_train_df are used for training
    current_features = [f for f in features if f in augmented_train_df.columns]
    current_cat_features = [
        f for f in categorical_features if f in current_features
    ]

    lgbm = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        random_state=42,
        n_estimators=500,
        n_jobs=-1,
        verbose=-1,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
    )
    try:
      lgbm.fit(
          augmented_train_df[current_features],
          augmented_train_df[TARGET_MODEL_STR],
          sample_weight=sample_weights,
          categorical_feature=current_cat_features,
      )
      models[q] = lgbm
    except Exception as e:
      print(
          f'Warning: Could not train model for quantile {q}: {e}. Predictions'
          ' for this quantile will be 0.'
      )
      models[q] = None

  # --- 4. Prediction Loop (Recursive Forecasting for Lags) ---

  test_y_hat_quantiles_dict = {}

  # The combined_tracker_lookup_series is already initialized with all historical data.
  # Now, extend the combined_tracker (DataFrame itself) with the test scaffold for recursive updates.
  test_tracker_scaffold = test_x[['location', 'target_end_date']].copy()
  test_tracker_scaffold[TARGET_MODEL_STR] = np.nan

  # Combine historical tracker data with future scaffold
  combined_tracker = pd.concat(
      [combined_tracker_initial_history.reset_index(), test_tracker_scaffold],
      ignore_index=True,
  )
  combined_tracker['target_end_date'] = pd.to_datetime(
      combined_tracker['target_end_date']
  )
  combined_tracker.set_index(['location', 'target_end_date'], inplace=True)
  combined_tracker.sort_index(inplace=True)

  # Update combined_tracker_lookup_series to be a view of the potentially larger combined_tracker
  combined_tracker_lookup_series = combined_tracker[TARGET_MODEL_STR]

  # Pre-process test_x for basic features once (year, month, week_of_year etc.), passing min_overall_date
  test_x_prepared = _engineer_features(
      test_x.copy(), min_overall_date=min_overall_date
  )

  # Iterate through horizons in order
  for h in sorted(test_x['horizon'].unique()):
    current_horizon_test_x = test_x_prepared[
        test_x_prepared['horizon'] == h
    ].copy()
    if current_horizon_test_x.empty:
      continue

    # Add lagged features using the optimized _add_lagged_features_from_tracker
    current_horizon_test_x_with_lags = _add_lagged_features_from_tracker(
        current_horizon_test_x,
        lags,
        'location',
        TARGET_MODEL_STR,
        combined_tracker_lookup_series,
    )

    # Add ILI lagged features to the current horizon test data - now using ili_signal_per_capita
    current_horizon_test_x_with_lags = _add_ili_lagged_features(
        current_horizon_test_x_with_lags,
        ili_lags,
        'location',
        ili_signal_history_lookup_series,
        'ili_signal_per_capita',
    )

    # IMPROVEMENT: Explicitly fill NaNs for lagged features in the test set
    # This is a pragmatic choice to prevent NaNs in features, making early predictions well-defined.
    lag_cols_to_fill_nan = [f'{TARGET_MODEL_STR}_lag_{l}' for l in lags] + [
        f'ili_signal_per_capita_lag_{l}' for l in ili_lags
    ]
    for col in lag_cols_to_fill_nan:
      if col in current_horizon_test_x_with_lags.columns:
        current_horizon_test_x_with_lags[col] = (
            current_horizon_test_x_with_lags[col].fillna(0.0)
        )

    # Ensure categorical features are aligned for prediction
    for col in ['location', 'season_year']:
      categories_to_use = (
          train_location_categories
          if col == 'location'
          else train_season_year_categories
      )
      current_horizon_test_x_with_lags[col] = pd.Categorical(
          current_horizon_test_x_with_lags[col], categories=categories_to_use
      )

    horizon_predictions = {}
    median_preds_for_tracker_update = None
    for q_val in QUANTILES:
      q_col_name = f'quantile_{q_val}'

      current_features = [
          f for f in features if f in current_horizon_test_x_with_lags.columns
      ]

      if models.get(q_val) is not None:
        preds_log1p_per_capita = models[q_val].predict(
            current_horizon_test_x_with_lags[current_features]
        )

        # Inverse transformations: expm1 then multiply by population
        # IMPROVEMENT: Use effective population for prediction to avoid division by zero issues
        effective_population_for_preds = np.where(
            current_horizon_test_x_with_lags['population'] == 0,
            1.0,
            current_horizon_test_x_with_lags['population'],
        )
        preds = (
            np.expm1(preds_log1p_per_capita) * effective_population_for_preds
        )

        preds = np.maximum(0, preds)
        preds = np.nan_to_num(preds, nan=0.0)
      else:
        # Fallback for failed models: predict 0
        preds = np.full(len(current_horizon_test_x_with_lags), 0.0)

      horizon_predictions[q_col_name] = preds

      if (
          q_val == 0.5
      ):  # Store median (on TARGET_MODEL_STR scale) for tracker update
        if models.get(q_val) is not None:
          median_preds_for_tracker_update_raw = models[q_val].predict(
              current_horizon_test_x_with_lags[current_features]
          )
          median_preds_for_tracker_update = np.nan_to_num(
              median_preds_for_tracker_update_raw, nan=0.0
          )
        else:  # If median model failed, use 0 for the tracker to avoid NaNs propagating
          median_preds_for_tracker_update = np.full(
              len(current_horizon_test_x_with_lags), 0.0
          )

    for q_col_name, preds_array in horizon_predictions.items():
      if q_col_name not in test_y_hat_quantiles_dict:
        test_y_hat_quantiles_dict[q_col_name] = pd.Series(
            index=original_test_x_index, dtype=float
        )
      test_y_hat_quantiles_dict[q_col_name].loc[
          current_horizon_test_x_with_lags.index
      ] = preds_array

    # OPTIMIZED: Update the combined_tracker with current horizon's median predictions
    if median_preds_for_tracker_update is not None:
      # Create a Series of median predictions, indexed by (location, target_end_date) for direct update
      update_multiindex = pd.MultiIndex.from_arrays(
          [
              current_horizon_test_x_with_lags['location'],
              current_horizon_test_x_with_lags['target_end_date'],
          ],
          names=['location', 'target_end_date'],
      )

      # Directly update the combined_tracker's TARGET_MODEL_STR column
      combined_tracker.loc[update_multiindex, TARGET_MODEL_STR] = (
          median_preds_for_tracker_update
      )
      # The combined_tracker_lookup_series is a view of combined_tracker[TARGET_MODEL_STR],
      # so updating combined_tracker directly updates the lookup_series as well.

  test_y_hat_quantiles = pd.DataFrame(test_y_hat_quantiles_dict)
  test_y_hat_quantiles = test_y_hat_quantiles.loc[original_test_x_index]

  # --- 5. Apply Baseline Substitution and Enforce Monotonicity/Non-negativity (Principle 3) ---

  recent_signal_window_weeks = 8
  low_signal_threshold_sum = 10  # Total admissions sum, not per-capita

  # IMPROVED: Data-driven low signal baseline using ALL historical per-capita target values (including zeros)
  full_train_df['effective_population'] = np.where(
      full_train_df['population'] == 0, 1.0, full_train_df['population']
  )
  full_train_df[TARGET_PER_CAPITA_STR] = (
      full_train_df[TARGET_STR] / full_train_df['effective_population']
  )
  full_train_df[TARGET_PER_CAPITA_STR] = np.maximum(
      0, full_train_df[TARGET_PER_CAPITA_STR]
  )

  # Calculate quantiles from all historical per-capita admissions, including zeros
  # Ensure minimum positive value for non-zero quantiles to maintain probabilistic nature
  if len(full_train_df[TARGET_PER_CAPITA_STR]) >= (
      len(QUANTILES) * 5
  ):  # Use similar threshold for data-driven
    raw_baseline_per_capita = np.quantile(
        full_train_df[TARGET_PER_CAPITA_STR], QUANTILES
    )
  else:
    # Fallback to a smoother linear baseline if not enough data for empirical quantiles
    raw_baseline_per_capita = np.linspace(
        0, 5e-6, len(QUANTILES)
    )  # 5 cases per million population, per-capita

  # Ensure monotonicity and minimum positive value (e.g., 1 case per billion population for very low quantiles)
  data_driven_low_signal_baseline_values_per_capita = np.maximum.accumulate(
      raw_baseline_per_capita
  )
  data_driven_low_signal_baseline_values_per_capita = np.maximum(
      data_driven_low_signal_baseline_values_per_capita, 1e-9
  )  # Smallest non-zero value

  # MODIFIED: Define a fixed, minimal probabilistic baseline for strictly zero recent signal (per-capita)
  num_initial_zeros_in_baseline = (
      4  # e.g., for quantiles 0.01, 0.025, 0.05, 0.1
  )
  max_baseline_value_for_zero_signal_per_capita = (
      5e-6  # 5 cases per million population, consistent with per-capita
  )

  # Create a smoothly increasing sequence for the non-zero part of the baseline
  smooth_non_zero_part_per_capita = np.linspace(
      0,
      max_baseline_value_for_zero_signal_per_capita,
      len(QUANTILES) - num_initial_zeros_in_baseline,
  )

  # Combine the initial zeros with the smoothly increasing part (per-capita values)
  zero_signal_quantiles_per_capita = np.array(
      [0] * num_initial_zeros_in_baseline
      + smooth_non_zero_part_per_capita.tolist()
  )
  zero_signal_quantiles_per_capita = np.maximum.accumulate(
      zero_signal_quantiles_per_capita
  )  # Ensure monotonicity
  zero_signal_quantiles_per_capita = np.maximum(
      0, zero_signal_quantiles_per_capita
  )  # Ensure non-negativity

  # Calculate recent signal for each location
  location_recent_signal = (
      full_train_df.groupby('location')
      .apply(
          lambda x: x.sort_values('target_end_date')
          .tail(recent_signal_window_weeks)[TARGET_STR]
          .sum()
      )
      .reset_index(name='recent_admissions_sum')
  )

  # Convert to dictionary for faster lookup
  location_recent_signal_dict = location_recent_signal.set_index('location')[
      'recent_admissions_sum'
  ].to_dict()
  location_population_dict = locations.set_index('location')[
      'population'
  ].to_dict()

  for idx in test_y_hat_quantiles.index:
    loc = test_x.loc[idx, 'location']
    recent_sum = location_recent_signal_dict.get(loc, 0)
    current_location_population = location_population_dict.get(
        loc, 1.0
    )  # Default to 1.0 to prevent div by zero
    if current_location_population == 0:
      current_location_population = 1.0

    forecast_row_quantiles = []
    model_predictions_for_row = np.array(
        [test_y_hat_quantiles.loc[idx, f'quantile_{q}'] for q in QUANTILES]
    )

    if recent_sum <= 0:
      # For strictly zero or non-positive recent signal, use a fixed, minimal probabilistic baseline.
      # Scale per-capita baseline to total count for the current location's population.
      forecast_row_quantiles = (
          zero_signal_quantiles_per_capita * current_location_population
      ).tolist()
    elif 0 < recent_sum <= low_signal_threshold_sum:
      # Little-to-no signal, but not strictly zero. Dynamically scale AND BLEND the data-driven baseline.
      scale_factor = recent_sum / low_signal_threshold_sum

      # Scale data-driven baseline from per-capita to total count for this location
      scaled_data_driven_baseline_total = (
          data_driven_low_signal_baseline_values_per_capita
          * current_location_population
      )

      # Blend model predictions with the scaled data-driven baseline
      # When scale_factor is small (recent_sum near 0), closer to baseline.
      # When scale_factor is large (recent_sum near threshold), closer to model prediction.
      blended_predictions = (scale_factor * model_predictions_for_row) + (
          (1 - scale_factor) * scaled_data_driven_baseline_total
      )
      forecast_row_quantiles = blended_predictions.tolist()
    else:
      # Sufficient signal, use model predictions
      forecast_row_quantiles = model_predictions_for_row.tolist()

    # Enforce monotonicity and non-negativity across all cases (model predictions and baselines).
    monotonic_quantiles = np.maximum.accumulate(
        np.array(forecast_row_quantiles)
    )
    monotonic_quantiles = np.maximum(0, monotonic_quantiles)

    for i, q_val in enumerate(QUANTILES):
      test_y_hat_quantiles.loc[idx, f'quantile_{q_val}'] = monotonic_quantiles[
          i
      ]

  for col in test_y_hat_quantiles.columns:
    test_y_hat_quantiles[col] = np.round(test_y_hat_quantiles[col]).astype(int)

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
