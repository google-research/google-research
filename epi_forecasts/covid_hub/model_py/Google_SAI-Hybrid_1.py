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
MODEL_NAME = 'Google_SAI-Hybrid_1'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of this Hybrid Forecasting Method:
# 1. LightGBM Quantile Regression for Flexibility: Utilizes LightGBM to directly predict quantiles, leveraging its non-linear modeling capabilities for robust and accurate forecasts.
# 2. Direct Multi-Horizon Forecasting with Rich Features: Employs an expanded training data approach where 'horizon' is a direct model feature, combined with extensive feature engineering (datetime, population covariates, multi-type lagged admissions including raw, smoothed, EWMA, and critical lagged differences for trend signals).
# 3. Population-Normalized Target with Log Transformation: Models per-capita admissions (normalized by population), and applies a log1p transformation to the target for better model behavior with count data and skewed distributions.
# 4. Comprehensive Robustness & Post-processing: Incorporates strict data cleaning for population, careful windowing of training data, efficient handling of known historical values (horizon -1), and essential post-processing steps like non-negativity, quantile monotonicity, and rounding.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Any, List, Dict, Tuple, Set


def _get_quantile_col_names():
  """Returns a list of quantile column names."""
  return [f'quantile_{q}' for q in QUANTILES]


def _return_empty_predictions(test_x):
  """Helper to return an empty (zero-filled) predictions DataFrame with correct columns.

  This is used for robust error handling in case model training or data
  preparation fails, ensuring the function always returns a DataFrame of the
  expected format.
  """
  empty_preds = pd.DataFrame(
      0, index=test_x.index, columns=_get_quantile_col_names()
  ).astype(int)
  return empty_preds


def _clean_population_series(
    series, fallback_value = 1.0
):
  """Cleans a population series by converting to numeric, filling NaNs, and handling zeros.

  Ensures that population values are positive for normalization. Uses a provided
  fallback_value for invalid entries.
  """
  series = pd.to_numeric(series, errors='coerce')
  # Fill NaNs and replace non-positive values with the determined fallback value
  cleaned_series = series.fillna(fallback_value)
  cleaned_series[cleaned_series <= 0] = fallback_value
  return cleaned_series


def _prepare_historical_source_data(
    train_x_cleaned_population,  # Expects population to be already cleaned
    train_y,
    smoothing_window,
):
  """Combines train_x and train_y, applies population cleaning, normalization,

  and smoothing. This function prepares the raw historical values that will be
  used to look up lagged features and the target.
  It does NOT generate lagged feature columns itself.
  """
  historical_data_df = train_x_cleaned_population.copy()
  historical_data_df[TARGET_STR] = train_y

  # Ensure 'target_end_date' is datetime and sort for correct rolling mean and lag calculations
  historical_data_df['target_end_date'] = pd.to_datetime(
      historical_data_df['target_end_date']
  )
  historical_data_df = historical_data_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # --- Population Normalization (Code 1 Principle: explicit normalization) ---
  # Population in historical_data_df is now already cleaned from fit_and_predict_fn, just ensure it's numeric
  # Removed epsilon, as population_clean is guaranteed to be positive.
  historical_data_df['admissions_norm'] = (
      historical_data_df[TARGET_STR] / historical_data_df['population']
  )
  historical_data_df['admissions_norm'] = historical_data_df[
      'admissions_norm'
  ].fillna(0)

  # --- Rolling Mean Smoothing (Code 1 Principle: explicit smoothing for AR-like features) ---
  if smoothing_window > 0:
    historical_data_df['smoothed_admissions_norm'] = historical_data_df.groupby(
        'location'
    )['admissions_norm'].transform(
        lambda x: x.rolling(
            window=smoothing_window, min_periods=1, center=False
        ).mean()
    )
  else:
    historical_data_df['smoothed_admissions_norm'] = historical_data_df[
        'admissions_norm'
    ]
  historical_data_df['smoothed_admissions_norm'] = historical_data_df[
      'smoothed_admissions_norm'
  ].fillna(0.0)

  # --- NEW: Exponential Weighted Moving Average (EWMA) (Hybrid improvement) ---
  # Captures trends with more weight on recent observations than simple rolling mean.
  # Span is roughly twice window for similar "effective" length, or adjusted based on specific requirements.
  # min_periods=1 allows early values to be non-NaN.
  historical_data_df['ewm_admissions_norm'] = historical_data_df.groupby(
      'location'
  )['admissions_norm'].transform(
      lambda x: x.ewm(
          span=smoothing_window * 2, adjust=False, min_periods=1
      ).mean()
  )
  historical_data_df['ewm_admissions_norm'] = historical_data_df[
      'ewm_admissions_norm'
  ].fillna(0.0)

  # --- Target Transformation (ML best practice for count data) ---
  historical_data_df['log_admissions_target'] = np.log1p(
      historical_data_df['admissions_norm']
  )
  historical_data_df['log_admissions_target'] = historical_data_df[
      'log_admissions_target'
  ].fillna(0.0)

  # Return necessary columns including the original 'population' for consistency in scaffold processing
  return historical_data_df[[
      'location',
      'target_end_date',
      'population',
      TARGET_STR,
      'admissions_norm',
      'smoothed_admissions_norm',
      'ewm_admissions_norm',
      'log_admissions_target',
  ]]


def _extract_features_for_scaffold(
    df_scaffold,
    historical_source_data_for_lags,
    config,
    train_categorical_feature_categories = None,
):
  """Generates features for a given scaffold (expanded training data or test_x)

  by extracting datetime features, population features, and dynamically
  merging lagged features from the historical source data.

  Hybrid Principle: This function combines feature engineering ideas from Code 1
  (lags) and Code 2 (rich datetime features, population as covariate, direct
  multi-horizon features)
  and uses the pre-computed historical source data for efficiency and
  correctness.
  """
  df = df_scaffold.copy()

  # Ensure date columns are datetime objects for consistent operations
  # These should already be datetime from fit_and_predict_fn, but defensive check
  df['target_end_date'] = pd.to_datetime(df['target_end_date'])
  df['reference_date'] = pd.to_datetime(df['reference_date'])

  # --- Population features (Code 2 Principle: population as covariate, uses already cleaned population) ---
  # The 'population' column in df is now expected to be already cleaned when passed to this function.
  df['log_population'] = np.log1p(df['population'])

  # --- Datetime features from reference_date and target_end_date (Code 2 Principle) ---
  df['ref_year'] = df['reference_date'].dt.year
  df['ref_month'] = df['reference_date'].dt.month
  df['ref_weekofyear'] = df['reference_date'].dt.isocalendar().week.astype(int)
  df['ref_dayofyear'] = df['reference_date'].dt.dayofyear

  df['target_year'] = df['target_end_date'].dt.year
  df['target_month'] = df['target_end_date'].dt.month
  df['target_weekofyear'] = (
      df['target_end_date'].dt.isocalendar().week.astype(int)
  )
  df['target_dayofyear'] = df['target_end_date'].dt.dayofyear

  # Cyclical features for week of year (Code 2 Principle) to capture seasonality
  max_week_for_cycle = 53  # A common upper bound for ISO week numbers
  df['ref_week_sin'] = np.sin(
      2 * np.pi * df['ref_weekofyear'] / max_week_for_cycle
  )
  df['ref_week_cos'] = np.cos(
      2 * np.pi * df['ref_weekofyear'] / max_week_for_cycle
  )
  df['target_week_sin'] = np.sin(
      2 * np.pi * df['target_weekofyear'] / max_week_for_cycle
  )
  df['target_week_cos'] = np.cos(
      2 * np.pi * df['target_weekofyear'] / max_week_for_cycle
  )

  # --- Weeks Since First Observation (Long-term temporal trend feature) ---
  if not historical_source_data_for_lags.empty:
    earliest_target_date = historical_source_data_for_lags[
        'target_end_date'
    ].min()
    df['weeks_since_first_observation'] = (
        df['target_end_date'] - earliest_target_date
    ).dt.days // 7
    df['weeks_since_first_observation'] = df[
        'weeks_since_first_observation'
    ].clip(lower=0)
  else:
    df['weeks_since_first_observation'] = (
        0.0  # Default to 0 if no historical data
    )

  # --- Horizon as a feature and interaction term (Code 2 Principle) ---
  df['horizon'] = df['horizon'].astype(int)
  # Use +1 to ensure interaction term is non-zero for horizon 0
  df['log_population_horizon_interaction'] = df['log_population'] * (
      df['horizon'] + 1
  )

  # --- Efficiently Merge Lagged Features (Hybrid: values known at reference_date) ---
  # Create a base lookup DataFrame from the historical source data, selected and indexed for efficiency
  historical_admissions_for_merge = historical_source_data_for_lags[[
      'location',
      'target_end_date',
      'admissions_norm',
      'smoothed_admissions_norm',
      'ewm_admissions_norm',
      'log_admissions_target',
  ]].set_index(
      ['location', 'target_end_date']
  )  # Use multi-index for fast lookup

  lag_weeks = sorted(config.get('lag_weeks', [1, 2, 3, 4]))
  all_lag_feat_cols: Set[str] = set()  # Use a set to avoid duplicates

  # Dynamically merge historical values for each lag using reindex for speed
  for lw in lag_weeks:
    # Calculate the lookup date for the current lag relative to reference_date
    lookup_keys = pd.MultiIndex.from_arrays(
        [df['location'], df['reference_date'] - pd.Timedelta(weeks=lw)],
        names=['location', 'target_end_date'],
    )

    # Reindex to efficiently get the lagged values and assign to df
    # Fill missing lags (e.g., at start of time series) with 0.0
    # LightGBM can handle NaNs for numerical features, but 0.0 for log-transformed admissions
    # implies 0 actual admissions which is a strong but reasonable assumption for missing history.
    df[f'feat_lag_norm_admissions_{lw}w'] = (
        historical_admissions_for_merge['admissions_norm']
        .reindex(lookup_keys, fill_value=0.0)
        .values
    )
    df[f'feat_lag_smoothed_norm_admissions_{lw}w'] = (
        historical_admissions_for_merge['smoothed_admissions_norm']
        .reindex(lookup_keys, fill_value=0.0)
        .values
    )
    df[f'feat_lag_ewm_norm_admissions_{lw}w'] = (
        historical_admissions_for_merge['ewm_admissions_norm']
        .reindex(lookup_keys, fill_value=0.0)
        .values
    )
    df[f'feat_lag_log_target_{lw}w'] = (
        historical_admissions_for_merge['log_admissions_target']
        .reindex(lookup_keys, fill_value=0.0)
        .values
    )
    all_lag_feat_cols.update([
        f'feat_lag_norm_admissions_{lw}w',
        f'feat_lag_smoothed_norm_admissions_{lw}w',
        f'feat_lag_ewm_norm_admissions_{lw}w',
        f'feat_lag_log_target_{lw}w',
    ])

  # --- Generate Lagged Differences for 'admissions_norm' and 'ewm_admissions_norm' (Hybrid improvement) ---
  # This captures week-over-week change, providing trend information.
  # Original logic (adjacent lags)
  sorted_lag_weeks = sorted(lag_weeks)
  for i in range(len(sorted_lag_weeks) - 1):
    current_lag_w = sorted_lag_weeks[i]
    next_lag_w = sorted_lag_weeks[i + 1]

    lag_col_current_norm = f'feat_lag_norm_admissions_{current_lag_w}w'
    lag_col_next_norm = f'feat_lag_norm_admissions_{next_lag_w}w'
    diff_col_name_norm = (
        f'feat_diff_lag_norm_{current_lag_w}w_minus_{next_lag_w}w'
    )
    if lag_col_current_norm in df.columns and lag_col_next_norm in df.columns:
      df[diff_col_name_norm] = df[lag_col_current_norm] - df[lag_col_next_norm]
      df[diff_col_name_norm] = df[diff_col_name_norm].fillna(0.0)
      all_lag_feat_cols.add(diff_col_name_norm)
    else:
      df[diff_col_name_norm] = 0.0

    lag_col_current_ewm = f'feat_lag_ewm_norm_admissions_{current_lag_w}w'
    lag_col_next_ewm = f'feat_lag_ewm_norm_admissions_{next_lag_w}w'
    diff_col_name_ewm = (
        f'feat_diff_lag_ewm_{current_lag_w}w_minus_{next_lag_w}w'
    )
    if lag_col_current_ewm in df.columns and lag_col_next_ewm in df.columns:
      df[diff_col_name_ewm] = df[lag_col_current_ewm] - df[lag_col_next_ewm]
      df[diff_col_name_ewm] = df[diff_col_name_ewm].fillna(0.0)
      all_lag_feat_cols.add(diff_col_name_ewm)
    else:
      df[diff_col_name_ewm] = 0.0

  # NEW IMPROVEMENT: Additional Lagged Differences for EWMA (most recent vs. longer terms)
  # This provides stronger signals for changes in trend over different time horizons.
  if len(sorted_lag_weeks) > 0:
    most_recent_lag_ewm_col = (  # Typically lag_1w
        f'feat_lag_ewm_norm_admissions_{sorted_lag_weeks[0]}w'
    )

    if most_recent_lag_ewm_col in df.columns:
      # Define longer-term lags to compare against the most recent lag
      # These are chosen to represent short-term (4w), medium-term (8w, 12w), and long-term/seasonal (26w, 52w) trends.
      longer_term_diff_lags = [
          lw
          for lw in sorted_lag_weeks
          if lw > sorted_lag_weeks[0] and lw in [4, 8, 12, 26, 52]
      ]

      for longer_lw in longer_term_diff_lags:
        longer_lag_ewm_col = f'feat_lag_ewm_norm_admissions_{longer_lw}w'
        diff_col_name_ewm_long = (  # More descriptive name
            f'feat_diff_lag_ewm_recent_minus_{longer_lw}w'
        )
        if longer_lag_ewm_col in df.columns:
          df[diff_col_name_ewm_long] = (
              df[most_recent_lag_ewm_col] - df[longer_lag_ewm_col]
          )
          df[diff_col_name_ewm_long] = df[diff_col_name_ewm_long].fillna(0.0)
          all_lag_feat_cols.add(diff_col_name_ewm_long)
        else:
          df[diff_col_name_ewm_long] = 0.0

  # Define features to use in the model.
  numerical_base_features = [
      'log_population',
      'log_population_horizon_interaction',
      'ref_year',
      'ref_month',
      'ref_weekofyear',
      'ref_dayofyear',
      'ref_week_sin',
      'ref_week_cos',
      'target_year',
      'target_month',
      'target_weekofyear',
      'target_dayofyear',
      'target_week_sin',
      'target_week_cos',
      'weeks_since_first_observation',  # NEW FEATURE
  ]
  feature_cols = [
      col for col in numerical_base_features if col in df.columns
  ] + list(all_lag_feat_cols)

  # --- Categorical features for LightGBM (Code 2 Principle: optimized categorical handling) ---
  # Only location and horizon are treated as categorical features.
  initial_categorical_cols = ['location', 'horizon']
  categorical_cols = [
      col for col in initial_categorical_cols if col in feature_cols
  ]

  # Initialize categories dict if not provided (for training data)
  current_categorical_feature_categories = (
      train_categorical_feature_categories
      if train_categorical_feature_categories is not None
      else {}
  )

  for col in categorical_cols:
    if col in df.columns:
      if col in current_categorical_feature_categories:
        try:
          # Cast using known categories from training for consistency
          df[col] = df[col].astype(
              pd.CategoricalDtype(
                  categories=current_categorical_feature_categories[col],
                  ordered=False,
              ),
              copy=False,
          )
        except (TypeError, ValueError) as e:
          # Fallback for unseen categories in test data or invalid casts
          # This warning is suppressed globally for LightGBM related messages, but good to keep for debug.
          warnings.warn(
              f"Casting categorical feature '{col}' to specified categories"
              ' failed or contained unseen values. Falling back to local'
              f' inference for test set. Error: {e}',
              UserWarning,
          )
          df[col] = df[col].astype('category')
      else:
        # Infer categories if not provided (for training data)
        df[col] = df[col].astype('category')
        current_categorical_feature_categories[col] = df[
            col
        ].cat.categories.tolist()

  return (
      df,
      feature_cols,
      categorical_cols,
      current_categorical_feature_categories,
  )


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make probabilistic predictions for test_x by modelling train_x to train_y.

  This function implements a robust hybrid strategy combining principles from
  Code 1
  (population normalization, rolling mean smoothing for some lags, training data
  windowing,
  robust post-processing) and Code 2 (LightGBM, rich datetime feature
  engineering,
  population as covariate, direct multi-horizon forecasting via expanded
  training data,
  direct quantile optimization).

  Crucially, it uses normalized, UNSMOOTHED admissions as the primary target for
  LGBM (log1p transformed),
  and intelligently handles horizon -1 predictions by directly looking up known
  historical values,
  preventing unnecessary model prediction for already observed data.
  """
  # Define config:
  config = {
      'description': 'Hybrid - Optimized Performance',
      'lag_weeks': [1, 2, 3, 4, 8, 12, 26, 52],  # Expanded lags
      'smoothing_window_weeks': 4,
      'prediction_horizons': [
          -1,
          0,
          1,
          2,
          3,
      ],  # Horizons for prediction (includes -1 for known values)
      'forecast_horizons_for_training': [
          0,
          1,
          2,
          3,
      ],  # Actual future horizons for training
      'train_historical_data_window_weeks': (
          169
      ),  # ~3.25 years of recent data for historical lookups
      'train_forecast_ref_date_window_weeks': (
          156
      ),  # ~3 years for reference date context
      'lgbm_params': {
          'n_estimators': 1200,  # Increased for more thorough learning
          'learning_rate': 0.007,  # Decreased for more careful learning
          'num_leaves': 70,  # Increased for more complex trees
          'max_depth': 10,
          'min_child_samples': 60,  # Increased for better robustness to noise
          'reg_alpha': 0.07,  # Increased regularization
          'reg_lambda': 0.07,  # Increased regularization
          'colsample_bytree': 0.7,
          'subsample': 0.7,
          'max_bin': 255,
          'min_gain_to_split': 0.0005,  # Decreased to allow more splits
      },
  }

  # Suppress specific LightGBM and general warnings for cleaner output
  warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
  warnings.filterwarnings('ignore', category=FutureWarning)
  warnings.filterwarnings(
      'ignore',
      message='The `num_features` argument to `LGBMRegressor` is deprecated',
  )
  warnings.filterwarnings(
      'ignore', message='The `categorical_feature` keyword has been deprecated'
  )

  # --- Initial Date Conversions and CENTRALIZED Population Cleaning ---
  # Standardize date types at the entry point for train_x and test_x
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  # Calculate a robust median for fallback from the entire training population
  valid_finite_positive_train_pops = train_x['population'][
      (train_x['population'] > 0) & (np.isfinite(train_x['population']))
  ]
  if not valid_finite_positive_train_pops.empty:
    global_fallback_median_pop = valid_finite_positive_train_pops.median()
    if pd.isna(global_fallback_median_pop) or global_fallback_median_pop <= 0:
      global_fallback_median_pop = 1.0  # Default if median is still problematic
  else:
    global_fallback_median_pop = 1.0  # Default if no valid pops in train_x

  # Apply global population cleaning to all relevant dataframes
  train_x['population'] = train_x.groupby('location')['population'].transform(
      lambda x: _clean_population_series(
          x, fallback_value=global_fallback_median_pop
      )
  )
  # Store original test_x populations for denormalization later, ensuring it's cleaned
  test_x['population'] = test_x.groupby('location')['population'].transform(
      lambda x: _clean_population_series(
          x, fallback_value=global_fallback_median_pop
      )
  )
  original_test_populations_series = test_x[
      'population'
  ].copy()  # This is now the cleaned population

  # --- Prepare Comprehensive Historical Source Data ---
  # This data contains normalized, smoothed admissions and their log-transformed target.
  # It will be the source for looking up lagged features for both train and and test.
  # Pass train_x with already cleaned population
  historical_source_data = _prepare_historical_source_data(
      train_x.copy(),
      train_y.copy(),
      config.get('smoothing_window_weeks', 4),
  )

  if historical_source_data.empty:
    warnings.warn(
        'No historical data available after preprocessing. Returning empty'
        ' predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  max_historical_target_date = historical_source_data['target_end_date'].max()
  if pd.isna(max_historical_target_date):
    warnings.warn(
        'No historical data dates found after preprocessing. Returning empty'
        ' predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  # Filter historical data for lag lookups to the specified window
  train_historical_data_window_weeks = config.get(
      'train_historical_data_window_weeks', 169
  )
  min_train_target_date_for_lags = max_historical_target_date - pd.Timedelta(
      weeks=train_historical_data_window_weeks
  )

  recent_historical_data_for_lags = historical_source_data[
      historical_source_data['target_end_date']
      >= min_train_target_date_for_lags
  ].copy()

  if recent_historical_data_for_lags.empty:
    warnings.warn(
        f'No recent historical data within {train_historical_data_window_weeks}'
        ' weeks for lag lookups. Falling back to all available historical data'
        ' for expansion.',
        UserWarning,
    )
    recent_historical_data_for_lags = historical_source_data.copy()
  elif (
      historical_source_data.empty
  ):  # Fallback to empty if even original is empty
    warnings.warn(
        'No historical data available at all after preprocessing. Returning'
        ' empty predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  # --- Streamlined Training Data Expansion (Code 2 principle: direct multi-horizon via tabular data) ---
  forecast_horizons_for_training = config.get(
      'forecast_horizons_for_training', [0, 1, 2, 3]
  )
  train_forecast_ref_date_window_weeks = config.get(
      'train_forecast_ref_date_window_weeks', 156
  )

  # Determine the range of reference dates to consider for training
  max_train_ref_date = max_historical_target_date
  min_train_ref_date = max_train_ref_date - pd.Timedelta(
      weeks=train_forecast_ref_date_window_weeks
  )

  # Get unique locations from the historical data
  unique_locations = recent_historical_data_for_lags['location'].unique()

  # Generate reference dates as a fixed series of Saturdays
  train_ref_dates = pd.date_range(
      start=min_train_ref_date, end=max_train_ref_date, freq='W-SAT'
  )

  # Create all combinations of (location, reference_date, horizon) directly
  expanded_train_df_base = pd.DataFrame(
      pd.MultiIndex.from_product(
          [unique_locations, train_ref_dates, forecast_horizons_for_training],
          names=['location', 'reference_date', 'horizon'],
      ).to_frame(index=False)
  )

  expanded_train_df_base['target_end_date'] = expanded_train_df_base[
      'reference_date'
  ] + pd.to_timedelta(expanded_train_df_base['horizon'], unit='W')

  # Merge target values (log_admissions_target) and population from historical data
  # 'population' from historical_source_data is already cleaned
  historical_lookup = historical_source_data[
      ['location', 'target_end_date', 'log_admissions_target', 'population']
  ]

  # Merge target_end_date specific data for the target
  expanded_train_df_base = pd.merge(
      expanded_train_df_base,
      historical_lookup[
          ['location', 'target_end_date', 'log_admissions_target']
      ],
      on=['location', 'target_end_date'],
      how=(  # Left merge to keep all forecast combinations, target might be NaN if not observed
          'left'
      ),
  )

  # Merge reference_date specific data for population.
  # We rename 'target_end_date' to 'reference_date' in the historical_lookup to join correctly.
  expanded_train_df_base = pd.merge(
      expanded_train_df_base,
      historical_lookup[['location', 'target_end_date', 'population']].rename(
          columns={'target_end_date': 'reference_date_for_pop_lookup'}
      ),
      left_on=[
          'location',
          'reference_date',
      ],  # Join on reference_date of the expanded_train_df_base
      right_on=['location', 'reference_date_for_pop_lookup'],
      how='left',
      suffixes=('', '_ref_date'),
  )
  expanded_train_df_base.drop(
      columns=['reference_date_for_pop_lookup'], inplace=True
  )  # Clean up temp column
  # The merged 'population' column is named 'population_ref_date' due to suffixes. Rename it.
  if 'population_ref_date' in expanded_train_df_base.columns:
    expanded_train_df_base['population'] = expanded_train_df_base[
        'population_ref_date'
    ]
    expanded_train_df_base.drop(columns=['population_ref_date'], inplace=True)

  # Drop rows where the target_end_date is beyond the actual historical data available
  # or where the target value is NaN (i.e., not observed in history)
  # Ensure population is also present for the reference_date (already cleaned to non-zero)
  expanded_train_df_base.dropna(
      subset=['log_admissions_target', 'population'], inplace=True
  )
  expanded_train_df_base.reset_index(drop=True, inplace=True)

  if expanded_train_df_base.empty:
    warnings.warn(
        'No valid training data could be formed after vectorized expansion and'
        ' windowing. Returning empty predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  # Generate features for the expanded training data. `train_categorical_feature_categories` will be populated here.
  train_categorical_feature_categories = (
      {}
  )  # Initialize for learning categories
  (
      X_train_features_df,
      feature_cols,
      categorical_cols,
      train_categorical_feature_categories,
  ) = _extract_features_for_scaffold(
      expanded_train_df_base.drop(
          columns=['log_admissions_target'], errors='ignore'
      ),
      recent_historical_data_for_lags,  # Use the windowed historical data for lag lookups
      config,
      train_categorical_feature_categories,  # Pass empty dict to populate
  )

  # Robust target alignment (already done implicitly by dropna above, but explicit check)
  X_train = X_train_features_df[feature_cols]
  y_train = expanded_train_df_base['log_admissions_target']

  if X_train.empty or y_train.empty or len(X_train) != len(y_train):
    warnings.warn(
        'Training data features or target is empty/mismatched after processing.'
        ' Returning empty predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  # --- Model Training (LightGBM Quantile Regression - Code 2 Principle: direct quantile optimization) ---
  models = {}
  lgbm_base_params = {
      'objective': 'quantile',
      'metric': 'quantile',
      'n_estimators': (
          config.get('lgbm_params', {}).get('n_estimators', 1200)
      ),  # Tuned
      'learning_rate': (
          config.get('lgbm_params', {}).get('learning_rate', 0.007)
      ),  # Tuned
      'num_leaves': (
          config.get('lgbm_params', {}).get('num_leaves', 70)
      ),  # Tuned
      'max_depth': config.get('lgbm_params', {}).get('max_depth', 10),
      'min_child_samples': (
          config.get('lgbm_params', {}).get('min_child_samples', 60)
      ),  # Tuned
      'random_state': 42,
      'n_jobs': -1,
      'verbose': -1,
      'colsample_bytree': (
          config.get('lgbm_params', {}).get('colsample_bytree', 0.7)
      ),
      'subsample': config.get('lgbm_params', {}).get('subsample', 0.7),
      'boosting_type': (
          config.get('lgbm_params', {}).get('boosting_type', 'gbdt')
      ),
      'reg_alpha': (
          config.get('lgbm_params', {}).get('reg_alpha', 0.07)
      ),  # Tuned
      'reg_lambda': (
          config.get('lgbm_params', {}).get('reg_lambda', 0.07)
      ),  # Tuned
      'min_gain_to_split': (
          config.get('lgbm_params', {}).get('min_gain_to_split', 0.0005)
      ),  # Tuned
      'max_bin': config.get('lgbm_params', {}).get('max_bin', 255),
  }

  try:
    valid_categorical_features = [
        col for col in categorical_cols if col in X_train.columns
    ]  # Compute once
    for q in QUANTILES:
      model = lgb.LGBMRegressor(alpha=q, **lgbm_base_params)
      model.fit(
          X_train, y_train, categorical_feature=valid_categorical_features
      )
      models[q] = model
  except Exception as e:
    warnings.warn(
        f'Error during LGBM model training: {e}. Returning empty predictions.',
        UserWarning,
    )
    return _return_empty_predictions(test_x)

  # Initialize prediction DataFrame using original test_x index
  test_y_hat_quantiles = pd.DataFrame(
      0.0, index=test_x.index, columns=_get_quantile_col_names()
  )

  # --- Optimized Horizon -1 Handling (Hybrid improvement) ---
  # Separate test_x into known historical values (horizon -1) and actual forecasts
  test_x_known = test_x.loc[test_x['horizon'] == -1].copy()
  test_x_forecast = test_x.loc[test_x['horizon'] != -1].copy()

  if not test_x_known.empty:
    # Use the raw target column from historical_source_data
    full_train_historical_data_raw_indexed = historical_source_data.set_index(
        ['location', 'target_end_date']
    )[TARGET_STR]

    # For horizon -1, fill all quantile columns with the known historical value
    for idx in test_x_known.index:
      row = test_x_known.loc[idx]
      lookup_key = (row['location'], row['target_end_date'])
      known_val = full_train_historical_data_raw_indexed.get(lookup_key, np.nan)
      # Fill all quantile columns for this row with the known value, handling NaNs with 0
      test_y_hat_quantiles.loc[idx, _get_quantile_col_names()] = np.nan_to_num(
          known_val, nan=0.0
      )

  # --- Make predictions for remaining horizons using trained LightGBM models ---
  if not test_x_forecast.empty:
    # Generate features for the remaining test data
    # Pass categories learned from training to ensure consistency in test data casting
    X_test_features_df, _, _, _ = _extract_features_for_scaffold(
        test_x_forecast,  # test_x_forecast already contains cleaned population
        recent_historical_data_for_lags,  # Use the windowed historical data for lag lookups
        config,
        train_categorical_feature_categories,  # Pass categories learned from training
    )

    # Ensure test features have the same columns and order as training features
    # Reindex to match feature_cols from training, filling missing numerical columns with 0.0
    X_test_features_df = X_test_features_df.reindex(
        columns=feature_cols, fill_value=0.0
    )

    # Predict for each quantile
    for q in QUANTILES:
      if q in models:
        log_normalized_preds = models[q].predict(X_test_features_df)
        normalized_preds = np.expm1(
            log_normalized_preds
        )  # Inverse transform from log1p

        # Align test populations with the current slice of test_x being predicted
        current_test_populations = original_test_populations_series.loc[
            test_x_forecast.index
        ]

        # Denormalize predictions back to original scale (total counts)
        denormalized_preds = normalized_preds * current_test_populations.values

        test_y_hat_quantiles.loc[test_x_forecast.index, f'quantile_{q}'] = (
            denormalized_preds
        )
      else:
        warnings.warn(
            f'Model for quantile {q} not found. Filling with 0 for this'
            ' quantile.',
            UserWarning,
        )

  # --- Post-processing: Non-negativity, Monotonicity, Rounding (Code 1 & 2 principles) ---
  test_y_hat_quantiles[test_y_hat_quantiles < 0] = 0.0

  # Apply monotonicity constraint (sort quantiles across the row)
  test_y_hat_quantiles = pd.DataFrame(
      np.sort(test_y_hat_quantiles.values, axis=1),
      columns=test_y_hat_quantiles.columns,
      index=test_y_hat_quantiles.index,
  )

  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(int)

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
