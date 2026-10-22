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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation,cell-var-from-loop

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
MODEL_NAME = 'Google_SAI-Novel_2'
TARGET_STR = 'Total RSV Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, List

# --- Configuration Constants ---

TARGET_STR = 'Total RSV Admissions'

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
horizons = [0, 1, 2, 3]


# NEW/IMPROVED CONSTANTS
PER_CAPITA_SCALE = 100000  # Scale admissions to per 100k population
# A broader range of lags for better capture of seasonal and short-term dynamics
LAG_PERIODS = [1, 2, 3, 4, 8, 12, 16, 20, 26, 30, 39, 52, 60, 78, 104]
# Add some rolling windows for trend/volatility
ROLLING_WINDOWS = [4, 8, 12, 26, 52]  # Rolling windows in weeks
EPSILON_POPULATION = (
    1e-6  # Small value to prevent division by zero for population
)

# Note: _create_fold_scaffold, np.seterr are defined outside this cell but needed for context.
# We assume _create_fold_scaffold, ilinet_state, locations are available in the global scope.

# --- Helper functions (IMPROVED and moved into rewrite cell) ---


def _create_time_features(
    df_input, date_col = 'target_end_date', prefix = ''
):
  """Generates time-based features from a date column, including cyclic features for seasonality.

  IMPROVEMENT: Added week_of_year, month_of_year, and their cyclic components.
  """
  df = df_input.copy()
  df[date_col] = pd.to_datetime(df[date_col])
  # Keep year as a general trend feature
  df[f'{prefix}year'] = df[date_col].dt.year
  # Day of year for granular seasonality
  df[f'{prefix}day_of_year'] = df[date_col].dt.dayofyear
  # Cyclic features for smooth seasonality (using 366 days for robustness against leap years)
  df[f'{prefix}sin_day_of_year'] = np.sin(
      2 * np.pi * df[f'{prefix}day_of_year'] / 366
  )
  df[f'{prefix}cos_day_of_year'] = np.cos(
      2 * np.pi * df[f'{prefix}day_of_year'] / 366
  )

  # NEW: Add week-based and month-based features for seasonality
  df[f'{prefix}week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
  df[f'{prefix}month_of_year'] = df[date_col].dt.month

  # Cyclic features for week of year (using 53 weeks for robustness against varying number of weeks in a year)
  df[f'{prefix}sin_week_of_year'] = np.sin(
      2 * np.pi * df[f'{prefix}week_of_year'] / 53
  )
  df[f'{prefix}cos_week_of_year'] = np.cos(
      2 * np.pi * df[f'{prefix}week_of_year'] / 53
  )

  return df


def _calculate_lags(
    df_input,
    base_cols,
    lag_periods,
    group_col = 'location',
):
  """Calculates lagged features for specified base columns within each group.

  Returns the dataframe with new lag columns and the list of new lag column
  names.
  """
  df = df_input.copy()
  all_lag_cols = []
  # Sort by group_col and target_end_date to ensure correct lag calculation
  df = df.sort_values(by=[group_col, 'target_end_date'])
  for col_base in base_cols:
    for lag in lag_periods:
      new_col_name = f'lag_{col_base}_{lag}'
      df[new_col_name] = df.groupby(group_col)[col_base].shift(lag)
      all_lag_cols.append(new_col_name)
  return df, all_lag_cols


def _calculate_rolling_features(
    df_input,
    base_col,
    windows,
    group_col = 'location',
    shift_amount = 1,
):
  """Calculates rolling mean and std for a specified base column within each group,

  then shifts them to be available *before* the current week (to avoid data
  leakage).
  """
  df = df_input.copy()
  all_rolling_cols = []
  df = df.sort_values(by=[group_col, 'target_end_date'])
  for window in windows:
    new_col_mean = f'rolling_mean_{base_col}_{window}w_lag{shift_amount}'
    new_col_std = f'rolling_std_{base_col}_{window}w_lag{shift_amount}'
    # Use .rolling(min_periods=1) to allow calculation even if not a full window is available,
    # and then shift to prevent data leakage (feature for week T uses data up to T-shift_amount).
    df[new_col_mean] = df.groupby(group_col)[base_col].transform(
        lambda x: x.rolling(window=window, min_periods=1)
        .mean()
        .shift(shift_amount)
    )
    df[new_col_std] = df.groupby(group_col)[base_col].transform(
        lambda x: x.rolling(window=window, min_periods=1)
        .std()
        .shift(shift_amount)
    )
    all_rolling_cols.extend([new_col_mean, new_col_std])
  return df, all_rolling_cols


def _prepare_historical_data(
    train_x,
    train_y,
    ilinet_state_raw,
    locations_df,
):
  """Combines actual RSV and historical ILINet data, engineers base features.

  IMPROVEMENT: Added new time features, delta features and now rolling features.
  """
  train_x_copy = train_x.copy()
  train_y_copy = train_y.copy()
  train_y_copy.name = TARGET_STR  # Ensure target series has a name for merging

  # 1. Prepare 'actual RSV' data (from train_x, train_y)
  train_actual_rsv_df = train_x_copy.merge(
      train_y_copy, left_index=True, right_index=True, how='inner'
  )
  train_actual_rsv_df['target_end_date'] = pd.to_datetime(
      train_actual_rsv_df['target_end_date']
  )

  # Map population from global locations DataFrame for consistency and add log_population
  locations_df['population'] = pd.to_numeric(
      locations_df['population'], errors='coerce'
  )
  location_population_map = locations_df.set_index('location')[
      'population'
  ].to_dict()
  train_actual_rsv_df['population'] = train_actual_rsv_df['location'].map(
      location_population_map
  )
  # Handle potential NaNs or zeros in population for per-capita calculations
  train_actual_rsv_df['population'] = (
      train_actual_rsv_df['population']
      .fillna(EPSILON_POPULATION)
      .replace(0, EPSILON_POPULATION)
  )
  train_actual_rsv_df['log_population'] = np.log1p(
      train_actual_rsv_df['population']
  )

  # Rename actual target for clarity in the combined historical dataset
  train_actual_rsv_df = train_actual_rsv_df.rename(
      columns={TARGET_STR: 'actual_rsv_admissions'}
  )

  # Select core columns needed for further processing
  train_actual_rsv_df = train_actual_rsv_df[[
      'target_end_date',
      'location',
      'population',
      'log_population',
      'actual_rsv_admissions',
      'Total COVID-19 Admissions',
      'Total Influenza Admissions',
  ]].copy()
  # Fill any missing COVID/Flu admissions in the modern RSV data with 0
  train_actual_rsv_df['Total COVID-19 Admissions'] = (
      train_actual_rsv_df['Total COVID-19 Admissions'].fillna(0).clip(lower=0)
  )
  train_actual_rsv_df['Total Influenza Admissions'] = (
      train_actual_rsv_df['Total Influenza Admissions'].fillna(0).clip(lower=0)
  )

  # 2. Prepare historical ILINet data
  ilinet_state_processed = ilinet_state_raw.copy()
  ilinet_state_processed['target_end_date'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  ) + pd.Timedelta(days=6)

  # Filter ILINet data as per problem constraints (available only before '2022-10-15')
  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] < '2022-10-15'
  ].copy()

  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations_df[
          ['location', 'location_name', 'population']
      ].drop_duplicates(),
      on='location_name',
      how='left',
  )
  ilinet_state_processed.dropna(
      subset=['location'], inplace=True
  )  # Drop if location code not found
  ilinet_state_processed['location'] = ilinet_state_processed[
      'location'
  ].astype(int)
  # Handle potential NaNs or zeros in population for per-capita calculations
  ilinet_state_processed['population'] = (
      ilinet_state_processed['population']
      .fillna(EPSILON_POPULATION)
      .replace(0, EPSILON_POPULATION)
  )
  ilinet_state_processed['log_population'] = np.log1p(
      ilinet_state_processed['population']
  )

  # Select core columns for ILINet data
  ilinet_state_processed = ilinet_state_processed[[
      'target_end_date',
      'location',
      'population',
      'log_population',
      'ilitotal',
  ]].copy()
  # For ILINet-only dates, COVID/Flu admissions are not available, fill with 0
  ilinet_state_processed['Total COVID-19 Admissions'] = 0
  ilinet_state_processed['Total Influenza Admissions'] = 0

  # 3. Combine all historical data sources
  # Use pd.concat for better control over columns and to avoid implicit merges on auxiliary signals.
  # Create consistent column sets before concatenating.
  common_historical_cols = [
      'target_end_date',
      'location',
      'population',
      'log_population',
      'Total COVID-19 Admissions',
      'Total Influenza Admissions',
  ]

  df_rsv_processed_for_concat = train_actual_rsv_df[
      common_historical_cols + ['actual_rsv_admissions']
  ].copy()
  df_rsv_processed_for_concat = df_rsv_processed_for_concat.rename(
      columns={'actual_rsv_admissions': 'combined_rsv_ili'}
  )

  df_ili_processed_for_concat = ilinet_state_processed[
      common_historical_cols + ['ilitotal']
  ].copy()
  df_ili_processed_for_concat = df_ili_processed_for_concat.rename(
      columns={'ilitotal': 'combined_rsv_ili'}
  )

  # Concatenate and then fill combined_rsv_ili, preferring actual_rsv_admissions
  full_historical_df = pd.concat(
      [df_rsv_processed_for_concat, df_ili_processed_for_concat],
      ignore_index=True,
  )
  # Sort and drop duplicates, preferring actual RSV if dates overlap (though they shouldn't with current filtering)
  full_historical_df = full_historical_df.sort_values(
      by=['location', 'target_end_date', 'combined_rsv_ili'],
      ascending=[True, True, False],
  )
  full_historical_df = full_historical_df.drop_duplicates(
      subset=['location', 'target_end_date'], keep='first'
  )
  full_historical_df['combined_rsv_ili'] = (
      full_historical_df['combined_rsv_ili'].fillna(0).clip(lower=0)
  )

  # Sort by location and date for proper time-series operations
  full_historical_df = full_historical_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Add time-based features (now including cyclic features)
  full_historical_df = _create_time_features(
      full_historical_df, date_col='target_end_date'
  )

  # Calculate per-capita versions of all admission metrics
  base_admission_features = [
      'combined_rsv_ili',
      'Total COVID-19 Admissions',
      'Total Influenza Admissions',
  ]
  for col in base_admission_features:
    col_per_capita = f'{col}_per_capita'
    full_historical_df[col_per_capita] = (
        full_historical_df[col]
        / full_historical_df['population']
        * PER_CAPITA_SCALE
    )
    full_historical_df[col_per_capita] = (
        full_historical_df[col_per_capita].fillna(0).clip(lower=0)
    )

  # NEW: Calculate delta features for auxiliary signals. These will be used directly as features.
  full_historical_df['delta_covid_per_capita_1wk'] = (
      full_historical_df.groupby('location')[
          'Total COVID-19 Admissions_per_capita'
      ]
      .diff(1)
      .fillna(0)
  )
  full_historical_df['delta_flu_per_capita_1wk'] = (
      full_historical_df.groupby('location')[
          'Total Influenza Admissions_per_capita'
      ]
      .diff(1)
      .fillna(0)
  )

  # NEW IMPROVEMENT: Calculate rolling features for the target.
  # Shift by 1 to prevent data leakage (rolling mean/std of current week should not include current week's value)
  full_historical_df, new_rolling_cols = _calculate_rolling_features(
      full_historical_df,
      'combined_rsv_ili_per_capita',
      ROLLING_WINDOWS,
      shift_amount=1,
  )

  return (
      full_historical_df,
      new_rolling_cols,
  )  # Return rolling column names for feature list


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """
  with warnings.catch_warnings():
    warnings.simplefilter(
        'ignore'
    )  # Suppress warnings from pandas/sklearn/lightgbm during this function

    # --- 1. Initial Data Preparation & Comprehensive Historical Data Integration ---
    # Determine current reference date and training end date
    test_x_copy = test_x.copy()
    test_x_copy['target_end_date'] = pd.to_datetime(
        test_x_copy['target_end_date']
    )
    test_x_copy['reference_date'] = pd.to_datetime(
        test_x_copy['reference_date']
    )
    current_reference_date = test_x_copy['reference_date'].iloc[0]
    train_end_date = current_reference_date - pd.Timedelta(weeks=1)

    # Use the helper function to prepare the full historical dataset.
    # This data goes up to the latest available historical date from combined sources.
    # `ilinet_state` and `locations` are assumed to be globally available as per problem description.
    full_historical_df_processed, new_rolling_cols = _prepare_historical_data(
        train_x, train_y, ilinet_state, locations
    )

    # Get population data from the global 'locations' DataFrame for all locations
    location_population_map = locations.set_index('location')[
        'population'
    ].to_dict()

    # --- 2. Feature Engineering for Training Data (IMPROVED LAG & ROLLING CALCULATION) ---
    # Define base admission features for lag calculation (per capita)
    # IMPROVEMENT: Include per-capita COVID and Influenza for lag features
    base_admission_features_per_capita = [
        'combined_rsv_ili_per_capita',
        'Total COVID-19 Admissions_per_capita',
        'Total Influenza Admissions_per_capita',
    ]

    # IMPROVEMENT: Calculate all lagged features on the ENTIRE processed historical data first.
    # This ensures that lag features for training data can reach as far back as possible.
    full_historical_df_with_lags, all_lag_cols = _calculate_lags(
        full_historical_df_processed,
        base_admission_features_per_capita,
        LAG_PERIODS,
    )

    # Now filter this data for the actual training period
    train_df_with_features = full_historical_df_with_lags[
        full_historical_df_with_lags['target_end_date'] <= train_end_date
    ].copy()

    # For training, horizon is always 0 as we're predicting the actual observed values (or their historical proxy)
    train_df_with_features['horizon'] = 0

    # Define features used by the model, ensuring consistency between training and prediction
    # IMPROVEMENT: Added 'location' as categorical feature, new time features, and rolling features
    model_features = [
        'location',  # NEW: Added location as a categorical feature
        'year',
        'day_of_year',
        'sin_day_of_year',
        'cos_day_of_year',  # Day-based cyclic features
        'week_of_year',
        'sin_week_of_year',
        'cos_week_of_year',
        'month_of_year',  # NEW: Week/Month-based cyclic features
        'population',
        'log_population',
        'horizon',
        'Total COVID-19 Admissions_per_capita',  # NEW: Direct auxiliary signals
        'Total Influenza Admissions_per_capita',  # NEW: Direct auxiliary signals
        'delta_covid_per_capita_1wk',  # NEW: Delta features
        'delta_flu_per_capita_1wk',  # NEW: Delta features
    ]
    model_features.extend(all_lag_cols)
    model_features.extend(new_rolling_cols)  # NEW: Add rolling features

    # Filter model features to only include those present in the DataFrame (defensive check)
    model_features = [
        f for f in model_features if f in train_df_with_features.columns
    ]

    # Identify categorical features for LightGBM
    categorical_features = ['location']
    categorical_features = [
        f for f in categorical_features if f in model_features
    ]  # Ensure they are also in model_features

    # Drop rows with NaNs, which primarily arise from lag and rolling calculations at the very beginning of the time series
    valid_train_indices = (
        train_df_with_features[model_features + ['combined_rsv_ili_per_capita']]
        .dropna()
        .index
    )
    X_train = train_df_with_features.loc[valid_train_indices, model_features]
    # IMPROVEMENT: Apply log1p transformation to the target variable
    y_train = np.log1p(
        train_df_with_features.loc[
            valid_train_indices, 'combined_rsv_ili_per_capita'
        ]
    )

    # --- 3. Train LightGBM Models for Each Quantile ---
    quantile_models = {}
    for q in QUANTILES:
      # Using LightGBM's quantile objective for probabilistic forecasts.
      # Parameters are chosen for robustness and performance.
      # IMPROVEMENT: Increased n_estimators and slightly reduced learning_rate for potentially better accuracy
      # Pass categorical_feature parameter
      lgbm = lgb.LGBMRegressor(
          objective='quantile',
          alpha=q,
          random_state=42,
          n_estimators=500,
          learning_rate=0.02,
          num_leaves=31,  # Tuned hyperparameters
          n_jobs=-1,
          verbose=-1,  # Suppress verbose output during training
          min_child_samples=20,
          colsample_bytree=0.7,
          subsample=0.7,
      )
      lgbm.fit(
          X_train, y_train, categorical_feature=categorical_features
      )  # Pass categorical features
      quantile_models[q] = lgbm

    # --- 4. Iterative Prediction for Test Data (Recursive Forecasting) ---
    test_y_hat_quantiles = pd.DataFrame(
        index=test_x_copy.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )

    # 4.1. Prepare full prediction scaffold (all known historical data + test horizons)
    historical_part = full_historical_df_processed[
        full_historical_df_processed['target_end_date'] <= train_end_date
    ].copy()
    historical_part['horizon'] = (
        0  # Consistent horizon value for historical data
    )

    # Test part: build from test_x_copy provided features
    test_part = test_x_copy.copy()
    # _create_time_features now generates all new time features as well
    test_part = _create_time_features(test_part, date_col='target_end_date')

    test_part['population'] = test_part['location'].map(location_population_map)
    # Handle potential NaNs or zeros in population for per-capita calculations
    test_part['population'] = (
        test_part['population']
        .fillna(EPSILON_POPULATION)
        .replace(0, EPSILON_POPULATION)
    )
    test_part['log_population'] = np.log1p(test_part['population'])

    # Initialize per-capita target for test part with NaNs; these will be filled by predictions.
    test_part['combined_rsv_ili_per_capita'] = np.nan

    # For horizon 0, use provided admissions. For future horizons, these will be 0 initially or forward-filled.
    # Ensure Total COVID/Influenza Admissions columns exist in test_x_copy, default to 0 if not.
    test_part['Total COVID-19 Admissions_per_capita'] = (
        (
            test_x_copy.get('Total COVID-19 Admissions', 0)
            / test_part['population']
            * PER_CAPITA_SCALE
        )
        .fillna(0)
        .clip(lower=0)
    )
    test_part['Total Influenza Admissions_per_capita'] = (
        (
            test_x_copy.get('Total Influenza Admissions', 0)
            / test_part['population']
            * PER_CAPITA_SCALE
        )
        .fillna(0)
        .clip(lower=0)
    )

    # Select consistent columns for concatenation between historical and test parts
    # These columns must be present in both historical_part and test_part before concatenation.
    # IMPROVEMENT: Added new time features to common_cols_for_scaffold
    common_cols_for_scaffold = [
        'target_end_date',
        'location',
        'population',
        'log_population',
        'horizon',
        'year',
        'day_of_year',
        'sin_day_of_year',
        'cos_day_of_year',  # Existing time features
        'week_of_year',
        'sin_week_of_year',
        'cos_week_of_year',
        'month_of_year',  # NEW time features
        'combined_rsv_ili_per_capita',
        'Total COVID-19 Admissions_per_capita',
        'Total Influenza Admissions_per_capita',
    ]
    # Ensure selected columns exist in respective dataframes
    historical_cols_to_use = [
        col
        for col in common_cols_for_scaffold
        if col in historical_part.columns
    ]
    test_cols_to_use = [
        col for col in common_cols_for_scaffold if col in test_part.columns
    ]

    full_forecast_scaffold = pd.concat(
        [historical_part[historical_cols_to_use], test_part[test_cols_to_use]],
        ignore_index=True,
    )

    # Sort the entire scaffold by location and date to enable correct lag lookup
    full_forecast_scaffold = full_forecast_scaffold.sort_values(
        by=['location', 'target_end_date']
    ).reset_index(drop=True)

    # --- IMPROVEMENT: Forward-fill non-target admissions for future lags ---
    # This handles the case where future COVID/Flu admissions are not known/provided in test_x.
    # It assumes the last known value persists.
    full_forecast_scaffold['Total COVID-19 Admissions_per_capita'] = (
        full_forecast_scaffold.groupby('location')[
            'Total COVID-19 Admissions_per_capita'
        ].ffill()
    )
    full_forecast_scaffold['Total Influenza Admissions_per_capita'] = (
        full_forecast_scaffold.groupby('location')[
            'Total Influenza Admissions_per_capita'
        ].ffill()
    )
    # Fill any remaining NaNs (e.g., if a location has no historical data for these) with 0.
    full_forecast_scaffold['Total COVID-19 Admissions_per_capita'] = (
        full_forecast_scaffold['Total COVID-19 Admissions_per_capita'].fillna(0)
    )
    full_forecast_scaffold['Total Influenza Admissions_per_capita'] = (
        full_forecast_scaffold['Total Influenza Admissions_per_capita'].fillna(
            0
        )
    )

    # NEW: Calculate delta features on the full scaffold AFTER ffill for consistency
    full_forecast_scaffold['delta_covid_per_capita_1wk'] = (
        full_forecast_scaffold.groupby('location')[
            'Total COVID-19 Admissions_per_capita'
        ]
        .diff(1)
        .fillna(0)
    )
    full_forecast_scaffold['delta_flu_per_capita_1wk'] = (
        full_forecast_scaffold.groupby('location')[
            'Total Influenza Admissions_per_capita'
        ]
        .diff(1)
        .fillna(0)
    )

    # Create a mapping from (target_end_date, location) to the original test_x_copy index.
    test_x_orig_index_map = (
        test_x_copy.reset_index()
        .set_index(['target_end_date', 'location'])['index']
        .to_dict()
    )
    full_forecast_scaffold['original_test_x_index'] = (
        full_forecast_scaffold.apply(
            lambda row: test_x_orig_index_map.get(
                (row['target_end_date'], row['location']), np.nan
            ),
            axis=1,
        )
    )

    unique_test_target_dates = sorted(test_x_copy['target_end_date'].unique())

    # 4.2. Perform iterative prediction for each target_end_date in the test set (horizon by horizon)
    for current_target_date in unique_test_target_dates:
      # Identify rows in full_forecast_scaffold that need prediction for the current target date
      rows_to_predict_mask = (
          full_forecast_scaffold['target_end_date'] == current_target_date
      ) & pd.notna(full_forecast_scaffold['original_test_x_index'])

      if not rows_to_predict_mask.any():
        continue  # Skip if no rows to predict for this specific date

      current_test_rows_df = full_forecast_scaffold[rows_to_predict_mask].copy()

      # Prepare base features for the current batch of predictions, including new time and delta features.
      X_pred_batch = current_test_rows_df[[
          'location',
          'year',
          'day_of_year',
          'sin_day_of_year',
          'cos_day_of_year',
          'week_of_year',
          'sin_week_of_year',
          'cos_week_of_year',
          'month_of_year',
          'population',
          'log_population',
          'horizon',
          'Total COVID-19 Admissions_per_capita',
          'Total Influenza Admissions_per_capita',
          'delta_covid_per_capita_1wk',
          'delta_flu_per_capita_1wk',
      ]].copy()

      # Dynamically populate lag features for the current batch of predictions.
      for col_base_per_capita in base_admission_features_per_capita:
        for lag in LAG_PERIODS:
          lag_date = current_target_date - pd.Timedelta(weeks=lag)
          lag_col_name = f'lag_{col_base_per_capita}_{lag}'

          # Retrieve lagged values from the full_forecast_scaffold for the specific lag_date.
          lagged_values_df = full_forecast_scaffold[
              (full_forecast_scaffold['target_end_date'] == lag_date)
              & (
                  full_forecast_scaffold['location'].isin(
                      X_pred_batch['location']
                  )
              )
          ][['location', col_base_per_capita]].rename(
              columns={col_base_per_capita: lag_col_name}
          )

          # Merge these lagged values into the current prediction batch.
          X_pred_batch = pd.merge(
              X_pred_batch, lagged_values_df, on='location', how='left'
          )
          # Fill NaNs from missing history (e.g., very early in the series) or locations without prior data with 0.
          X_pred_batch[lag_col_name] = X_pred_batch[lag_col_name].fillna(0)

      # Dynamically populate rolling features for the current batch of predictions.
      # We need to re-calculate these on the portion of `full_forecast_scaffold` *up to and including* current_target_date.
      # The `_calculate_rolling_features` function includes a `shift_amount=1` which means it will use data *before* the current week.
      temp_scaffold_for_rolling = full_forecast_scaffold[
          full_forecast_scaffold['target_end_date'] <= current_target_date
      ].copy()

      # Re-calculate rolling features on this temporary scaffold
      temp_scaffold_for_rolling, current_rolling_cols_names = (
          _calculate_rolling_features(
              temp_scaffold_for_rolling,
              'combined_rsv_ili_per_capita',
              ROLLING_WINDOWS,
              shift_amount=1,
          )
      )

      # Extract the rolling features for the current target date
      rolling_features_for_current_date = temp_scaffold_for_rolling[
          (temp_scaffold_for_rolling['target_end_date'] == current_target_date)
          & (
              temp_scaffold_for_rolling['location'].isin(
                  X_pred_batch['location']
              )
          )
      ][['location'] + current_rolling_cols_names]

      # Merge these rolling features into X_pred_batch
      X_pred_batch = pd.merge(
          X_pred_batch,
          rolling_features_for_current_date,
          on='location',
          how='left',
      )
      # Fill any NaNs resulting from insufficient history for rolling windows with 0
      for col_name in current_rolling_cols_names:
        X_pred_batch[col_name] = X_pred_batch[col_name].fillna(0)

      # IMPROVEMENT: Ensure prediction feature order matches training feature order
      # This line remains crucial as it aligns the features for prediction to those trained on.
      X_pred_input = X_pred_batch[X_train.columns]

      # Predict quantiles for the current batch using each trained LightGBM model.
      preds_per_capita = pd.DataFrame(index=X_pred_input.index)
      for q_model in quantile_models:
        # IMPROVEMENT: Apply inverse log1p transformation (expm1) to predictions
        preds_per_capita[f'quantile_{q_model}'] = np.expm1(
            quantile_models[q_model].predict(X_pred_input)
        )

      # Scale predictions back from per-capita to total admissions using the population for the current row.
      population_for_scaling = current_test_rows_df['population'].values
      scaled_preds_values = (
          preds_per_capita.values
          / PER_CAPITA_SCALE
          * population_for_scaling[:, None]
      )

      # Map the scaled predictions back to the original test_x_copy index.
      original_test_indices = current_test_rows_df[
          'original_test_x_index'
      ].values.astype(int)
      test_y_hat_quantiles.loc[
          original_test_indices, [f'quantile_{q}' for q in QUANTILES]
      ] = scaled_preds_values

      # Update full_forecast_scaffold with the median prediction (per-capita) for this date.
      # This is crucial for subsequent iterations, as future lags and rolling features will depend on these forecasts.
      # Ensure median prediction used for recursive forecasting is also expm1'd.
      median_prediction_per_capita = np.expm1(
          quantile_models[0.5].predict(X_pred_input)
      )
      # Use `.loc` with the mask and column to avoid SettingWithCopyWarning
      full_forecast_scaffold.loc[
          rows_to_predict_mask, 'combined_rsv_ili_per_capita'
      ] = median_prediction_per_capita

    # --- 5. Post-processing: Ensure Monotonicity and Non-negativity ---
    # Round predictions to the nearest integer and ensure they are not negative.
    for col in test_y_hat_quantiles.columns:
      test_y_hat_quantiles[col] = test_y_hat_quantiles[col].apply(
          lambda x: max(0, np.round(x))
      )

    # Crucial step: Ensure predicted quantiles for any given row are monotonically increasing.
    test_y_hat_quantiles = test_y_hat_quantiles.apply(
        np.maximum.accumulate, axis=1
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
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc rsv hosp')
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
