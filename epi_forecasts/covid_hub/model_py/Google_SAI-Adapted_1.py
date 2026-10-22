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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,g-bare-generic,reimported,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation

import datetime

from absl import app
import pandas as pd

from constant_defs import HORIZONS
from constant_defs import QUANTILES
from constant_defs import REQUIRED_CDC_LOCATIONS
from constant_defs import TARGET_STR
from epi_utils import _create_fold_scaffold
from epi_utils import compute_rolling_evaluation
from epi_utils import format_for_cdc
from epi_utils import get_most_recent_saturday_date_str
from epi_utils import get_next_saturday_date_str
from epi_utils import get_saturdays_between_dates
from plotting_utils import plot_season_forecasts

timedelta = datetime.timedelta


INPUT_DIR = ''
MODEL_NAME = 'Google_SAI-Adapted_1'
TARGET_STR = 'Total COVID-19 Admissions'
# Alias matching the global defined in the Kaggle notebooks the models were
# developed in; some fit_and_predict_fn implementations reference it directly.
horizons = HORIZONS

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the forecasting method:
# 1. Enhanced Tabular Feature Engineering: Generation of a rich set of time-series features (lags, diverse rolling statistics, differences, ratios, acceleration, cyclical components) from historical data and static covariates, implicitly encoding sequence information into a tabular format.
# 2. Global Multi-Region LightGBM Training: A single LightGBM model is trained simultaneously across all geographic regions on the tabular features, enabling shared learning patterns (implicit weight sharing) for a global loss objective.
# 3. Autoregressive Point Forecasting: Predictions are generated step-by-step for multiple horizons. Forecasts for future horizons leverage previous forecasts as lagged inputs through dynamic feature recalculation during the inference loop, simulating a sequential decoding process.
# 4. Adaptive Conformal Quantile Calibration (ACI): Prediction intervals are derived by combining point forecasts with empirical quantiles of recent *relative* residuals, calculated on a rolling hold-out set. This method dynamically adjusts interval width proportional to the prediction magnitude, improving calibration for heteroscedastic data and guaranteeing valid uncertainty.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Union, List, Dict, Tuple

# Global constants (TARGET_STR, QUANTILES, horizons) are assumed to be defined outside this cell.
# Redefining for local clarity if this cell were executed standalone, but they're globally available.
# TARGET_STR = 'Total COVID-19 Admissions'
TARGET_STR = 'Total COVID-19 Admissions'

# QUANTILES = [0.01, ..., 0.99]
# horizons = [0, 1, 2, 3]

TARGET_PER_CAPITA_STR = TARGET_STR + '_per_capita'

# --- Configuration Constants for the improved model ---
LAG_FEATURES_COUNT = 26  # Captures up to half a year of lags.
ROLLING_WINDOW_SIZES = [
    2,
    4,
    8,
    12,
    16,
]  # Added 12 and 16-week window for more diverse rolling statistics
CALIBRATION_WINDOW_WEEKS = (
    16  # Increased from 12 for more robust residual estimation
)
EPSILON_FOR_RELATIVE_RESIDUALS = (
    1e-6  # Small constant to prevent division by zero in relative residuals.
)
MAX_RATIO_CAP = (
    100.0  # Cap for extremely high growth ratios to prevent outliers.
)


def _add_date_features(df):
  """Adds cyclical date features to the DataFrame."""
  df['target_end_date'] = pd.to_datetime(df['target_end_date'])
  df['week_of_year'] = df['target_end_date'].dt.isocalendar().week.astype(int)
  df['month'] = df['target_end_date'].dt.month
  df['year'] = df['target_end_date'].dt.year
  df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
  df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
  return df.drop(columns=['week_of_year'])


def _add_lag_features(
    df,
    target_col_name,
    lag_count,
    context_series = None,
    is_training_data = False,
):
  """Generates lagged features for the target variable.

  Handles both training (shift) and prediction (context lookup).
  """
  for lag in range(1, lag_count + 1):
    lag_col_name = f'{target_col_name}_lag_{lag}'
    if is_training_data:
      df[lag_col_name] = df.groupby('location')[target_col_name].shift(lag)
    else:
      if (
          context_series is not None and len(context_series) > 0
      ):  # Ensure context_series is not empty for lookup
        # Calculate the date corresponding to the lagged value for each row
        lag_dates_for_current_df = df['target_end_date'] - pd.Timedelta(
            weeks=lag
        )
        valid_rows_mask = (
            df['location'].notna() & lag_dates_for_current_df.notna()
        )

        if valid_rows_mask.any():
          # Create lookup keys (e.g., '1_2021-01-02') and map from context_series
          keys = (
              df.loc[valid_rows_mask, 'location'].astype(str)
              + '_'
              + lag_dates_for_current_df.loc[valid_rows_mask].dt.strftime(
                  '%Y-%m-%d'
              )
          )
          df.loc[valid_rows_mask, lag_col_name] = keys.map(context_series)
        else:
          df[lag_col_name] = np.nan  # No valid rows to compute lags for
      else:
        df[lag_col_name] = (
            np.nan
        )  # No context provided or context is empty for prediction mode
  return df


def _add_rolling_stats_from_lags(
    df,
    target_col_name,
    lag_count,
    rolling_window_sizes,
):
  """Generates rolling statistics *across* specified lag features for each row.

  E.g., for window=4, it takes mean(lag_1, lag_2, lag_3, lag_4).
  """
  for window_size in rolling_window_sizes:
    # Define the lags that form this "rolling window" of features
    lag_cols_for_window = [
        f'{target_col_name}_lag_{i}'
        for i in range(1, min(window_size, lag_count) + 1)
    ]
    actual_lag_cols_for_rolling = [
        col for col in lag_cols_for_window if col in df.columns
    ]

    if actual_lag_cols_for_rolling:
      # Ensure numeric type for calculations, coerce errors to NaN
      lagged_values_matrix = df[actual_lag_cols_for_rolling].apply(
          pd.to_numeric, errors='coerce'
      )

      df[f'{target_col_name}_rolling_mean_{window_size}'] = (
          lagged_values_matrix.mean(axis=1)
      )
      df[f'{target_col_name}_rolling_std_{window_size}'] = (
          lagged_values_matrix.std(axis=1)
      )
      df[f'{target_col_name}_rolling_min_{window_size}'] = (
          lagged_values_matrix.min(axis=1)
      )
      df[f'{target_col_name}_rolling_max_{window_size}'] = (
          lagged_values_matrix.max(axis=1)
      )
      df[f'{target_col_name}_rolling_median_{window_size}'] = (
          lagged_values_matrix.median(axis=1)
      )
      df[f'{target_col_name}_rolling_sum_{window_size}'] = (
          lagged_values_matrix.sum(axis=1)
      )

      # NEW: Add rolling coefficient of variation
      # To avoid division by zero for rolling_std / rolling_mean, use a small epsilon
      df[f'{target_col_name}_rolling_cov_{window_size}'] = df[
          f'{target_col_name}_rolling_std_{window_size}'
      ] / (
          df[f'{target_col_name}_rolling_mean_{window_size}']
          + EPSILON_FOR_RELATIVE_RESIDUALS
      )
      df[f'{target_col_name}_rolling_cov_{window_size}'] = df[
          f'{target_col_name}_rolling_cov_{window_size}'
      ].replace([np.inf, -np.inf], np.nan)
  return df


def _add_growth_features(
    df, target_col_name, max_ratio_cap = MAX_RATIO_CAP
):
  """Adds difference, ratio, and acceleration features derived from lagged values."""

  def _calculate_ratio_robust(
      numerator_series, denominator_series, cap
  ):
    """Helper to calculate ratio robustly handling zero denominators."""
    ratio_series = pd.Series(np.nan, index=numerator_series.index, dtype=float)

    valid_mask = numerator_series.notna() & denominator_series.notna()

    # Case 1: Denominator is not zero
    non_zero_denom_mask = valid_mask & (denominator_series != 0)
    ratio_series.loc[non_zero_denom_mask] = (
        numerator_series[non_zero_denom_mask]
        / denominator_series[non_zero_denom_mask]
    )

    # Case 2: Denominator is zero
    zero_denom_mask = valid_mask & (denominator_series == 0)

    # Subcase 2a: Denominator is zero and numerator is positive (infinite growth)
    infinite_growth_mask = zero_denom_mask & (numerator_series > 0)
    ratio_series.loc[infinite_growth_mask] = cap

    # Subcase 2b: Denominator is zero and numerator is zero (no change)
    zero_to_zero_mask = zero_denom_mask & (numerator_series == 0)
    ratio_series.loc[zero_to_zero_mask] = 1.0  # Treat as no change

    return ratio_series.clip(upper=cap)

  # Pre-fetch relevant lags for efficiency and clean access
  # Need up to lag_5 for some features like diff_4, ratio_4, acceleration_1_3_5
  lag_values = {i: df.get(f'{target_col_name}_lag_{i}') for i in range(1, 6)}

  # Diff and Ratio 1 (Current vs. 1-week prior)
  if lag_values[1] is not None and lag_values[2] is not None:
    df[f'{target_col_name}_diff_1'] = lag_values[1] - lag_values[2]
    df[f'{target_col_name}_ratio_1'] = _calculate_ratio_robust(
        lag_values[1], lag_values[2], max_ratio_cap
    )

  # Diff 2 (Current vs. 2-weeks prior)
  if lag_values[1] is not None and lag_values[3] is not None:
    df[f'{target_col_name}_diff_2'] = lag_values[1] - lag_values[3]

  # Acceleration 1-2-3 ((t-1)-(t-2) vs (t-2)-(t-3))
  if (
      lag_values[1] is not None
      and lag_values[2] is not None
      and lag_values[3] is not None
  ):
    first_order_diff_1_2 = lag_values[1] - lag_values[2]
    first_order_diff_2_3 = lag_values[2] - lag_values[3]
    df[f'{target_col_name}_acceleration_1_2_3'] = (
        first_order_diff_1_2 - first_order_diff_2_3
    )

  # Longer-term difference and ratio features (Current vs. 4-weeks prior, which is lag_1 vs lag_5)
  if lag_values[1] is not None and lag_values[5] is not None:
    df[f'{target_col_name}_diff_4'] = lag_values[1] - lag_values[5]
    df[f'{target_col_name}_ratio_4'] = _calculate_ratio_robust(
        lag_values[1], lag_values[5], max_ratio_cap
    )

  # Additional acceleration feature (1-3 vs 3-5) using lags 1, 3, 5
  if (
      lag_values[1] is not None
      and lag_values[3] is not None
      and lag_values[5] is not None
  ):
    first_order_diff_1_3 = lag_values[1] - lag_values[3]
    first_order_diff_3_5 = lag_values[3] - lag_values[5]
    df[f'{target_col_name}_acceleration_1_3_5'] = (
        first_order_diff_1_3 - first_order_diff_3_5
    )

  # NEW: Ratio of rolling means for capturing longer-term growth trends
  # Requires rolling_mean features to be already computed. Assume a window of 4 vs 8.
  if (
      f'{target_col_name}_rolling_mean_4' in df.columns
      and f'{target_col_name}_rolling_mean_8' in df.columns
  ):
    df[f'{target_col_name}_ratio_rolling_mean_4_8'] = _calculate_ratio_robust(
        df[f'{target_col_name}_rolling_mean_4'],
        df[f'{target_col_name}_rolling_mean_8'],
        max_ratio_cap,
    )

  return df


def _add_interaction_features(
    df, target_col_name
):
  """Adds interaction features, specifically lag-1 target with log-population and cyclical date features."""
  if (
      f'{target_col_name}_lag_1' in df.columns
      and 'log_population' in df.columns
  ):
    df[f'{target_col_name}_lag_1_pop_interaction'] = (
        df[f'{target_col_name}_lag_1'] * df['log_population']
    )

  if 'log_population' in df.columns and 'week_of_year_sin' in df.columns:
    df['log_pop_week_sin_interaction'] = (
        df['log_population'] * df['week_of_year_sin']
    )
  if 'log_population' in df.columns and 'week_of_year_cos' in df.columns:
    df['log_pop_week_cos_interaction'] = (
        df['log_population'] * df['week_of_year_cos']
    )

  # NEW: Interaction features with horizon
  if 'horizon' in df.columns and 'log_population' in df.columns:
    df['horizon_log_pop_interaction'] = df['horizon'] * df['log_population']
  if 'horizon' in df.columns and 'week_of_year_sin' in df.columns:
    df['horizon_week_sin_interaction'] = df['horizon'] * df['week_of_year_sin']
  if 'horizon' in df.columns and 'week_of_year_cos' in df.columns:
    df['horizon_week_cos_interaction'] = df['horizon'] * df['week_of_year_cos']

  return df


def _generate_all_features(
    df,
    target_col_name,
    lag_count,
    rolling_window_sizes,
    context_series = None,
    is_training_data = False,
    max_ratio_cap = MAX_RATIO_CAP,
):
  """Orchestrates the generation of all time-series and static features.

  This serves as the "Enhanced Sequence Feature Engineering" step.
  """
  df_processed = df.copy()

  # Add log-transformed population feature
  if 'population' in df_processed.columns:
    # Replace 0 or NaN population with 1 for log1p safety, assuming population cannot be truly zero or negative for relevant calculations
    df_processed['log_population'] = np.log1p(
        df_processed['population'].replace(0, 1).fillna(1).astype(float)
    )
  else:
    df_processed['log_population'] = (
        np.nan
    )  # Ensure column exists even if population is missing

  df_processed = _add_date_features(df_processed)
  df_processed = _add_lag_features(
      df_processed, target_col_name, lag_count, context_series, is_training_data
  )
  df_processed = _add_rolling_stats_from_lags(
      df_processed, target_col_name, lag_count, rolling_window_sizes
  )
  df_processed = _add_growth_features(
      df_processed, target_col_name, max_ratio_cap
  )
  df_processed = _add_interaction_features(df_processed, target_col_name)

  return df_processed


def _generate_model_features_list_base(
    lag_count, rolling_window_sizes, target_per_capita_str
):
  """Dynamically generates the *base* list of potential feature names."""
  features = []
  # Lag features
  features.extend(
      [f'{target_per_capita_str}_lag_{i}' for i in range(1, lag_count + 1)]
  )

  # Rolling statistics based on lags
  for window in rolling_window_sizes:
    features.extend([
        f'{target_per_capita_str}_rolling_mean_{window}',
        f'{target_per_capita_str}_rolling_std_{window}',
        f'{target_per_capita_str}_rolling_min_{window}',
        f'{target_per_capita_str}_rolling_max_{window}',
        f'{target_per_capita_str}_rolling_median_{window}',
        f'{target_per_capita_str}_rolling_sum_{window}',
        f'{target_per_capita_str}_rolling_cov_{window}',
    ])  # NEW

  # Difference, ratio, and acceleration features (from _add_growth_features)
  # Ensure these names align with those generated in _add_growth_features
  features.extend([
      f'{target_per_capita_str}_diff_1',
      f'{target_per_capita_str}_ratio_1',
      f'{target_per_capita_str}_diff_2',
      f'{target_per_capita_str}_acceleration_1_2_3',
      f'{target_per_capita_str}_diff_4',
      f'{target_per_capita_str}_ratio_4',
      f'{target_per_capita_str}_acceleration_1_3_5',
      f'{target_per_capita_str}_ratio_rolling_mean_4_8',
  ])  # NEW

  # Static and cyclical date features
  features.extend([
      'population',
      'log_population',
      'month',
      'year',
      'location',
      'week_of_year_sin',
      'week_of_year_cos',
  ])

  # Interaction features (Updated with new ones)
  features.append(f'{target_per_capita_str}_lag_1_pop_interaction')
  features.append('log_pop_week_sin_interaction')
  features.append('log_pop_week_cos_interaction')

  # NEW: Direct horizon feature and its interactions
  features.append('horizon')  # Direct horizon feature
  features.append('horizon_log_pop_interaction')
  features.append('horizon_week_sin_interaction')
  features.append('horizon_week_cos_interaction')

  return features


def _get_historical_context_dictionary_optimized(
    global_history_per_capita_series,
    global_history_date_series,
    cutoff_date,
):
  """Generates a dictionary mapping 'location_YYYY-MM-DD' to historical

  per-capita target values for dates strictly before the cutoff_date,
  using pre-computed global history series for efficiency.
  """
  relevant_keys_mask = global_history_date_series < cutoff_date
  relevant_keys_index = global_history_date_series.index[relevant_keys_mask]

  if relevant_keys_index.empty:
    return {}
  return global_history_per_capita_series.loc[relevant_keys_index].to_dict()


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Makes probabilistic forecasts for Total COVID-19 Hospital Admissions using a LightGBM

  model with enhanced time-series features and Adaptive Conformal Inference
  (ACI).
  """

  # 0. Initial Data Preparation and Type Coercion
  train_df_full = train_x.copy()
  train_df_full[TARGET_STR] = train_y

  train_df_full['target_end_date'] = pd.to_datetime(
      train_df_full['target_end_date']
  )
  test_x_processed = test_x.copy()
  test_x_processed['target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  )
  test_x_processed['reference_date'] = pd.to_datetime(
      test_x_processed['reference_date']
  )

  train_df_full = train_df_full.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Calculate per-capita target for training data, handling zero/missing population robustly
  population_safe_train = (
      train_df_full['population'].replace(0, 1).fillna(1).astype(float)
  )
  train_df_full[TARGET_PER_CAPITA_STR] = (
      train_df_full[TARGET_STR] / population_safe_train
  )
  train_df_full[TARGET_PER_CAPITA_STR] = train_df_full[
      TARGET_PER_CAPITA_STR
  ].fillna(0.0)

  # Pre-calculate population for test data for later conversion from per-capita
  test_x_processed['population_safe'] = (
      test_x_processed['population'].replace(0, 1).fillna(1).astype(float)
  )

  # Create a global history series for efficient lagged feature lookup during autoregression
  train_df_full['key'] = (
      train_df_full['location'].astype(str)
      + '_'
      + train_df_full['target_end_date'].dt.strftime('%Y-%m-%d')
  )
  _GLOBAL_HISTORY_PER_CAPITA_SERIES = train_df_full.set_index('key')[
      TARGET_PER_CAPITA_STR
  ]
  _GLOBAL_HISTORY_DATE_SERIES = train_df_full.set_index('key')[
      'target_end_date'
  ]

  # Determine the last date in the training data
  train_end_date = train_df_full['target_end_date'].max()

  # DataFrame to store final quantile predictions
  test_y_hat_quantiles = pd.DataFrame(index=test_x_processed.index)

  # Dynamically generate the *base* list of potential model features
  MODEL_FEATURES_BASE = _generate_model_features_list_base(
      LAG_FEATURES_COUNT, ROLLING_WINDOW_SIZES, TARGET_PER_CAPITA_STR
  )

  # --- 1. Enhanced Sequence Feature Engineering for Training Data ---
  train_df_model = _generate_all_features(
      df=train_df_full.copy(),  # Pass a copy to avoid modifying original train_df_full
      target_col_name=TARGET_PER_CAPITA_STR,
      lag_count=LAG_FEATURES_COUNT,
      rolling_window_sizes=ROLLING_WINDOW_SIZES,
      is_training_data=True,  # Indicate this is training data to use shift method for lags
      max_ratio_cap=MAX_RATIO_CAP,
  )

  # NEW: Add 'horizon' feature to training data (will be NaN for most, but useful for feature list)
  # This makes the feature set consistent. For training, 'horizon' is typically 0, or not applicable.
  # However, for consistency with the prediction features, we add it. Its values will be 0 or NaN.
  if 'horizon' not in train_df_model.columns:
    train_df_model['horizon'] = (
        0  # Or NaN, depending on how this feature is used for past data
    )

  # Finalize model features: filter to only those actually generated and configured in the training data
  FINAL_MODEL_FEATURES = [
      col for col in MODEL_FEATURES_BASE if col in train_df_model.columns
  ]

  # Drop rows with NaN values in target or key features for clean training
  required_cols_for_training = FINAL_MODEL_FEATURES + [TARGET_PER_CAPITA_STR]
  train_df_model_cleaned = train_df_model.dropna(
      subset=required_cols_for_training
  ).copy()

  X_train = train_df_model_cleaned[FINAL_MODEL_FEATURES]
  y_train = train_df_model_cleaned[TARGET_PER_CAPITA_STR]

  # Explicitly convert categorical features to 'category' dtype for LightGBM
  categorical_features_to_convert = ['location', 'month']
  for col in categorical_features_to_convert:
    if col in X_train.columns:
      X_train[col] = X_train[col].astype('category')

  # Fallback for insufficient training data
  if X_train.empty or y_train.empty:
    warnings.warn(
        'Training data is insufficient after feature engineering (X_train or'
        ' y_train is empty). Falling back to simple median forecast.',
        UserWarning,
    )
    global_median_admissions = train_y.median() if not train_y.empty else 0
    for q in QUANTILES:
      test_y_hat_quantiles[f'quantile_{q}'] = test_x_processed['horizon'].apply(
          lambda h: global_median_admissions
      )
    return test_y_hat_quantiles

  # --- 2. Global Multi-Task Point Forecast Model Training (LightGBM) ---
  actual_categorical_features = [
      col for col in categorical_features_to_convert if col in X_train.columns
  ]

  model = lgb.LGBMRegressor(
      objective='regression_l1',  # Mean Absolute Error objective, robust to outliers
      n_estimators=3000,  # Increased estimators for better learning
      learning_rate=0.003,  # Conservative learning rate
      num_leaves=31,  # Moderate complexity
      max_depth=-1,  # No limit on tree depth
      min_child_samples=10,  # Minimum data in a leaf
      random_state=42,  # For reproducibility
      n_jobs=-1,  # Use all available CPU cores
      verbose=-1,  # Suppress verbose output during training
      colsample_bytree=0.7,  # Feature subsampling
      subsample=0.7,  # Data subsampling
      reg_alpha=0.1,  # L1 regularization
      reg_lambda=0.1,  # L2 regularization
  )
  model.fit(X_train, y_train, categorical_feature=actual_categorical_features)

  # --- 3. Autoregressive Point Forecasting for Test Set ---
  all_point_forecasts_for_test = pd.DataFrame(index=test_x_processed.index)
  all_point_forecasts_for_test['point_forecast'] = np.nan
  all_point_forecasts_for_test['point_forecast_pc'] = (
      np.nan
  )  # Store per-capita predictions

  unique_ref_dates = sorted(test_x_processed['reference_date'].unique())

  # Loop through each unique forecast reference date in the test set
  for ref_date in unique_ref_dates:
    ref_date_test_rows = (
        test_x_processed[test_x_processed['reference_date'] == ref_date]
        .sort_values('horizon')
        .copy()
    )

    # Initialize context dictionary for current reference date's autoregressive predictions
    current_forecast_context_dict_for_this_ref_date = (
        _get_historical_context_dictionary_optimized(
            _GLOBAL_HISTORY_PER_CAPITA_SERIES,
            _GLOBAL_HISTORY_DATE_SERIES,
            ref_date,
        )
    )

    # Autoregressive loop: predict horizon by horizon for the current reference date
    for h in sorted(horizons):
      current_horizon_rows = ref_date_test_rows[
          ref_date_test_rows['horizon'] == h
      ].copy()

      if current_horizon_rows.empty:
        continue

      # Generate features for the current horizon using the dynamic context
      # Crucially, pass horizon as a feature here:
      features_for_pred = _generate_all_features(
          df=current_horizon_rows.copy(),  # Pass a copy to avoid modifying original current_horizon_rows
          target_col_name=TARGET_PER_CAPITA_STR,
          lag_count=LAG_FEATURES_COUNT,
          rolling_window_sizes=ROLLING_WINDOW_SIZES,
          context_series=current_forecast_context_dict_for_this_ref_date,
          is_training_data=False,  # Indicate this is prediction data to use context lookup
          max_ratio_cap=MAX_RATIO_CAP,
      )

      # Filter features to only those the model was trained on
      features_for_pred_filtered = features_for_pred[[
          col
          for col in FINAL_MODEL_FEATURES
          if col in features_for_pred.columns
      ]]

      # Explicitly convert categorical features to 'category' dtype for LightGBM prediction
      for col in categorical_features_to_convert:
        if col in features_for_pred_filtered.columns:
          features_for_pred_filtered[col] = features_for_pred_filtered[
              col
          ].astype('category')

      if (
          features_for_pred_filtered.empty
          or features_for_pred_filtered.shape[0] == 0
      ):
        current_horizon_predictions_pc = np.array([])
      else:
        # Make per-capita point predictions
        current_horizon_predictions_pc = model.predict(
            features_for_pred_filtered
        )
        current_horizon_predictions_pc = np.maximum(
            0, current_horizon_predictions_pc
        )  # Ensure non-negative predictions

      # Convert per-capita predictions back to total admissions
      current_horizon_predictions = (
          current_horizon_predictions_pc
          * current_horizon_rows['population_safe'].values
      )
      current_horizon_predictions = np.maximum(0, current_horizon_predictions)

      # Store predictions and update the context dictionary for subsequent horizons
      for i, idx in enumerate(current_horizon_rows.index):
        loc = current_horizon_rows.loc[idx, 'location']
        target_date = current_horizon_rows.loc[idx, 'target_end_date']

        predicted_raw_val = (
            current_horizon_predictions[i]
            if i < len(current_horizon_predictions)
            else 0.0
        )
        predicted_pc_val = (
            current_horizon_predictions_pc[i]
            if i < len(current_horizon_predictions_pc)
            else 0.0
        )

        all_point_forecasts_for_test.loc[idx, 'point_forecast'] = (
            predicted_raw_val
        )
        all_point_forecasts_for_test.loc[idx, 'point_forecast_pc'] = (
            predicted_pc_val  # Store per-capita prediction
        )

        key = str(loc) + '_' + target_date.strftime('%Y-%m-%d')
        current_forecast_context_dict_for_this_ref_date[key] = predicted_pc_val

  # --- 4. Adaptive Conformal Quantile Calibration (ACI) Simulation ---
  calibration_relative_residuals_by_horizon = {h: [] for h in horizons}
  all_relative_residuals_collected_globally = []

  # Define the calibration window (e.g., last CALIBRATION_WINDOW_WEEKS of training data as reference dates)
  calibration_end_ref_date = train_end_date
  calibration_start_ref_date = calibration_end_ref_date - pd.Timedelta(
      weeks=CALIBRATION_WINDOW_WEEKS - 1
  )
  calibration_ref_dates = pd.date_range(
      start=calibration_start_ref_date,
      end=calibration_end_ref_date,
      freq='W-SAT',
  )

  # Handle case where calibration window cannot be formed
  if calibration_ref_dates.empty:
    warnings.warn(
        'Not enough data to form calibration window. Quantiles will be based on'
        ' point forecast only (zero uncertainty).',
        UserWarning,
    )
    for q in QUANTILES:
      test_y_hat_quantiles[f'quantile_{q}'] = all_point_forecasts_for_test[
          'point_forecast'
      ]
    return test_y_hat_quantiles

  # Prepare location information for scaffold creation (assumes 'locations' is globally defined, or derived)
  unique_location_codes = test_x_processed['location'].unique()

  # Create a scaffold for all calibration reference dates and horizons
  all_cal_scaffolds = []
  for cal_ref_date in calibration_ref_dates:
    cal_fold_scaffold = _create_fold_scaffold(  # _create_fold_scaffold is defined outside this cell
        cal_ref_date,
        horizons,
        unique_location_codes,
        locations,  # `locations` DataFrame is assumed to be global
    )
    all_cal_scaffolds.append(cal_fold_scaffold)

  calibration_scaffold = pd.concat(all_cal_scaffolds, ignore_index=True)
  calibration_scaffold['target_end_date'] = pd.to_datetime(
      calibration_scaffold['target_end_date']
  )
  calibration_scaffold['reference_date'] = pd.to_datetime(
      calibration_scaffold['reference_date']
  )

  # Filter the calibration scaffold to only include dates within the training period that have actuals
  calibration_scaffold_filtered = calibration_scaffold[
      calibration_scaffold['target_end_date'] <= train_end_date
  ].copy()

  # Merge with actual per-capita target values from the training data
  calibration_scaffold_with_actuals = calibration_scaffold_filtered.merge(
      train_df_full[['target_end_date', 'location', TARGET_PER_CAPITA_STR]],
      on=['target_end_date', 'location'],
      how=(  # Use left join to keep all calibration scaffold entries and add actuals where available
          'left'
      ),
  )

  # Simulate autoregressive predictions for the calibration period to get residuals
  for cal_ref_date in calibration_ref_dates:
    current_cal_forecast_context_dict = (
        _get_historical_context_dictionary_optimized(
            _GLOBAL_HISTORY_PER_CAPITA_SERIES,
            _GLOBAL_HISTORY_DATE_SERIES,
            cal_ref_date,
        )
    )

    cal_rows_for_this_ref_date = (
        calibration_scaffold_with_actuals[(
            calibration_scaffold_with_actuals['reference_date'] == cal_ref_date
        )]
        .sort_values('horizon')
        .copy()
    )

    for h in sorted(horizons):
      current_horizon_cal_rows = cal_rows_for_this_ref_date[
          cal_rows_for_this_ref_date['horizon'] == h
      ].copy()

      if current_horizon_cal_rows.empty:
        continue

      # Generate features for calibration prediction
      features_for_cal_pred = _generate_all_features(
          df=current_horizon_cal_rows.copy(),  # Pass a copy
          target_col_name=TARGET_PER_CAPITA_STR,
          lag_count=LAG_FEATURES_COUNT,
          rolling_window_sizes=ROLLING_WINDOW_SIZES,
          context_series=current_cal_forecast_context_dict,
          is_training_data=False,
          max_ratio_cap=MAX_RATIO_CAP,
      )
      features_for_cal_pred_filtered = features_for_cal_pred[[
          col
          for col in FINAL_MODEL_FEATURES
          if col in features_for_cal_pred.columns
      ]]

      # Explicitly convert categorical features to 'category' dtype for LightGBM prediction
      for col in categorical_features_to_convert:
        if col in features_for_cal_pred_filtered.columns:
          features_for_cal_pred_filtered[col] = features_for_cal_pred_filtered[
              col
          ].astype('category')

      if (
          features_for_cal_pred_filtered.empty
          or features_for_cal_pred_filtered.shape[0] == 0
      ):
        cal_predictions_pc = np.array([])
      else:
        cal_predictions_pc = model.predict(features_for_cal_pred_filtered)
        cal_predictions_pc = np.maximum(
            0, cal_predictions_pc
        )  # Ensure non-negative

      # Calculate and collect relative residuals for each prediction point
      for i, idx in enumerate(current_horizon_cal_rows.index):
        loc = current_horizon_cal_rows.loc[idx, 'location']
        target_date = current_horizon_cal_rows.loc[idx, 'target_end_date']
        actual_pc_val = current_horizon_cal_rows.loc[idx, TARGET_PER_CAPITA_STR]

        pred_val_pc = (
            cal_predictions_pc[i] if i < len(cal_predictions_pc) else 0.0
        )

        if pd.notna(actual_pc_val):
          # Calculate the relative residual: (actual - predicted) / (predicted + epsilon)
          relative_residual = (actual_pc_val - pred_val_pc) / (
              pred_val_pc + EPSILON_FOR_RELATIVE_RESIDUALS
          )
          calibration_relative_residuals_by_horizon[h].append(relative_residual)
          all_relative_residuals_collected_globally.append(relative_residual)

        # Update the context dictionary with the current prediction for autoregressive use in later horizons
        key = str(loc) + '_' + target_date.strftime('%Y-%m-%d')
        current_cal_forecast_context_dict[key] = pred_val_pc

  # Check if any relative residuals were collected for calibration
  any_relative_residuals_collected_globally_flag = bool(
      all_relative_residuals_collected_globally
  )

  if not any_relative_residuals_collected_globally_flag:
    warnings.warn(
        'No calibration relative residuals could be computed globally.'
        ' Quantiles will be based on point forecast only (zero uncertainty).',
        UserWarning,
    )
    for q in QUANTILES:
      test_y_hat_quantiles[f'quantile_{q}'] = all_point_forecasts_for_test[
          'point_forecast'
      ]
    return test_y_hat_quantiles

  # Calculate empirical quantiles of relative residuals for each horizon
  horizon_error_quantiles = {}
  for h in horizons:
    if calibration_relative_residuals_by_horizon[h]:
      horizon_error_quantiles[h] = np.quantile(
          calibration_relative_residuals_by_horizon[h], QUANTILES
      )
    else:
      # Fallback to global error distribution if horizon-specific residuals are insufficient
      warnings.warn(
          f'No calibration relative residuals for horizon {h}. Using global'
          ' error distribution for this horizon.',
          UserWarning,
      )
      horizon_error_quantiles[h] = np.quantile(
          all_relative_residuals_collected_globally, QUANTILES
      )

  # Apply ACI to test set point forecasts to generate final quantile predictions
  for idx in test_x_processed.index:
    h = test_x_processed.loc[idx, 'horizon']
    point_forecast_raw = all_point_forecasts_for_test.loc[idx, 'point_forecast']
    point_forecast_pc = all_point_forecasts_for_test.loc[
        idx, 'point_forecast_pc'
    ]  # Use stored per-capita forecast
    location_pop = test_x_processed.loc[
        idx, 'population_safe'
    ]  # Use safe population for final total conversion

    if (
        h in horizon_error_quantiles
        and any_relative_residuals_collected_globally_flag
    ):  # Check this flag for overall ACI validity
      for i, q in enumerate(QUANTILES):
        # Scale the residual quantile by the point forecast (per-capita) to get absolute error
        # Adding EPSILON_FOR_RELATIVE_RESIDUALS ensures scaling is always positive
        absolute_error_estimate_pc = horizon_error_quantiles[h][i] * (
            point_forecast_pc + EPSILON_FOR_RELATIVE_RESIDUALS
        )

        # Add this absolute error estimate to the point forecast to get the quantile forecast
        forecast_pc_q = point_forecast_pc + absolute_error_estimate_pc

        # Ensure non-negativity of per-capita quantile forecast before scaling
        forecast_pc_q = np.maximum(0, forecast_pc_q)

        # Convert the per-capita quantile forecast back to total admissions
        test_y_hat_quantiles.loc[idx, f'quantile_{q}'] = (
            forecast_pc_q * location_pop
        )
    else:
      # In the rare case of no error quantiles for a horizon (should be caught by earlier warnings), use point forecast
      for q in QUANTILES:
        test_y_hat_quantiles.loc[idx, f'quantile_{q}'] = point_forecast_raw

  # Final Post-processing: Ensure non-negativity, monotonicity, and integer rounding
  # First, ensure all quantile columns are numeric before operations
  for col in test_y_hat_quantiles.columns:
    test_y_hat_quantiles[col] = pd.to_numeric(
        test_y_hat_quantiles[col], errors='coerce'
    ).fillna(
        0
    )  # Coerce non-numeric to NaN then fill 0
    test_y_hat_quantiles[col] = test_y_hat_quantiles[col].clip(lower=0)

  for idx in test_y_hat_quantiles.index:
    row_values = test_y_hat_quantiles.loc[
        idx, [f'quantile_{q}' for q in QUANTILES]
    ].values
    # Apply cummax to ensure monotonicity: each quantile must be >= previous one
    monotonic_values = np.maximum.accumulate(row_values)
    test_y_hat_quantiles.loc[idx, [f'quantile_{q}' for q in QUANTILES]] = (
        monotonic_values
    )

  for col in test_y_hat_quantiles.columns:
    test_y_hat_quantiles[col] = test_y_hat_quantiles[col].round().astype(int)

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
