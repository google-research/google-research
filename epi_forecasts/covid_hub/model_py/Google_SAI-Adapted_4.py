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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,g-bare-generic,reimported,g-doc-args,g-short-docstring-punctuation,g-doc-return-or-yield

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
MODEL_NAME = 'Google_SAI-Adapted_4'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Forecasting Method:
# 1. Target Transformation & Variance Stabilization: Apply a fourth-root transformation to the raw hospitalization counts.
# 2. Comprehensive Feature Engineering: Construct a feature matrix including autoregressive lags, demographic constants, cyclical calendar features (e.g., sin/cos of week-of-year), and the integer forecast horizon.
# 3. Quantile Gradient Boosting Ensemble: Train separate LightGBM regressors for each required quantile, optimizing the pinball loss function.
# 4. Iterative Prediction & Inverse Transformation: Generate transformed quantile forecasts for the horizon iteratively, using previous predictions for autoregressive features, and then apply the inverse fourth-power transformation to return to the natural count scale, followed by robust post-processing.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Any, List, Tuple, Dict, Set

# Configuration Constants (moved inside rewrite cell for self-containment)
MAX_INT64 = np.iinfo(np.int64).max
# IMPROVEMENT: Expanded LAG_WEEKS for more extensive feature engineering as per problem statement
LAG_WEEKS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
]  # Expanded lags for richer features
ROLLING_WINDOW_SIZES = [
    2,
    4,
    8,
    12,
]  # For 2-week, 4-week, 8-week, and 12-week rolling means
TARGET_SCALING_FACTOR = 100_000.0  # For per 100k population scaling
MIN_POPULATION_FOR_SCALING = (
    1.0  # Minimum population to prevent division by zero or extreme scaling
)

# Define a safe maximum for transformed predictions BEFORE raising to power 4
# (MAX_INT64)**0.25 is approximately 30958.8. Using 31000.0 as a safe upper bound
# to prevent float64 overflow during intermediate calculations (e.g., 31000^4).
MAX_TRANSFORMED_PREDICTION = 31000.0


# TARGET_STR and QUANTILES are assumed to be available from the notebook's global scope


def _add_base_features(
    df,
    all_locations_categories,
    all_year_categories,
    all_month_categories,
    first_train_date,
):
  """Adds static, time-index, cyclical calendar, and trend features to the DataFrame IN PLACE.

  Ensures 'location', 'year', and 'month' are categorical for LightGBM.
  'target_end_date' and 'location' are assumed pre-processed.
  """
  # Ensure categorical type for 'year' and 'month' columns for LightGBM with consistent categories
  year_categorical_dtype = pd.CategoricalDtype(
      categories=all_year_categories, ordered=False
  )
  month_categorical_dtype = pd.CategoricalDtype(
      categories=all_month_categories, ordered=False
  )

  df['year'] = df['target_end_date'].dt.year.astype(year_categorical_dtype)
  df['month'] = df['target_end_date'].dt.month.astype(month_categorical_dtype)
  df['weekofyear'] = df['target_end_date'].dt.isocalendar().week.astype(int)

  # Minor refinement: Normalize weekofyear for cyclical features to a 52-week cycle
  df['weekofyear_for_cycles'] = df['weekofyear'].apply(
      lambda w: (w - 1) % 52 + 1
  )  # Map week 53 to 1, week 1 to 1, etc.
  df['sin_weekofyear'] = np.sin(2 * np.pi * df['weekofyear_for_cycles'] / 52)
  df['cos_weekofyear'] = np.cos(2 * np.pi * df['weekofyear_for_cycles'] / 52)

  if 'horizon' not in df.columns:
    # 'horizon' is typically provided in test_x, but might be missing if df is just a historical slice.
    # Set to 0 if not present, indicating a 0-week forecast horizon (i.e., historical data point).
    df['horizon'] = 0

  # Add weeks since start feature to capture linear time trend
  df['weeks_since_start'] = (
      (df['target_end_date'] - first_train_date).dt.days // 7
  ).astype(int)

  # Ensure categorical type for 'location' column for LightGBM with consistent categories
  location_categorical_dtype = pd.CategoricalDtype(
      categories=all_locations_categories, ordered=False
  )
  df['location'] = df['location'].astype(location_categorical_dtype)


def _add_time_series_features_to_df(
    df,
    lag_source_series,  # MultiIndex Series (location, target_end_date) -> transformed_target
    rows_to_process_idx,  # The index of rows in df to fill features for
    lags,
    rolling_windows,
):
  """Adds/fills autoregressive lag and true rolling mean features to the input DataFrame IN PLACE

  for specified rows. Optimized for efficient lookup and calculation.
  """
  if rows_to_process_idx.empty:
    return

  current_locations = df.loc[rows_to_process_idx, 'location']
  current_target_end_dates = df.loc[rows_to_process_idx, 'target_end_date']

  # Determine all unique individual lag values needed (for direct lags and for rolling means)
  all_needed_individual_lags_set = set(lags)
  for window_size in rolling_windows:
    all_needed_individual_lags_set.update(range(1, window_size + 1))

  all_needed_individual_lags = sorted(list(all_needed_individual_lags_set))

  if not all_needed_individual_lags:  # No features to add
    return

  # Create a temporary DataFrame to hold all fetched lag values, indexed by original row index
  # This avoids repeated DataFrame creations for df_lookup_requests and pivot
  temp_fetched_lags_df = pd.DataFrame(index=rows_to_process_idx)

  for k in all_needed_individual_lags:
    lookup_dates_k = current_target_end_dates - pd.Timedelta(weeks=k)
    lookup_multiindex_k = pd.MultiIndex.from_arrays(
        [current_locations, lookup_dates_k],
        names=['location', 'target_end_date'],
    )

    # Fetch values from the lag_source_series for this lag k
    # The reindex aligns by (location, lookup_date_k) and returns a Series indexed by lookup_multiindex_k
    # We then extract values and assign to temp_fetched_lags_df where index matches rows_to_process_idx
    fetched_values_for_lag_k = lag_source_series.reindex(
        lookup_multiindex_k
    ).values
    temp_fetched_lags_df[f'_lag_{k}'] = fetched_values_for_lag_k

  # Now, process the fetched values to assign to the main DataFrame 'df'

  # For direct lags:
  for k in lags:
    df.loc[rows_to_process_idx, f'transformed_target_lag_{k}'] = (
        temp_fetched_lags_df[f'_lag_{k}']
    )

  # For rolling means:
  if rolling_windows:
    for window_size in rolling_windows:
      col_name = f'transformed_target_rolling_mean_{window_size}w'

      # Select the columns corresponding to lags for the current window (e.g., _lag_1 to _lag_window_size)
      window_cols_for_mean = [f'_lag_{k}' for k in range(1, window_size + 1)]

      # Check if all values within the window are non-NaN for each row (original_idx)
      # This enforces the "minimum period" requirement: all `window_size` lags must be present.
      window_data = temp_fetched_lags_df[window_cols_for_mean]
      all_lags_present_mask = window_data.notna().all(axis=1)

      # Compute the mean across the window_cols.
      computed_means = window_data.mean(axis=1)

      # Apply the mask: set mean to NaN if not all lags for the window were present
      computed_means[~all_lags_present_mask] = np.nan

      # Assign the computed rolling means back to the main DataFrame 'df' for the current rows.
      df.loc[rows_to_process_idx, col_name] = computed_means


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Returns quantiles.
  """
  # Store original test_x index to ensure the output DataFrame matches its structure.
  test_x_original_index = test_x.index.copy()

  # Ensure target_end_date is datetime and location is int for consistent processing
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  train_x['location'] = train_x['location'].astype(int)
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['location'] = test_x['location'].astype(int)

  # 1. Prepare Training Data and Variance Stabilization (with population scaling)
  # Combine features (train_x) and target (train_y) into a single DataFrame.
  train_df = pd.concat([train_x, train_y], axis=1)

  # Ensure target column is numeric before transformation, handle non-finite values if any.
  train_df[TARGET_STR] = pd.to_numeric(
      train_df[TARGET_STR], errors='coerce'
  ).fillna(0)

  # Ensure target values are non-negative before transformation.
  train_df[TARGET_STR] = train_df[TARGET_STR].clip(lower=0)

  # IMPROVEMENT: Robustly handle potential NaN values in 'population' before clipping
  # This prevents NaNs from propagating into 'population_clipped' and subsequent calculations.
  train_df['population_clipped'] = (
      train_df['population']
      .fillna(0)
      .astype(float)
      .clip(lower=MIN_POPULATION_FOR_SCALING)
  )
  train_df['scaled_target'] = train_df[TARGET_STR] / (
      train_df['population_clipped'] / TARGET_SCALING_FACTOR
  )

  # Apply the fourth-root transformation to the scaled hospitalization counts.
  train_df['transformed_target'] = np.power(train_df['scaled_target'], 0.25)

  # Get all unique location codes from both train and test to define categorical types consistently
  all_locations_categories = sorted(
      pd.concat([train_x['location'], test_x['location']]).unique().tolist()
  )

  # Get all unique year and month values across train and test for consistent categorical encoding
  all_target_end_dates = pd.concat(
      [train_x['target_end_date'], test_x['target_end_date']]
  ).drop_duplicates()
  all_year_categories = sorted(all_target_end_dates.dt.year.unique().tolist())
  all_month_categories = sorted(all_target_end_dates.dt.month.unique().tolist())

  # Determine the earliest date in the training data for weeks_since_start feature
  first_train_date = train_df['target_end_date'].min()

  # 2. Feature Engineering (Base Features + Lags + Rolling Means + Trend)

  # Add base features to both training and test dataframes (in-place)
  processed_train_df = train_df.copy()  # Work on a copy
  _add_base_features(
      processed_train_df,
      all_locations_categories,
      all_year_categories,
      all_month_categories,
      first_train_date,
  )

  processed_test_x = test_x.copy()  # Work on a copy
  # IMPROVEMENT: Robustly handle potential NaN values in 'population' for test_x too, before clipping
  processed_test_x['population_clipped'] = (
      processed_test_x['population']
      .fillna(0)
      .astype(float)
      .clip(lower=MIN_POPULATION_FOR_SCALING)
  )
  _add_base_features(
      processed_test_x,
      all_locations_categories,
      all_year_categories,
      all_month_categories,
      first_train_date,
  )

  # Create a unified MultiIndex Series for efficient lag lookups across train and test.
  # This series will hold all historical transformed targets and be updated with predictions.
  all_location_dates_for_multi_index = (
      pd.concat([
          processed_train_df[['location', 'target_end_date']],
          processed_test_x[['location', 'target_end_date']],
      ])
      .drop_duplicates()
      .sort_values(by=['location', 'target_end_date'])
  )

  full_multi_index = pd.MultiIndex.from_frame(
      all_location_dates_for_multi_index
  )

  # Initialize lag_store_series_data with transformed training targets
  lag_store_series_data = processed_train_df.set_index(
      ['location', 'target_end_date']
  )['transformed_target'].copy()
  # Reindex to ensure it covers all relevant future dates with NaN, ready for updates
  lag_store_series_data = lag_store_series_data.reindex(
      full_multi_index, fill_value=np.nan
  )

  # Generate autoregressive lag and rolling mean features for the TRAINING data (in-place).
  # This uses the `lag_store_series_data` which currently only contains historical values.
  _add_time_series_features_to_df(
      processed_train_df,
      lag_store_series_data,
      processed_train_df.index,
      LAG_WEEKS,
      ROLLING_WINDOW_SIZES,
  )

  # Define the complete list of feature columns.
  base_feature_cols = [
      'location',
      'population_clipped',
      'horizon',
      'year',
      'month',
      'weekofyear',
      'sin_weekofyear',
      'cos_weekofyear',
      'weeks_since_start',
  ]
  lag_feature_cols = [f'transformed_target_lag_{lag}' for lag in LAG_WEEKS]
  rolling_feature_cols = [
      f'transformed_target_rolling_mean_{window}w'
      for window in ROLLING_WINDOW_SIZES
  ]
  full_feature_cols = (
      base_feature_cols + lag_feature_cols + rolling_feature_cols
  )

  # Prepare feature matrix (X_train) and transformed target vector (y_train_transformed) for training.
  X_train = processed_train_df[full_feature_cols].copy()
  y_train_transformed = processed_train_df['transformed_target'].copy()

  # Drop rows with any NaN values (typically due to initial lags that extend beyond historical data).
  nan_rows_mask = X_train.isna().any(axis=1) | y_train_transformed.isna()
  X_train = X_train[~nan_rows_mask]
  y_train_transformed = y_train_transformed[~nan_rows_mask]

  if X_train.empty:
    warnings.warn(
        'No valid training data after feature engineering. Returning zero'
        ' predictions.'
    )
    test_y_hat_quantiles = pd.DataFrame(
        0,
        index=test_x_original_index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    return test_y_hat_quantiles

  # 3. Ensemble Training (LightGBM)
  models = {}
  # Explicitly define categorical feature columns for LightGBM
  categorical_feature_cols = ['location', 'year', 'month']

  for q in QUANTILES:
    lgbm = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        random_state=42,
        verbose=-1,
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        categorical_feature=categorical_feature_cols,
        n_jobs=-1,
    )
    lgbm.fit(X_train, y_train_transformed)
    models[f'quantile_{q}'] = lgbm

  # 4. Iterative Prediction for Test Data (Handling Autoregressive Lags and Rolling Means)

  # Pre-allocate DataFrame for all test predictions.
  all_test_predictions_transformed = pd.DataFrame(
      index=processed_test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Get unique target dates from the test set, sorted chronologically.
  unique_test_target_dates = sorted(
      processed_test_x['target_end_date'].unique()
  )

  for current_date in unique_test_target_dates:
    # Identify rows in processed_test_x that correspond to the current prediction date.
    current_date_mask = processed_test_x['target_end_date'] == current_date
    current_date_mask_idx = processed_test_x.index[
        current_date_mask
    ]  # Get the actual index for these rows

    # Fill autoregressive lag and rolling mean features for the current period's test rows IN PLACE.
    # This uses `lag_store_series_data` which now contains historical data and previous median predictions.
    _add_time_series_features_to_df(
        processed_test_x,
        lag_store_series_data,
        rows_to_process_idx=current_date_mask_idx,
        lags=LAG_WEEKS,
        rolling_windows=ROLLING_WINDOW_SIZES,
    )

    # Extract the features for prediction, ensuring their column order matches `X_train`.
    X_pred_current = processed_test_x.loc[
        current_date_mask_idx, full_feature_cols
    ]

    # Predict all quantiles for the current_date using the trained ensemble models.
    current_preds_transformed = pd.DataFrame(index=X_pred_current.index)
    for q_label, model in models.items():
      current_preds_transformed[q_label] = model.predict(X_pred_current)

    # Store predictions in the pre-allocated DataFrame.
    all_test_predictions_transformed.loc[current_preds_transformed.index] = (
        current_preds_transformed
    )

    # Update `lag_store_series_data` IN-PLACE with the median prediction for the `current_date`.
    # This is crucial for subsequent autoregressive predictions.
    median_preds = current_preds_transformed['quantile_0.5']

    # Create MultiIndex for the current predictions to update lag_store_series efficiently.
    current_idx_for_update = pd.MultiIndex.from_arrays(
        [
            processed_test_x.loc[current_date_mask_idx, 'location'],
            processed_test_x.loc[current_date_mask_idx, 'target_end_date'],
        ],
        names=['location', 'target_end_date'],
    )

    # Update `lag_store_series_data` in place
    lag_store_series_data.loc[current_idx_for_update] = median_preds.values

  if all_test_predictions_transformed.empty:
    warnings.warn(
        'No predictions generated for test set. Returning zero predictions.'
    )
    test_y_hat_quantiles = pd.DataFrame(
        0,
        index=test_x_original_index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    return test_y_hat_quantiles

  # Reindex the combined predictions to match the original order of `test_x`.
  test_y_hat_quantiles_transformed = all_test_predictions_transformed.reindex(
      test_x_original_index
  )

  # 5. Inverse Transformation & Post-processing
  # Clip transformed predictions to prevent potential float64 overflow when raising to the power of 4.
  # Values are also clipped to be non-negative before the power transformation.
  clipped_transformed_preds = test_y_hat_quantiles_transformed.clip(
      0, MAX_TRANSFORMED_PREDICTION
  )

  # Raise predicted values to the fourth power to revert the variance stabilization.
  unscaled_transformed_preds = np.power(clipped_transformed_preds, 4)

  # Inverse population scaling using 'population_clipped' from processed_test_x
  test_y_hat_quantiles = unscaled_transformed_preds.multiply(
      processed_test_x.loc[
          unscaled_transformed_preds.index, 'population_clipped'
      ]
      / TARGET_SCALING_FACTOR,
      axis=0,
  )

  # Crucial robustness: Fill any NaNs that might have emerged (e.g., from 0.0 ** NaN issues) with 0
  # BEFORE clipping, rounding, and casting to integer.
  test_y_hat_quantiles = test_y_hat_quantiles.fillna(0)

  # Clip again to ensure non-negativity and values are within int64 bounds,
  # then round, and cast to np.int64. Use float(MAX_INT64) for precise clipping.
  max_clip_value_float = float(MAX_INT64)
  test_y_hat_quantiles = test_y_hat_quantiles.clip(
      lower=0, upper=max_clip_value_float
  )
  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(np.int64)

  # 6. Monotonicity Enforcement
  # Apply cumulative maximum across columns for each row to ensure quantiles are
  # strictly monotonically increasing (e.g., q0.01 <= q0.025 <= ... <= q0.99).
  test_y_hat_quantiles = test_y_hat_quantiles.cummax(axis=1)

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
