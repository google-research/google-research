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

# pylint: disable=g-bad-import-order,reimported,g-importing-member,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,f-string-without-interpolation,cell-var-from-loop

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
MODEL_NAME = 'Google_SAI-Novel_1'
TARGET_STR = ''

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
from collections import deque
import math  # Import math for sine/cosine functions

# The following constants are assumed to be globally available as per the notebook structure:
# TARGET_STR = 'Total RSV Admissions'
# QUANTILES = [...]
# locations: pd.DataFrame
# ilinet_state: pd.DataFrame
# ilinet_hhs, ilinet: (not used in this specific plan for state-level forecasting, but available)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a LightGBM model with historical augmentation and quantile forecasts.

  Args:
      train_x: DataFrame of historical features.
      train_y: Series of historical target values (Total RSV Admissions).
      test_x: DataFrame of future dates for which to make predictions.

  Returns:
      pd.DataFrame: Quantile predictions matching test_x index and required
      columns.
  """
  warnings.filterwarnings(
      'ignore', category=UserWarning
  )  # Suppress warnings from LGBM if verbose=-1 not enough

  # 1. Data Preprocessing and Merging
  # Combine train_x and train_y
  df_train_actual = train_x.merge(
      train_y.rename(TARGET_STR), left_index=True, right_index=True, how='left'
  )
  df_train_actual['target_end_date'] = pd.to_datetime(
      df_train_actual['target_end_date']
  )

  # Drop unnecessary columns first to keep data cleaner. These columns often contain mostly NaNs in practice
  # and are not used in the model.
  cols_to_drop_from_raw_train = [
      'geography',
      'ed_trends_ari',
      'ed_trends_covid',
      'ed_trends_influenza',
      'ed_trends_rsv',
      'hsa',
      'hsa_counties',
      'hsa_nci_id',
      'fips',
      'trend_source',
      'ari_threshold_classification',
      'covid_threshold_classification',
      'influenza_threshold_classification',
      'rsv_threshold_classification',
      'abbreviation',
      'count_rate0p3',
      'count_rate0p5',
      'count_rate0p7',
      'count_rate1',
      'count_rate1p7',
      'count_rate3',
      'count_rate4',
      'count_rate5',
  ]
  df_train_actual = df_train_actual.drop(
      columns=[
          col
          for col in cols_to_drop_from_raw_train
          if col in df_train_actual.columns
      ],
      errors='ignore',
  )

  # Merge population data from 'locations' (globally available)
  # Drop existing population/location_name from train_x to use authoritative 'locations' data
  df_train_actual = df_train_actual.drop(
      columns=['population', 'location_name'], errors='ignore'
  )
  df_train_actual = pd.merge(
      df_train_actual,
      locations[['location', 'population', 'location_name']],
      on='location',
      how='left',
  )

  # Rename 'percent_visits_rsv' from train_x to 'NSSP_RSV_ED_percent' for consistency
  if 'percent_visits_rsv' in df_train_actual.columns:
    df_train_actual['NSSP_RSV_ED_percent'] = df_train_actual[
        'percent_visits_rsv'
    ]
  else:
    df_train_actual['NSSP_RSV_ED_percent'] = (
        0.0  # Default to 0 if column doesn't exist or is missing
    )

  # Ensure necessary columns are present and filled for features. These are typically counts, so 0 is a reasonable fill.
  for col in [
      'Total COVID-19 Admissions',
      'Total Influenza Admissions',
      'NSSP_RSV_ED_percent',
  ]:
    if (
        col not in df_train_actual.columns
    ):  # Should not happen if previous steps are correct, but for robustness
      df_train_actual[col] = 0.0
    df_train_actual[col] = df_train_actual[col].fillna(
        0.0
    )  # Fill NaNs if present

  # 2. Prepare ilinet_state for historical augmentation
  ilinet_state_processed = ilinet_state.copy()
  ilinet_state_processed['target_end_date'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  ) + pd.Timedelta(
      days=6
  )  # Adjust to Saturday
  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] < '2022-10-15'
  ]  # Filter as per problem statement

  # Merge ilinet with location FIPS codes using location_name
  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations[['location', 'location_name', 'population']],
      left_on='region',
      right_on='location_name',
      how='left',
  )
  ilinet_state_processed = ilinet_state_processed.dropna(
      subset=['location']
  )  # Drop rows where location merge failed (e.g., non-state regions)

  # Filter for relevant ILI columns
  ilinet_state_processed = ilinet_state_processed[
      ['location', 'target_end_date', 'unweighted_ili', 'population']
  ]
  ilinet_state_processed['unweighted_ili'] = ilinet_state_processed[
      'unweighted_ili'
  ].fillna(
      0.0
  )  # Fill ILI NaNs

  # 3. Historical Augmentation - Learn Transformation (Strategy 2 - Refined)
  # Merge actual RSV with ILI data to find an overlap for ratio calculation
  overlap_df = pd.merge(
      df_train_actual.dropna(subset=[TARGET_STR]),
      ilinet_state_processed,
      on=['location', 'target_end_date'],
      how='inner',
      suffixes=('_rsv', '_ili'),
  )

  median_rsv_ili_ratio_per_loc = {}
  global_median_rsv_ili_ratio = (
      1.0  # Default fallback if no valid overlap exists
  )

  if not overlap_df.empty:
    # Calculate global median ratio from non-zero ILI overlap
    non_zero_ili_overlap_corrected = overlap_df[
        overlap_df['unweighted_ili'] > 0
    ]
    if not non_zero_ili_overlap_corrected.empty:
      ratios = (
          non_zero_ili_overlap_corrected[TARGET_STR]
          / non_zero_ili_overlap_corrected['unweighted_ili']
      )
      global_median_rsv_ili_ratio = ratios.median()
      global_median_rsv_ili_ratio = np.clip(
          global_median_rsv_ili_ratio, 0.0, 100.0
      )  # Clip extreme ratios

    # Calculate location-specific median ratios
    for loc_id in overlap_df['location'].unique():
      loc_overlap = overlap_df[overlap_df['location'] == loc_id]
      loc_non_zero_ili = loc_overlap[loc_overlap['unweighted_ili'] > 0]
      if not loc_non_zero_ili.empty:
        loc_ratios = (
            loc_non_zero_ili[TARGET_STR] / loc_non_zero_ili['unweighted_ili']
        )
        loc_median_ratio = loc_ratios.median()
        median_rsv_ili_ratio_per_loc[loc_id] = np.clip(
            loc_median_ratio, 0.0, 100.0
        )
      else:
        median_rsv_ili_ratio_per_loc[loc_id] = (
            global_median_rsv_ili_ratio  # Fallback to global
        )

  # Generate synthetic RSV using the determined ratios
  synthetic_ilinet_rsv = ilinet_state_processed.copy()
  synthetic_ilinet_rsv[TARGET_STR] = synthetic_ilinet_rsv.apply(
      lambda row: row['unweighted_ili']
      * median_rsv_ili_ratio_per_loc.get(
          row['location'], global_median_rsv_ili_ratio
      ),
      axis=1,
  ).clip(lower=0)

  # For synthetic data, set modern-era admission counts to 0 and population from 'locations'
  synthetic_ilinet_rsv['Total COVID-19 Admissions'] = 0.0
  synthetic_ilinet_rsv['Total Influenza Admissions'] = 0.0
  synthetic_ilinet_rsv['NSSP_RSV_ED_percent'] = (
      0.0  # ED percent not relevant for old ILINet data
  )

  # Combine actual training data and synthetic ILI data for the full historical training set
  base_features = [
      TARGET_STR,
      'Total COVID-19 Admissions',
      'Total Influenza Admissions',
      'NSSP_RSV_ED_percent',
  ]
  cols_to_combine = [
      'location',
      'target_end_date',
      'population',
  ] + base_features

  df_combined_past = pd.concat(
      [df_train_actual[cols_to_combine], synthetic_ilinet_rsv[cols_to_combine]],
      ignore_index=True,
  )

  df_combined_past = df_combined_past.sort_values(
      by=['location', 'target_end_date']
  )
  # Keep actual RSV if both actual and synthetic exist for the same date/location
  # Use 'first' to prioritize df_train_actual as it's the first in concat
  df_combined_past = df_combined_past.drop_duplicates(
      subset=['location', 'target_end_date'], keep='first'
  )

  # Add 'is_synthetic' flag: True if original TARGET_STR was NaN and filled by synthetic
  actual_dates_locs = df_train_actual[
      ['location', 'target_end_date']
  ].drop_duplicates()
  df_combined_past = pd.merge(
      df_combined_past,
      actual_dates_locs.assign(is_actual=1),
      on=['location', 'target_end_date'],
      how='left',
  )
  df_combined_past['is_synthetic'] = df_combined_past['is_actual'].isna()
  df_combined_past.drop(columns='is_actual', inplace=True)

  df_combined_past[TARGET_STR] = df_combined_past[TARGET_STR].fillna(
      0.0
  )  # Ensure target is not NaN for training

  # 4. Feature Engineering on df_combined_past
  df_combined_past['week_of_year'] = (
      df_combined_past['target_end_date'].dt.isocalendar().week.astype(int)
  )
  df_combined_past['month'] = df_combined_past['target_end_date'].dt.month
  df_combined_past['day_of_year'] = df_combined_past[
      'target_end_date'
  ].dt.dayofyear
  df_combined_past['year'] = df_combined_past['target_end_date'].dt.year

  # --- Start of Improvement 1: Add Seasonal Cyclical Features (already present) ---
  # Max week of year is 52 for most years, sometimes 53. Use 52.14 for average weeks in a year.
  # For simplicity and robustness against 52/53 week years, normalize by 52.
  df_combined_past['week_sin'] = np.sin(
      2 * np.pi * df_combined_past['week_of_year'] / 52.14
  )
  df_combined_past['week_cos'] = np.cos(
      2 * np.pi * df_combined_past['week_of_year'] / 52.14
  )

  df_combined_past['day_sin'] = np.sin(
      2 * np.pi * df_combined_past['day_of_year'] / 365.25
  )
  df_combined_past['day_cos'] = np.cos(
      2 * np.pi * df_combined_past['day_of_year'] / 365.25
  )
  # --- End of Improvement 1 ---

  # Location encoding
  le = LabelEncoder()
  # Fit LabelEncoder on all unique locations from combined data (actual + synthetic + test_x locations)
  all_locations = pd.concat(
      [df_combined_past['location'], test_x['location']]
  ).unique()
  le.fit(all_locations)
  df_combined_past['location_encoded'] = le.transform(
      df_combined_past['location']
  )

  # Lag features for all relevant columns
  lags = [1, 2, 3, 4, 8, 12, 26, 52]

  for col in base_features:
    for l in lags:
      df_combined_past[f'lag_{l}_{col}'] = df_combined_past.groupby('location')[
          col
      ].shift(l)

  # Rolling mean/std features
  roll_windows = [4, 8, 12]
  for col in base_features:
    for rw in roll_windows:
      df_combined_past[f'roll_mean_{rw}_{col}'] = df_combined_past.groupby(
          'location'
      )[col].transform(
          lambda x: x.rolling(window=rw, min_periods=1).mean().shift(1)
      )
      df_combined_past[f'roll_std_{rw}_{col}'] = df_combined_past.groupby(
          'location'
      )[col].transform(
          lambda x: x.rolling(window=rw, min_periods=1).std().shift(1).fillna(0)
      )  # Fill std NaN with 0

  # Interaction features (using lagged values to avoid data leakage)
  df_combined_past['pop_rsv_interaction_lag1'] = (
      df_combined_past['population'] * df_combined_past[f'lag_1_{TARGET_STR}']
  )
  df_combined_past['rsv_flu_ratio_lag1'] = df_combined_past[
      f'lag_1_{TARGET_STR}'
  ] / (df_combined_past[f'lag_1_Total Influenza Admissions'] + 1e-6)
  df_combined_past['rsv_covid_ratio_lag1'] = df_combined_past[
      f'lag_1_{TARGET_STR}'
  ] / (df_combined_past[f'lag_1_Total COVID-19 Admissions'] + 1e-6)

  # --- Start of Improvement 2: Add additional interaction feature ---
  df_combined_past['pop_NSSP_RSV_ED_interaction_lag1'] = (
      df_combined_past['population']
      * df_combined_past[f'lag_1_NSSP_RSV_ED_percent']
  )
  # --- End of Improvement 2 ---

  # Fill NaNs created by feature engineering for training with 0
  df_combined_past = df_combined_past.fillna(0.0)

  # Filter out rows with target NaN (should not happen after previous fills, but good for robustness)
  df_combined_past = df_combined_past.dropna(subset=[TARGET_STR])

  # 5. Model Training
  feature_columns = (
      [
          'week_of_year',
          'month',
          'day_of_year',
          'year',
          'population',
          'location_encoded',
          'is_synthetic',
          'week_sin',
          'week_cos',
          'day_sin',
          'day_cos',  # Added cyclical features
      ]
      + [f'lag_{l}_{col}' for col in base_features for l in lags]
      + [
          f'roll_mean_{rw}_{col}'
          for col in base_features
          for rw in roll_windows
      ]
      + [f'roll_std_{rw}_{col}' for col in base_features for rw in roll_windows]
      + [
          'pop_rsv_interaction_lag1',
          'rsv_flu_ratio_lag1',
          'rsv_covid_ratio_lag1',
          'pop_NSSP_RSV_ED_interaction_lag1',
      ]
  )  # Added new interaction feature

  # Filter feature_columns to only include those actually present in df_combined_past
  feature_columns = [
      col for col in feature_columns if col in df_combined_past.columns
  ]

  X_train = df_combined_past[feature_columns]
  y_train = df_combined_past[TARGET_STR]

  quantile_models = {}
  for q in QUANTILES:
    # Hyperparameters for LGBMRegressor (tuned for reasonable performance)
    # --- Start of Improvement 1: Increase num_leaves ---
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=q,
        random_state=42,
        n_estimators=500,  # Increased from 300 to 500
        learning_rate=0.03,
        n_jobs=-1,
        verbose=-1,
        max_depth=7,
        num_leaves=63,
    )  # Increased num_leaves from 31 to 63
    # --- End of Improvement 1 ---
    model.fit(X_train, y_train)
    quantile_models[q] = model

  # 6. Prediction for test_x (Accurate Recursive State Management)
  test_x_copy = test_x.copy()
  test_x_copy['target_end_date'] = pd.to_datetime(
      test_x_copy['target_end_date']
  )
  test_x_copy['week_of_year'] = (
      test_x_copy['target_end_date'].dt.isocalendar().week.astype(int)
  )
  test_x_copy['month'] = test_x_copy['target_end_date'].dt.month
  test_x_copy['day_of_year'] = test_x_copy['target_end_date'].dt.dayofyear
  test_x_copy['year'] = test_x_copy['target_end_date'].dt.year

  # --- Apply Seasonal Cyclical Features to test_x_copy ---
  test_x_copy['week_sin'] = np.sin(
      2 * np.pi * test_x_copy['week_of_year'] / 52.14
  )
  test_x_copy['week_cos'] = np.cos(
      2 * np.pi * test_x_copy['week_of_year'] / 52.14
  )
  test_x_copy['day_sin'] = np.sin(
      2 * np.pi * test_x_copy['day_of_year'] / 365.25
  )
  test_x_copy['day_cos'] = np.cos(
      2 * np.pi * test_x_copy['day_of_year'] / 365.25
  )
  # --- End of Seasonal Cyclical Features for test_x_copy ---

  test_x_copy['location_encoded'] = le.transform(test_x_copy['location'])
  test_x_copy['is_synthetic'] = (
      False  # These are future predictions, not synthetic historical data points
  )

  # The `train_end_date` for the current fold, used to determine cutoff for historical data.
  current_fold_train_end_date = test_x_copy[
      'reference_date'
  ].min() - pd.Timedelta(weeks=1)

  # Max lookback for lags and rolling windows (plus one for current value)
  max_lookback = max(lags + roll_windows) + 1

  # Initialize location_history_state with deques for raw values of base_features
  location_history_state = {}

  # Helper to get an empty state for a new location
  def get_empty_history_deque_state():
    return {col: deque(maxlen=max_lookback) for col in base_features}

  # Populate initial location_history_state from df_combined_past up to current_fold_train_end_date
  historical_data_for_lags = df_combined_past[
      df_combined_past['target_end_date'] <= current_fold_train_end_date
  ].copy()
  historical_data_for_lags = historical_data_for_lags.sort_values(
      by=['location', 'target_end_date']
  )

  for loc_id in historical_data_for_lags['location'].unique():
    location_history_state[loc_id] = get_empty_history_deque_state()

  # Fill deques with actual historical data
  for _, row_hist in historical_data_for_lags.iterrows():
    loc_id = row_hist['location']
    for col in base_features:
      location_history_state[loc_id][col].append(row_hist[col])

  # Dictionary to store median predictions for (location, target_end_date) within the current test_x batch.
  # This allows for recursive updates of TARGET_STR lags.
  predictions_as_lags = {}

  # Store predictions in a DataFrame
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x_copy.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Sort test_x to ensure chronological processing for recursive lag updates
  test_x_sorted = test_x_copy.sort_values(by=['location', 'target_end_date'])

  for original_idx, row in test_x_sorted.iterrows():
    location = row['location']
    target_end_date = row['target_end_date']

    # Get current state for this location, or an empty one if new
    current_location_state = location_history_state.get(
        location, get_empty_history_deque_state()
    )

    X_test_row = pd.DataFrame(index=[0], columns=feature_columns)

    # Populate direct features
    for feat in [
        'week_of_year',
        'month',
        'day_of_year',
        'year',
        'population',
        'location_encoded',
        'is_synthetic',
        'week_sin',
        'week_cos',
        'day_sin',
        'day_cos',
    ]:  # Include new cyclical features
      if feat in X_test_row.columns:
        X_test_row[feat] = row[feat]

    # Dynamically calculate lags and rolling features for the current test_x row
    for col in base_features:
      past_values = list(
          current_location_state[col]
      )  # Get current history for this col (up to previous week)

      # Calculate lags
      for l_idx, l in enumerate(lags):
        lag_val = 0.0
        if col == TARGET_STR:
          # Priority 1: Check if this lag_date was predicted in the current test_x batch (recursive prediction)
          lag_date_key = (location, target_end_date - pd.Timedelta(weeks=l))
          if lag_date_key in predictions_as_lags:
            lag_val = predictions_as_lags[lag_date_key]
          # Priority 2: Look into the 'past_values' deque (actual historical data)
          elif len(past_values) >= l:  # Check if enough history
            lag_val = past_values[-l]
          # Else, lag_val remains 0.0 (e.g., if history is too short)
        else:  # For non-TARGET_STR base_features (COVID, Flu, ED%)
          # For these features, future values are unknown. Lags should refer to the last known actuals
          # from the training period, carried forward.
          if not past_values:  # If no history, assume 0
            lag_val = 0.0
          elif len(past_values) >= l:  # Get actual historical lag if available
            lag_val = past_values[-l]
          else:  # If lag points beyond available history, use the last available historical value
            lag_val = past_values[
                -1
            ]  # This implements "carry forward last known value" for lags

        if f'lag_{l}_{col}' in X_test_row.columns:
          X_test_row[f'lag_{l}_{col}'] = lag_val

      # Calculate rolling mean/std features
      for rw in roll_windows:
        if not past_values:  # If no history at all
          if f'roll_mean_{rw}_{col}' in X_test_row.columns:
            X_test_row[f'roll_mean_{rw}_{col}'] = 0.0
          if f'roll_std_{rw}_{col}' in X_test_row.columns:
            X_test_row[f'roll_std_{rw}_{col}'] = 0.0
        else:
          # Use the last 'rw' elements from past_values, or all if less than 'rw'
          # These values implicitly represent the 'shifted' rolling window based on history up to the previous week.
          effective_roll_data = (
              past_values[-rw:] if len(past_values) >= rw else past_values
          )

          if f'roll_mean_{rw}_{col}' in X_test_row.columns:
            X_test_row[f'roll_mean_{rw}_{col}'] = np.mean(effective_roll_data)
          if f'roll_std_{rw}_{col}' in X_test_row.columns:
            # np.std returns 0 for a single element, which is correct for std of 1 value
            X_test_row[f'roll_std_{rw}_{col}'] = np.std(effective_roll_data)

    # Populate interaction features using the newly computed lags
    # Ensure lag_1 values are from the X_test_row features that were just computed
    lag1_rsv = (
        X_test_row[f'lag_1_{TARGET_STR}'].iloc[0]
        if f'lag_1_{TARGET_STR}' in X_test_row.columns
        else 0.0
    )
    lag1_flu = (
        X_test_row[f'lag_1_Total Influenza Admissions'].iloc[0]
        if f'lag_1_Total Influenza Admissions' in X_test_row.columns
        else 0.0
    )
    lag1_covid = (
        X_test_row[f'lag_1_Total COVID-19 Admissions'].iloc[0]
        if f'lag_1_Total COVID-19 Admissions' in X_test_row.columns
        else 0.0
    )
    lag1_nssp_rsv_ed = (
        X_test_row[f'lag_1_NSSP_RSV_ED_percent'].iloc[0]
        if f'lag_1_NSSP_RSV_ED_percent' in X_test_row.columns
        else 0.0
    )  # Get lag for new interaction feature

    if 'pop_rsv_interaction_lag1' in X_test_row.columns:
      X_test_row['pop_rsv_interaction_lag1'] = (
          X_test_row['population'] * lag1_rsv
      )
    if 'rsv_flu_ratio_lag1' in X_test_row.columns:
      X_test_row['rsv_flu_ratio_lag1'] = lag1_rsv / (lag1_flu + 1e-6)
    if 'rsv_covid_ratio_lag1' in X_test_row.columns:
      X_test_row['rsv_covid_ratio_lag1'] = lag1_rsv / (lag1_covid + 1e-6)

    # --- Start of Improvement 2: Calculate new interaction feature for prediction ---
    if 'pop_NSSP_RSV_ED_interaction_lag1' in X_test_row.columns:
      X_test_row['pop_NSSP_RSV_ED_interaction_lag1'] = (
          X_test_row['population'] * lag1_nssp_rsv_ed
      )
    # --- End of Improvement 2 ---

    # Ensure all feature columns are numeric and fill any last NaNs
    X_test_row = X_test_row[feature_columns].fillna(0.0).astype(float)

    # Predict quantiles
    preds = []
    for q in QUANTILES:
      preds.append(quantile_models[q].predict(X_test_row)[0])

    # Enforce monotonicity and non-negativity
    preds = np.maximum(0.0, np.sort(preds))

    # Store predictions using the original index from test_x
    for i, q in enumerate(QUANTILES):
      test_y_hat_quantiles.loc[original_idx, f'quantile_{q}'] = preds[i]

    # Update predictions_as_lags for the next horizon's lags (using the median prediction)
    median_pred_val = preds[QUANTILES.index(0.5)]
    predictions_as_lags[(location, target_end_date)] = median_pred_val

    # Update deques for the next prediction step
    current_location_state[TARGET_STR].append(
        median_pred_val
    )  # Target is updated with its own forecast
    for col in [c for c in base_features if c != TARGET_STR]:
      # For non-target features, propagate the last known actual value forward.
      # This means if the deque is not empty, append its last element.
      # If empty (e.g., new location with no history), append 0.0.
      if not current_location_state[col]:
        current_location_state[col].append(0.0)
      else:
        current_location_state[col].append(
            current_location_state[col][-1]
        )  # Propagate last known value

    location_history_state[location] = (
        current_location_state  # Update the global state dict
    )

  # Ensure integer type and non-negativity for final output
  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(int).clip(lower=0)

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
