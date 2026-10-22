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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,g-bare-generic,reimported,missing-function-docstring,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation,unused-argument,broad-exception-caught,g-long-ternary

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
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core principles of the method:
# 1. Ensemble Forecasting: Combine predictions from three distinct models.
# 2. Model Diversification: Utilize a population-normalized autoregressive model, a covariate-assisted quantile regression model, and a regime-switching forecaster.
# 3. Probabilistic Ensemble Averaging: Integrate component forecasts by converting to Cumulative Distribution Functions (CDFs), performing linear pooling of probabilities, and inverting the averaged CDF to extract final quantiles.
# 4. Monotonicity Enforcement: Apply post-processing to ensure all predicted quantiles for any given forecast are strictly non-decreasing.

import warnings
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Set, Tuple
from scipy.interpolate import interp1d
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

# Constants for the problem (assumed to be available in the global scope)
# TARGET_STR = 'Total COVID-19 Admissions'
# QUANTILES = [...]

# Configuration Constants specific to this implementation
CDF_RESOLUTION_POINTS = (
    2000  # Increased for better CDF approximation resolution and robustness
)
AR_RESIDUAL_SAMPLING_RANDOM_STATE = (
    42  # Added for reproducibility of AR component
)

# Define common lag periods (in weeks) for feature engineering
LAG_PERIODS_TARGET = [
    1,
    2,
    3,
    4,
]  # Focused on more recent lags of the target itself
LAG_PERIODS_CASES_PROXY = [
    1,
    2,
]  # Lags for the 'case count' proxy feature (diff of admissions)


# NEW CLASS: RobustStandardScaler to handle constant features gracefully
class RobustStandardScaler:
  """A StandardScaler wrapper that handles constant features gracefully.

  It stores all features seen during fit and ensures that transform operates
  only on those features, filling missing ones with 0 and dropping new ones.
  """

  def __init__(self, epsilon = 1e-9):
    self.scaler = StandardScaler()
    self.epsilon = epsilon
    self._non_constant_features = pd.Index([])
    self._all_features = pd.Index([])  # Store all feature names seen during fit

  def fit(self, X):
    self._all_features = X.columns
    stds = X.std(axis=0)
    self._non_constant_features = X.columns[stds > self.epsilon]

    if not self._non_constant_features.empty:
      self.scaler.fit(X[self._non_constant_features])
    return self

  def transform(self, X):
    # Reindex X to ensure it has all features seen during fit.
    # Fill any features that were in self._all_features but not in X with 0.0.
    # Drop any features in X that were not in self._all_features.
    X_processed = X.reindex(columns=self._all_features, fill_value=0.0)

    # Create a copy to store scaled values to avoid SettingWithCopyWarning
    X_scaled = X_processed.copy()

    # Scale only the non_constant_features if they exist
    if not self._non_constant_features.empty:
      X_scaled[self._non_constant_features] = self.scaler.transform(
          X_processed[self._non_constant_features]
      )
    return X_scaled

  def fit_transform(self, X):
    return self.fit(X).transform(X)


def _ensure_datetime(
    df, col = 'target_end_date'
):
  """Ensures a specified column in a DataFrame is of datetime type."""
  df = df.copy()
  if pd.api.types.is_object_dtype(df[col]):
    df[col] = pd.to_datetime(df[col])
  return df


def _get_fallback_quantiles(base_value, quantiles):
  """Generates a spread of quantiles around a base value for fallback scenarios.

  Ensures non-negativity, rounding to int, and monotonicity. Simplifies the
  logic to provide a more stable and calibrated spread.
  """
  base_value = np.maximum(0, int(np.round(base_value)))

  # Define a simplified set of anchor quantiles and corresponding values/multipliers
  # These are designed to provide a reasonable spread, especially for low values.
  anchor_quantiles = np.array([
      0.00,
      0.01,
      0.025,
      0.05,
      0.1,
      0.25,
      0.5,
      0.75,
      0.9,
      0.95,
      0.975,
      0.99,
      1.00,
  ])

  # Values/multipliers for the anchor quantiles based on base_value
  if base_value == 0:
    # For base 0, ensure all quantiles are 0 but allow a small positive tail for higher quantiles
    anchor_values_raw = np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 8, 10])
  elif (
      base_value <= 5
  ):  # For very small positive values, ensure a meaningful spread
    anchor_values_raw = np.array([
        0,
        0,
        0,
        0,
        1,
        np.maximum(1, int(base_value * 0.5)),
        base_value,
        base_value + 2,
        base_value + 5,
        base_value + 8,
        base_value + 12,
        base_value + 15,
        base_value + 20,
    ])
  elif (
      base_value <= 50
  ):  # For small to medium values, proportional spread with a floor
    multipliers = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    )
    anchor_values_raw = base_value * multipliers
    # Ensure minimal absolute spread for larger quantiles if base is small
    anchor_values_raw[anchor_quantiles > 0.5] = np.maximum(
        anchor_values_raw[anchor_quantiles > 0.5],
        base_value
        + (anchor_quantiles[anchor_quantiles > 0.5] - 0.5)
        * np.maximum(5, base_value * 0.5),
    )
  else:  # For larger values, more proportional spread
    multipliers = np.array(
        [0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
    )
    anchor_values_raw = base_value * multipliers

  # Ensure non-negativity and integer type
  anchor_values = np.round(np.maximum(0, anchor_values_raw)).astype(int)
  # Ensure strict monotonicity by cumulative maximum
  anchor_values = np.maximum.accumulate(anchor_values)

  # Filter out duplicate points that would break interp1d
  # Use pandas to handle sorting and dropping duplicates cleanly
  temp_df_interp = pd.DataFrame({'q': anchor_quantiles, 'val': anchor_values})
  temp_df_interp = temp_df_interp.sort_values(by='q').drop_duplicates(
      subset='q', keep='last'
  )

  # If there are still less than 2 unique points, enforce a minimal spread
  if len(temp_df_interp) < 2:
    if base_value == 0:
      temp_df_interp = pd.DataFrame({'q': [0.0, 1.0], 'val': [0, 5]})
    else:
      temp_df_interp = pd.DataFrame(
          {'q': [0.0, 0.5, 1.0], 'val': [0, base_value, base_value + 20]}
      )
    temp_df_interp['val'] = np.maximum.accumulate(
        np.maximum(0, np.round(temp_df_interp['val']).astype(int))
    )

  f_interp = interp1d(
      temp_df_interp['q'].values,
      temp_df_interp['val'].values,
      kind='linear',
      bounds_error=False,
      fill_value=(
          temp_df_interp['val'].iloc[0],
          temp_df_interp['val'].iloc[-1],
      ),
      assume_sorted=True,
  )

  fallback_qs = f_interp(quantiles)
  fallback_qs = np.round(np.maximum(0, fallback_qs)).astype(int)
  return np.maximum.accumulate(fallback_qs)  # Final monotonicity check


def _generate_features_for_location(
    loc_id,
    train_df_loc,
    test_x_loc,
    target_str,
    lag_periods_target,
    lag_periods_cases_proxy,
):
  """Generates features for a single location, combining train and test data safely,

  preventing future data leakage for lag features.
  """

  # 1. Prepare a combined series of dates and actual observed targets
  # Combine all unique dates from train and test.
  combined_dates = pd.concat(
      [train_df_loc['target_end_date'], test_x_loc['target_end_date']]
  ).unique()
  combined_df = pd.DataFrame({'target_end_date': combined_dates})
  combined_df = combined_df.sort_values('target_end_date').reset_index(
      drop=True
  )

  # Merge actual target values from training data
  combined_df = combined_df.merge(
      train_df_loc[['target_end_date', target_str]],
      on='target_end_date',
      how='left',
  )

  # Merge population data (test_x's population values take precedence for its dates)
  pop_train = train_df_loc[['target_end_date', 'population']].drop_duplicates()
  pop_test = test_x_loc[['target_end_date', 'population']].drop_duplicates()

  all_pop = pd.concat([pop_train, pop_test]).drop_duplicates(
      subset=['target_end_date'], keep='last'
  )
  combined_df = combined_df.merge(all_pop, on='target_end_date', how='left')

  # Ensure population is robust (non-zero, ffill/bfill for any missing) and cast to int
  combined_df['population'] = np.maximum(
      1, combined_df['population'].fillna(method='ffill').fillna(method='bfill')
  ).astype(int)

  # Determine the last observed value from the training data for filling future lags
  last_observed_value = (
      combined_df[target_str].dropna().iloc[-1]
      if not combined_df[target_str].dropna().empty
      else 0
  )
  last_observed_value = np.maximum(0, last_observed_value).astype(int)

  # Create a temporary 'fill_target' column for generating lags.
  # This column contains actual historical values. For *future* (test) dates,
  # it is filled with the last known observed value from training data (persistence forecast for lags).
  combined_df['temp_fill_target'] = (
      combined_df[target_str].fillna(method='ffill').fillna(last_observed_value)
  )
  combined_df['temp_fill_target'] = np.maximum(
      0, combined_df['temp_fill_target']
  ).astype(int)

  # Set index for time-series operations
  combined_df = combined_df.set_index('target_end_date').sort_index()

  # Generate TARGET lags
  for lag in lag_periods_target:
    combined_df[f'lag_{lag}'] = combined_df['temp_fill_target'].shift(lag)

  # Re-introduce "case count" proxy features as per problem statement (Component 2)
  # Using 1-week difference in admissions as a proxy for new cases/activity change
  combined_df['admissions_1wk_diff'] = (
      combined_df['temp_fill_target'].diff(1).fillna(0)
  )  # Fill initial NaN with 0
  for lag in lag_periods_cases_proxy:
    combined_df[f'lag_cases_{lag}'] = combined_df['admissions_1wk_diff'].shift(
        lag
    )

  # --- NEW FEATURE ENGINEERING ---
  # Add week of year as a cyclical feature
  combined_df['weekofyear'] = combined_df.index.isocalendar().week.astype(int)
  combined_df['weekofyear_sin'] = np.sin(
      2 * np.pi * combined_df['weekofyear'] / 52
  )
  combined_df['weekofyear_cos'] = np.cos(
      2 * np.pi * combined_df['weekofyear'] / 52
  )

  # Add rolling mean and standard deviation of population-normalized admissions
  combined_df['normalized_admissions_for_rolling'] = (
      combined_df['temp_fill_target'] / combined_df['population']
  )

  # Use a 4-week window for rolling stats. `closed='left'` to prevent future leakage.
  combined_df['rolling_mean_norm_admissions_4wk'] = (
      combined_df['normalized_admissions_for_rolling']
      .rolling(window=4, min_periods=1, closed='left')
      .mean()
      .fillna(0)
  )
  combined_df['rolling_std_norm_admissions_4wk'] = (
      combined_df['normalized_admissions_for_rolling']
      .rolling(window=4, min_periods=1, closed='left')
      .std()
      .fillna(0)
  )

  # Drop temporary target, normalized admissions, and reset index, remove intermediate diff column
  combined_df = combined_df.drop(
      columns=[
          'temp_fill_target',
          target_str,
          'admissions_1wk_diff',
          'normalized_admissions_for_rolling',
      ]
  ).reset_index()

  # Separate into training and test features
  max_train_date = train_df_loc['target_end_date'].max()

  train_data_with_features = combined_df[
      combined_df['target_end_date'] <= max_train_date
  ].copy()
  train_data_with_features = train_data_with_features.merge(
      train_df_loc[['target_end_date', target_str]],
      on='target_end_date',
      how='left',
  )

  # Ensure `test_data_with_features` only contains rows corresponding to `test_x_loc`'s original index and dates.
  test_data_with_features = combined_df[
      combined_df['target_end_date'].isin(test_x_loc['target_end_date'])
  ].copy()
  # Map back to original test_x index
  test_data_with_features = (
      test_data_with_features.set_index('target_end_date')
      .loc[test_x_loc['target_end_date']]
      .reset_index(drop=True)
  )
  test_data_with_features.index = test_x_loc.index

  # Fill any remaining NaNs in lag features (e.g., initial lags or those pointing before first observation)
  all_lag_features = (
      [f'lag_{lag}' for lag in LAG_PERIODS_TARGET]
      + [f'lag_cases_{lag}' for lag in LAG_PERIODS_CASES_PROXY]
      + [
          'weekofyear_sin',
          'weekofyear_cos',
          'rolling_mean_norm_admissions_4wk',
          'rolling_std_norm_admissions_4wk',
      ]
  )

  for col in all_lag_features:
    # Fill NaNs with the last observed value from the training data for robustness
    # and ensure non-negativity and integer type.
    if col in [
        'weekofyear_sin',
        'weekofyear_cos',
    ]:  # These are not counts, fill with 0
      train_data_with_features[col] = train_data_with_features[col].fillna(0)
      test_data_with_features[col] = test_data_with_features[col].fillna(0)
    else:  # For count-like features, fill with last observed and ensure int
      train_data_with_features[col] = np.maximum(
          0, train_data_with_features[col].fillna(last_observed_value)
      ).astype(int)
      test_data_with_features[col] = np.maximum(
          0, test_data_with_features[col].fillna(last_observed_value)
      ).astype(int)

  return train_data_with_features, test_data_with_features, last_observed_value


def _post_process_quantiles(
    pred_df, quantiles
):
  """Ensures quantiles are non-negative, integer, and monotonically increasing."""
  quantile_cols = [f'quantile_{q}' for q in quantiles]

  # Ensure non-negativity and integer type for all quantile columns
  for col in quantile_cols:
    if col in pred_df.columns:
      pred_df[col] = np.round(
          np.maximum(0, pd.to_numeric(pred_df[col], errors='coerce').fillna(0))
      ).astype(int)

  # Apply cumulative maximum row-wise to enforce monotonicity
  # Convert to numpy for faster operation, then back to pandas
  if not pred_df.empty and all(col in pred_df.columns for col in quantile_cols):
    processed_values = pred_df[quantile_cols].values
    processed_values = np.maximum.accumulate(processed_values, axis=1)
    pred_df[quantile_cols] = processed_values

  return pred_df


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Returns quantile predictions.
  """
  # np.seterr(over='warn') is the default from the preamble, suppressing over/underflow errors.

  # 1. Initial Data Preparation
  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y
  train_df = _ensure_datetime(train_df)
  test_x_orig_index = test_x.index  # Preserve original index for final output
  test_x = _ensure_datetime(test_x).copy()  # Work on a copy of test_x

  # Ensure 'population' is numeric and handle potential zero or NaN populations
  train_df['population'] = (
      pd.to_numeric(train_df['population'], errors='coerce')
      .fillna(1)
      .replace(0, 1)
      .astype(int)
  )
  test_x['population'] = (
      pd.to_numeric(test_x['population'], errors='coerce')
      .fillna(1)
      .replace(0, 1)
      .astype(int)
  )

  # Remove rows with NaN target values for training (crucial for model fitting)
  train_df.dropna(subset=[TARGET_STR], inplace=True)
  if train_df.empty:
    warnings.warn(
        'Training data became empty after dropping NaNs. Returning zeros.',
        UserWarning,
    )
    return pd.DataFrame(
        0, index=test_x_orig_index, columns=[f'quantile_{q}' for q in QUANTILES]
    )

  # Sort data for time-series operations
  train_df = train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Determine the maximum observed value globally (for CDF scaling, as a robust baseline)
  max_observed_val_global = (
      train_df[TARGET_STR].max() if not train_df.empty else 0
  )
  max_observed_val_global = np.maximum(10, max_observed_val_global).astype(
      int
  )  # Ensure a reasonable floor for scaling

  all_locations = train_df['location'].unique()

  # Store component predictions (dictionaries mapping location_id to DataFrame of quantiles)
  component_1_preds = {}
  component_2_preds = {}
  component_3_preds = {}
  # Store latest observed value per location for consistent fallbacks
  location_latest_observed_values = {}

  # Features for Quantile Regression models (Component 2 and 3)
  qr_features = (
      [f'lag_{lag}' for lag in LAG_PERIODS_TARGET]
      + [f'lag_cases_{lag}' for lag in LAG_PERIODS_CASES_PROXY]
      + [
          'weekofyear_sin',
          'weekofyear_cos',
          'rolling_mean_norm_admissions_4wk',
          'rolling_std_norm_admissions_4wk',
      ]
  )

  for loc_id in all_locations:
    loc_train_df = train_df[train_df['location'] == loc_id].copy()
    loc_test_x = test_x[test_x['location'] == loc_id].copy()

    # Fallback if no training or test data for this location
    if loc_train_df.empty or loc_test_x.empty:
      warnings.warn(
          f'Empty training or test data for location {loc_id}. Filling with'
          ' fallback values.',
          UserWarning,
      )
      preds_df_fallback = pd.DataFrame(
          [
              _get_fallback_quantiles(0, QUANTILES)
          ],  # Use 0 as base if no train data
          index=loc_test_x.index,
          columns=[f'quantile_{q}' for q in QUANTILES],
      )
      component_1_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      component_2_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      component_3_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      location_latest_observed_values[loc_id] = 0  # No observed value
      continue

    loc_train_df = loc_train_df.sort_values(by='target_end_date').reset_index(
        drop=True
    )

    # --- Centralized Feature Engineering for this location ---
    train_data_with_features, test_data_with_features, latest_observed_value = (
        _generate_features_for_location(
            loc_id,
            loc_train_df,
            loc_test_x,
            TARGET_STR,
            LAG_PERIODS_TARGET,
            LAG_PERIODS_CASES_PROXY,
        )
    )
    location_latest_observed_values[loc_id] = latest_observed_value

    # Use a fixed 28-day window for component 1 and 2, as per instructions.
    loc_max_train_date = loc_train_df['target_end_date'].max()
    window_start_date = loc_max_train_date - pd.Timedelta(
        weeks=3
    )  # 28-day window inclusive of max_date

    loc_train_window_for_models = train_data_with_features[
        (train_data_with_features['target_end_date'] >= window_start_date)
        & (train_data_with_features['target_end_date'] <= loc_max_train_date)
    ].copy()
    # Ensure that the training window data used for fitting has no NaNs in features or target.
    loc_train_window_for_models.dropna(
        subset=qr_features + [TARGET_STR], inplace=True
    )

    loc_test_x_features = (
        test_data_with_features.copy()
    )  # Features for making predictions

    # Check for sufficient data for model fitting after lag creation and cleaning
    min_ar_series_length = (
        max(LAG_PERIODS_TARGET) + 1
    )  # Needs at least max lag + 1 points for AR
    min_qr_data_points = (
        len(qr_features) + 5
    )  # Need more points than features + 1 for meaningful QR

    if (
        len(loc_train_window_for_models) < min_ar_series_length
        or len(loc_train_window_for_models) < min_qr_data_points
    ):
      warnings.warn(
          f'Insufficient robust training data for location {loc_id} after'
          f' creating lags for AR/QR (len={len(loc_train_window_for_models)}).'
          ' Using simple fallback for all components.',
          UserWarning,
      )
      preds_df_fallback = pd.DataFrame(
          [_get_fallback_quantiles(latest_observed_value, QUANTILES)],
          index=loc_test_x.index,
          columns=[f'quantile_{q}' for q in QUANTILES],
      )
      component_1_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      component_2_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      component_3_preds[loc_id] = _post_process_quantiles(
          preds_df_fallback.copy(), QUANTILES
      )
      continue

    # --- Component 1: Population-normalized Autoregressive Model with Studentized Residual Resampling ---
    ar_forecasts_quantiles = pd.DataFrame(
        index=loc_test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    ar_model_successful = False

    ar_train_series_raw = loc_train_window_for_models[TARGET_STR]
    ar_train_population = loc_train_window_for_models['population']

    if ar_train_population.min() == 0:
      ar_train_population = ar_train_population.replace(0, 1)

    ar_train_series_normalized = (
        ar_train_series_raw / ar_train_population
    ).reset_index(drop=True)

    ar_lags_to_use = min(
        len(ar_train_series_normalized) - 1, max(LAG_PERIODS_TARGET)
    )
    ar_lags_to_use = max(
        1, ar_lags_to_use
    )  # Ensure at least 1 lag if data allows

    # NEW CHECK: Handle nearly constant AR training series
    if (
        ar_train_series_normalized.std() < 1e-8
    ):  # Lower tolerance for "nearly constant"
      warnings.warn(
          f'AR training series is nearly constant for {loc_id}. Skipping AR'
          ' model and using fallback spread.',
          UserWarning,
      )
      # ar_model_successful remains False, triggering fallback.
    elif len(ar_train_series_normalized) >= (ar_lags_to_use + 1):
      try:
        if ar_lags_to_use >= 1:
          with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore'
            )  # Suppress convergence warnings from statsmodels
            ar_model = AutoReg(
                ar_train_series_normalized,
                lags=ar_lags_to_use,
                seasonal=False,
                trend='c',
            )
            ar_results = ar_model.fit(disp=False)

            # --- IMPROVEMENT: Robustness check for AR model parameters and forecasts ---
            if (
                ar_results.params.isnull().any()
                or np.isinf(ar_results.params).any()
            ):
              warnings.warn(
                  f'AR model parameters contain NaN or Inf for {loc_id}. Using'
                  ' fallback spread for AR component.',
                  UserWarning,
              )
            else:
              residuals = ar_results.resid
              # Ensure enough residuals with sufficient variance for resampling
              # Also check if the fitted model's innovations have sufficient variance
              if (
                  len(residuals) > max(LAG_PERIODS_TARGET)
                  and np.std(residuals) > 1e-9
                  and ar_results.sigma2 > 1e-9
              ):
                num_samples = 1000

                # Generate point forecasts for the test set from the AR model
                forecast_steps = len(loc_test_x)
                ar_point_forecasts_normalized_series = ar_results.predict(
                    start=len(ar_train_series_normalized),
                    end=len(ar_train_series_normalized) + forecast_steps - 1,
                )

                # Check point forecasts for validity
                if (
                    ar_point_forecasts_normalized_series.isnull().any()
                    or np.isinf(ar_point_forecasts_normalized_series).any()
                ):
                  warnings.warn(
                      f'AR point forecasts contain NaN or Inf for {loc_id}.'
                      ' Using fallback spread for AR component.',
                      UserWarning,
                  )
                else:
                  # Align forecast series with the test_x index
                  ar_point_forecasts_normalized_series.index = loc_test_x.index

                  # Studentized residuals: divide by estimated standard deviation of the innovations
                  std_dev_innovations = np.sqrt(ar_results.sigma2)
                  studentized_residuals = residuals / std_dev_innovations

                  # Sample from studentized residuals
                  # MODIFICATION: Added random_state for reproducibility
                  sampled_studentized_errors = np.random.choice(
                      studentized_residuals,
                      size=num_samples,
                      replace=True,
                      random_state=AR_RESIDUAL_SAMPLING_RANDOM_STATE,
                  )

                  for i, (idx, row) in enumerate(loc_test_x.iterrows()):
                    point_pred_norm = ar_point_forecasts_normalized_series.loc[
                        idx
                    ]

                    # Scale sampled studentized errors by the estimated standard deviation and horizon factor
                    horizon_scaling_factor = np.sqrt(row['horizon'] + 1)
                    scaled_errors = (
                        sampled_studentized_errors
                        * std_dev_innovations
                        * horizon_scaling_factor
                    )
                    simulated_normalized = point_pred_norm + scaled_errors

                    simulated_admissions = (
                        simulated_normalized * row['population']
                    )
                    ar_forecasts_quantiles.loc[idx] = np.quantile(
                        simulated_admissions, QUANTILES
                    )
                  ar_model_successful = True
              else:
                warnings.warn(
                    'AR innovations too small, residuals too few, or sigma2'
                    f' near zero for {loc_id}. Using fallback spread for AR'
                    ' component.',
                    UserWarning,
                )
        else:
          warnings.warn(
              'AR lags could not be determined or were less than 1 for'
              f' {loc_id} (series too short). Skipping AR model.',
              UserWarning,
          )
      except Exception as e:
        warnings.warn(
            f'Component 1 AR model failed for {loc_id}: {e}. Falling back to a'
            ' simple spread for AR component.',
            UserWarning,
        )
    else:
      warnings.warn(
          f'Insufficient training data for AR model for {loc_id} after'
          ' considering lags. Skipping AR model.',
          UserWarning,
      )

    if not ar_model_successful:
      for idx in loc_test_x.index:
        ar_forecasts_quantiles.loc[idx] = _get_fallback_quantiles(
            latest_observed_value, QUANTILES
        )

    component_1_preds[loc_id] = _post_process_quantiles(
        ar_forecasts_quantiles, QUANTILES
    )

    # --- Component 2: Covariate-assisted Quantile Regression Model ---
    qr_train_data = loc_train_window_for_models.copy()

    qr_forecasts_quantiles = pd.DataFrame(
        index=loc_test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    qr_model_successful_overall = True

    if (
        len(qr_train_data) >= min_qr_data_points
        and not loc_test_x_features.empty
    ):
      X_train_qr = qr_train_data[qr_features]
      y_train_qr = qr_train_data[TARGET_STR]

      # Additional check for QR: ensure sufficient unique values in target for meaningful regression
      if y_train_qr.nunique() < 2:
        warnings.warn(
            f'QR training target has fewer than 2 unique values for {loc_id}.'
            ' Falling back for Component 2.',
            UserWarning,
        )
        qr_model_successful_overall = False
      else:
        qr_scaler = RobustStandardScaler()
        X_train_qr_scaled = qr_scaler.fit_transform(X_train_qr)
        X_test_qr_scaled = qr_scaler.transform(
            loc_test_x_features[qr_features]
        )  # Ensure features match

        if (
            X_train_qr_scaled.shape[0] >= min_qr_data_points
            and y_train_qr.shape[0] >= min_qr_data_points
        ):
          for q in QUANTILES:
            try:
              # Setting max_iter and tol for robustness in highs solver
              qr_model = QuantileRegressor(
                  quantile=q,
                  alpha=0.01,
                  solver='highs',
                  verbose=0,
                  max_iter=5000,
                  tol=1e-4,
              )
              qr_model.fit(X_train_qr_scaled, y_train_qr)
              qr_preds = qr_model.predict(X_test_qr_scaled)
              qr_forecasts_quantiles[f'quantile_{q}'] = qr_preds
            except Exception as e:
              warnings.warn(
                  f'Component 2 QR model for quantile {q} failed for {loc_id}:'
                  f' {e}.',
                  UserWarning,
              )
              qr_model_successful_overall = False
              break  # Break out of quantile loop if one model fails for this component/location
        else:
          warnings.warn(
              f'Insufficient robust QR training data for {loc_id}. Falling back'
              ' for Component 2.',
              UserWarning,
          )
          qr_model_successful_overall = False
    else:
      warnings.warn(
          f'QR models failed or had insufficient data for {loc_id}. Falling'
          ' back for Component 2.',
          UserWarning,
      )
      qr_model_successful_overall = False

    if not qr_model_successful_overall:
      # Fallback to general robust spread
      for idx in loc_test_x.index:
        qr_forecasts_quantiles.loc[idx] = _get_fallback_quantiles(
            latest_observed_value, QUANTILES
        )

    component_2_preds[loc_id] = _post_process_quantiles(
        qr_forecasts_quantiles, QUANTILES
    )

    # --- Component 3: Regime-Switching Forecaster based on Epidemic Direction ---
    loc_train_data_for_regimes = (
        loc_train_window_for_models.copy()
    )  # Use the same 28-day window for consistency

    trend_smoothing_window = 3  # Use a 3-week rolling average for trend
    loc_train_data_for_regimes['raw_trend_val'] = loc_train_data_for_regimes[
        TARGET_STR
    ].diff(periods=1)
    loc_train_data_for_regimes['trend_val'] = (
        loc_train_data_for_regimes['raw_trend_val']
        .rolling(window=trend_smoothing_window, min_periods=1)
        .mean()
        .fillna(0)
    )

    # Improvement: Base dynamic_regime_threshold on median of recent target values
    recent_target_median = loc_train_data_for_regimes[TARGET_STR].median()
    dynamic_regime_threshold = np.maximum(
        5, int(recent_target_median * 0.1) + 1
    )  # Floor of 5, or 10% of median + 1
    dynamic_regime_threshold = np.maximum(
        dynamic_regime_threshold, 20
    )  # Ensure a reasonable absolute minimum for threshold.

    loc_train_data_for_regimes['regime'] = 'steady'
    loc_train_data_for_regimes.loc[
        loc_train_data_for_regimes['trend_val'] > dynamic_regime_threshold,
        'regime',
    ] = 'up'
    loc_train_data_for_regimes.loc[
        loc_train_data_for_regimes['trend_val'] < -dynamic_regime_threshold,
        'regime',
    ] = 'down'

    regime_models: Dict[str, Dict[float, QuantileRegressor]] = {
        'up': {},
        'down': {},
        'steady': {},
    }
    regime_scalers: Dict[str, RobustStandardScaler] = {
        'up': None,
        'down': None,
        'steady': None,
    }
    regime_model_status: Dict[str, bool] = {
        'up': False,
        'down': False,
        'steady': False,
    }

    # Train regime-specific QR models
    for regime_type in ['up', 'down', 'steady']:
      regime_train_data = loc_train_data_for_regimes[
          loc_train_data_for_regimes['regime'] == regime_type
      ].dropna(subset=qr_features + [TARGET_STR])

      if (
          len(regime_train_data) >= min_qr_data_points
          and regime_train_data[TARGET_STR].nunique() >= 2
      ):
        X_train_regime_qr = regime_train_data[qr_features]
        y_train_regime_qr = regime_train_data[TARGET_STR]

        scaler = RobustStandardScaler()
        X_train_regime_qr_scaled = scaler.fit_transform(X_train_regime_qr)
        regime_scalers[regime_type] = scaler

        all_regime_quantiles_successful = True
        for q in QUANTILES:
          try:
            qr_model = QuantileRegressor(
                quantile=q,
                alpha=0.01,
                solver='highs',
                verbose=0,
                max_iter=5000,
                tol=1e-4,
            )
            qr_model.fit(X_train_regime_qr_scaled, y_train_regime_qr)
            regime_models[regime_type][q] = qr_model
          except Exception as e:
            warnings.warn(
                f'Regime QR model for {regime_type}, quantile {q} failed for'
                f' {loc_id}: {e}.',
                UserWarning,
            )
            all_regime_quantiles_successful = False
            break  # Break if one quantile model fails for this regime
        regime_model_status[regime_type] = all_regime_quantiles_successful
      else:
        warnings.warn(
            f'Insufficient robust training data for {regime_type} regime for'
            f' {loc_id} (using 28-day window). Component 3 will fallback for'
            ' this regime.',
            UserWarning,
        )

    cr_forecasts_quantiles = pd.DataFrame(
        index=loc_test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    cr_model_successful_overall = True

    # Determine the current regime for forecasting based on the last smoothed trend from the training window
    if (
        not loc_train_data_for_regimes.empty
        and 'trend_val' in loc_train_data_for_regimes.columns
    ):
      last_smoothed_trend_val = loc_train_data_for_regimes['trend_val'].iloc[-1]
    else:
      last_smoothed_trend_val = 0

    if last_smoothed_trend_val > dynamic_regime_threshold:
      forecast_regime = 'up'
    elif last_smoothed_trend_val < -dynamic_regime_threshold:
      forecast_regime = 'down'
    else:
      forecast_regime = 'steady'

    # Make predictions using the chosen regime's model and its corresponding scaler
    current_regime_scaler = regime_scalers.get(forecast_regime)
    if (
        regime_model_status[forecast_regime]
        and current_regime_scaler is not None
    ):
      X_test_qr = loc_test_x_features[qr_features]
      X_test_qr_scaled = current_regime_scaler.transform(X_test_qr)

      for q in QUANTILES:
        try:
          cr_preds = regime_models[forecast_regime][q].predict(X_test_qr_scaled)
          cr_forecasts_quantiles[f'quantile_{q}'] = cr_preds
        except Exception as e:
          warnings.warn(
              f'Failed to apply regime model for {forecast_regime}, quantile'
              f' {q} for {loc_id}: {e}.',
              UserWarning,
          )
          cr_model_successful_overall = False
          break  # Break if one quantile model fails for this regime
    else:
      warnings.warn(
          f'Regime models for {forecast_regime} for {loc_id} are'
          ' incomplete/failed or no scaler. Falling back for Component 3.',
          UserWarning,
      )
      cr_model_successful_overall = False

    if not cr_model_successful_overall:
      # Fallback to general robust spread
      for idx in loc_test_x.index:
        cr_forecasts_quantiles.loc[idx] = _get_fallback_quantiles(
            latest_observed_value, QUANTILES
        )

    component_3_preds[loc_id] = _post_process_quantiles(
        cr_forecasts_quantiles, QUANTILES
    )

  # --- 4 & 5: Ensemble Integration & Probabilistic Averaging ---
  final_ensemble_quantiles = pd.DataFrame(
      index=test_x_orig_index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  for idx, row in test_x.iterrows():
    loc_id = row['location']

    preds_comp1 = (
        component_1_preds.get(loc_id, pd.DataFrame()).loc[idx]
        if loc_id in component_1_preds
        and idx in component_1_preds[loc_id].index
        else None
    )
    preds_comp2 = (
        component_2_preds.get(loc_id, pd.DataFrame()).loc[idx]
        if loc_id in component_2_preds
        and idx in component_2_preds[loc_id].index
        else None
    )
    preds_comp3 = (
        component_3_preds.get(loc_id, pd.DataFrame()).loc[idx]
        if loc_id in component_3_preds
        and idx in component_3_preds[loc_id].index
        else None
    )

    valid_component_preds_values = []
    if preds_comp1 is not None and not preds_comp1.isnull().all():
      valid_component_preds_values.append(preds_comp1.values)
    if preds_comp2 is not None and not preds_comp2.isnull().all():
      valid_component_preds_values.append(preds_comp2.values)
    if preds_comp3 is not None and not preds_comp3.isnull().all():
      valid_component_preds_values.append(preds_comp3.values)

    current_latest_observed_val = location_latest_observed_values.get(loc_id, 0)
    current_latest_observed_val = np.maximum(
        0, current_latest_observed_val
    ).astype(int)

    if not valid_component_preds_values:
      warnings.warn(
          f'All components failed for forecast point {idx}. Using robust'
          ' fallback spread.',
          UserWarning,
      )
      final_ensemble_quantiles.loc[idx] = _get_fallback_quantiles(
          current_latest_observed_val, QUANTILES
      )
      continue

    # --- Calculate a more robust current_max_val for the CDF grid ---
    max_val_across_all_preds_for_this_row = 0
    if valid_component_preds_values:
      max_val_across_all_preds_for_this_row = np.max([
          np.max(p_vals)
          for p_vals in valid_component_preds_values
          if len(p_vals) > 0
      ])

    # Determine the maximum value for the CDF grid dynamically.
    # A buffer that scales with the magnitude of predictions, with a floor
    dynamic_buffer = np.maximum(
        100, int(max_val_across_all_preds_for_this_row * 0.5)
    )  # Increased buffer factor to 0.5

    current_max_val_for_grid = np.maximum.reduce([
        max_observed_val_global * 2,  # A multiple of global max observed
        max_val_across_all_preds_for_this_row
        + dynamic_buffer,  # Highest prediction plus dynamic buffer
        current_latest_observed_val
        * 3,  # Latest observed value scaled up (more aggressive)
        500,  # Ensure a reasonable floor for the grid max
    ]).astype(int)

    # Ensure current_max_val_for_grid is strictly positive
    current_max_val_for_grid = np.maximum(1, current_max_val_for_grid)

    x_cdf_grid = np.linspace(0, current_max_val_for_grid, CDF_RESOLUTION_POINTS)

    avg_cdf_values_on_grid = np.zeros_like(x_cdf_grid, dtype=float)
    num_valid_components = 0
    for q_preds_values in valid_component_preds_values:
      q_preds_values_sorted = np.array(q_preds_values)

      # Combine prediction points with 0 and current_max_val_for_grid for robust interpolation.
      x_interp_points_raw = np.concatenate(
          ([0.0], q_preds_values_sorted, [current_max_val_for_grid + 1.0])
      )
      y_interp_points_raw = np.concatenate(([0.0], QUANTILES, [1.0]))

      # Create DataFrame, sort, and drop duplicates to ensure strict monotonicity for x and y
      temp_df = pd.DataFrame(
          {'x': x_interp_points_raw, 'y': y_interp_points_raw}
      )
      temp_df = temp_df.sort_values(by='x').drop_duplicates(
          subset='x', keep='last'
      )  # Keep 'last' to represent highest quantile for a given value

      x_for_interp = temp_df['x'].values
      y_for_interp = temp_df['y'].values

      # Final check for monotonicity of y values if x values are strictly increasing but y isn't
      y_for_interp = np.maximum.accumulate(y_for_interp)

      # Ensure at least 2 unique points for interpolation
      if len(x_for_interp) < 2:
        warnings.warn(
            'Too few unique points for CDF interpolation after processing for'
            f' {loc_id}, point {idx}. Skipping component.',
            UserWarning,
        )
        continue  # Skip this component's contribution to the average CDF

      cdf_func = interp1d(
          x_for_interp,
          y_for_interp,
          kind='linear',
          bounds_error=False,
          fill_value=(0, 1),
          assume_sorted=True,
      )
      avg_cdf_values_on_grid += cdf_func(x_cdf_grid)
      num_valid_components += 1

    if num_valid_components > 0:
      avg_cdf_values_on_grid /= num_valid_components
    else:
      warnings.warn(
          'No valid components for ensemble averaging for forecast point'
          f' {idx}. Using fallback.',
          UserWarning,
      )
      final_ensemble_quantiles.loc[idx] = _get_fallback_quantiles(
          current_latest_observed_val, QUANTILES
      )
      continue

    # Ensure averaged CDF values are unique and monotonically increasing for inverse interpolation
    # Rounding avg_cdf values to handle precision issues before taking unique values
    avg_cdf_values_unique, unique_indices = np.unique(
        avg_cdf_values_on_grid.round(8), return_index=True
    )
    x_cdf_grid_unique = x_cdf_grid[unique_indices]

    # Final safety for interp1d: ensure the points for inverse CDF are strictly monotonic on both axes.
    temp_inv_df = (
        pd.DataFrame({'y': avg_cdf_values_unique, 'x': x_cdf_grid_unique})
        .sort_values('y')
        .drop_duplicates(subset='y', keep='last')
    )

    if len(temp_inv_df['y']) < 2:
      warnings.warn(
          'Insufficient unique CDF points after averaging for inverse CDF for'
          f' forecast point {idx}. Using fallback.',
          UserWarning,
      )
      final_ensemble_quantiles.loc[idx] = _get_fallback_quantiles(
          current_latest_observed_val, QUANTILES
      )
      continue

    inverse_cdf_func = interp1d(
        temp_inv_df['y'],
        temp_inv_df['x'],
        kind='linear',
        bounds_error=False,
        fill_value=(temp_inv_df['x'].iloc[0], temp_inv_df['x'].iloc[-1]),
        assume_sorted=True,
    )

    final_qs = inverse_cdf_func(QUANTILES)
    final_ensemble_quantiles.loc[idx] = final_qs

  test_y_hat_quantiles = _post_process_quantiles(
      final_ensemble_quantiles, QUANTILES
  )
  test_y_hat_quantiles.index = test_x_orig_index

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
