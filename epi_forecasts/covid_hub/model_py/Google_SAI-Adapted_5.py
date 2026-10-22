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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-short-docstring-punctuation,g-doc-return-or-yield,g-doc-args,broad-exception-caught

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
MODEL_NAME = 'Google_SAI-Adapted_5'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles:
# 1. Multi-Method Ensemble: Aggregates forecasts from SARIMAX (seasonal ARIMA), ETS, Linear Regression, and Random Forest.
# 2. Diverse Data Streams / Feature Engineering: The problem statement mandates ingesting and cleaning historical time series
#    from three distinct sources (Clinical Laboratory data, Public Health Laboratory data, and Hospitalization census).
#    However, the provided `train_x` and `train_y` input data *only contain* the 'Total COVID-19 Admissions' target series
#    along with metadata. To adhere to the spirit of "multiple surveillance data streams" and "feature construction" despite
#    this data limitation, the model performs extensive feature engineering. It creates a comprehensive set of predictive
#    signals (features) *derived from this one available target time series*. These features include current values, multiple
#    lagged observations (e.g., y_t-1, y_t-52), rolling means of the target, various time-based components (weekofyear,
#    month, year, dayofyear, quarter, sin_weekofyear, cos_weekofyear), log-transformed population, and new non-linear/
#    interaction terms like 'lag_1_squared' and 'lagged_diff_52_1'. The target variable is log1p-transformed for modeling
#    to handle skewness and stabilize variance.
# 3. Time-Series & ML Models: Explicitly employs SARIMAX and ETS for time series analysis, and Linear Regression and
#    Random Forest for machine learning regression.
# 4. Quantile Aggregation & Recursive Forecasting: Derives quantiles from the combined ensemble distribution of predictions
#    by collecting samples from each successful model's output for each forecast point. The median prediction from this
#    ensemble is then used recursively as a proxy for future known values in subsequent feature generation.

from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

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

N_SAMPLES = 5000  # Increased for more stable quantile estimation
MIN_STD_EPSILON = 1e-6
MAX_LOG_VALUE = 16.0  # Arbitrary upper bound to prevent log(very_large_number) issues or numerical instability
LAGS = [1, 2, 3, 4, 8, 52]
ROLLING_WINDOWS = [2, 4, 8]

# Dynamically generate FEATURE_COLS_ML to ensure consistency with LAGS and ROLLING_WINDOWS.
_LAG_COLS = [f'lag_{l}' for l in LAGS]
_ROLLING_MEAN_COLS = [f'rolling_mean_{w}' for w in ROLLING_WINDOWS]
_TIME_POP_COLS = [
    'log_population',
    'weekofyear',
    'month',
    'year',
    'dayofyear',
    'quarter',
    'sin_weekofyear',
    'cos_weekofyear',
]

# Use a temporary set to ensure uniqueness of interaction terms
_INTERACTION_COLS_SET = set()
# Ensure lag_1 is present before creating lag_1 dependent interactions
if 1 in LAGS:
  _INTERACTION_COLS_SET.update([
      'lag_1_log_pop_interaction',
      'lag_1_weekofyear_interaction',
      'lag_1_month_interaction',
      'lag_1_sin_week_interaction',
      'lag_1_squared',
  ])
  if 2 in LAGS:
    _INTERACTION_COLS_SET.add('lagged_diff_1_2')
  if 52 in LAGS:
    _INTERACTION_COLS_SET.update([
        'lagged_diff_1_52',
        'lagged_diff_52_1',
        'lag_1_lag_52_interaction',
    ])
_INTERACTION_COLS = sorted(
    list(_INTERACTION_COLS_SET)
)  # Sort for consistent column order

# Combine all feature categories into the final list and sort for consistency.
FEATURE_COLS_ML = sorted(
    list(
        set(_LAG_COLS + _ROLLING_MEAN_COLS + _TIME_POP_COLS + _INTERACTION_COLS)
    )
)


def _generate_ml_features(
    df_to_featurize,
    historical_log_target_series,
    is_training = False,
):
  """Generates ML features for a given DataFrame using vectorized operations (training)

  or an optimized single-point lookup (prediction).

  Args:
      df_to_featurize: A DataFrame with 'target_end_date' and 'population' for
        the specific rows/dates for which features need to be generated.
        'target_end_date' will be set as the index temporarily. For prediction,
        this DataFrame should contain only one row.
      historical_log_target_series: A pandas Series (indexed by target_end_date)
        containing log1p-transformed historical actual target values and/or
        previous median predictions. This series is assumed to be sorted by
        index. For prediction, this series should be strictly up to
        `target_end_date - 1 week`.
      is_training: If True, NaNs for initial missing lagged/rolling features are
        retained (to be dropped later by the caller during training). If False
        (for prediction), features are generated for a single target date, using
        the provided `historical_log_target_series` which serves as the "known
        past."

  Returns:
      A DataFrame with the generated ML features, with 'target_end_date' as
      index.
      For prediction, it will contain a single row of features.
  """
  df_features = df_to_featurize.set_index('target_end_date').copy()

  # Time-based features
  df_features['weekofyear'] = df_features.index.isocalendar().week.astype(int)
  df_features['month'] = df_features.index.month
  df_features['year'] = df_features.index.year
  df_features['dayofyear'] = df_features.index.dayofyear
  df_features['quarter'] = df_features.index.quarter

  # Sin/Cos transform for weekofyear to capture seasonality smoothly
  df_features['sin_weekofyear'] = np.sin(
      2 * np.pi * df_features['weekofyear'] / 52.0
  )
  df_features['cos_weekofyear'] = np.cos(
      2 * np.pi * df_features['weekofyear'] / 52.0
  )

  # Log-transformed population (assuming population is already numeric and non-zero)
  df_features['log_population'] = np.log1p(df_features['population'])

  # --- Optimized Lag and Rolling Mean Calculation ---
  if is_training:
    # Training path: Uses the full historical_log_target_series for vectorized operations
    # Ensure the full index includes all dates from historical series and df_features
    full_index_for_lags = pd.Index(
        np.sort(
            historical_log_target_series.index.union(df_features.index).unique()
        )
    )
    unified_target_log_series_source = historical_log_target_series.reindex(
        full_index_for_lags
    )

    # Lagged features using vectorized shift
    for lag in LAGS:
      df_features[f'lag_{lag}'] = unified_target_log_series_source.shift(
          lag
      ).loc[df_features.index]

    # Rolling means using vectorized rolling().mean() with closed='left' for no look-ahead
    for window in ROLLING_WINDOWS:
      # Ensure rolling mean uses the same unified_target_log_series_source
      rolling_mean_series = unified_target_log_series_source.rolling(
          window=window, min_periods=1, closed='left'
      ).mean()
      df_features[f'rolling_mean_{window}'] = rolling_mean_series.loc[
          df_features.index
      ]
  else:
    # Prediction path (single point recursive): Optimized for performance and robustness
    target_date = df_features.index[0]

    # The historical_log_target_series should already contain all actuals and prior median
    # predictions up to (target_date - 1 week). We use this directly for lags.

    # Lagged features: directly query the history series for specific lag dates
    for lag in LAGS:
      lag_date = target_date - pd.Timedelta(weeks=lag)
      # Use .get() with 0.0 as default if the lag_date is not in history_series_log
      # This handles cases where history is too short for a particular lag.
      df_features.loc[target_date, f'lag_{lag}'] = (
          historical_log_target_series.get(lag_date, 0.0)
      )

    # Rolling Mean Calculation for Prediction:
    if not historical_log_target_series.empty:
      for window in ROLLING_WINDOWS:
        # Get the last `window` values from the *actual available history*.
        # `iloc[-window:]` correctly handles cases where len(history) < window.
        rolling_values = historical_log_target_series.iloc[-window:].values
        # Ensure np.mean is not called on empty array for older numpy versions without keepdims=False
        rolling_mean_val = (
            np.mean(rolling_values) if rolling_values.size > 0 else 0.0
        )
        if pd.isna(
            rolling_mean_val
        ):  # Safeguard against NaN if rolling_values were all NaNs
          rolling_mean_val = 0.0
        df_features.loc[target_date, f'rolling_mean_{window}'] = (
            rolling_mean_val
        )
    else:
      # If history is empty, all rolling means are 0.0
      for window in ROLLING_WINDOWS:
        df_features.loc[target_date, f'rolling_mean_{window}'] = 0.0

    # Ensure any basic lagged/rolling features that might still be NaN are filled with 0.0 for prediction.
    # This is a safeguard if the above logic missed any scenario.
    for col in _LAG_COLS + _ROLLING_MEAN_COLS:
      if col not in df_features.columns or pd.isna(
          df_features.loc[target_date, col]
      ):
        df_features.loc[target_date, col] = 0.0

  # Interaction terms, using robust retrieval of lagged/time-based features.
  # Use .get() with a default Series of 0.0 (or 0 for ints) to ensure the column exists
  # and operations are robust even if base features are missing due to LAGS configuration.
  lag_1_series = df_features.get(
      'lag_1', pd.Series(0.0, index=df_features.index)
  )
  lag_2_series = df_features.get(
      'lag_2', pd.Series(0.0, index=df_features.index)
  )
  lag_52_series = df_features.get(
      'lag_52', pd.Series(0.0, index=df_features.index)
  )

  log_pop_series = df_features.get(
      'log_population', pd.Series(0.0, index=df_features.index)
  )
  weekofyear_series = df_features.get(
      'weekofyear', pd.Series(0, index=df_features.index, dtype=int)
  )
  month_series = df_features.get(
      'month', pd.Series(0, index=df_features.index, dtype=int)
  )
  sin_week_series = df_features.get(
      'sin_weekofyear', pd.Series(0.0, index=df_features.index)
  )

  # Conditionally create interaction terms only if they are defined in _INTERACTION_COLS
  if 'lagged_diff_1_2' in _INTERACTION_COLS:
    df_features['lagged_diff_1_2'] = lag_1_series - lag_2_series
  if 'lag_1_log_pop_interaction' in _INTERACTION_COLS:
    df_features['lag_1_log_pop_interaction'] = lag_1_series * log_pop_series
  if 'lag_1_weekofyear_interaction' in _INTERACTION_COLS:
    df_features['lag_1_weekofyear_interaction'] = (
        lag_1_series * weekofyear_series
    )
  if 'lag_1_month_interaction' in _INTERACTION_COLS:
    df_features['lag_1_month_interaction'] = lag_1_series * month_series
  if 'lag_1_sin_week_interaction' in _INTERACTION_COLS:
    df_features['lag_1_sin_week_interaction'] = lag_1_series * sin_week_series
  if 'lagged_diff_1_52' in _INTERACTION_COLS:
    df_features['lagged_diff_1_52'] = lag_1_series - lag_52_series
  if 'lagged_diff_52_1' in _INTERACTION_COLS:
    df_features['lagged_diff_52_1'] = lag_52_series - lag_1_series
  if 'lag_1_lag_52_interaction' in _INTERACTION_COLS:
    df_features['lag_1_lag_52_interaction'] = lag_1_series * lag_52_series
  if 'lag_1_squared' in _INTERACTION_COLS:
    df_features['lag_1_squared'] = lag_1_series**2

  # Final pass to ensure all defined FEATURE_COLS_ML are present and clean.
  # This is crucial for consistency between training and prediction feature sets.
  for col in FEATURE_COLS_ML:
    if col not in df_features.columns:
      df_features[col] = 0.0  # Add missing feature columns with default 0.0
    # For prediction (is_training=False), fill any remaining NaNs with 0.0.
    elif df_features[col].isnull().any() and not is_training:
      df_features[col] = df_features[col].fillna(0.0)

  return df_features[FEATURE_COLS_ML].copy()


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """
  # Initialize a local random number generator for reproducible sampling within this function call
  rng = np.random.default_rng(42)

  # Suppress specific warnings from statsmodels and numpy to keep output clean,
  # as instructed to configure models to be as quiet as possible.
  warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
  warnings.filterwarnings(
      'ignore', category=FutureWarning, module='statsmodels'
  )
  warnings.filterwarnings(
      'ignore', category=RuntimeWarning, module='statsmodels'
  )
  # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)   ## Manually changed
  warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

  # --- Centralized Date Conversions ---
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  # Combine train_x and train_y
  train_df = train_x.copy()
  train_df[TARGET_STR] = train_y

  all_final_predictions = []
  unique_locations_train = train_df['location'].unique()
  unique_locations_test = test_x['location'].unique()

  # --- Pre-compute Location Populations ---
  # Ensure population is numeric and non-zero. Fillna(1.0) for safety against missing or zero populations.
  location_populations_map = locations.set_index('location')['population']
  location_populations_map = (
      pd.to_numeric(location_populations_map, errors='coerce')
      .fillna(1.0)
      .replace(0, 1.0)
  )

  # --- Global ML Training Data Preparation ---
  # Store processed (features and target, with NaNs dropped) per-location data
  loc_processed_ml_data = {}

  for loc_id in unique_locations_train:
    loc_train_df = (
        train_df[train_df['location'] == loc_id]
        .sort_values('target_end_date')
        .copy()
    )

    # Assign population for this location
    current_loc_population = location_populations_map.get(loc_id, 1.0)
    loc_train_df['population'] = current_loc_population

    # Prepare historical target series for ML models (log1p transformed)
    loc_train_df[f'log1p_{TARGET_STR}'] = np.log1p(loc_train_df[TARGET_STR])

    # Initialize the history series (actuals for training data)
    history_series_log_for_features = loc_train_df.set_index('target_end_date')[
        f'log1p_{TARGET_STR}'
    ].dropna()

    # Generate features for the training period for this location
    X_train_ml_loc_df = _generate_ml_features(
        loc_train_df[['target_end_date', 'population']],
        history_series_log_for_features,
        is_training=True,
    )

    y_train_ml_log_loc_series = loc_train_df.set_index('target_end_date').loc[
        X_train_ml_loc_df.index, f'log1p_{TARGET_STR}'
    ]

    # Drop rows with NaNs from incomplete lags/rolling means at the start of the series.
    # This aligns X and y, and prepares clean data for global ML model training.
    combined_ml_train_loc_df = X_train_ml_loc_df.join(
        y_train_ml_log_loc_series
    ).dropna()

    if not combined_ml_train_loc_df.empty:
      loc_processed_ml_data[loc_id] = {
          'X': combined_ml_train_loc_df[FEATURE_COLS_ML],
          'y': combined_ml_train_loc_df[f'log1p_{TARGET_STR}'],
      }

  # Aggregate global ML training data from processed location data
  all_loc_features_list = [data['X'] for data in loc_processed_ml_data.values()]
  all_loc_targets_list = [data['y'] for data in loc_processed_ml_data.values()]

  # Initialize ML models and scaler to None, and residual std to min value
  lr_model, rf_model = None, None
  scaler = None
  global_lr_residuals_std = MIN_STD_EPSILON

  if (
      all_loc_features_list and all_loc_targets_list
  ):  # If there's any data to train global models
    X_train_ml_global = pd.concat(all_loc_features_list)
    y_train_ml_log_global = pd.concat(all_loc_targets_list)

    # --- Global ML Model Training ---
    try:
      # Fit scaler first
      scaler = StandardScaler()
      # Ensure X_train_ml_global is not empty before scaling
      if not X_train_ml_global.empty:
        X_train_scaled_global = scaler.fit_transform(X_train_ml_global)
      else:
        scaler = None  # No data to scale
        X_train_scaled_global = None
    except Exception:
      scaler = None  # If scaler fitting fails, it means no scaling can be done
      X_train_scaled_global = None  # Fallback

    # Train Linear Regression (requires scaling)
    if (
        lr_model is None
        and scaler is not None
        and X_train_scaled_global is not None
        and X_train_scaled_global.shape[0] > 0
    ):
      try:
        lr_model = LinearRegression(n_jobs=-1).fit(
            X_train_scaled_global, y_train_ml_log_global
        )
        # Calculate LR residuals std (on log scale) from training for sampling.
        lr_residuals = y_train_ml_log_global - lr_model.predict(
            X_train_scaled_global
        )
        lr_residuals_std = np.std(lr_residuals)
        global_lr_residuals_std = max(
            MIN_STD_EPSILON,
            lr_residuals_std if pd.notna(lr_residuals_std) else MIN_STD_EPSILON,
        )
      except Exception:
        lr_model = None  # LR model training failed
        global_lr_residuals_std = MIN_STD_EPSILON  # Fallback to minimum epsilon

    # Train Random Forest (can use scaled features for consistency, though not strictly required)
    # Using scaled features for RF aligns with consistent preprocessing for ML components
    rf_train_features = (
        X_train_scaled_global
        if X_train_scaled_global is not None and scaler is not None
        else X_train_ml_global.values
    )
    if (
        rf_model is None
        and rf_train_features is not None
        and rf_train_features.shape[0] > 0
    ):
      try:
        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, verbose=0, n_jobs=-1
        ).fit(rf_train_features, y_train_ml_log_global)
      except Exception:
        rf_model = None

  # --- Per-Location Forecasting for all locations in test_x ---
  for loc_id in unique_locations_test:
    loc_train_df = (
        train_df[train_df['location'] == loc_id]
        .sort_values('target_end_date')
        .copy()
    )
    loc_test_x = (
        test_x[test_x['location'] == loc_id].sort_values('horizon').copy()
    )

    # Get current location's population
    current_loc_population = location_populations_map.get(loc_id, 1.0)

    # Fallback: predict 0s for locations with no training data at all
    if loc_train_df.empty:
      pred_quantiles = pd.DataFrame(
          0,
          index=loc_test_x.index,
          columns=[f'quantile_{q}' for q in QUANTILES],
      )
      all_final_predictions.append(pred_quantiles)
      continue

    # Prepare historical target series for TS models and recursive ML feature generation
    loc_train_df['population'] = current_loc_population
    loc_train_df[f'log1p_{TARGET_STR}'] = np.log1p(loc_train_df[TARGET_STR])
    ts_train_series_log = loc_train_df.set_index('target_end_date')[
        f'log1p_{TARGET_STR}'
    ].dropna()

    # Minimum data length should be at least 52 for seasonal models,
    # or the largest lag/rolling window if those are larger, plus 1 for the target.
    max_lag_or_window = 0
    if LAGS:
      max_lag_or_window = max(max_lag_or_window, max(LAGS))
    if ROLLING_WINDOWS:
      max_lag_or_window = max(max_lag_or_window, max(ROLLING_WINDOWS))
    min_data_len_for_ts = max(
        52, max_lag_or_window + 1
    )  # Ensure enough data for lags/rolling and seasonality

    # Fallback if insufficient historical data for meaningful modeling
    if (
        ts_train_series_log.empty
        or len(ts_train_series_log) < min_data_len_for_ts
    ):
      pred_quantiles = pd.DataFrame(
          0,
          index=loc_test_x.index,
          columns=[f'quantile_{q}' for q in QUANTILES],
      )
      all_final_predictions.append(pred_quantiles)
      continue

    # Calculate a robust historical standard deviation of the log-transformed target for fallbacks.
    historical_log_std_loc = ts_train_series_log.std()
    historical_log_std_loc = max(
        MIN_STD_EPSILON,
        historical_log_std_loc
        if pd.notna(historical_log_std_loc)
        else MIN_STD_EPSILON,
    )

    # --- Location-Specific LR Residuals Standard Deviation Calculation ---
    loc_lr_residuals_std_base = global_lr_residuals_std  # Default to global
    if (
        lr_model
        and scaler
        and loc_id in loc_processed_ml_data
        and not loc_processed_ml_data[loc_id]['X'].empty
    ):
      try:
        X_train_ml_loc_for_resids = loc_processed_ml_data[loc_id]['X']
        y_train_ml_log_loc_for_resids = loc_processed_ml_data[loc_id]['y']

        if not X_train_ml_loc_for_resids.empty:
          X_train_scaled_loc_for_resids = scaler.transform(
              X_train_ml_loc_for_resids
          )
          lr_loc_residuals = y_train_ml_log_loc_for_resids - lr_model.predict(
              X_train_scaled_loc_for_resids
          )
          loc_lr_residuals_std_calculated = np.std(lr_loc_residuals)
          loc_lr_residuals_std_base = max(
              MIN_STD_EPSILON,
              loc_lr_residuals_std_calculated
              if pd.notna(loc_lr_residuals_std_calculated)
              else global_lr_residuals_std,
          )
      except Exception:
        pass  # Fallback to global_lr_residuals_std if calculation for loc fails

    # Initialize TS models (to None) to track successful training.
    sarimax_model_fit, ets_model_fit = None, None

    # --- Per-Location Time Series Model Training ---
    if (
        len(ts_train_series_log) >= 52
        and not ts_train_series_log.isnull().all()
    ):
      # Check for constant series before fitting, as TS models can struggle with them
      if ts_train_series_log.nunique() > 1:
        try:
          sarimax_model_fit = SARIMAX(
              ts_train_series_log,
              order=(0, 1, 1),
              seasonal_order=(0, 1, 1, 52),
              enforce_stationarity=False,
              enforce_invertibility=False,
          ).fit(disp=False)
        except Exception:
          sarimax_model_fit = None

        try:
          ets_model_fit = ExponentialSmoothing(
              ts_train_series_log,
              seasonal_periods=52,
              trend='add',
              seasonal='add',
              initialization_method='estimated',
              damped_trend=True,
          ).fit(disp=False)
        except Exception:
          ets_model_fit = None

    # If all models (including global ML models if they failed training) failed for this location, fallback to zeros.
    if all(
        m is None
        for m in [sarimax_model_fit, ets_model_fit, lr_model, rf_model]
    ):
      pred_quantiles = pd.DataFrame(
          0,
          index=loc_test_x.index,
          columns=[f'quantile_{q}' for q in QUANTILES],
      )
      all_final_predictions.append(pred_quantiles)
      continue

    # Cache SARIMAX and ETS forecasts for efficiency as they are generated for the full horizon at once.
    sarimax_forecast_means_log, sarimax_forecast_stds_log = None, None
    ets_simulations_log = (
        None  # This will store N_SAMPLES simulations per horizon
    )

    if sarimax_model_fit:
      try:
        sarimax_results = sarimax_model_fit.get_forecast(steps=len(loc_test_x))
        sarimax_forecast_means_log = sarimax_results.predicted_mean.values
        sarimax_forecast_stds_log = sarimax_results.se_mean.values
        # Ensure stds are not too small
        sarimax_forecast_stds_log = np.maximum(
            MIN_STD_EPSILON, sarimax_forecast_stds_log
        )
      except Exception:
        sarimax_model_fit = None

    if ets_model_fit:
      try:
        # `simulate` in ExponentialSmoothing forecasts `nsimulations` steps,
        # each with `repetitions` paths. So `ets_simulations_log` is (steps, repetitions).
        ets_simulations_log = ets_model_fit.simulate(
            nsimulations=len(loc_test_x), repetitions=N_SAMPLES, anchor='end'
        )
      except Exception:
        ets_model_fit = None

    # Initialize history_series_log for recursive ML prediction for *this specific location*.
    # It contains actuals and will be updated with median predictions for future horizons.
    history_series_log = ts_train_series_log.copy()

    # --- Recursive Forecasting for Test Horizons ---
    for idx_row, row in loc_test_x.iterrows():
      target_end_date_h = row['target_end_date']
      h_idx = row['horizon']

      ensemble_samples_h_log = []

      # --- SARIMAX Prediction Samples ---
      if (
          sarimax_model_fit
          and sarimax_forecast_means_log is not None
          and h_idx < len(sarimax_forecast_means_log)
      ):
        mean_h_log = sarimax_forecast_means_log[h_idx]
        std_h_log = sarimax_forecast_stds_log[h_idx]
        samples_log = norm.rvs(
            loc=mean_h_log, scale=std_h_log, size=N_SAMPLES, random_state=rng
        )
        if (
            samples_log is not None
            and len(samples_log) > 0
            and np.all(np.isfinite(samples_log))
        ):
          ensemble_samples_h_log.extend(samples_log)

      # --- ETS Prediction Samples ---
      if (
          ets_model_fit
          and ets_simulations_log is not None
          and h_idx < ets_simulations_log.shape[0]
      ):
        samples_log = ets_simulations_log[
            h_idx, :
        ]  # Get all N_SAMPLES for this horizon
        if (
            samples_log is not None
            and len(samples_log) > 0
            and np.all(np.isfinite(samples_log))
        ):
          ensemble_samples_h_log.extend(samples_log)

      # --- Global ML Models Prediction (Linear Regression & Random Forest) ---
      current_forecast_df_slice = pd.DataFrame([{
          'target_end_date': target_end_date_h,
          'population': current_loc_population,
      }])

      # Generate features for the current horizon recursively using the updated history_series_log.
      X_test_ml_h_df = _generate_ml_features(
          current_forecast_df_slice,
          history_series_log,
          is_training=False,  # This call is now optimized for single point prediction
      )

      if not X_test_ml_h_df.empty:
        # Ensure the input to scaler/predict is a 2D array, even for a single sample.
        X_test_ml_h_values = X_test_ml_h_df.values

        # Prepare scaled features if scaler is available
        X_test_scaled_h = None
        if scaler is not None:
          try:
            X_test_scaled_h = scaler.transform(X_test_ml_h_values)
          except Exception:
            X_test_scaled_h = None  # Scaling failed for this input

        # Linear Regression samples (in log scale)
        # Use scaled features for LR if available and model exists
        if lr_model and X_test_scaled_h is not None:
          try:
            y_pred_lr_h_log = lr_model.predict(X_test_scaled_h)[0]

            # Apply horizon-dependent scaling to LR standard deviation
            lr_std_for_horizon = loc_lr_residuals_std_base * np.sqrt(h_idx + 1)
            lr_std_for_horizon = max(
                MIN_STD_EPSILON, lr_std_for_horizon
            )  # Ensure not too small

            samples_log = norm.rvs(
                loc=y_pred_lr_h_log,
                scale=lr_std_for_horizon,
                size=N_SAMPLES,
                random_state=rng,
            )
            if (
                samples_log is not None
                and len(samples_log) > 0
                and np.all(np.isfinite(samples_log))
            ):
              ensemble_samples_h_log.extend(samples_log)
          except Exception:
            pass

        # Random Forest samples (from individual trees, in log scale)
        # Use scaled features for RF if available for consistency, otherwise fall back to unscaled values.
        if rf_model:
          try:
            rf_input_features = (
                X_test_scaled_h
                if X_test_scaled_h is not None
                else X_test_ml_h_values
            )
            # Randomly sample from individual tree predictions to form distribution
            if rf_model.estimators_:  # Ensure there are estimators
              rf_tree_predictions_log = [
                  tree.predict(rf_input_features)[0]
                  for tree in rf_model.estimators_
              ]
              if (
                  rf_tree_predictions_log
              ):  # Ensure there are predictions from trees
                rf_samples_log = rng.choice(
                    rf_tree_predictions_log, size=N_SAMPLES, replace=True
                )
                if (
                    rf_samples_log is not None
                    and len(rf_samples_log) > 0
                    and np.all(np.isfinite(rf_samples_log))
                ):
                  ensemble_samples_h_log.extend(rf_samples_log)
          except Exception:
            pass

      # Inverse transform samples and calculate quantiles for the current (location, target_end_date_h)
      if ensemble_samples_h_log:
        clamped_samples_log = np.clip(
            ensemble_samples_h_log, a_min=None, a_max=MAX_LOG_VALUE
        )
        samples_h = np.maximum(0, np.expm1(clamped_samples_log))
        quant_values = np.quantile(samples_h, QUANTILES)
        median_prediction_h = np.median(samples_h)
      else:
        quant_values = np.zeros(len(QUANTILES))
        median_prediction_h = 0

      # Ensure non-negativity and monotonicity of quantiles, then round to nearest integer.
      quant_values = np.maximum(0, quant_values)
      quant_values = np.maximum.accumulate(quant_values)
      quant_values = np.round(quant_values).astype(int)

      temp_row_quantiles = pd.DataFrame(
          [quant_values],
          columns=[f'quantile_{q}' for q in QUANTILES],
          index=[idx_row],
      )
      all_final_predictions.append(temp_row_quantiles)

      # IMPORTANT: Update `history_series_log` with the median prediction for the current horizon (in log-scale).
      # This is crucial for recursive feature generation in subsequent horizons for this location.
      # Only update if the target_end_date_h is a future date relative to the last actual observed.
      # The history_series_log is strictly for values *prior* to the current target_end_date being forecasted.
      # The next forecast (h_idx+1) will then use this new value as `lag_1`.
      history_series_log.loc[target_end_date_h] = np.clip(
          np.log1p(median_prediction_h), a_min=None, a_max=MAX_LOG_VALUE
      )
      # Ensure history_series_log remains sorted by index for correct lagging and rolling mean calculations
      history_series_log = history_series_log.sort_index()

  # Concatenate all predictions from all locations and re-index to match the original test_x order.
  final_predictions_df = pd.concat(all_final_predictions).loc[test_x.index]

  return final_predictions_df


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
