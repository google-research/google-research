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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation,broad-exception-caught,unused-argument,missing-function-docstring,cell-var-from-loop

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
MODEL_NAME = 'Google_SAI-Adapted_11'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Forecasting Method:
# 1. Hierarchical Data Aggregation: Aggregate state-level data to form Regional (HHS regions) and National time series.
# 2. Multi-Level Modeling: Fit independent SARIMAX models to each state's series, and LSTM networks to regional and national series to generate probabilistic forecasts.
# 3. Hierarchical Disaggregation: Map regional and national LSTM forecasts down to the state level using a blend of population weighting and historical proportional contributions.
# 4. Ensemble Blending & Quantile Generation: Compute the final probabilistic quantile forecast for each state by taking a horizon-dependent weighted combination of state-specific SARIMAX, and disaggregated regional/national LSTM predictions, ensuring non-negativity and monotonicity.

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm
from typing import Any, Dict, List, Tuple

# For LSTMs - NEW IMPORTS
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow logging to keep output clean, as per instruction 'as quiet as possible'.
tf.get_logger().setLevel('ERROR')
# Suppress statsmodels convergence warnings for cleaner output during SARIMAX fitting.
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


# Assume TARGET_STR and QUANTILES are defined globally in the notebook context.
# TARGET_STR = 'Total COVID-19 Admissions'
# QUANTILES = [...]

# Correct HHS Region mapping based on official HHS regions.
# This mapping is crucial for hierarchical aggregation.
HHS_REGION_MAPPING = {
    'Connecticut': 'HHS_Region_1',
    'Maine': 'HHS_Region_1',
    'Massachusetts': 'HHS_Region_1',
    'New Hampshire': 'HHS_Region_1',
    'Rhode Island': 'HHS_Region_1',
    'Vermont': 'HHS_Region_1',
    'New Jersey': 'HHS_Region_2',
    'New York': 'HHS_Region_2',
    'Puerto Rico': 'HHS_Region_2',
    'Delaware': 'HHS_Region_3',
    'District of Columbia': 'HHS_Region_3',
    'Maryland': 'HHS_Region_3',
    'Pennsylvania': 'HHS_Region_3',
    'Virginia': 'HHS_Region_3',
    'West Virginia': 'HHS_Region_3',
    'Alabama': 'HHS_Region_4',
    'Florida': 'HHS_Region_4',
    'Georgia': 'HHS_Region_4',
    'Kentucky': 'HHS_Region_4',
    'Mississippi': 'HHS_Region_4',
    'North Carolina': 'HHS_Region_4',
    'South Carolina': 'HHS_Region_4',
    'Tennessee': 'HHS_Region_4',
    'Illinois': 'HHS_Region_5',
    'Indiana': 'HHS_Region_5',
    'Michigan': 'HHS_Region_5',
    'Minnesota': 'HHS_Region_5',
    'Ohio': 'HHS_Region_5',
    'Wisconsin': 'HHS_Region_5',
    'Arkansas': 'HHS_Region_6',
    'Louisiana': 'HHS_Region_6',
    'New Mexico': 'HHS_Region_6',
    'Oklahoma': 'HHS_Region_6',
    'Texas': 'HHS_Region_6',
    'Iowa': 'HHS_Region_7',
    'Kansas': 'HHS_Region_7',
    'Missouri': 'HHS_Region_7',
    'Nebraska': 'HHS_Region_7',
    'Colorado': 'HHS_Region_8',
    'Montana': 'HHS_Region_8',
    'North Dakota': 'HHS_Region_8',
    'South Dakota': 'HHS_Region_8',
    'Utah': 'HHS_Region_8',
    'Wyoming': 'HHS_Region_8',
    'Arizona': 'HHS_Region_9',
    'California': 'HHS_Region_9',
    'Hawaii': 'HHS_Region_9',
    'Nevada': 'HHS_Region_9',
    'Alaska': 'HHS_Region_10',
    'Idaho': 'HHS_Region_10',
    'Oregon': 'HHS_Region_10',
    'Washington': 'HHS_Region_10',
}

# --- Consolidated and refined SARIMAX_ORDER_CANDIDATES ---
SARIMAX_ORDER_CANDIDATES = [
    # Group 1: High-priority, robust seasonal models (often excellent baselines for weekly data)
    (
        (0, 1, 0),
        (0, 1, 0, 52),
    ),  # Seasonal Random Walk with seasonal differencing (very robust)
    ((0, 1, 1), (0, 1, 1, 52)),  # Common robust seasonal model for count data
    # Group 2: Simpler non-seasonal and general purpose models
    ((0, 1, 0), (0, 0, 0, 0)),  # Non-seasonal Random Walk with drift
    (
        (1, 1, 0),
        (0, 0, 0, 0),
    ),  # ARIMA(1,1,0) - Non-seasonal AR with differencing
    (
        (0, 1, 1),
        (0, 0, 0, 0),
    ),  # ARIMA(0,1,1) - Non-seasonal MA with differencing
    ((1, 1, 0), (0, 1, 1, 52)),  # Combining non-seasonal AR with seasonal ARIMA
    # Group 3: More complex, but potentially better fitting seasonal models (try last if data supports)
    ((1, 1, 1), (0, 1, 1, 52)),  # ARIMA(1,1,1)(0,1,1,52)
    (
        (1, 1, 1),
        (1, 1, 1, 52),
    ),  # ARIMA(1,1,1)(1,1,1,52) - Most complex, try last if data supports
]

# NEW CONSTANT: Maximum allowed log-transformed value before expm1 to prevent OverflowError.
MAX_LOG_VALUE_BEFORE_EXPM1_CLIP = 700.0
MIN_LOG_STD_ERR_VAL_FLOOR = (
    0.18  # Consistent floor for log-transformed std error across models
)


def _generate_quantiles_from_ci(
    point_forecast_series,
    lower_ci_series,
    upper_ci_series,
    min_std_err_val_original_scale,
):
  """Generates all specified quantiles from a point forecast and its 95% confidence interval.

  Ensures non-negativity and monotonicity of quantiles, respecting a minimum
  standard error on the original scale. Input series (point_forecast, lower_ci,
  upper_ci) are assumed to be on the original scale.
  """
  Z_975 = norm.ppf(0.975)
  ci_width = (upper_ci_series - lower_ci_series).clip(lower=0)
  std_err_series = ci_width / (2 * Z_975)
  # Ensure std_err_series is never below the robust minimum.
  std_err_series = std_err_series.fillna(min_std_err_val_original_scale).clip(
      lower=min_std_err_val_original_scale
  )

  quantile_predictions = pd.DataFrame(
      index=point_forecast_series.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
  )

  for q in QUANTILES:
    quantile_predictions[f'quantile_{q}'] = (
        point_forecast_series.values + norm.ppf(q) * std_err_series.values
    )

  quantile_predictions = quantile_predictions.clip(lower=0)
  quantile_predictions = quantile_predictions.cummax(axis=1)
  return quantile_predictions


def _calculate_series_min_std_err_original_scale(series):
  """Calculates a robust minimum standard error value for a given series on the original scale.

  This helps prevent overly narrow prediction intervals for low-variance or
  sparse series, contributing to better calibration.
  """
  MIN_ABS_STD_ERROR_FLOOR = 3.0
  RELATIVE_STD_ERROR_FLOOR_PERCENTAGE = 0.08
  MAX_PROPORTIONAL_STD_ERROR_FLOOR = 50.0
  MIN_STD_ERR_MEAN_LOOKBACK_WEEKS = 52

  recent_mean = 0.0
  if (
      not series.empty
      and isinstance(series.index, pd.DatetimeIndex)
      and series.index.is_monotonic_increasing
  ):
    max_date = series.index.max()
    recent_series = series[
        series.index
        >= (max_date - pd.Timedelta(weeks=MIN_STD_ERR_MEAN_LOOKBACK_WEEKS))
    ]
    if not recent_series.empty:
      recent_mean = recent_series.mean()
  elif not series.empty:
    recent_mean = series.mean()

  if recent_mean > 0:
    proportional_floor = RELATIVE_STD_ERROR_FLOOR_PERCENTAGE * recent_mean
    proportional_floor = min(
        proportional_floor, MAX_PROPORTIONAL_STD_ERROR_FLOOR
    )
    series_min_std_err_val_original_scale = max(
        MIN_ABS_STD_ERROR_FLOOR, proportional_floor
    )
  else:
    series_min_std_err_val_original_scale = MIN_ABS_STD_ERROR_FLOOR

  return series_min_std_err_val_original_scale


def _generate_naive_quantiles(
    series,
    forecast_dates_array,
    horizons_map,
    series_min_std_err_val_original_scale,
):
  """Generates naive quantiles for given forecast dates based on the last observed value."""
  forecast_dates = pd.to_datetime(forecast_dates_array)
  num_steps = len(forecast_dates)
  Z_975 = norm.ppf(0.975)

  series = series.clip(lower=0)

  if series.empty or series.sum() == 0:
    zero_forecasts = pd.DataFrame(
        0, index=forecast_dates, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return zero_forecasts.cummax(axis=1)

  log_transformed_series = np.log1p(series)

  # NEW: Handle series that become constant after log1p transformation
  if (
      log_transformed_series.min() == log_transformed_series.max()
      or log_transformed_series.nunique() < 2
  ):
    last_val_log = (
        log_transformed_series.iloc[-1]
        if not log_transformed_series.empty
        else np.log1p(0)
    )
    point_forecast_series_log = pd.Series(
        [last_val_log] * num_steps, index=forecast_dates
    )
    estimated_log_std_err_series = pd.Series(
        [MIN_LOG_STD_ERR_VAL_FLOOR] * num_steps, index=forecast_dates
    )

    lower_ci_log = (
        point_forecast_series_log - Z_975 * estimated_log_std_err_series
    )
    upper_ci_log = (
        point_forecast_series_log + Z_975 * estimated_log_std_err_series
    )

    point_forecast_series_log = point_forecast_series_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )
    lower_ci_log = lower_ci_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )
    upper_ci_log = upper_ci_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )

    point_forecast_original = np.expm1(point_forecast_series_log)
    lower_ci_original = np.expm1(lower_ci_log)
    upper_ci_original = np.expm1(upper_ci_log)

    return _generate_quantiles_from_ci(
        point_forecast_original,
        lower_ci_original,
        upper_ci_original,
        series_min_std_err_val_original_scale,
    )

  last_val_log = (
      log_transformed_series.iloc[-1]
      if not log_transformed_series.empty
      else np.log1p(0)
  )
  point_forecast_series_log = pd.Series(
      [last_val_log] * num_steps, index=forecast_dates
  )

  diffs_log = log_transformed_series.diff().dropna()

  # NEW: Robust std dev calculation for diffs_log
  std_dev_from_data_base_log = np.nan
  if len(diffs_log) >= 5:  # Use a minimum number of diffs for robust std/MAD
    recent_diffs_log = diffs_log.tail(52)  # Use recent year for volatility
    std_dev_from_data_base_log = np.std(recent_diffs_log)

    mad_log_diffs = np.median(
        np.abs(recent_diffs_log - np.median(recent_diffs_log))
    )
    if not np.isnan(mad_log_diffs) and mad_log_diffs > 0:
      std_dev_from_data_base_log = np.nanmax(
          [std_dev_from_data_base_log, mad_log_diffs * 1.4826]
      )
  elif len(diffs_log) > 1:
    std_dev_from_data_base_log = np.std(
        diffs_log
    )  # For very short series, use all
  else:
    std_dev_from_data_base_log = np.nan  # Will be floored later

  # Apply global min log std error floor
  if (
      not np.isnan(std_dev_from_data_base_log)
      and std_dev_from_data_base_log > MIN_LOG_STD_ERR_VAL_FLOOR
  ):
    base_log_std_err = std_dev_from_data_base_log
  else:
    base_log_std_err = MIN_LOG_STD_ERR_VAL_FLOOR

  estimated_log_std_err_series = pd.Series(index=forecast_dates, dtype=float)
  for date in forecast_dates:
    horizon_val = horizons_map.get(date, 0)
    estimated_log_std_err_series.loc[date] = base_log_std_err * np.sqrt(
        horizon_val + 1
    )

  estimated_log_std_err_series = estimated_log_std_err_series.clip(
      lower=MIN_LOG_STD_ERR_VAL_FLOOR
  )

  lower_ci_log = (
      point_forecast_series_log - Z_975 * estimated_log_std_err_series
  )
  upper_ci_log = (
      point_forecast_series_log + Z_975 * estimated_log_std_err_series
  )

  point_forecast_series_log = point_forecast_series_log.clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )
  lower_ci_log = lower_ci_log.clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )
  upper_ci_log = upper_ci_log.clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )

  point_forecast_original = np.expm1(point_forecast_series_log)
  lower_ci_original = np.expm1(lower_ci_log)
  upper_ci_original = np.expm1(upper_ci_log)

  point_forecast_original.index = forecast_dates
  lower_ci_original.index = forecast_dates
  upper_ci_original.index = forecast_dates

  return _generate_quantiles_from_ci(
      point_forecast_original,
      lower_ci_original,
      upper_ci_original,
      series_min_std_err_val_original_scale,
  )


def _fit_and_forecast_sarimax(
    series,
    forecast_dates_array,
    model_id,
    horizons_map,
):
  """Fits SARIMAX models to a given time series, attempting various orders for robustness,

  and generates quantile forecasts. Falls back to naive forecasts if SARIMAX
  fails.
  """
  forecast_dates = pd.to_datetime(forecast_dates_array)
  series = series.clip(lower=0)

  if not isinstance(series.index, pd.DatetimeIndex):
    series.index = pd.to_datetime(series.index)
  series = series.asfreq('W-SAT', fill_value=0)

  series_min_std_err_val_original_scale = (
      _calculate_series_min_std_err_original_scale(series)
  )

  SEASONAL_PERIOD = 52
  MIN_LEN_FOR_NON_SEASONAL_ARMA = 20
  MIN_LEN_FOR_SEASONAL_DIFF = SEASONAL_PERIOD + 1
  MIN_LEN_FOR_FULL_SEASONAL_ARMA = SEASONAL_PERIOD * 2
  MIN_OBSERVATIONS_FOR_SARIMAX_ATTEMPT = 5
  MIN_NON_ZERO_OBSERVATIONS_FOR_SARIMAX = 15

  if (
      series.empty
      or series.sum() == 0
      or len(series) < MIN_OBSERVATIONS_FOR_SARIMAX_ATTEMPT
      or np.count_nonzero(series) < MIN_NON_ZERO_OBSERVATIONS_FOR_SARIMAX
  ):
    return _generate_naive_quantiles(
        series,
        forecast_dates,
        horizons_map,
        series_min_std_err_val_original_scale,
    )

  log_transformed_series = np.log1p(series)
  current_series_length = len(log_transformed_series)

  if not np.isfinite(log_transformed_series).all():
    return _generate_naive_quantiles(
        series,
        forecast_dates,
        horizons_map,
        series_min_std_err_val_original_scale,
    )

  if (
      log_transformed_series.min() == log_transformed_series.max()
      or log_transformed_series.nunique() < 2
  ):
    return _generate_naive_quantiles(
        series,
        forecast_dates,
        horizons_map,
        series_min_std_err_val_original_scale,
    )

  num_forecast_steps = len(forecast_dates)

  filtered_candidates = []
  for order, seasonal_order in SARIMAX_ORDER_CANDIDATES:
    is_seasonal_model = seasonal_order[3] > 0
    has_seasonal_differencing = seasonal_order[1] > 0
    has_seasonal_arma_components = (
        seasonal_order[0] > 0 or seasonal_order[2] > 0
    )
    has_non_seasonal_arma_components = order[0] > 0 or order[2] > 0

    if is_seasonal_model:
      if (
          has_seasonal_differencing
          and current_series_length < MIN_LEN_FOR_SEASONAL_DIFF
      ):
        continue
      if (
          has_seasonal_arma_components
          and current_series_length < MIN_LEN_FOR_FULL_SEASONAL_ARMA
      ):
        continue

    if (
        has_non_seasonal_arma_components
        and current_series_length < MIN_LEN_FOR_NON_SEASONAL_ARMA
    ):
      continue

    filtered_candidates.append((order, seasonal_order))

  best_sarimax_res = None
  min_aic = np.inf

  for order, seasonal_order in filtered_candidates:
    try:
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sarimax_model = SARIMAX(
            log_transformed_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
            initialization='approximate',
            freq='W-SAT',
        )
        sarimax_res = sarimax_model.fit(disp=False, maxiter=1000)

        if (
            sarimax_res.aic is not None
            and not np.isinf(sarimax_res.aic)
            and not np.isnan(sarimax_res.aic)
            and sarimax_res.aic < min_aic
        ):
          min_aic = sarimax_res.aic
          best_sarimax_res = sarimax_res

    except Exception:
      pass

  if best_sarimax_res is None:
    robust_fallback_order = ((0, 1, 0), (0, 1, 0, SEASONAL_PERIOD))
    if current_series_length >= MIN_LEN_FOR_SEASONAL_DIFF:
      try:
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          sarimax_model_fallback = SARIMAX(
              log_transformed_series,
              order=robust_fallback_order[0],
              seasonal_order=robust_fallback_order[1],
              enforce_stationarity=True,
              enforce_invertibility=True,
              initialization='approximate',
              freq='W-SAT',
          )
          sarimax_res_fallback = sarimax_model_fallback.fit(
              disp=False, maxiter=1000
          )
          if (
              sarimax_res_fallback.aic is not None
              and not np.isinf(sarimax_res_fallback.aic)
              and not np.isnan(sarimax_res_fallback.aic)
          ):
            best_sarimax_res = sarimax_res_fallback
      except Exception:
        pass

  if best_sarimax_res is None:
    return _generate_naive_quantiles(
        series,
        forecast_dates,
        horizons_map,
        series_min_std_err_val_original_scale,
    )

  forecast_obj_log = best_sarimax_res.get_forecast(steps=num_forecast_steps)
  forecast_result_log = forecast_obj_log.predicted_mean
  conf_int_log = forecast_obj_log.conf_int(alpha=0.05)

  if (
      forecast_result_log.isnull().any()
      or conf_int_log.isnull().any().any()
      or not np.isfinite(forecast_result_log).all()
      or not np.isfinite(conf_int_log).all().all()
  ):
    return _generate_naive_quantiles(
        series,
        forecast_dates,
        horizons_map,
        series_min_std_err_val_original_scale,
    )

  forecast_result_log = forecast_result_log.clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )
  conf_int_log.iloc[:, 0] = conf_int_log.iloc[:, 0].clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )
  conf_int_log.iloc[:, 1] = conf_int_log.iloc[:, 1].clip(
      lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
  )

  point_forecast_original = np.expm1(forecast_result_log)
  lower_ci_original = np.expm1(conf_int_log.iloc[:, 0])
  upper_ci_original = np.expm1(conf_int_log.iloc[:, 1])

  point_forecast_original.index = forecast_dates
  lower_ci_original.index = forecast_dates
  upper_ci_original.index = forecast_dates

  return _generate_quantiles_from_ci(
      point_forecast_original,
      lower_ci_original,
      upper_ci_original,
      series_min_std_err_val_original_scale,
  )


# --- NEW LSTM helper functions ---


def _create_lstm_sequences(
    data, n_steps_in
):
  """Creates sequences of data for LSTM training.

  X: [n_steps_in lookback values] y: [next value]
  """
  X, y = [], []
  for i in range(len(data)):
    end_ix = i + n_steps_in
    if end_ix > len(data) - 1:
      break
    seq_x, seq_y = data[i:end_ix], data[end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)


def _fit_and_forecast_lstm(
    series,
    forecast_dates_array,
    model_id,
    horizons_map,
):
  """Fits an LSTM model to a given time series and generates quantile forecasts.

  Falls back to SARIMAX forecasts if LSTM training or prediction fails.
  """
  # NEW: Set random seed for reproducibility
  tf.random.set_seed(42)

  forecast_dates = pd.to_datetime(forecast_dates_array)
  num_forecast_steps = len(forecast_dates)

  # Ensure input series is non-negative and has continuous weekly frequency.
  series = series.clip(lower=0)
  if not isinstance(series.index, pd.DatetimeIndex):
    series.index = pd.to_datetime(series.index)
  series = series.asfreq('W-SAT', fill_value=0)

  series_min_std_err_val_original_scale = (
      _calculate_series_min_std_err_original_scale(series)
  )

  # Define minimum data length for LSTM. LSTMs need a decent amount of data.
  LSTM_MIN_OBSERVATIONS = (
      104  # Roughly 2 years of weekly data for seasonal patterns
  )
  LSTM_LOOKBACK_WEEKS = 52  # Input sequence length for LSTM (1 year of history)
  LSTM_EPOCHS = 50  # Number of training epochs
  LSTM_BATCH_SIZE = 16  # Batch size for training

  # Early exit for very sparse series or too few observations.
  if (
      series.empty
      or series.sum() == 0
      or len(series) < LSTM_MIN_OBSERVATIONS
      or np.count_nonzero(series) < (LSTM_MIN_OBSERVATIONS / 2)
  ):
    # NEW: Fallback to SARIMAX for regional/national models if LSTM conditions not met
    return _fit_and_forecast_sarimax(
        series, forecast_dates_array, model_id, horizons_map
    )

  # Apply log1p transformation
  log_transformed_series = np.log1p(series)

  # Handle non-finite values after transformation
  if not np.isfinite(log_transformed_series).all():
    # NEW: Fallback to SARIMAX
    return _fit_and_forecast_sarimax(
        series, forecast_dates_array, model_id, horizons_map
    )

  # Handle constant series after transformation
  if (
      log_transformed_series.min() == log_transformed_series.max()
      or log_transformed_series.nunique() < 2
  ):
    # NEW: Fallback to SARIMAX
    return _fit_and_forecast_sarimax(
        series, forecast_dates_array, model_id, horizons_map
    )

  # Normalize the log-transformed data
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(
      log_transformed_series.values.reshape(-1, 1)
  )

  # Create sequences for LSTM
  X, y = _create_lstm_sequences(scaled_data, LSTM_LOOKBACK_WEEKS)

  # If not enough sequences can be formed, fallback to SARIMAX
  if len(X) == 0:
    # NEW: Fallback to SARIMAX
    return _fit_and_forecast_sarimax(
        series, forecast_dates_array, model_id, horizons_map
    )

  try:
    # Define LSTM model
    model = keras.Sequential([
        keras.layers.LSTM(
            50, activation='relu', input_shape=(LSTM_LOOKBACK_WEEKS, 1)
        ),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train LSTM model (verbose=0 to suppress output)
    model.fit(X, y, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)

    # Generate multi-step point forecasts recursively
    last_sequence = scaled_data[-LSTM_LOOKBACK_WEEKS:].reshape(
        1, LSTM_LOOKBACK_WEEKS, 1
    )
    scaled_log_forecasts_list = []

    for _ in range(num_forecast_steps):
      predicted_scaled_val = model.predict(last_sequence, verbose=0)[0, 0]
      scaled_log_forecasts_list.append(predicted_scaled_val)
      # Update sequence: remove oldest, add new prediction
      last_sequence = np.append(
          last_sequence[:, 1:, :], [[predicted_scaled_val]], axis=1
      )

    # Inverse transform scaled log forecasts to original log scale
    log_forecasts_unscaled = scaler.inverse_transform(
        np.array(scaled_log_forecasts_list).reshape(-1, 1)
    ).flatten()

    # Calculate prediction intervals based on training residuals on the *log scale*
    # Ensure proper alignment of `y_train_actual_log` and `train_predictions_log`
    y_train_actual_log = log_transformed_series.iloc[
        LSTM_LOOKBACK_WEEKS:
    ].values
    train_predictions_scaled = model.predict(X, verbose=0)
    train_predictions_log = scaler.inverse_transform(
        train_predictions_scaled
    ).flatten()

    # Ensure lengths match for residual calculation
    if len(y_train_actual_log) == len(train_predictions_log):
      base_std_dev_log = np.std(y_train_actual_log - train_predictions_log)
    else:
      base_std_dev_log = np.nan

    # Use a minimum floor for the log standard error, consistent with naive, for calibration.
    # Use the same MIN_LOG_STD_ERR_VAL_FLOOR for consistency
    if (
        np.isnan(base_std_dev_log)
        or base_std_dev_log < MIN_LOG_STD_ERR_VAL_FLOOR
    ):
      base_std_dev_log = MIN_LOG_STD_ERR_VAL_FLOOR

    # Scale standard deviation by horizon for increasing uncertainty further out
    Z_975 = norm.ppf(0.975)
    estimated_log_std_err_series = pd.Series(index=forecast_dates, dtype=float)
    for i, date in enumerate(forecast_dates):
      horizon_val = horizons_map.get(date, 0)
      # The +1 ensures that horizon 0 also has some uncertainty and scales up for longer horizons.
      estimated_log_std_err_series.loc[date] = base_std_dev_log * np.sqrt(
          horizon_val + 1
      )
    estimated_log_std_err_series = estimated_log_std_err_series.clip(
        lower=MIN_LOG_STD_ERR_VAL_FLOOR
    )

    point_forecast_series_log = pd.Series(
        log_forecasts_unscaled, index=forecast_dates
    )
    lower_ci_log = (
        point_forecast_series_log - Z_975 * estimated_log_std_err_series
    )
    upper_ci_log = (
        point_forecast_series_log + Z_975 * estimated_log_std_err_series
    )

    # Clip log-transformed values before expm1
    point_forecast_series_log = point_forecast_series_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )
    lower_ci_log = lower_ci_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )
    upper_ci_log = upper_ci_log.clip(
        lower=0, upper=MAX_LOG_VALUE_BEFORE_EXPM1_CLIP
    )

    point_forecast_original = np.expm1(point_forecast_series_log)
    lower_ci_original = np.expm1(lower_ci_log)
    upper_ci_original = np.expm1(upper_ci_log)

    return _generate_quantiles_from_ci(
        point_forecast_original,
        lower_ci_original,
        upper_ci_original,
        series_min_std_err_val_original_scale,
    )

  except Exception as e:
    # print(f"LSTM for {model_id} failed: {e}. Falling back to SARIMAX.") # Optional: for debug only.
    # NEW: Fallback to SARIMAX for regional/national models if LSTM training fails
    return _fit_and_forecast_sarimax(
        series, forecast_dates_array, model_id, horizons_map
    )


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # No blanket warning suppression, as per instructions. Specific model suppressions remain if appropriate.

  # --- 0. Pre-computation and Data Preparation ---

  # Ensure 'target_end_date' is a datetime object for consistent time series operations.
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  # train_y's index is assumed to be aligned with train_x's index.

  # Combine train_x and train_y into a single DataFrame for easier manipulation.
  # Use .loc to avoid SettingWithCopyWarning if train_x is a slice.
  train_df = train_x.loc[train_y.index].copy()
  train_df[TARGET_STR] = train_y

  # Sort data by location and date to ensure correct time series order for modeling.
  train_df = train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Assume 'locations' DataFrame is available in the global scope based on notebook context.
  # Create a copy of global 'locations' to add region info without modifying global state.
  locations_with_regions = locations.copy()

  # Map location_name to correct HHS_Region. Locations not in the mapping default to 'Unknown_Region'.
  locations_with_regions['hhs_region'] = (
      locations_with_regions['location_name']
      .map(HHS_REGION_MAPPING)
      .fillna('Unknown_Region')
  )

  # Merge region information into both training and test dataframes.
  train_df = train_df.merge(
      locations_with_regions[['location', 'hhs_region']],
      on='location',
      how='left',
  )
  test_x = test_x.merge(
      locations_with_regions[['location', 'hhs_region']],
      on='location',
      how='left',
  )

  # Extract unique locations from test_x for which forecasts are needed.
  forecast_locations = test_x['location'].unique()

  if test_x.empty:
    # If no forecast steps are required or test_x is empty, return an empty DataFrame with expected columns.
    return pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )

  # Pre-extract sorted unique forecast dates for each location, regional, and national levels.
  # These are np.ndarrays as they come from .unique(), consistent with _fit_and_forecast_sarimax expects.
  forecast_dates_by_location = test_x.groupby('location')[
      'target_end_date'
  ].apply(lambda x: x.sort_values().unique())
  forecast_dates_global = test_x['target_end_date'].sort_values().unique()

  # Create a map from target_end_date to horizon for dynamic std dev scaling in naive forecasts.
  # This map will be used for all levels (state, regional, national) since target_end_date defines horizon.
  horizons_map = test_x.set_index('target_end_date')['horizon'].to_dict()

  # --- 1. Hierarchical Aggregation ---
  # Aggregate training data to Regional and National levels by summing the target variable.
  regional_train_df = (
      train_df.groupby(['hhs_region', 'target_end_date'])[TARGET_STR]
      .sum()
      .reset_index()
  )
  national_train_df = (
      train_df.groupby('target_end_date')[TARGET_STR].sum().reset_index()
  )

  # Store the latest populations for disaggregation ratios.
  latest_state_populations = locations_with_regions.set_index('location')[
      'population'
  ]
  latest_regional_populations = locations_with_regions.groupby('hhs_region')[
      'population'
  ].sum()
  latest_national_population = locations_with_regions['population'].sum()

  # Create a template for empty forecasts, crucial for consistent reindexing and error handling.
  empty_forecast_template_global = pd.DataFrame(
      0,
      index=pd.to_datetime(forecast_dates_global),
      columns=[f'quantile_{q}' for q in QUANTILES],
  )

  # Calculate Historical Ratios for Disaggregation
  RATIO_LOOKBACK_WEEKS = (
      52  # Use one year (52 weeks) of recent data for calculating ratios
  )

  # Filter train_df to the look-back window for ratio calculation.
  max_train_date = (
      train_df['target_end_date'].max() if not train_df.empty else pd.NaT
  )
  recent_train_df = train_df[
      train_df['target_end_date']
      >= (max_train_date - pd.Timedelta(weeks=RATIO_LOOKBACK_WEEKS))
  ]

  if recent_train_df.empty:
    recent_train_df = train_df

  state_admissions_totals = recent_train_df.groupby('location')[
      TARGET_STR
  ].sum()
  regional_admissions_totals = recent_train_df.groupby('hhs_region')[
      TARGET_STR
  ].sum()
  national_admissions_total = (
      recent_train_df[TARGET_STR].sum() if not recent_train_df.empty else 0
  )

  historical_state_to_region_ratios = {}
  historical_state_to_national_ratios = {}

  MIN_ADMISSIONS_SUM_EPSILON = 1e-6

  for loc_id in forecast_locations:
    state_region_series = locations_with_regions[
        locations_with_regions['location'] == loc_id
    ]['hhs_region']
    if (
        state_region_series.empty
        or state_region_series.iloc[0] == 'Unknown_Region'
    ):
      historical_state_to_region_ratios[loc_id] = 0.0
      historical_state_to_national_ratios[loc_id] = 0.0
      continue
    state_region = state_region_series.iloc[0]

    state_sum = state_admissions_totals.get(loc_id, 0)
    region_sum = regional_admissions_totals.get(state_region, 0)

    region_sum_for_ratio = region_sum + MIN_ADMISSIONS_SUM_EPSILON
    nation_sum_for_ratio = (
        national_admissions_total + MIN_ADMISSIONS_SUM_EPSILON
    )

    hist_to_region = min(1.0, max(0.0, state_sum) / region_sum_for_ratio)
    hist_to_national = min(1.0, max(0.0, state_sum) / nation_sum_for_ratio)

    historical_state_to_region_ratios[loc_id] = hist_to_region
    historical_state_to_national_ratios[loc_id] = hist_to_national

  # --- 2. Local Modeling (SARIMAX for each state) ---
  state_sarimax_preds: Dict[int, pd.DataFrame] = {}
  for loc_id in forecast_locations:
    state_data = train_df[train_df['location'] == loc_id].set_index(
        'target_end_date'
    )[TARGET_STR]
    current_forecast_dates = forecast_dates_by_location.get(
        loc_id, np.array([])
    )

    if current_forecast_dates.size == 0:
      state_sarimax_preds[loc_id] = empty_forecast_template_global
      continue

    state_forecast_df = _fit_and_forecast_sarimax(
        state_data, current_forecast_dates, f'state {loc_id}', horizons_map
    )
    state_sarimax_preds[loc_id] = state_forecast_df.reindex(
        pd.to_datetime(forecast_dates_global), fill_value=0
    )

  # --- 3. Regional/National Modeling (LSTM for broader dynamics, with SARIMAX fallback) ---
  regional_lstm_preds: Dict[str, pd.DataFrame] = {}
  for region_id in regional_train_df['hhs_region'].unique():
    region_data = regional_train_df[
        regional_train_df['hhs_region'] == region_id
    ].set_index('target_end_date')[TARGET_STR]
    regional_lstm_preds[region_id] = _fit_and_forecast_lstm(
        region_data, forecast_dates_global, f'region {region_id}', horizons_map
    )

  national_data = national_train_df.set_index('target_end_date')[TARGET_STR]
  national_lstm_preds = _fit_and_forecast_lstm(
      national_data, forecast_dates_global, 'national', horizons_map
  )

  # --- 4. Disaggregation ---
  # Disaggregate regional and national forecasts to state level based on a blend of population and historical ratios.
  disaggregated_regional_forecasts = {}
  disaggregated_national_forecasts = {}

  W_POP_DISAGG = 0.5
  W_HIST_DISAGG = 0.5

  for loc_id in forecast_locations:
    state_pop = latest_state_populations.get(loc_id, 0)

    if state_pop <= 0:
      disaggregated_regional_forecasts[loc_id] = empty_forecast_template_global
      disaggregated_national_forecasts[loc_id] = empty_forecast_template_global
      continue

    state_region_series = locations_with_regions[
        locations_with_regions['location'] == loc_id
    ]['hhs_region']
    if (
        state_region_series.empty
        or state_region_series.iloc[0] == 'Unknown_Region'
    ):
      disaggregated_regional_forecasts[loc_id] = empty_forecast_template_global
      disaggregated_national_forecasts[loc_id] = empty_forecast_template_global
      continue
    state_region = state_region_series.iloc[0]

    region_pop = latest_regional_populations.get(state_region, 0)
    national_pop = (
        latest_national_population if latest_national_population > 0 else 0
    )

    population_ratio_regional = state_pop / region_pop if region_pop > 0 else 0
    population_ratio_national = (
        state_pop / national_pop if national_pop > 0 else 0
    )

    historical_ratio_regional = historical_state_to_region_ratios.get(
        loc_id, 0.0
    )
    historical_ratio_national = historical_state_to_national_ratios.get(
        loc_id, 0.0
    )

    blended_ratio_regional = max(
        0.0,
        (population_ratio_regional * W_POP_DISAGG)
        + (historical_ratio_regional * W_HIST_DISAGG),
    )
    blended_ratio_national = max(
        0.0,
        (population_ratio_national * W_POP_DISAGG)
        + (historical_ratio_national * W_HIST_DISAGG),
    )

    # Disaggregate regional forecasts using blended ratio.
    # Now using regional_lstm_preds.
    if state_region in regional_lstm_preds:
      disaggregated_regional_forecasts[loc_id] = regional_lstm_preds[
          state_region
      ].apply(lambda x: x * blended_ratio_regional)
    else:
      disaggregated_regional_forecasts[loc_id] = empty_forecast_template_global

    # Disaggregate national forecasts using blended ratio.
    # Now using national_lstm_preds.
    if not national_lstm_preds.empty:
      disaggregated_national_forecasts[loc_id] = national_lstm_preds.apply(
          lambda x: x * blended_ratio_national
      )
    else:
      disaggregated_national_forecasts[loc_id] = empty_forecast_template_global

  # --- 5. Ensemble Weighting ---
  # Define ensemble weights dynamically based on forecast horizon.
  ENSEMBLE_WEIGHTS_PER_HORIZON = {
      0: {'local': 0.85, 'regional': 0.075, 'national': 0.075},
      1: {'local': 0.7, 'regional': 0.15, 'national': 0.15},
      2: {'local': 0.5, 'regional': 0.25, 'national': 0.25},
      3: {'local': 0.35, 'regional': 0.325, 'national': 0.325},
  }
  DEFAULT_HORIZON_WEIGHTS = ENSEMBLE_WEIGHTS_PER_HORIZON[
      max(ENSEMBLE_WEIGHTS_PER_HORIZON.keys())
  ]

  final_predictions_list = []
  # Iterate through the original test_x index to ensure the output DataFrame's index matches.
  for test_idx, row in test_x.iterrows():
    loc_id = row['location']
    target_end_date = row['target_end_date']
    horizon = row['horizon']

    weights = ENSEMBLE_WEIGHTS_PER_HORIZON.get(horizon, DEFAULT_HORIZON_WEIGHTS)

    w_local = weights['local']
    w_regional = weights['regional']
    w_national = weights['national']

    # Defensive normalization of weights.
    total_weight = w_local + w_regional + w_national
    if total_weight > 0:
      w_local /= total_weight
      w_regional /= total_weight
      w_national /= total_weight
    else:  # Fallback for extremely unlikely case where all weights are zero.
      w_local, w_regional, w_national = 1 / 3, 1 / 3, 1 / 3

    # Retrieve forecasts using target_end_date as index.
    # Local model (state-level) is SARIMAX. Regional and National models are LSTM (with SARIMAX fallback).
    local_pred = state_sarimax_preds.get(
        loc_id, empty_forecast_template_global
    ).loc[target_end_date]
    regional_pred = disaggregated_regional_forecasts.get(
        loc_id, empty_forecast_template_global
    ).loc[target_end_date]
    national_pred = disaggregated_national_forecasts.get(
        loc_id, empty_forecast_template_global
    ).loc[target_end_date]

    # Weighted combination for each quantile.
    ensemble_pred = (
        w_local * local_pred
        + w_regional * regional_pred
        + w_national * national_pred
    )
    final_predictions_list.append(ensemble_pred)

  test_y_hat_quantiles = pd.DataFrame(
      final_predictions_list,
      columns=[f'quantile_{q}' for q in QUANTILES],
      index=test_x.index,
  )

  # Final check for non-negativity and monotonicity across all forecasts.
  test_y_hat_quantiles = test_y_hat_quantiles.clip(lower=0)
  test_y_hat_quantiles = test_y_hat_quantiles.cummax(axis=1)

  # Round all quantile predictions to the nearest integer.
  test_y_hat_quantiles = test_y_hat_quantiles.round()

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
