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
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning from HuberRegressor if it's not converging perfectly
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Assume QUANTILES and TARGET_STR are globally defined as in the notebook preamble
# For local definition if not globally available:
# QUANTILES = [
#     0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
# ]
# TARGET_STR = 'Total Influenza Admissions'

# Constants used internally for this function
ADMISSIONS_PROPORTION_COL = 'admissions_proportion'
LOGIT_ADMISSIONS_PROPORTION_COL = 'logit_admissions_proprtion'
ILI_RATE_COL = 'ili_rate'
LOGIT_ILI_RATE_COL = 'logit_ili_rate'
RESIDUAL_FROM_TREND_COL = (  # Renamed for clarity (delta_j,t)
    'season_specific_residual'
)
GLOBAL_EWMA_DISCREPANCY_COL = (  # Dynamic global discrepancy (mu_t_proxy)
    'global_ewma_discrepancy'
)
STATISTICAL_SEASONAL_BASELINE = (  # Renamed for clarity (was BASELINE_TREND_COL)
    'statistical_seasonal_baseline'
)

EWMA_ALPHA = 0.1  # Alpha for Exponentially Weighted Moving Average
N_SAMPLES = (
    2000  # IMPROVEMENT 7: Increased for better quantile stability (from 1000)
)
MIN_PREDICTED_ADMISSIONS = 0  # Ensure non-negative predictions
EPSILON_PROPORTION = 1e-7  # Epsilon for logit/expit to handle 0 and 1
MIN_STD_DEV_LOGIT = 1e-3  # Minimum standard deviation in logit space
# IMPROVEMENT 2: Overdispersion parameters, adapting to horizon
OVERDISPERSION_BASE_FACTOR = 1.15  # Baseline overdispersion multiplier for std dev (instead of fixed sqrt(1.2)) - INCREASED FROM 1.05
OVERDISPERSION_PER_HORIZON_INCREMENT = 0.1  # Additional factor per horizon (multiplied by horizon number) - INCREASED FROM 0.05
MAX_SLOPE_SCALER = (
    5.0  # Max slope for robust linear scaler to prevent extreme values
)


def logit_safe(p):
  """Safely apply logit transformation, clipping p to avoid log(0) or log(1)."""
  p = np.clip(p, EPSILON_PROPORTION, 1 - EPSILON_PROPORTION)
  return logit(p)


def expit_safe(x):
  """Safely apply expit (inverse logit) transformation."""
  return expit(x)


class RobustSimpleLinearScaler:
  """A robust linear scaler that estimates slope and intercept using various regression

  techniques, with safeguards for edge cases like constant or empty input.
  Enforces a positive and bounded slope.
  """

  def __init__(self, fallback_to_median=True):
    self.slope = 1.0  # Default fallback
    self.intercept = 0.0  # Default fallback
    self.fallback_to_median = fallback_to_median
    self.model_used = 'Default'

  def fit(self, X_input, y_input):
    X_flat = np.asarray(X_input).flatten()
    y_flat = np.asarray(y_input).flatten()

    # IMPROVEMENT 1: Explicitly handle empty inputs
    if X_flat.size == 0 or y_flat.size == 0:
      self.slope = 1.0
      self.intercept = 0.0
      self.model_used = 'Default_empty_input'
      return self

    # Handle single data point
    if X_flat.size == 1:
      self.slope = 1.0
      self.intercept = (
          y_flat[0] - X_flat[0]
      )  # Assume a slope of 1, adjust intercept
      if not np.isfinite(self.intercept):
        self.intercept = 0.0
      self.slope = np.clip(
          self.slope, EPSILON_PROPORTION, MAX_SLOPE_SCALER
      )  # Ensure positive and bounded slope
      self.model_used = 'Default_single_point'
      return self

    fit_successful = False

    # 1. Try LinearRegression with positive constraint
    try:
      lr_model = LinearRegression(positive=True)
      lr_model.fit(X_flat.reshape(-1, 1), y_flat)
      # IMPROVEMENT 1: Ensure parameters are finite and slope is positive
      if (
          np.isfinite(lr_model.coef_).all()
          and np.isfinite(lr_model.intercept_)
          and lr_model.coef_[0] >= EPSILON_PROPORTION
      ):
        self.slope = lr_model.coef_[0]
        self.intercept = lr_model.intercept_
        self.slope = np.clip(
            self.slope, EPSILON_PROPORTION, MAX_SLOPE_SCALER
        )  # Clip slope
        self.model_used = 'LinearRegression'
        fit_successful = True
      else:
        warnings.warn(
            'LinearRegression (positive=True) for RobustSimpleLinearScaler'
            ' produced unsuitable results.',
            RuntimeWarning,
        )
    except Exception as e:
      warnings.warn(
          'LinearRegression (positive=True) for RobustSimpleLinearScaler'
          f' failed: {e}.',
          RuntimeWarning,
      )

    # 2. If LinearRegression failed or unsuitable, try HuberRegressor
    if not fit_successful:
      try:
        huber_model = HuberRegressor(
            max_iter=1000, alpha=0.01, fit_intercept=True
        )  # IMPROVEMENT 1: Increased max_iter for Huber
        huber_model.fit(X_flat.reshape(-1, 1), y_flat)
        # IMPROVEMENT 1: Ensure parameters are finite and slope is positive
        if (
            np.isfinite(huber_model.coef_).all()
            and np.isfinite(huber_model.intercept_)
            and huber_model.coef_[0] >= EPSILON_PROPORTION
        ):
          self.slope = huber_model.coef_[0]
          self.intercept = huber_model.intercept_
          self.slope = np.clip(
              self.slope, EPSILON_PROPORTION, MAX_SLOPE_SCALER
          )  # Clip slope
          self.model_used = 'HuberRegressor'
          fit_successful = True
        else:
          warnings.warn(
              'HuberRegressor for RobustSimpleLinearScaler produced unsuitable'
              ' results.',
              RuntimeWarning,
          )
      except Exception as e:
        warnings.warn(
            f'HuberRegressor for RobustSimpleLinearScaler failed: {e}.',
            RuntimeWarning,
        )

    # 3. If both failed, use median-based fallback (if enabled)
    if not fit_successful and self.fallback_to_median:
      median_x = np.median(X_flat)
      median_y = np.median(y_flat)

      Q1_x, Q3_x = np.percentile(X_flat, [25, 75])
      Q1_y, Q3_y = np.percentile(y_flat, [25, 75])

      # IMPROVEMENT 1: Robust check for constant X values
      if (Q3_x - Q1_x) < EPSILON_PROPORTION:  # No variance in X values
        self.slope = 0.0  # Set slope to 0 if X is constant
        self.intercept = median_y
      elif (Q3_y - Q1_y) < EPSILON_PROPORTION:  # No variance in Y values
        self.slope = 0.0
        self.intercept = median_y
      else:  # Robust slope using IQR
        self.slope = (Q3_y - Q1_y) / ((Q3_x - Q1_x) + EPSILON_PROPORTION)
        self.intercept = median_y - self.slope * median_x

      self.slope = np.clip(
          self.slope, EPSILON_PROPORTION, MAX_SLOPE_SCALER
      )  # Ensure positive and bounded slope
      self.model_used = 'Median_Fallback'
      fit_successful = True

    # Ensure positive slope and finite values, even if fallback_to_median was off or failed
    # IMPROVEMENT 1: Final safety clip and finite check
    self.slope = np.maximum(
        self.slope, EPSILON_PROPORTION
    )  # Ensure slope is never too close to zero, or negative
    if not np.isfinite(self.slope):
      self.slope = 1.0
    if not np.isfinite(self.intercept):
      self.intercept = 0.0
    if not fit_successful:
      self.model_used = 'Default_no_fit'  # Fallback if no method succeeded
    return self

  def predict(self, X_input):
    X_flat = np.asarray(X_input).flatten()
    if X_flat.size == 0:
      return np.array([])
    return self.slope * X_flat + self.intercept


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a hybrid model by modelling train_x to train_y.

  Returns quantile predictions.
  """

  # Ensure date columns are datetime objects consistently
  for df in [train_x, test_x]:
    if not pd.api.types.is_datetime64_any_dtype(df['target_end_date']):
      df['target_end_date'] = pd.to_datetime(df['target_end_date'])
  if not pd.api.types.is_datetime64_any_dtype(test_x['reference_date']):
    test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  current_train_end_date = train_x['target_end_date'].max()

  # --- 1. Data Preprocessing and Augmentation (Strategy 2-like for ILI history) ---

  # Combine train_x and train_y
  train_df_actuals = pd.concat([train_x, train_y], axis=1)

  # Calculate admission proportions and their logit transforms robustly
  train_df_actuals['population_for_prop'] = train_df_actuals[
      'population'
  ].replace(
      0, 1
  )  # Safeguard against zero population
  train_df_actuals[ADMISSIONS_PROPORTION_COL] = (
      train_df_actuals[TARGET_STR] / train_df_actuals['population_for_prop']
  )
  train_df_actuals[LOGIT_ADMISSIONS_PROPORTION_COL] = logit_safe(
      train_df_actuals[ADMISSIONS_PROPORTION_COL]
  )

  # Prepare historical ILINet data (assuming 'ilinet_state' and 'locations' are globally available)
  ilinet_state_df_full = ilinet_state.copy()
  ilinet_state_df_full = ilinet_state_df_full[
      ilinet_state_df_full['region_type'] == 'States'
  ].copy()

  ilinet_state_df_full['target_end_date'] = pd.to_datetime(
      ilinet_state_df_full['week_start']
  ) + pd.Timedelta(days=6)

  # Filter ILINet data to be only up to the current training end date (NO DATA LEAKAGE)
  ilinet_state_df_filtered = ilinet_state_df_full[
      ilinet_state_df_full['target_end_date'] <= current_train_end_date
  ].copy()

  # Merge ILINet with locations to get FIPS codes and populations
  ilinet_state_df_filtered = ilinet_state_df_filtered.merge(
      locations[['location_name', 'location', 'population']],
      left_on='region',
      right_on='location_name',
      how='left',
  )
  ilinet_state_df_filtered.dropna(
      subset=['location', 'population'], inplace=True
  )
  ilinet_state_df_filtered['location'] = ilinet_state_df_filtered[
      'location'
  ].astype(int)

  # Calculate ILI rate, handling potential NaN/zeros
  ilinet_state_df_filtered[ILI_RATE_COL] = (
      ilinet_state_df_filtered['unweighted_ili'].fillna(0) / 100
  )
  ilinet_state_df_filtered[ILI_RATE_COL] = np.clip(
      ilinet_state_df_filtered[ILI_RATE_COL],
      EPSILON_PROPORTION,
      1 - EPSILON_PROPORTION,
  )
  ilinet_state_df_filtered[LOGIT_ILI_RATE_COL] = logit_safe(
      ilinet_state_df_filtered[ILI_RATE_COL]
  )

  # --- Learn Transformation (ILI Rate to Admissions Proportion) using overlap ---
  overlap_df = pd.merge(
      train_df_actuals[
          ['target_end_date', 'location', LOGIT_ADMISSIONS_PROPORTION_COL]
      ],
      ilinet_state_df_filtered[
          ['target_end_date', 'location', LOGIT_ILI_RATE_COL]
      ],
      on=['target_end_date', 'location'],
      how='inner',
  )
  overlap_df.dropna(
      subset=[LOGIT_ADMISSIONS_PROPORTION_COL, LOGIT_ILI_RATE_COL], inplace=True
  )

  # Initialize RobustSimpleLinearScaler which has internal robust fallback logic
  ili_to_admissions_scaler_model = RobustSimpleLinearScaler()

  # Fit the scaler. RobustSimpleLinearScaler handles cases where overlap_df is empty or contains sparse data.
  ili_to_admissions_scaler_model.fit(
      overlap_df[[LOGIT_ILI_RATE_COL]],
      overlap_df[LOGIT_ADMISSIONS_PROPORTION_COL],
  )

  # Create synthetic logit_admissions_proportion from historical ILINet
  synthetic_df = pd.DataFrame()
  valid_ili_rows = ilinet_state_df_filtered[
      ilinet_state_df_filtered[LOGIT_ILI_RATE_COL].notna()
  ].copy()
  if not valid_ili_rows.empty:
    synthetic_logit_preds = ili_to_admissions_scaler_model.predict(
        valid_ili_rows[[LOGIT_ILI_RATE_COL]]
    )

    valid_ili_rows[LOGIT_ADMISSIONS_PROPORTION_COL] = synthetic_logit_preds

    synthetic_df = valid_ili_rows[[
        'target_end_date',
        'location',
        'population',
        LOGIT_ADMISSIONS_PROPORTION_COL,
    ]].copy()
    synthetic_df['population_for_prop'] = synthetic_df['population'].replace(
        0, 1
    )  # Ensure non-zero for synthetic data
    synthetic_df[ADMISSIONS_PROPORTION_COL] = expit_safe(
        synthetic_df[LOGIT_ADMISSIONS_PROPORTION_COL]
    )
    synthetic_df[TARGET_STR] = (
        synthetic_df[ADMISSIONS_PROPORTION_COL]
        * synthetic_df['population_for_prop']
    )

  # Combine actual training data and synthetic historical data
  train_data_for_features = pd.concat(
      [train_df_actuals, synthetic_df], ignore_index=True
  )
  train_data_for_features.drop_duplicates(
      subset=['target_end_date', 'location'],
      keep='first',  # Keep actual values over synthetic for overlap dates
      inplace=True,
  )
  train_data_for_features.sort_values(
      by=['location', 'target_end_date'], inplace=True
  )
  train_data_for_features = train_data_for_features.reset_index(drop=True)

  # Determine overall median logit for fallbacks, now based on combined history for current training period
  overall_mean_logit = (
      train_data_for_features[LOGIT_ADMISSIONS_PROPORTION_COL].median()
      if not train_data_for_features.empty
      else logit_safe(EPSILON_PROPORTION)
  )
  if not np.isfinite(overall_mean_logit):
    overall_mean_logit = logit_safe(EPSILON_PROPORTION)

  # --- 2. Feature Engineering on `train_data_for_features` ---
  train_data_for_features['epiweek'] = (
      train_data_for_features['target_end_date']
      .dt.isocalendar()
      .week.astype(int)
  )
  train_data_for_features['year'] = train_data_for_features[
      'target_end_date'
  ].dt.year
  train_data_for_features['epiweek_sin'] = np.sin(
      2 * np.pi * train_data_for_features['epiweek'] / 52
  )
  train_data_for_features['epiweek_cos'] = np.cos(
      2 * np.pi * train_data_for_features['epiweek'] / 52
  )
  train_data_for_features['epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * train_data_for_features['epiweek'] / 52
  )
  train_data_for_features['epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * train_data_for_features['epiweek'] / 52
  )

  train_data_for_features['population_scaled'] = np.log1p(
      train_data_for_features['population']
  )
  train_data_for_features['season_year_start'] = train_data_for_features[
      'year'
  ] - (train_data_for_features['epiweek'] < 40).astype(int)

  # Horizon features for training data (always 0 for observed past data)
  train_data_for_features['horizon'] = 0

  # Reference date features for training data (assumed to be target_end_date for historical observations)
  train_data_for_features['reference_date'] = train_data_for_features[
      'target_end_date'
  ]
  train_data_for_features['ref_date_epiweek'] = (
      train_data_for_features['reference_date']
      .dt.isocalendar()
      .week.astype(int)
  )
  train_data_for_features['ref_date_epiweek_sin'] = np.sin(
      2 * np.pi * train_data_for_features['ref_date_epiweek'] / 52
  )
  train_data_for_features['ref_date_epiweek_cos'] = np.cos(
      2 * np.pi * train_data_for_features['ref_date_epiweek'] / 52
  )
  train_data_for_features['ref_date_epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * train_data_for_features['ref_date_epiweek'] / 52
  )
  train_data_for_features['ref_date_epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * train_data_for_features['ref_date_epiweek'] / 52
  )

  # --- Smoothed Global Epiweek Trend (Fourier Series based STATISTICAL_SEASONAL_BASELINE) ---
  # This acts as a statistical approximation of a "typical seasonal curve"
  weekly_medians = (
      train_data_for_features.groupby('epiweek')[
          LOGIT_ADMISSIONS_PROPORTION_COL
      ]
      .median()
      .reset_index()
  )
  weekly_medians['epiweek_sin'] = np.sin(
      2 * np.pi * weekly_medians['epiweek'] / 52
  )
  weekly_medians['epiweek_cos'] = np.cos(
      2 * np.pi * weekly_medians['epiweek'] / 52
  )
  weekly_medians['epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * weekly_medians['epiweek'] / 52
  )
  weekly_medians['epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * weekly_medians['epiweek'] / 52
  )

  # Ensure to drop NaNs before fitting, and handle if weekly_medians becomes empty
  weekly_medians.dropna(subset=[LOGIT_ADMISSIONS_PROPORTION_COL], inplace=True)

  global_seasonal_features_for_smoothing = [
      'epiweek_sin',
      'epiweek_cos',
      'epiweek_sin_2',
      'epiweek_cos_2',
  ]
  smoothed_global_trend_model = LinearRegression()

  if len(weekly_medians) >= (
      len(global_seasonal_features_for_smoothing) + 1
  ):  # Enough data points for meaningful regression (features + intercept)
    try:
      smoothed_global_trend_model.fit(
          weekly_medians[global_seasonal_features_for_smoothing],
          weekly_medians[LOGIT_ADMISSIONS_PROPORTION_COL],
      )
      # Apply smoothed trend to all training data
      train_data_for_features[STATISTICAL_SEASONAL_BASELINE] = (
          smoothed_global_trend_model.predict(
              train_data_for_features[global_seasonal_features_for_smoothing]
          )
      )
    except Exception as e:
      warnings.warn(
          f'LinearRegression for smoothed global trend failed: {e}. Falling'
          ' back to raw median.',
          RuntimeWarning,
      )
      # Fallback to raw median if regression fails
      raw_global_epiweek_mean_logit_map = weekly_medians.set_index('epiweek')[
          LOGIT_ADMISSIONS_PROPORTION_COL
      ]
      train_data_for_features[STATISTICAL_SEASONAL_BASELINE] = (
          train_data_for_features['epiweek']
          .map(raw_global_epiweek_mean_logit_map)
          .fillna(overall_mean_logit)
      )
  else:
    warnings.warn(
        'Not enough weekly median data to fit smoothed global trend model.'
        ' Falling back to raw median.',
        RuntimeWarning,
    )
    # Fallback to raw median if insufficient data
    raw_global_epiweek_mean_logit_map = (
        weekly_medians.set_index('epiweek')[LOGIT_ADMISSIONS_PROPORTION_COL]
        if not weekly_medians.empty
        else pd.Series(dtype=float)
    )
    train_data_for_features[STATISTICAL_SEASONAL_BASELINE] = (
        train_data_for_features['epiweek']
        .map(raw_global_epiweek_mean_logit_map)
        .fillna(overall_mean_logit)
    )

  # Fill any remaining NaNs (e.g., if epiweek was missing or outside training range) with overall median
  train_data_for_features[STATISTICAL_SEASONAL_BASELINE].fillna(
      overall_mean_logit, inplace=True
  )

  # Weeks from global peak proxy (based on the smoothed global trend)
  # Re-calculate peak_epiweek from the smoothed trend model
  all_epiweeks_df = pd.DataFrame({'epiweek': range(1, 53)})
  all_epiweeks_df['epiweek_sin'] = np.sin(
      2 * np.pi * all_epiweeks_df['epiweek'] / 52
  )
  all_epiweeks_df['epiweek_cos'] = np.cos(
      2 * np.pi * all_epiweeks_df['epiweek'] / 52
  )
  all_epiweeks_df['epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * all_epiweeks_df['epiweek'] / 52
  )
  all_epiweeks_df['epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * all_epiweeks_df['epiweek'] / 52
  )

  peak_epiweek = 1  # Default fallback
  # Check if smoothed_global_trend_model was successfully fitted before using it
  if hasattr(smoothed_global_trend_model, 'coef_') and len(weekly_medians) >= (
      len(global_seasonal_features_for_smoothing) + 1
  ):
    smoothed_global_trend_preds = smoothed_global_trend_model.predict(
        all_epiweeks_df[global_seasonal_features_for_smoothing]
    )
    if smoothed_global_trend_preds.size > 0:
      peak_epiweek = all_epiweeks_df['epiweek'].iloc[
          np.argmax(smoothed_global_trend_preds)
      ]
  else:
    # Fallback to fixed peak or raw median if smoothing failed
    if (
        not weekly_medians.empty
    ):  # Use raw median if available and smoothing failed
      peak_epiweek = weekly_medians.set_index('epiweek')[
          LOGIT_ADMISSIONS_PROPORTION_COL
      ].idxmax()
    # If weekly_medians is also empty, peak_epiweek remains 1 (default)
  if not np.isfinite(peak_epiweek):
    peak_epiweek = 1

  train_data_for_features['weeks_from_peak_proxy'] = np.abs(
      ((train_data_for_features['epiweek'] - peak_epiweek + 26 + 52) % 52) - 26
  )

  # --- REFINED HYBRID DISCREPANCY MODELING (Code 1 Style) ---

  # 1. Calculate deviation from the statistical seasonal baseline
  train_data_for_features['deviation_from_baseline'] = (
      train_data_for_features[LOGIT_ADMISSIONS_PROPORTION_COL]
      - train_data_for_features[STATISTICAL_SEASONAL_BASELINE]
  )

  # 2. Calculate global median of these deviations for each week
  # Reindex for robustness, ensuring all `all_dates_in_train_idx` will have an entry, with NaN for missing weeks.
  all_dates_in_train_idx = (
      pd.Series(index=train_data_for_features['target_end_date'].unique())
      .sort_index()
      .index
  )
  weekly_global_median_deviations = (
      train_data_for_features.groupby('target_end_date')[
          'deviation_from_baseline'
      ]
      .median()
      .reindex(all_dates_in_train_idx)
  )

  # 3. Calculate the EWMA of these global median deviations, ignoring NaNs, then shift by 1 week (mu_t_proxy)
  # The ignore_na=True parameter ensures EWMA is calculated only on observed values, carrying forward the last valid EWMA.
  global_ewma_series = (
      weekly_global_median_deviations.ewm(
          alpha=EWMA_ALPHA, adjust=False, ignore_na=True
      )
      .mean()
      .shift(1)
      .fillna(0)
  )

  # Merge this lagged EWMA back to the train_data_for_features as a feature for training
  train_data_for_features = pd.merge(
      train_data_for_features,
      global_ewma_series.rename(GLOBAL_EWMA_DISCREPANCY_COL),
      left_on='target_end_date',
      right_index=True,
      how='left',
  )
  train_data_for_features[GLOBAL_EWMA_DISCREPANCY_COL].fillna(0, inplace=True)

  # 4. Calculate season-specific residual (delta_j,t)
  train_data_for_features[RESIDUAL_FROM_TREND_COL] = (
      train_data_for_features['deviation_from_baseline']
      - train_data_for_features[GLOBAL_EWMA_DISCREPANCY_COL]
  )

  # Lagged features for logit_admissions_proportion
  lags = [1, 2, 3, 4]
  for lag in lags:
    train_data_for_features[f'lag_{lag}'] = train_data_for_features.groupby(
        'location'
    )[LOGIT_ADMISSIONS_PROPORTION_COL].shift(lag)

  # Weekly change (lag1 - lag2)
  train_data_for_features['weekly_change_lag1'] = (
      train_data_for_features['lag_1'] - train_data_for_features['lag_2']
  )
  train_data_for_features['weekly_change_lag1'].fillna(0.0, inplace=True)

  # Lagged season-specific residuals
  lagged_residuals = [1, 2]
  for lag in lagged_residuals:
    train_data_for_features[f'lag_{RESIDUAL_FROM_TREND_COL}_{lag}'] = (
        train_data_for_features.groupby('location')[
            RESIDUAL_FROM_TREND_COL
        ].shift(lag)
    )
  train_data_for_features[f'lag_{RESIDUAL_FROM_TREND_COL}_1'].fillna(
      0.0, inplace=True
  )
  train_data_for_features[f'lag_{RESIDUAL_FROM_TREND_COL}_2'].fillna(
      0.0, inplace=True
  )

  # Define features and target for the models
  model_features = (
      [
          'population_scaled',
          'epiweek_sin',
          'epiweek_cos',
          'epiweek_sin_2',
          'epiweek_cos_2',
          'year',
          'season_year_start',
          STATISTICAL_SEASONAL_BASELINE,
          'weeks_from_peak_proxy',
          'weekly_change_lag1',
          GLOBAL_EWMA_DISCREPANCY_COL,
          'horizon',
          'ref_date_epiweek_sin',
          'ref_date_epiweek_cos',
          'ref_date_epiweek_sin_2',
          'ref_date_epiweek_cos_2',
      ]
      + [f'lag_{lag}' for lag in lags]
      + [f'lag_{RESIDUAL_FROM_TREND_COL}_{lag}' for lag in lagged_residuals]
  )
  model_target = LOGIT_ADMISSIONS_PROPORTION_COL

  # Filter for training data with all required features and target
  train_for_model = train_data_for_features.dropna(
      subset=model_features + [model_target]
  ).copy()

  X_train = train_for_model[model_features]
  y_train = train_for_model[model_target]

  # Handle cases where training data is insufficient
  if X_train.empty or len(X_train) < 2:
    # Create a dummy DataFrame with the correct index and columns
    dummy_preds = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    for q_col in dummy_preds.columns:
      dummy_preds[q_col] = MIN_PREDICTED_ADMISSIONS
    return dummy_preds

  # --- 3. Model Training (Hybrid: HGB for Mean, HGB for Std Dev, plus historical std) ---

  hgb_mean_model = HistGradientBoostingRegressor(
      loss='squared_error',
      random_state=42,
      max_iter=450,  # IMPROVEMENT 4: INCREASED from 350
      learning_rate=0.05,
      max_depth=5,
      verbose=0,
  )
  hgb_mean_model.fit(X_train, y_train)

  train_predictions = hgb_mean_model.predict(X_train)
  residuals = y_train - train_predictions

  # Target for STD model: squared residuals (variance)
  y_train_var = residuals**2

  # IMPROVEMENT 3: Define richer features for the standard deviation model
  std_model_features = [
      'population_scaled',
      'epiweek_sin',
      'epiweek_cos',
      'epiweek_sin_2',
      'epiweek_cos_2',  # Added 2nd harmonics
      'year',
      'season_year_start',
      STATISTICAL_SEASONAL_BASELINE,
      'weeks_from_peak_proxy',
      GLOBAL_EWMA_DISCREPANCY_COL,
      'horizon',
      'lag_1',
      'lag_2',  # Added lagged target to std_model_features
      'weekly_change_lag1',  # Added weekly change to std_model_features
      f'lag_{RESIDUAL_FROM_TREND_COL}_1',
      f'lag_{RESIDUAL_FROM_TREND_COL}_2',  # Added lagged residual to std_model_features
      'ref_date_epiweek_sin',
      'ref_date_epiweek_cos',
      'ref_date_epiweek_sin_2',
      'ref_date_epiweek_cos_2',  # Added ref_date features (including 2nd harmonics)
      (  # The mean prediction itself might correlate with residual std
          'mean_prediction'
      ),
  ]
  # IMPROVEMENT 3: Prepare X_train for std model by correctly selecting specified features from train_for_model and adding mean_prediction
  X_train_std = train_for_model[
      [f for f in std_model_features if f != 'mean_prediction']
  ].copy()
  X_train_std['mean_prediction'] = train_predictions

  hgb_std_model = None
  # Only train std model if there's enough variance and data points (e.g., at least 20 samples)
  if (
      len(X_train_std) >= 20 and y_train_var.var() > 1e-10
  ):  # Ensure enough data and variance to fit a meaningful std model
    try:
      hgb_std_model = HistGradientBoostingRegressor(
          loss='squared_error',  # Predict squared residuals (variance) directly
          random_state=42,
          max_iter=350,  # IMPROVEMENT 4: INCREASED from 250
          learning_rate=0.05,
          max_depth=3,
          verbose=0,
      )
      hgb_std_model.fit(X_train_std, y_train_var)
    except Exception as e:
      warnings.warn(
          f'HistGradientBoostingRegressor for variance failed: {e}. Falling'
          ' back to global std dev.',
          RuntimeWarning,
      )

  global_logit_std = y_train.std() if not y_train.empty else MIN_STD_DEV_LOGIT
  if not np.isfinite(global_logit_std) or global_logit_std < MIN_STD_DEV_LOGIT:
    global_logit_std = MIN_STD_DEV_LOGIT

  location_logit_stds = (
      train_for_model.groupby('location')[LOGIT_ADMISSIONS_PROPORTION_COL]
      .std()
      .fillna(global_logit_std)
  )
  location_logit_stds = location_logit_stds.apply(
      lambda x: np.maximum(x, MIN_STD_DEV_LOGIT)
  )

  # IMPROVEMENT 6: Pre-calculate location-epiweek specific medians/means for robust lag imputation
  location_epiweek_median_logit_map = (
      train_data_for_features.groupby(['location', 'epiweek'])[
          LOGIT_ADMISSIONS_PROPORTION_COL
      ]
      .median()
      .unstack('epiweek')
  )
  # Use ffill/bfill to fill missing epiweeks within a location, then global median for completely missing locations
  location_epiweek_median_logit_map = (
      location_epiweek_median_logit_map.ffill(axis=1)
      .bfill(axis=1)
      .fillna(overall_mean_logit)
  )

  location_epiweek_mean_residual_map = (
      train_data_for_features.groupby(['location', 'epiweek'])[
          RESIDUAL_FROM_TREND_COL
      ]
      .mean()
      .unstack('epiweek')
  )
  # Use ffill/bfill to fill missing epiweeks within a location, then 0.0 for completely missing locations
  location_epiweek_mean_residual_map = (
      location_epiweek_mean_residual_map.ffill(axis=1).bfill(axis=1).fillna(0.0)
  )

  # --- 4. Forecasting for `test_x` (Recursive Loop) ---

  forecast_df = test_x.copy()  # Preserves original index of test_x
  ref_date = forecast_df['reference_date'].iloc[
      0
  ]  # The single reference_date for this test_x batch

  forecast_df['epiweek'] = (
      forecast_df['target_end_date'].dt.isocalendar().week.astype(int)
  )
  forecast_df['year'] = forecast_df['target_end_date'].dt.year
  forecast_df['epiweek_sin'] = np.sin(2 * np.pi * forecast_df['epiweek'] / 52)
  forecast_df['epiweek_cos'] = np.cos(2 * np.pi * forecast_df['epiweek'] / 52)
  forecast_df['epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * forecast_df['epiweek'] / 52
  )
  forecast_df['epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * forecast_df['epiweek'] / 52
  )

  forecast_df['population_scaled'] = np.log1p(forecast_df['population'])
  forecast_df['season_year_start'] = forecast_df['year'] - (
      forecast_df['epiweek'] < 40
  ).astype(int)

  # Reference date features for test_x
  forecast_df['ref_date_epiweek'] = (
      forecast_df['reference_date'].dt.isocalendar().week.astype(int)
  )
  forecast_df['ref_date_epiweek_sin'] = np.sin(
      2 * np.pi * forecast_df['ref_date_epiweek'] / 52
  )
  forecast_df['ref_date_epiweek_cos'] = np.cos(
      2 * np.pi * forecast_df['ref_date_epiweek'] / 52
  )
  forecast_df['ref_date_epiweek_sin_2'] = np.sin(
      2 * np.pi * 2 * forecast_df['ref_date_epiweek'] / 52
  )
  forecast_df['ref_date_epiweek_cos_2'] = np.cos(
      2 * np.pi * 2 * forecast_df['ref_date_epiweek'] / 52
  )

  # Predict STATISTICAL_SEASONAL_BASELINE for test_x using the trained smoothed_global_trend_model
  if hasattr(smoothed_global_trend_model, 'coef_') and len(weekly_medians) >= (
      len(global_seasonal_features_for_smoothing) + 1
  ):
    forecast_df[STATISTICAL_SEASONAL_BASELINE] = (
        smoothed_global_trend_model.predict(
            forecast_df[global_seasonal_features_for_smoothing]
        )
    )
  else:  # Fallback to raw median if smoothing failed or insufficient data
    raw_global_epiweek_map_for_test = (
        weekly_medians.set_index('epiweek')[LOGIT_ADMISSIONS_PROPORTION_COL]
        if not weekly_medians.empty
        else pd.Series(dtype=float)
    )
    forecast_df[STATISTICAL_SEASONAL_BASELINE] = (
        forecast_df['epiweek']
        .map(raw_global_epiweek_map_for_test)
        .fillna(overall_mean_logit)
    )

  # Ensure STATISTICAL_SEASONAL_BASELINE is finite
  forecast_df[STATISTICAL_SEASONAL_BASELINE] = np.nan_to_num(
      forecast_df[STATISTICAL_SEASONAL_BASELINE], nan=overall_mean_logit
  )

  # Weeks from global peak proxy (from smoothed trend, re-use existing peak_epiweek)
  forecast_df['weeks_from_peak_proxy'] = np.abs(
      ((forecast_df['epiweek'] - peak_epiweek + 26 + 52) % 52) - 26
  )

  all_quantile_preds = []

  # Initialize state for recursive forecasting (lags and residuals) for this single reference date
  current_state_for_ref_date = {}

  # IMPROVEMENT 5: Initialize current_global_ewma_value based on the *last valid* EWMA value from training data.
  initial_global_ewma = 0.0
  if not global_ewma_series.empty:
    # Use ffill to ensure a non-NaN value, then take the last one
    last_valid_ewma_series = global_ewma_series.ffill()
    if not last_valid_ewma_series.empty:
      last_ewma_val = last_valid_ewma_series.iloc[-1]
      if np.isfinite(last_ewma_val):
        initial_global_ewma = last_ewma_val

  current_global_ewma_value = initial_global_ewma

  # Determine maximum lags required for initialization
  max_lags = max(lags) if lags else 0
  max_residual_lags = max(lagged_residuals) if lagged_residuals else 0
  max_all_lags = max(max_lags, max_residual_lags)

  # Populate initial lags for each location using historical data up to current_train_end_date
  for loc in forecast_df['location'].unique():
    loc_lags_history = []
    loc_residual_lags_history = []

    # Iterate backwards from reference_date to populate lags
    for lag_offset in range(1, max_all_lags + 1):
      prev_week_date = ref_date - pd.Timedelta(weeks=lag_offset)
      prev_week_data = train_data_for_features[
          (train_data_for_features['location'] == loc)
          & (train_data_for_features['target_end_date'] == prev_week_date)
      ]

      if not prev_week_data.empty:
        if lag_offset <= max_lags:
          loc_lags_history.append(
              prev_week_data[LOGIT_ADMISSIONS_PROPORTION_COL].iloc[0]
          )
        if lag_offset <= max_residual_lags:
          loc_residual_lags_history.append(
              prev_week_data[RESIDUAL_FROM_TREND_COL].iloc[0]
          )
      else:  # IMPROVEMENT 6: Fallback to location-epiweek specific median/mean if no exact history
        prev_week_epiweek = (
            pd.to_datetime(prev_week_date).isocalendar().week.astype(int)
        )

        # Fallback for LOGIT_ADMISSIONS_PROPORTION_COL from robust map
        loc_epiweek_median_logit = (
            location_epiweek_median_logit_map.loc[loc, prev_week_epiweek]
            if loc in location_epiweek_median_logit_map.index
            else overall_mean_logit
        )
        if not np.isfinite(loc_epiweek_median_logit):
          loc_epiweek_median_logit = overall_mean_logit
        if lag_offset <= max_lags:
          loc_lags_history.append(loc_epiweek_median_logit)

        # Fallback for RESIDUAL_FROM_TREND_COL from robust map
        loc_epiweek_mean_residual = (
            location_epiweek_mean_residual_map.loc[loc, prev_week_epiweek]
            if loc in location_epiweek_mean_residual_map.index
            else 0.0
        )
        if not np.isfinite(loc_epiweek_mean_residual):
          loc_epiweek_mean_residual = 0.0
        if lag_offset <= max_residual_lags:
          loc_residual_lags_history.append(loc_epiweek_mean_residual)

    # Ensure enough elements for all lags, padding with fallbacks if history is too short
    loc_lags_padded = loc_lags_history + [overall_mean_logit] * max(
        0, max_lags - len(loc_lags_history)
    )
    loc_residual_lags_padded = loc_residual_lags_history + [0.0] * max(
        0, max_residual_lags - len(loc_residual_lags_history)
    )

    current_state_for_ref_date[loc] = {
        'lags': loc_lags_padded[:max_lags],
        'weekly_change_lag1': (
            (loc_lags_padded[0] - loc_lags_padded[1])
            if len(loc_lags_padded) >= 2
            else 0.0
        ),
        'residual_lags': loc_residual_lags_padded[:max_residual_lags],
    }

  # Loop through horizons for the *single* reference_date represented by test_x
  for h in sorted(forecast_df['horizon'].unique()):
    current_horizon_rows = forecast_df[
        (forecast_df['reference_date'] == ref_date)
        & (forecast_df['horizon'] == h)
    ].copy()

    if current_horizon_rows.empty:
      continue

    # Populate features for the current horizon for each row
    for idx, row in current_horizon_rows.iterrows():
      loc = row['location']

      current_horizon_rows.loc[idx, 'horizon'] = h

      # The STATISTICAL_SEASONAL_BASELINE is already pre-calculated for all forecast_df rows above

      # Retrieve location-specific state for lags and residuals, with default if location not seen
      loc_state = current_state_for_ref_date.get(
          loc,
          {
              'lags': [overall_mean_logit] * max_lags,
              'weekly_change_lag1': 0.0,
              'residual_lags': [0.0] * max_residual_lags,
          },
      )

      loc_lags_values = loc_state['lags']
      for i_lag, lag_val_num in enumerate(lags):
        if i_lag < len(loc_lags_values):
          current_horizon_rows.loc[idx, f'lag_{lag_val_num}'] = loc_lags_values[
              i_lag
          ]
        else:  # Fallback if lag history is shorter than expected (should be covered by padding above but good for safety)
          current_horizon_rows.loc[idx, f'lag_{lag_val_num}'] = (
              overall_mean_logit
          )

      current_horizon_rows.loc[idx, 'weekly_change_lag1'] = loc_state[
          'weekly_change_lag1'
      ]

      loc_residual_lags_values = loc_state['residual_lags']
      for i_lag, lag_val_num in enumerate(lagged_residuals):
        if i_lag < len(loc_residual_lags_values):
          current_horizon_rows.loc[
              idx, f'lag_{RESIDUAL_FROM_TREND_COL}_{lag_val_num}'
          ] = loc_residual_lags_values[i_lag]
        else:  # Fallback if residual lag history is shorter than expected
          current_horizon_rows.loc[
              idx, f'lag_{RESIDUAL_FROM_TREND_COL}_{lag_val_num}'
          ] = 0.0

      # Populate the dynamic EWMA feature for each row using the current state
      current_horizon_rows.loc[idx, GLOBAL_EWMA_DISCREPANCY_COL] = (
          current_global_ewma_value
      )

    # Make predictions for the current horizon
    X_predict_mean = current_horizon_rows[model_features]
    logit_mean_preds = hgb_mean_model.predict(X_predict_mean)

    # Determine std_preds (now based on predicted variance)
    std_preds = np.full_like(
        logit_mean_preds, global_logit_std
    )  # Initialize with global std dev

    if hgb_std_model:
      # IMPROVEMENT 3: Prepare X_predict for std model using the specified std_model_features
      X_predict_std = current_horizon_rows[
          [f for f in std_model_features if f != 'mean_prediction']
      ].copy()
      X_predict_std['mean_prediction'] = (
          logit_mean_preds  # Add current mean prediction
      )

      # Predict variance, ensure non-negative, then take sqrt for standard deviation
      predicted_variance = np.maximum(0, hgb_std_model.predict(X_predict_std))
      hgb_std_raw_preds = np.sqrt(predicted_variance)

      blended_std_preds = []
      for i, idx in enumerate(current_horizon_rows.index):
        loc = current_horizon_rows.loc[idx, 'location']
        loc_hist_std = location_logit_stds.get(loc, global_logit_std)

        hgb_std_val = np.maximum(hgb_std_raw_preds[i], MIN_STD_DEV_LOGIT)
        # Blending HGB-predicted std with historical location std (e.g., 80% HGB, 20% historical)
        blended_std = (hgb_std_val * 0.8) + (loc_hist_std * 0.2)
        blended_std_preds.append(np.maximum(blended_std, MIN_STD_DEV_LOGIT))
      std_preds = np.array(blended_std_preds)
    else:
      std_preds = np.array([
          location_logit_stds.get(loc, global_logit_std)
          for loc in current_horizon_rows['location']
      ])
      std_preds = np.maximum(std_preds, MIN_STD_DEV_LOGIT)

    # IMPROVEMENT 2: Apply an adaptive overdispersion factor to the standard deviation directly
    # This scales uncertainty based on horizon, assuming uncertainty grows with forecast lead time.
    overdispersion_factor = OVERDISPERSION_BASE_FACTOR + (
        OVERDISPERSION_PER_HORIZON_INCREMENT * h
    )
    std_preds = std_preds * overdispersion_factor
    std_preds = np.maximum(
        std_preds, MIN_STD_DEV_LOGIT
    )  # Re-ensure minimum std after scaling

    quantile_predictions_for_horizon = pd.DataFrame(
        index=current_horizon_rows.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )

    predicted_deviation_from_baseline_h = []

    for i, idx in enumerate(current_horizon_rows.index):
      mean_val = logit_mean_preds[i]  # Use the predicted mean directly
      std_val = std_preds[i]
      population_val = current_horizon_rows.loc[idx, 'population']

      # Generate samples from Normal distribution in logit space
      logit_samples = np.random.normal(
          loc=mean_val, scale=std_val, size=N_SAMPLES
      )  # IMPROVEMENT 7: N_SAMPLES increased

      # Convert logit samples directly to proportions, then to admissions counts
      proportion_samples = expit_safe(logit_samples)
      admissions_samples = proportion_samples * (
          population_val if population_val > 0 else 1
      )
      admissions_samples = np.maximum(
          admissions_samples, MIN_PREDICTED_ADMISSIONS
      )  # Ensure non-negative

      for q_idx, q_level in enumerate(QUANTILES):
        quantile_predictions_for_horizon.loc[idx, f'quantile_{q_level}'] = (
            np.percentile(admissions_samples, q_level * 100)
        )

      # Update state for recursive forecasting
      # Use MEAN prediction for the current week to update lags for next week (consistency with HGB)
      population_for_prop_val = population_val if population_val > 0 else 1
      proportion = np.clip(
          expit_safe(mean_val), EPSILON_PROPORTION, 1 - EPSILON_PROPORTION
      )  # Use mean_val
      mean_logit_pred_for_recursive_update = logit(proportion)

      current_baseline_trend = current_horizon_rows.loc[
          idx, STATISTICAL_SEASONAL_BASELINE
      ]
      current_local_ewma_state_for_loc = current_horizon_rows.loc[
          idx, GLOBAL_EWMA_DISCREPANCY_COL
      ]

      # For updating global EWMA
      current_predicted_deviation_from_baseline = (
          mean_logit_pred_for_recursive_update - current_baseline_trend
      )
      predicted_deviation_from_baseline_h.append(
          current_predicted_deviation_from_baseline
      )

      # For updating local residual lags
      current_local_residual_pred = (
          mean_logit_pred_for_recursive_update
          - current_baseline_trend
          - current_local_ewma_state_for_loc
      )

      loc = current_horizon_rows.loc[idx, 'location']
      loc_state = current_state_for_ref_date.get(
          loc,
          {  # Get current state or initialize with defaults
              'lags': [overall_mean_logit] * max_lags,
              'weekly_change_lag1': 0.0,
              'residual_lags': [0.0] * max_residual_lags,
          },
      )

      new_lags = [mean_logit_pred_for_recursive_update] + loc_state['lags'][:-1]
      new_weekly_change = (
          new_lags[0] - new_lags[1] if len(new_lags) >= 2 else 0.0
      )
      new_residual_lags = [current_local_residual_pred] + loc_state[
          'residual_lags'
      ][:-1]

      current_state_for_ref_date[loc] = {
          'lags': new_lags,
          'weekly_change_lag1': new_weekly_change,
          'residual_lags': new_residual_lags,
      }

    all_quantile_preds.append(quantile_predictions_for_horizon)

    # Update global EWMA for the next horizon with robustness checks
    if (
        predicted_deviation_from_baseline_h
    ):  # Check if there were any predictions to update from
      global_median_predicted_deviation_h = np.median(
          predicted_deviation_from_baseline_h
      )

      # Ensure values are finite before updating EWMA
      if np.isfinite(global_median_predicted_deviation_h):
        current_global_ewma_value = (
            EWMA_ALPHA * global_median_predicted_deviation_h
            + (1 - EWMA_ALPHA) * current_global_ewma_value
        )
      else:
        warnings.warn(
            'Non-finite median predicted deviation for EWMA update at horizon'
            f' {h}, retaining previous EWMA value.',
            RuntimeWarning,
        )

      if not np.isfinite(current_global_ewma_value):
        warnings.warn(
            f'Non-finite EWMA value encountered after update at horizon {h},'
            ' resetting to 0.',
            RuntimeWarning,
        )
        current_global_ewma_value = 0.0
    else:
      warnings.warn(
          f'No predictions made for horizon {h} to update global EWMA, current'
          ' EWMA value retained.',
          RuntimeWarning,
      )

  if not all_quantile_preds:
    dummy_preds = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    for q_col in dummy_preds.columns:
      dummy_preds[q_col] = MIN_PREDICTED_ADMISSIONS
    return dummy_preds

  # Combine all forecast dataframes and ensure the index matches test_x
  final_predictions_df_raw = pd.concat(all_quantile_preds)

  # Reindex to match the original test_x index precisely.
  # This is crucial for satisfying the problem's output requirement.
  final_predictions_df = final_predictions_df_raw.reindex(test_x.index)

  final_predictions_df = final_predictions_df.apply(
      lambda x: np.floor(x).astype(int)
  )
  final_predictions_df[final_predictions_df < MIN_PREDICTED_ADMISSIONS] = (
      MIN_PREDICTED_ADMISSIONS
  )

  # Enforce monotonicity across quantiles
  for col_idx in range(1, len(QUANTILES)):
    prev_col = f'quantile_{QUANTILES[col_idx-1]}'
    curr_col = f'quantile_{QUANTILES[col_idx]}'
    final_predictions_df[curr_col] = np.maximum(
        final_predictions_df[curr_col], final_predictions_df[prev_col]
    )

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
