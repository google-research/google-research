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
MODEL_NAME = 'Google_SAI-Adapted_8'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import nbinom
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from typing import Any, Dict


# --- Configuration Constants ---


def _create_fold_scaffold(
    reference_date,
    horizons,
    location_codes,
    locations_df,
):
  """Creates the feature scaffold for a single forecast date."""
  all_combinations = pd.MultiIndex.from_product(
      [[reference_date], horizons, location_codes],
      names=['reference_date', 'horizon', 'location'],
  )
  scaffold = pd.DataFrame(index=all_combinations).reset_index()
  scaffold['target_end_date'] = scaffold.apply(
      lambda row: row['reference_date'] + pd.Timedelta(weeks=row['horizon']),
      axis=1,
  )
  return scaffold.merge(locations_df, on='location', how='left')


np.seterr(over='raise')


# Core principles of the method described:
# 1. Autoregressive/Branching Dynamics: The model explicitly uses lagged target values as features.
#    Crucially, in the forecasting phase, it recursively feeds its own median predictions from
#    earlier horizons back into the feature set for later horizons, directly simulating a
#    step-by-step branching process.
# 2. Time-Varying Transmission/Growth: Cyclical `week_of_year` features (sin, cos) capture strong seasonality.
#    The inclusion of `month` and `year` features, along with newly added explicit growth rate features
#    (both raw and log-transformed), allows the XGBoost model to implicitly learn non-linear and longer-term
#    patterns in growth rates, resembling variations in R_t and directly modeling epidemic trajectory.
# 3. Integration of Related Epidemic Signals: ILINet data is integrated as an additional,
#    leading epidemic indicator. A location-specific (with global fallback) linear transformation
#    (now applied to log1p-transformed values for robustness) maps historical ILINet data to the
#    scale of hospital admissions, creating an extended synthetic history that provides a longer,
#    mechanistically-linked view of disease activity. This augmented history, along with direct
#    ILI features and ILI growth features, helps inform the branching process.
# 4. Probabilistic Forecasts for Count Data: Forecasts are generated as quantiles of a Negative
#    Binomial distribution. The mean (mu) is predicted by the XGBoost model, and a robustly
#    estimated dispersion parameter (alpha) characterizes the uncertainty appropriate for
#    overdispersed count data. Quantiles are then derived from this distribution, ensuring
#    non-negativity and monotonicity. The alpha estimation is now more robustly calculated by
#    using Maximum Likelihood Estimation (MLE) on informative observations and clamping within
#    a more appropriate range.


def _add_features(df):
  """Generates time-series features for the given DataFrame.

  Assumes 'target_end_date', 'location', 'population', 'Total Influenza
  Admissions', 'admissions_per_capita', 'log1p_admissions', 'unweighted_ili',
  'ili_per_capita', 'log1p_ili' are already present and properly
  imputed/transformed.
  """
  df_copy = df.copy()

  # Time features (Enhanced cyclical features and Year)
  df_copy['week_of_year'] = (
      df_copy['target_end_date'].dt.isocalendar().week.astype(int)
  )
  df_copy['sin_week_of_year'] = np.sin(2 * np.pi * df_copy['week_of_year'] / 52)
  df_copy['cos_week_of_year'] = np.cos(2 * np.pi * df_copy['week_of_year'] / 52)
  df_copy['month'] = df_copy['target_end_date'].dt.month
  df_copy['year'] = df_copy['target_end_date'].dt.year

  # Lagged features (autoregressive on actual/synthetic target AND per capita)
  lags = [
      1,
      2,
      3,
      4,
      8,
      12,
      16,
      20,
      24,
      28,
      52,
  ]  # Extended lags for seasonality
  for lag in lags:
    df_copy[f'lag_{lag}'] = df_copy.groupby('location')[TARGET_STR].shift(lag)
    df_copy[f'log1p_lag_{lag}'] = df_copy.groupby('location')[
        'log1p_admissions'
    ].shift(lag)
    df_copy[f'lag_per_capita_{lag}'] = df_copy.groupby('location')[
        'admissions_per_capita'
    ].shift(lag)
    df_copy[f'ili_lag_{lag}'] = df_copy.groupby('location')[
        'unweighted_ili'
    ].shift(lag)
    df_copy[f'log1p_ili_lag_{lag}'] = df_copy.groupby('location')[
        'log1p_ili'
    ].shift(lag)
    df_copy[f'ili_lag_per_capita_{lag}'] = df_copy.groupby('location')[
        'ili_per_capita'
    ].shift(lag)

  # Growth rate and difference features for branching process dynamics
  df_copy['lag_1_val'] = df_copy.groupby('location')[TARGET_STR].shift(1)
  df_copy['lag_2_val'] = df_copy.groupby('location')[TARGET_STR].shift(2)

  # Log1p difference (more stable for zeros and small numbers, corresponds to log-growth)
  df_copy['log1p_lag_1_val'] = np.log1p(df_copy['lag_1_val'].fillna(0))
  df_copy['log1p_lag_2_val'] = np.log1p(df_copy['lag_2_val'].fillna(0))
  df_copy['log1p_growth_rate_1_week'] = (
      df_copy['log1p_lag_1_val'] - df_copy['log1p_lag_2_val']
  ).fillna(
      0
  )  # If both zero, diff is 0

  # Original growth rate, robustified (add 1 to numerator and denominator to handle zeros more gracefully)
  df_copy['growth_rate_1_week'] = (df_copy['lag_1_val'] + 1) / (
      df_copy['lag_2_val'] + 1
  )
  df_copy['growth_rate_1_week'] = (
      df_copy['growth_rate_1_week'].fillna(1).replace([np.inf, -np.inf], 1)
  )

  df_copy['diff_1_week'] = df_copy['lag_1_val'] - df_copy['lag_2_val']

  # --- NEW GROWTH FEATURES (building on existing ones) ---
  df_copy['log1p_lag_3_val'] = np.log1p(
      df_copy.groupby('location')[TARGET_STR].shift(3).fillna(0)
  )
  df_copy['log1p_growth_rate_2_week'] = (
      df_copy['log1p_lag_1_val'] - df_copy['log1p_lag_3_val']
  ).fillna(0)

  # Ensure log1p_growth_rate_1_week is available for shift operation
  df_copy['lagged_log1p_growth_rate_1_week'] = (
      df_copy.groupby('location')['log1p_growth_rate_1_week'].shift(1).fillna(0)
  )
  df_copy['log1p_acceleration'] = (
      df_copy['log1p_growth_rate_1_week']
      - df_copy['lagged_log1p_growth_rate_1_week']
  ).fillna(0)

  # Ratio to last year's value, for annual trend
  df_copy['ratio_current_to_lag52'] = (df_copy['lag_1_val'] + 1) / (
      df_copy.groupby('location')[TARGET_STR].shift(52) + 1
  ).fillna(1)
  df_copy['ratio_current_to_lag52'] = (
      df_copy['ratio_current_to_lag52'].replace([np.inf, -np.inf], 1).fillna(1)
  )
  # --- End NEW GROWTH FEATURES ---

  # Growth features for ILI as well
  df_copy['ili_lag_1_val'] = df_copy.groupby('location')[
      'unweighted_ili'
  ].shift(1)
  df_copy['ili_lag_2_val'] = df_copy.groupby('location')[
      'unweighted_ili'
  ].shift(2)

  df_copy['log1p_ili_lag_1_val'] = np.log1p(df_copy['ili_lag_1_val'].fillna(0))
  df_copy['log1p_ili_lag_2_val'] = np.log1p(df_copy['ili_lag_2_val'].fillna(0))
  df_copy['log1p_ili_growth_rate_1_week'] = (
      df_copy['log1p_ili_lag_1_val'] - df_copy['log1p_ili_lag_2_val']
  ).fillna(0)

  df_copy['ili_growth_rate_1_week'] = (df_copy['ili_lag_1_val'] + 1) / (
      df_copy['ili_lag_2_val'] + 1
  )
  df_copy['ili_growth_rate_1_week'] = (
      df_copy['ili_growth_rate_1_week'].fillna(1).replace([np.inf, -np.inf], 1)
  )  # Robustify

  df_copy['ili_diff_1_week'] = (
      df_copy['ili_lag_1_val'] - df_copy['ili_lag_2_val']
  )

  # Rolling window features on per capita admissions/ILI
  windows = [2, 4, 8]
  for window in windows:
    # Use .transform(lambda x: ...) to apply rolling operations grouped by location
    # shift(1) ensures these are truly 'lagged' rolling features
    df_copy[f'rolling_mean_per_capita_{window}'] = df_copy.groupby('location')[
        'admissions_per_capita'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df_copy[f'rolling_std_per_capita_{window}'] = df_copy.groupby('location')[
        'admissions_per_capita'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
    df_copy[f'ili_rolling_mean_per_capita_{window}'] = df_copy.groupby(
        'location'
    )['ili_per_capita'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df_copy[f'ili_rolling_std_per_capita_{window}'] = df_copy.groupby(
        'location'
    )['ili_per_capita'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )

    # Original target rolling features (for robustness in areas with low pop/where per-capita might be noisy)
    df_copy[f'rolling_mean_{window}'] = df_copy.groupby('location')[
        TARGET_STR
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df_copy[f'rolling_std_{window}'] = df_copy.groupby('location')[
        TARGET_STR
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )

    # --- Rolling window features on log1p admissions/ILI ---
    df_copy[f'log1p_rolling_mean_{window}'] = df_copy.groupby('location')[
        'log1p_admissions'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df_copy[f'log1p_rolling_std_{window}'] = df_copy.groupby('location')[
        'log1p_admissions'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
    df_copy[f'log1p_ili_rolling_mean_{window}'] = df_copy.groupby('location')[
        'log1p_ili'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df_copy[f'log1p_ili_rolling_std_{window}'] = df_copy.groupby('location')[
        'log1p_ili'
    ].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
    # --- End Rolling window features ---

  # Drop temporary columns used for growth rates and log1p intermediates
  df_copy = df_copy.drop(
      columns=[
          'lag_1_val',
          'lag_2_val',
          'log1p_lag_1_val',
          'log1p_lag_2_val',
          'log1p_lag_3_val',
          'lagged_log1p_growth_rate_1_week',
          'ili_lag_1_val',
          'ili_lag_2_val',
          'log1p_ili_lag_1_val',
          'log1p_ili_lag_2_val',
      ],
      errors='ignore',
  )

  return df_copy


# Helper for MLE alpha estimation
def _nbinom_neg_log_likelihood(alpha, y_true, mu_pred):
  """Negative log-likelihood of a Negative Binomial distribution."""
  # Ensure alpha is positive and finite
  alpha = np.clip(
      alpha, 1e-6, 50.0
  )  # Clamped to be consistent with minimize bounds

  # Calculate n and p parameters for Negative Binomial
  n = 1 / alpha
  # Ensure mu_pred is strictly positive to avoid issues with p calculation if alpha is large
  mu_pred_safe = np.maximum(1e-9, mu_pred)
  p = 1 / (1 + alpha * mu_pred_safe)

  # Clamp p to avoid numerical issues (must be between 0 and 1 exclusive)
  p = np.clip(p, 1e-6, 1 - 1e-6)

  # Filter for non-zero mu_pred or y_true > 0 for more stable calculation
  informative_indices = (y_true > 0) | (mu_pred_safe > 0.1)
  if (
      np.sum(informative_indices) < 50
  ):  # Increased minimum informative points for MLE
    return np.inf  # Cannot estimate if not enough informative data

  y_true_informative = y_true[informative_indices]
  n_informative = n[informative_indices]
  p_informative = p[informative_indices]

  # Calculate log PMF. Handle potential issues with zero counts for y_true
  log_pmfs = nbinom.logpmf(y_true_informative, n=n_informative, p=p_informative)

  # Sum of log PMFs, handling potential NaNs (e.g., from invalid parameters) by heavily penalizing them
  log_pmfs = np.nan_to_num(log_pmfs, nan=-1e10, posinf=-1e10, neginf=-1e10)

  return -np.sum(log_pmfs)


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.

  Do not do any cross-validation in here.
  """
  # Ensure target_end_date is datetime for all inputs
  train_x = train_x.copy()
  train_y = train_y.copy()
  test_x = test_x.copy()

  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  # --- Data Augmentation with ILINet (Refined Strategy 2) ---
  # Prepare ilinet_state data for merging
  ilinet_state_clean = ilinet_state[
      ilinet_state['region_type'] == 'States'
  ].copy()
  ilinet_state_clean['target_end_date'] = pd.to_datetime(
      ilinet_state_clean['week_start']
  ) + pd.Timedelta(
      days=6
  )  # Adjust to Saturday end date
  ilinet_state_clean = ilinet_state_clean.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_clean = ilinet_state_clean.merge(
      locations[['location', 'location_name']], on='location_name', how='left'
  )

  # Filter for relevant columns and drop duplicates to ensure unique (date, location) pairs
  ilinet_state_clean = (
      ilinet_state_clean[['target_end_date', 'location', 'unweighted_ili']]
      .dropna(subset=['location'])
      .drop_duplicates(subset=['target_end_date', 'location'])
  )

  # Get all unique locations and relevant dates
  all_locations = pd.concat([train_x['location'], test_x['location']]).unique()
  min_date_train = train_x['target_end_date'].min()
  min_date_test = test_x['target_end_date'].min()
  min_date_ili = (
      ilinet_state_clean['target_end_date'].min()
      if not ilinet_state_clean.empty
      else pd.NaT
  )

  # The earliest date from any source is the effective start for features
  effective_min_date = pd.NaT
  if not pd.isna(min_date_train):
    effective_min_date = min_date_train
  if not pd.isna(min_date_test) and (
      pd.isna(effective_min_date) or min_date_test < effective_min_date
  ):
    effective_min_date = min_date_test
  if not pd.isna(min_date_ili) and (
      pd.isna(effective_min_date) or min_date_ili < effective_min_date
  ):
    effective_min_date = min_date_ili

  # Go back further to ensure enough history for lags (e.g., max_lag + buffer for rolling/growth)
  lags_list = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 52]  # from _add_features
  windows_list = [2, 4, 8]  # from _add_features
  max_lag_needed = (
      max(lags_list + [3]) + max(windows_list) + 4
  )  # Max lag + max rolling window + small buffer for shift(1), and +3 for log1p_lag_3_val

  if not pd.isna(effective_min_date):
    effective_min_date = effective_min_date - pd.Timedelta(weeks=max_lag_needed)
  else:  # Fallback if no dates at all
    effective_min_date = pd.Timestamp('2010-01-01')  # Arbitrary safe start

  max_date = test_x['target_end_date'].max()  # Up to the latest forecast date

  date_range = pd.date_range(
      start=effective_min_date, end=max_date, freq='W-SAT'
  )

  # Create an empty scaffold for all historical data
  full_history_df = pd.DataFrame(
      index=pd.MultiIndex.from_product(
          [date_range, all_locations], names=['target_end_date', 'location']
      )
  ).reset_index()

  # Add population data to the full historical scaffold
  full_history_df = full_history_df.merge(
      locations[['location', 'population', 'location_name']],
      on='location',
      how='left',
  )
  full_history_df['population'] = (
      full_history_df['population'].astype(float).replace(0, 1e-6)
  )  # Ensure population is float and not zero

  # Merge with actual training data (target variable)
  train_df_for_merge = train_x.merge(
      train_y, left_index=True, right_index=True
  )[['target_end_date', 'location', TARGET_STR]]
  full_history_df = full_history_df.merge(
      train_df_for_merge, on=['target_end_date', 'location'], how='left'
  )

  # Merge with ilinet data
  full_history_df = full_history_df.merge(
      ilinet_state_clean,
      on=['target_end_date', 'location'],
      how='left',
      suffixes=('', '_ilinet'),
  )

  # Sort for correct lag calculations
  full_history_df = full_history_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Impute 'unweighted_ili' NaNs BEFORE learning transformation
  full_history_df['unweighted_ili'] = (
      full_history_df.groupby('location')['unweighted_ili'].ffill().bfill()
  )
  full_history_df['unweighted_ili'] = full_history_df['unweighted_ili'].fillna(
      0
  )
  full_history_df['unweighted_ili'] = np.maximum(
      0, full_history_df['unweighted_ili']
  )  # Ensure non-negative

  # --- Centralized base feature derivation (log1p, per capita) ---
  full_history_df['log1p_admissions'] = np.log1p(
      full_history_df[TARGET_STR].fillna(0)
  )
  full_history_df['admissions_per_capita'] = (
      full_history_df[TARGET_STR].fillna(0)
      / full_history_df['population']
      * 100000
  )
  full_history_df['log1p_ili'] = np.log1p(full_history_df['unweighted_ili'])
  full_history_df['ili_per_capita'] = (
      full_history_df['unweighted_ili'] / full_history_df['population'] * 100000
  )

  # Learn a linear transformation from ILI to admissions using the overlap period
  overlap_data_for_transform = full_history_df.dropna(
      subset=[TARGET_STR]
  ).copy()

  location_models: Dict[int, LinearRegression] = {}
  global_ili_to_admissions_model = (
      LinearRegression()
  )  # Default fallback model, will be updated

  # Scaler for log1p_ili input to linear regression models
  ili_log1p_scaler = StandardScaler()

  min_samples_for_local_model = 15

  # Fit global scaler on all available (non-NaN) log1p_ili data across full_history_df
  valid_ili_data_for_scaler_fit = full_history_df[
      full_history_df['log1p_ili'].notna()
  ][['log1p_ili']]
  if (
      not valid_ili_data_for_scaler_fit.empty
      and valid_ili_data_for_scaler_fit['log1p_ili'].nunique() > 1
  ):
    ili_log1p_scaler.fit(valid_ili_data_for_scaler_fit)
  else:
    # If no varied ILI data, scaler will do nothing. Initialize with identity if not fitted.
    warnings.warn(
        'Not enough varied ILI data to fit scaler. Scaler may not be effective.'
    )
    # StandardScaler can't be easily initialized to an identity transform if not fitted
    # Best practice is to ensure fitting is robust or handle cases where it's not needed.

  # Filter for data suitable for fitting global model (overlap)
  valid_global_data_for_fit = overlap_data_for_transform[
      overlap_data_for_transform['log1p_ili'].notna()
      & overlap_data_for_transform['log1p_admissions'].notna()
  ]

  # Fit global model
  if (
      not valid_global_data_for_fit.empty
      and valid_global_data_for_fit['log1p_ili'].nunique() > 1
      and valid_global_data_for_fit['log1p_ili'].var()
      > 1e-9  # Ensure ILI has variance
      and valid_global_data_for_fit['log1p_admissions'].nunique() > 1
  ):  # Ensure target has variance
    try:
      X_overlap_global_raw = valid_global_data_for_fit[['log1p_ili']].copy()
      X_overlap_global_scaled = ili_log1p_scaler.transform(X_overlap_global_raw)
      global_ili_to_admissions_model.fit(
          X_overlap_global_scaled, valid_global_data_for_fit['log1p_admissions']
      )
    except ValueError as e:
      warnings.warn(
          f'Could not fit global ILI to admissions model: {e}. Using a constant'
          ' mean model.'
      )
      global_ili_to_admissions_model.coef_ = np.array([0.0])
      global_ili_to_admissions_model.intercept_ = (
          valid_global_data_for_fit['log1p_admissions'].mean()
          if not valid_global_data_for_fit.empty
          else 0.0
      )
  else:
    warnings.warn(
        'Not enough varied data to fit global ILI to admissions model. Using a'
        ' constant mean model.'
    )
    global_ili_to_admissions_model.coef_ = np.array([0.0])
    global_ili_to_admissions_model.intercept_ = (
        valid_global_data_for_fit['log1p_admissions'].mean()
        if not valid_global_data_for_fit.empty
        else 0.0
    )

  # Fit local models
  for loc_id in all_locations:
    loc_overlap = overlap_data_for_transform[
        overlap_data_for_transform['location'] == loc_id
    ]
    valid_loc_data_for_fit = loc_overlap[
        loc_overlap['log1p_ili'].notna()
        & loc_overlap['log1p_admissions'].notna()
    ]

    if (
        not valid_loc_data_for_fit.empty
        and len(valid_loc_data_for_fit) >= min_samples_for_local_model
        and valid_loc_data_for_fit['log1p_ili'].nunique() > 1
        and valid_loc_data_for_fit['log1p_ili'].var() > 1e-9
        and valid_loc_data_for_fit['log1p_admissions'].nunique() > 1
    ):
      try:
        local_model = LinearRegression()
        X_loc_overlap_raw = valid_loc_data_for_fit[['log1p_ili']].copy()
        X_loc_overlap_scaled = ili_log1p_scaler.transform(
            X_loc_overlap_raw
        )  # Use global scaler
        local_model.fit(
            X_loc_overlap_scaled, valid_loc_data_for_fit['log1p_admissions']
        )
        location_models[loc_id] = local_model
      except ValueError as e:
        warnings.warn(
            'Could not fit local ILI to admissions model for location'
            f' {loc_id}: {e}. Falling back to global model.'
        )
        pass

  # Generate synthetic admissions for periods without actual target data using ILI
  full_history_df['synthetic_admissions_temp'] = np.nan
  for loc_id in all_locations:
    loc_df_idx = full_history_df['location'] == loc_id

    # Only fill if TARGET_STR is NaN (i.e., this is a historical gap to fill)
    needs_synthetic = loc_df_idx & full_history_df[TARGET_STR].isna()

    if needs_synthetic.any():
      X_ili_for_pred_raw = full_history_df.loc[
          needs_synthetic, ['unweighted_ili', 'log1p_ili']
      ].copy()

      model_to_use = location_models.get(loc_id, global_ili_to_admissions_model)

      # Only attempt prediction if the model is fitted and there's ILI data for this subset
      if (
          hasattr(model_to_use, 'coef_')
          and hasattr(model_to_use, 'intercept_')
          and not X_ili_for_pred_raw.empty
          and X_ili_for_pred_raw['log1p_ili'].notna().all()
      ):  # Use log1p_ili for prediction

        X_ili_for_pred_scaled = ili_log1p_scaler.transform(
            X_ili_for_pred_raw[['log1p_ili']]
        )

        predicted_log1p_admissions = model_to_use.predict(X_ili_for_pred_scaled)
        full_history_df.loc[needs_synthetic, 'synthetic_admissions_temp'] = (
            np.round(
                np.expm1(np.maximum(0, predicted_log1p_admissions))
            ).astype(int)
        )
      else:
        full_history_df.loc[needs_synthetic, 'synthetic_admissions_temp'] = (
            0  # Default to 0 if no valid model/data
        )

  # Fill missing actual TARGET_STR values with synthetic ones, prioritizing actual
  full_history_df[TARGET_STR] = full_history_df[TARGET_STR].fillna(
      full_history_df['synthetic_admissions_temp']
  )
  full_history_df = full_history_df.drop(columns=['synthetic_admissions_temp'])

  # Ensure TARGET_STR is non-negative and integer
  full_history_df[TARGET_STR] = np.maximum(
      0, full_history_df[TARGET_STR].fillna(0)
  ).astype('Int64')

  # --- Recalculate base derived features after TARGET_STR augmentation ---
  # This is crucial so log1p_admissions and admissions_per_capita reflect the full augmented history
  full_history_df['log1p_admissions'] = np.log1p(full_history_df[TARGET_STR])
  full_history_df['admissions_per_capita'] = (
      full_history_df[TARGET_STR] / full_history_df['population'] * 100000
  )
  # No need to recalculate log1p_ili/ili_per_capita as unweighted_ili was imputed earlier

  # --- Feature Engineering for training data (actuals + synthetic history) ---
  train_end_date_actual = train_x['target_end_date'].max()

  # Apply _add_features to the full historical data up to train_end_date_actual for training set.
  # This includes actuals and synthetic history, so lags/rolling are based on the full augmented past.
  X_train_full = full_history_df[
      full_history_df['target_end_date'] <= train_end_date_actual
  ].copy()
  X_train_full_with_features = _add_features(X_train_full)
  y_train_full = X_train_full_with_features[TARGET_STR]

  # Identify feature columns
  feature_cols = [
      col
      for col in X_train_full_with_features.columns
      if col.startswith((
          'sin_week_',
          'cos_week_',
          'lag_',
          'ili_lag_',
          'rolling_',
          'growth_rate_',
          'diff_',
          'log1p_growth_rate_',
          'log1p_rolling_',
          'log1p_lag_',
          'log1p_acceleration',
          'ratio_current_to_lag52',
      ))
      or col in ['population', 'month', 'year']
  ]
  feature_cols = [
      col for col in feature_cols if col in X_train_full_with_features.columns
  ]  # Ensure existence

  X_train = X_train_full_with_features[feature_cols]

  # Impute NaNs in features (e.g., for initial lags) with 0 before scaling.
  X_train = X_train.fillna(0)

  # --- Target Transformation for XGBoost ---
  y_train_transformed = np.log1p(y_train_full.fillna(0))

  # Scale numerical features
  scaler = StandardScaler()
  numerical_cols_to_scale = [
      col
      for col in feature_cols
      if not col.startswith(('sin_', 'cos_'))
      and col not in ['month', 'week_of_year', 'year', 'location']
  ]

  X_train_scaled = X_train.copy()
  if not X_train.empty and not X_train[numerical_cols_to_scale].empty:
    scaler.fit(X_train[numerical_cols_to_scale])
    X_train_scaled[numerical_cols_to_scale] = scaler.transform(
        X_train[numerical_cols_to_scale]
    )

  # --- Model Training (XGBoost for transformed mean prediction) ---
  xgbr = xgb.XGBRegressor(
      objective='reg:squarederror',
      n_estimators=1000,
      learning_rate=0.01,
      max_depth=8,
      subsample=0.7,
      colsample_bytree=0.7,
      random_state=42,
      n_jobs=-1,
      verbosity=0,
      tree_method='hist',
  )

  if X_train_scaled.empty or y_train_transformed.empty:
    test_y_hat_quantiles = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return test_y_hat_quantiles.fillna(0).astype(int)

  xgbr.fit(X_train_scaled, y_train_transformed)

  # --- Estimate Dispersion Parameter (alpha) for Negative Binomial (MLE) ---
  alpha_est = 0.1  # Default fallback alpha value, slightly higher than 0.05

  try:
    train_log_mu_preds = xgbr.predict(X_train_scaled)
    train_mu_preds = np.expm1(train_log_mu_preds)
    train_mu_preds = np.maximum(1e-6, train_mu_preds)  # Ensure mu is positive

    y_train_actual = y_train_full.fillna(0).values.astype(
        int
    )  # Ensure integer type for nbinom.logpmf

    # Filter for informative points to avoid numerical issues with all zeros
    informative_indices = (y_train_actual > 0) | (train_mu_preds > 0.1)

    if (
        np.sum(informative_indices) > 50
    ):  # Need enough data points for MLE to be meaningful
      y_informative = y_train_actual[informative_indices]
      mu_informative = train_mu_preds[informative_indices]

      # Initial guess for alpha (could use moment estimator or a fixed value)
      var_estimate = np.var(y_informative)
      mean_estimate = np.mean(y_informative)

      # More robust initial alpha calculation: clamp moment estimate to a reasonable range
      if mean_estimate > 0 and var_estimate >= mean_estimate:
        initial_alpha = (var_estimate - mean_estimate) / (mean_estimate**2)
        initial_alpha = np.clip(
            initial_alpha, 1e-4, 5.0
        )  # Confine initial guess to a more reasonable range
      else:
        initial_alpha = 0.1  # Slightly higher fallback for potentially underdispersed or low count data

      # Bounds for alpha: must be positive. A practical upper bound helps stability.
      alpha_bounds = [(1e-6, 50.0)]

      res = minimize(
          _nbinom_neg_log_likelihood,
          x0=[initial_alpha],
          args=(y_informative, mu_informative),
          method='L-BFGS-B',
          bounds=alpha_bounds,
          options={'disp': False, 'gtol': 1e-5},
      )

      if res.success:
        alpha_est = np.clip(res.x[0], alpha_bounds[0][0], alpha_bounds[0][1])
      else:
        warnings.warn(
            f'MLE for alpha did not converge: {res.message}. Using initial or'
            ' default alpha.'
        )
        alpha_est = (
            initial_alpha if initial_alpha > 0 else 0.1
        )  # Use slightly higher fallback
    else:
      warnings.warn(
          f'Not enough informative data ({np.sum(informative_indices)} points)'
          f' for MLE alpha estimation. Using default alpha ({alpha_est}).'
      )
      alpha_est = 0.1  # Fallback to 0.1

  except Exception as e:
    warnings.warn(
        f'Error during MLE alpha estimation: {e}. Using default alpha'
        f' ({alpha_est}).'
    )
    alpha_est = 0.1  # Fallback to 0.1

  if alpha_est <= 0:
    alpha_est = 1e-6  # Ensure positive alpha, consistent with lower bound

  # --- Optimized Recursive Prediction Loop for Branching Process Dynamics ---
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # prediction_df_state will contain actuals up to train_end_date_actual,
  # and will be progressively filled with median predictions for future dates.
  # It will be the single source of truth for features needed during recursion.

  # Initialize prediction_df_state with the full history and future scaffold
  prediction_df_state = full_history_df.set_index(
      ['target_end_date', 'location']
  ).copy()

  # Set future TARGET_STR and derived values to NaN initially for prediction
  future_dates_idx = (
      prediction_df_state.index.get_level_values('target_end_date')
      > train_end_date_actual
  )
  prediction_df_state.loc[future_dates_idx, TARGET_STR] = np.nan
  prediction_df_state.loc[future_dates_idx, 'admissions_per_capita'] = np.nan
  prediction_df_state.loc[future_dates_idx, 'log1p_admissions'] = (
      np.nan
  )  # Crucial: reset log1p too

  # Sort test_x to ensure we process dates chronologically
  test_x_sorted = test_x.sort_values(by=['target_end_date', 'location']).copy()
  unique_test_dates = sorted(test_x_sorted['target_end_date'].unique())

  # Iterate through unique prediction weeks (horizons)
  for current_target_end_date in unique_test_dates:
    # Get current test points for this week
    current_week_test_points = test_x_sorted[
        test_x_sorted['target_end_date'] == current_target_end_date
    ]
    original_indices_for_this_week = (
        current_week_test_points.index
    )  # Store original indices from test_x
    current_week_locations = current_week_test_points['location'].unique()

    # Define the history window needed for feature calculation for current_target_end_date
    # This slice includes 'current_target_end_date' itself and max_lag_needed weeks prior.
    # This temporary DataFrame will contain actuals for dates <= train_end_date_actual and
    # median predictions for dates between train_end_date_actual and current_target_end_date - 1.
    start_date_for_temp_features = current_target_end_date - pd.Timedelta(
        weeks=max_lag_needed
    )

    # Filter prediction_df_state to get the relevant window for _add_features
    temp_df_for_features = prediction_df_state.loc[
        (
            prediction_df_state.index.get_level_values('target_end_date')
            >= start_date_for_temp_features
        )
        & (
            prediction_df_state.index.get_level_values('target_end_date')
            <= current_target_end_date
        )
        & (
            prediction_df_state.index.get_level_values('location').isin(
                current_week_locations
            )
        )
    ].reset_index()

    # Recalculate features for this temp_df_for_features.
    # This call is now on a much smaller, focused dataset for each unique test week.
    # The lags/rolling means will correctly incorporate any previous median predictions.
    history_with_features = _add_features(temp_df_for_features)

    # Extract the specific feature vectors for the current week's prediction points
    X_current_week_raw_features = history_with_features[
        history_with_features['target_end_date'] == current_target_end_date
    ].copy()

    # Reindex X_current_week_raw_features to match the order of current_week_test_points,
    # which ensures predictions are mapped correctly back to the original test_x index.
    X_current_week_raw_features = X_current_week_raw_features.set_index(
        ['target_end_date', 'location']
    )
    current_week_test_points_indexed = current_week_test_points.set_index(
        ['target_end_date', 'location']
    )
    X_current_week_raw_features = X_current_week_raw_features.reindex(
        current_week_test_points_indexed.index
    )

    # Select feature columns and impute NaNs (e.g., for very early lags in prediction horizon)
    X_current_pred_final = X_current_week_raw_features[feature_cols].fillna(0)

    if X_current_pred_final.empty:
      warnings.warn(
          f'No features generated for {current_target_end_date} for any'
          ' location. Filling with zeros.'
      )
      quant_values_matrix = np.zeros(
          (len(current_week_test_points), len(QUANTILES)), dtype=int
      )
    else:
      # Scale the feature vectors using the pre-fitted scaler
      X_current_pred_scaled = X_current_pred_final.copy()
      X_current_pred_scaled[numerical_cols_to_scale] = scaler.transform(
          X_current_pred_final[numerical_cols_to_scale]
      )

      # Predict log(mu), then inverse transform to get mu on original scale
      log_mu_preds = xgbr.predict(X_current_pred_scaled)
      mu_preds = np.expm1(log_mu_preds)
      mu_preds = np.maximum(0, mu_preds)  # Ensure non-negative results

      # Generate quantiles for the current mu values
      quant_values_matrix = np.zeros((len(mu_preds), len(QUANTILES)), dtype=int)
      for i, mu_pred in enumerate(mu_preds):
        if mu_pred < 1e-6:  # If mean is practically zero, all quantiles are 0
          quant_values_single = np.zeros(len(QUANTILES), dtype=int)
        else:
          # Negative Binomial parameters
          n_param = 1 / alpha_est
          p_param = 1 / (1 + alpha_est * mu_pred)

          # Clamp parameters to valid ranges for nbinom.ppf to prevent errors
          n_param = np.maximum(1e-6, n_param)
          p_param = np.maximum(1e-6, np.minimum(1 - 1e-6, p_param))

          quant_values_single = nbinom.ppf(QUANTILES, n=n_param, p=p_param)
          quant_values_single = np.round(
              np.maximum(0, quant_values_single)
          ).astype(int)
          quant_values_single = np.maximum.accumulate(
              quant_values_single
          )  # Enforce monotonicity
        quant_values_matrix[i] = quant_values_single

    # Store predictions in the output DataFrame, using the original test_x indices
    test_y_hat_quantiles.loc[
        original_indices_for_this_week, [f'quantile_{q}' for q in QUANTILES]
    ] = quant_values_matrix

    # Update prediction_df_state with the predicted median for the current week.
    # This is crucial for the recursive 'branching' aspect: these medians will be used as
    # lagged inputs for feature calculations in subsequent prediction weeks.
    for i, (idx, row_test_x) in enumerate(current_week_test_points.iterrows()):
      loc_to_update = row_test_x['location']
      median_pred = (
          quant_values_matrix[i, QUANTILES.index(0.5)]
          if 0.5 in QUANTILES
          else quant_values_matrix[i, len(QUANTILES) // 2]
      )

      prediction_df_state.loc[
          (current_target_end_date, loc_to_update), TARGET_STR
      ] = median_pred

      # Also update derived features based on the new median prediction
      current_population_for_update = prediction_df_state.loc[
          (current_target_end_date, loc_to_update), 'population'
      ]
      if (
          current_population_for_update is not None
          and current_population_for_update > 1e-6
      ):
        prediction_df_state.loc[
            (current_target_end_date, loc_to_update), 'admissions_per_capita'
        ] = (median_pred / current_population_for_update * 100000)
      else:
        prediction_df_state.loc[
            (current_target_end_date, loc_to_update), 'admissions_per_capita'
        ] = 0

      prediction_df_state.loc[
          (current_target_end_date, loc_to_update), 'log1p_admissions'
      ] = np.log1p(median_pred)

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
