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
MODEL_NAME = 'Google_SAI-Hybrid_3'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from typing import Dict, List, Tuple, Callable
import scipy.stats as st  # Added for normal distribution quantiles

# TARGET_STR and QUANTILES are assumed to be available from the global scope of the notebook.
# Similarly, `locations`, `ilinet`, `ilinet_hhs`, `ilinet_state` are assumed to be globally available.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make probabilistic forecasts for Total Influenza Hospital Admissions.

  This function implements a hybrid strategy leveraging:
  1. Historical ILINet data for augmenting training history with robust linear
  regression-based transformation,
     where both ILI and admissions are population-normalized and
     *sqrt-transformed*.
     Revised fallback hierarchy for ILI-to-admissions model fitting with a
     *data-driven global fallback*.
     The ILI-to-Admissions linear regression models are now trained on all
     available overlap data,
     including zero-admission weeks, to better capture low-incidence dynamics.
     The ultimate fallback for ILI-to-admissions model parameters is now a
     robust mean ratio,
     rather than fixed arbitrary constants, for better data-driven
     initialization.
  2. *Sqrt-transformation* of population-normalized target variable for better
  model performance on count data.
  3. Enhanced feature engineering incorporating:
     - `location` as a categorical feature, seasonal components, year,
     `relative_season_week`.
     - `relative_season_week_sq` to capture more complex seasonal shapes.
     - `sin_relative_season_week` and `cos_relative_season_week` to robustly
     capture seasonal cycles (GPR x1, x2 inspiration).
     - Lagged values, rolling means, season cumulative activity, and previous
     season peak activity
       (all population-normalized and sqrt-transformed).
     - Log-difference based trend (growth and acceleration) features.
     - Exponentially Weighted Moving Averages (EWMA) for
     `current_vs_historical_admissions_diff_lag1`
       and `current_vs_historical_cumulative_diff_lagged` to create a more
       dynamic and adaptive
       'season severity' signal.
     - `ewma_admissions_diff_deviation_lagged` to capture how much the latest
     severity assessment
       deviates from its smoothed EWMA trend, signaling rapid changes.
     - `rolling_std_current_vs_historical_admissions_diff_lagged` to quantify
     the volatility in the
       season's severity signal, inspired by GPR's category-specific noise.
     - **NEW:** `ewma_admissions_diff_deviation_lagged_x_relative_season_week`
     for nuanced interaction.
     - National-level aggregates of key features to provide broader
     epidemiological context to state-level predictions,
       using population-weighted averages for representativeness.
     - Corrected calculation of `national_previous_season_peak_norm_target`
     based on actual national peaks.
     - Robust fallback for `current_vs_historical` features using
     population-weighted blending of local and national historical climatology.
     - `previous_season_end_of_season_avg_norm_target` to capture previous
     season's lingering activity.
     - `previous_season_max_ewma_admissions_diff` to capture previous season's
     peak severity anomaly.
     - More interaction features with `population_log` to capture
     population-specific dynamics,
       including interactions with seasonal components.
     - Interaction features involving `horizon` and key time-series components
       (`lag_1_norm`, `relative_season_week`, `ewma_admissions_diff_lagged`) to
       capture horizon-dependent relationships.
     - `population_log_x_horizon` to capture population-specific horizon
     effects.
  4. LightGBM for *median* prediction and a *second* LightGBM for *prediction of
  residuals' scale*
     (heteroskedasticity), providing robust statistical predictions with dynamic
     uncertainty.
     **LGBM hyperparameters adjusted for slight increase in capacity and
     generalization.**
  5. Iterative prediction for handling lagged, national, and dynamically
  evolving features in the forecast horizon.
     The propagation of national features has been refined for consistency and
     population weighting.
     Optimized calculation of season cumulative features in the prediction loop.
  6. Post-processing to ensure quantile monotonicity and non-negativity,
  including inverse transformation.
     Quantile generation uses a Student's t-distribution instead of a normal
     distribution
     to account for heavier tails in error distributions. A
     `SCALE_INFLATION_FACTOR` is applied to widen
     prediction intervals for better calibration (inspired by SIRS ensemble
     spread).
     **Adjusted `SCALE_INFLATION_FACTOR` for potentially better calibration.**
  """
  # --- Configuration Constants ---
  MAX_LAG = 4  # Use up to 4 weeks lag for lagged normalized admissions features
  ROLLING_MEAN_WINDOW = (
      3  # Window for rolling mean feature (e.g., last 3 weeks)
  )
  MIN_ILI_TO_ADMISSIONS_OVERLAP = (
      10  # Minimum weeks of overlap to learn a local linear model
  )
  MIN_UNIVERSAL_OVERLAP = 30  # Minimum weeks for a universal linear model
  MIN_GLOBAL_OVERLAP = (
      40  # Minimum weeks for a global linear model for better initial default
  )
  POP_NORM_FACTOR = 100_000  # Normalize admissions/ILI per 100,000 population
  INTERNAL_TARGET_COL = (  # The actual target for LGBM median model (sqrt(x+1)-1)
      'admissions_per_100k_sqrt_plus_1_minus_1'
  )
  INTERNAL_SCALE_TARGET_COL = (  # The target for LGBM scale model
      'admissions_residuals_sqrt_abs'
  )
  LAST_N_SEASONS_FOR_CLIMATOLOGY = (
      7  # Number of most recent seasons to use for climatology features
  )
  EWMA_ALPHA = 0.20  # Alpha for Exponentially Weighted Moving Average for severity features
  MIN_CLIMATOLOGY_OBS = (
      10  # Minimum observations for climatology for robustness
  )

  # Constants for climatology blending and previous season features
  CLIMATOLOGY_WEIGHT_K = (
      9  # Smoothing factor for blending local/national climatology
  )
  END_SEASON_RELATIVE_WEEKS = [
      20,
      21,
      22,
      23,
      24,
      25,
  ]  # Relative weeks to define 'end of season' (typically March-April)

  # Constants for Quantile Generation Refinement
  DF_T_DISTRIBUTION = (
      4  # Degrees of freedom for Student's t-distribution for quantiles
  )
  SCALE_INFLATION_FACTOR = (
      1.28  # Factor to inflate predicted scale (increased from 1.22)
  )

  # --- 0. Initial Data Preparation ---

  # Ensure consistent data types for dates and populations
  train_x = train_x.copy()
  test_x = test_x.copy()
  train_y = train_y.copy()

  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  train_y.name = TARGET_STR  # Ensure target series has correct name

  train_x['population'] = train_x['population'].astype(float)
  test_x['population'] = test_x['population'].astype(float)

  # Merge train_x and train_y to have a single DataFrame for current observed data
  current_observed_df = train_x.copy()
  current_observed_df[TARGET_STR] = train_y

  # Calculate normalized target for current observed data
  # NEW TRANSFORMATION: sqrt(x+1)-1
  normalized_target_val = (
      current_observed_df[TARGET_STR]
      / current_observed_df['population'].replace(0, np.nan)
  ) * POP_NORM_FACTOR
  current_observed_df[INTERNAL_TARGET_COL] = (
      np.sqrt(normalized_target_val + 1) - 1
  )
  current_observed_df[INTERNAL_TARGET_COL] = current_observed_df[
      INTERNAL_TARGET_COL
  ].fillna(
      0.0
  )  # Handle zero admissions/population cases

  # --- 1. Data Augmentation with ILINet (Improved Linear Regression Transformation with Global Fallback) ---

  # Process ilinet_state data. The ILINet data is available before 2022-10-15.
  ilinet_state_processed = ilinet_state.copy()
  ilinet_state_processed['target_end_date'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  )
  # Adjust ILI 'week_start' (typically Sunday) to match 'target_end_date' (Saturday)
  ilinet_state_processed['target_end_date'] = ilinet_state_processed[
      'target_end_date'
  ] + pd.Timedelta(days=6)

  # Filter to only data available before the competition's historical cutoff
  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] <= pd.to_datetime('2022-10-15')
  ].copy()

  # Map ilinet 'region' (location_name) to 'location' (FIPS code)
  location_name_to_fips = locations.set_index('location_name')[
      'location'
  ].to_dict()
  ilinet_state_processed['location'] = ilinet_state_processed['region'].map(
      location_name_to_fips
  )
  ilinet_state_processed = ilinet_state_processed.dropna(
      subset=['location', 'ilitotal']
  ).copy()
  ilinet_state_processed['location'] = ilinet_state_processed[
      'location'
  ].astype(int)
  # Merge population data
  ilinet_state_processed = ilinet_state_processed.merge(
      locations[['location', 'population']], on='location', how='left'
  )
  ilinet_state_processed['population'] = ilinet_state_processed[
      'population'
  ].astype(float)

  # Calculate normalized ILI for historical data
  ilinet_state_processed['ilitotal_per_100k'] = (
      ilinet_state_processed['ilitotal']
      / ilinet_state_processed['population'].replace(0, np.nan)
  ) * POP_NORM_FACTOR
  ilinet_state_processed['ilitotal_per_100k'] = ilinet_state_processed[
      'ilitotal_per_100k'
  ].fillna(0.0)

  # Apply the same sqrt transformation to ILI as to the target admissions
  ilinet_state_processed['ilitotal_per_100k_transformed'] = (
      np.sqrt(ilinet_state_processed['ilitotal_per_100k'] + 1) - 1
  )

  # Prepare overlap data for learning ILI-to-Admissions transformation
  overlap_data_for_regression = pd.merge(
      current_observed_df[[
          'target_end_date',
          'location',
          'population',
          TARGET_STR,
          INTERNAL_TARGET_COL,
      ]],
      ilinet_state_processed[[
          'target_end_date',
          'location',
          'ilitotal',
          'ilitotal_per_100k',
          'ilitotal_per_100k_transformed',
      ]],
      on=['target_end_date', 'location'],
      how='inner',
  )
  overlap_data_for_regression = overlap_data_for_regression.dropna(
      subset=[TARGET_STR, 'ilitotal']
  )
  overlap_data_for_regression_for_fitting = overlap_data_for_regression.copy()

  # Helper to calculate robust data-driven initial defaults for slope and intercept
  def _get_data_driven_initial_defaults(
      df_overlap_for_fitting,
  ):
    """Calculates robust, data-driven initial default slope and intercept.

    Operates on sqrt-transformed ILI and Admissions. Prioritizes a linear
    regression fit, then a robust mean ratio, then a hardcoded fallback.
    """
    X_val = df_overlap_for_fitting[
        'ilitotal_per_100k_transformed'
    ].values.reshape(-1, 1)
    y_val = df_overlap_for_fitting[INTERNAL_TARGET_COL].values

    # 1. Try a full linear regression on all available overlap data if sufficient
    try:
      if len(df_overlap_for_fitting) >= MIN_GLOBAL_OVERLAP:
        model = LinearRegression()
        model.fit(X_val, y_val)
        # Ensure positive or near-zero slope and reasonable intercept
        if (
            model.coef_[0] >= -1e-6
            and not np.isnan(model.coef_[0])
            and not np.isnan(model.intercept_)
        ):
          return model.coef_[0], model.intercept_
    except Exception:
      pass  # Fallback to next strategy

    # 2. Robust mean ratio fallback if linear regression is not feasible or fails
    # Consider only positive values for means to avoid issues with zeros
    mean_y = y_val[y_val > 0].mean() if y_val[y_val > 0].size > 0 else 0.0
    mean_x = X_val[X_val > 0].mean() if X_val[X_val > 0].size > 0 else 0.0

    if mean_x > 0.001 and mean_y > 0.001:
      # Ensure slope is not zero or negative, minimum of 0.001 for a tiny positive slope
      return max(0.001, mean_y / mean_x), 0.0

    # 3. Ultimate hardcoded fallback
    return 0.1, 0.0  # Default if no robust data-driven option works

  # Helper to learn a linear model for ilitotal_per_100k_transformed -> INTERNAL_TARGET_COL
  def _learn_ili_admissions_model(
      df_overlap,
      min_samples,
      default_slope,
      default_intercept,
  ):
    """Attempts to fit a LinearRegression model, returning a model with provided default parameters if unsuccessful or invalid.

    Operates on sqrt-transformed ILI and Admissions.
    """
    model = LinearRegression()

    # Define the default model with provided default parameters
    default_result_model = LinearRegression()
    default_result_model.coef_ = np.array([default_slope])
    default_result_model.intercept_ = default_intercept

    if len(df_overlap) < min_samples:
      return default_result_model

    X = df_overlap['ilitotal_per_100k_transformed'].values.reshape(-1, 1)
    y = df_overlap[INTERNAL_TARGET_COL].values

    try:
      model.fit(X, y)
      # Check for reasonable model parameters: positive or very small non-negative slope and non-NaN coefficients/intercepts
      if (
          model.coef_[0] >= -1e-6
          and not np.isnan(model.coef_[0])
          and not np.isnan(model.intercept_)
      ):
        return model
      else:  # Fallback due to unreasonable fitted parameters (e.g., negative slope or NaN)
        return default_result_model
    except Exception:  # Catch any errors during fitting
      # Fallback due to fitting error (e.g., singular matrix)
      return default_result_model

  # Refined ILI Transformation Fallback Hierarchy (Data-driven global fallback)
  # 1. Calculate robust data-driven initial defaults
  initial_default_slope, initial_default_intercept = (
      _get_data_driven_initial_defaults(overlap_data_for_regression_for_fitting)
  )

  # 2. Global model: learned from ALL overlap data, provides the most robust general fallback
  global_ili_model = _learn_ili_admissions_model(
      overlap_data_for_regression_for_fitting,
      min_samples=MIN_GLOBAL_OVERLAP,
      default_slope=initial_default_slope,
      default_intercept=initial_default_intercept,
  )

  # 3. Universal model: learned from all overlap data, falls back to the robust global_ili_model parameters
  universal_ili_model = _learn_ili_admissions_model(
      overlap_data_for_regression_for_fitting,
      min_samples=MIN_UNIVERSAL_OVERLAP,
      default_slope=global_ili_model.coef_[0],
      default_intercept=global_ili_model.intercept_,
  )

  # 4. Location-specific models: each falls back to the universal model if local data is insufficient
  loc_ili_to_admissions_models: Dict[int, LinearRegression] = {}
  for loc in locations['location'].unique():
    loc_overlap = overlap_data_for_regression_for_fitting[
        overlap_data_for_regression_for_fitting['location'] == loc
    ].copy()

    local_model = _learn_ili_admissions_model(
        loc_overlap,
        MIN_ILI_TO_ADMISSIONS_OVERLAP,
        default_slope=universal_ili_model.coef_[0],
        default_intercept=universal_ili_model.intercept_,
    )
    loc_ili_to_admissions_models[loc] = local_model

  # Create full historical dataset using ILI and then overlay real admissions
  historical_df = ilinet_state_processed[[
      'target_end_date',
      'location',
      'region',
      'population',
      'ilitotal',
      'ilitotal_per_100k',
      'ilitotal_per_100k_transformed',
  ]].copy()
  historical_df = historical_df.rename(columns={'region': 'location_name'})
  historical_df = historical_df.drop_duplicates(
      subset=['location', 'target_end_date']
  )

  historical_df[INTERNAL_TARGET_COL] = 0.0  # Initialize with 0

  for loc in historical_df['location'].unique():
    loc_rows_indices = historical_df['location'] == loc
    ilitotal_per_100k_transformed_for_loc = historical_df.loc[
        loc_rows_indices, 'ilitotal_per_100k_transformed'
    ].copy()

    # Explicitly set 0 for INTERNAL_TARGET_COL if ilitotal_per_100k_transformed is 0
    historical_df.loc[
        loc_rows_indices & (ilitotal_per_100k_transformed_for_loc == 0),
        INTERNAL_TARGET_COL,
    ] = 0.0

    if loc in loc_ili_to_admissions_models:
      model_to_use = loc_ili_to_admissions_models[loc]
      X_predict = ilitotal_per_100k_transformed_for_loc.values.reshape(-1, 1)
      predicted_internal_target = model_to_use.predict(X_predict)
      # Ensure predictions are non-negative after transformation
      predicted_internal_target[predicted_internal_target < 0] = 0.0
      historical_df.loc[loc_rows_indices, INTERNAL_TARGET_COL] = (
          predicted_internal_target
      )

  # Convert synthetic INTERNAL_TARGET_COL back to Total Influenza Admissions
  # INVERSE TRANSFORMATION: (y+1)^2 - 1
  predicted_normalized_target = (
      historical_df[INTERNAL_TARGET_COL] + 1
  ) ** 2 - 1
  historical_df[TARGET_STR] = (
      (
          predicted_normalized_target
          * historical_df['population']
          / POP_NORM_FACTOR
      )
      .round()
      .astype(int)
  )
  historical_df[TARGET_STR] = np.maximum(0, historical_df[TARGET_STR])

  # Combine synthetic history with actual observations, prioritizing actual
  full_train_df = pd.concat(
      [
          historical_df[[
              'target_end_date',
              'location',
              'location_name',
              'population',
              TARGET_STR,
          ]],
          current_observed_df[[
              'target_end_date',
              'location',
              'location_name',
              'population',
              TARGET_STR,
          ]],
      ],
      ignore_index=True,
  )

  full_train_df = full_train_df.sort_values(
      by=['location', 'target_end_date', TARGET_STR],
      ascending=[True, True, False],
  )
  full_train_df = full_train_df.drop_duplicates(
      subset=['location', 'target_end_date'], keep='first'
  )
  full_train_df = full_train_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Recalculate INTERNAL_TARGET_COL after combining actual and synthetic data
  # NEW TRANSFORMATION: sqrt(x+1)-1
  normalized_target_val = (
      full_train_df[TARGET_STR] / full_train_df['population'].replace(0, np.nan)
  ) * POP_NORM_FACTOR
  full_train_df[INTERNAL_TARGET_COL] = np.sqrt(normalized_target_val + 1) - 1
  full_train_df[INTERNAL_TARGET_COL] = full_train_df[
      INTERNAL_TARGET_COL
  ].fillna(0.0)

  # --- 2. Feature Engineering (Enhanced with Season Severity and Inter-Season Carry-over) ---

  # Helper to determine flu season start year (e.g., 2023 for 2023/2024 season)
  def get_season_year(date):
    return date.year if date.isocalendar().week >= 40 else date.year - 1

  # Helper to get relative season week
  def get_relative_season_week(date):
    week = date.isocalendar().week
    return week - 40 if week >= 40 else week + 12

  # Helper to generate basic features
  def create_base_features(df_input):
    df = df_input.copy()
    df['epiweek'] = df['target_end_date'].dt.isocalendar().week.astype(int)
    df['sin_epiweek'] = np.sin(2 * np.pi * df['epiweek'] / 52)
    df['cos_epiweek'] = np.cos(2 * np.pi * df['epiweek'] / 52)
    df['year'] = df['target_end_date'].dt.year
    df['population_log'] = np.log1p(df['population'])
    df['season_year_start'] = df['target_end_date'].apply(get_season_year)
    df['relative_season_week'] = df['target_end_date'].apply(
        get_relative_season_week
    )
    # Add quadratic term for relative_season_week
    df['relative_season_week_sq'] = df['relative_season_week'] ** 2
    # Add sinusoidal terms for relative_season_week (GPR x2 inspiration)
    df['sin_relative_season_week'] = np.sin(
        2 * np.pi * df['relative_season_week'] / 52
    )
    df['cos_relative_season_week'] = np.cos(
        2 * np.pi * df['relative_season_week'] / 52
    )
    # Add horizon (will be filled for test_x)
    if 'horizon' in df.columns:
      df['horizon'] = df['horizon'].astype(int)
    else:
      df['horizon'] = 0  # For training data, horizon is 0 for past observations
    return df

  full_train_df_features = create_base_features(full_train_df.copy())
  full_train_df_features = full_train_df_features.set_index(
      ['location', 'target_end_date']
  )

  # Populate lags and rolling mean for training data
  for lag in range(1, MAX_LAG + 1):
    full_train_df_features[f'lag_{lag}_norm'] = full_train_df_features.groupby(
        level='location'
    )[INTERNAL_TARGET_COL].shift(lag)

  full_train_df_features[
      f'rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm'
  ] = full_train_df_features.groupby(level='location')[
      INTERNAL_TARGET_COL
  ].transform(
      lambda x: x.shift(1).rolling(ROLLING_MEAN_WINDOW, min_periods=1).mean()
  )

  # Feature: season_cumulative_norm_target_lagged
  full_train_df_features[
      'season_cumulative_norm_target_lagged'
  ] = full_train_df_features.groupby(['location', 'season_year_start'])[
      INTERNAL_TARGET_COL
  ].transform(
      lambda x: x.shift(1).expanding(min_periods=1).sum()
  )
  full_train_df_features['season_cumulative_norm_target_lagged'] = (
      full_train_df_features['season_cumulative_norm_target_lagged'].fillna(0.0)
  )

  # Feature: previous_season_peak_norm_target (state-level)
  seasonal_peaks_norm = (
      full_train_df_features.groupby(['location', 'season_year_start'])[
          INTERNAL_TARGET_COL
      ]
      .max()
      .reset_index()
  )
  previous_season_peak_lookup_norm = {
      (row['location'], row['season_year_start']): row[INTERNAL_TARGET_COL]
      for _, row in seasonal_peaks_norm.iterrows()
  }
  full_train_df_features['previous_season_peak_norm_target'] = (
      full_train_df_features.apply(
          lambda row: previous_season_peak_lookup_norm.get(
              (row.name[0], row['season_year_start'] - 1), 0.0
          ),
          axis=1,
      )
  )
  full_train_df_features['previous_season_peak_norm_target'] = (
      full_train_df_features['previous_season_peak_norm_target'].fillna(0.0)
  )

  # Feature: admissions_growth_rate_log_diff_lag1
  full_train_df_features['admissions_growth_rate_log_diff_lag1'] = (
      full_train_df_features[f'lag_1_norm']
      - full_train_df_features[f'lag_2_norm']
  )
  full_train_df_features['admissions_growth_rate_log_diff_lag1'] = (
      full_train_df_features['admissions_growth_rate_log_diff_lag1'].fillna(0.0)
  )

  # Feature: admissions_acceleration_log_diff_lag1
  full_train_df_features[
      'admissions_acceleration_log_diff_lag1'
  ] = full_train_df_features['admissions_growth_rate_log_diff_lag1'] - (
      full_train_df_features[f'lag_2_norm']
      - full_train_df_features[f'lag_3_norm']
  )
  full_train_df_features['admissions_acceleration_log_diff_lag1'] = (
      full_train_df_features['admissions_acceleration_log_diff_lag1'].fillna(
          0.0
      )
  )

  # --- HYBRID FEATURES: Season Severity/Climatology and National Aggregates ---

  # 1. Historical Averages for Severity Feature (GPR `x4` inspiration)
  latest_season_year = full_train_df_features['season_year_start'].max()
  # Exclude the current season (latest_season_year) from climatology calculation
  seasons_for_climatology = range(
      max(
          full_train_df_features['season_year_start'].min(),
          latest_season_year - LAST_N_SEASONS_FOR_CLIMATOLOGY,
      ),
      latest_season_year,
  )

  # Filter full_train_df_features to only include recent seasons for climatology
  recent_seasons_data = full_train_df_features.reset_index()[
      full_train_df_features.reset_index()['season_year_start'].isin(
          seasons_for_climatology
      )
  ].copy()

  # Calculate national historical climatology for fallback using population-weighted mean
  national_historical_climatology = (
      recent_seasons_data.groupby('relative_season_week')
      .apply(
          lambda x: pd.Series({
              'national_historical_avg_target_norm_by_week': (
                  np.average(x[INTERNAL_TARGET_COL], weights=x['population'])
                  if not x[INTERNAL_TARGET_COL].isnull().all()
                  and x['population'].sum() > 0
                  else 0.0
              ),
              'national_historical_avg_cumulative_norm_by_week': (
                  np.average(
                      x['season_cumulative_norm_target_lagged'],
                      weights=x['population'],
                  )
                  if not x['season_cumulative_norm_target_lagged']
                  .isnull()
                  .all()
                  and x['population'].sum() > 0
                  else 0.0
              ),
              'count_observations_national': x[INTERNAL_TARGET_COL].count(),
          })
      )
      .reset_index()
  )
  national_historical_climatology['count_observations_national'] = (
      national_historical_climatology['count_observations_national'].astype(int)
  )

  # Calculate state-level historical climatology using population-weighted mean AND count of observations
  historical_climatology = (
      recent_seasons_data.groupby(['location', 'relative_season_week'])
      .apply(
          lambda x: pd.Series({
              'historical_avg_target_norm_by_week_loc': (
                  np.average(x[INTERNAL_TARGET_COL], weights=x['population'])
                  if not x[INTERNAL_TARGET_COL].isnull().all()
                  and x['population'].sum() > 0
                  else 0.0
              ),
              'historical_avg_cumulative_norm_by_week_loc': (
                  np.average(
                      x['season_cumulative_norm_target_lagged'],
                      weights=x['population'],
                  )
                  if not x['season_cumulative_norm_target_lagged']
                  .isnull()
                  .all()
                  and x['population'].sum() > 0
                  else 0.0
              ),
              'count_observations_loc': x[INTERNAL_TARGET_COL].count(),
          })
      )
      .reset_index()
  )
  historical_climatology['count_observations_loc'] = historical_climatology[
      'count_observations_loc'
  ].astype(int)

  full_train_df_features = full_train_df_features.reset_index().merge(
      historical_climatology,
      on=['location', 'relative_season_week'],
      how='left',
  )
  full_train_df_features = full_train_df_features.merge(
      national_historical_climatology, on=['relative_season_week'], how='left'
  )

  # Fill NaNs for climatology counts, ensuring they are at least 0
  full_train_df_features['count_observations_loc'] = (
      full_train_df_features['count_observations_loc'].fillna(0).astype(int)
  )
  full_train_df_features['count_observations_national'] = (
      full_train_df_features['count_observations_national']
      .fillna(0)
      .astype(int)
  )

  # Calculate differences from historical averages with improved national fallback logic
  # Determine the raw baseline components for `admissions_diff_lag1`
  base_target_loc = full_train_df_features[
      'historical_avg_target_norm_by_week_loc'
  ]
  base_target_nat = full_train_df_features[
      'national_historical_avg_target_norm_by_week'
  ]
  count_loc_target = full_train_df_features['count_observations_loc']

  # Weighted blending of local and national climatology
  weight_local_target = np.minimum(1.0, count_loc_target / CLIMATOLOGY_WEIGHT_K)

  national_target_avg_fallback = base_target_nat.fillna(0.0)
  local_target_for_blend = base_target_loc.fillna(national_target_avg_fallback)

  full_train_df_features['baseline_target_for_diff'] = (
      weight_local_target * local_target_for_blend
  ) + ((1 - weight_local_target) * national_target_avg_fallback)

  # Determine the raw baseline components for `cumulative_diff_lagged`
  base_cumulative_loc = full_train_df_features[
      'historical_avg_cumulative_norm_by_week_loc'
  ]
  base_cumulative_nat = full_train_df_features[
      'national_historical_avg_cumulative_norm_by_week'
  ]

  weight_local_cumulative = np.minimum(
      1.0, count_loc_target / CLIMATOLOGY_WEIGHT_K
  )

  national_cumulative_avg_fallback = base_cumulative_nat.fillna(0.0)
  local_cumulative_for_blend = base_cumulative_loc.fillna(
      national_cumulative_avg_fallback
  )

  full_train_df_features['baseline_cumulative_for_diff'] = (
      weight_local_cumulative * local_cumulative_for_blend
  ) + ((1 - weight_local_cumulative) * national_cumulative_avg_fallback)

  # Final check for any remaining NaNs in blended baselines
  full_train_df_features['baseline_target_for_diff'] = full_train_df_features[
      'baseline_target_for_diff'
  ].fillna(0.0)
  full_train_df_features['baseline_cumulative_for_diff'] = (
      full_train_df_features['baseline_cumulative_for_diff'].fillna(0.0)
  )

  full_train_df_features['current_vs_historical_admissions_diff_lag1'] = (
      full_train_df_features['lag_1_norm']
      - full_train_df_features['baseline_target_for_diff']
  )
  full_train_df_features['current_vs_historical_cumulative_diff_lagged'] = (
      full_train_df_features['season_cumulative_norm_target_lagged']
      - full_train_df_features['baseline_cumulative_for_diff']
  )

  # Fill any NaNs from the difference features (e.g., very early in history where lag_1_norm is NaN)
  full_train_df_features[[
      'current_vs_historical_admissions_diff_lag1',
      'current_vs_historical_cumulative_diff_lagged',
  ]] = full_train_df_features[[
      'current_vs_historical_admissions_diff_lag1',
      'current_vs_historical_cumulative_diff_lagged',
  ]].fillna(
      0.0
  )

  # NEW FEATURE (GPR x4 inspiration): Exponentially Weighted Moving Average (EWMA) of severity difference features
  # These are calculated based on *lagged* differences within each season for a more dynamic severity signal.
  full_train_df_features[
      'ewma_current_vs_historical_admissions_diff_lagged'
  ] = full_train_df_features.groupby(['location', 'season_year_start'])[
      'current_vs_historical_admissions_diff_lag1'
  ].transform(
      lambda x: x.shift(1)
      .ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=1)
      .mean()
  )
  full_train_df_features[
      'ewma_current_vs_historical_cumulative_diff_lagged'
  ] = full_train_df_features.groupby(['location', 'season_year_start'])[
      'current_vs_historical_cumulative_diff_lagged'
  ].transform(
      lambda x: x.shift(1)
      .ewm(alpha=EWMA_ALPHA, adjust=False, min_periods=1)
      .mean()
  )
  full_train_df_features[
      'ewma_current_vs_historical_admissions_diff_lagged'
  ] = full_train_df_features[
      'ewma_current_vs_historical_admissions_diff_lagged'
  ].fillna(
      0.0
  )
  full_train_df_features[
      'ewma_current_vs_historical_cumulative_diff_lagged'
  ] = full_train_df_features[
      'ewma_current_vs_historical_cumulative_diff_lagged'
  ].fillna(
      0.0
  )

  # --- HYBRID SEVERITY DYNAMICS FEATURES ---
  # 1. NEW FEATURE: ewma_admissions_diff_deviation_lagged (Dynamic Severity Deviation)
  full_train_df_features['ewma_admissions_diff_deviation_lagged'] = (
      full_train_df_features['current_vs_historical_admissions_diff_lag1']
      - full_train_df_features[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ]
  )
  full_train_df_features['ewma_admissions_diff_deviation_lagged'] = (
      full_train_df_features['ewma_admissions_diff_deviation_lagged'].fillna(
          0.0
      )
  )

  # 2. NEW FEATURE: rolling_std_current_vs_historical_admissions_diff_lagged (Severity Volatility)
  full_train_df_features[
      'rolling_std_current_vs_historical_admissions_diff_lagged'
  ] = full_train_df_features.groupby(['location', 'season_year_start'])[
      'current_vs_historical_admissions_diff_lag1'
  ].transform(
      lambda x: x.shift(1).rolling(ROLLING_MEAN_WINDOW, min_periods=1).std()
  )
  full_train_df_features[
      'rolling_std_current_vs_historical_admissions_diff_lagged'
  ] = full_train_df_features[
      'rolling_std_current_vs_historical_admissions_diff_lagged'
  ].fillna(
      0.0
  )

  # NEW IMPROVEMENT: Add Interaction Features for Severity
  full_train_df_features['rel_week_x_lag1_norm'] = (
      full_train_df_features['relative_season_week']
      * full_train_df_features['lag_1_norm']
  )
  full_train_df_features['rel_week_x_current_vs_hist_admissions'] = (
      full_train_df_features['relative_season_week']
      * full_train_df_features['current_vs_historical_admissions_diff_lag1']
  )
  full_train_df_features['rel_week_x_current_vs_hist_cumulative'] = (
      full_train_df_features['relative_season_week']
      * full_train_df_features['current_vs_historical_cumulative_diff_lagged']
  )
  full_train_df_features['lag1_x_current_vs_hist_admissions'] = (
      full_train_df_features['lag_1_norm']
      * full_train_df_features['current_vs_historical_admissions_diff_lag1']
  )
  full_train_df_features['lag1_x_current_vs_hist_cumulative'] = (
      full_train_df_features['lag_1_norm']
      * full_train_df_features['current_vs_historical_cumulative_diff_lagged']
  )
  # NEW INTERACTION FEATURE: Interaction between the two main severity difference features
  full_train_df_features[
      'current_vs_historical_admissions_x_cumulative_diff'
  ] = (
      full_train_df_features['current_vs_historical_admissions_diff_lag1']
      * full_train_df_features['current_vs_historical_cumulative_diff_lagged']
  )

  # More nuanced interactions with EWMA severity differences
  full_train_df_features['rel_week_x_ewma_admissions_diff'] = (
      full_train_df_features['relative_season_week']
      * full_train_df_features[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ]
  )
  full_train_df_features['rel_week_x_ewma_cumulative_diff'] = (
      full_train_df_features['relative_season_week']
      * full_train_df_features[
          'ewma_current_vs_historical_cumulative_diff_lagged'
      ]
  )
  full_train_df_features['lag1_x_ewma_admissions_diff'] = (
      full_train_df_features['lag_1_norm']
      * full_train_df_features[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ]
  )
  full_train_df_features['lag1_x_ewma_cumulative_diff'] = (
      full_train_df_features['lag_1_norm']
      * full_train_df_features[
          'ewma_current_vs_historical_cumulative_diff_lagged'
      ]
  )
  # NEW INTERACTION FEATURE: Interaction between the two EWMA severity features
  full_train_df_features['ewma_admissions_x_ewma_cumulative_diff'] = (
      full_train_df_features[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ]
      * full_train_df_features[
          'ewma_current_vs_historical_cumulative_diff_lagged'
      ]
  )
  # NEW INTERACTION FEATURE: EWMA Deviation x Relative Season Week (ADDED THIS ONE)
  full_train_df_features[
      'ewma_admissions_diff_deviation_lagged_x_relative_season_week'
  ] = (
      full_train_df_features['ewma_admissions_diff_deviation_lagged']
      * full_train_df_features['relative_season_week']
  )

  # NEW: Population interaction features
  full_train_df_features['population_log_x_lag_1_norm'] = (
      full_train_df_features['population_log']
      * full_train_df_features['lag_1_norm']
  )
  full_train_df_features[
      'population_log_x_current_vs_historical_admissions_diff_lag1'
  ] = (
      full_train_df_features['population_log']
      * full_train_df_features['current_vs_historical_admissions_diff_lag1']
  )
  # More population interactions
  full_train_df_features['population_log_x_relative_season_week'] = (
      full_train_df_features['population_log']
      * full_train_df_features['relative_season_week']
  )
  full_train_df_features['population_log_x_sin_relative_season_week'] = (
      full_train_df_features['population_log']
      * full_train_df_features['sin_relative_season_week']
  )
  full_train_df_features['population_log_x_cos_relative_season_week'] = (
      full_train_df_features['population_log']
      * full_train_df_features['cos_relative_season_week']
  )

  # --- NEW INTER-SEASON CARRY-OVER FEATURES ---
  # 1. Previous season's end-of-season average normalized target
  prev_season_end_avg_lookup = {}
  for loc in full_train_df_features['location'].unique():
    for season_year in full_train_df_features['season_year_start'].unique():
      # Get data for the defined 'end of season' relative weeks
      end_of_season_data = full_train_df_features[
          (full_train_df_features['location'] == loc)
          & (full_train_df_features['season_year_start'] == season_year)
          & (
              full_train_df_features['relative_season_week'].isin(
                  END_SEASON_RELATIVE_WEEKS
              )
          )
      ]
      if not end_of_season_data.empty:
        prev_season_end_avg_lookup[(loc, season_year)] = end_of_season_data[
            INTERNAL_TARGET_COL
        ].mean()
      else:
        prev_season_end_avg_lookup[(loc, season_year)] = 0.0

  full_train_df_features['previous_season_end_of_season_avg_norm_target'] = (
      full_train_df_features.apply(
          lambda row: prev_season_end_avg_lookup.get(
              (row['location'], row['season_year_start'] - 1), 0.0
          ),
          axis=1,
      )
  )
  full_train_df_features['previous_season_end_of_season_avg_norm_target'] = (
      full_train_df_features[
          'previous_season_end_of_season_avg_norm_target'
      ].fillna(0.0)
  )

  # 2. Previous season's maximum EWMA admissions difference (peak severity anomaly)
  prev_season_max_ewma_lookup = {}
  for loc in full_train_df_features['location'].unique():
    for season_year in full_train_df_features['season_year_start'].unique():
      season_ewma_data = full_train_df_features[
          (full_train_df_features['location'] == loc)
          & (full_train_df_features['season_year_start'] == season_year)
      ]
      if not season_ewma_data[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ].empty:
        prev_season_max_ewma_lookup[(loc, season_year)] = season_ewma_data[
            'ewma_current_vs_historical_admissions_diff_lagged'
        ].max()
      else:
        prev_season_max_ewma_lookup[(loc, season_year)] = 0.0

  full_train_df_features['previous_season_max_ewma_admissions_diff'] = (
      full_train_df_features.apply(
          lambda row: prev_season_max_ewma_lookup.get(
              (row['location'], row['season_year_start'] - 1), 0.0
          ),
          axis=1,
      )
  )
  full_train_df_features['previous_season_max_ewma_admissions_diff'] = (
      full_train_df_features['previous_season_max_ewma_admissions_diff'].fillna(
          0.0
      )
  )

  # NEW: Horizon interaction features
  full_train_df_features['horizon_x_lag_1_norm'] = (
      full_train_df_features['horizon'] * full_train_df_features['lag_1_norm']
  )
  full_train_df_features['horizon_x_relative_season_week'] = (
      full_train_df_features['horizon']
      * full_train_df_features['relative_season_week']
  )
  full_train_df_features['horizon_x_ewma_admissions_diff_lagged'] = (
      full_train_df_features['horizon']
      * full_train_df_features[
          'ewma_current_vs_historical_admissions_diff_lagged'
      ]
  )
  # NEW INTERACTION FEATURE: Population-Horizon Interaction
  full_train_df_features['population_log_x_horizon'] = (
      full_train_df_features['population_log']
      * full_train_df_features['horizon']
  )

  # Fill any NaNs created by interactions with 0.0
  # Consolidate fillna calls for readability and efficiency
  interaction_features = [
      'rel_week_x_lag1_norm',
      'rel_week_x_current_vs_hist_admissions',
      'rel_week_x_current_vs_hist_cumulative',
      'lag1_x_current_vs_hist_admissions',
      'lag1_x_current_vs_hist_cumulative',
      'current_vs_historical_admissions_x_cumulative_diff',
      'rel_week_x_ewma_admissions_diff',
      'rel_week_x_ewma_cumulative_diff',
      'lag1_x_ewma_admissions_diff',
      'lag1_x_ewma_cumulative_diff',
      'ewma_admissions_x_ewma_cumulative_diff',
      'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
      'population_log_x_lag_1_norm',
      'population_log_x_current_vs_historical_admissions_diff_lag1',
      'population_log_x_relative_season_week',
      'population_log_x_sin_relative_season_week',
      'population_log_x_cos_relative_season_week',
      'ewma_admissions_diff_deviation_lagged',
      'rolling_std_current_vs_historical_admissions_diff_lagged',
      'previous_season_end_of_season_avg_norm_target',
      'previous_season_max_ewma_admissions_diff',
      'horizon_x_lag_1_norm',
      'horizon_x_relative_season_week',
      'horizon_x_ewma_admissions_diff_lagged',
      'population_log_x_horizon',
  ]
  for feat_col in interaction_features:
    if feat_col in full_train_df_features.columns:
      full_train_df_features[feat_col] = full_train_df_features[
          feat_col
      ].fillna(0.0)

  full_train_df_features = full_train_df_features.set_index(
      ['location', 'target_end_date']
  )

  # IMPROVEMENT: Corrected National Previous Season Peak Calculation
  national_actual_target_norm_temp = (
      full_train_df_features.reset_index()
      .groupby('target_end_date')
      .apply(
          lambda x: pd.Series({
              'national_actual_target_norm': (
                  np.average(x[INTERNAL_TARGET_COL], weights=x['population'])
                  if not x[INTERNAL_TARGET_COL].isnull().all()
                  and x['population'].sum() > 0
                  else 0.0
              )
          })
      )
      .reset_index()
  )
  national_actual_target_norm_temp['season_year_start'] = (
      national_actual_target_norm_temp['target_end_date'].apply(get_season_year)
  )

  national_season_actual_peak_lookup = (
      national_actual_target_norm_temp.groupby('season_year_start')[
          'national_actual_target_norm'
      ]
      .max()
      .to_dict()
  )

  full_train_df_features[
      'national_previous_season_peak_norm_target'
  ] = full_train_df_features['season_year_start'].apply(
      lambda s_year: national_season_actual_peak_lookup.get(s_year - 1, 0.0)
  )
  full_train_df_features['national_previous_season_peak_norm_target'] = (
      full_train_df_features[
          'national_previous_season_peak_norm_target'
      ].fillna(0.0)
  )

  # 2. National Aggregates (SIRS global context inspiration) - with POPULATION WEIGHTING
  # Define features to aggregate nationally.
  national_agg_features_to_derive = [
      f'lag_{i}_norm' for i in range(1, MAX_LAG + 1)
  ] + [
      f'rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm',
      'season_cumulative_norm_target_lagged',
      'admissions_growth_rate_log_diff_lag1',
      'admissions_acceleration_log_diff_lag1',
      'current_vs_historical_admissions_diff_lag1',
      'current_vs_historical_cumulative_diff_lagged',
      'ewma_current_vs_historical_admissions_diff_lagged',
      'ewma_current_vs_historical_cumulative_diff_lagged',
      'ewma_admissions_diff_deviation_lagged',
      'rolling_std_current_vs_historical_admissions_diff_lagged',
      'rel_week_x_lag1_norm',
      'rel_week_x_current_vs_hist_admissions',
      'rel_week_x_current_vs_hist_cumulative',
      'lag1_x_current_vs_hist_admissions',
      'lag1_x_current_vs_hist_cumulative',
      'current_vs_historical_admissions_x_cumulative_diff',
      'rel_week_x_ewma_admissions_diff',
      'rel_week_x_ewma_cumulative_diff',
      'lag1_x_ewma_admissions_diff',
      'lag1_x_ewma_cumulative_diff',
      'ewma_admissions_x_ewma_cumulative_diff',
      'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
      'population_log_x_lag_1_norm',
      'population_log_x_current_vs_historical_admissions_diff_lag1',
      'population_log_x_relative_season_week',
      'population_log_x_sin_relative_season_week',
      'population_log_x_cos_relative_season_week',
      'horizon_x_lag_1_norm',
      'horizon_x_relative_season_week',
      'horizon_x_ewma_admissions_diff_lagged',
      'population_log_x_horizon',
  ]

  # Calculate national aggregates on training data using population weighting
  national_aggs_df = (
      full_train_df_features.reset_index()
      .groupby('target_end_date')
      .apply(
          lambda x: pd.Series({
              f'national_{feat}': (
                  np.average(x[feat], weights=x['population'])
                  if not x[feat].isnull().all() and x['population'].sum() > 0
                  else 0.0
              )
              for feat in national_agg_features_to_derive
          })
      )
      .reset_index()
  )

  full_train_df_features = (
      full_train_df_features.reset_index()
      .merge(national_aggs_df, on='target_end_date', how='left')
      .set_index(['location', 'target_end_date'])
  )
  # Fill NaNs created by national merges (e.g. very early dates where no national data)
  for col in national_aggs_df.columns:
    if col != 'target_end_date':
      full_train_df_features[col] = full_train_df_features[col].fillna(0.0)

  full_train_df_features = full_train_df_features.reset_index()

  # Define features to be used in the model
  features = (
      [
          'population_log',
          'epiweek',
          'sin_epiweek',
          'cos_epiweek',
          'year',
          'location',
          'horizon',
          'relative_season_week',
          'relative_season_week_sq',
          'sin_relative_season_week',
          'cos_relative_season_week',
      ]
      + [f'lag_{i}_norm' for i in range(1, MAX_LAG + 1)]
      + [
          f'rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm',
          'season_cumulative_norm_target_lagged',
          'previous_season_peak_norm_target',
          'admissions_growth_rate_log_diff_lag1',
          'admissions_acceleration_log_diff_lag1',
          'current_vs_historical_admissions_diff_lag1',
          'current_vs_historical_cumulative_diff_lagged',
          'ewma_current_vs_historical_admissions_diff_lagged',
          'ewma_current_vs_historical_cumulative_diff_lagged',
          'ewma_admissions_diff_deviation_lagged',
          'rolling_std_current_vs_historical_admissions_diff_lagged',
          'rel_week_x_lag1_norm',
          'rel_week_x_current_vs_hist_admissions',
          'rel_week_x_current_vs_hist_cumulative',
          'lag1_x_current_vs_hist_admissions',
          'lag1_x_current_vs_hist_cumulative',
          'current_vs_historical_admissions_x_cumulative_diff',
          'rel_week_x_ewma_admissions_diff',
          'rel_week_x_ewma_cumulative_diff',
          'lag1_x_ewma_admissions_diff',
          'lag1_x_ewma_cumulative_diff',
          'ewma_admissions_x_ewma_cumulative_diff',
          'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
          'population_log_x_lag_1_norm',
          'population_log_x_current_vs_historical_admissions_diff_lag1',
          'population_log_x_relative_season_week',
          'population_log_x_sin_relative_season_week',
          'population_log_x_cos_relative_season_week',
          'national_previous_season_peak_norm_target',
          'previous_season_end_of_season_avg_norm_target',
          'previous_season_max_ewma_admissions_diff',
          'horizon_x_lag_1_norm',
          'horizon_x_relative_season_week',
          'horizon_x_ewma_admissions_diff_lagged',
          'population_log_x_horizon',
      ]
      + [f'national_{feat}' for feat in national_agg_features_to_derive]
  )

  categorical_features = ['location']

  train_features_df = full_train_df_features[features].copy()
  train_target_series = full_train_df_features[INTERNAL_TARGET_COL].copy()

  # Fill NaNs in features (created by lags/rolling means/season features at start of data) with 0.0
  train_features_df = train_features_df.fillna(0.0)

  # --- 3. Model Training (Median and Scale Models) ---

  # 3.1 Train a LightGBM model for the MEDIAN forecast
  lgbm_median_params = {
      'objective': (
          'quantile'
      ),  # Using quantile objective with alpha=0.5 for median
      'alpha': 0.5,
      'metric': 'quantile',
      'n_estimators': 2200,  # Increased from 2000
      'learning_rate': 0.006,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'bagging_freq': 1,
      'lambda_l1': 0.60,
      'lambda_l2': 0.60,
      'num_leaves': 70,  # Increased from 60
      'min_child_samples': 75,  # Decreased from 85
      'verbose': -1,
      'n_jobs': -1,
      'seed': 42,
  }
  lgbm_median = lgb.LGBMRegressor(**lgbm_median_params)
  lgbm_median.fit(
      train_features_df,
      train_target_series,
      categorical_feature=categorical_features,
  )

  # 3.2 Calculate residuals from the median model
  train_median_preds = lgbm_median.predict(train_features_df)
  train_residuals = train_target_series - train_median_preds

  # 3.3 Train a LightGBM model for the SCALE (magnitude of residuals, approximating std)
  # Target for scale model: square root of absolute residuals to ensure non-negativity and scale transformation
  # Add a small epsilon to avoid sqrt(0) issues and to ensure minimum uncertainty
  train_scale_target = np.sqrt(np.abs(train_residuals) + 1e-6)

  lgbm_scale_params = {
      'objective': 'regression',  # Standard regression to predict scale
      'metric': 'rmse',
      'n_estimators': 1900,  # Increased from 1700
      'learning_rate': 0.006,
      'feature_fraction': 0.75,
      'bagging_fraction': 0.75,
      'bagging_freq': 1,
      'lambda_l1': 0.60,
      'lambda_l2': 0.60,
      'num_leaves': 65,  # Increased from 55
      'min_child_samples': 70,  # Decreased from 80
      'verbose': -1,
      'n_jobs': -1,
      'seed': 42,
  }
  lgbm_scale = lgb.LGBMRegressor(**lgbm_scale_params)
  lgbm_scale.fit(
      train_features_df,
      train_scale_target,
      categorical_feature=categorical_features,
  )

  # --- 4. Iterative Prediction for Test Data (Horizon by Horizon with Dynamic Lags, Season Features, and National Context) ---

  # Create a scaffold for test features, initially with NaN for dynamic features
  test_x_scaffold = create_base_features(test_x.copy())

  # Prepare a dictionary to store the latest known/predicted values for each location
  # This will be updated iteratively with median forecasts for future lags, in INTERNAL_TARGET_COL space
  location_latest_values: Dict[Tuple[int, pd.Timestamp], float] = {}
  for (loc, date), val in full_train_df_features.set_index(
      ['location', 'target_end_date']
  )[INTERNAL_TARGET_COL].items():
    location_latest_values[(loc, date)] = val

  # Initialize location_season_cumulative_tracker
  location_season_cumulative_tracker: Dict[
      Tuple[int, int, pd.Timestamp], float
  ] = {}
  for _, row in full_train_df_features.reset_index().iterrows():
    loc = row['location']
    season_year = row['season_year_start']
    # Store cumulative *up to and including* the current week's actual value
    cumulative_val = (
        row['season_cumulative_norm_target_lagged'] + row[INTERNAL_TARGET_COL]
    )
    location_season_cumulative_tracker[
        (loc, season_year, row['target_end_date'])
    ] = cumulative_val

  # Initialize national historical context from full_train_df_features
  national_historical_context: Dict[pd.Timestamp, Dict[str, float]] = {}
  train_end_date = full_train_df_features['target_end_date'].max()

  # Populate national_historical_context from the population-weighted national_aggs_df
  # Add national_previous_season_peak_norm_target to national_aggs_df before merging, for historical context.
  national_aggs_df[
      'national_previous_season_peak_norm_target'
  ] = national_aggs_df['target_end_date'].apply(
      lambda d: national_season_actual_peak_lookup.get(
          get_season_year(d) - 1, 0.0
      )
  )

  for _, row in national_aggs_df.iterrows():
    national_historical_context[row['target_end_date']] = {
        col: row[col]
        for col in national_aggs_df.columns
        if col not in ['target_end_date']
    }

  # This will store national predicted features (actual means of state-level medians and derived from them)
  # for dates beyond train_end_date
  national_predicted_context: Dict[pd.Timestamp, Dict[str, float]] = {}

  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )

  # Sort test_x_scaffold by target_end_date to process horizons sequentially
  sorted_test_dates = sorted(test_x_scaffold['target_end_date'].unique())

  # Create a Series mapping the original test_x index to (location, target_end_date, population, horizon) for easy lookup
  test_idx_to_loc_date = test_x.set_index(test_x.index)[
      ['location', 'target_end_date', 'population', 'horizon']
  ]

  # Helper to get *any* national feature for a given date from either historical or predicted context
  def _get_national_feature_for_date(
      d, feature_name, fallback_value = 0.0
  ):
    if d in national_predicted_context:
      return national_predicted_context[d].get(feature_name, fallback_value)
    if d in national_historical_context:
      return national_historical_context[d].get(feature_name, fallback_value)
    return fallback_value

  # Track latest EWMA severity differences for calculation in loop
  # Stored as: { (location, season_year_start): latest_ewma_value }
  location_ewma_admissions_tracker: Dict[Tuple[int, int], float] = {}
  location_ewma_cumulative_tracker: Dict[Tuple[int, int], float] = {}
  # For rolling std, we need to store past actual differences, not EWMA
  # Stored as: { (location, target_end_date): current_vs_historical_admissions_diff }
  location_current_vs_historical_admissions_diff_tracker: Dict[
      Tuple[int, pd.Timestamp], float
  ] = {}

  # Populate initial EWMA values from training data (the EWMA for train_end_date)
  for loc in locations['location'].unique():
    loc_train_features = full_train_df_features[
        full_train_df_features['location'] == loc
    ].set_index('target_end_date')

    current_season_year_at_train_end = get_season_year(train_end_date)

    ad_ewma_series = loc_train_features[
        'ewma_current_vs_historical_admissions_diff_lagged'
    ].dropna()
    if not ad_ewma_series.empty and train_end_date in ad_ewma_series.index:
      location_ewma_admissions_tracker[
          (loc, current_season_year_at_train_end)
      ] = ad_ewma_series.loc[train_end_date]
    else:
      location_ewma_admissions_tracker[
          (loc, current_season_year_at_train_end)
      ] = 0.0

    cum_ewma_series = loc_train_features[
        'ewma_current_vs_historical_cumulative_diff_lagged'
    ].dropna()
    if not cum_ewma_series.empty and train_end_date in cum_ewma_series.index:
      location_ewma_cumulative_tracker[
          (loc, current_season_year_at_train_end)
      ] = cum_ewma_series.loc[train_end_date]
    else:
      location_ewma_cumulative_tracker[
          (loc, current_season_year_at_train_end)
      ] = 0.0

    # Initialize current_vs_historical_admissions_diff_tracker for rolling_std
    # Store values for ROLLING_MEAN_WINDOW weeks prior to train_end_date
    for i in range(ROLLING_MEAN_WINDOW):
      hist_date = train_end_date - pd.Timedelta(weeks=i)
      if hist_date in loc_train_features.index:
        location_current_vs_historical_admissions_diff_tracker[
            (loc, hist_date)
        ] = loc_train_features.loc[
            hist_date, 'current_vs_historical_admissions_diff_lag1'
        ]

  for forecast_date in sorted_test_dates:
    current_horizon_test_rows_idx = test_x_scaffold[
        test_x_scaffold['target_end_date'] == forecast_date
    ].index
    current_horizon_test_x_batch = test_x_scaffold.loc[
        current_horizon_test_rows_idx
    ].copy()

    for idx in current_horizon_test_x_batch.index:
      loc = current_horizon_test_x_batch.loc[idx, 'location']
      current_season_year_start_forecast = current_horizon_test_x_batch.loc[
          idx, 'season_year_start'
      ]
      relative_season_week = current_horizon_test_x_batch.loc[
          idx, 'relative_season_week'
      ]
      horizon_val = current_horizon_test_x_batch.loc[idx, 'horizon']

      # Populate state-specific lag features (for INTERNAL_TARGET_COL)
      lag_values = {}
      for lag in range(1, MAX_LAG + 1):
        lag_date = forecast_date - pd.Timedelta(weeks=lag)
        lag_value = location_latest_values.get((loc, lag_date), 0.0)
        current_horizon_test_x_batch.loc[idx, f'lag_{lag}_norm'] = lag_value
        lag_values[lag] = lag_value

      # Populate state-specific rolling mean feature (for INTERNAL_TARGET_COL)
      rolling_mean_vals = [
          lag_values.get(i, 0.0) for i in range(1, ROLLING_MEAN_WINDOW + 1)
      ]
      current_horizon_test_x_batch.loc[
          idx, f'rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm'
      ] = (np.mean(rolling_mean_vals) if rolling_mean_vals else 0.0)

      # Populate state-specific season_cumulative_norm_target_lagged for test data
      prev_week_cumulative_sum = location_season_cumulative_tracker.get(
          (
              loc,
              current_season_year_start_forecast,
              forecast_date - pd.Timedelta(weeks=1),
          ),
          0.0,
      )
      current_horizon_test_x_batch.loc[
          idx, 'season_cumulative_norm_target_lagged'
      ] = prev_week_cumulative_sum

      # Populate state-specific previous_season_peak_norm_target for test data
      prev_season_year_for_lookup = current_season_year_start_forecast - 1
      current_horizon_test_x_batch.loc[
          idx, 'previous_season_peak_norm_target'
      ] = previous_season_peak_lookup_norm.get(
          (loc, prev_season_year_for_lookup), 0.0
      )

      # Populate admissions_growth_rate_log_diff_lag1 for test data
      current_horizon_test_x_batch.loc[
          idx, 'admissions_growth_rate_log_diff_lag1'
      ] = lag_values.get(1, 0.0) - lag_values.get(2, 0.0)

      # Populate admissions_acceleration_log_diff_lag1 for test data
      current_horizon_test_x_batch.loc[
          idx, 'admissions_acceleration_log_diff_lag1'
      ] = (lag_values.get(1, 0.0) - lag_values.get(2, 0.0)) - (
          lag_values.get(2, 0.0) - lag_values.get(3, 0.0)
      )

      # Populate current_vs_historical_admissions_diff_lag1 and current_vs_historical_cumulative_diff_lagged
      # Use population-weighted blending of local and national climatology

      # Get local historical climatology and its observation count
      loc_climat_row = historical_climatology.loc[
          (historical_climatology['location'] == loc)
          & (
              historical_climatology['relative_season_week']
              == relative_season_week
          )
      ]
      local_historical_avg_target = (
          loc_climat_row['historical_avg_target_norm_by_week_loc'].values[0]
          if not loc_climat_row.empty
          else np.nan
      )
      local_historical_avg_cumulative = (
          loc_climat_row['historical_avg_cumulative_norm_by_week_loc'].values[0]
          if not loc_climat_row.empty
          else np.nan
      )
      local_obs_count = (
          loc_climat_row['count_observations_loc'].values[0]
          if not loc_climat_row.empty
          else 0
      )

      # Get national historical climatology
      nat_climat_row = national_historical_climatology.loc[
          national_historical_climatology['relative_season_week']
          == relative_season_week
      ]
      national_historical_avg_target = (
          nat_climat_row['national_historical_avg_target_norm_by_week'].values[
              0
          ]
          if not nat_climat_row.empty
          else np.nan
      )
      national_historical_avg_cumulative = (
          nat_climat_row[
              'national_historical_avg_cumulative_norm_by_week'
          ].values[0]
          if not nat_climat_row.empty
          else np.nan
      )

      # Blending logic: Weighted blend based on local observation count
      weight_local_target_pred = min(
          1.0, local_obs_count / CLIMATOLOGY_WEIGHT_K
      )

      national_target_avg_fallback_pred = (
          national_historical_avg_target
          if not np.isnan(national_historical_avg_target)
          else 0.0
      )
      local_target_for_blend_pred = (
          local_historical_avg_target
          if not np.isnan(local_historical_avg_target)
          else national_target_avg_fallback_pred
      )

      baseline_target_for_diff = (
          weight_local_target_pred * local_target_for_blend_pred
      ) + ((1 - weight_local_target_pred) * national_target_avg_fallback_pred)

      weight_local_cumulative_pred = min(
          1.0, local_obs_count / CLIMATOLOGY_WEIGHT_K
      )

      national_cumulative_avg_fallback_pred = (
          national_historical_avg_cumulative
          if not np.isnan(national_historical_avg_cumulative)
          else 0.0
      )
      local_cumulative_for_blend_pred = (
          local_historical_avg_cumulative
          if not np.isnan(local_historical_avg_cumulative)
          else national_cumulative_avg_fallback_pred
      )

      baseline_cumulative_for_diff = (
          weight_local_cumulative_pred * local_cumulative_for_blend_pred
      ) + (
          (1 - weight_local_cumulative_pred)
          * national_cumulative_avg_fallback_pred
      )

      # Final safety check for any remaining NaNs in blended baselines
      baseline_target_for_diff = (
          0.0
          if np.isnan(baseline_target_for_diff)
          else baseline_target_for_diff
      )
      baseline_cumulative_for_diff = (
          0.0
          if np.isnan(baseline_cumulative_for_diff)
          else baseline_cumulative_for_diff
      )

      current_vs_historical_admissions_diff = (
          current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
          - baseline_target_for_diff
      )
      current_vs_historical_cumulative_diff = (
          current_horizon_test_x_batch.loc[
              idx, 'season_cumulative_norm_target_lagged'
          ]
          - baseline_cumulative_for_diff
      )

      current_horizon_test_x_batch.loc[
          idx, 'current_vs_historical_admissions_diff_lag1'
      ] = current_vs_historical_admissions_diff
      current_horizon_test_x_batch.loc[
          idx, 'current_vs_historical_cumulative_diff_lagged'
      ] = current_vs_historical_cumulative_diff

      # Populate EWMA of severity differences using the trackers
      prev_ewma_admissions = location_ewma_admissions_tracker.get(
          (loc, current_season_year_start_forecast), 0.0
      )
      prev_ewma_cumulative = location_ewma_cumulative_tracker.get(
          (loc, current_season_year_start_forecast), 0.0
      )
      current_horizon_test_x_batch.loc[
          idx, 'ewma_current_vs_historical_admissions_diff_lagged'
      ] = prev_ewma_admissions
      current_horizon_test_x_batch.loc[
          idx, 'ewma_current_vs_historical_cumulative_diff_lagged'
      ] = prev_ewma_cumulative

      # NEW FEATURE: ewma_admissions_diff_deviation_lagged
      current_horizon_test_x_batch.loc[
          idx, 'ewma_admissions_diff_deviation_lagged'
      ] = (current_vs_historical_admissions_diff - prev_ewma_admissions)

      # NEW FEATURE: rolling_std_current_vs_historical_admissions_diff_lagged
      rolling_std_vals = []
      for i in range(ROLLING_MEAN_WINDOW):
        hist_diff_date = forecast_date - pd.Timedelta(weeks=1 + i)
        rolling_std_vals.append(
            location_current_vs_historical_admissions_diff_tracker.get(
                (loc, hist_diff_date), 0.0
            )
        )

      current_horizon_test_x_batch.loc[
          idx, 'rolling_std_current_vs_historical_admissions_diff_lagged'
      ] = (np.std(rolling_std_vals) if len(rolling_std_vals) > 1 else 0.0)

      # Add Interaction Features for test data
      current_horizon_test_x_batch.loc[idx, 'rel_week_x_lag1_norm'] = (
          relative_season_week
          * current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
      )
      current_horizon_test_x_batch.loc[
          idx, 'rel_week_x_current_vs_hist_admissions'
      ] = (
          relative_season_week
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_admissions_diff_lag1'
          ]
      )
      current_horizon_test_x_batch.loc[
          idx, 'rel_week_x_current_vs_hist_cumulative'
      ] = (
          relative_season_week
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_cumulative_diff_lagged'
          ]
      )
      current_horizon_test_x_batch.loc[
          idx, 'lag1_x_current_vs_hist_admissions'
      ] = (
          current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_admissions_diff_lag1'
          ]
      )
      current_horizon_test_x_batch.loc[
          idx, 'lag1_x_current_vs_hist_cumulative'
      ] = (
          current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_cumulative_diff_lagged'
          ]
      )
      # NEW INTERACTION FEATURE:
      current_horizon_test_x_batch.loc[
          idx, 'current_vs_historical_admissions_x_cumulative_diff'
      ] = (
          current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_admissions_diff_lag1'
          ]
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_cumulative_diff_lagged'
          ]
      )

      # Populate more nuanced interaction features with EWMA for test data
      current_horizon_test_x_batch.loc[
          idx, 'rel_week_x_ewma_admissions_diff'
      ] = (
          relative_season_week
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_admissions_diff_lagged'
          ]
      )
      current_horizon_test_x_batch.loc[
          idx, 'rel_week_x_ewma_cumulative_diff'
      ] = (
          relative_season_week
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_cumulative_diff_lagged'
          ]
      )
      current_horizon_test_x_batch.loc[idx, 'lag1_x_ewma_admissions_diff'] = (
          current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_admissions_diff_lagged'
          ]
      )
      current_horizon_test_x_batch.loc[idx, 'lag1_x_ewma_cumulative_diff'] = (
          current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_cumulative_diff_lagged'
          ]
      )
      # NEW INTERACTION FEATURE: Interaction between the two EWMA severity features
      current_horizon_test_x_batch.loc[
          idx, 'ewma_admissions_x_ewma_cumulative_diff'
      ] = (
          current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_admissions_diff_lagged'
          ]
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_cumulative_diff_lagged'
          ]
      )
      # NEW INTERACTION FEATURE: EWMA Deviation x Relative Season Week (ADDED THIS ONE)
      current_horizon_test_x_batch.loc[
          idx, 'ewma_admissions_diff_deviation_lagged_x_relative_season_week'
      ] = (
          current_horizon_test_x_batch.loc[
              idx, 'ewma_admissions_diff_deviation_lagged'
          ]
          * relative_season_week
      )

      # Populate Population interaction features for test data
      population_log_val = current_horizon_test_x_batch.loc[
          idx, 'population_log'
      ]
      current_horizon_test_x_batch.loc[idx, 'population_log_x_lag_1_norm'] = (
          population_log_val
          * current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
      )
      current_horizon_test_x_batch.loc[
          idx, 'population_log_x_current_vs_historical_admissions_diff_lag1'
      ] = (
          population_log_val
          * current_horizon_test_x_batch.loc[
              idx, 'current_vs_historical_admissions_diff_lag1'
          ]
      )
      # More population interactions
      current_horizon_test_x_batch.loc[
          idx, 'population_log_x_relative_season_week'
      ] = (
          population_log_val
          * current_horizon_test_x_batch.loc[idx, 'relative_season_week']
      )
      current_horizon_test_x_batch.loc[
          idx, 'population_log_x_sin_relative_season_week'
      ] = (
          population_log_val
          * current_horizon_test_x_batch.loc[idx, 'sin_relative_season_week']
      )
      current_horizon_test_x_batch.loc[
          idx, 'population_log_x_cos_relative_season_week'
      ] = (
          population_log_val
          * current_horizon_test_x_batch.loc[idx, 'cos_relative_season_week']
      )

      # Populate Horizon interaction features for test data
      current_horizon_test_x_batch.loc[idx, 'horizon_x_lag_1_norm'] = (
          horizon_val * current_horizon_test_x_batch.loc[idx, 'lag_1_norm']
      )
      current_horizon_test_x_batch.loc[
          idx, 'horizon_x_relative_season_week'
      ] = (
          horizon_val
          * current_horizon_test_x_batch.loc[idx, 'relative_season_week']
      )
      current_horizon_test_x_batch.loc[
          idx, 'horizon_x_ewma_admissions_diff_lagged'
      ] = (
          horizon_val
          * current_horizon_test_x_batch.loc[
              idx, 'ewma_current_vs_historical_admissions_diff_lagged'
          ]
      )
      # NEW INTERACTION FEATURE: Population-Horizon Interaction
      current_horizon_test_x_batch.loc[idx, 'population_log_x_horizon'] = (
          current_horizon_test_x_batch.loc[idx, 'population_log']
          * current_horizon_test_x_batch.loc[idx, 'horizon']
      )

      # ADDED: Populate national_previous_season_peak_norm_target for test data
      current_horizon_test_x_batch.loc[
          idx, 'national_previous_season_peak_norm_target'
      ] = national_season_actual_peak_lookup.get(
          current_season_year_start_forecast - 1, 0.0
      )

      # NEW: Populate previous_season_end_of_season_avg_norm_target for test data
      current_horizon_test_x_batch.loc[
          idx, 'previous_season_end_of_season_avg_norm_target'
      ] = prev_season_end_avg_lookup.get(
          (loc, current_season_year_start_forecast - 1), 0.0
      )

      # NEW: Populate previous_season_max_ewma_admissions_diff for test data
      current_horizon_test_x_batch.loc[
          idx, 'previous_season_max_ewma_admissions_diff'
      ] = prev_season_max_ewma_lookup.get(
          (loc, current_season_year_start_forecast - 1), 0.0
      )

      # --- Populate national-level features for test data, using a consistent lookup ---

      # National lagged features (e.g., national_lag_1_norm for current forecast_date is the national_lag_1_norm from forecast_date - 1 weeks)
      for lag in range(1, MAX_LAG + 1):
        lag_date = forecast_date - pd.Timedelta(weeks=lag)
        current_horizon_test_x_batch.loc[idx, f'national_lag_{lag}_norm'] = (
            _get_national_feature_for_date(lag_date, 'national_lag_1_norm')
        )

      # National Rolling Mean (lagged) - derive from national_lag_1_norm history
      national_rolling_vals = [
          _get_national_feature_for_date(
              forecast_date - pd.Timedelta(weeks=i), 'national_lag_1_norm'
          )
          for i in range(1, ROLLING_MEAN_WINDOW + 1)
      ]
      current_horizon_test_x_batch.loc[
          idx, f'national_rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm'
      ] = (np.mean(national_rolling_vals) if national_rolling_vals else 0.0)

      # National Growth Rate (lagged) - derive from national_lag_1_norm history
      national_lag1_target = _get_national_feature_for_date(
          forecast_date - pd.Timedelta(weeks=1), 'national_lag_1_norm'
      )
      national_lag2_target = _get_national_feature_for_date(
          forecast_date - pd.Timedelta(weeks=2), 'national_lag_1_norm'
      )
      current_horizon_test_x_batch.loc[
          idx, 'national_admissions_growth_rate_log_diff_lag1'
      ] = (national_lag1_target - national_lag2_target)

      # National Acceleration (lagged) - derive from national_lag_1_norm history
      national_lag3_target = _get_national_feature_for_date(
          forecast_date - pd.Timedelta(weeks=3), 'national_lag_1_norm'
      )
      current_horizon_test_x_batch.loc[
          idx, 'national_admissions_acceleration_log_diff_lag1'
      ] = (national_lag1_target - national_lag2_target) - (
          national_lag2_target - national_lag3_target
      )

      # National Season Cumulative (lagged) - derive from national_lag_1_norm history
      national_cumulative_sum = 0.0
      current_nat_season_year = get_season_year(forecast_date)
      temp_date_for_cumulative = forecast_date - pd.Timedelta(weeks=1)
      # Loop backwards to sum previous 'national_lag_1_norm' values within the current flu season
      while (
          get_season_year(temp_date_for_cumulative) == current_nat_season_year
          and temp_date_for_cumulative > train_end_date
      ):
        national_cumulative_sum += _get_national_feature_for_date(
            temp_date_for_cumulative, 'national_lag_1_norm'
        )
        temp_date_for_cumulative -= pd.Timedelta(weeks=1)
      # Add historical national cumulative from the training data for the previous week if available
      if forecast_date - pd.Timedelta(weeks=1) <= train_end_date:
        national_cumulative_sum += _get_national_feature_for_date(
            forecast_date - pd.Timedelta(weeks=1),
            'national_season_cumulative_norm_target_lagged',
        )

      current_horizon_test_x_batch.loc[
          idx, 'national_season_cumulative_norm_target_lagged'
      ] = national_cumulative_sum

      # For the national-level severity and interaction features used as *inputs* to LGBM:
      for feat_suffix in [
          'current_vs_historical_admissions_diff_lag1',
          'current_vs_historical_cumulative_diff_lagged',
          'ewma_current_vs_historical_admissions_diff_lagged',
          'ewma_current_vs_historical_cumulative_diff_lagged',
          'ewma_admissions_diff_deviation_lagged',
          'rolling_std_current_vs_historical_admissions_diff_lagged',
          'rel_week_x_lag1_norm',
          'rel_week_x_current_vs_hist_admissions',
          'rel_week_x_current_vs_hist_cumulative',
          'lag1_x_current_vs_hist_admissions',
          'lag1_x_current_vs_hist_cumulative',
          'current_vs_historical_admissions_x_cumulative_diff',
          'rel_week_x_ewma_admissions_diff',
          'rel_week_x_ewma_cumulative_diff',
          'lag1_x_ewma_admissions_diff',
          'lag1_x_ewma_cumulative_diff',
          'ewma_admissions_x_ewma_cumulative_diff',
          'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
          'population_log_x_lag_1_norm',
          'population_log_x_current_vs_historical_admissions_diff_lag1',
          'population_log_x_relative_season_week',
          'population_log_x_sin_relative_season_week',
          'population_log_x_cos_relative_season_week',
          'horizon_x_lag_1_norm',
          'horizon_x_relative_season_week',
          'horizon_x_ewma_admissions_diff_lagged',
          'population_log_x_horizon',
      ]:
        national_col_name = f'national_{feat_suffix}'
        current_horizon_test_x_batch.loc[idx, national_col_name] = (
            _get_national_feature_for_date(
                forecast_date - pd.Timedelta(weeks=1), national_col_name
            )
        )

    # Predict median and scale for the current horizon
    median_predictions_batch = lgbm_median.predict(
        current_horizon_test_x_batch[features]
    )
    scale_predictions_batch = lgbm_scale.predict(
        current_horizon_test_x_batch[features]
    )

    # Ensure scale predictions are non-negative and have a minimum value
    scale_predictions_batch = np.maximum(scale_predictions_batch, 1e-6)

    # Apply scale inflation factor
    scale_predictions_batch *= SCALE_INFLATION_FACTOR

    # Generate quantiles based on predicted median and scale (using Student's t-distribution)
    quantile_preds_for_current_horizon = {}
    for q_level in QUANTILES:
      z_score_or_t_value = st.t.ppf(q_level, df=DF_T_DISTRIBUTION)
      predictions = (
          median_predictions_batch
          + z_score_or_t_value * scale_predictions_batch
      )
      test_y_hat_quantiles.loc[
          current_horizon_test_rows_idx, f'quantile_{q_level}'
      ] = predictions
      quantile_preds_for_current_horizon[q_level] = predictions

    # Calculate the median prediction for the current horizon and update the map for future lags
    median_predictions_series = pd.Series(
        median_predictions_batch, index=current_horizon_test_rows_idx
    )

    # Update location_latest_values with these new median forecasts (transformed values)
    for original_idx, median_val in median_predictions_series.items():
      loc_val = test_idx_to_loc_date.loc[original_idx, 'location']
      date_val = test_idx_to_loc_date.loc[original_idx, 'target_end_date']
      current_season_year_start_forecast = get_season_year(date_val)

      location_latest_values[(loc_val, date_val)] = median_val

      # Update location_season_cumulative_tracker with current week's prediction
      prev_cumulative_from_tracker = location_season_cumulative_tracker.get(
          (
              loc_val,
              current_season_year_start_forecast,
              date_val - pd.Timedelta(weeks=1),
          ),
          0.0,
      )
      current_cumulative_sum_for_week = (
          prev_cumulative_from_tracker + median_val
      )
      location_season_cumulative_tracker[
          (loc_val, current_season_year_start_forecast, date_val)
      ] = current_cumulative_sum_for_week

      # Update location_ewma_trackers and current_vs_historical_admissions_diff_tracker
      # Recalculate current_vs_historical_admissions_diff_lag1 for the *current week's median prediction*

      # Get local historical climatology and its observation count for *this week*
      loc_climat_row_curr = historical_climatology.loc[
          (historical_climatology['location'] == loc_val)
          & (
              historical_climatology['relative_season_week']
              == relative_season_week
          )
      ]
      local_historical_avg_target_curr = (
          loc_climat_row_curr['historical_avg_target_norm_by_week_loc'].values[
              0
          ]
          if not loc_climat_row_curr.empty
          else np.nan
      )
      local_obs_count_curr = (
          loc_climat_row_curr['count_observations_loc'].values[0]
          if not loc_climat_row_curr.empty
          else 0
      )
      nat_climat_row_curr = national_historical_climatology.loc[
          national_historical_climatology['relative_season_week']
          == relative_season_week
      ]
      national_historical_avg_target_curr = (
          nat_climat_row_curr[
              'national_historical_avg_target_norm_by_week'
          ].values[0]
          if not nat_climat_row_curr.empty
          else np.nan
      )

      weight_local_target_curr = min(
          1.0, local_obs_count_curr / CLIMATOLOGY_WEIGHT_K
      )
      national_target_avg_fallback_curr = (
          national_historical_avg_target_curr
          if not np.isnan(national_historical_avg_target_curr)
          else 0.0
      )
      local_target_for_blend_curr = (
          local_historical_avg_target_curr
          if not np.isnan(local_historical_avg_target_curr)
          else national_target_avg_fallback_curr
      )
      baseline_target_for_diff_curr = (
          weight_local_target_curr * local_target_for_blend_curr
      ) + ((1 - weight_local_target_curr) * national_target_avg_fallback_curr)
      baseline_target_for_diff_curr = (
          0.0
          if np.isnan(baseline_target_for_diff_curr)
          else baseline_target_for_diff_curr
      )

      current_admissions_diff_for_update = (
          median_val - baseline_target_for_diff_curr
      )

      # Use the EWMA value that was used as a *lagged feature* for the current forecast_date (which is for forecast_date - 1)
      prev_ewma_admissions_from_tracker = location_ewma_admissions_tracker.get(
          (loc_val, current_season_year_start_forecast), 0.0
      )
      prev_ewma_cumulative_from_tracker = location_ewma_cumulative_tracker.get(
          (loc_val, current_season_year_start_forecast), 0.0
      )

      # Calculate the new EWMA for the *current forecast_date*
      new_ewma_admissions = (
          EWMA_ALPHA * current_admissions_diff_for_update
          + (1 - EWMA_ALPHA) * prev_ewma_admissions_from_tracker
      )

      # Recalculate current_vs_historical_cumulative_diff for the *current week's median prediction*
      prev_week_cumulative_sum_for_curr_diff = (
          location_season_cumulative_tracker.get(
              (
                  loc_val,
                  current_season_year_start_forecast,
                  date_val - pd.Timedelta(weeks=1),
              ),
              0.0,
          )
      )
      current_cumulative_val_for_diff = (
          prev_week_cumulative_sum_for_curr_diff + median_val
      )

      loc_climat_row_curr = historical_climatology.loc[
          (historical_climatology['location'] == loc_val)
          & (
              historical_climatology['relative_season_week']
              == relative_season_week
          )
      ]
      local_historical_avg_cumulative_curr = (
          loc_climat_row_curr[
              'historical_avg_cumulative_norm_by_week_loc'
          ].values[0]
          if not loc_climat_row_curr.empty
          else np.nan
      )
      nat_climat_row_curr = national_historical_climatology.loc[
          national_historical_climatology['relative_season_week']
          == relative_season_week
      ]
      national_historical_avg_cumulative_curr = (
          nat_climat_row_curr[
              'national_historical_avg_cumulative_norm_by_week'
          ].values[0]
          if not nat_climat_row_curr.empty
          else np.nan
      )

      weight_local_cumulative_curr = min(
          1.0, local_obs_count_curr / CLIMATOLOGY_WEIGHT_K
      )
      national_cumulative_avg_fallback_curr = (
          national_historical_avg_cumulative_curr
          if not np.isnan(national_historical_avg_cumulative_curr)
          else 0.0
      )
      local_cumulative_for_blend_curr = (
          local_historical_avg_cumulative_curr
          if not np.isnan(local_historical_avg_cumulative_curr)
          else national_cumulative_avg_fallback_curr
      )
      baseline_cumulative_for_diff_curr = (
          weight_local_cumulative_curr * local_cumulative_for_blend_curr
      ) + (
          (1 - weight_local_cumulative_curr)
          * national_cumulative_avg_fallback_curr
      )
      baseline_cumulative_for_diff_curr = (
          0.0
          if np.isnan(baseline_cumulative_for_diff_curr)
          else baseline_cumulative_for_diff_curr
      )

      current_cumulative_diff_for_update = (
          current_cumulative_val_for_diff - baseline_cumulative_for_diff_curr
      )
      new_ewma_cumulative = (
          EWMA_ALPHA * current_cumulative_diff_for_update
          + (1 - EWMA_ALPHA) * prev_ewma_cumulative_from_tracker
      )

      # Store this new EWMA (for current forecast_date) into the tracker,
      # so it becomes the lagged EWMA for the *next* forecast week.
      location_ewma_admissions_tracker[
          (loc_val, current_season_year_start_forecast)
      ] = new_ewma_admissions
      location_ewma_cumulative_tracker[
          (loc_val, current_season_year_start_forecast)
      ] = new_ewma_cumulative

      # Store the current week's actual current_vs_historical_admissions_diff_lag1 for rolling_std calculation in future weeks
      location_current_vs_historical_admissions_diff_tracker[
          (loc_val, date_val)
      ] = current_admissions_diff_for_update

    # --- Update national_predicted_context for the current forecast_date ---
    current_nat_forecast_features_for_storage = {}

    # Population-weighted average of state-level median predictions for national_lag_1_norm
    current_nat_forecast_features_for_storage['national_lag_1_norm'] = (
        np.average(
            median_predictions_series,
            weights=current_horizon_test_x_batch['population'],
        )
        if not median_predictions_series.empty
        and current_horizon_test_x_batch['population'].sum() > 0
        else 0.0
    )

    # Calculate other national lagged features (e.g., national_lag_2_norm for forecast_date is national_lag_1_norm from forecast_date-1)
    for lag in range(2, MAX_LAG + 1):
      current_nat_forecast_features_for_storage[f'national_lag_{lag}_norm'] = (
          _get_national_feature_for_date(
              forecast_date - pd.Timedelta(weeks=lag - 1), 'national_lag_1_norm'
          )
      )

    # Calculate derived national features for the current forecast_date using the sequence of _get_national_feature_for_date
    national_rolling_vals_for_storage = [
        _get_national_feature_for_date(
            forecast_date - pd.Timedelta(weeks=i), 'national_lag_1_norm'
        )
        for i in range(1, ROLLING_MEAN_WINDOW + 1)
    ]
    current_nat_forecast_features_for_storage[
        f'national_rolling_mean_{ROLLING_MEAN_WINDOW}wk_lag1_norm'
    ] = (
        np.mean(national_rolling_vals_for_storage)
        if national_rolling_vals_for_storage
        else 0.0
    )

    national_lag1_target_for_storage = _get_national_feature_for_date(
        forecast_date - pd.Timedelta(weeks=1), 'national_lag_1_norm'
    )
    national_lag2_target_for_storage = _get_national_feature_for_date(
        forecast_date - pd.Timedelta(weeks=2), 'national_lag_1_norm'
    )
    current_nat_forecast_features_for_storage[
        'national_admissions_growth_rate_log_diff_lag1'
    ] = (national_lag1_target_for_storage - national_lag2_target_for_storage)

    national_lag3_target_for_storage = _get_national_feature_for_date(
        forecast_date - pd.Timedelta(weeks=3), 'national_lag_1_norm'
    )
    current_nat_forecast_features_for_storage[
        'national_admissions_acceleration_log_diff_lag1'
    ] = (
        national_lag1_target_for_storage - national_lag2_target_for_storage
    ) - (
        national_lag2_target_for_storage - national_lag3_target_for_storage
    )

    national_cumulative_sum_for_storage = 0.0
    current_nat_season_year_for_storage = get_season_year(forecast_date)
    temp_date_for_storage = forecast_date - pd.Timedelta(weeks=1)
    while (
        get_season_year(temp_date_for_storage)
        == current_nat_season_year_for_storage
        and temp_date_for_storage > train_end_date
    ):
      national_cumulative_sum_for_storage += _get_national_feature_for_date(
          temp_date_for_storage, 'national_lag_1_norm'
      )
      temp_date_for_storage -= pd.Timedelta(weeks=1)
    # Add historical national cumulative from the training data for the previous week if available
    if forecast_date - pd.Timedelta(weeks=1) <= train_end_date:
      national_cumulative_sum_for_storage += _get_national_feature_for_date(
          forecast_date - pd.Timedelta(weeks=1),
          'national_season_cumulative_norm_target_lagged',
      )

    current_nat_forecast_features_for_storage[
        'national_season_cumulative_norm_target_lagged'
    ] = national_cumulative_sum_for_storage

    # Populate national_previous_season_peak_norm_target for storage
    current_nat_forecast_features_for_storage[
        'national_previous_season_peak_norm_target'
    ] = national_season_actual_peak_lookup.get(
        current_nat_season_year_for_storage - 1, 0.0
    )

    # Recalculate national-level continuous severity and interaction features
    pop_weights = current_horizon_test_x_batch['population']
    pop_sum = pop_weights.sum()

    if pop_sum > 0:
      for feat_suffix in [
          'current_vs_historical_admissions_diff_lag1',
          'current_vs_historical_cumulative_diff_lagged',
          'ewma_current_vs_historical_admissions_diff_lagged',
          'ewma_current_vs_historical_cumulative_diff_lagged',
          'ewma_admissions_diff_deviation_lagged',
          'rolling_std_current_vs_historical_admissions_diff_lagged',
          'rel_week_x_lag1_norm',
          'rel_week_x_current_vs_hist_admissions',
          'rel_week_x_current_vs_hist_cumulative',
          'lag1_x_current_vs_hist_admissions',
          'lag1_x_current_vs_hist_cumulative',
          'current_vs_historical_admissions_x_cumulative_diff',
          'rel_week_x_ewma_admissions_diff',
          'rel_week_x_ewma_cumulative_diff',
          'lag1_x_ewma_admissions_diff',
          'lag1_x_ewma_cumulative_diff',
          'ewma_admissions_x_ewma_cumulative_diff',
          'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
          'population_log_x_lag_1_norm',
          'population_log_x_current_vs_historical_admissions_diff_lag1',
          'population_log_x_relative_season_week',
          'population_log_x_sin_relative_season_week',
          'population_log_x_cos_relative_season_week',
          'horizon_x_lag_1_norm',
          'horizon_x_relative_season_week',
          'horizon_x_ewma_admissions_diff_lagged',
          'population_log_x_horizon',
      ]:
        col_name = feat_suffix
        national_col_name = f'national_{feat_suffix}'

        values = current_horizon_test_x_batch[col_name].fillna(0.0)

        current_nat_forecast_features_for_storage[national_col_name] = (
            np.average(values, weights=pop_weights)
        )
    else:
      for feat_suffix in [
          'current_vs_historical_admissions_diff_lag1',
          'current_vs_historical_cumulative_diff_lagged',
          'ewma_current_vs_historical_admissions_diff_lagged',
          'ewma_current_vs_historical_cumulative_diff_lagged',
          'ewma_admissions_diff_deviation_lagged',
          'rolling_std_current_vs_historical_admissions_diff_lagged',
          'rel_week_x_lag1_norm',
          'rel_week_x_current_vs_hist_admissions',
          'rel_week_x_current_vs_hist_cumulative',
          'lag1_x_current_vs_hist_admissions',
          'lag1_x_current_vs_hist_cumulative',
          'current_vs_historical_admissions_x_cumulative_diff',
          'rel_week_x_ewma_admissions_diff',
          'rel_week_x_ewma_cumulative_diff',
          'lag1_x_ewma_admissions_diff',
          'lag1_x_ewma_cumulative_diff',
          'ewma_admissions_x_ewma_cumulative_diff',
          'ewma_admissions_diff_deviation_lagged_x_relative_season_week',  # ADDED THIS ONE
          'population_log_x_lag_1_norm',
          'population_log_x_current_vs_historical_admissions_diff_lag1',
          'population_log_x_relative_season_week',
          'population_log_x_sin_relative_season_week',
          'population_log_x_cos_relative_season_week',
          'horizon_x_lag_1_norm',
          'horizon_x_relative_season_week',
          'horizon_x_ewma_admissions_diff_lagged',
          'population_log_x_horizon',
      ]:
        national_col_name = f'national_{feat_suffix}'
        current_nat_forecast_features_for_storage[national_col_name] = 0.0

    national_predicted_context[forecast_date] = (
        current_nat_forecast_features_for_storage
    )

  # --- 5. Post-processing ---
  # Inverse transform predictions from INTERNAL_TARGET_COL to admissions_per_100k
  # INVERSE TRANSFORMATION: (y+1)^2 - 1
  test_y_hat_quantiles = (test_y_hat_quantiles + 1) ** 2 - 1

  # Convert admissions_per_100k back to Total Influenza Admissions
  test_x_populations = test_x_scaffold.set_index(test_x_scaffold.index)[
      'population'
  ]
  test_y_hat_quantiles = test_y_hat_quantiles.apply(
      lambda col: (col * test_x_populations / POP_NORM_FACTOR)
  )

  # Ensure non-negative integers
  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(int)
  test_y_hat_quantiles[test_y_hat_quantiles < 0] = 0

  # Crucial constraint: predicted quantiles must be monotonically increasing for each row.
  test_y_hat_quantiles = test_y_hat_quantiles.apply(
      lambda row: np.maximum.accumulate(row.values),
      axis=1,
      result_type='broadcast',
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
