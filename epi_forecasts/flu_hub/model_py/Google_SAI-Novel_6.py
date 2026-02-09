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
MODEL_NAME = 'Google_SAI-Novel_6'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from collections import deque
from scipy.stats import norm, median_abs_deviation


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # Suppress specific warnings that might arise from numerical operations or sklearn
  warnings.filterwarnings(
      'ignore', message='The behavior of Series.append', category=FutureWarning
  )
  warnings.filterwarnings(
      'ignore',
      message='The behavior of DataFrame.product and Series.product',
      category=FutureWarning,
  )
  warnings.filterwarnings(
      'ignore',
      message='The `needs_categories` parameter',
      category=FutureWarning,
  )
  warnings.filterwarnings(
      'ignore',
      message='Initializing distribution with no arguments',
      category=UserWarning,
  )

  # --- 0. Initial Data Preparation ---
  # Combine train_x and train_y
  train_data = train_x.copy()
  train_data[TARGET_STR] = train_y.copy()

  # Convert date columns to datetime
  train_data['target_end_date'] = pd.to_datetime(train_data['target_end_date'])
  test_x_processed = (
      test_x.copy()
  )  # Use a copy to avoid modifying original test_x
  test_x_processed['target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  )
  test_x_processed['reference_date'] = pd.to_datetime(
      test_x_processed['reference_date']
  )

  # Merge population information into train_data and test_x_processed if not present
  if 'population' not in train_data.columns:
    train_data = train_data.merge(
        locations[['location', 'population']], on='location', how='left'
    )
  if 'population' not in test_x_processed.columns:
    test_x_processed = test_x_processed.merge(
        locations[['location', 'population']], on='location', how='left'
    )

  # Ensure populations are numeric and not NaN and apply MIN_POPULATION
  global_mean_population = (
      locations['population'].mean() if not locations.empty else 1_000_000
  )
  MIN_POPULATION = 1.0  # Use 1.0 instead of epsilon for more robust floor
  train_data['population'] = (
      train_data['population']
      .fillna(global_mean_population)
      .apply(lambda x: max(x, MIN_POPULATION))
  )
  test_x_processed['population'] = (
      test_x_processed['population']
      .fillna(global_mean_population)
      .apply(lambda x: max(x, MIN_POPULATION))
  )

  # --- 1. Feature Engineering (Common for both ILINet and NHSN data) ---
  def create_time_features(
      df, min_date
  ):
    df['year'] = df['target_end_date'].dt.isocalendar().year.astype(int)
    df['week'] = df['target_end_date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['target_end_date'].dt.dayofweek  # Saturday is 5

    # Define season_year and season_week based on CDC epi-week definitions
    # Season typically starts week 40 of year N and ends week 39 of year N+1
    df['season_year'] = df['year'].copy()
    df.loc[df['week'] >= 40, 'season_year'] = df.loc[df['week'] >= 40, 'year']
    df.loc[df['week'] < 40, 'season_year'] = df.loc[df['week'] < 40, 'year'] - 1

    df['season_week'] = df['week'] - 39
    df.loc[df['season_week'] <= 0, 'season_week'] = (
        df.loc[df['season_week'] <= 0, 'season_week'] + 52
    )

    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)

    df['time_idx'] = (
        df['target_end_date'] - min_date
    ).dt.days / 7  # New linear time trend feature
    return df

  min_global_date = min(
      train_data['target_end_date'].min(),
      test_x_processed['target_end_date'].min(),
  )
  train_data = create_time_features(train_data, min_global_date)
  test_x_features = create_time_features(
      test_x_processed.copy(), min_global_date
  )  # create features for test_x scaffold

  # --- 2. ILINet Data Augmentation & Feature Creation (Strategy 2: Learn a Transformation + Feature) ---
  ilinet_state_clean = ilinet_state.copy()
  ilinet_state_clean = ilinet_state_clean.rename(
      columns={'region': 'location_name', 'week_start': 'target_end_date'}
  )
  ilinet_state_clean['target_end_date'] = pd.to_datetime(
      ilinet_state_clean['target_end_date']
  )

  # Filter to dates before the cutoff (as ILINet is only available historically)
  ILINET_CUTOFF_DATE = pd.to_datetime('2022-10-15')
  ilinet_state_clean = ilinet_state_clean[
      ilinet_state_clean['target_end_date'] < ILINET_CUTOFF_DATE
  ]

  # Merge with locations to get FIPS and population
  ilinet_state_clean = ilinet_state_clean.merge(
      locations[['location', 'location_name', 'population']],
      on='location_name',
      how='left',
  )
  ilinet_state_clean['population'] = (
      ilinet_state_clean['population']
      .fillna(global_mean_population)
      .apply(lambda x: max(x, MIN_POPULATION))
  )
  ilinet_state_clean = ilinet_state_clean.dropna(
      subset=['location', 'population', 'unweighted_ili']
  )

  # Convert unweighted_ili (percentage) to ili_per_100k
  ilinet_state_clean['ili_per_100k'] = (
      ilinet_state_clean['unweighted_ili'] * 1000
  ).astype(float)
  # Clip ili_per_100k to prevent negative values from impacting Huber Regressor
  ilinet_state_clean['ili_per_100k'] = np.maximum(
      0, ilinet_state_clean['ili_per_100k']
  )

  # Apply 4th root transformation to ILINet data for consistency with NHSN target
  ilinet_state_clean['ili_per_100k_4rt'] = np.power(
      ilinet_state_clean['ili_per_100k'], 1 / 4
  )
  ilinet_state_clean['ili_per_100k_4rt'] = np.maximum(
      0, ilinet_state_clean['ili_per_100k_4rt']
  )  # Ensure non-negative after transform

  ilinet_state_clean = create_time_features(
      ilinet_state_clean, min_global_date
  )  # Add time features to ILINet data

  # Identify overlap period for learning the transformation
  min_nhs_date = train_data['target_end_date'].min()
  overlap_data = train_data[
      train_data['target_end_date'] < ILINET_CUTOFF_DATE
  ].copy()
  overlap_data = overlap_data.dropna(subset=[TARGET_STR, 'population'])
  overlap_data['admissions_per_100k'] = (
      overlap_data[TARGET_STR] / overlap_data['population']
  ) * 100_000

  # Target for Huber is 4th-rooted NHSN admissions for transformation learning
  overlap_data['admissions_per_100k_4rt'] = np.power(
      np.maximum(0, overlap_data['admissions_per_100k']), 1 / 4
  )

  # Merge ILINet with overlap data for transformation learning
  overlap_with_ili = pd.merge(
      overlap_data,
      ilinet_state_clean[
          ['target_end_date', 'location', 'ili_per_100k_4rt']
      ],  # Use 4th-rooted ILI
      on=['target_end_date', 'location'],
      how='inner',
  )
  overlap_with_ili = overlap_with_ili.dropna(
      subset=['admissions_per_100k_4rt', 'ili_per_100k_4rt']
  )

  # Learn Huber Regressor transformation for each location
  location_huber_models = {}

  # Fallback global Huber model - trained on 4th-rooted ili_per_100k
  global_huber_fallback_ili_transform = None
  min_samples_for_huber = 5  # Minimum samples to train HuberRegressor

  if (
      overlap_with_ili.shape[0] >= min_samples_for_huber
      and overlap_with_ili['ili_per_100k_4rt'].std() > 0
      and overlap_with_ili['admissions_per_100k_4rt'].std() > 0
  ):
    X_ili_global = overlap_with_ili['ili_per_100k_4rt'].values.reshape(-1, 1)
    y_admissions_global = overlap_with_ili['admissions_per_100k_4rt'].values
    try:
      global_huber_fallback_ili_transform = HuberRegressor(
          epsilon=1.35, max_iter=1000, tol=1e-3
      )
      global_huber_fallback_ili_transform.fit(X_ili_global, y_admissions_global)
    except Exception:
      global_huber_fallback_ili_transform = None

  for loc_fips in overlap_with_ili['location'].unique():
    loc_data = overlap_with_ili[overlap_with_ili['location'] == loc_fips]

    # Only try to fit Huber if enough data points and sufficient variation in target and feature
    if (
        loc_data.shape[0] >= min_samples_for_huber
        and loc_data['ili_per_100k_4rt'].std() > 0
        and loc_data['admissions_per_100k_4rt'].std() > 0
    ):
      X_ili = loc_data['ili_per_100k_4rt'].values.reshape(-1, 1)
      y_admissions = loc_data[
          'admissions_per_100k_4rt'
      ].values  # Use 4th-rooted admissions

      try:
        huber = HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-3)
        huber.fit(X_ili, y_admissions)
        location_huber_models[loc_fips] = huber
      except Exception:
        pass  # Silently ignore locations where Huber fails to fit robustly

  # Generate synthetic admissions for historical ILINet data (before min_nhs_date)
  synthetic_admissions_list = []
  historical_ilinet_for_synth = ilinet_state_clean[
      ilinet_state_clean['target_end_date'] < min_nhs_date
  ].copy()

  # Use 4th-rooted ili_per_100k for prediction
  historical_ilinet_for_synth['ili_per_100k_to_use'] = (
      historical_ilinet_for_synth['ili_per_100k_4rt']
  )

  for loc_fips in historical_ilinet_for_synth['location'].unique():
    loc_ili_data = historical_ilinet_for_synth[
        historical_ilinet_for_synth['location'] == loc_fips
    ].copy()

    huber_model_to_use = location_huber_models.get(
        loc_fips, global_huber_fallback_ili_transform
    )

    if huber_model_to_use is not None:
      X_ili_to_predict = loc_ili_data['ili_per_100k_to_use'].values.reshape(
          -1, 1
      )

      with warnings.catch_warnings():
        warnings.simplefilter(
            'ignore', RuntimeWarning
        )  # Ignore warnings from np.power on edge cases
        synthetic_admissions_per_100k_4rt = huber_model_to_use.predict(
            X_ili_to_predict
        )

      synthetic_admissions_per_100k_4rt = np.maximum(
          0, synthetic_admissions_per_100k_4rt
      )  # Ensure non-negative in 4rt space

      # Inverse transform from 4th root space
      synthetic_admissions_per_100k = np.power(
          synthetic_admissions_per_100k_4rt, 4
      )

      loc_ili_data['admissions_per_100k'] = np.maximum(
          0, synthetic_admissions_per_100k
      )  # Ensure non-negative
      loc_ili_data[TARGET_STR] = (
          loc_ili_data['admissions_per_100k']
          * loc_ili_data['population']
          / 100_000
      )
      loc_ili_data[TARGET_STR] = np.round(loc_ili_data[TARGET_STR]).astype(int)

      synthetic_admissions_list.append(
          loc_ili_data[[
              'target_end_date',
              'location',
              'location_name',
              'population',
              TARGET_STR,
              'admissions_per_100k',
              'year',
              'week',
              'season_year',
              'season_week',
              'sin_week',
              'cos_week',
              'time_idx',  # Include time_idx
          ]]
      )

  if synthetic_admissions_list:
    synthetic_admissions_df = pd.concat(
        synthetic_admissions_list, ignore_index=True
    )
    full_train_data = pd.concat(
        [synthetic_admissions_df, train_data], ignore_index=True
    )
    full_train_data = full_train_data.drop_duplicates(
        subset=['target_end_date', 'location'], keep='last'
    )
  else:
    full_train_data = train_data.copy()

  full_train_data = full_train_data.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  # Integrate ili_per_100k_4rt as a direct feature in full_train_data
  ilinet_feature_data = ilinet_state_clean[
      ['target_end_date', 'location', 'ili_per_100k_4rt']
  ].copy()
  full_train_data = pd.merge(
      full_train_data,
      ilinet_feature_data,
      on=['target_end_date', 'location'],
      how='left',
  )
  full_train_data = full_train_data.rename(
      columns={'ili_per_100k_4rt': 'ili_4rt_feature'}
  )
  full_train_data['ili_4rt_feature'] = full_train_data[
      'ili_4rt_feature'
  ].fillna(0.0)

  # Prepare test_x_features for ili_4rt_feature (will be 0 as ILINet data is historical only)
  test_x_features['ili_4rt_feature'] = (
      0.0  # ILINet is not available for future dates in test_x
  )

  # --- 3. Target Transformation: 4th Root & Logit ---

  # 4th Root Transformation (from Code 2) - This will be the PRIMARY target for LGBM and AR models
  full_train_data['admissions_per_100k'] = (
      full_train_data[TARGET_STR] / full_train_data['population']
  ) * 100_000
  full_train_data['admissions_per_100k'] = np.maximum(
      full_train_data['admissions_per_100k'], 1e-6
  )  # Floor to avoid issues
  full_train_data['value_rate_4rt'] = np.power(
      full_train_data['admissions_per_100k'], 1 / 4
  )

  # Calculate p95 and mean per location for standardization
  loc_stats = (
      full_train_data.groupby('location')['value_rate_4rt']
      .agg(p95_loc=lambda x: x.quantile(0.95), mean_loc='mean')
      .reset_index()
  )

  global_fallback_p95_loc_value = (
      full_train_data['value_rate_4rt'].quantile(0.95)
      if not full_train_data['value_rate_4rt'].empty
      else 1.0
  )
  if (
      pd.isna(global_fallback_p95_loc_value)
      or global_fallback_p95_loc_value <= 1e-6
  ):
    global_fallback_p95_loc_value = 1.0

  loc_stats['p95_loc'] = np.where(
      loc_stats['p95_loc'] <= 0.1,
      np.maximum(global_fallback_p95_loc_value, 0.1),
      loc_stats['p95_loc'],
  )
  loc_stats['p95_loc'] = np.maximum(
      loc_stats['p95_loc'], 0.1
  )  # Ensure p95 is at least 0.1
  loc_stats['mean_loc'] = loc_stats['mean_loc'].fillna(0.0)

  full_train_data = full_train_data.merge(loc_stats, on='location', how='left')
  full_train_data['value_rate_4rt_std'] = (
      full_train_data['value_rate_4rt'] / full_train_data['p95_loc']
  ) - full_train_data['mean_loc']

  # Logit Transformation (from Code 1) - This will be used as a FEATURE
  # Define a robust maximum admission rate per 100k to bound the proportion
  # Use a high quantile to make it robust to outliers, ensure a sensible minimum
  max_admissions_per_100k_q999 = (
      full_train_data['admissions_per_100k'].quantile(0.999)
      if not full_train_data['admissions_per_100k'].empty
      else 0
  )
  MAX_RATE_FOR_PROPORTION = np.maximum(
      max_admissions_per_100k_q999 * 1.05, 1000.0
  )  # Multiply by 1.05 and ensure a floor of 1000 for robustness
  EPSILON = 1e-6

  full_train_data['admissions_proportion'] = (
      full_train_data['admissions_per_100k'] / MAX_RATE_FOR_PROPORTION
  )
  full_train_data['admissions_proportion'] = np.clip(
      full_train_data['admissions_proportion'], EPSILON, 1 - EPSILON
  )
  full_train_data['logit_admissions_proportion'] = np.log(
      full_train_data['admissions_proportion']
      / (1 - full_train_data['admissions_proportion'])
  )

  # --- 4. Advanced Feature Engineering (Lags, Rolling Stats, Categorical Encoding) ---
  LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  ROLLING_WINDOWS = [4, 8, 12]
  AR_LAGS = [1, 2, 3, 4, 5, 6, 7, 8]  # For Huber AR model
  DIFF_LAGS = [1, 2, 3]  # Lags for diff prediction model

  def add_lag_features(
      df, target_col, lags, fill_value=0.0
  ):
    df = df.sort_values(by=['location', 'target_end_date'])
    for lag in lags:
      df[f'{target_col}_lag_{lag}'] = (
          df.groupby('location')[target_col].shift(lag).fillna(fill_value)
      )
    return df

  def add_rolling_features(
      df,
      target_col,
      windows,
      min_periods_std = 3,
      fill_mean=0.0,
      fill_std=0.01,
  ):
    df = df.sort_values(by=['location', 'target_end_date'])
    for window in windows:
      df[f'{target_col}_rolling_mean_{window}'] = (
          df.groupby('location')[target_col]
          .transform(
              lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
          )
          .fillna(fill_mean)
      )
      df[f'{target_col}_rolling_std_{window}'] = (
          df.groupby('location')[target_col]
          .transform(
              lambda x: x.shift(1)
              .rolling(window=window, min_periods=min_periods_std)
              .std()
          )
          .fillna(fill_std)
      )
      # Ensure std is always positive
      df[f'{target_col}_rolling_std_{window}'] = np.where(
          df[f'{target_col}_rolling_std_{window}'] <= 0,
          fill_std,
          df[f'{target_col}_rolling_std_{window}'],
      )
    return df

  # Global MAD for fallbacks, ensure strictly positive
  global_mad_value_rate_4rt_std = median_abs_deviation(
      full_train_data['value_rate_4rt_std'].dropna()
  )
  if (
      pd.isna(global_mad_value_rate_4rt_std)
      or global_mad_value_rate_4rt_std <= 0
  ):
    global_mad_value_rate_4rt_std = 0.1

  full_train_data = add_lag_features(
      full_train_data, 'value_rate_4rt_std', LAGS, fill_value=0.0
  )
  full_train_data = add_rolling_features(
      full_train_data,
      'value_rate_4rt_std',
      ROLLING_WINDOWS,
      min_periods_std=3,
      fill_mean=0.0,
      fill_std=global_mad_value_rate_4rt_std,
  )

  full_train_data['diff_value_rate_4rt_std'] = (
      full_train_data.groupby('location')['value_rate_4rt_std']
      .diff(1)
      .fillna(0.0)
  )
  full_train_data = add_lag_features(
      full_train_data, 'diff_value_rate_4rt_std', DIFF_LAGS, fill_value=0.0
  )

  # Lag features for logit_admissions_proportion (used as features for primary models)
  global_mad_logit_admissions_proportion = median_abs_deviation(
      full_train_data['logit_admissions_proportion'].dropna()
  )
  if (
      pd.isna(global_mad_logit_admissions_proportion)
      or global_mad_logit_admissions_proportion <= 0
  ):
    global_mad_logit_admissions_proportion = 0.1

  full_train_data = add_lag_features(
      full_train_data, 'logit_admissions_proportion', LAGS, fill_value=0.0
  )
  full_train_data = add_rolling_features(
      full_train_data,
      'logit_admissions_proportion',
      ROLLING_WINDOWS,
      min_periods_std=3,
      fill_mean=0.0,
      fill_std=global_mad_logit_admissions_proportion,
  )

  full_train_data['horizon'] = 0

  robust_max_season_week = max(
      52,
      full_train_data['season_week'].max()
      if not full_train_data['season_week'].empty
      else 52,
  )

  full_train_data['season_progress'] = (
      full_train_data['season_week'] / robust_max_season_week
  )
  full_train_data['season_end_proximity'] = (
      robust_max_season_week - full_train_data['season_week']
  ) / robust_max_season_week

  test_x_features['season_progress'] = (
      test_x_features['season_week'] / robust_max_season_week
  )
  test_x_features['season_end_proximity'] = (
      robust_max_season_week - test_x_features['season_week']
  ) / robust_max_season_week

  # --- 4.1. Hybrid Element: Robust Seasonal Baseline Model for value_rate_4rt_std (Primary Target's Baseline) ---
  location_baseline_seasonal_models = {}

  baseline_features_4rt = [
      'sin_week',
      'cos_week',
      'season_week',
      'season_progress',
      'season_end_proximity',
      'time_idx',
      'ili_4rt_feature',
  ]
  min_samples_for_baseline_huber_4rt = len(baseline_features_4rt) + 2

  global_huber_seasonal_fallback_4rt = None
  loc_data_clean_baseline_global_4rt = full_train_data.dropna(
      subset=['value_rate_4rt_std']
  )
  for feature in baseline_features_4rt:
    if feature in loc_data_clean_baseline_global_4rt.columns:
      loc_data_clean_baseline_global_4rt[
          feature
      ] = loc_data_clean_baseline_global_4rt[feature].fillna(
          loc_data_clean_baseline_global_4rt[feature].mean()
          if not loc_data_clean_baseline_global_4rt[feature].empty
          else 0.0
      )

  if (
      loc_data_clean_baseline_global_4rt.shape[0]
      >= min_samples_for_baseline_huber_4rt
      and loc_data_clean_baseline_global_4rt['value_rate_4rt_std'].nunique() > 1
      and loc_data_clean_baseline_global_4rt['value_rate_4rt_std'].std() > 0
  ):
    try:
      global_huber_seasonal_fallback_4rt = HuberRegressor(
          epsilon=1.35, max_iter=1000, tol=1e-3
      )
      global_huber_seasonal_fallback_4rt.fit(
          loc_data_clean_baseline_global_4rt[baseline_features_4rt],
          loc_data_clean_baseline_global_4rt['value_rate_4rt_std'],
      )
    except Exception:
      global_huber_seasonal_fallback_4rt = None

  for loc_fips in full_train_data['location'].unique():
    loc_data = full_train_data[full_train_data['location'] == loc_fips].copy()

    loc_data_clean_baseline_4rt = loc_data.dropna(subset=['value_rate_4rt_std'])

    for feature in baseline_features_4rt:
      if feature in loc_data_clean_baseline_4rt.columns:
        loc_data_clean_baseline_4rt[feature] = loc_data_clean_baseline_4rt[
            feature
        ].fillna(
            loc_data_clean_baseline_4rt[feature].mean()
            if not loc_data_clean_baseline_4rt[feature].empty
            else 0.0
        )

    huber_seasonal_to_use_4rt = None
    if (
        loc_data_clean_baseline_4rt.shape[0]
        >= min_samples_for_baseline_huber_4rt
        and loc_data_clean_baseline_4rt['value_rate_4rt_std'].nunique() > 1
        and loc_data_clean_baseline_4rt['value_rate_4rt_std'].std() > 0
    ):
      try:
        huber_seasonal = HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-3)
        huber_seasonal.fit(
            loc_data_clean_baseline_4rt[baseline_features_4rt],
            loc_data_clean_baseline_4rt['value_rate_4rt_std'],
        )
        huber_seasonal_to_use_4rt = huber_seasonal
      except Exception:
        pass

    if huber_seasonal_to_use_4rt is not None:
      location_baseline_seasonal_models[loc_fips] = huber_seasonal_to_use_4rt
    elif global_huber_seasonal_fallback_4rt is not None:
      location_baseline_seasonal_models[loc_fips] = (
          global_huber_seasonal_fallback_4rt
      )

  full_train_data['baseline_seasonal_pred'] = np.nan
  for loc_fips, model in location_baseline_seasonal_models.items():
    loc_idx = full_train_data['location'] == loc_fips
    train_baseline_features_loc = full_train_data.loc[
        loc_idx, baseline_features_4rt
    ].copy()
    for feature in baseline_features_4rt:
      train_baseline_features_loc[feature] = train_baseline_features_loc[
          feature
      ].fillna(
          train_baseline_features_loc[feature].mean()
          if not train_baseline_features_loc[feature].empty
          else 0.0
      )
    full_train_data.loc[loc_idx, 'baseline_seasonal_pred'] = model.predict(
        train_baseline_features_loc
    )

  global_baseline_mean_fallback_4rt = full_train_data[
      'baseline_seasonal_pred'
  ].mean()
  full_train_data['baseline_seasonal_pred'] = full_train_data[
      'baseline_seasonal_pred'
  ].fillna(
      global_baseline_mean_fallback_4rt
      if not pd.isna(global_baseline_mean_fallback_4rt)
      else 0.0
  )

  test_x_features['baseline_seasonal_pred'] = np.nan
  for loc_fips in test_x_features['location'].unique():
    loc_idx = test_x_features['location'] == loc_fips
    model_to_use = location_baseline_seasonal_models.get(
        loc_fips, global_huber_seasonal_fallback_4rt
    )

    if model_to_use is not None:
      test_x_loc_data = test_x_features.loc[
          loc_idx, baseline_features_4rt
      ].copy()
      for feature in baseline_features_4rt:
        test_x_loc_data[feature] = test_x_loc_data[feature].fillna(
            test_x_loc_data[feature].mean()
            if not test_x_loc_data[feature].empty
            else 0.0
        )
      test_x_features.loc[loc_idx, 'baseline_seasonal_pred'] = (
          model_to_use.predict(test_x_loc_data)
      )
    else:
      test_x_features.loc[loc_idx, 'baseline_seasonal_pred'] = (
          global_baseline_mean_fallback_4rt
      )

  test_x_features['baseline_seasonal_pred'] = test_x_features[
      'baseline_seasonal_pred'
  ].fillna(
      global_baseline_mean_fallback_4rt
      if not pd.isna(global_baseline_mean_fallback_4rt)
      else 0.0
  )

  # --- 4.1.1. Hybrid Element: Robust Seasonal Baseline Model for logit_admissions_proportion (Used as a FEATURE's Baseline) ---
  location_baseline_seasonal_models_logit = {}
  baseline_features_logit = [
      'sin_week',
      'cos_week',
      'season_week',
      'season_progress',
      'season_end_proximity',
      'time_idx',
      'ili_4rt_feature',
  ]
  min_samples_for_baseline_huber_logit = len(baseline_features_logit) + 2

  global_huber_seasonal_fallback_logit = None
  loc_data_clean_baseline_global_logit = full_train_data.dropna(
      subset=['logit_admissions_proportion']
  )
  for feature in baseline_features_logit:
    if feature in loc_data_clean_baseline_global_logit.columns:
      loc_data_clean_baseline_global_logit[
          feature
      ] = loc_data_clean_baseline_global_logit[feature].fillna(
          loc_data_clean_baseline_global_logit[feature].mean()
          if not loc_data_clean_baseline_global_logit[feature].empty
          else 0.0
      )

  if (
      loc_data_clean_baseline_global_logit.shape[0]
      >= min_samples_for_baseline_huber_logit
      and loc_data_clean_baseline_global_logit[
          'logit_admissions_proportion'
      ].nunique()
      > 1
      and loc_data_clean_baseline_global_logit[
          'logit_admissions_proportion'
      ].std()
      > 0
  ):
    try:
      global_huber_seasonal_fallback_logit = HuberRegressor(
          epsilon=1.35, max_iter=1000, tol=1e-3
      )
      global_huber_seasonal_fallback_logit.fit(
          loc_data_clean_baseline_global_logit[baseline_features_logit],
          loc_data_clean_baseline_global_logit['logit_admissions_proportion'],
      )
    except Exception:
      global_huber_seasonal_fallback_logit = None

  for loc_fips in full_train_data['location'].unique():
    loc_data = full_train_data[full_train_data['location'] == loc_fips].copy()
    loc_data_clean_baseline_logit = loc_data.dropna(
        subset=['logit_admissions_proportion']
    )

    for feature in baseline_features_logit:
      if feature in loc_data_clean_baseline_logit.columns:
        loc_data_clean_baseline_logit[feature] = loc_data_clean_baseline_logit[
            feature
        ].fillna(
            loc_data_clean_baseline_logit[feature].mean()
            if not loc_data_clean_baseline_logit[feature].empty
            else 0.0
        )

    huber_seasonal_to_use_logit = None
    if (
        loc_data_clean_baseline_logit.shape[0]
        >= min_samples_for_baseline_huber_logit
        and loc_data_clean_baseline_logit[
            'logit_admissions_proportion'
        ].nunique()
        > 1
        and loc_data_clean_baseline_logit['logit_admissions_proportion'].std()
        > 0
    ):
      try:
        huber_seasonal = HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-3)
        huber_seasonal.fit(
            loc_data_clean_baseline_logit[baseline_features_logit],
            loc_data_clean_baseline_logit['logit_admissions_proportion'],
        )
        huber_seasonal_to_use_logit = huber_seasonal
      except Exception:
        pass

    if huber_seasonal_to_use_logit is not None:
      location_baseline_seasonal_models_logit[loc_fips] = (
          huber_seasonal_to_use_logit
      )
    elif global_huber_seasonal_fallback_logit is not None:
      location_baseline_seasonal_models_logit[loc_fips] = (
          global_huber_seasonal_fallback_logit
      )

  full_train_data['baseline_seasonal_pred_logit'] = np.nan
  for loc_fips, model in location_baseline_seasonal_models_logit.items():
    loc_idx = full_train_data['location'] == loc_fips
    train_baseline_features_loc = full_train_data.loc[
        loc_idx, baseline_features_logit
    ].copy()
    for feature in baseline_features_logit:
      train_baseline_features_loc[feature] = train_baseline_features_loc[
          feature
      ].fillna(
          train_baseline_features_loc[feature].mean()
          if not train_baseline_features_loc[feature].empty
          else 0.0
      )
    full_train_data.loc[loc_idx, 'baseline_seasonal_pred_logit'] = (
        model.predict(train_baseline_features_loc)
    )

  global_baseline_mean_fallback_logit = full_train_data[
      'baseline_seasonal_pred_logit'
  ].mean()
  full_train_data['baseline_seasonal_pred_logit'] = full_train_data[
      'baseline_seasonal_pred_logit'
  ].fillna(
      global_baseline_mean_fallback_logit
      if not pd.isna(global_baseline_mean_fallback_logit)
      else 0.0
  )

  test_x_features['baseline_seasonal_pred_logit'] = np.nan
  for loc_fips in test_x_features['location'].unique():
    loc_idx = test_x_features['location'] == loc_fips
    model_to_use = location_baseline_seasonal_models_logit.get(
        loc_fips, global_huber_seasonal_fallback_logit
    )

    if model_to_use is not None:
      test_x_loc_data = test_x_features.loc[
          loc_idx, baseline_features_logit
      ].copy()
      for feature in baseline_features_logit:
        test_x_loc_data[feature] = test_x_loc_data[feature].fillna(
            test_x_loc_data[feature].mean()
            if not test_x_loc_data[feature].empty
            else 0.0
        )
      test_x_features.loc[loc_idx, 'baseline_seasonal_pred_logit'] = (
          model_to_use.predict(test_x_loc_data)
      )
    else:
      test_x_features.loc[loc_idx, 'baseline_seasonal_pred_logit'] = (
          global_baseline_mean_fallback_logit
      )
  test_x_features['baseline_seasonal_pred_logit'] = test_x_features[
      'baseline_seasonal_pred_logit'
  ].fillna(
      global_baseline_mean_fallback_logit
      if not pd.isna(global_baseline_mean_fallback_logit)
      else 0.0
  )

  # --- 4.2. Hybrid Element: Dynamic Global Discrepancy (EWMA of Residuals) for value_rate_4rt_std (Primary Target's Discrepancy) ---
  EWMA_SPAN = 8
  ewma_alpha = 2 / (EWMA_SPAN + 1)

  full_train_data['seasonal_residual'] = (
      full_train_data['value_rate_4rt_std']
      - full_train_data['baseline_seasonal_pred']
  )

  global_median_seasonal_residuals_series = (
      full_train_data.groupby('target_end_date')['seasonal_residual']
      .median()
      .sort_index()
  )

  global_median_seasonal_residual_fallback = (
      global_median_seasonal_residuals_series.median()
      if not global_median_seasonal_residuals_series.empty
      else 0.0
  )
  causal_common_discrepancy_ewma_series = (
      global_median_seasonal_residuals_series.ewm(
          span=EWMA_SPAN, min_periods=1, adjust=False
      )
      .mean()
      .shift(1)
  )  # Added adjust=False for consistent alpha behavior

  full_train_data['common_discrepancy_ewma'] = full_train_data[
      'target_end_date'
  ].map(causal_common_discrepancy_ewma_series)
  full_train_data['common_discrepancy_ewma'] = full_train_data[
      'common_discrepancy_ewma'
  ].fillna(global_median_seasonal_residual_fallback)

  full_train_data = add_lag_features(
      full_train_data,
      'common_discrepancy_ewma',
      [1],
      fill_value=global_median_seasonal_residual_fallback,
  )

  # Initialize dynamic EWMA state for recursive forecasting (main improvement)
  current_dynamic_discrepancy_ewma = global_median_seasonal_residual_fallback
  current_dynamic_discrepancy_ewma_lag1 = (
      global_median_seasonal_residual_fallback
  )

  causal_common_discrepancy_ewma_series_non_nan = (
      causal_common_discrepancy_ewma_series.dropna()
  )

  if not causal_common_discrepancy_ewma_series_non_nan.empty:
    current_dynamic_discrepancy_ewma = (
        causal_common_discrepancy_ewma_series_non_nan.iloc[-1]
    )
    if len(causal_common_discrepancy_ewma_series_non_nan) >= 2:
      current_dynamic_discrepancy_ewma_lag1 = (
          causal_common_discrepancy_ewma_series_non_nan.iloc[-2]
      )
    else:  # If only one historical EWMA value, lag1 is the same as current
      current_dynamic_discrepancy_ewma_lag1 = current_dynamic_discrepancy_ewma

  # --- 4.2.1. Hybrid Element: Dynamic Global Discrepancy (EWMA of Residuals) for logit_admissions_proportion (Used as a FEATURE's Discrepancy) ---
  full_train_data['seasonal_residual_logit'] = (
      full_train_data['logit_admissions_proportion']
      - full_train_data['baseline_seasonal_pred_logit']
  )
  global_median_seasonal_residuals_series_logit = (
      full_train_data.groupby('target_end_date')['seasonal_residual_logit']
      .median()
      .sort_index()
  )
  global_median_seasonal_residual_fallback_logit = (
      global_median_seasonal_residuals_series_logit.median()
      if not global_median_seasonal_residuals_series_logit.empty
      else 0.0
  )
  causal_common_discrepancy_ewma_series_logit = (
      global_median_seasonal_residuals_series_logit.ewm(
          span=EWMA_SPAN, min_periods=1, adjust=False
      )
      .mean()
      .shift(1)
  )

  full_train_data['common_discrepancy_ewma_logit'] = full_train_data[
      'target_end_date'
  ].map(causal_common_discrepancy_ewma_series_logit)
  full_train_data['common_discrepancy_ewma_logit'] = full_train_data[
      'common_discrepancy_ewma_logit'
  ].fillna(global_median_seasonal_residual_fallback_logit)
  full_train_data = add_lag_features(
      full_train_data,
      'common_discrepancy_ewma_logit',
      [1],
      fill_value=global_median_seasonal_residual_fallback_logit,
  )

  # Initialize dynamic EWMA state for logit discrepancy for recursive forecasting (NEW IMPROVEMENT)
  current_dynamic_discrepancy_ewma_logit = (
      global_median_seasonal_residual_fallback_logit
  )
  current_dynamic_discrepancy_ewma_lag1_logit = (
      global_median_seasonal_residual_fallback_logit
  )

  causal_common_discrepancy_ewma_series_non_nan_logit = (
      causal_common_discrepancy_ewma_series_logit.dropna()
  )

  if not causal_common_discrepancy_ewma_series_non_nan_logit.empty:
    current_dynamic_discrepancy_ewma_logit = (
        causal_common_discrepancy_ewma_series_non_nan_logit.iloc[-1]
    )
    if len(causal_common_discrepancy_ewma_series_non_nan_logit) >= 2:
      current_dynamic_discrepancy_ewma_lag1_logit = (
          causal_common_discrepancy_ewma_series_non_nan_logit.iloc[-2]
      )
    else:
      current_dynamic_discrepancy_ewma_lag1_logit = (
          current_dynamic_discrepancy_ewma_logit
      )

  # --- NEW: SIR-inspired Season Phase Feature ---
  # Define season phases based on season_week (1-52/53)
  def get_season_phase(season_week):
    if season_week >= 40 or season_week <= 10:
      return 'early_season'  # Weeks 40-52/53, then 1-10
    elif season_week <= 20:
      return 'rising_limb'  # Weeks 11-20
    elif season_week <= 30:
      return 'peak_season'  # Weeks 21-30
    elif season_week <= 39:
      return 'falling_limb'  # Weeks 31-39
    else:
      return 'unknown'  # Should not happen with proper season_week definition

  full_train_data['season_phase'] = full_train_data['season_week'].apply(
      get_season_phase
  )
  test_x_features['season_phase'] = test_x_features['season_week'].apply(
      get_season_phase
  )

  le_season_phase = LabelEncoder()
  # Fit transform on full_train_data, then transform test_x, handling unseen phases
  full_train_data['season_phase_encoded'] = le_season_phase.fit_transform(
      full_train_data['season_phase']
  )
  test_x_features['season_phase_encoded'] = (
      test_x_features['season_phase']
      .map(
          lambda x: le_season_phase.transform([x])[0]
          if x in le_season_phase.classes_
          else -1
      )
      .astype(int)
  )  # Ensure integer type

  # Categorical encoding for 'location' and 'season_year'
  le_location = LabelEncoder()
  full_train_data['location_encoded'] = le_location.fit_transform(
      full_train_data['location']
  )

  le_season_year = LabelEncoder()
  full_train_data['season_year_encoded'] = le_season_year.fit_transform(
      full_train_data['season_year']
  )

  test_x_features['location_encoded'] = (
      test_x_features['location']
      .map(
          lambda x: le_location.transform([x])[0]
          if x in le_location.classes_
          else -1
      )
      .astype(int)
  )
  test_x_features['season_year_encoded'] = (
      test_x_features['season_year']
      .map(
          lambda x: le_season_year.transform([x])[0]
          if x in le_season_year.classes_
          else -1
      )
      .astype(int)
  )

  # --- Define features for LGBM-Level, LGBM-Diff, Huber-AR Models (all predicting in 4th-root-std space) ---

  # Common base features
  base_features = [
      'sin_week',
      'cos_week',
      'horizon',
      'population',
      'season_progress',
      'season_end_proximity',
      'location_encoded',
      'season_year_encoded',
      'time_idx',
      'ili_4rt_feature',
      'season_phase_encoded',
  ]
  categorical_features_lgb = [
      'location_encoded',
      'season_year_encoded',
      'season_phase_encoded',
  ]

  # Logit-based features to be added to 4th-root models (as discussed in plan)
  logit_lags_for_4rt_models = [
      f'logit_admissions_proportion_lag_{lag}' for lag in LAGS
  ]
  logit_rolling_means_for_4rt_models = [
      f'logit_admissions_proportion_rolling_mean_{window}'
      for window in ROLLING_WINDOWS
  ]
  logit_rolling_stds_for_4rt_models = [
      f'logit_admissions_proportion_rolling_std_{window}'
      for window in ROLLING_WINDOWS
  ]
  logit_direct_features_for_4rt_models = [
      'baseline_seasonal_pred_logit',
      'common_discrepancy_ewma_logit',
      'common_discrepancy_ewma_logit_lag_1',  # Add lagged logit discrepancy
  ]

  # LGBM-Level Quantile Model Features (predicts LEVEL 'value_rate_4rt_std')
  lgbm_level_features = (
      [col for col in base_features]
      + [
          'baseline_seasonal_pred',  # Primary target's baseline
          'common_discrepancy_ewma',  # Primary target's discrepancy
          'common_discrepancy_ewma_lag_1',  # Primary target's lagged discrepancy
          'diff_value_rate_4rt_std_lag_1',
      ]
      + [f'value_rate_4rt_std_lag_{lag}' for lag in LAGS]
      + [
          f'value_rate_4rt_std_rolling_mean_{window}'
          for window in ROLLING_WINDOWS
      ]
      + [
          f'value_rate_4rt_std_rolling_std_{window}'
          for window in ROLLING_WINDOWS
      ]
      + logit_lags_for_4rt_models
      + logit_rolling_means_for_4rt_models
      + logit_rolling_stds_for_4rt_models
      + logit_direct_features_for_4rt_models
  )  # Logit features as inputs

  # LGBM-Diff Quantile Model Features (predicts CHANGE 'diff_value_rate_4rt_std')
  lgbm_diff_features = (
      [col for col in base_features]
      + [
          'baseline_seasonal_pred',  # Primary target's baseline
          'common_discrepancy_ewma',  # Primary target's discrepancy
          'common_discrepancy_ewma_lag_1',  # Primary target's lagged discrepancy
          'value_rate_4rt_std_lag_1',
      ]
      + [f'diff_value_rate_4rt_std_lag_{lag}' for lag in DIFF_LAGS]
      + [
          f'value_rate_4rt_std_rolling_mean_{window}'
          for window in ROLLING_WINDOWS
      ]
      + logit_lags_for_4rt_models
      + logit_rolling_means_for_4rt_models
      + logit_rolling_stds_for_4rt_models
      + logit_direct_features_for_4rt_models
  )  # Logit features as inputs

  # Huber AR Model Features (predicts LEVEL 'value_rate_4rt_std')
  ar_model_features = (
      [f'value_rate_4rt_std_lag_{lag}' for lag in AR_LAGS]
      + [col for col in base_features]
      + [
          'baseline_seasonal_pred',  # Primary target's baseline
          'common_discrepancy_ewma_lag_1',  # Primary target's lagged discrepancy
      ]
      + [f'logit_admissions_proportion_lag_{lag}' for lag in [1, 2, 4]]
      + ['baseline_seasonal_pred_logit', 'common_discrepancy_ewma_logit_lag_1']
  )  # Logit baseline and lagged discrepancy as input

  # --- Prepare Training Data for Models ---

  # LGBM-Level Model Training Data
  target_lgbm_level = 'value_rate_4rt_std'
  train_features_target_lgbm_level = full_train_data[
      [target_lgbm_level] + lgbm_level_features
  ].copy()
  train_features_target_lgbm_level = train_features_target_lgbm_level.dropna(
      subset=[target_lgbm_level, 'value_rate_4rt_std_lag_1']
  )
  for col in lgbm_level_features:
    if col not in categorical_features_lgb:
      if train_features_target_lgbm_level[col].dtype in ['float64', 'int64']:
        col_mean = train_features_target_lgbm_level[col].mean()
        train_features_target_lgbm_level[col] = (
            train_features_target_lgbm_level[col].fillna(
                col_mean if not pd.isna(col_mean) else 0.0
            )
        )
  X_train_lgbm_level = train_features_target_lgbm_level[lgbm_level_features]
  y_train_lgbm_level = train_features_target_lgbm_level[target_lgbm_level]
  for col in categorical_features_lgb:
    X_train_lgbm_level[col] = X_train_lgbm_level[col].astype('category')

  # LGBM-Diff Model Training Data
  target_lgbm_diff = 'diff_value_rate_4rt_std'
  train_features_target_lgbm_diff = full_train_data[
      [target_lgbm_diff] + lgbm_diff_features
  ].copy()
  train_features_target_lgbm_diff = train_features_target_lgbm_diff.dropna(
      subset=[target_lgbm_diff, 'value_rate_4rt_std_lag_1']
  )
  for col in lgbm_diff_features:
    if col not in categorical_features_lgb:
      if train_features_target_lgbm_diff[col].dtype in ['float64', 'int64']:
        col_mean = train_features_target_lgbm_diff[col].mean()
        train_features_target_lgbm_diff[col] = train_features_target_lgbm_diff[
            col
        ].fillna(col_mean if not pd.isna(col_mean) else 0.0)
  X_train_lgbm_diff = train_features_target_lgbm_diff[lgbm_diff_features]
  y_train_lgbm_diff = train_features_target_lgbm_diff[target_lgbm_diff]
  for col in categorical_features_lgb:
    X_train_lgbm_diff[col] = X_train_lgbm_diff[col].astype('category')

  # --- 5. Model Training (LGBM Quantile Regression) ---
  lgbm_quantile_models_level = {}
  lgbm_quantile_models_diff = {}

  lgb_params = {
      'objective': 'quantile',
      'metric': 'quantile',
      'n_estimators': 1000,  # Increased estimators for more learning capacity
      'learning_rate': 0.02,  # Slightly reduced learning rate for stability
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'bagging_freq': 1,
      'verbose': -1,
      'n_jobs': -1,
      'seed': 42,
      'num_leaves': 40,  # Slightly increased complexity
      'min_child_samples': 20,
      'max_depth': 12,  # Slightly increased depth
      'reg_alpha': 0.1,  # L1 regularization
      'reg_lambda': 0.1,  # L2 regularization
  }

  for q in QUANTILES:
    lgb_params['alpha'] = q

    # LGBM-Level model
    model_lgbm_level = lgb.LGBMRegressor(**lgb_params)
    model_lgbm_level.fit(
        X_train_lgbm_level,
        y_train_lgbm_level,
        categorical_feature=categorical_features_lgb,
    )
    lgbm_quantile_models_level[q] = model_lgbm_level

    # LGBM-Diff model
    model_lgbm_diff = lgb.LGBMRegressor(**lgb_params)
    model_lgbm_diff.fit(
        X_train_lgbm_diff,
        y_train_lgbm_diff,
        categorical_feature=categorical_features_lgb,
    )
    lgbm_quantile_models_diff[q] = model_lgbm_diff

  # --- 5.1. Huber AR Model Training ---

  # Ensure all AR features are present and handled for training
  for col in ar_model_features:
    if col not in full_train_data.columns:
      full_train_data[col] = 0.0
    if (
        col in categorical_features_lgb
    ):  # Use categorical features list for consistency
      full_train_data[col] = full_train_data[col].astype(int)
    elif full_train_data[col].dtype in ['float64', 'int64']:
      if (
          'lag_' in col
          or 'horizon' == col
          or 'common_discrepancy_ewma_lag_1' == col
          or 'common_discrepancy_ewma_logit_lag_1' == col
      ):
        full_train_data[col] = full_train_data[col].fillna(0.0)
      elif 'time_idx' == col:
        full_train_data[col] = full_train_data[col].fillna(
            full_train_data[col].mean()
            if not full_train_data[col].empty
            else 0.0
        )
      else:
        full_train_data[col] = full_train_data[col].fillna(0.0)

  X_train_ar_raw = full_train_data[ar_model_features].copy()
  y_train_ar = full_train_data['value_rate_4rt_std'].copy()

  ar_train_df_clean = pd.concat([X_train_ar_raw, y_train_ar], axis=1).dropna(
      subset=[f'value_rate_4rt_std_lag_{lag}' for lag in AR_LAGS]
      + ['value_rate_4rt_std']
  )

  X_train_ar = ar_train_df_clean[ar_model_features]
  y_train_ar = ar_train_df_clean['value_rate_4rt_std']

  huber_ar_model = HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-3)
  location_ar_mad_residuals_map = {}

  min_samples_for_ar_huber = len(ar_model_features) + 2
  if (
      X_train_ar.shape[0] >= min_samples_for_ar_huber
      and y_train_ar.nunique() > 1
      and y_train_ar.std() > 0
  ):
    try:
      huber_ar_model.fit(X_train_ar, y_train_ar)
      ar_train_predictions = huber_ar_model.predict(X_train_ar)
      ar_train_residuals = y_train_ar - ar_train_predictions
      ar_residuals_df = ar_train_df_clean[['location']].copy()
      ar_residuals_df['residuals'] = ar_train_residuals

      location_ar_mad_residuals = ar_residuals_df.groupby('location')[
          'residuals'
      ].apply(lambda x: median_abs_deviation(x.dropna()))
      location_ar_mad_residuals = location_ar_mad_residuals.fillna(
          global_mad_value_rate_4rt_std
      )
      location_ar_mad_residuals = location_ar_mad_residuals.replace(
          0, global_mad_value_rate_4rt_std
      )  # Ensure no zero MAD
      location_ar_mad_residuals = np.maximum(
          location_ar_mad_residuals, 0.01
      )  # Ensure positive lower bound
      location_ar_mad_residuals_map = location_ar_mad_residuals.to_dict()
    except Exception:
      huber_ar_model = None
  else:
    huber_ar_model = None

  z_scores = {q: norm.ppf(q) for q in QUANTILES}

  # --- 6. Recursive Prediction for test_x (Ensemble) ---

  test_x_df_full = test_x_features.copy()
  test_x_df_full = test_x_df_full.merge(loc_stats, on='location', how='left')

  global_p95_loc_fallback_test = (
      loc_stats['p95_loc'].mean() if not loc_stats.empty else 1.0
  )
  global_mean_loc_fallback_test = (
      loc_stats['mean_loc'].mean() if not loc_stats.empty else 0.0
  )

  test_x_df_full['p95_loc'] = test_x_df_full['p95_loc'].fillna(
      global_p95_loc_fallback_test
  )
  test_x_df_full['mean_loc'] = test_x_df_full['mean_loc'].fillna(
      global_mean_loc_fallback_test
  )

  predictions_df = test_x_df_full.copy()
  for q in QUANTILES:
    predictions_df[f'quantile_{q}'] = np.nan

  location_history = (
      {}
  )  # Stores `value_rate_4rt_std` history for each location for lags
  location_history_logit = (
      {}
  )  # Stores `logit_admissions_proportion` history for each location for lags (as features)

  required_history_len = max(max(LAGS), max(AR_LAGS), max(DIFF_LAGS) + 1) + 1

  for loc_fips in test_x_df_full['location'].unique():
    last_train_values = full_train_data[
        full_train_data['location'] == loc_fips
    ].sort_values('target_end_date')

    # Initialize value_rate_4rt_std history
    history_data_std = (
        last_train_values['value_rate_4rt_std']
        .tail(required_history_len)
        .to_list()
    )
    if len(history_data_std) < required_history_len:
      pad_val = (
          last_train_values['value_rate_4rt_std'].iloc[-1]
          if not last_train_values['value_rate_4rt_std'].empty
          else global_mean_loc_fallback_test
      )
      history_data_std = [pad_val] * (
          required_history_len - len(history_data_std)
      ) + history_data_std
    location_history[loc_fips] = deque(
        history_data_std, maxlen=required_history_len
    )

    # Initialize logit_admissions_proportion history (for features)
    history_data_logit = (
        last_train_values['logit_admissions_proportion']
        .tail(required_history_len)
        .to_list()
    )
    if len(history_data_logit) < required_history_len:
      pad_val_logit = (
          last_train_values['logit_admissions_proportion'].iloc[-1]
          if not last_train_values['logit_admissions_proportion'].empty
          else 0.0
      )  # Logit mean is often around 0
      history_data_logit = [pad_val_logit] * (
          required_history_len - len(history_data_logit)
      ) + history_data_logit
    location_history_logit[loc_fips] = deque(
        history_data_logit, maxlen=required_history_len
    )

  # Initialize dynamic EWMA state for primary target (value_rate_4rt_std)
  current_dynamic_discrepancy_ewma = global_median_seasonal_residual_fallback
  current_dynamic_discrepancy_ewma_lag1 = (
      global_median_seasonal_residual_fallback
  )

  if not causal_common_discrepancy_ewma_series_non_nan.empty:
    current_dynamic_discrepancy_ewma = (
        causal_common_discrepancy_ewma_series_non_nan.iloc[-1]
    )
    if len(causal_common_discrepancy_ewma_series_non_nan) >= 2:
      current_dynamic_discrepancy_ewma_lag1 = (
          causal_common_discrepancy_ewma_series_non_nan.iloc[-2]
      )
    else:
      current_dynamic_discrepancy_ewma_lag1 = (
          current_dynamic_discrepancy_ewma  # Fallback if not enough history
      )

  # Initialize dynamic EWMA state for logit features (NEW IMPROVEMENT)
  current_dynamic_discrepancy_ewma_logit = (
      global_median_seasonal_residual_fallback_logit
  )
  current_dynamic_discrepancy_ewma_lag1_logit = (
      global_median_seasonal_residual_fallback_logit
  )

  if not causal_common_discrepancy_ewma_series_non_nan_logit.empty:
    current_dynamic_discrepancy_ewma_logit = (
        causal_common_discrepancy_ewma_series_non_nan_logit.iloc[-1]
    )
    if len(causal_common_discrepancy_ewma_series_non_nan_logit) >= 2:
      current_dynamic_discrepancy_ewma_lag1_logit = (
          causal_common_discrepancy_ewma_series_non_nan_logit.iloc[-2]
      )
    else:
      current_dynamic_discrepancy_ewma_lag1_logit = current_dynamic_discrepancy_ewma_logit  # Fallback if not enough history

  # Define a robust overall max admissions per 100k for final capping
  global_max_admissions_per_100k = (
      full_train_data['admissions_per_100k'].max()
      if not full_train_data['admissions_per_100k'].empty
      else 1000.0
  )
  FINAL_ADMISSIONS_PER_100K_CAP = np.maximum(
      global_max_admissions_per_100k * 1.5, 5000.0
  )  # Cap at 1.5x observed max or 5000

  # Recursive forecasting by horizon
  for h in sorted(test_x_df_full['horizon'].unique()):
    current_horizon_rows = test_x_df_full[test_x_df_full['horizon'] == h].copy()

    # Update dynamic common discrepancy for the current horizon (main target)
    current_horizon_rows['common_discrepancy_ewma'] = (
        current_dynamic_discrepancy_ewma
    )
    current_horizon_rows['common_discrepancy_ewma_lag_1'] = (
        current_dynamic_discrepancy_ewma_lag1
    )

    # Update dynamic common discrepancy for logit features (NEW IMPROVEMENT)
    current_horizon_rows['common_discrepancy_ewma_logit'] = (
        current_dynamic_discrepancy_ewma_logit
    )
    current_horizon_rows['common_discrepancy_ewma_logit_lag_1'] = (
        current_dynamic_discrepancy_ewma_lag1_logit
    )
    current_horizon_rows['horizon'] = h

    for loc_fips in current_horizon_rows['location'].unique():
      loc_idx_in_current_horizon = current_horizon_rows['location'] == loc_fips

      # --- Update features for current horizon ---
      current_lags_data_std = list(location_history[loc_fips])
      current_lags_data_logit = list(
          location_history_logit[loc_fips]
      )  # For logit features

      all_lags = sorted(
          list(
              set(
                  LAGS
                  + AR_LAGS
                  + [
                      l
                      for l in DIFF_LAGS
                      if f'diff_value_rate_4rt_std_lag_{l}'
                      in lgbm_diff_features
                  ]
              )
          )
      )

      last_value_rate_4rt_std = (
          current_lags_data_std[-1]
          if current_lags_data_std
          else global_mean_loc_fallback_test
      )
      last_logit_admissions_proportion = (
          current_lags_data_logit[-1] if current_lags_data_logit else 0.0
      )  # Last logit value for feature lags

      for lag_val_index in all_lags:
        # value_rate_4rt_std lags
        if len(current_lags_data_std) >= lag_val_index + 1:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'value_rate_4rt_std_lag_{lag_val_index}',
          ] = current_lags_data_std[-(lag_val_index)]
        else:
          fallback_val = (
              current_lags_data_std[0]
              if current_lags_data_std
              else global_mean_loc_fallback_test
          )
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'value_rate_4rt_std_lag_{lag_val_index}',
          ] = fallback_val

        # logit_admissions_proportion lags (as features)
        if len(current_lags_data_logit) >= lag_val_index + 1:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'logit_admissions_proportion_lag_{lag_val_index}',
          ] = current_lags_data_logit[-(lag_val_index)]
        else:
          fallback_val_logit = (
              current_lags_data_logit[0] if current_lags_data_logit else 0.0
          )
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'logit_admissions_proportion_lag_{lag_val_index}',
          ] = fallback_val_logit

      for diff_lag_val_index in DIFF_LAGS:
        if len(current_lags_data_std) >= diff_lag_val_index + 1:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'diff_value_rate_4rt_std_lag_{diff_lag_val_index}',
          ] = (
              current_lags_data_std[-(diff_lag_val_index)]
              - current_lags_data_std[-(diff_lag_val_index + 1)]
          )
        else:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'diff_value_rate_4rt_std_lag_{diff_lag_val_index}',
          ] = 0.0

      MIN_PERIODS_ROLLING_STD = 3
      for window in ROLLING_WINDOWS:
        # Rolling features for value_rate_4rt_std
        if len(current_lags_data_std) >= window:
          values_for_rolling_std = list(current_lags_data_std)[-window:]
          if len(values_for_rolling_std) >= 1:
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'value_rate_4rt_std_rolling_mean_{window}',
            ] = np.mean(values_for_rolling_std)
            if (
                len(values_for_rolling_std) >= MIN_PERIODS_ROLLING_STD
                and np.std(values_for_rolling_std) > 0
            ):
              current_horizon_rows.loc[
                  loc_idx_in_current_horizon,
                  f'value_rate_4rt_std_rolling_std_{window}',
              ] = np.std(values_for_rolling_std)
            else:
              current_horizon_rows.loc[
                  loc_idx_in_current_horizon,
                  f'value_rate_4rt_std_rolling_std_{window}',
              ] = global_mad_value_rate_4rt_std
          else:
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'value_rate_4rt_std_rolling_mean_{window}',
            ] = current_horizon_rows.loc[
                loc_idx_in_current_horizon, 'mean_loc'
            ].iloc[
                0
            ]
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'value_rate_4rt_std_rolling_std_{window}',
            ] = global_mad_value_rate_4rt_std
        else:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'value_rate_4rt_std_rolling_mean_{window}',
          ] = current_horizon_rows.loc[
              loc_idx_in_current_horizon, 'mean_loc'
          ].iloc[
              0
          ]
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'value_rate_4rt_std_rolling_std_{window}',
          ] = global_mad_value_rate_4rt_std

        # Rolling features for logit_admissions_proportion (as features)
        if len(current_lags_data_logit) >= window:
          values_for_rolling_logit = list(current_lags_data_logit)[-window:]
          if len(values_for_rolling_logit) >= 1:
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'logit_admissions_proportion_rolling_mean_{window}',
            ] = np.mean(values_for_rolling_logit)
            if (
                len(values_for_rolling_logit) >= MIN_PERIODS_ROLLING_STD
                and np.std(values_for_rolling_logit) > 0
            ):
              current_horizon_rows.loc[
                  loc_idx_in_current_horizon,
                  f'logit_admissions_proportion_rolling_std_{window}',
              ] = np.std(values_for_rolling_logit)
            else:
              current_horizon_rows.loc[
                  loc_idx_in_current_horizon,
                  f'logit_admissions_proportion_rolling_std_{window}',
              ] = global_mad_logit_admissions_proportion
          else:
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'logit_admissions_proportion_rolling_mean_{window}',
            ] = 0.0  # Logit mean often near 0
            current_horizon_rows.loc[
                loc_idx_in_current_horizon,
                f'logit_admissions_proportion_rolling_std_{window}',
            ] = global_mad_logit_admissions_proportion
        else:
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'logit_admissions_proportion_rolling_mean_{window}',
          ] = 0.0
          current_horizon_rows.loc[
              loc_idx_in_current_horizon,
              f'logit_admissions_proportion_rolling_std_{window}',
          ] = global_mad_logit_admissions_proportion

      current_horizon_rows.loc[
          loc_idx_in_current_horizon, 'value_rate_4rt_std_lag_1'
      ] = last_value_rate_4rt_std
      current_horizon_rows.loc[
          loc_idx_in_current_horizon, 'logit_admissions_proportion_lag_1'
      ] = last_logit_admissions_proportion

    # Create feature sets for the three models for the current horizon
    X_test_horizon_lgbm_level = current_horizon_rows[lgbm_level_features].copy()
    X_test_horizon_lgbm_diff = current_horizon_rows[lgbm_diff_features].copy()
    X_test_horizon_ar = current_horizon_rows[ar_model_features].copy()

    # Fill any remaining NaNs in X_test_horizon for all models (safety net)
    for df_h, model_features in [
        (X_test_horizon_lgbm_level, lgbm_level_features),
        (X_test_horizon_lgbm_diff, lgbm_diff_features),
        (X_test_horizon_ar, ar_model_features),
    ]:
      for col in model_features:
        if col not in df_h.columns:
          df_h[col] = 0.0
        if col in categorical_features_lgb:
          df_h[col] = df_h[col].astype(int)
        elif df_h[col].dtype in ['float64', 'int64']:
          col_mean_val = df_h[col].mean()
          df_h[col] = df_h[col].fillna(
              col_mean_val if not pd.isna(col_mean_val) else 0.0
          )

    for col in categorical_features_lgb:
      X_test_horizon_lgbm_level[col] = X_test_horizon_lgbm_level[col].astype(
          'category'
      )
      X_test_horizon_lgbm_diff[col] = X_test_horizon_lgbm_diff[col].astype(
          'category'
      )

    # 1. Predict with LGBM-Level Quantile Model (predicts LEVEL 'value_rate_4rt_std')
    lgbm_level_predictions_std = pd.DataFrame(
        index=X_test_horizon_lgbm_level.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    for q in QUANTILES:
      lgbm_level_predictions_std[f'quantile_{q}'] = lgbm_quantile_models_level[
          q
      ].predict(X_test_horizon_lgbm_level)

    # 2. Predict with LGBM-Diff Quantile Model and convert to LEVEL 'value_rate_4rt_std'
    lgbm_diff_predictions_level_std = pd.DataFrame(
        index=X_test_horizon_lgbm_diff.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    for q in QUANTILES:
      diff_preds = lgbm_quantile_models_diff[q].predict(
          X_test_horizon_lgbm_diff
      )
      lgbm_diff_predictions_level_std[f'quantile_{q}'] = (
          current_horizon_rows['value_rate_4rt_std_lag_1'].values + diff_preds
      )

    # 3. Predict with Huber AR Model and derive quantiles (predicts LEVEL 'value_rate_4rt_std')
    ar_predictions_huber_std = pd.DataFrame(
        index=X_test_horizon_ar.index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )
    if huber_ar_model is not None:
      ar_median_preds = huber_ar_model.predict(X_test_horizon_ar)
      for idx, row in X_test_horizon_ar.iterrows():
        loc_fips = current_horizon_rows.loc[idx, 'location']

        mad_residual_loc = location_ar_mad_residuals_map.get(
            loc_fips, global_mad_value_rate_4rt_std
        )
        mad_residual_loc = np.maximum(
            mad_residual_loc, 0.01
        )  # Ensure positive std

        scaled_mad = mad_residual_loc * np.sqrt(h + 1)

        for q_idx, q in enumerate(QUANTILES):
          ar_predictions_huber_std.loc[idx, f'quantile_{q}'] = (
              ar_median_preds[X_test_horizon_ar.index.get_loc(idx)]
              + z_scores[q] * scaled_mad
          )
    else:
      for q_idx, q in enumerate(QUANTILES):
        ar_predictions_huber_std[f'quantile_{q}'] = (
            lgbm_level_predictions_std.iloc[:, QUANTILES.index(0.5)].values
            + z_scores[q] * global_mad_value_rate_4rt_std * np.sqrt(h + 1)
        )

    # Ensemble averaging of quantile predictions from the THREE models (all now in LEVEL space)
    # Ensure the unstandardization and power transformation are robust
    pred_admissions_per_100k_lgbm_level_4rt = (
        lgbm_level_predictions_std.values
        + current_horizon_rows['mean_loc'].values[:, np.newaxis]
    ) * current_horizon_rows['p95_loc'].values[:, np.newaxis]
    pred_admissions_per_100k_lgbm_diff_4rt = (
        lgbm_diff_predictions_level_std.values
        + current_horizon_rows['mean_loc'].values[:, np.newaxis]
    ) * current_horizon_rows['p95_loc'].values[:, np.newaxis]
    pred_admissions_per_100k_ar_4rt = (
        ar_predictions_huber_std.values
        + current_horizon_rows['mean_loc'].values[:, np.newaxis]
    ) * current_horizon_rows['p95_loc'].values[:, np.newaxis]

    # Clip values before applying the 4th power to avoid overflow/extreme values (removed cap_for_4rt_unstd_base)
    pred_admissions_per_100k_lgbm_level = np.power(
        np.maximum(0, pred_admissions_per_100k_lgbm_level_4rt), 4
    )
    pred_admissions_per_100k_lgbm_diff = np.power(
        np.maximum(0, pred_admissions_per_100k_lgbm_diff_4rt), 4
    )
    pred_admissions_per_100k_ar = np.power(
        np.maximum(0, pred_admissions_per_100k_ar_4rt), 4
    )

    # Now average the three predictions (in admissions_per_100k space)
    ensemble_admissions_per_100k = (
        pred_admissions_per_100k_lgbm_level
        + pred_admissions_per_100k_lgbm_diff
        + pred_admissions_per_100k_ar
    ) / 3  # Averaging THREE models now

    ensemble_admissions_per_100k[ensemble_admissions_per_100k < 0] = 0
    # Apply final cap to admissions_per_100k after ensembling
    ensemble_admissions_per_100k = np.minimum(
        ensemble_admissions_per_100k, FINAL_ADMISSIONS_PER_100K_CAP
    )

    pred_admissions = (
        ensemble_admissions_per_100k
        * current_horizon_rows['population'].values[:, np.newaxis]
        / 100_000
    )
    pred_admissions = np.round(pred_admissions).astype(int)

    for q_idx, q in enumerate(QUANTILES):
      predictions_df.loc[current_horizon_rows.index, f'quantile_{q}'] = (
          pred_admissions[:, q_idx]
      )

    # Update location histories for the next horizon
    if h < max(test_x_df_full['horizon']):
      # For value_rate_4rt_std, average the medians of the three models that predict in that space
      ensembled_median_std = (
          lgbm_level_predictions_std['quantile_0.5']
          + lgbm_diff_predictions_level_std['quantile_0.5']
          + ar_predictions_huber_std['quantile_0.5']
      ) / 3

      # For logit_admissions_proportion, re-calculate based on the ensembled median from 4rt-std space.
      # This ensures consistency: future logit features are derived from the same source as primary predictions.

      # Reconstruct the 'admissions_per_100k' from the ensembled median in 4rt-std space
      ensembled_median_4rt_unstd = (
          ensembled_median_std + current_horizon_rows['mean_loc']
      ) * current_horizon_rows['p95_loc']
      ensembled_median_admissions_per_100k = np.power(
          np.maximum(0, ensembled_median_4rt_unstd), 4
      )
      # Apply same final cap as to the full quantile set
      ensembled_median_admissions_per_100k = np.minimum(
          ensembled_median_admissions_per_100k, FINAL_ADMISSIONS_PER_100K_CAP
      )

      # Convert to proportion using the global MAX_RATE_FOR_PROPORTION
      ensembled_median_admissions_proportion = (
          ensembled_median_admissions_per_100k / MAX_RATE_FOR_PROPORTION
      )
      ensembled_median_admissions_proportion = np.clip(
          ensembled_median_admissions_proportion, EPSILON, 1 - EPSILON
      )

      # Re-transform to logit space for history
      median_logit_val = np.log(
          ensembled_median_admissions_proportion
          / (1 - ensembled_median_admissions_proportion)
      )

      for loc_fips in current_horizon_rows['location'].unique():
        # Update main value history with ensembled median (LEVEL)
        median_val_std = ensembled_median_std[
            current_horizon_rows['location'] == loc_fips
        ].iloc[0]
        location_history[loc_fips].append(median_val_std)

        # Update logit value history (for logit-based features in next horizon)
        median_val_logit = median_logit_val[
            current_horizon_rows['location'] == loc_fips
        ].iloc[0]
        location_history_logit[loc_fips].append(median_val_logit)

      # Update the dynamic EWMA for the *primary target* (value_rate_4rt_std) for the next iteration
      median_pred_std_for_discrepancy = (
          ensembled_median_std.median()
      )  # Median of the ensemble medians for this horizon
      median_baseline_pred_std = current_horizon_rows[
          'baseline_seasonal_pred'
      ].median()  # Median of baseline predictions for this horizon

      # Calculate the median predicted discrepancy for this horizon
      predicted_discrepancy_h = (
          median_pred_std_for_discrepancy - median_baseline_pred_std
      )
      if pd.isna(predicted_discrepancy_h):
        predicted_discrepancy_h = 0.0  # Robustness

      # Update EWMA state for the next iteration
      current_dynamic_discrepancy_ewma_lag1 = current_dynamic_discrepancy_ewma
      current_dynamic_discrepancy_ewma = (
          ewma_alpha * predicted_discrepancy_h
          + (1 - ewma_alpha) * current_dynamic_discrepancy_ewma
      )

      # Ensure updated EWMA doesn't become NaN if inputs are NaN
      if pd.isna(current_dynamic_discrepancy_ewma):
        current_dynamic_discrepancy_ewma = (
            global_median_seasonal_residual_fallback
        )

      # Update the dynamic EWMA for the *logit feature* (NEW IMPROVEMENT) for the next iteration
      median_pred_logit_for_discrepancy = (
          median_logit_val.median()
      )  # Median of ensembled logit values for this horizon
      median_baseline_pred_logit = current_horizon_rows[
          'baseline_seasonal_pred_logit'
      ].median()  # Median of logit baseline predictions for this horizon

      # Calculate the median predicted logit discrepancy for this horizon
      predicted_discrepancy_logit_h = (
          median_pred_logit_for_discrepancy - median_baseline_pred_logit
      )
      if pd.isna(predicted_discrepancy_logit_h):
        predicted_discrepancy_logit_h = 0.0  # Robustness

      # Update EWMA state for the next iteration
      current_dynamic_discrepancy_ewma_lag1_logit = (
          current_dynamic_discrepancy_ewma_logit
      )
      current_dynamic_discrepancy_ewma_logit = (
          ewma_alpha * predicted_discrepancy_logit_h
          + (1 - ewma_alpha) * current_dynamic_discrepancy_ewma_logit
      )

      # Ensure updated EWMA doesn't become NaN if inputs are NaN
      if pd.isna(current_dynamic_discrepancy_ewma_logit):
        current_dynamic_discrepancy_ewma_logit = (
            global_median_seasonal_residual_fallback_logit
        )

  # --- 7. Post-processing ---
  quantile_cols = [f'quantile_{q}' for q in QUANTILES]
  predictions_df[quantile_cols] = predictions_df[quantile_cols].apply(
      lambda x: np.maximum(0, x)
  )

  predictions_df[quantile_cols] = predictions_df[quantile_cols].cummax(axis=1)
  predictions_df[quantile_cols] = predictions_df[quantile_cols].apply(
      lambda x: np.maximum(0, x)
  )

  final_predictions = predictions_df.set_index(test_x.index)[quantile_cols]

  return final_predictions


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
