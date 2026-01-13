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
MODEL_NAME = 'Google_SAI-Adapted_11'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from lightgbm import LGBMRegressor
from scipy.stats import norm
import warnings
from collections import deque
from typing import Any  # Keep for completeness, though not explicitly used in the final version of the rewrite cell

# Assume QUANTILES, locations, location_codes, TARGET_STR are available from global scope.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Return quantiles.
  """

  # Deep copy to avoid modifying original dataframes outside this function
  train_x = train_x.copy()
  train_y = train_y.copy()
  test_x = test_x.copy()

  # Store original test_x index for the final output alignment
  original_test_x_index = test_x.index

  # --- 0. Global Mappings & Constants ---
  # Standardize HHS region names to 'Region1', 'Region2', etc.
  hhs_region_map = {
      'Region1': [9, 23, 25, 33, 44, 50],  # CT, ME, MA, NH, RI, VT
      'Region2': [34, 36, 72],  # NJ, NY, PR
      'Region3': [10, 11, 24, 42, 51, 54],  # DE, DC, MD, PA, VA, WV
      'Region4': [
          1,
          12,
          13,
          21,
          28,
          37,
          45,
          47,
      ],  # AL, FL, GA, KY, MS, NC, SC, TN
      'Region5': [17, 18, 26, 27, 39, 55],  # IL, IN, MI, MN, OH, WI
      'Region6': [5, 22, 35, 40, 48],  # AR, LA, NM, OK, TX
      'Region7': [19, 20, 29, 31],  # IA, KS, MO, NE
      'Region8': [8, 30, 38, 46, 49, 56],  # CO, MT, ND, SD, UT, WY
      'Region9': [
          4,
          6,
          32,
      ],  # AZ, CA, NV (HI removed if not in REQUIRED_CDC_LOCATIONS)
      'Region10': [2, 16, 41, 53],  # AK, ID, OR, WA
  }

  NATIONAL_FIPS_STR = '999'  # Dummy FIPS for National level data

  # Calculate populations for HHS regions and National.
  # Ensure 'location' column in 'locations' is int for direct comparison
  locations_filtered_int = locations[
      locations['location'].isin([int(loc) for loc in REQUIRED_CDC_LOCATIONS])
  ]
  hhs_populations = {
      region_name: (
          locations_filtered_int[
              locations_filtered_int['location'].isin(fips_list)
          ]['population'].sum()
      )
      for region_name, fips_list in hhs_region_map.items()
  }
  national_population = locations_filtered_int['population'].sum()

  # Consolidate all population data for easy lookup by location string/FIPS
  all_populations = (
      locations_filtered_int.set_index('location')['population']
      .astype(int)
      .astype(str)
      .to_dict()
  )  # FIPS are int, so map FIPS int to str
  all_populations.update(
      {region_name: pop for region_name, pop in hhs_populations.items()}
  )
  all_populations[NATIONAL_FIPS_STR] = national_population

  scaling_params = (
      {}
  )  # Store scaling parameters (mean, 95th percentile) for inverse transformation
  MAX_LAG = (
      12  # Max lag needed for features or AR model (e.g., lag_12 for LGBM)
  )
  ILI_DATA_CUTOFF_DATE = pd.to_datetime(
      '2022-10-15'
  ).date()  # From problem description
  MIN_OVERLAP_POINTS = 10  # Minimum data points required for linear regression in transformation (increased from 5)

  # NEW: Constants for overflow prevention during inverse transformation
  # Calculate upper bound for 4th root before power(4) to prevent OverflowError
  MAX_INT_COUNT = np.iinfo(np.int64).max
  MAX_SAFE_4RT_VALUE = (
      np.power(MAX_INT_COUNT, 1 / 4) - 1e-6
  )  # Subtract small epsilon for margin

  # --- 1. Initial Data Preparation (NHSN) ---
  df_nhs_actual = pd.DataFrame({
      'target_end_date': train_x['target_end_date'],
      'location': train_x['location'].astype(str),  # Ensure location is string
      'raw_value': train_y,  # This is Total Influenza Admissions (counts)
      'population': train_x['population'],
  }).copy()
  df_nhs_actual['data_source_type'] = 'NHSN_ACTUAL'
  df_nhs_actual['target_end_date'] = pd.to_datetime(
      df_nhs_actual['target_end_date']
  )

  # Convert NHSN raw_value (counts) to rate per 100,000 population
  valid_population_mask_nhs = df_nhs_actual['population'] > 0
  df_nhs_actual['value_rate'] = 0.0
  df_nhs_actual.loc[valid_population_mask_nhs, 'value_rate'] = (
      df_nhs_actual.loc[valid_population_mask_nhs, 'raw_value']
      / df_nhs_actual.loc[valid_population_mask_nhs, 'population']
  ) * 100000

  # --- 2. Initial Data Preparation (ILINet - Raw) ---
  # 2.a. Process ILINet HHS data
  df_hhs_ili = ilinet_hhs.copy()
  df_hhs_ili['target_end_date'] = pd.to_datetime(
      df_hhs_ili['week_start']
  ) + pd.Timedelta(days=6)
  df_hhs_ili['location'] = df_hhs_ili['region'].map(
      lambda x: x.replace(' ', '')
  )  # e.g. 'Region 1' -> 'Region1'
  df_hhs_ili['ili_value'] = df_hhs_ili['weighted_ili']
  df_hhs_ili['population'] = df_hhs_ili['location'].map(hhs_populations)

  # 2.b. Process ILINet National data
  df_nat_ili = ilinet.copy()
  df_nat_ili['target_end_date'] = pd.to_datetime(
      df_nat_ili['week_start']
  ) + pd.Timedelta(days=6)
  df_nat_ili['location'] = NATIONAL_FIPS_STR
  df_nat_ili['ili_value'] = df_nat_ili['weighted_ili']
  df_nat_ili['population'] = national_population

  # 2.c. Process ILINet State data
  df_state_ili = ilinet_state.copy()
  df_state_ili['target_end_date'] = pd.to_datetime(
      df_state_ili['week_start']
  ) + pd.Timedelta(days=6)
  df_state_ili = df_state_ili.rename(columns={'region': 'location_name'})
  df_state_ili = df_state_ili.merge(
      locations[['location', 'location_name']], on='location_name', how='inner'
  )
  df_state_ili['location'] = df_state_ili['location'].astype(
      str
  )  # Ensure location is string
  df_state_ili['ili_value'] = df_state_ili['unweighted_ili']
  df_state_ili['population'] = df_state_ili['location'].map(
      all_populations
  )  # Use combined population map

  # Combine all ILINet data and apply cutoff
  df_ilinet_combined_raw = pd.concat(
      [
          df_hhs_ili[
              ['target_end_date', 'location', 'ili_value', 'population']
          ],
          df_nat_ili[
              ['target_end_date', 'location', 'ili_value', 'population']
          ],
          df_state_ili[
              ['target_end_date', 'location', 'ili_value', 'population']
          ],
      ],
      ignore_index=True,
  )

  df_ilinet_combined_raw = df_ilinet_combined_raw[
      df_ilinet_combined_raw['target_end_date'].dt.date <= ILI_DATA_CUTOFF_DATE
  ]
  df_ilinet_combined_raw = df_ilinet_combined_raw.dropna(
      subset=['ili_value', 'population']
  )
  df_ilinet_combined_raw['ili_value'] = df_ilinet_combined_raw[
      'ili_value'
  ].clip(
      lower=0
  )  # ILI cannot be negative

  # --- 3. Learn Transformation from ILINet to NHSN Rates ---
  transformation_models = {}

  # Calculate global fallback parameters if a location lacks enough overlap data
  global_median_nhs_rate = df_nhs_actual['value_rate'].median()
  global_median_ili_value = df_ilinet_combined_raw['ili_value'].median()

  # IMPROVED: More robust global fallback slope calculation
  global_fallback_slope = 0.0
  global_fallback_intercept = 0.0
  epsilon_ili = 1e-6  # Small constant to prevent division by zero for ILI values close to zero

  if global_median_ili_value < epsilon_ili:  # ILI is zero or near zero
    if global_median_nhs_rate < epsilon_ili:
      # Both near zero, map to 0
      global_fallback_slope = 0.0
      global_fallback_intercept = 0.0
    else:
      # ILI near zero, NHSN positive, map ILI to constant NHSN rate
      global_fallback_slope = 0.0
      global_fallback_intercept = global_median_nhs_rate
  else:  # ILI is positive
    global_fallback_slope = global_median_nhs_rate / (
        global_median_ili_value + epsilon_ili
    )  # Add epsilon to denominator
    global_fallback_intercept = global_median_nhs_rate - (
        global_fallback_slope * global_median_ili_value
    )

  # Ensure fallback slope is reasonable
  global_fallback_slope = np.clip(
      global_fallback_slope, 0.0001, 1000.0
  )  # Lower bound slightly reduced to allow very small positive slopes

  for loc in df_ilinet_combined_raw['location'].unique():
    ili_series_loc = df_ilinet_combined_raw[
        df_ilinet_combined_raw['location'] == loc
    ].set_index('target_end_date')['ili_value']
    nhs_rate_series_loc = df_nhs_actual[
        df_nhs_actual['location'] == loc
    ].set_index('target_end_date')['value_rate']

    overlap_df = pd.merge(
        ili_series_loc.to_frame(),
        nhs_rate_series_loc.to_frame(),
        left_index=True,
        right_index=True,
        how='inner',
        suffixes=('_ili', '_nhs'),
    )
    overlap_df = overlap_df.replace([np.inf, -np.inf], np.nan).dropna()

    if (
        len(overlap_df) >= MIN_OVERLAP_POINTS
        and overlap_df['ili_value_ili'].std() > 1e-6
        and overlap_df['value_nhs'].std() > 1e-6
    ):
      try:
        # Use HuberRegressor for a more robust linear transformation
        model = HuberRegressor(max_iter=5000, epsilon=1.35)
        model.fit(overlap_df[['ili_value_ili']], overlap_df['value_nhs'])
        slope = model.coef_[0]
        intercept = model.intercept_
        slope = np.clip(
            slope, 0.0001, 1000.0
        )  # Ensure slope is positive and not excessively large/small
        transformation_models[loc] = {'slope': slope, 'intercept': intercept}
      except ValueError:
        transformation_models[loc] = {
            'slope': global_fallback_slope,
            'intercept': global_fallback_intercept,
        }
    else:
      transformation_models[loc] = {
          'slope': global_fallback_slope,
          'intercept': global_fallback_intercept,
      }

  # --- 4. Generate Synthetic NHSN History ---
  df_synthetic_nhs_counts_list = []
  for loc, params in transformation_models.items():
    ili_data_for_loc = df_ilinet_combined_raw[
        df_ilinet_combined_raw['location'] == loc
    ].copy()

    ili_data_for_loc['synthetic_nhs_rate'] = (
        ili_data_for_loc['ili_value'] * params['slope'] + params['intercept']
    ).clip(lower=0)

    # Convert synthetic NHSN rate back to counts
    # Ensure population is available for this location
    ili_data_for_loc['population'] = (
        ili_data_for_loc['location'].map(all_populations).fillna(1).astype(int)
    )  # Use a fallback pop of 1 to prevent NaN during computation

    valid_population_mask_ili = ili_data_for_loc['population'] > 0
    ili_data_for_loc['raw_value'] = 0  # Initialize
    ili_data_for_loc.loc[valid_population_mask_ili, 'raw_value'] = (
        (
            ili_data_for_loc.loc[
                valid_population_mask_ili, 'synthetic_nhs_rate'
            ]
            * ili_data_for_loc.loc[valid_population_mask_ili, 'population']
            / 100000
        )
        .round()
        .astype(int)
        .clip(lower=0)
    )

    ili_data_for_loc['data_source_type'] = 'NHSN_SYNTHETIC'
    df_synthetic_nhs_counts_list.append(
        ili_data_for_loc[[
            'target_end_date',
            'location',
            'raw_value',
            'population',
            'data_source_type',
        ]]
    )
  df_synthetic_nhs_counts_overall = pd.concat(
      df_synthetic_nhs_counts_list, ignore_index=True
  )

  # --- 5. Create Augmented Target Data (`df_augmented_target_data`) ---
  earliest_nhs_dates = (
      df_nhs_actual.groupby('location')['target_end_date'].min().to_dict()
  )

  filtered_synthetic_dfs = []
  for loc in df_synthetic_nhs_counts_overall['location'].unique():
    loc_synthetic_data = df_synthetic_nhs_counts_overall[
        df_synthetic_nhs_counts_overall['location'] == loc
    ]
    if loc in earliest_nhs_dates:
      earliest_date = earliest_nhs_dates[loc]
      filtered_synthetic_dfs.append(
          loc_synthetic_data[
              loc_synthetic_data['target_end_date'] < earliest_date
          ]
      )
    else:  # For locations with no actual NHSN data, keep all synthetic data
      filtered_synthetic_dfs.append(loc_synthetic_data)

  df_filtered_synthetic = (
      pd.concat(filtered_synthetic_dfs, ignore_index=True)
      if filtered_synthetic_dfs
      else pd.DataFrame()
  )

  df_augmented_target_data = pd.concat(
      [df_nhs_actual, df_filtered_synthetic], ignore_index=True
  )
  df_augmented_target_data = df_augmented_target_data.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)
  df_augmented_target_data = df_augmented_target_data.drop_duplicates(
      subset=['target_end_date', 'location'], keep='first'
  )

  # --- 6. Data Standardization (Refactored) ---
  all_raw_data = df_augmented_target_data.copy()

  valid_population_mask = all_raw_data['population'] > 0
  all_raw_data['value_rate'] = 0.0
  all_raw_data.loc[valid_population_mask, 'value_rate'] = (
      all_raw_data.loc[valid_population_mask, 'raw_value']
      / all_raw_data.loc[valid_population_mask, 'population']
  ) * 100000

  all_raw_data['value_4rt'] = np.power(
      all_raw_data['value_rate'].clip(lower=0), 1 / 4
  )

  # Calculate global mean for padding deque later AND consistent fillna
  global_standardized_mean = all_raw_data['value_4rt'].mean()

  for loc_val, group in all_raw_data.groupby('location'):
    if (
        group['value_4rt'].empty
        or group['value_4rt'].min() == group['value_4rt'].max()
    ):
      scaling_params[loc_val] = {'mean': 0, 'p95': 1}
    else:
      p95 = group['value_4rt'].quantile(0.95)
      mean_val = group['value_4rt'].mean()
      # IMPROVED: Ensure p95 is not too small to avoid excessively large scaled values
      scaling_params[loc_val] = {
          'mean': mean_val,
          'p95': max(1e-6, p95),
      }  # Ensure p95 is at least a small positive number

  all_raw_data['standardized_value'] = 0.0
  for loc_val, params in scaling_params.items():
    mask = all_raw_data['location'] == loc_val
    all_raw_data.loc[mask, 'standardized_value'] = (
        all_raw_data.loc[mask, 'value_4rt'] / params['p95']
    ) - params['mean']

  # --- 7. Temporal Feature Engineering ---
  all_raw_data['year'] = all_raw_data['target_end_date'].dt.year
  all_raw_data['week'] = (
      all_raw_data['target_end_date'].dt.isocalendar().week.astype(int)
  )
  all_raw_data['sin_week'] = np.sin(2 * np.pi * all_raw_data['week'] / 52)
  all_raw_data['cos_week'] = np.cos(2 * np.pi * all_raw_data['week'] / 52)

  def get_time_until_christmas(date_val):
    """Calculates time in weeks until nearest Christmas for a single date."""
    # Ensure date_val is a date object for consistent arithmetic
    if isinstance(date_val, pd.Timestamp):
      date_val = date_val.date()

    current_year = date_val.year

    christmas_this_year = pd.to_datetime(f'{current_year}-12-25').date()
    christmas_next_year = pd.to_datetime(f'{current_year + 1}-12-25').date()
    christmas_prev_year = pd.to_datetime(f'{current_year - 1}-12-25').date()

    dist_this = (christmas_this_year - date_val).days / 7
    dist_next = (christmas_next_year - date_val).days / 7
    dist_prev = (christmas_prev_year - date_val).days / 7

    min_abs_dist = np.min(
        [np.abs(dist_this), np.abs(dist_next), np.abs(dist_prev)]
    )

    if np.abs(dist_this) == min_abs_dist:
      return dist_this
    elif np.abs(dist_next) == min_abs_dist:
      return dist_next
    else:
      return dist_prev

  # Apply to series with a vectorized approach if possible, or element-wise
  all_raw_data['time_until_christmas'] = all_raw_data['target_end_date'].apply(
      get_time_until_christmas
  )

  all_raw_data['season_id'] = all_raw_data['year'].where(
      all_raw_data['week'] >= 40, all_raw_data['year'] - 1
  )

  # --- 8. Lag & Rolling Features (Refactored: Unified Target Signal) ---
  lag_features_to_gen = [1, 2, 3, 4, 5, 6, 7, 8, 12]
  rolling_window_features_to_gen = [3, 6]

  combined_features_long_list = []

  for loc_val, group in all_raw_data.groupby('location'):
    group = group.sort_values('target_end_date').copy()

    temp_df = group[
        ['target_end_date', 'location', 'standardized_value']
    ].copy()

    for lag in lag_features_to_gen:
      temp_df[f'lag_{lag}'] = temp_df['standardized_value'].shift(lag)

    for window in rolling_window_features_to_gen:
      temp_df[f'rolling_mean_{window}'] = (
          temp_df['standardized_value']
          .shift(1)
          .rolling(window=window, min_periods=1)
          .mean()
      )
      temp_df[f'rolling_std_{window}'] = (
          temp_df['standardized_value']
          .shift(1)
          .rolling(window=window, min_periods=1)
          .std()
      )

    # Trend and curvature are also based on lagged standardized values
    temp_df[f'trend_1_2'] = temp_df[f'lag_1'] - temp_df[f'lag_2']
    temp_df[f'curvature_1_2_3'] = (temp_df[f'lag_1'] - temp_df[f'lag_2']) - (
        temp_df[f'lag_2'] - temp_df[f'lag_3']
    )

    combined_features_long_list.append(temp_df)

  combined_features_long_df = pd.concat(
      combined_features_long_list, ignore_index=True
  )

  all_features_lookup_df = pd.merge(
      all_raw_data[[
          'target_end_date',
          'location',
          'standardized_value',
          'year',
          'week',
          'sin_week',
          'cos_week',
          'time_until_christmas',
          'season_id',
      ]],
      combined_features_long_df.drop(
          columns=['standardized_value'], errors='ignore'
      ),
      on=['target_end_date', 'location'],
      how='left',
  )
  all_features_lookup_df = all_features_lookup_df.set_index(
      ['target_end_date', 'location']
  ).sort_index()

  # --- 9. Training Data Assembly (Vectorized) ---
  train_max_date = pd.to_datetime(train_x['target_end_date'].max())
  train_target_data_filtered = all_raw_data[
      all_raw_data['target_end_date'] <= train_max_date
  ].copy()

  lgbm_horizons_to_train = [-1, 0, 1, 2, 3]

  # Create a base dataframe of all (target_end_date, location, horizon) combinations
  # and initial standardized values for the target week
  base_lgbm_train_data = []
  for h in lgbm_horizons_to_train:
    temp_df = train_target_data_filtered[
        ['target_end_date', 'location', 'standardized_value', 'season_id']
    ].copy()
    temp_df['horizon'] = h
    temp_df['reference_date'] = temp_df['target_end_date'] - pd.Timedelta(
        weeks=h
    )
    temp_df['last_available_data_date'] = temp_df[
        'reference_date'
    ] - pd.Timedelta(weeks=1)
    base_lgbm_train_data.append(temp_df)

  lgbm_train_data_expanded = pd.concat(base_lgbm_train_data, ignore_index=True)
  lgbm_train_data_expanded = lgbm_train_data_expanded.rename(
      columns={'standardized_value': 'standardized_value_at_target_end_date'}
  )

  # Merge features for the target_end_date (year, week, sin/cos_week, time_until_christmas)
  features_target_end_date = all_features_lookup_df.reset_index()[[
      'target_end_date',
      'location',
      'year',
      'week',
      'sin_week',
      'cos_week',
      'time_until_christmas',
  ]]
  lgbm_train_data_expanded = pd.merge(
      lgbm_train_data_expanded,
      features_target_end_date,
      on=['target_end_date', 'location'],
      how='left',
  )

  # Merge features for the last_available_data_date (including its standardized_value and derived features)
  features_from_last_available_data = (
      all_features_lookup_df.reset_index().copy()
  )
  features_from_last_available_data = features_from_last_available_data.rename(
      columns={
          'target_end_date': (
              'last_available_data_date_key'
          ),  # Temporary key for merge
          'standardized_value': (
              'standardized_value_at_last_available_data_date'
          ),
      }
  )

  # These lagged features are now correctly indexed for 'last_available_data_date'
  cols_to_merge_from_last_obs = [
      'last_available_data_date_key',
      'location',
      'standardized_value_at_last_available_data_date',
  ] + [
      col
      for col in all_features_lookup_df.columns
      if any(
          col.startswith(p)
          for p in ['lag_', 'rolling_', 'trend_', 'curvature_']
      )
  ]

  lgbm_train_data_expanded = pd.merge(
      lgbm_train_data_expanded,
      features_from_last_available_data[cols_to_merge_from_last_obs],
      left_on=['last_available_data_date', 'location'],
      right_on=['last_available_data_date_key', 'location'],
      how='left',
  ).drop(
      columns=['last_available_data_date_key']
  )  # Drop temporary key

  # Calculate the target variable: change_standardized_value
  lgbm_train_data_expanded['change_standardized_value'] = (
      lgbm_train_data_expanded['standardized_value_at_target_end_date']
      - lgbm_train_data_expanded[
          'standardized_value_at_last_available_data_date'
      ]
  )

  # Crucial step: Drop rows where the key target components were originally NaN
  lgbm_train_data_expanded = lgbm_train_data_expanded.dropna(
      subset=[
          'standardized_value_at_target_end_date',
          'standardized_value_at_last_available_data_date',
          'change_standardized_value',
      ]
  )

  # Prepare feature sets for LGBM models
  # NEW: Add 'last_observed_std_value' as a feature for LGBM full model
  lgbm_train_data_expanded['last_observed_std_value'] = (
      lgbm_train_data_expanded['standardized_value_at_last_available_data_date']
  )

  full_feature_names_template = [
      'year',
      'week',
      'sin_week',
      'cos_week',
      'time_until_christmas',
      'horizon',
      'last_observed_std_value',
  ] + [  # Added last_observed_std_value
      col
      for col in all_features_lookup_df.columns
      if any(
          col.startswith(p)
          for p in ['lag_', 'rolling_', 'trend_', 'curvature_']
      )
  ]

  full_feature_names = [
      col
      for col in full_feature_names_template
      if col in lgbm_train_data_expanded.columns
  ]

  # Consistent fillna for training features for LGBM models
  X_train_full_lgbm = lgbm_train_data_expanded[full_feature_names].fillna(
      global_standardized_mean
  )
  y_train_lgbm_change = lgbm_train_data_expanded['change_standardized_value']

  # Features for Model 2 (No Level Features)
  level_feature_bases = (
      [f'lag_{i}' for i in lag_features_to_gen]
      + [f'rolling_mean_{w}' for w in rolling_window_features_to_gen]
      + ['last_observed_std_value']
  )  # Also exclude this new level feature

  no_level_feature_names = [
      col for col in full_feature_names if col not in level_feature_bases
  ]
  X_train_no_level_lgbm = lgbm_train_data_expanded[
      no_level_feature_names
  ].fillna(global_standardized_mean)

  # AR Model Training Data (Vectorized)
  # The AR model directly predicts standardized_value.
  ar_lags_cols = [f'lag_{i}' for i in range(1, 9)]  # AR(8) uses lag_1 to lag_8
  ar_features_to_train_template = ar_lags_cols + [
      'time_until_christmas',
      'sin_week',
      'cos_week',
  ]

  # Start with filtered train_target_data
  ar_train_df = train_target_data_filtered[
      ['target_end_date', 'location', 'standardized_value']
  ].copy()

  # Merge AR features from all_features_lookup_df
  # The `lag_N` features in all_features_lookup_df are already relative to the `target_end_date` in that row.
  # So `lag_1` here means value at `target_end_date - 1 week`.
  ar_train_df = pd.merge(
      ar_train_df,
      all_features_lookup_df.reset_index()[
          ['target_end_date', 'location']
          + [
              col
              for col in ar_features_to_train_template
              if col in all_features_lookup_df.columns
          ]
      ],
      on=['target_end_date', 'location'],
      how='left',
  )

  # Consistent fillna for AR model training features
  ar_features_to_train = [
      col for col in ar_features_to_train_template if col in ar_train_df.columns
  ]
  X_ar_train = ar_train_df[ar_features_to_train].fillna(
      global_standardized_mean
  )
  y_ar_train = ar_train_df['standardized_value']

  # Drop rows where standardized_value (the target) might be NaN (should be handled by train_target_data_filtered)
  X_ar_train, y_ar_train = (
      X_ar_train.loc[y_ar_train.dropna().index],
      y_ar_train.dropna(),
  )

  # --- 10. Component Model 1: Tree-Based Quantile Regression (Full Features) ---
  lgbm_model_full_quantiles = {}
  unique_seasons = lgbm_train_data_expanded['season_id'].unique()
  for q in QUANTILES:
    bagged_models = []
    for _ in range(3):  # 3 bags for bagging
      sample_seasons = np.random.choice(
          unique_seasons, size=int(len(unique_seasons) * 0.7), replace=True
      )
      sample_indices = lgbm_train_data_expanded[
          lgbm_train_data_expanded['season_id'].isin(sample_seasons)
      ].index

      sample_train_X = X_train_full_lgbm.loc[sample_indices]
      sample_train_y = y_train_lgbm_change.loc[sample_indices]

      lgbm = LGBMRegressor(
          objective='quantile',
          alpha=q,
          random_state=42 + _,
          n_estimators=100,
          learning_rate=0.05,
          verbose=-1,
          n_jobs=-1,
      )
      lgbm.fit(sample_train_X, sample_train_y)
      bagged_models.append(lgbm)
    lgbm_model_full_quantiles[q] = bagged_models

  # --- 11. Component Model 2: Tree-Based Quantile Regression (No Level Features) ---
  lgbm_model_no_level_quantiles = {}
  for q in QUANTILES:
    bagged_models = []
    for _ in range(3):  # 3 bags for bagging
      sample_seasons = np.random.choice(
          unique_seasons, size=int(len(unique_seasons) * 0.7), replace=True
      )
      sample_indices = lgbm_train_data_expanded[
          lgbm_train_data_expanded['season_id'].isin(sample_seasons)
      ].index

      sample_train_X = X_train_no_level_lgbm.loc[sample_indices]
      sample_train_y = y_train_lgbm_change.loc[sample_indices]

      lgbm = LGBMRegressor(
          objective='quantile',
          alpha=q,
          random_state=42 + _,
          n_estimators=100,
          learning_rate=0.05,
          verbose=-1,
          n_jobs=-1,
      )
      lgbm.fit(sample_train_X, sample_train_y)
      bagged_models.append(lgbm)
    lgbm_model_no_level_quantiles[q] = bagged_models

  # --- 12. Component Model 3: Autoregressive with Covariates (Linear Regression + Innovation Variance) ---
  # Train a single Linear Regression model to predict the median standardized value (shared coefficients)
  # The method contract describes "Fit an autoregressive model of order 8" and mentions "share autoregressive coefficients".
  # HuberRegressor is used here for robustness.
  ar_median_model = HuberRegressor(max_iter=5000, epsilon=1.35, alpha=0.0)
  ar_median_model.fit(X_ar_train, y_ar_train)

  # Calculate residuals and estimate innovation variance for each location
  ar_train_preds_median = ar_median_model.predict(X_ar_train)

  # Need to merge predictions back to ar_train_df to access 'location' for grouping
  ar_train_df_with_preds = ar_train_df.copy()
  ar_train_df_with_preds['median_pred'] = ar_train_preds_median
  ar_train_df_with_preds['residuals'] = (
      ar_train_df_with_preds['standardized_value']
      - ar_train_df_with_preds['median_pred']
  )

  # Estimate separate innovation variance (standard deviation of residuals) for each location
  # Use a small constant to prevent zero std dev which would lead to zero-width intervals
  location_innovation_std_devs = (
      ar_train_df_with_preds.groupby('location')['residuals']
      .std()
      .fillna(1e-6)
      .to_dict()
  )

  # --- 13. Test Prediction (Refactored: Optimized Recursive Multi-Step Forecasting) ---
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])
  test_x['location'] = test_x['location'].astype(
      str
  )  # Ensure location is string

  output_df_cols = [f'quantile_{q}' for q in QUANTILES]
  final_predictions_standardized = pd.DataFrame(
      index=test_x.set_index(['reference_date', 'horizon', 'location']).index,
      columns=output_df_cols,
      dtype=float,
  )

  # Pre-calculate location means for padding
  location_standardized_means = (
      all_features_lookup_df.groupby('location')['standardized_value']
      .mean()
      .to_dict()
  )

  for (ref_date, loc_str), group_test_x in test_x.groupby(
      ['reference_date', 'location']
  ):
    horizons_for_this_group = sorted(group_test_x['horizon'].unique())

    last_observed_date = ref_date - pd.Timedelta(weeks=1)

    history_series = all_features_lookup_df.loc[
        (slice(None), loc_str), 'standardized_value'
    ]
    history_series = history_series[
        history_series.index.get_level_values('target_end_date')
        <= last_observed_date
    ].sort_index()

    # Initialize deque with available history, pad with location-specific mean or global mean
    initial_history = history_series.tolist()
    if len(initial_history) < MAX_LAG:
      padding_value = location_standardized_means.get(
          loc_str, global_standardized_mean
      )
      predicted_standardized_values_deque = deque(
          [padding_value] * (MAX_LAG - len(initial_history)) + initial_history,
          maxlen=MAX_LAG,
      )
    else:
      predicted_standardized_values_deque = deque(
          initial_history[-MAX_LAG:], maxlen=MAX_LAG
      )

    for horizon_val in range(
        min(horizons_for_this_group), max(horizons_for_this_group) + 1
    ):
      current_target_date = ref_date + pd.Timedelta(weeks=horizon_val)

      # Store common temporal features for current_target_date
      common_temporal_features = {
          'year': current_target_date.year,
          'week': int(current_target_date.isocalendar().week),
          'sin_week': np.sin(
              2 * np.pi * int(current_target_date.isocalendar().week) / 52
          ),
          'cos_week': np.cos(
              2 * np.pi * int(current_target_date.isocalendar().week) / 52
          ),
          'time_until_christmas': get_time_until_christmas(current_target_date),
          'horizon': horizon_val,
      }

      # The last available standardized value (X_{t-1} relative to current_target_date).
      # This is the last value in the deque, representing the state right before the current forecast.
      current_last_observed_std_value = predicted_standardized_values_deque[
          MAX_LAG - 1
      ]

      # --- Feature generation for LGBM Models ---
      # LGBM models predict the CHANGE in standardized value: X_t - X_{t-1}
      # Features `lag_N` for LGBM represent `X_{t-1-N}` (value N weeks before `X_{t-1}`)
      lgbm_pred_features = common_temporal_features.copy()
      lgbm_pred_features['last_observed_std_value'] = (
          current_last_observed_std_value  # X_{t-1}
      )

      current_lags_for_lgbm = {}
      for lag in lag_features_to_gen:
        # `lag_N` feature in LGBM refers to the value N weeks prior to `current_last_observed_std_value` (which is X_{t-1})
        # So, it's X_{t-1-N}. Deque index: MAX_LAG - 1 (for X_{t-1}) - N
        deque_index = MAX_LAG - 1 - lag
        lag_val = (
            predicted_standardized_values_deque[deque_index]
            if deque_index >= 0
            else global_standardized_mean
        )
        current_lags_for_lgbm[f'lag_{lag}'] = lag_val
        lgbm_pred_features[f'lag_{lag}'] = lag_val

      for window in rolling_window_features_to_gen:
        # Rolling features for LGBM are based on values from `window` weeks prior to X_{t-1}, up to X_{t-2}
        # Slice should be `deque[MAX_LAG - 1 - window : MAX_LAG - 1]`
        rolling_data_slice = list(predicted_standardized_values_deque)[
            max(0, MAX_LAG - 1 - window) : MAX_LAG - 1
        ]
        lgbm_pred_features[f'rolling_mean_{window}'] = (
            np.mean(rolling_data_slice)
            if len(rolling_data_slice) > 0
            else global_standardized_mean
        )
        lgbm_pred_features[f'rolling_std_{window}'] = (
            np.std(rolling_data_slice) if len(rolling_data_slice) > 0 else 0.0
        )

      # Populate trend and curvature based on correctly indexed lags for LGBM
      lgbm_pred_features['trend_1_2'] = current_lags_for_lgbm.get(
          'lag_1', global_standardized_mean
      ) - current_lags_for_lgbm.get('lag_2', global_standardized_mean)
      lgbm_pred_features['curvature_1_2_3'] = (
          current_lags_for_lgbm.get('lag_1', global_standardized_mean)
          - current_lags_for_lgbm.get('lag_2', global_standardized_mean)
      ) - (
          current_lags_for_lgbm.get('lag_2', global_standardized_mean)
          - current_lags_for_lgbm.get('lag_3', global_standardized_mean)
      )

      # Ensure prediction DataFrames use the correct and aligned feature sets
      X_test_pred_full = pd.DataFrame(
          [lgbm_pred_features], columns=full_feature_names
      ).fillna(global_standardized_mean)
      X_test_pred_no_level = pd.DataFrame(
          [lgbm_pred_features], columns=no_level_feature_names
      ).fillna(global_standardized_mean)

      # --- Feature generation for AR Model ---
      # AR model predicts X_t directly using X_{t-1}, X_{t-2}, ..., X_{t-8}
      # `lag_N` feature for AR refers to X_{t-N}. Deque index: MAX_LAG - N
      ar_pred_features = {
          'time_until_christmas': common_temporal_features[
              'time_until_christmas'
          ],
          'sin_week': common_temporal_features['sin_week'],
          'cos_week': common_temporal_features['cos_week'],
      }
      for lag in range(1, 9):  # AR(8)
        deque_index = MAX_LAG - lag
        lag_val = (
            predicted_standardized_values_deque[deque_index]
            if deque_index >= 0
            else global_standardized_mean
        )
        ar_pred_features[f'lag_{lag}'] = lag_val

      X_test_pred_ar = pd.DataFrame(
          [ar_pred_features], columns=ar_features_to_train
      ).fillna(global_standardized_mean)

      preds_q_model1 = pd.DataFrame(index=[0], columns=output_df_cols)
      preds_q_model2 = pd.DataFrame(index=[0], columns=output_df_cols)
      preds_q_model3 = pd.DataFrame(
          index=[0], columns=output_df_cols
      )  # For AR model direct quantiles

      # --- Model 3 (AR Quantile Regression with Median + Innovation Variance) Predictions ---
      # Predict the median using the AR median model (HuberRegressor)
      ar_median_pred = ar_median_model.predict(X_test_pred_ar)[0]

      # Get location-specific standard deviation of residuals
      ar_innovation_std = location_innovation_std_devs.get(
          loc_str, 1e-6
      )  # Fallback to a small value

      # --- CORRECTION: Calculate steps_ahead for AR model innovation variance scaling ---
      # steps_ahead is the number of forecast steps from the LAST ACTUAL OBSERVED data (ref_date - 1 week)
      # to the current target date (ref_date + horizon_val weeks).
      # For horizon_val = -1 (X_{ref_date-1}), it's 1 step from X_{ref_date-1} (current_last_observed_std_value)
      # For horizon_val = 0 (X_{ref_date}), it's 1 step from X_{ref_date-1} (current_last_observed_std_value)
      # If the base observation is ref_date - 1 week, then the horizon steps are (horizon_val - (-1)) + 1 = horizon_val + 2
      # No, this is incorrect. The AR model uses `current_last_observed_std_value` (X_{t-1})
      # So from last *actual* data `X_{ref_date-1}`, how many steps to `X_{ref_date+horizon_val}`?
      # It's (ref_date + horizon_val) - (ref_date - 1) = horizon_val + 1 weeks.
      # We enforce a minimum of 1 step for the standard deviation scaling.
      steps_ahead = max(1, horizon_val + 1)
      ar_innovation_std_scaled = ar_innovation_std * np.sqrt(steps_ahead)

      for q in QUANTILES:
        # Model 1 predictions (standardized value = change in standardized value + last observed standardized value)
        bagged_preds_1_change = np.mean([
            model.predict(X_test_pred_full)[0]
            for model in lgbm_model_full_quantiles[q]
        ])
        preds_q_model1.loc[0, f'quantile_{q}'] = (
            bagged_preds_1_change + current_last_observed_std_value
        )

        # Model 2 predictions (standardized value = change in standardized value + last observed standardized value)
        bagged_preds_2_change = np.mean([
            model.predict(X_test_pred_no_level)[0]
            for model in lgbm_model_no_level_quantiles[q]
        ])
        preds_q_model2.loc[0, f'quantile_{q}'] = (
            bagged_preds_2_change + current_last_observed_std_value
        )

        # Model 3 (AR) quantiles derived from median forecast and location-specific innovation std dev
        preds_q_model3.loc[0, f'quantile_{q}'] = (
            ar_median_pred + norm.ppf(q) * ar_innovation_std_scaled
        )

      ensemble_pred_standardized = (
          preds_q_model1.loc[0] + preds_q_model2.loc[0] + preds_q_model3.loc[0]
      ) / 3

      # Ensure the index tuple exists before assignment to avoid potential KeyError on missing combinations
      if (
          ref_date,
          horizon_val,
          loc_str,
      ) in final_predictions_standardized.index:
        final_predictions_standardized.loc[(ref_date, horizon_val, loc_str)] = (
            ensemble_pred_standardized.values
        )

      # Update deque with the ensemble median forecast for the current target date
      # This makes the multi-step ahead forecasts recursive.
      median_std_pred = ensemble_pred_standardized['quantile_0.5']
      predicted_standardized_values_deque.append(median_std_pred)

  # --- 14. Ensemble Formation and Inverse Transformation ---
  final_predictions_df = pd.DataFrame(
      index=final_predictions_standardized.index, columns=output_df_cols
  )

  test_x_indexed = test_x.set_index(['reference_date', 'horizon', 'location'])

  for idx in final_predictions_standardized.index:
    loc_str = idx[2]

    params = scaling_params.get(loc_str, {'mean': 0, 'p95': 1})
    mean_val = params['mean']
    p95_val = params['p95']

    # Ensure values are float before operations
    inverse_standardized = (
        final_predictions_standardized.loc[idx].astype(float) + mean_val
    ) * p95_val

    # IMPROVED: Clip inverse_standardized BEFORE applying power(4) to prevent OverflowError
    inverse_standardized = inverse_standardized.clip(
        lower=0, upper=MAX_SAFE_4RT_VALUE
    )
    inverse_4rt = np.power(inverse_standardized, 4)

    location_population = test_x_indexed.loc[idx]['population']

    # Clip to max int value after rounding and converting to int to prevent overflow
    final_counts = (
        (inverse_4rt * location_population / 100000)
        .round()
        .clip(lower=0, upper=MAX_INT_COUNT)
        .astype(int)
    )
    final_predictions_df.loc[idx] = final_counts

  # --- 15. Monotonicity Enforcement ---
  for idx_tuple in final_predictions_df.index:
    final_predictions_df.loc[idx_tuple] = final_predictions_df.loc[
        idx_tuple
    ].cummax()

  # --- 16. Final Output Index Alignment ---
  final_predictions_df_reset = final_predictions_df.reset_index()

  test_x_temp = test_x.copy()
  test_x_temp['location'] = test_x_temp['location'].astype(str)

  output_predictions_aligned = pd.merge(
      test_x_temp,
      final_predictions_df_reset,
      on=['reference_date', 'horizon', 'location'],
      how='left',
  )

  output_predictions_final = output_predictions_aligned.set_index(
      original_test_x_index
  )
  output_predictions_final = output_predictions_final[
      [f'quantile_{q}' for q in QUANTILES]
  ]

  return output_predictions_final


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
