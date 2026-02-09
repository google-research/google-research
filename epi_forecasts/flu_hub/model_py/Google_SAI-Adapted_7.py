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
MODEL_NAME = 'Google_SAI-Adapted_7'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import warnings

# Global constants as defined in the preamble, accessible in this scope.
# QUANTILES = [...]
# HORIZONS = [0, 1, 2, 3]
# TARGET_STR = 'Total Influenza Admissions'

# Configuration Constants for the model
N_BAGS = 3  # Number of bagging iterations
LAG_WINDOW = 8  # Number of past weeks to consider for lags, trend, curvature (Increased from 4 to 8)
FLU_SEASON_START_WEEK = 40  # Week 40 of year Y (approx. Sep/Oct)
FLU_SEASON_END_WEEK = 20  # Week 20 of year Y+1 (approx. May)
# Pandemic seasons to exclude for training. These are the *start years* of the seasons.
# The 2020/21 season (starting in late 2020) has some target data and is included.
PANDEMIC_SEASONS_YEARS = [
    2019
]  # Only exclude the 2019/20 season which pre-dates target data availability.
ILINET_CUTOFF_DATE = pd.Timestamp(
    '2022-10-15'
)  # Date after which ILINet data is no longer available
NHSN_DATA_START_DATE = pd.Timestamp(
    '2020-08-01'
)  # Approximate start of NHSN target data availability

# Mapping of State FIPS to HHS Region FIPS - based on CDC HHS region definitions
STATE_TO_HHS_REGION_MAP = {
    # HHS Region 1 (FIPS 90): CT, ME, MA, NH, RI, VT
    9: 90,
    23: 90,
    25: 90,
    33: 90,
    44: 90,
    50: 90,
    # HHS Region 2 (FIPS 91): NJ, NY, PR, VI
    34: 91,
    36: 91,
    72: 91,  # FIPS 72 (Puerto Rico)
    # HHS Region 3 (FIPS 92): DE, DC, MD, PA, VA, WV
    10: 92,
    11: 92,
    24: 92,
    42: 92,
    51: 92,
    54: 92,  # FIPS 11 (District of Columbia)
    # HHS Region 4 (FIPS 93): AL, FL, GA, KY, MS, NC, SC, TN
    1: 93,
    12: 93,
    13: 93,
    21: 93,
    28: 93,
    37: 93,
    45: 93,
    47: 93,
    # HHS Region 5 (FIPS 94): IL, IN, MI, MN, OH, WI
    17: 94,
    18: 94,
    26: 94,
    27: 94,
    39: 94,
    55: 94,
    # HHS Region 6 (FIPS 95): AR, LA, NM, OK, TX
    5: 95,
    22: 95,
    35: 95,
    40: 95,
    48: 95,
    # HHS Region 7 (FIPS 96): IA, KS, MO, NE
    19: 96,
    20: 96,
    29: 96,
    31: 96,
    # HHS Region 8 (FIPS 97): CO, MT, ND, SD, UT, WY
    8: 97,
    30: 97,
    38: 97,
    46: 97,
    49: 97,
    56: 97,
    # HHS Region 9 (FIPS 98): AZ, CA, HI, NV (also includes American Samoa, Guam, N. Mariana Islands, but not explicitly in 'locations')
    4: 98,
    6: 98,
    15: 98,
    32: 98,
    # HHS Region 10 (FIPS 99): AK, ID, OR, WA
    2: 99,
    16: 99,
    41: 99,
    53: 99,
}

# National FIPS code
NATIONAL_FIPS = 0


# Custom DataScaler to handle per-group scaling with global fallback
class DataScaler:

  def __init__(self):
    self.means = {}  # Key: (location, data_source), Value: mean
    self.stds = {}  # Key: (location, data_source), Value: std
    self._global_means = (
        {}
    )  # Key: data_source, Value: mean (global fallback per source)
    self._global_stds = (
        {}
    )  # Key: data_source, Value: std (global fallback per source)

  def fit_transform(
      self,
      df,
      group_cols,
      value_col,
      train_end_date,
  ):
    df_copy = df.copy()

    unique_data_sources = df_copy['data_source'].unique()

    # 1. Calculate global means/stds for each data_source for fallbacks
    for ds in unique_data_sources:
      ds_train_data = df_copy[
          (df_copy['data_source'] == ds)
          & (df_copy['target_end_date'] <= train_end_date)
      ][value_col]

      if not ds_train_data.empty:
        if ds_train_data.nunique() > 1:
          self._global_means[ds] = ds_train_data.mean()
          self._global_stds[ds] = ds_train_data.std()
        else:  # Only one unique value, std is 0. Treat as std=1 for scaling, but mean is the value itself.
          self._global_means[ds] = ds_train_data.iloc[0]
          self._global_stds[ds] = 1.0
      else:  # No data for this source in training period
        self._global_means[ds] = 0.0
        self._global_stds[ds] = 1.0

    # 2. Fit individual scalers or use global fallback for each (location, data_source) group
    transform_params = []
    for (location, data_source), group in df_copy.groupby(group_cols):
      train_group = group[group['target_end_date'] <= train_end_date]

      mean_val = None
      std_val = None

      if not train_group.empty:
        if train_group[value_col].nunique() > 1:
          mean_val = train_group[value_col].mean()
          std_val = train_group[value_col].std()
        else:  # Only one unique value in training, std is 0.
          mean_val = train_group[value_col].iloc[0]
          std_val = 1.0  # std is 0, treat as 1.0 for inverse transform to return the mean

      # If no specific scaler could be fitted, use the global default for this data_source.
      if mean_val is None:
        mean_val = self._global_means.get(data_source, 0.0)
        std_val = self._global_stds.get(data_source, 1.0)

      self.means[(location, data_source)] = mean_val
      self.stds[(location, data_source)] = std_val

      transform_params.append({
          group_cols[0]: location,
          group_cols[1]: data_source,
          'mean_val_merged': mean_val,
          'std_val_merged': std_val,
      })

    # 3. Create a DataFrame for transform parameters and merge
    transform_params_df = pd.DataFrame(transform_params)
    df_copy = pd.merge(df_copy, transform_params_df, on=group_cols, how='left')

    # Fill any missing merged params with global fallbacks (e.g., if a group appeared only in test)
    df_copy['mean_val_merged'].fillna(
        df_copy['data_source'].map(self._global_means), inplace=True
    )
    df_copy['std_val_merged'].fillna(
        df_copy['data_source'].map(self._global_stds), inplace=True
    )

    # Defensive fill for any remaining NaNs (should not happen if fallbacks are comprehensive)
    df_copy['mean_val_merged'].fillna(0.0, inplace=True)
    df_copy['std_val_merged'].fillna(1.0, inplace=True)

    # 4. Apply scaling in a vectorized manner
    # Replace 0 std with 1.0 for division, then handle the constant case
    effective_std = df_copy['std_val_merged'].replace(0, 1.0)
    df_copy['value_scaled'] = (
        df_copy[value_col] - df_copy['mean_val_merged']
    ) / effective_std

    df_copy.drop(
        columns=['mean_val_merged', 'std_val_merged'], inplace=True
    )  # Clean up temporary columns

    return df_copy

  def inverse_transform_df(
      self,
      df_to_unscale,
      value_col,
      location_col,
      data_source,
  ):
    """Inverse transforms a DataFrame of scaled values, given corresponding locations and data_source,

    using vectorized pandas operations.

    Args:
        df_to_unscale: DataFrame containing the scaled values (`value_col`) and
          location identifiers (`location_col`).
        value_col: The name of the column in `df_to_unscale` that contains the
          scaled values.
        location_col: The name of the column in `df_to_unscale` that contains
          the location FIPS codes.
        data_source: The data source string (e.g., 'NHSN') for which the scaler
          was fitted.

    Returns:
        np.ndarray: An array of inverse-transformed values.
    """
    if df_to_unscale.empty:
      return np.array([])

    temp_df = df_to_unscale[[value_col, location_col]].copy()
    temp_df['data_source_lookup'] = (
        data_source  # Use a distinct column name for the lookup data_source
    )

    # Create a DataFrame for means and stds for efficient merging
    # Ensure that std_val defaults to 1.0 if not found, to prevent division by zero in case of edge errors
    means_stds_data = []
    for (loc_key, ds_key), mean_val in self.means.items():
      std_val = self.stds.get(
          (loc_key, ds_key), 1.0
      )  # Default std to 1.0 if not found
      means_stds_data.append({
          'location': loc_key,
          'data_source_lookup': ds_key,
          'mean_val': mean_val,
          'std_val': std_val,
      })
    means_stds_df = pd.DataFrame(means_stds_data)

    # Merge means and stds into the temporary DataFrame
    temp_df = pd.merge(
        temp_df,
        means_stds_df,
        on=['location', 'data_source_lookup'],
        how='left',
    )

    # Fill any missing means/stds with global/default fallbacks (should be handled by fit_transform, but defensive)
    # Global means/stds need to be pulled for the specific 'data_source' argument passed to inverse_transform_df
    temp_df['mean_val'].fillna(
        self._global_means.get(data_source, 0.0), inplace=True
    )
    temp_df['std_val'].fillna(
        self._global_stds.get(data_source, 1.0), inplace=True
    )

    # Guard against zero standard deviation for inverse transform (replace with 1.0 if it somehow becomes 0)
    temp_df['std_val'] = temp_df['std_val'].replace(0, 1.0)

    # Perform vectorized inverse transformation
    unscaled_values = (temp_df[value_col] * temp_df['std_val']) + temp_df[
        'mean_val'
    ]

    return unscaled_values.values


# Helper functions
def _get_augmented_locations(locations_df):
  augmented_locations = locations_df.copy()
  augmented_locations['region_type'] = 'State'

  # National (FIPS 0)
  national_pop = locations_df['population'].sum()
  national_entry = pd.DataFrame([{
      'location': NATIONAL_FIPS,
      'abbreviation': 'US',
      'location_name': 'National',
      'population': national_pop,
      'region_type': 'National',
  }])
  augmented_locations = pd.concat(
      [augmented_locations, national_entry], ignore_index=True
  )

  # HHS Regions (dummy FIPS 90-99 for Region 1-10)
  hhs_fips_start = 90
  hhs_entries = []

  hhs_to_state_map = {}
  for state_fips, hhs_fips in STATE_TO_HHS_REGION_MAP.items():
    if hhs_fips not in hhs_to_state_map:
      hhs_to_state_map[hhs_fips] = []
    hhs_to_state_map[hhs_fips].append(state_fips)

  for hhs_fips_code in sorted(hhs_to_state_map.keys()):
    states_in_region = hhs_to_state_map[hhs_fips_code]
    # Ensure we only sum populations for locations present in the initial locations_df
    region_population = locations_df[
        locations_df['location'].isin(states_in_region)
    ]['population'].sum()

    region_num = (
        hhs_fips_code - hhs_fips_start + 1
    )  # Convert FIPS 90 -> Region 1
    hhs_entries.append({
        'location': hhs_fips_code,
        'abbreviation': f'HHS{region_num}',
        'location_name': f'Region {region_num}',
        'population': region_population,
        'region_type': 'HHS Region',
    })
  augmented_locations = pd.concat(
      [augmented_locations, pd.DataFrame(hhs_entries)], ignore_index=True
  )
  return augmented_locations


def _process_ilinet_data(
    ilinet_df,
    source_name,
    region_type,
    augmented_locations_df,
):
  df = ilinet_df.copy()
  value_col = (
      'unweighted_ili' if 'unweighted_ili' in df.columns else 'weighted_ili'
  )

  # Select columns that are consistently available
  selected_cols = ['week_start', 'region', value_col, 'season']
  selected_cols_exist = [col for col in selected_cols if col in df.columns]
  df = df[selected_cols_exist].copy()

  # Rename original 'region' column to avoid conflict with 'location_name' from augmented_locations_df
  df.rename(
      columns={
          'region': 'location_name_ilinet',
          value_col: 'value',
          'week_start': 'target_end_date',
      },
      inplace=True,
  )
  df['data_source'] = source_name
  df['signal_type'] = 'ILI'
  df['horizon'] = 0  # Observed data
  df['target_end_date'] = pd.to_datetime(df['target_end_date'])
  df['region_type'] = region_type

  # Map ILINet's 'region' name to a FIPS code and merge location details
  if region_type == 'National':
    df['location'] = NATIONAL_FIPS  # Assign National FIPS
    # Merge population and canonical location_name from augmented_locations_df
    df = pd.merge(
        df,
        augmented_locations_df[
            augmented_locations_df['location'] == NATIONAL_FIPS
        ][['location', 'location_name', 'population']],
        on='location',
        how='left',
    )
    df['location_name'] = df['location_name'].fillna(
        'National'
    )  # Ensure it's 'National' if merge failed for any reason
  elif region_type == 'HHS Region':
    # Create a mapping from ILINet's region name (e.g., 'Region 1') to HHS FIPS code
    hhs_region_name_to_fips = {
        entry['location_name']: entry['location']
        for _, entry in augmented_locations_df[
            augmented_locations_df['region_type'] == 'HHS Region'
        ].iterrows()
    }
    df['location'] = df['location_name_ilinet'].map(hhs_region_name_to_fips)
    # Now merge population and the consistent location_name from augmented_locations_df
    df = pd.merge(
        df,
        augmented_locations_df[
            augmented_locations_df['region_type'] == 'HHS Region'
        ][['location', 'location_name', 'population']],
        on='location',
        how='left',
    )
    df['location_name'] = df['location_name'].fillna(
        df['location_name_ilinet']
    )  # Fallback if canonical name is missing
  elif region_type == 'State':
    # For states, ILINet's region name is the state name.
    location_mapping = augmented_locations_df[
        augmented_locations_df['region_type'] == 'State'
    ][['location_name', 'location', 'population']].drop_duplicates()
    df = pd.merge(
        df,
        location_mapping,
        left_on='location_name_ilinet',
        right_on='location_name',
        how='left',
        suffixes=('_ilinet', ''),
    )
    # Ensure the 'location_name' from augmented_locations_df is used
    df['location_name'] = df['location_name'].fillna(
        df['location_name_ilinet']
    )  # Fallback if canonical name is missing

  # Drop the temporary ilinet_location_name and ensure location is Int64 (to allow NaNs)
  df.drop(columns=['location_name_ilinet'], errors='ignore', inplace=True)
  df['location'] = df['location'].astype('Int64')  # Use Int64 to allow for NaNs

  if df['location'].isnull().any():
    missing_regions_names = df.loc[
        df['location'].isnull(), 'location_name'
    ].unique()
    warnings.warn(
        f'Missing location FIPS codes for {source_name}:'
        f' {missing_regions_names}. Dropping unmapped rows.',
        UserWarning,
    )
    df.dropna(
        subset=['location'], inplace=True
    )  # Drop rows that cannot be mapped to a location
    df['location'] = df['location'].astype(
        int
    )  # Convert to int after dropping NaNs

  return df[[
      'target_end_date',
      'location',
      'location_name',
      'population',
      'value',
      'data_source',
      'signal_type',
      'horizon',
      'season',
      'region_type',
  ]]


def _calculate_season_week(df):
  df['week_of_year'] = df['target_end_date'].dt.isocalendar().week.astype(int)
  # The 'season_year' is the calendar year in which the flu season *starts*.
  # E.g., for season '2020/21', the season_year is 2020 (starting in week 40 of 2020).
  df['season_year'] = df['target_end_date'].dt.year.where(
      df['week_of_year'] >= FLU_SEASON_START_WEEK,
      df['target_end_date'].dt.year - 1,
  )
  # Calculate season_week: week 1 of the season corresponds to FLU_SEASON_START_WEEK
  # Example: week 40 is season_week 1, week 41 is season_week 2, week 1 (next year) is season_week 14 (52-40+1 + 1)
  df['season_week'] = (df['week_of_year'] - FLU_SEASON_START_WEEK + 52) % 52 + 1
  return df


def _get_christmas_diff(df):
  # Calculates the difference in weeks from the closest Christmas (week 52)
  # Corrected vectorized logic for symmetric distance to week 52 (Christmas)
  df_copy = df.copy()
  week_of_year = df_copy['week_of_year']

  # Formula for symmetric difference to a target week (e.g., week 52)
  # (week_value - target_week + half_year_weeks) % num_weeks_in_year - half_year_weeks
  # For a 52-week year, target_week=52, num_weeks_in_year=52, half_year_weeks=26
  df_copy['christmas_diff'] = (week_of_year - 52 + 26) % 52 - 26

  return df_copy


def _add_lag_trend_curvature(
    df_group, value_col, lag_window
):
  df_sorted = df_group.sort_values(by=['target_end_date']).copy()

  # Generate lags
  for i in range(1, lag_window + 1):
    df_sorted[f'lag_{i}'] = df_sorted[value_col].shift(i)

  # Calculate trend and curvature from lagged values
  df_sorted[f'trend'] = df_sorted[value_col] - df_sorted[f'lag_1']
  df_sorted[f'curvature'] = (df_sorted[value_col] - df_sorted[f'lag_1']) - (
      df_sorted[f'lag_1'] - df_sorted[f'lag_2']
  )
  return df_sorted


# Refactored function for dynamic feature computation per source
def _compute_dynamic_features_per_source(
    full_historical_df,
):
  df = full_historical_df.copy()

  processed_groups = []
  # Identify dynamic feature columns before processing for easier prefixing later
  base_dynamic_cols_to_compute = [
      f'lag_{i}' for i in range(1, LAG_WINDOW + 1)
  ] + ['trend', 'curvature', 'current_value_scaled']

  for (loc, ds), group in df.groupby(['location', 'data_source']):
    if not group.empty:
      group_with_features = _add_lag_trend_curvature(
          group, 'value_scaled', LAG_WINDOW
      )

      # Also include the 'value_scaled' itself as the 'current' level
      group_with_features['current_value_scaled'] = group_with_features[
          'value_scaled'
      ]

      # Prefix dynamic feature names with data_source
      # Use data_source directly for prefixing for clarity (e.g., 'NHSN_lag_1', 'ILINet_State_lag_1')
      sanitized_ds = ds.replace(
          'ILINet_', ''
      )  # Keep suffix for ILINet types, remove common prefix

      # Select relevant dynamic columns, using base_dynamic_cols_to_compute as the source for column names
      dynamic_cols = [
          col
          for col in group_with_features.columns
          if col in base_dynamic_cols_to_compute
      ]
      rename_map = {col: f'{sanitized_ds}_{col}' for col in dynamic_cols}
      group_with_features.rename(columns=rename_map, inplace=True)
      processed_groups.append(group_with_features)

  if not processed_groups:
    return pd.DataFrame()  # Return empty if no processed signals

  return pd.concat(processed_groups, ignore_index=True)


# New function for assembling model features - OPTIMIZED
def _assemble_model_features(
    dynamic_features_per_source_df,
    augmented_locations_df,
    features_obs_dates,
    location_ilinet_nhs_ratio_map,  # NEW: Ratio map for imputation
):

  # 1. Preparation of Static and Temporal Features
  # Filter augmented_locations_df to only 'State' region_type for the base scaffold
  # Only select location, population, and location_cat, as region_type_cat would be constant here
  state_location_info = augmented_locations_df[
      augmented_locations_df['region_type'] == 'State'
  ][
      ['location', 'population', 'location_cat', 'region_type_cat']
  ].copy()  # Added region_type_cat

  # Pre-calculate temporal features for all observation dates
  temporal_features_df = pd.DataFrame(
      {'target_end_date': features_obs_dates}
  ).drop_duplicates()
  temporal_features_df = _calculate_season_week(temporal_features_df)
  temporal_features_df = _get_christmas_diff(temporal_features_df)
  temporal_features_df = temporal_features_df.drop(
      columns=['week_of_year'], errors='ignore'
  )  # week_of_year is used to derive, but season_week is more stable and direct. Remove to avoid redundancy.

  # Create base scaffold for all (observation_date, location) combinations
  base_scaffold = pd.MultiIndex.from_product(
      [features_obs_dates, state_location_info['location'].unique()],
      names=['target_end_date', 'location'],
  ).to_frame(index=False)

  # Merge static location info (including already encoded categorical features)
  all_features_df = pd.merge(
      base_scaffold, state_location_info, on='location', how='left'
  )  # Renamed to all_features_df directly

  # Merge temporal features
  all_features_df = pd.merge(
      all_features_df, temporal_features_df, on='target_end_date', how='left'
  )

  # 2. Preparation of Dynamic Features from dynamic_features_per_source_df
  # Identify all dynamic columns created by _compute_dynamic_features_per_source (e.g. NHSN_lag_1, State_trend)
  all_source_prefixes_dynamic = [
      'NHSN',
      'State',
      'HHS',
      'National',
  ]  # Explicitly define prefixes
  prefixed_dynamic_cols = [
      col
      for col in dynamic_features_per_source_df.columns
      if any(p in col for p in all_source_prefixes_dynamic)
  ]

  # Split dynamic_features_per_source_df by data_source
  # Use generic ILINet_State, ILINet_HHS, ILINet_National directly for prefix checking here
  nhs_dynamic = dynamic_features_per_source_df[
      dynamic_features_per_source_df['data_source'] == 'NHSN'
  ][
      ['target_end_date', 'location']
      + [c for c in prefixed_dynamic_cols if c.startswith('NHSN_')]
  ]
  state_ilinet_dynamic = dynamic_features_per_source_df[
      dynamic_features_per_source_df['data_source'] == 'ILINet_State'
  ][
      ['target_end_date', 'location']
      + [c for c in prefixed_dynamic_cols if c.startswith('State_')]
  ]
  hhs_ilinet_dynamic = dynamic_features_per_source_df[
      dynamic_features_per_source_df['data_source'] == 'ILINet_HHS'
  ][
      ['target_end_date', 'location']
      + [c for c in prefixed_dynamic_cols if c.startswith('HHS_')]
  ].rename(
      columns={'location': 'hhs_location'}
  )
  national_ilinet_dynamic = dynamic_features_per_source_df[
      dynamic_features_per_source_df['data_source'] == 'ILINet_National'
  ][
      ['target_end_date']
      + [c for c in prefixed_dynamic_cols if c.startswith('National_')]
  ]  # National features don't have a 'location' column specific to states

  # 3. Merging Dynamic Features onto all_features_df
  # Merge NHSN dynamic features FIRST
  all_features_df = pd.merge(
      all_features_df,
      nhs_dynamic,
      on=['target_end_date', 'location'],
      how='left',
  )

  # The 'NHSN_current_value_scaled' is a special feature as it's the baseline for diff prediction
  original_nhs_current_col_name = 'NHSN_current_value_scaled'
  renamed_nhs_current_col_name = 'current_nhs_scaled_value'

  if original_nhs_current_col_name in all_features_df.columns:
    all_features_df.rename(
        columns={original_nhs_current_col_name: renamed_nhs_current_col_name},
        inplace=True,
    )
  else:
    all_features_df[renamed_nhs_current_col_name] = (
        np.nan
    )  # Create as NaN if missing

  # Merge other ILINet dynamic features (these will include their own 'current_value_scaled' e.g., 'State_current_value_scaled')
  all_features_df = pd.merge(
      all_features_df,
      state_ilinet_dynamic,
      on=['target_end_date', 'location'],
      how='left',
  )

  state_to_hhs_df = pd.DataFrame(
      list(STATE_TO_HHS_REGION_MAP.items()),
      columns=['location', 'hhs_location'],
  ).drop_duplicates()
  all_features_df = pd.merge(
      all_features_df, state_to_hhs_df, on='location', how='left'
  )
  all_features_df = pd.merge(
      all_features_df,
      hhs_ilinet_dynamic,
      on=['target_end_date', 'hhs_location'],
      how='left',
  )
  all_features_df.drop(columns=['hhs_location'], errors='ignore', inplace=True)

  all_features_df = pd.merge(
      all_features_df, national_ilinet_dynamic, on='target_end_date', how='left'
  )

  # NEW: Imputation for current_nhs_scaled_value and is_current_nhs_synthetic feature
  all_features_df['is_current_nhs_synthetic'] = 0  # Initialize as not synthetic

  # Identify potential synthetic candidates
  mask_current_nhs_missing = all_features_df[
      renamed_nhs_current_col_name
  ].isna()
  mask_ilinet_available = (
      all_features_df['target_end_date'] <= ILINET_CUTOFF_DATE
  )
  mask_ilinet_state_current_available = all_features_df[
      'State_current_value_scaled'
  ].notna()

  imputation_mask = (
      mask_current_nhs_missing
      & mask_ilinet_available
      & mask_ilinet_state_current_available
  )

  if imputation_mask.any():
    locations_for_imputation = all_features_df.loc[imputation_mask, 'location']

    # Get ratios for each location in the imputation_mask, with global fallback
    ratios_for_imputation = locations_for_imputation.map(
        location_ilinet_nhs_ratio_map
    ).fillna(
        location_ilinet_nhs_ratio_map.get(
            'global_fallback_median', 1.0
        )  # Use 1.0 if even global is missing
    )

    imputed_values = (
        all_features_df.loc[imputation_mask, 'State_current_value_scaled']
        * ratios_for_imputation
    )
    all_features_df.loc[imputation_mask, renamed_nhs_current_col_name] = (
        imputed_values
    )
    all_features_df.loc[imputation_mask, 'is_current_nhs_synthetic'] = 1

  # Fill any remaining NaNs for current_nhs_scaled_value with 0.0 (e.g., beyond ILINet_Cutoff or if ILINet_State also missing)
  all_features_df[renamed_nhs_current_col_name] = all_features_df[
      renamed_nhs_current_col_name
  ].fillna(0.0)

  # Add ILINet data availability indicators (original, but useful)
  all_features_df['ilinet_state_available'] = (
      all_features_df['target_end_date'] <= ILINET_CUTOFF_DATE
  ).astype(int)
  all_features_df['ilinet_hhs_available'] = (
      all_features_df['target_end_date'] <= ILINET_CUTOFF_DATE
  ).astype(int)
  all_features_df['ilinet_national_available'] = (
      all_features_df['target_end_date'] <= ILINET_CUTOFF_DATE
  ).astype(int)

  # Fill any NaNs from missing dynamic features with 0.0. This is important for LGBM.
  expected_dynamic_feature_suffixes_for_fill = [
      f'lag_{i}' for i in range(1, LAG_WINDOW + 1)
  ] + ['trend', 'curvature', 'current_value_scaled']

  for (
      prefix
  ) in all_source_prefixes_dynamic:  # Use the explicit list of dynamic prefixes
    for suffix in expected_dynamic_feature_suffixes_for_fill:
      col_name = f'{prefix}_{suffix}'

      # Skip the renamed NHSN_current_value_scaled as it's handled separately
      if col_name == original_nhs_current_col_name:
        continue

      if col_name not in all_features_df.columns:
        all_features_df[col_name] = (
            0.0  # Create the column if it's entirely missing after all merges
        )
      all_features_df[col_name] = all_features_df[col_name].fillna(
          0.0
      )  # Fill NaNs that resulted from merges for existing columns

  return all_features_df


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # IMPLEMENTATION PLAN.
  # ## Core Principles Checklist
  # 1. Probabilistic Forecasts: The code will train 23 separate LightGBM Quantile Regression models, one for each quantile, directly predicting the required probabilistic intervals. Bagging will be applied by training multiple models per quantile on bootstrapped seasons and averaging their predictions.
  # 2. Weighted Interval Score (WIS): LightGBM models will be configured with `objective='quantile'` and `alpha` set to the specific quantile level, directly optimizing for quantile loss which is a component of WIS.
  # 3. Monotonically Increasing Quantiles: After generating predictions for all quantiles, the predicted values for each forecast row will be sorted along the quantile axis to enforce the monotonicity constraint.

  # ## Step-by-Step Logic
  # 1.  **Initialize & Augment Data:**
  #     a.  Determine `train_end_date` from `test_x` by taking the minimum `reference_date` and subtracting one week.
  #     b.  Augment `locations` DataFrame (`augmented_locations_df`) to include National (FIPS 0) and HHS Region entries (dummy FIPS codes 90-99), accurately calculating their populations and assigning `region_type`.
  #     c.  Initialize and fit `LabelEncoder`s for `location_cat` and `region_type_cat` using all unique values from `augmented_locations_df` and apply them to `augmented_locations_df`.
  # 2.  **Process and Transform Historical Signals (Pre-scaling):**
  #     a.  Process `train_x`/`train_y` (NHSN data): Combine into a single DataFrame, add `data_source='NHSN'`, `signal_type='Hospitalizations'`, `horizon=0`. Convert `target_end_date` to datetime and merge with `augmented_locations_df` to get `location_name`, `population`, and `region_type`.
  #     b.  Process `ilinet_state`, `ilinet_hhs`, and `ilinet` (National ILINet): Use a helper function `_process_ilinet_data` to rename columns, assign appropriate `data_source` and `signal_type`, map `region` names to FIPS `location` codes (including HHS FIPS for `ilinet_hhs` and National FIPS 0 for `ilinet`), and ensure `target_end_date` is datetime.
  #     c.  Concatenate processed NHSN and ILINet DataFrames into `full_historical_df_raw`. Sort by `location`, `data_source`, and `target_end_date`.
  #     d.  Apply the *already fitted* `location_encoder` and `region_type_encoder` to `full_historical_df_raw`. Fit and apply `data_source_encoder`.
  #     e.  Calculate global temporal features (`season_week`, `christmas_diff`, `season_year`) on `full_historical_df_raw`.
  #     f.  **Apply first stage transformation:** Convert `value` (NHSN) to `rate` per 100,000 population. Apply a fourth-root transformation to all `value`s (NHSN rates, ILINet values), ensuring values are clipped at 0 before transformation. Store as `value_transformed`.
  # 3.  **Centering/Scaling and Learn ILINet-to-NHSN Transformation:**
  #     a.  Instantiate `DataScaler`. Fit/transform the `value_transformed` column to `value_scaled` for each `(location, data_source)` group, using data up to `train_end_date`. Store `mean` and `std` for inverse transformation. `full_historical_df_raw` now has `value_scaled`.
  #     b.  **Calculate ILINet-State to NHSN `value_scaled` ratio mapping for synthetic target creation:** Create an `overlap_data` DataFrame by merging `full_historical_df_raw` (filtered for 'NHSN' and 'ILINet_State') on `location` and `target_end_date` for dates within the overlap period (`NHSN_DATA_START_DATE` to `ILINET_CUTOFF_DATE`). For each `location`, compute the median ratio of `NHSN_value_scaled` to `ILINet_State_value_scaled` from `overlap_data` and store in `location_ilinet_nhs_ratio_map`. Add a small epsilon to the denominator *only* for stability. Include a global median ratio as a fallback.
  # 4.  **Create Synthetic NHSN Target Data:**
  #     a.  Filter `full_historical_df_raw` for rows where `data_source == 'ILINet_State'`, `target_end_date < NHSN_DATA_START_DATE`, and `target_end_date <= ILINET_CUTOFF_DATE`.
  #     b.  For these filtered ILINet_State rows, create `synthetic_target_scaled_value` by multiplying their `value_scaled` by the corresponding `location`-specific ratio from `location_ilinet_nhs_ratio_map` (using global fallback if location-specific is missing).
  #     c.  Construct `synthetic_target_df` from these, containing `target_end_date`, `location`, `target_scaled_value`. This DataFrame represents synthetic target history.
  # 5.  **Feature Engineering - Dynamic Features Per Source and Assembled Model Features:**
  #     a.  Call `_compute_dynamic_features_per_source` on `full_historical_df_raw` (which now contains all scaled signal data) to compute all temporal, lagged, trend, and curvature features for all data sources and locations. This returns `dynamic_features_per_source_df`.
  #     b.  Define `all_available_features_obs_dates` as unique `target_end_date`s from 'NHSN' or 'ILINet_State' data in `dynamic_features_per_source_df` up to `train_end_date`.
  #     c.  Call `_assemble_model_features` with `dynamic_features_per_source_df`, `augmented_locations_df` (which has `_cat` columns), `all_available_features_obs_dates`, and the `location_ilinet_nhs_ratio_map` to create `all_features_df` for training. This function will *impute* missing `current_nhs_scaled_value` using `State_current_value_scaled` and the `location_ilinet_nhs_ratio_map` when actual NHSN data is not available and ILINet data is, and adds `is_current_nhs_synthetic`.
  #     d.  Define `lgbm_features` to be a comprehensive, static list of all *expected* feature columns, including `location_cat`, `region_type_cat`, `is_current_nhs_synthetic`, availability indicators, numerical temporal features, and all prefixed dynamic features, ensuring consistency across training and test sets.
  # 6.  **Construct Model Training Data (`X_train_model`, `y_train_model`):**
  #     a.  Combine `target_df_nhs` (actual targets from `full_historical_df_raw` where `data_source=='NHSN'`) and `synthetic_target_df` (from step 4) into `combined_target_df`.
  #     b.  Create a master DataFrame by merging `all_features_df` (filtered for `features_obs_date` up to `train_end_date`) with `combined_target_df` across `location` and `target_observation_date` (which is `features_obs_date + pd.Timedelta(weeks=1) + pd.Timedelta(weeks=horizon)`)
  #     c.  Filter out training records corresponding to `season_year`s in `PANDEMIC_SEASONS_YEARS` or falling within summer off-seasons.
  #     d.  Calculate the target `y_diff_scaled` as `target_scaled_value - current_nhs_scaled_value`.
  #     e.  Drop rows with missing `y_diff_scaled` values.
  #     f.  Ensure `X_train_model` explicitly contains all columns from the static `lgbm_features` list, filling with 0.0 if a column is missing, and selecting these features. `y_train_model` will be `y_diff_scaled`, and `season_year_for_bagging` will be extracted.
  # 7.  **Construct Model Test Data (`X_test_model`):**
  #     a.  Determine `features_obs_date_for_test` (one week prior to `test_x`'s `reference_date`).
  #     b.  Call the `_assemble_model_features` with `dynamic_features_per_source_df`, `augmented_locations_df`, `[features_obs_date_for_test]`, and `location_ilinet_nhs_ratio_map` to create `test_date_features_df`.
  #     c.  Merge `test_x` with `test_date_features_df` (on `location`), and then add `horizon` column from `test_x`.
  #     d.  Extract `current_nhs_scaled_value` for inverse transformation later.
  #     e.  Ensure `X_test_model` explicitly contains all columns from the static `lgbm_features` list, filling with 0.0 if a column is missing after merges, and selecting these features to form `X_test_model`. Set `X_test_model`'s index to match `test_x.index`.
  # 8.  **Model Training (Bagging & Quantile LightGBM):**
  #     a.  Initialize an empty dictionary `trained_models`.
  #     b.  For each `q` in `QUANTILES` and for `N_BAGS` iterations:
  #         i.   Randomly sample `season_year`s with replacement from the available training seasons for bagging.
  #         ii.  Filter training data (`X_train_model`, `y_train_model`) based on the sampled seasons.
  #         iii. Train an `lgb.LGBMRegressor` with `objective='quantile'`, `alpha=q`, `verbose=-1`, and explicitly specify categorical features by their indices. Store the trained model.
  # 9.  **Prediction Generation:**
  #     a.  Initialize `test_y_hat_quantiles` DataFrame.
  #     b.  For each `q` in `QUANTILES`:
  #         i.   Generate predictions from all bagged models for `q` on `X_test_model` and average them.
  #         ii.  Add back `current_nhs_scaled_values_for_test` to obtain the predicted absolute scaled values.
  #         iii. Apply inverse transformations: use the `DataScaler` to unscale (using 'NHSN' parameters), then apply `power(4)` for the inverse fourth-root, and finally convert the rate back to admission counts by multiplying by population and dividing by 100,000.
  #         iv.  Cap predictions at 0, round to the nearest integer, and store.
  #     c.  Enforce monotonicity: For each row, sort the predicted quantile values.
  #     d.  Return the final `test_y_hat_quantiles` DataFrame.

  # 5d. Define all feature columns for LGBM
  # Removed 'season_year' from lgbm_categorical_features to treat it as numerical
  lgbm_categorical_features = [
      'location_cat',
      'region_type_cat',
      'ilinet_state_available',
      'ilinet_hhs_available',
      'ilinet_national_available',
      'is_current_nhs_synthetic',
  ]
  lgbm_numerical_features_base = [
      'population',
      'season_week',
      'christmas_diff',
      'season_year',
      'current_nhs_scaled_value',
  ]

  # Define all expected dynamic feature names statically
  all_source_prefixes_for_lgbm = [
      'NHSN',
      'State',
      'HHS',
      'National',
  ]  # Use full prefixes for LGBM features
  expected_dynamic_feature_suffixes = [
      f'lag_{i}' for i in range(1, LAG_WINDOW + 1)
  ] + ['trend', 'curvature', 'current_value_scaled']

  expected_dynamic_features = []
  for prefix in all_source_prefixes_for_lgbm:
    for suffix in expected_dynamic_feature_suffixes:
      # Skip the specific NHSN_current_value_scaled as it's renamed to 'current_nhs_scaled_value' and already in base numerical features.
      if prefix == 'NHSN' and suffix == 'current_value_scaled':
        continue
      expected_dynamic_features.append(f'{prefix}_{suffix}')

  # Combine all feature names
  lgbm_features = (
      lgbm_categorical_features
      + lgbm_numerical_features_base
      + expected_dynamic_features
      + ['horizon']
  )

  # 1a. Determine train_end_date
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])
  train_end_date = test_x['reference_date'].min() - pd.Timedelta(weeks=1)

  # 1b. Augment locations DataFrame
  augmented_locations_df = _get_augmented_locations(locations)

  # 1c. Global Categorical Encoding for augmented_locations_df
  all_locations = augmented_locations_df['location'].unique()
  all_region_types = augmented_locations_df['region_type'].unique()

  loc_encoder = LabelEncoder()
  region_type_encoder = LabelEncoder()

  loc_encoder.fit(all_locations)
  region_type_encoder.fit(all_region_types)

  augmented_locations_df['location_cat'] = loc_encoder.transform(
      augmented_locations_df['location']
  )
  augmented_locations_df['region_type_cat'] = region_type_encoder.transform(
      augmented_locations_df['region_type']
  )

  # 2. Process and Transform Historical Signals (Pre-scaling)
  # 2a. Process NHSN data (train_x, train_y)
  nhs_df = train_x.copy()
  nhs_df[TARGET_STR] = train_y
  nhs_df.rename(columns={TARGET_STR: 'value'}, inplace=True)
  nhs_df['data_source'] = 'NHSN'
  nhs_df['signal_type'] = 'Hospitalizations'
  nhs_df['horizon'] = 0  # This is observed data for horizon 0
  nhs_df['target_end_date'] = pd.to_datetime(nhs_df['target_end_date'])
  # Ensure location data is consistent and region_type is added
  nhs_df = pd.merge(
      nhs_df.drop(columns=['population', 'location_name'], errors='ignore'),
      augmented_locations_df[
          ['location', 'location_name', 'population', 'region_type']
      ],
      on='location',
      how='left',
  )
  nhs_df = nhs_df[[
      'target_end_date',
      'location',
      'location_name',
      'population',
      'value',
      'data_source',
      'signal_type',
      'horizon',
      'region_type',
  ]]

  # 2b. Process ILINet data
  ilinet_state_processed = _process_ilinet_data(
      ilinet_state, 'ILINet_State', 'State', augmented_locations_df
  )
  ilinet_hhs_processed = _process_ilinet_data(
      ilinet_hhs, 'ILINet_HHS', 'HHS Region', augmented_locations_df
  )
  ilinet_national_processed = _process_ilinet_data(
      ilinet, 'ILINet_National', 'National', augmented_locations_df
  )

  # 2c. Concatenate all raw historical data before further processing
  full_historical_df_raw = pd.concat(
      [
          nhs_df,
          ilinet_state_processed,
          ilinet_hhs_processed,
          ilinet_national_processed,
      ],
      ignore_index=True,
  )

  full_historical_df_raw['target_end_date'] = pd.to_datetime(
      full_historical_df_raw['target_end_date']
  )
  full_historical_df_raw = full_historical_df_raw.sort_values(
      by=['location', 'data_source', 'target_end_date']
  ).reset_index(drop=True)

  # Remove the original 'season' column which is now redundant after 'season_year' is derived
  full_historical_df_raw = full_historical_df_raw.drop(
      columns=['season'], errors='ignore'
  )

  # 2d. Apply encoders
  full_historical_df_raw['location_cat'] = loc_encoder.transform(
      full_historical_df_raw['location']
  )
  full_historical_df_raw['region_type_cat'] = region_type_encoder.transform(
      full_historical_df_raw['region_type']
  )

  all_data_sources = full_historical_df_raw['data_source'].unique()
  data_source_encoder = LabelEncoder()
  data_source_encoder.fit(all_data_sources)
  full_historical_df_raw['data_source_cat'] = data_source_encoder.transform(
      full_historical_df_raw['data_source']
  )

  # 2e. Calculate global temporal features
  full_historical_df_raw = _calculate_season_week(full_historical_df_raw)
  full_historical_df_raw = _get_christmas_diff(full_historical_df_raw)

  # 2f. Apply first stage transformation
  full_historical_df_raw['value_for_transform'] = full_historical_df_raw[
      'value'
  ].astype(float)
  # Convert NHSN counts to rates
  full_historical_df_raw.loc[
      full_historical_df_raw['data_source'] == 'NHSN', 'value_for_transform'
  ] = (
      full_historical_df_raw['value']
      / full_historical_df_raw['population']
      * 100_000
  ).fillna(
      0
  )

  # Apply fourth-root transformation. Cap at 0 to ensure non-negative for power transform.
  full_historical_df_raw['value_transformed'] = (
      full_historical_df_raw['value_for_transform'].clip(lower=0)
  ) ** (1 / 4)
  full_historical_df_raw['value_transformed'] = (
      full_historical_df_raw['value_transformed']
      .replace([np.inf, -np.inf], np.nan)
      .fillna(0)
  )

  # 3. Centering/Scaling and Learn ILINet-to-NHSN Transformation
  data_scaler = DataScaler()
  full_historical_df_raw = data_scaler.fit_transform(
      full_historical_df_raw,
      ['location', 'data_source'],
      'value_transformed',
      train_end_date,
  )

  # 3b. Calculate ILINet-State to NHSN `value_scaled` ratio mapping for synthetic target creation
  location_ilinet_nhs_ratio_map = {}
  epsilon = 1e-6  # Small value to avoid division by zero in denominator

  # Filter for overlap data for ratio calculation
  overlap_data = (
      full_historical_df_raw[
          (full_historical_df_raw['target_end_date'] >= NHSN_DATA_START_DATE)
          & (full_historical_df_raw['target_end_date'] <= ILINET_CUTOFF_DATE)
          & (
              full_historical_df_raw['data_source'].isin(
                  ['NHSN', 'ILINet_State']
              )
          )
      ]
      .pivot_table(
          index=['target_end_date', 'location'],
          columns='data_source',
          values='value_scaled',
      )
      .reset_index()
  )

  overlap_data.columns.name = None  # Remove columns name for easier access

  # Calculate median ratio per location
  if 'NHSN' in overlap_data.columns and 'ILINet_State' in overlap_data.columns:
    for loc in overlap_data['location'].unique():
      loc_overlap = overlap_data[overlap_data['location'] == loc].copy()
      # Only consider rows where both signals have non-zero scaled values for ratio
      # Corrected: epsilon only in denominator
      valid_ratios = loc_overlap['NHSN'] / (
          loc_overlap['ILINet_State'] + epsilon
      )
      # Filter out infinite/NaN ratios that might arise if ILINet_State is zero even with epsilon
      valid_ratios = valid_ratios[np.isfinite(valid_ratios)]
      if not valid_ratios.empty:
        location_ilinet_nhs_ratio_map[loc] = valid_ratios.median()

    # Global fallback median ratio
    all_valid_ratios_global = overlap_data['NHSN'] / (
        overlap_data['ILINet_State'] + epsilon
    )
    all_valid_ratios_global = all_valid_ratios_global[
        np.isfinite(all_valid_ratios_global)
    ]
    if not all_valid_ratios_global.empty:
      location_ilinet_nhs_ratio_map['global_fallback_median'] = (
          all_valid_ratios_global.median()
      )
    else:
      location_ilinet_nhs_ratio_map['global_fallback_median'] = (
          1.0  # Default to 1.0 if no overlap data at all
      )

  else:  # If one of the required columns is missing (e.g., no NHSN data in overlap)
    location_ilinet_nhs_ratio_map['global_fallback_median'] = 1.0

  # 4. Create Synthetic NHSN Target Data
  synthetic_target_df_list = []
  ilinet_state_pre_nhs_data = full_historical_df_raw[
      (full_historical_df_raw['data_source'] == 'ILINet_State')
      & (full_historical_df_raw['target_end_date'] < NHSN_DATA_START_DATE)
      & (
          full_historical_df_raw['target_end_date'] <= ILINET_CUTOFF_DATE
      )  # only if ILINet data is available
  ].copy()

  if not ilinet_state_pre_nhs_data.empty:
    # Map ratios to these historical ILINet_State records
    ratios_for_synthetic = (
        ilinet_state_pre_nhs_data['location']
        .map(location_ilinet_nhs_ratio_map)
        .fillna(
            location_ilinet_nhs_ratio_map.get('global_fallback_median', 1.0)
        )
    )
    ilinet_state_pre_nhs_data['synthetic_target_scaled_value'] = (
        ilinet_state_pre_nhs_data['value_scaled'] * ratios_for_synthetic
    )

    # Create synthetic target DataFrame
    synthetic_target_df = ilinet_state_pre_nhs_data[
        ['target_end_date', 'location', 'synthetic_target_scaled_value']
    ].copy()
    synthetic_target_df.rename(
        columns={'synthetic_target_scaled_value': 'target_scaled_value'},
        inplace=True,
    )
    synthetic_target_df_list.append(synthetic_target_df)

  # 5. Feature Engineering - Dynamic Features Per Source and Assembled Model Features
  # 5a. Compute dynamic features per source
  dynamic_features_per_source_df = _compute_dynamic_features_per_source(
      full_historical_df_raw
  )

  if dynamic_features_per_source_df.empty:
    warnings.warn(
        'No dynamic historical features could be created. Returning empty'
        ' predictions.',
        UserWarning,
    )
    return (
        pd.DataFrame(
            index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
        )
        .fillna(0)
        .astype(int)
    )

  # 5b. Define all available observation dates for features (up to train_end_date)
  all_available_features_obs_dates = dynamic_features_per_source_df[(
      dynamic_features_per_source_df['data_source'].isin(
          ['NHSN', 'ILINet_State']
      )
  )][
      'target_end_date'
  ].unique()  # Use NHSN and ILINet_State as proxies for available observation dates for features
  all_available_features_obs_dates = np.sort(all_available_features_obs_dates)

  # 5c. Assemble all features for the training data (pass ratio map)
  all_features_df = _assemble_model_features(
      dynamic_features_per_source_df,
      augmented_locations_df,
      all_available_features_obs_dates,
      location_ilinet_nhs_ratio_map,  # NEW parameter
  )

  if all_features_df.empty:
    warnings.warn(
        'No comprehensive historical features could be assembled. Returning'
        ' empty predictions.',
        UserWarning,
    )
    return (
        pd.DataFrame(
            index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
        )
        .fillna(0)
        .astype(int)
    )

  # The `lgbm_features` list is already defined statically at the top of the function.

  # 6. Construct Model Training Data (X_train_model, y_train_model)

  # 6a. Combine actual and synthetic targets for y_train_model
  target_df_nhs = full_historical_df_raw[
      full_historical_df_raw['data_source'] == 'NHSN'
  ][['target_end_date', 'location', 'value_scaled']]
  target_df_nhs.rename(
      columns={'value_scaled': 'target_scaled_value'}, inplace=True
  )

  combined_target_df = pd.concat(
      [target_df_nhs] + synthetic_target_df_list, ignore_index=True
  )
  combined_target_df = combined_target_df.drop_duplicates(
      subset=['target_end_date', 'location'], keep='first'
  )  # In case of overlap, keep actual NHSN

  all_possible_train_features = []

  # Filter all_features_df to dates relevant for training (<= train_end_date for features itself)
  relevant_features_df = all_features_df[
      all_features_df['target_end_date'] <= train_end_date
  ].copy()

  for horizon_val in horizons:
    # Calculate the future target_end_date for this horizon
    # features_obs_date + 1 week (to align with reference_date for horizon 0) + horizon_val weeks
    relevant_features_df['target_observation_date'] = (
        relevant_features_df['target_end_date']
        + pd.Timedelta(weeks=1)
        + pd.Timedelta(weeks=horizon_val)
    )

    # Merge current features with future targets for this horizon
    merged_for_horizon = pd.merge(
        relevant_features_df,
        combined_target_df,  # Use combined_target_df here
        left_on=['target_observation_date', 'location'],
        right_on=['target_end_date', 'location'],
        how='left',
        suffixes=('', '_target'),
    )

    merged_for_horizon['horizon'] = horizon_val
    all_possible_train_features.append(merged_for_horizon)

  X_train_master_df = pd.concat(all_possible_train_features, ignore_index=True)

  # Filter out records where target_scaled_value is NaN (no future target available)
  X_train_master_df.dropna(subset=['target_scaled_value'], inplace=True)

  # Calculate target_diff_scaled
  X_train_master_df['target_diff_scaled'] = (
      X_train_master_df['target_scaled_value']
      - X_train_master_df['current_nhs_scaled_value']
  )

  # Filter out pandemic seasons and summer off-seasons
  FLU_SEASON_END_SEASON_WEEK_MAPPED = (
      FLU_SEASON_END_WEEK - FLU_SEASON_START_WEEK + 52
  ) % 52 + 1

  X_train_filtered_df = X_train_master_df[
      (~X_train_master_df['season_year'].isin(PANDEMIC_SEASONS_YEARS))
      & (X_train_master_df['season_week'] >= 1)
      & (X_train_master_df['season_week'] <= FLU_SEASON_END_SEASON_WEEK_MAPPED)
  ].copy()

  if X_train_filtered_df.empty:
    warnings.warn(
        'No valid training data after filtering for seasons/off-seasons.'
        ' Returning empty predictions.',
        UserWarning,
    )
    return (
        pd.DataFrame(
            index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
        )
        .fillna(0)
        .astype(int)
    )

  y_train_model = X_train_filtered_df['target_diff_scaled']
  season_years_for_bagging = X_train_filtered_df['season_year'].unique()

  # Ensure all LGBM features exist in X_train_filtered_df, fill with 0 if missing
  for col in lgbm_features:
    if col not in X_train_filtered_df.columns:
      X_train_filtered_df[col] = 0.0

  X_train_model = X_train_filtered_df[
      lgbm_features
  ]  # Select only LGBM features

  # 7. Construct Model Test Data (X_test_model)
  test_reference_date = test_x['reference_date'].iloc[0]
  features_obs_date_for_test = test_reference_date - pd.Timedelta(weeks=1)

  # 7b. Call _assemble_model_features for the single test observation date (pass ratio map)
  test_date_features_df = _assemble_model_features(
      dynamic_features_per_source_df,
      augmented_locations_df,
      [features_obs_date_for_test],
      location_ilinet_nhs_ratio_map,  # NEW parameter
  )

  if test_date_features_df.empty:
    warnings.warn(
        f'No features assembled for test date {features_obs_date_for_test}.'
        ' Returning empty predictions.',
        UserWarning,
    )
    return (
        pd.DataFrame(
            index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
        )
        .fillna(0)
        .astype(int)
    )

  # 7c. Merge test_x with the assembled features for the test observation date
  X_test_merged_df = pd.merge(
      test_x.drop(
          columns=[
              'target_end_date',
              'abbreviation',
              'location_name',
              'population',
          ],
          errors='ignore',
      ),
      test_date_features_df,
      on='location',
      how='left',
  )  # Keep all test_x rows, fill features with NaN if no match

  # The current_nhs_scaled_value is now directly in test_date_features_df (and thus in X_test_merged_df)
  current_nhs_scaled_values_for_test = X_test_merged_df[
      'current_nhs_scaled_value'
  ]

  # Ensure all LGBM features exist in X_test_merged_df, fill with 0 if missing
  for col in lgbm_features:
    if col not in X_test_merged_df.columns:
      X_test_merged_df[col] = 0.0

  X_test_model = X_test_merged_df[lgbm_features]
  X_test_model.index = (
      test_x.index
  )  # Ensure the index matches the input test_x for final output

  # 8. Model Training (Bagging & Quantile LightGBM)
  trained_models = {q: [] for q in QUANTILES}

  # Ensure categorical features are correctly identified for LightGBM
  lgbm_categorical_feature_indices = [
      X_train_model.columns.get_loc(col)
      for col in lgbm_categorical_features
      if col in X_train_model.columns
  ]

  for q in QUANTILES:
    if len(season_years_for_bagging) == 0:
      warnings.warn(
          f'No season years available for bagging. Skipping quantile {q}.',
          UserWarning,
      )
      continue

    for bag_idx in range(N_BAGS):
      rng = np.random.default_rng(seed=hash((q, bag_idx)) % (2**32 - 1))

      sampled_seasons = rng.choice(
          season_years_for_bagging,
          size=len(season_years_for_bagging),
          replace=True,
      )

      bagging_mask = X_train_filtered_df['season_year'].isin(
          sampled_seasons
      )  # Use 'season_year' from filtered df
      X_bag = X_train_model.loc[bagging_mask].copy()
      y_bag = y_train_model.loc[bagging_mask].copy()

      if X_bag.empty or y_bag.empty:
        warnings.warn(
            f'Empty bagging subset for quantile {q} in bag {bag_idx}. Skipping'
            ' this model.',
            UserWarning,
        )
        continue

      lgbm = lgb.LGBMRegressor(
          objective='quantile',
          alpha=q,
          random_state=hash((q, bag_idx))
          % (2**32 - 1),  # Unique random state per bag
          n_estimators=200,  # Increased from 100 to 200
          learning_rate=0.05,  # Reduced learning rate for better generalization
          num_leaves=31,
          max_depth=8,  # Added max_depth to control tree complexity
          min_child_samples=20,  # Added min_child_samples for robustness against overfitting
          lambda_l1=0.1,  # Added L1 regularization
          lambda_l2=0.1,  # Added L2 regularization
          verbose=-1,
          n_jobs=-1,
          categorical_feature=lgbm_categorical_feature_indices,
      )
      lgbm.fit(X_bag, y_bag)
      trained_models[q].append(lgbm)

    if not trained_models[q]:
      warnings.warn(
          f'No models trained for quantile {q}. Predictions for this quantile'
          ' will be zero.',
          UserWarning,
      )

  # 9. Prediction Generation
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Prepare DataFrame for vectorized inverse transform
  df_for_inverse_transform = pd.DataFrame(
      {
          'predicted_transformed_value': (
              current_nhs_scaled_values_for_test.values
              + np.zeros(len(X_test_model))
          ),  # Initialize with current values
          'location': (
              X_test_merged_df['location'].values
          ),  # Ensure locations are aligned
      },
      index=test_x.index,
  )

  test_populations = X_test_merged_df[
      'population'
  ]  # Keep original population for final rate conversion

  for q in QUANTILES:
    predictions_for_q = []
    if trained_models[q]:
      for model in trained_models[q]:
        predictions_for_q.append(model.predict(X_test_model))

      avg_transformed_change = np.mean(predictions_for_q, axis=0)
    else:
      avg_transformed_change = np.zeros(len(X_test_model))

    # Update the 'predicted_transformed_value' column with the predicted scaled value for current quantile
    df_for_inverse_transform['predicted_transformed_value'] = (
        current_nhs_scaled_values_for_test.values + avg_transformed_change
    )

    # Apply inverse transformations using the vectorized DataFrame method
    val_unscaled_after_transform_series = pd.Series(
        data_scaler.inverse_transform_df(
            df_for_inverse_transform,
            'predicted_transformed_value',
            'location',
            'NHSN',
        ),
        index=df_for_inverse_transform.index,
    )

    predicted_rate = val_unscaled_after_transform_series**4
    predicted_rate_capped = predicted_rate.clip(lower=0)

    # Convert rate back to admission counts (vectorized)
    predicted_count = predicted_rate_capped * test_populations / 100_000

    test_y_hat_quantiles[f'quantile_{q}'] = predicted_count.values

  for col in test_y_hat_quantiles.columns:
    test_y_hat_quantiles[col] = (
        test_y_hat_quantiles[col].clip(lower=0).round().astype(int)
    )

  # Ensure monotonicity of quantiles
  sorted_quantiles = np.sort(test_y_hat_quantiles.values, axis=1)
  test_y_hat_quantiles = pd.DataFrame(
      sorted_quantiles, columns=test_y_hat_quantiles.columns, index=test_x.index
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
