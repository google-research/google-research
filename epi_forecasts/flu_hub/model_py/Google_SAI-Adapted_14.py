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
MODEL_NAME = 'Google_SAI-Adapted_14'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import warnings
from pandas.api.types import CategoricalDtype


# Helper for epiweek and season
def get_epiweek_and_season(date_series):
  isocalendar = date_series.dt.isocalendar()
  epiweek = isocalendar.week.astype(int)
  # A flu season typically runs from epiweek 40 of one year to epiweek 39 of the next year.
  # So, epiweeks 40-53 are part of the season starting in that year.
  # Epiweeks 1-39 are part of the season starting in the *previous* year.
  season_start_year = np.where(
      epiweek >= 40,
      isocalendar.year.astype(int),
      isocalendar.year.astype(int) - 1,
  )
  return epiweek, season_start_year


class LocationProcessor(BaseEstimator, TransformerMixin):
  """A transformer to standardize location data, map names to FIPS, and ensure

  'location' (FIPS int), 'location_cat' (CategoricalDtype), and 'population'
  columns are present and consistent.
  It expects 'global_locations_df' and 'global_location_categorical_type' to be
  available.
  """

  def __init__(self, global_locations_df, global_location_categorical_type):
    self.global_locations_df = global_locations_df
    self.global_location_categorical_type = global_location_categorical_type
    self.name_to_fips = dict(
        zip(
            global_locations_df['location_name'],
            global_locations_df['location'],
        )
    )

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_copy = X.copy()

    # Ensure 'location' (FIPS) column exists and is int
    if 'location' not in X_copy.columns:
      # Try mapping from 'location_name' or 'region'
      if 'location_name' in X_copy.columns:
        X_copy['location'] = (
            X_copy['location_name'].map(self.name_to_fips).astype('Int64')
        )
      elif (
          'region' in X_copy.columns
          and 'region_type' in X_copy.columns
          and (X_copy['region_type'] == 'States').any()
      ):
        state_rows = X_copy['region_type'] == 'States'
        X_copy.loc[state_rows, 'location'] = (
            X_copy.loc[state_rows, 'region']
            .map(self.name_to_fips)
            .astype('Int64')
        )
      else:
        # If location cannot be derived, we will drop rows later.
        X_copy['location'] = pd.NA

    # Drop rows where location mapping failed or is irrelevant (e.g., non-state regions in ilinet_state)
    X_copy = X_copy.dropna(subset=['location'])
    X_copy['location'] = X_copy['location'].astype(
        int
    )  # Convert to non-nullable int after dropping NaNs
    X_copy['location_cat'] = X_copy['location'].astype(
        self.global_location_categorical_type
    )

    # Merge population from global locations_df, ensuring it's the canonical source
    X_copy = X_copy.drop(
        columns=['population'], errors='ignore'
    )  # Drop existing population if present
    X_copy = X_copy.merge(
        self.global_locations_df[['location', 'population']],
        on='location',
        how='left',
    )
    X_copy['population'] = X_copy['population'].astype(
        int
    )  # Ensure population is int

    return X_copy.drop(
        columns=['location_name', 'region', 'region_type'], errors='ignore'
    )


# Helper function for adding interaction features to a DataFrame
# Modified to return a new DataFrame, to mitigate fragmentation warnings.
def _add_interaction_features_to_df(
    df, base_features, interaction_partners, suffix_sep='_x_'
):
  new_cols_data = {}
  new_interaction_cols = []
  # Filter interaction_partners to only those columns present in df
  valid_interaction_partners = [
      ip for ip in interaction_partners if ip in df.columns
  ]
  for bf in base_features:
    if bf in df.columns:  # Ensure base feature exists
      for ip in valid_interaction_partners:
        col_name = f'{bf}{suffix_sep}{ip}'
        if (
            col_name not in df.columns
        ):  # Avoid creating duplicates if already present
          new_cols_data[col_name] = df[bf] * df[ip]
          new_interaction_cols.append(col_name)
  if new_cols_data:
    # Concatenate new columns to the original dataframe
    df_with_interactions = pd.concat(
        [df, pd.DataFrame(new_cols_data, index=df.index)], axis=1
    )
    return df_with_interactions, new_interaction_cols
  return df, []


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # Suppress verbose output from statsmodels and common warnings
  warnings.filterwarnings('ignore', category=FutureWarning)
  warnings.filterwarnings('ignore', category=UserWarning)
  warnings.filterwarnings(
      'ignore', message='The default of `normalize` has been deprecated'
  )  # For LinearRegression
  warnings.filterwarnings(
      'ignore', message='`df_residual` is deprecated'
  )  # For QuantReg
  warnings.filterwarnings(
      'ignore', message='is_sparse is deprecated'
  )  # For pd.api.types.is_sparse
  warnings.filterwarnings(
      'ignore',
      message=(
          "The default method 'interior-point' in `scipy.optimize.linprog` will"
          " be changed to 'highs' in SciPy 1.11.0."
      ),
  )  # For QuantReg

  # Make copies to avoid modifying original DFs passed to the function
  train_x = train_x.copy()
  train_y = train_y.copy()
  test_x = test_x.copy()

  # Ensure `locations` and `ilinet_state` are available from global scope
  global locations, ilinet_state

  # --- 1. Data Preprocessing and Unification ---

  # Define a global CategoricalDtype for locations for consistent dummy variable generation
  all_unique_locations = sorted(locations['location'].unique().tolist())
  global_location_categorical_type = CategoricalDtype(
      categories=all_unique_locations, ordered=False
  )

  loc_processor = LocationProcessor(locations, global_location_categorical_type)

  # Process train_df and train_y (the 'dataset' part)
  train_df = pd.concat([train_x, train_y.rename(TARGET_STR)], axis=1)
  train_df['target_end_date'] = pd.to_datetime(train_df['target_end_date'])
  train_df = loc_processor.transform(train_df)
  train_df['Admissions_Per_Capita'] = (
      train_df[TARGET_STR] / train_df['population']
  )
  train_df['epiweek'], train_df['FluSeasonYear'] = get_epiweek_and_season(
      train_df['target_end_date']
  )

  # Process ILINet data (`ilinet_state`)
  ilinet_state_proc = ilinet_state.copy()
  ilinet_state_proc['week_start'] = pd.to_datetime(
      ilinet_state_proc['week_start']
  )
  ilinet_state_proc['target_end_date'] = ilinet_state_proc[
      'week_start'
  ] + pd.Timedelta(
      days=6
  )  # Saturday end date
  ilinet_state_proc = loc_processor.transform(ilinet_state_proc)
  ilinet_state_proc['epiweek'], ilinet_state_proc['FluSeasonYear'] = (
      get_epiweek_and_season(ilinet_state_proc['target_end_date'])
  )
  ilinet_state_proc['ILI_Value'] = ilinet_state_proc['unweighted_ili'].fillna(0)

  # Process test_x
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])
  test_x = loc_processor.transform(test_x)
  test_x['epiweek'], _ = get_epiweek_and_season(test_x['target_end_date'])

  # --- Whitening Process (Global RobustScaler) ---
  adm_scaler = RobustScaler()
  train_df['Admissions_Per_Capita_Clean'] = np.maximum(
      0, train_df['Admissions_Per_Capita']
  )

  if train_df['Admissions_Per_Capita_Clean'].nunique() > 1:
    train_df['Scaled_Admissions_PC'] = adm_scaler.fit_transform(
        train_df[['Admissions_Per_Capita_Clean']]
    )
  else:
    train_df['Scaled_Admissions_PC'] = 0.0
    warnings.warn(
        'Admissions data has no variance. Scaled_Admissions_PC set to 0.0.'
    )

  ili_scaler = RobustScaler()
  ilinet_state_proc['ILI_Value_Clean'] = np.maximum(
      0, ilinet_state_proc['ILI_Value']
  )

  if ilinet_state_proc['ILI_Value_Clean'].nunique() > 1:
    ilinet_state_proc['Scaled_ILI_Value'] = ili_scaler.fit_transform(
        ilinet_state_proc[['ILI_Value_Clean']]
    )
  else:
    ilinet_state_proc['Scaled_ILI_Value'] = 0.0
    warnings.warn('ILI data has no variance. Scaled_ILI_Value set to 0.0.')

  ilinet_state_proc['sin_epiweek'] = np.sin(
      2 * np.pi * ilinet_state_proc['epiweek'] / 52
  )
  ilinet_state_proc['cos_epiweek'] = np.cos(
      2 * np.pi * ilinet_state_proc['epiweek'] / 52
  )

  # Generate all possible location dummy columns (drop_first=False for consistent feature space)
  all_loc_dummy_cols_df = pd.get_dummies(
      pd.Series(
          global_location_categorical_type.categories,
          dtype=global_location_categorical_type,
      ),
      prefix='loc',
      drop_first=False,
  )
  all_loc_dummy_cols = all_loc_dummy_cols_df.columns.tolist()

  # --- 2. Learn a Transformation (Scaled ILI -> Scaled Admissions) ---
  overlap_cols = [
      'location',
      'target_end_date',
      'epiweek',
      'FluSeasonYear',
      'location_cat',
  ]
  overlap_merged = pd.merge(
      train_df[overlap_cols + ['Scaled_Admissions_PC']],
      ilinet_state_proc[
          overlap_cols + ['Scaled_ILI_Value', 'sin_epiweek', 'cos_epiweek']
      ],
      on=overlap_cols,
      how='inner',
  ).dropna()

  ili_transformation_base_features = [
      'Scaled_ILI_Value',
      'sin_epiweek',
      'cos_epiweek',
  ]

  ili_transform_features_pre = overlap_merged[
      ili_transformation_base_features
  ].copy()
  location_dummies_overlap = pd.get_dummies(
      overlap_merged['location_cat'], prefix='loc', drop_first=False
  )
  ili_transform_features_pre = pd.concat(
      [ili_transform_features_pre, location_dummies_overlap], axis=1
  )

  # Simplified key interactions for ILI transformation
  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre, ['Scaled_ILI_Value'], ['Scaled_ILI_Value']
  )  # Squared term
  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre,
      ['Scaled_ILI_Value'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre,
      ['Scaled_ILI_Value_x_Scaled_ILI_Value'],
      ['sin_epiweek', 'cos_epiweek'],
  )  # sq * sin/cos

  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre,
      ['sin_epiweek', 'cos_epiweek'],
      all_loc_dummy_cols,
  )
  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre, ['Scaled_ILI_Value'], all_loc_dummy_cols
  )
  ili_transform_features_pre, _ = _add_interaction_features_to_df(
      ili_transform_features_pre,
      ['Scaled_ILI_Value_x_Scaled_ILI_Value'],
      all_loc_dummy_cols,
  )

  all_ili_transform_model_features_potential = (
      ili_transformation_base_features
      + ['Scaled_ILI_Value_x_Scaled_ILI_Value']  # squared term
      + [
          f'Scaled_ILI_Value_x_{s}' for s in ['sin_epiweek', 'cos_epiweek']
      ]  # ili x sin/cos
      + [
          f'Scaled_ILI_Value_x_Scaled_ILI_Value_x_{s}'
          for s in ['sin_epiweek', 'cos_epiweek']
      ]  # ili_sq x sin/cos
      + all_loc_dummy_cols
      + [f'sin_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [f'cos_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [f'Scaled_ILI_Value_x_{c}' for c in all_loc_dummy_cols]
      + [
          f'Scaled_ILI_Value_x_Scaled_ILI_Value_x_{c}'
          for c in all_loc_dummy_cols
      ]
  )
  all_ili_transform_model_features = sorted(
      list(set(all_ili_transform_model_features_potential))
  )
  X_ili_overlap_for_transform = ili_transform_features_pre.reindex(
      columns=all_ili_transform_model_features, fill_value=0
  )
  y_adm_overlap = overlap_merged['Scaled_Admissions_PC']

  constant_cols_ili_transform_global = X_ili_overlap_for_transform.columns[
      X_ili_overlap_for_transform.nunique() <= 1
  ]
  ili_transform_features_filtered = [
      col
      for col in all_ili_transform_model_features
      if col not in constant_cols_ili_transform_global
  ]
  X_ili_overlap_for_transform_filtered = X_ili_overlap_for_transform.reindex(
      columns=ili_transform_features_filtered, fill_value=0
  )

  ili_full_history_features_pre = ilinet_state_proc[
      ili_transformation_base_features
  ].copy()
  location_dummies_full_history = pd.get_dummies(
      ilinet_state_proc['location_cat'], prefix='loc', drop_first=False
  )
  ili_full_history_features_pre = pd.concat(
      [ili_full_history_features_pre, location_dummies_full_history], axis=1
  )

  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre, ['Scaled_ILI_Value'], ['Scaled_ILI_Value']
  )
  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre,
      ['Scaled_ILI_Value'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre,
      ['Scaled_ILI_Value_x_Scaled_ILI_Value'],
      ['sin_epiweek', 'cos_epiweek'],
  )

  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre,
      ['sin_epiweek', 'cos_epiweek'],
      all_loc_dummy_cols,
  )
  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre, ['Scaled_ILI_Value'], all_loc_dummy_cols
  )
  ili_full_history_features_pre, _ = _add_interaction_features_to_df(
      ili_full_history_features_pre,
      ['Scaled_ILI_Value_x_Scaled_ILI_Value'],
      all_loc_dummy_cols,
  )

  X_ili_full_history_for_transform_filtered = (
      ili_full_history_features_pre.reindex(
          columns=ili_transform_features_filtered, fill_value=0
      )
  )

  fit_main_ili_transform_model = False
  lin_reg_ili_to_adm = LinearRegression(n_jobs=-1, fit_intercept=True)
  min_samples_for_ili_transform = (
      X_ili_overlap_for_transform_filtered.shape[1] + 5
      if not X_ili_overlap_for_transform_filtered.empty
      else 10
  )

  if (
      not overlap_merged.empty
      and len(y_adm_overlap.unique()) > 1
      and (
          X_ili_overlap_for_transform_filtered.shape[0]
          >= min_samples_for_ili_transform
      )
      and X_ili_overlap_for_transform_filtered.shape[1] > 0
  ):
    try:
      lin_reg_ili_to_adm.fit(
          X_ili_overlap_for_transform_filtered, y_adm_overlap
      )
      ilinet_state_proc['Synthetic_Scaled_Admissions_PC'] = (
          lin_reg_ili_to_adm.predict(X_ili_full_history_for_transform_filtered)
      )
      fit_main_ili_transform_model = True
    except Exception as e:
      warnings.warn(
          f'Main ILI to Admissions transformation model failed to fit: {e}.'
          ' Attempting simpler fallback.'
      )
  else:
    warnings.warn(
        'Main ILI to Admissions transformation failed to fit (insufficient'
        ' samples/features/variance). Attempting simpler fallback.'
    )

  if not fit_main_ili_transform_model:
    warnings.warn(
        'Main ILI to Admissions transformation failed. Trying a simpler linear'
        ' model fallback.'
    )

    simple_ili_transformation_base_features = [
        'Scaled_ILI_Value',
        'sin_epiweek',
        'cos_epiweek',
    ]
    X_ili_overlap_simple_pre = overlap_merged[
        simple_ili_transformation_base_features
    ].copy()
    location_dummies_overlap_simple = pd.get_dummies(
        overlap_merged['location_cat'], prefix='loc', drop_first=False
    )
    X_ili_overlap_simple_pre = pd.concat(
        [X_ili_overlap_simple_pre, location_dummies_overlap_simple], axis=1
    )

    simple_transform_model_features_potential = (
        simple_ili_transformation_base_features + all_loc_dummy_cols
    )
    simple_transform_model_features = sorted(
        list(set(simple_transform_model_features_potential))
    )

    X_ili_overlap_simple = X_ili_overlap_simple_pre.reindex(
        columns=simple_transform_model_features, fill_value=0
    )

    constant_cols_ili_transform_simple_global = X_ili_overlap_simple.columns[
        X_ili_overlap_simple.nunique() <= 1
    ]
    simple_ili_transform_features_filtered = [
        col
        for col in simple_transform_model_features
        if col not in constant_cols_ili_transform_simple_global
    ]
    X_ili_overlap_simple_filtered = X_ili_overlap_simple.reindex(
        columns=simple_ili_transform_features_filtered, fill_value=0
    )

    X_ili_full_history_simple_pre = ilinet_state_proc[
        simple_ili_transformation_base_features
    ].copy()
    location_dummies_full_history_simple = pd.get_dummies(
        ilinet_state_proc['location_cat'], prefix='loc', drop_first=False
    )
    X_ili_full_history_simple_pre = pd.concat(
        [X_ili_full_history_simple_pre, location_dummies_full_history_simple],
        axis=1,
    )
    X_ili_full_history_simple_filtered = X_ili_full_history_simple_pre.reindex(
        columns=simple_ili_transform_features_filtered, fill_value=0
    )

    min_samples_for_simple_ili_transform = (
        X_ili_overlap_simple_filtered.shape[1] + 5
        if not X_ili_overlap_simple_filtered.empty
        else 10
    )

    fit_simple_ili_transform_model = False
    if (
        not overlap_merged.empty
        and len(y_adm_overlap.unique()) > 1
        and (
            X_ili_overlap_simple_filtered.shape[0]
            >= min_samples_for_simple_ili_transform
        )
        and X_ili_overlap_simple_filtered.shape[1] > 0
    ):
      try:
        lin_reg_ili_to_adm_simple = LinearRegression(
            n_jobs=-1, fit_intercept=True
        )
        lin_reg_ili_to_adm_simple.fit(
            X_ili_overlap_simple_filtered, y_adm_overlap
        )
        ilinet_state_proc['Synthetic_Scaled_Admissions_PC'] = (
            lin_reg_ili_to_adm_simple.predict(
                X_ili_full_history_simple_filtered
            )
        )
        fit_simple_ili_transform_model = True
      except Exception as e:
        warnings.warn(
            f'Simple ILI to Admissions transformation model failed to fit: {e}.'
            ' Falling back to global scaling fallback.'
        )
    else:
      warnings.warn(
          'Simple ILI to Admissions transformation failed to fit (insufficient'
          ' samples/features/variance). Falling back to global scaling'
          ' fallback.'
      )

  if not fit_main_ili_transform_model and not fit_simple_ili_transform_model:
    warnings.warn(
        'No sufficient overlap data or variance to learn ILI to Admissions'
        ' transformation even with simpler models. Using enhanced scaling'
        ' fallback.'
    )
    if (
        not overlap_merged.empty
        and len(y_adm_overlap.unique()) > 1
        and overlap_merged['Scaled_ILI_Value'].nunique() > 1
    ):
      mean_adm_overlap = y_adm_overlap.mean()
      std_adm_overlap = y_adm_overlap.std()
      mean_ili_overlap = overlap_merged['Scaled_ILI_Value'].mean()
      std_ili_overlap = overlap_merged['Scaled_ILI_Value'].std()
    else:
      mean_adm_overlap = (
          train_df['Scaled_Admissions_PC'].mean() if not train_df.empty else 0.0
      )
      std_adm_overlap = (
          train_df['Scaled_Admissions_PC'].std()
          if train_df['Scaled_Admissions_PC'].nunique() > 1
          else 1.0
      )
      mean_ili_overlap = (
          ilinet_state_proc['Scaled_ILI_Value'].mean()
          if not ilinet_state_proc.empty
          else 0.0
      )
      std_ili_overlap = (
          ilinet_state_proc['Scaled_ILI_Value'].std()
          if ilinet_state_proc['Scaled_ILI_Value'].nunique() > 1
          else 1.0
      )

    if std_ili_overlap > 1e-6:
      ilinet_state_proc['Synthetic_Scaled_Admissions_PC'] = mean_adm_overlap + (
          ilinet_state_proc['Scaled_ILI_Value'] - mean_ili_overlap
      ) * (std_adm_overlap / std_ili_overlap)
    else:
      ilinet_state_proc['Synthetic_Scaled_Admissions_PC'] = mean_adm_overlap

  full_history_real = train_df[overlap_cols + ['Scaled_Admissions_PC']].rename(
      columns={'Scaled_Admissions_PC': 'Normalized_Target_PC'}
  )
  full_history_synthetic = ilinet_state_proc[
      overlap_cols + ['Synthetic_Scaled_Admissions_PC']
  ].rename(columns={'Synthetic_Scaled_Admissions_PC': 'Normalized_Target_PC'})

  real_data_keys = set(
      zip(full_history_real['location'], full_history_real['target_end_date'])
  )
  full_history_synthetic_filtered = full_history_synthetic[
      ~full_history_synthetic.apply(
          lambda row: (row['location'], row['target_end_date'])
          in real_data_keys,
          axis=1,
      )
  ]

  full_history_df = pd.concat(
      [full_history_real, full_history_synthetic_filtered], ignore_index=True
  )
  full_history_df = full_history_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)
  full_history_df['Normalized_Target_PC'] = np.maximum(
      0, full_history_df['Normalized_Target_PC'].fillna(0)
  )

  lags = [1, 2, 3, 52]  # Lags for Linear Model
  qa_lags = [1, 52]  # Reduced lags for Quantile Regression Model

  lag_features = []
  lag_squared_features = []
  for lag in lags:
    lag_col = f'lag_{lag}_target_pc'
    lag_sq_col = f'lag_{lag}_target_pc_sq'
    full_history_df[lag_col] = full_history_df.groupby('location')[
        'Normalized_Target_PC'
    ].shift(lag)
    full_history_df[lag_sq_col] = full_history_df[lag_col] ** 2
    lag_features.append(lag_col)
    lag_squared_features.append(lag_sq_col)

  full_history_df['sin_epiweek'] = np.sin(
      2 * np.pi * full_history_df['epiweek'] / 52
  )
  full_history_df['cos_epiweek'] = np.cos(
      2 * np.pi * full_history_df['epiweek'] / 52
  )

  lags_to_check = lag_features + lag_squared_features
  full_history_df_model = full_history_df.dropna(subset=lags_to_check)

  global_climatological_scaled_admissions_quantiles = {}
  if not full_history_df.empty:
    for q in QUANTILES:
      global_climatological_scaled_admissions_quantiles[q] = full_history_df[
          'Normalized_Target_PC'
      ].quantile(q, interpolation='linear')
  else:
    for q in QUANTILES:
      global_climatological_scaled_admissions_quantiles[q] = 0.0
  global_median_scaled_target_pc = (
      global_climatological_scaled_admissions_quantiles[0.5]
  )

  climatological_medians_pc = (
      full_history_df.groupby(['location', 'epiweek'])['Normalized_Target_PC']
      .median()
      .reset_index()
  )
  climatological_medians_pc = climatological_medians_pc.set_index(
      ['location', 'epiweek']
  )['Normalized_Target_PC']

  # --- Pre-train Linear Model (for baseline and recursive lags) ---
  linear_model_base_features = (
      lag_features + lag_squared_features + ['sin_epiweek', 'cos_epiweek']
  )

  linear_model_features_pre = full_history_df_model[
      linear_model_base_features + ['location_cat']
  ].copy()
  linear_model_features_pre = pd.get_dummies(
      linear_model_features_pre, columns=['location_cat'], drop_first=False
  )

  location_dummy_cols = [
      col for col in linear_model_features_pre.columns if col.startswith('loc_')
  ]
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre,
      ['sin_epiweek', 'cos_epiweek'],
      location_dummy_cols,
  )
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre,
      ['lag_1_target_pc', 'lag_1_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre,
      ['lag_52_target_pc', 'lag_52_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
  )  # lag_1_x_lag_52_pc
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre, ['lag_1_target_pc'], location_dummy_cols
  )
  linear_model_features_pre, _ = _add_interaction_features_to_df(
      linear_model_features_pre, ['lag_52_target_pc'], location_dummy_cols
  )

  all_linear_model_features_potential = (
      linear_model_base_features
      + all_loc_dummy_cols
      + [f'sin_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [f'cos_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [
          'lag_1_target_pc_x_sin_epiweek',
          'lag_1_target_pc_x_cos_epiweek',
          'lag_1_target_pc_sq_x_sin_epiweek',
          'lag_1_target_pc_sq_x_cos_epiweek',
      ]
      + [
          'lag_52_target_pc_x_sin_epiweek',
          'lag_52_target_pc_x_cos_epiweek',
          'lag_52_target_pc_sq_x_sin_epiweek',
          'lag_52_target_pc_sq_x_cos_epiweek',
      ]
      + ['lag_1_target_pc_x_lag_52_target_pc']
      + [f'lag_1_target_pc_x_{c}' for c in all_loc_dummy_cols]
      + [f'lag_52_target_pc_x_{c}' for c in all_loc_dummy_cols]
  )
  all_linear_model_features = sorted(
      list(set(all_linear_model_features_potential))
  )

  X_linear_train = linear_model_features_pre.reindex(
      columns=all_linear_model_features, fill_value=0
  )

  constant_cols_lin_global = X_linear_train.columns[
      X_linear_train.nunique() <= 1
  ]
  linear_model_features_filtered = [
      col
      for col in all_linear_model_features
      if col not in constant_cols_lin_global
  ]
  X_linear_train_filtered = X_linear_train.reindex(
      columns=linear_model_features_filtered, fill_value=0
  )
  y_linear_train_scaled = full_history_df_model['Normalized_Target_PC']

  lin_model = None
  linear_predictions_scaled = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )
  for q in QUANTILES:
    linear_predictions_scaled[f'quantile_{q}'] = (
        global_climatological_scaled_admissions_quantiles[q]
    )

  if not X_linear_train_filtered.empty:
    if len(y_linear_train_scaled.unique()) > 1:
      min_samples_for_lin_model = X_linear_train_filtered.shape[1] + 5
      if (
          X_linear_train_filtered.shape[0] >= min_samples_for_lin_model
          and X_linear_train_filtered.shape[1] > 0
      ):
        try:
          lin_model = LinearRegression(n_jobs=-1, fit_intercept=True)
          lin_model.fit(X_linear_train_filtered, y_linear_train_scaled)
        except Exception as e:
          warnings.warn(
              f'Linear model pre-training failed to fit: {e}. Using'
              ' climatological fallback.'
          )
      else:
        warnings.warn(
            'Skipping Linear model pre-training due to insufficient samples'
            f' ({X_linear_train_filtered.shape[0]} <'
            f' {min_samples_for_lin_model}) or no features. Using'
            ' climatological fallback.'
        )
    else:
      warnings.warn(
          'Skipping Linear model pre-training due to insufficient variance in'
          ' training target. Using climatological fallback.'
      )
  else:
    warnings.warn(
        'Skipping Linear model pre-training due to insufficient training data.'
        ' Using climatological fallback.'
    )

  epiweek_residual_quantiles = {}
  global_residual_quantiles = {q: 0.0 for q in QUANTILES}

  if lin_model is not None and not full_history_df_model.empty:
    train_residuals_df = full_history_df_model.copy()

    X_linear_train_for_preds_pre = train_residuals_df[
        linear_model_base_features + ['location_cat']
    ].copy()
    X_linear_train_for_preds_pre = pd.get_dummies(
        X_linear_train_for_preds_pre, columns=['location_cat'], drop_first=False
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre,
        ['sin_epiweek', 'cos_epiweek'],
        location_dummy_cols,
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre,
        ['lag_1_target_pc', 'lag_1_target_pc_sq'],
        ['sin_epiweek', 'cos_epiweek'],
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre,
        ['lag_52_target_pc', 'lag_52_target_pc_sq'],
        ['sin_epiweek', 'cos_epiweek'],
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre, ['lag_1_target_pc'], location_dummy_cols
    )
    X_linear_train_for_preds_pre, _ = _add_interaction_features_to_df(
        X_linear_train_for_preds_pre, ['lag_52_target_pc'], location_dummy_cols
    )

    X_linear_train_for_preds = X_linear_train_for_preds_pre.reindex(
        columns=lin_model.feature_names_in_, fill_value=0
    )

    train_residuals_df['prediction_scaled'] = lin_model.predict(
        X_linear_train_for_preds
    )
    train_residuals_df['residual_scaled'] = (
        train_residuals_df['Normalized_Target_PC']
        - train_residuals_df['prediction_scaled']
    )

    if not train_residuals_df.empty:
      global_residual_quantiles = {
          q: (
              train_residuals_df['residual_scaled'].quantile(
                  q, interpolation='linear'
              )
          )
          for q in QUANTILES
      }
      for ew in train_residuals_df['epiweek'].unique():
        ew_residuals = train_residuals_df[train_residuals_df['epiweek'] == ew][
            'residual_scaled'
        ]
        if len(ew_residuals) >= 10 and ew_residuals.nunique() > 1:
          epiweek_residual_quantiles[ew] = {
              q: ew_residuals.quantile(q, interpolation='linear')
              for q in QUANTILES
          }
        else:
          epiweek_residual_quantiles[ew] = global_residual_quantiles
    else:
      warnings.warn(
          'No residuals generated from linear model for dynamic quantile'
          ' estimation. Using global residual quantiles (likely 0s).'
      )

  test_x_processed = test_x.reset_index().rename(
      columns={'index': 'original_index'}
  )
  test_x_processed['sin_epiweek'] = np.sin(
      2 * np.pi * test_x_processed['epiweek'] / 52
  )
  test_x_processed['cos_epiweek'] = np.cos(
      2 * np.pi * test_x_processed['epiweek'] / 52
  )
  test_x_processed = test_x_processed.sort_values(
      by=['reference_date', 'horizon', 'location']
  )
  test_x_processed = test_x_processed.set_index('original_index')

  full_ts_data_for_lags = full_history_df.set_index(
      ['location', 'target_end_date']
  )['Normalized_Target_PC'].copy()

  for (
      lag
  ) in lags:  # Use all lags for recursive filling as linear model uses them
    lag_dates = test_x_processed['target_end_date'] - pd.Timedelta(weeks=lag)
    test_x_processed[f'lag_{lag}_target_pc'] = [
        full_ts_data_for_lags.get((loc, date), np.nan)
        for loc, date in zip(test_x_processed['location'], lag_dates)
    ]
    test_x_processed[f'lag_{lag}_target_pc_sq'] = (
        test_x_processed[f'lag_{lag}_target_pc'] ** 2
    )

  test_x_processed['lag_1_target_pc_x_lag_52_target_pc'] = (
      test_x_processed['lag_1_target_pc'] * test_x_processed['lag_52_target_pc']
  )

  # Initialize lag_x_location interactions for test_x_processed
  for lag_col_name in ['lag_1_target_pc', 'lag_52_target_pc']:
    for loc_col in all_loc_dummy_cols:
      loc_fips = int(loc_col.split('_')[1])
      test_x_processed[f'{lag_col_name}_x_{loc_col}'] = test_x_processed[
          lag_col_name
      ] * (test_x_processed['location'] == loc_fips).astype(int)

  for original_idx, row in test_x_processed.iterrows():
    current_loc = row['location']
    current_target_end_date = row['target_end_date']
    current_epiweek = row['epiweek']

    lags_to_fill_mask = test_x_processed.loc[original_idx, lags_to_check].isna()
    if lags_to_fill_mask.any():
      for lag_col_name in np.array(lags_to_check)[lags_to_fill_mask]:
        if '_sq' in lag_col_name:
          base_lag_col_name = lag_col_name.replace('_sq', '')
          lag_num = int(base_lag_col_name.split('_')[1])
        else:
          base_lag_col_name = lag_col_name
          lag_num = int(lag_col_name.split('_')[1])

        lag_date = current_target_end_date - pd.Timedelta(weeks=lag_num)

        lag_value_scaled = full_ts_data_for_lags.get(
            (current_loc, lag_date), np.nan
        )

        if pd.isna(lag_value_scaled):
          lag_epiweek_for_clim = (lag_date.isocalendar().week - 1) % 52 + 1
          lag_value_scaled = climatological_medians_pc.get(
              (current_loc, lag_epiweek_for_clim),
              global_median_scaled_target_pc,
          )

        test_x_processed.loc[original_idx, base_lag_col_name] = lag_value_scaled
        test_x_processed.loc[original_idx, f'{base_lag_col_name}_sq'] = (
            lag_value_scaled**2
        )

      # Recalculate interaction features after filling lags for this row
      test_x_processed.loc[
          original_idx, 'lag_1_target_pc_x_lag_52_target_pc'
      ] = (
          test_x_processed.loc[original_idx, 'lag_1_target_pc']
          * test_x_processed.loc[original_idx, 'lag_52_target_pc']
      )
      for lag_col_name in ['lag_1_target_pc', 'lag_52_target_pc']:
        for loc_col in all_loc_dummy_cols:
          loc_fips = int(loc_col.split('_')[1])
          test_x_processed.loc[
              original_idx, f'{lag_col_name}_x_{loc_col}'
          ] = test_x_processed.loc[original_idx, lag_col_name] * (
              test_x_processed.loc[original_idx, 'location'] == loc_fips
          ).astype(
              int
          )

    if lin_model:
      single_row_features_pre = test_x_processed.loc[[original_idx]][
          linear_model_base_features + ['location_cat']
      ].copy()
      single_row_features_pre = pd.get_dummies(
          single_row_features_pre, columns=['location_cat'], drop_first=False
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre,
          ['sin_epiweek', 'cos_epiweek'],
          location_dummy_cols,
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre,
          ['lag_1_target_pc', 'lag_1_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre,
          ['lag_52_target_pc', 'lag_52_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre, ['lag_1_target_pc'], location_dummy_cols
      )
      single_row_features_pre, _ = _add_interaction_features_to_df(
          single_row_features_pre, ['lag_52_target_pc'], location_dummy_cols
      )

      X_linear_single_row = single_row_features_pre.reindex(
          columns=lin_model.feature_names_in_, fill_value=0
      )

      median_pred = lin_model.predict(X_linear_single_row)[0]
    else:
      median_pred = global_median_scaled_target_pc

    full_ts_data_for_lags[(current_loc, current_target_end_date)] = median_pred

  # --- Feature set for Quantile Auto-Regression (QA) model ---
  # Use a reduced set of lags for QA to improve robustness and convergence with limited windowed data
  qa_reduced_lag_features = [f'lag_{lag}_target_pc' for lag in qa_lags]
  qa_reduced_lag_squared_features = [
      f'lag_{lag}_target_pc_sq' for lag in qa_lags
  ]

  qa_features_base = (
      qa_reduced_lag_features
      + qa_reduced_lag_squared_features
      + ['sin_epiweek', 'cos_epiweek']
  )

  qa_model_features_pre = full_history_df_model[
      qa_features_base + ['location_cat']
  ].copy()
  qa_model_features_pre = pd.get_dummies(
      qa_model_features_pre, columns=['location_cat'], drop_first=False
  )

  # Use all_loc_dummy_cols for consistency in generating interactions
  qa_location_dummy_cols = all_loc_dummy_cols

  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre,
      ['sin_epiweek', 'cos_epiweek'],
      qa_location_dummy_cols,
  )
  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre,
      ['lag_1_target_pc', 'lag_1_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre,
      ['lag_52_target_pc', 'lag_52_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
  )  # lag_1_x_lag_52_pc
  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre, ['lag_1_target_pc'], qa_location_dummy_cols
  )
  qa_model_features_pre, _ = _add_interaction_features_to_df(
      qa_model_features_pre, ['lag_52_target_pc'], qa_location_dummy_cols
  )

  all_qa_model_features_potential = (
      qa_features_base
      + all_loc_dummy_cols
      + [f'sin_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [f'cos_epiweek_x_{c}' for c in all_loc_dummy_cols]
      + [
          'lag_1_target_pc_x_sin_epiweek',
          'lag_1_target_pc_x_cos_epiweek',
          'lag_1_target_pc_sq_x_sin_epiweek',
          'lag_1_target_pc_sq_x_cos_epiweek',
      ]
      + [
          'lag_52_target_pc_x_sin_epiweek',
          'lag_52_target_pc_x_cos_epiweek',
          'lag_52_target_pc_sq_x_sin_epiweek',
          'lag_52_target_pc_sq_x_cos_epiweek',
      ]
      + ['lag_1_target_pc_x_lag_52_target_pc']
      + [f'lag_1_target_pc_x_{c}' for c in all_loc_dummy_cols]
      + [f'lag_52_target_pc_x_{c}' for c in all_loc_dummy_cols]
  )
  all_qa_model_features = sorted(list(set(all_qa_model_features_potential)))

  X_full_history_qa_base = qa_model_features_pre.reindex(
      columns=all_qa_model_features, fill_value=0
  )
  X_full_history_qa = sm.add_constant(
      X_full_history_qa_base, has_constant='add'
  )
  y_full_history_qa = full_history_df_model['Normalized_Target_PC']

  globally_constant_qa_cols = X_full_history_qa.columns[
      X_full_history_qa.std() < 1e-9
  ]
  qa_model_full_cols_filtered = [
      col
      for col in X_full_history_qa.columns
      if col not in globally_constant_qa_cols
  ]

  test_x_qa_base_pre = test_x_processed[
      qa_features_base + ['location_cat']
  ].copy()
  test_x_qa_base_pre = pd.get_dummies(
      test_x_qa_base_pre, columns=['location_cat'], drop_first=False
  )

  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre, ['sin_epiweek', 'cos_epiweek'], qa_location_dummy_cols
  )
  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre,
      ['lag_1_target_pc', 'lag_1_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre,
      ['lag_52_target_pc', 'lag_52_target_pc_sq'],
      ['sin_epiweek', 'cos_epiweek'],
  )
  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
  )
  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre, ['lag_1_target_pc'], qa_location_dummy_cols
  )
  test_x_qa_base_pre, _ = _add_interaction_features_to_df(
      test_x_qa_base_pre, ['lag_52_target_pc'], qa_location_dummy_cols
  )

  test_x_qa_base = test_x_qa_base_pre.reindex(
      columns=all_qa_model_features, fill_value=0
  )
  test_x_qa = sm.add_constant(test_x_qa_base, has_constant='add')
  test_x_qa_filtered = test_x_qa.reindex(
      columns=qa_model_full_cols_filtered, fill_value=0
  )

  # --- 4. Baseline Model Generation (Linear Model first, for QA fallback) ---
  if lin_model:
    X_linear_test_pre = test_x_processed[
        linear_model_base_features + ['location_cat']
    ].copy()
    X_linear_test_pre = pd.get_dummies(
        X_linear_test_pre, columns=['location_cat'], drop_first=False
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre, ['sin_epiweek', 'cos_epiweek'], location_dummy_cols
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre,
        ['lag_1_target_pc', 'lag_1_target_pc_sq'],
        ['sin_epiweek', 'cos_epiweek'],
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre,
        ['lag_52_target_pc', 'lag_52_target_pc_sq'],
        ['sin_epiweek', 'cos_epiweek'],
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre, ['lag_1_target_pc'], location_dummy_cols
    )
    X_linear_test_pre, _ = _add_interaction_features_to_df(
        X_linear_test_pre, ['lag_52_target_pc'], location_dummy_cols
    )

    X_linear_test = X_linear_test_pre.reindex(
        columns=lin_model.feature_names_in_, fill_value=0
    )

    mean_preds_scaled_linear = lin_model.predict(X_linear_test)

    for q_idx, q in enumerate(QUANTILES):
      linear_predictions_scaled.loc[:, f'quantile_{q}'] = (
          test_x_processed.apply(
              lambda row: mean_preds_scaled_linear[
                  test_x_processed.index.get_loc(row.name)
              ]
              + epiweek_residual_quantiles.get(
                  row['epiweek'], global_residual_quantiles
              )[q],
              axis=1,
          )
      )
  else:
    warnings.warn(
        'Linear model was not trained; linear_predictions_scaled relies on'
        ' climatological fallbacks already set.'
    )

  linear_predictions_pc = np.maximum(
      0,
      pd.DataFrame(
          adm_scaler.inverse_transform(linear_predictions_scaled),
          index=linear_predictions_scaled.index,
          columns=linear_predictions_scaled.columns,
      ),
  )
  linear_predictions_abs = linear_predictions_pc.multiply(
      test_x_processed['population'], axis=0
  )

  linear_predictions_final = np.maximum(0, linear_predictions_abs)
  linear_predictions_final = pd.DataFrame(
      np.maximum.accumulate(linear_predictions_final.values, axis=1),
      index=linear_predictions_final.index,
      columns=linear_predictions_final.columns,
  )

  clim_predictions_scaled = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )
  augmented_historical_scaled_admissions = full_history_df[
      ['location', 'epiweek', 'Normalized_Target_PC']
  ]
  min_samples_for_clim_quantile = 10

  for idx, row in test_x_processed.iterrows():
    loc = row['location']
    ew = row['epiweek']

    historical_values_scaled = augmented_historical_scaled_admissions[
        (augmented_historical_scaled_admissions['location'] == loc)
        & (augmented_historical_scaled_admissions['epiweek'] == ew)
    ]['Normalized_Target_PC']

    if (
        not historical_values_scaled.empty
        and len(historical_values_scaled) >= min_samples_for_clim_quantile
        and historical_values_scaled.nunique() > 1
    ):
      for q in QUANTILES:
        clim_predictions_scaled.loc[idx, f'quantile_{q}'] = (
            historical_values_scaled.quantile(q, interpolation='linear')
        )
    else:
      for q in QUANTILES:
        clim_predictions_scaled.loc[idx, f'quantile_{q}'] = (
            global_climatological_scaled_admissions_quantiles[q]
        )

  clim_predictions_pc = np.maximum(
      0,
      pd.DataFrame(
          adm_scaler.inverse_transform(clim_predictions_scaled),
          index=clim_predictions_scaled.index,
          columns=clim_predictions_scaled.columns,
      ),
  )
  clim_predictions_abs = clim_predictions_pc.multiply(
      test_x_processed['population'], axis=0
  )

  clim_predictions_final = np.maximum(0, clim_predictions_abs)
  clim_predictions_final = pd.DataFrame(
      np.maximum.accumulate(clim_predictions_final.values, axis=1),
      index=clim_predictions_final.index,
      columns=clim_predictions_final.columns,
  )

  # --- 3. Windowed Quantile Auto-Regression (with improved fallback) ---
  qa_raw_predictions_scaled = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )
  qa_raw_predictions_scaled = (
      linear_predictions_scaled.copy()
  )  # Initialize with linear predictions as a robust fallback

  unique_test_epiweeks = test_x_processed['epiweek'].unique()

  if not full_history_df_model.empty:
    for current_epiweek_in_test in unique_test_epiweeks:
      window_epiweeks = [
          (current_epiweek_in_test + offset - 1) % 52 + 1
          for offset in range(-3, 4)
      ]

      windowed_train_df = full_history_df_model[
          full_history_df_model['epiweek'].isin(window_epiweeks)
      ]

      # Use qa_features_base for windowed model (reduced lags)
      qa_current_window_lag_features = [
          f'lag_{lag}_target_pc' for lag in qa_lags
      ]
      qa_current_window_lag_squared_features = [
          f'lag_{lag}_target_pc_sq' for lag in qa_lags
      ]
      qa_current_window_features_base = (
          qa_current_window_lag_features
          + qa_current_window_lag_squared_features
          + ['sin_epiweek', 'cos_epiweek']
      )

      X_windowed_base_pre = windowed_train_df[
          qa_current_window_features_base + ['location_cat']
      ].copy()
      X_windowed_base_pre = pd.get_dummies(
          X_windowed_base_pre, columns=['location_cat'], drop_first=False
      )
      qa_location_dummy_cols_window = (
          all_loc_dummy_cols  # Consistent dummy columns
      )

      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre,
          ['sin_epiweek', 'cos_epiweek'],
          qa_location_dummy_cols_window,
      )
      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre,
          ['lag_1_target_pc', 'lag_1_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre,
          ['lag_52_target_pc', 'lag_52_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre, ['lag_1_target_pc'], ['lag_52_target_pc']
      )
      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre,
          ['lag_1_target_pc'],
          qa_location_dummy_cols_window,
      )
      X_windowed_base_pre, _ = _add_interaction_features_to_df(
          X_windowed_base_pre,
          ['lag_52_target_pc'],
          qa_location_dummy_cols_window,
      )

      # Reconstruct all_qa_model_features_potential based on reduced qa_lags
      all_qa_model_features_potential = (
          qa_current_window_features_base
          + all_loc_dummy_cols
          + [f'sin_epiweek_x_{c}' for c in all_loc_dummy_cols]
          + [f'cos_epiweek_x_{c}' for c in all_loc_dummy_cols]
          + [
              'lag_1_target_pc_x_sin_epiweek',
              'lag_1_target_pc_x_cos_epiweek',
              'lag_1_target_pc_sq_x_sin_epiweek',
              'lag_1_target_pc_sq_x_cos_epiweek',
          ]
          + [
              'lag_52_target_pc_x_sin_epiweek',
              'lag_52_target_pc_x_cos_epiweek',
              'lag_52_target_pc_sq_x_sin_epiweek',
              'lag_52_target_pc_sq_x_cos_epiweek',
          ]
          + ['lag_1_target_pc_x_lag_52_target_pc']
          + [f'lag_1_target_pc_x_{c}' for c in all_loc_dummy_cols]
          + [f'lag_52_target_pc_x_{c}' for c in all_loc_dummy_cols]
      )
      all_qa_model_features = sorted(list(set(all_qa_model_features_potential)))

      X_windowed_base = X_windowed_base_pre.reindex(
          columns=all_qa_model_features, fill_value=0
      )
      X_windowed = sm.add_constant(X_windowed_base, has_constant='add')

      # Apply global constant column filtering
      # Recreate test_x_qa_filtered based on the current qa_features_base
      test_x_qa_base_current_features_pre = test_x_processed[
          qa_current_window_features_base + ['location_cat']
      ].copy()
      test_x_qa_base_current_features_pre = pd.get_dummies(
          test_x_qa_base_current_features_pre,
          columns=['location_cat'],
          drop_first=False,
      )

      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['sin_epiweek', 'cos_epiweek'],
          qa_location_dummy_cols,
      )
      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['lag_1_target_pc', 'lag_1_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['lag_52_target_pc', 'lag_52_target_pc_sq'],
          ['sin_epiweek', 'cos_epiweek'],
      )
      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['lag_1_target_pc'],
          ['lag_52_target_pc'],
      )
      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['lag_1_target_pc'],
          qa_location_dummy_cols,
      )
      test_x_qa_base_current_features_pre, _ = _add_interaction_features_to_df(
          test_x_qa_base_current_features_pre,
          ['lag_52_target_pc'],
          qa_location_dummy_cols,
      )

      test_x_qa_base_current_features = (
          test_x_qa_base_current_features_pre.reindex(
              columns=all_qa_model_features, fill_value=0
          )
      )
      test_x_qa_current_features = sm.add_constant(
          test_x_qa_base_current_features, has_constant='add'
      )

      globally_constant_qa_cols_current_features = X_windowed.columns[
          X_windowed.std() < 1e-9
      ]  # Recalculate based on current X_windowed
      qa_model_full_cols_filtered_current_features = [
          col
          for col in X_windowed.columns
          if col not in globally_constant_qa_cols_current_features
      ]

      X_windowed_filtered_global = X_windowed.reindex(
          columns=qa_model_full_cols_filtered_current_features, fill_value=0
      )
      current_window_non_constant_cols = X_windowed_filtered_global.columns[
          X_windowed_filtered_global.std() > 1e-9
      ]
      X_windowed_for_fit_dynamic = X_windowed_filtered_global.reindex(
          columns=current_window_non_constant_cols, fill_value=0
      )

      test_x_slice_indices = test_x_processed[
          test_x_processed['epiweek'] == current_epiweek_in_test
      ].index

      # Align test data to the dynamically selected features *before* predicting
      test_x_qa_filtered_current_features = test_x_qa_current_features.reindex(
          columns=qa_model_full_cols_filtered_current_features, fill_value=0
      )
      X_test_slice_aligned_to_window = test_x_qa_filtered_current_features.loc[
          test_x_slice_indices
      ].reindex(columns=X_windowed_for_fit_dynamic.columns, fill_value=0)

      min_samples_for_qr = (
          X_windowed_for_fit_dynamic.shape[1] + 5
          if not X_windowed_for_fit_dynamic.empty
          else 10
      )  # Adjust min_samples based on dynamic features

      if (
          windowed_train_df.empty
          or windowed_train_df['Normalized_Target_PC'].nunique() < 2
          or len(windowed_train_df) < min_samples_for_qr
          or X_windowed_for_fit_dynamic.empty
          or X_windowed_for_fit_dynamic.shape[1] == 0
      ):  # Explicitly check dynamic features
        warnings.warn(
            'Insufficient training data or target variance for QA model in'
            f' epiweek window around {current_epiweek_in_test}'
            f' (samples={len(windowed_train_df)},'
            f' dynamic_features={X_windowed_for_fit_dynamic.shape[1]}). Falling'
            ' back to linear model predictions for this window.'
        )
        continue

      y_windowed = windowed_train_df['Normalized_Target_PC']

      for q in QUANTILES:
        try:
          # Fit QuantReg with dynamically filtered features
          model = sm.QuantReg(y_windowed, X_windowed_for_fit_dynamic).fit(
              q=q, disp=0, max_iter=15000
          )  # Increased max_iter slightly

          # Check for NaN or inf coefficients
          if model.params.isnull().any() or np.isinf(model.params).any():
            raise ValueError(
                'QuantReg produced NaN/inf coefficients, indicating'
                ' instability.'
            )

          # Predict using test data aligned to the features the model was actually fitted on
          predictions_q = model.predict(X_test_slice_aligned_to_window)
          qa_raw_predictions_scaled.loc[
              test_x_slice_indices, f'quantile_{q}'
          ] = predictions_q
        except Exception as e:
          warnings.warn(
              f'QuantReg failed for quantile {q}, epiweek'
              f' {current_epiweek_in_test} with dynamic feature filtering: {e}.'
              ' Falling back to linear model predictions for this quantile in'
              ' this window.'
          )
  else:
    warnings.warn(
        'No sufficient history available to train Windowed Quantile'
        ' Auto-Regression. Falling back to linear model predictions.'
    )

  qa_predictions_pc = np.maximum(
      0,
      pd.DataFrame(
          adm_scaler.inverse_transform(qa_raw_predictions_scaled),
          index=qa_raw_predictions_scaled.index,
          columns=qa_raw_predictions_scaled.columns,
      ),
  )
  qa_predictions_abs = qa_predictions_pc.multiply(
      test_x_processed['population'], axis=0
  )

  qa_predictions_final = np.maximum(0, qa_predictions_abs)
  qa_predictions_final = pd.DataFrame(
      np.maximum.accumulate(qa_predictions_final.values, axis=1),
      index=qa_predictions_final.index,
      columns=qa_predictions_final.columns,
  )

  # --- 5. Ensemble Averaging ---
  final_predictions = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )

  weight_qa = 6
  weight_clim = 1
  weight_linear = 1
  total_weight = weight_qa + weight_clim + weight_linear

  for q in QUANTILES:
    final_predictions[f'quantile_{q}'] = (
        weight_qa * qa_predictions_final[f'quantile_{q}']
        + weight_clim * clim_predictions_final[f'quantile_{q}']
        + weight_linear * linear_predictions_final[f'quantile_{q}']
    ) / total_weight

  final_predictions = np.maximum(0, final_predictions)
  final_predictions = pd.DataFrame(
      np.maximum.accumulate(final_predictions.values, axis=1),
      index=final_predictions.index,
      columns=final_predictions.columns,
  )

  final_predictions = final_predictions.round().astype(int)

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
