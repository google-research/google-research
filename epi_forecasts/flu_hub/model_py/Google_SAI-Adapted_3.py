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
MODEL_NAME = 'Google_SAI-Adapted_3'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from scipy.stats import norm

TARGET_STR_LOG = 'Total Influenza Admissions_log1p'

# --- Configuration Constants ---


# Global constants for the model (can be tweaked)
N_SAMPLES = 1000  # Number of samples to draw for quantile estimation
LAG_COUNT = 6  # Short-term lags for previous target values and ILI values

# Static mapping from location_name to HHS region name
HHS_REGION_MAP = {
    'Alabama': 'Region 4',
    'Alaska': 'Region 10',
    'Arizona': 'Region 9',
    'Arkansas': 'Region 6',
    'California': 'Region 9',
    'Colorado': 'Region 8',
    'Connecticut': 'Region 1',
    'Delaware': 'Region 3',
    'District of Columbia': 'Region 3',
    'Florida': 'Region 4',
    'Georgia': 'Region 4',
    'Hawaii': 'Region 9',
    'Idaho': 'Region 10',
    'Illinois': 'Region 5',
    'Indiana': 'Region 5',
    'Iowa': 'Region 7',
    'Kansas': 'Region 7',
    'Kentucky': 'Region 4',
    'Louisiana': 'Region 6',
    'Maine': 'Region 1',
    'Maryland': 'Region 3',
    'Massachusetts': 'Region 1',
    'Michigan': 'Region 5',
    'Minnesota': 'Region 5',
    'Mississippi': 'Region 4',
    'Missouri': 'Region 7',
    'Montana': 'Region 8',
    'Nebraska': 'Region 7',
    'Nevada': 'Region 9',
    'New Hampshire': 'Region 1',
    'New Jersey': 'Region 2',
    'New Mexico': 'Region 6',
    'New York': 'Region 2',
    'North Carolina': 'Region 4',
    'North Dakota': 'Region 8',
    'Ohio': 'Region 5',
    'Oklahoma': 'Region 6',
    'Oregon': 'Region 10',
    'Pennsylvania': 'Region 3',
    'Puerto Rico': 'Region 2',
    'Rhode Island': 'Region 1',
    'South Carolina': 'Region 4',
    'South Dakota': 'Region 8',
    'Tennessee': 'Region 4',
    'Texas': 'Region 6',
    'Utah': 'Region 8',
    'Vermont': 'Region 1',
    'Virginia': 'Region 3',
    'Washington': 'Region 10',
    'West Virginia': 'Region 3',
    'Wisconsin': 'Region 5',
    'Wyoming': 'Region 8',
}


# Helper function to align ILINet week_start to target_end_date (Saturday)
def _align_ilinet_dates(df):
  df_copy = df.copy()  # Operate on a copy to avoid SettingWithCopyWarning
  df_copy['week_start'] = pd.to_datetime(df_copy['week_start'])
  df_copy['target_end_date'] = df_copy['week_start'] + pd.Timedelta(days=6)
  return df_copy


# IMPLEMENTATION PLAN.
# ## Core Principles Checklist
# 1.  **Spatial time-series framework**: The model will be trained on data from all locations simultaneously, incorporating `location` as one-hot encoded features, `population_scaled`, `sin_week`, `cos_week`, `days_since_start`, `national_ili_lag` (including `lag_51`, `lag_52`, `lag_53`), `TARGET_STR_LOG_lag_51`, `TARGET_STR_LOG_lag_52`, `TARGET_STR_LOG_lag_53`, `rest_of_us_target_log_lag`, and `hhs_ili_lag` features (including `lag_51`, `lag_52`, `lag_53` for `hhs_ili`) as broad-scale and direct spatial drivers, generating forecasts for all locations. The calculation of `rest_of_us_target_log` will remain vectorized.
# 2.  **Seasonality terms**: Periodic features derived from the `target_end_date` (`week_of_year` sine/cosine transformations) will be explicitly added to the feature set to capture seasonal patterns. Additionally, explicit `lag_51`, `lag_52`, `lag_53` features for `unweighted_ili`, `national_ili`, `hhs_ili`, and the `log(1+TARGET_STR)` target will be incorporated to capture strong year-on-year seasonal dependencies, leveraging the long historical ILI data provided and allowing for slight shifts in flu season timing. The transformation from ILI to admissions will also become seasonally aware by incorporating `sin_week` and `cos_week` in its linear model.
# 3.  **State-specific connectivity**: State-specific effects will be modeled by including one-hot encoded `location` identifiers (fixed effects). `rest_of_us_target_log_lag` features (population-weighted average of past influenza activity from other states) will continue to capture broad inter-state influence. The `hhs_ili_lag` features (including the new `lag_51`, `lag_52`, `lag_53` for HHS ILI), reflecting historical ILI activity within a state's HHS region, will provide a more refined regional-level spatial connectivity signal, differentiating between national and adjacent region impacts.
# 4.  **Random walk component**: Autoregressive features, specifically lagged `log(1+TARGET_STR)` values (with `LAG_COUNT` of 6, and seasonal lags 51, 52, 53), will be included. A `days_since_start` feature will also be added to capture general temporal trends. The `lag_51`, `lag_52`, `lag_53` features for `unweighted_ili`, `national_ili`, `hhs_ili`, and `log(1+TARGET_STR)` will significantly strengthen the random walk's ability to extrapolate seasonal patterns over longer terms by providing robust seasonal anchors by referring to data 51, 52, and 53 weeks prior to the target end date.
# 5.  **Unified Bayesian inference method designed for hierarchical models**: A `BayesianRidge` model will be trained on the `log(1+TARGET_STR)` target, directly implementing a Bayesian approach for linear regression. Probabilistic forecasts are generated by sampling from a normal distribution with the model's point predictions and an empirically estimated, *location-specific* residual standard deviation. The inherent Bayesian regularization in `BayesianRidge`, combined with location-specific fixed effects (one-hot encoded `location`) and explicitly modeled seasonal and spatial dependencies (such as `hhs_ili_lag` features, including the new seasonal lags, and advanced seasonal lags for target and other ILI), aims to capture the key structural elements (seasonality, connectivity, random walk) and provide a more robust and calibrated approximation of a full hierarchical Bayesian framework.
#
# ## Step-by-Step Logic
# 1.  **Initial Data Preparation**:
#    a. Convert `target_end_date` and `reference_date` in all input DataFrames to datetime objects.
#    b. Create `df_train_actual` by merging `train_x` and `train_y`.
#    c. Calculate `max_pop_overall` from `locations['population'].max()`.
#    d. Explicitly merge `locations` data (including `population`) into `df_train_actual` and `test_x`. Drop original `population` columns from `df_train_actual` and `test_x` before merging.
#    e. Define a static `hhs_region_map` dictionary to map `location_name` to `hhs_region_name`. Add `hhs_region_name` to the `locations` DataFrame using this map, and then merge this into `df_train_actual` and `test_x`.
#    f. Apply `np.log1p` transformation to `TARGET_STR` in `df_train_actual` to create `TARGET_STR_LOG`.
#    g. **NEW**: Calculate `week_of_year`, `sin_week`, and `cos_week` for `df_train_actual`.
# 2.  **Historical ILINet Data Preprocessing**:
#    a. Process `ilinet_state` and `ilinet` (national ILI) as described in the original plan.
#    b. Process `ilinet_hhs`: Apply `_align_ilinet_dates`. Rename `region` to `hhs_region_name` and `unweighted_ili` to `hhs_ili`. Select `target_end_date`, `hhs_region_name`, `hhs_ili`. Drop NaNs or duplicates. Calculate `week_of_year` for `ilinet_hhs_processed`.
#    c. Determine the overall `ili_data_cutoff_date`.
#    d. Calculate `week_of_year` for all processed ILI dataframes.
#    e. Compute `location_seasonal_ili_avg_df`, `global_weekly_ili_avg_df`, `overall_ili_mean`, `national_seasonal_ili_avg_df`, `overall_national_ili_mean` as before.
#    f. Compute `hhs_seasonal_ili_avg_df` (mean `hhs_ili` per `hhs_region_name` per `week_of_year`) and `overall_hhs_ili_mean` (mean `hhs_ili` across all history).
# 3.  **Historical Data Augmentation (Strategy 2 - Learn a Transformation)**:
#    a. Create `overlap_df` by merging `df_train_actual` (now with seasonal features) and `ilinet_state_processed`.
#    b. **MODIFIED**: For both global and location-specific `LinearRegression` models, use `['unweighted_ili', 'sin_week', 'cos_week']` as features.
#    c. Store `intercept`, `coef_ili`, `coef_sin`, `coef_cos` for each transformation model. Handle cases with insufficient data by falling back to global model parameters or small defaults.
#    d. Apply this **seasonally-aware transformation** to the entire `ilinet_state_processed` to create `ilinet_synthetic_target_df`, ensuring `sin_week` and `cos_week` are present for the calculation.
# 4.  **Consolidate Full Historical Data (`df_full_history`)**:
#    a. Create `df_synthetic_full` from `ilinet_synthetic_target_df`.
#    b. Merge `national_ili` from `ilinet_national_processed` and `population` from `locations` into `df_synthetic_full` as before.
#    c. Merge `hhs_ili` from `ilinet_hhs_processed` into `df_synthetic_full` using `target_end_date` and `hhs_region_name`.
#    d. Concatenate `df_synthetic_full` and `df_train_actual`, sort, and drop duplicates (keeping `last`). This creates `df_full_history`.
#    e. Define `week_of_year` for `df_full_history`.
#    f. Implement vectorized ILI Imputation for `unweighted_ili`, `national_ili`, and `hhs_ili` as before (hhs-seasonal -> overall hhs mean).
#    g. Drop rows with NaNs in `TARGET_STR_LOG`, `population`, `unweighted_ili`, `national_ili`, or `hhs_ili` from `df_full_history`.
# 5.  **Feature Engineering for Training**:
#    a. Define `create_base_features` helper, applying it to `df_full_history`.
#    b. Vectorize `rest_of_us_target_log` (Population-Weighted) as before.
#    c. Add lagged features to `full_history_features` using `groupby('location').shift(i)`:
#        *   `TARGET_STR_LOG_lag_{i}` for `i` in `[1, ..., LAG_COUNT]`
#        *   `TARGET_STR_LOG_lag_{i}` for `i` in `[51, 52, 53]`
#        *   `ili_lag_{i}` for `i` in `[1, ..., LAG_COUNT]`
#        *   `ili_lag_{i}` for `i` in `[51, 52, 53]`
#        *   `national_ili_lag_{i}` for `i` in `[1, ..., LAG_COUNT]`
#        *   `national_ili_lag_{i}` for `i` in `[51, 52, 53]`
#        *   `hhs_ili_lag_{i}` for `i` in `[1, ..., LAG_COUNT]`
#        *   `hhs_ili_lag_{i}` for `i` in `[51, 52, 53]`
#        *   `rest_of_us_target_log_lag_{i}` for `i` in `[1, ..., LAG_COUNT]`
#    d. Identify all unique `location` identifiers and one-hot encode `location` for `full_history_features`.
#    e. Define the list of `features` to be used by the model, including all original lagged features, the new `TARGET_STR_LOG_lag_51`, `TARGET_STR_LOG_lag_52`, `TARGET_STR_LOG_lag_53`, `ili_lag_51`, `ili_lag_52`, `ili_lag_53`, `national_ili_lag_51`, `national_ili_lag_52`, `national_ili_lag_53`, and the `hhs_ili_lag_51`, `hhs_ili_lag_52`, `hhs_ili_lag_53`.
#    f. Create `X_train` and `y_train` by selecting relevant columns and dropping rows with NaNs resulting from lags. The `lag_subset_cols` for `dropna` will include the largest lag needed, which is `53` for `TARGET_STR_LOG`, `ili`, `national_ili`, and `hhs_ili`.
#    g. Store `X_train_means` for filling NaNs in test features.
# 6.  **Model Training**:
#    a. Train the `BayesianRidge` model.
#    b. Calculate `train_residuals = y_train - model.predict(X_train)`.
#    c. Calculate `residual_std_per_location`: Merge `train_residuals` back to a DataFrame containing `location` for accurate grouping. Group `train_residuals` by `location` and compute the standard deviation. Handle locations with zero or insufficient residuals (e.g., set to a small default, like `0.1`). Store this in a dictionary or Series.
#    d. Define a `global_fallback_std`: Compute an overall `residual_std` for cases where `residual_std_per_location` might not be available or robust for a specific location.
# 7.  **Recursive Prediction and Quantile Generation for `test_x`**:
#    a. Initialize `test_y_hat_quantiles`.
#    b. Prepare `df_test_full` by applying `create_base_features` and one-hot encoding `location`, and merging `hhs_region_name`.
#    c. Sort `df_test_full`.
#    d. **Create a comprehensive `history_imputation_lookup_dict`**:
#        i. Determine the full date range needed for historical lags and future ILI imputations. This will go from the earliest date in `df_full_history` minus 53 weeks, up to the latest `target_end_date` in `test_x`.
#        ii. Create a scaffold of all `(target_end_date, location)` combinations within this range, merging `population` and `hhs_region_name`. **NEW**: Calculate `week_of_year`, `sin_week`, `cos_week` for this scaffold.
#        iii. Merge `df_full_history`'s `target_end_date`, `location`, `TARGET_STR_LOG`, `unweighted_ili`, `national_ili`, `hhs_ili` onto this scaffold. This populates actual/synthetic values where available.
#        iv. For any NaNs in `unweighted_ili`, `national_ili`, `hhs_ili` in the scaffold, impute them using the seasonal/global averages (`location_seasonal_ili_avg_df`, `global_weekly_ili_avg_df`, `national_seasonal_ili_avg_df`, `hhs_seasonal_ili_avg_df`, and overall means).
#        v. For any NaNs in `TARGET_STR_LOG` in the scaffold (e.g., dates before `df_full_history` or in the future test range without actual `TARGET_STR` data), impute them using the `location_transformation_models` (now with `intercept`, `coef_ili`, `coef_sin`, `coef_cos`) and the *imputed* `unweighted_ili`, `sin_week`, `cos_week` from the current scaffold. Clip results at `np.log1p(0)`.
#        vi. Set `(target_end_date, location)` as the index for the scaffold and convert relevant columns (`TARGET_STR_LOG`, `unweighted_ili`, `national_ili`, `hhs_ili`) to a dictionary for fast lookup: `history_imputation_lookup_dict`.
#    e. Initialize `current_lags_store`: A dictionary mapping each `location` to its lists of short-term lagged features, taken from `full_history_features` up to the earliest `reference_date` in `test_x`. Pad with `0.0` if history is shorter. *The long-term lags (51, 52, 53) will be looked up from `history_imputation_lookup_dict`.*
#    f. Iterate `horizon_idx` from 0 up to `max_horizon`:
#        i.  Initialize `horizon_predictions_log`, `horizon_populations`, `horizon_unweighted_ili_imputed`, `horizon_national_ili_imputed`, `horizon_hhs_ili_imputed` dictionaries for the current horizon.
#        ii. For each `location` in the current horizon:
#            1.  Populate all short-term lagged features in `current_test_row_features` using values from `current_lags_store[loc]`.
#            2.  **Simplified dynamic computation of seasonal lags (51, 52, 53)**: For each `s_lag` in `[51, 52, 53]`:
#                *   Calculate `lag_s_date = t_end_date - pd.Timedelta(weeks=s_lag)`.
#                *   Lookup the `TARGET_STR_LOG`, `unweighted_ili`, `national_ili`, `hhs_ili` values for `(lag_s_date, loc)` from `history_imputation_lookup_dict`.
#                *   Assign these directly to `current_test_row_features` with the appropriate lagged column names (`TARGET_STR_LOG_lag_{s_lag}`, `ili_lag_{s_lag}`, `national_ili_lag_{s_lag}`, `hhs_ili_lag_{s_lag}`).
#                *   Use sensible default fallbacks (e.g., `np.log1p(0)` for `TARGET_STR_LOG`, and respective overall means or `0.0` for ILI values) if a specific `(lag_s_date, loc)` is not found in the lookup.
#            3.  Retrieve current step's imputed `unweighted_ili`, `national_ili`, `hhs_ili` for `(t_end_date, loc)` from `history_imputation_lookup_dict`. Store these in `horizon_*_imputed` dictionaries.
#            4.  Populate other base features. Fill any remaining NaNs in the feature vector using `X_train_means`.
#            5.  Clip `pred_mean_log` to ensure numerical stability.
#            6.  Make point prediction (`pred_mean_log`). Store it and `population` in `horizon_*` dictionaries.
#            7.  Retrieve `residual_std` for the current location.
#            8.  Generate samples, inverse transform, calculate quantiles. Enforce monotonic quantiles *before* rounding to integer. Store in `test_y_hat_quantiles`.
#        iii. After predicting for all locations at `horizon_idx`:
#            1.  Create a temporary DataFrame from `horizon_predictions_log` and `horizon_populations` for vectorized calculations.
#            2.  Vectorize calculation of `rest_of_us_target_log` for all locations in this horizon.
#            3.  For each `loc` in `all_forecast_locations`:
#                a. Update `current_lags_store[loc]['target_log_lags']`, `current_lags_store[loc]['ili_lags']`, `current_lags_store[loc]['national_ili_lags']`, `current_lags_store[loc]['hhs_ili_lags']`, and `current_lags_store[loc]['rest_of_us_target_log_lags']` by popping the oldest and appending the respective predicted or imputed values for that location. (Short-term lags update as before).
# 8.  Convert the final `test_y_hat_quantiles` DataFrame to integer types.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # 1. Initial Data Preparation
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(
      test_x['reference_date']
  )  # Ensure reference_date is datetime

  df_train_actual = train_x.set_index(train_y.index).copy()
  df_train_actual[TARGET_STR] = train_y

  # OPTIMIZATION: Calculate max_pop_overall from the global locations DataFrame
  max_pop_overall = locations['population'].max()

  # Create a copy of locations to add HHS region mapping
  locations_extended = locations.copy()
  locations_extended['hhs_region_name'] = locations_extended[
      'location_name'
  ].map(HHS_REGION_MAP)

  # Explicitly remove original population before merging from locations
  if 'population' in df_train_actual.columns:
    df_train_actual = df_train_actual.drop(columns=['population'])
  if 'population' in test_x.columns:
    test_x = test_x.drop(columns=['population'])

  # Ensure population and hhs_region_name are correctly sourced from locations_extended
  df_train_actual = pd.merge(
      df_train_actual,
      locations_extended[[
          'location',
          'abbreviation',
          'location_name',
          'population',
          'hhs_region_name',
      ]],
      on='location',
      how='left',
  )
  test_x = pd.merge(
      test_x,
      locations_extended[[
          'location',
          'abbreviation',
          'location_name',
          'population',
          'hhs_region_name',
      ]],
      on='location',
      how='left',
  )

  df_train_actual[TARGET_STR_LOG] = np.log1p(df_train_actual[TARGET_STR])

  # NEW: Calculate week_of_year, sin_week, cos_week for df_train_actual
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df_train_actual['week_of_year'] = (
        df_train_actual['target_end_date'].dt.isocalendar().week.astype(int)
    )
  df_train_actual['sin_week'] = np.sin(
      2 * np.pi * df_train_actual['week_of_year'] / 52
  )
  df_train_actual['cos_week'] = np.cos(
      2 * np.pi * df_train_actual['week_of_year'] / 52
  )

  # 2. Historical ILINet Data Preprocessing
  ilinet_state_processed = _align_ilinet_dates(ilinet_state)
  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_processed = ilinet_state_processed.merge(
      locations_extended[['location', 'location_name']],
      on='location_name',
      how='left',
  )
  ilinet_state_processed = (
      ilinet_state_processed[['target_end_date', 'location', 'unweighted_ili']]
      .dropna(subset=['unweighted_ili', 'location'])
      .drop_duplicates(subset=['target_end_date', 'location'])
  )

  ilinet_national_processed = _align_ilinet_dates(ilinet)
  ilinet_national_processed = ilinet_national_processed[
      ilinet_national_processed['region_type'] == 'National'
  ]
  ilinet_national_processed = ilinet_national_processed.rename(
      columns={'unweighted_ili': 'national_ili'}
  )
  ilinet_national_processed = (
      ilinet_national_processed[['target_end_date', 'national_ili']]
      .dropna()
      .drop_duplicates(subset=['target_end_date'])
  )

  # NEW: Process ilinet_hhs
  ilinet_hhs_processed = _align_ilinet_dates(ilinet_hhs)
  ilinet_hhs_processed = ilinet_hhs_processed.rename(
      columns={'region': 'hhs_region_name', 'unweighted_ili': 'hhs_ili'}
  )
  ilinet_hhs_processed = (
      ilinet_hhs_processed[['target_end_date', 'hhs_region_name', 'hhs_ili']]
      .dropna()
      .drop_duplicates(subset=['target_end_date', 'hhs_region_name'])
  )

  ili_data_cutoff_date_state = ilinet_state_processed['target_end_date'].max()
  ili_data_cutoff_date_national = ilinet_national_processed[
      'target_end_date'
  ].max()
  ili_data_cutoff_date_hhs = ilinet_hhs_processed[
      'target_end_date'
  ].max()  # NEW
  ili_data_cutoff_date = min(
      ili_data_cutoff_date_state,
      ili_data_cutoff_date_national,
      ili_data_cutoff_date_hhs,
  )  # Overall ILI cutoff

  # Calculate seasonal averages from processed ILI dataframes
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    ilinet_state_processed['week_of_year'] = (
        ilinet_state_processed['target_end_date']
        .dt.isocalendar()
        .week.astype(int)
    )
    ilinet_national_processed['week_of_year'] = (
        ilinet_national_processed['target_end_date']
        .dt.isocalendar()
        .week.astype(int)
    )
    ilinet_hhs_processed['week_of_year'] = (
        ilinet_hhs_processed['target_end_date']
        .dt.isocalendar()
        .week.astype(int)
    )  # NEW

  # NEW: Add sin_week and cos_week to ilinet_state_processed
  ilinet_state_processed['sin_week'] = np.sin(
      2 * np.pi * ilinet_state_processed['week_of_year'] / 52
  )
  ilinet_state_processed['cos_week'] = np.cos(
      2 * np.pi * ilinet_state_processed['week_of_year'] / 52
  )

  location_seasonal_ili_avg_df = (
      ilinet_state_processed.groupby(['location', 'week_of_year'])[
          'unweighted_ili'
      ]
      .mean()
      .reset_index(name='unweighted_ili_seasonal_avg')
  )
  global_weekly_ili_avg_df = (
      ilinet_state_processed.groupby('week_of_year')['unweighted_ili']
      .mean()
      .reset_index(name='unweighted_ili_global_weekly_avg')
  )
  overall_ili_mean = ilinet_state_processed['unweighted_ili'].mean()

  national_seasonal_ili_avg_df = (
      ilinet_national_processed.groupby('week_of_year')['national_ili']
      .mean()
      .reset_index(name='national_ili_seasonal_avg')
  )
  overall_national_ili_mean = ilinet_national_processed['national_ili'].mean()

  # NEW: HHS seasonal averages
  hhs_seasonal_ili_avg_df = (
      ilinet_hhs_processed.groupby(['hhs_region_name', 'week_of_year'])[
          'hhs_ili'
      ]
      .mean()
      .reset_index(name='hhs_ili_seasonal_avg')
  )
  overall_hhs_ili_mean = ilinet_hhs_processed['hhs_ili'].mean()

  # 3. Strategy 2: Learn Transformation (ILI to log Admissions)
  # MODIFIED: Merge sin_week and cos_week from df_train_actual
  overlap_df = pd.merge(
      df_train_actual[[
          'target_end_date',
          'location',
          TARGET_STR_LOG,
          'sin_week',
          'cos_week',
      ]],
      ilinet_state_processed[['target_end_date', 'location', 'unweighted_ili']],
      on=['target_end_date', 'location'],
      how='inner',
  )
  overlap_df = overlap_df.dropna(
      subset=[TARGET_STR_LOG, 'unweighted_ili', 'sin_week', 'cos_week']
  )  # Ensure all features are present

  location_transformation_models = {}

  # MODIFIED: Global transformation now includes seasonal features
  global_intercept = 0.0
  global_coef_ili = 0.01  # Default to a small positive coef if no data
  global_coef_sin = 0.0
  global_coef_cos = 0.0

  transformation_features = ['unweighted_ili', 'sin_week', 'cos_week']

  if (
      not overlap_df.empty
      and (overlap_df[transformation_features].nunique() > 1).any()
  ):
    global_model = LinearRegression(n_jobs=-1)
    try:
      global_model.fit(
          overlap_df[transformation_features], overlap_df[TARGET_STR_LOG]
      )
      global_intercept = global_model.intercept_
      global_coef_ili = global_model.coef_[0]
      global_coef_sin = global_model.coef_[1]
      global_coef_cos = global_model.coef_[2]
      if global_coef_ili < 0:
        global_coef_ili = 0.01  # Ensure positive ILI relationship
    except ValueError:
      warnings.warn(
          'Failed to fit global LinearRegression for ILI transformation. Using'
          ' default parameters.',
          UserWarning,
      )
  else:
    warnings.warn(
        'No sufficient overlap data to fit global ILI transformation. Using'
        ' default parameters.',
        UserWarning,
    )

  for loc in overlap_df['location'].unique():
    loc_overlap_df = overlap_df[overlap_df['location'] == loc]
    loc_X = loc_overlap_df[transformation_features]
    loc_y = loc_overlap_df[TARGET_STR_LOG]

    if (
        len(loc_overlap_df) > len(transformation_features)
        and (loc_X.nunique() > 1).any()
    ):
      try:
        loc_model = LinearRegression()
        loc_model.fit(loc_X, loc_y)
        loc_coef_ili = loc_model.coef_[0]
        if loc_coef_ili < 0:
          loc_coef_ili = 0.01
        location_transformation_models[loc] = {
            'intercept': loc_model.intercept_,
            'coef_ili': loc_coef_ili,
            'coef_sin': loc_model.coef_[1],
            'coef_cos': loc_model.coef_[2],
        }
      except ValueError:
        warnings.warn(
            f'Failed to fit LinearRegression for location {loc} during ILI'
            ' transformation. Using global model.',
            UserWarning,
        )
        location_transformation_models[loc] = {
            'intercept': global_intercept,
            'coef_ili': global_coef_ili,
            'coef_sin': global_coef_sin,
            'coef_cos': global_coef_cos,
        }
    else:
      warnings.warn(
          f'Insufficient data/variance for location {loc} during ILI'
          ' transformation. Using global model.',
          UserWarning,
      )
      location_transformation_models[loc] = {
          'intercept': global_intercept,
          'coef_ili': global_coef_ili,
          'coef_sin': global_coef_sin,
          'coef_cos': global_coef_cos,
      }

  # Ensure all locations in `ilinet_state_processed` have a transformation model defined
  for loc in ilinet_state_processed['location'].unique():
    if loc not in location_transformation_models:
      location_transformation_models[loc] = {
          'intercept': global_intercept,
          'coef_ili': global_coef_ili,
          'coef_sin': global_coef_sin,
          'coef_cos': global_coef_cos,
      }

  # Apply transformation to the entire `ilinet_state_processed` to create synthetic log target (VECTORIZED)
  ilinet_synthetic_target_df = ilinet_state_processed.copy()

  # Create Series for intercepts and coefficients, mapped by location
  intercepts = pd.Series(
      {
          loc: data['intercept']
          for loc, data in location_transformation_models.items()
      },
      name='intercept',
  )
  coefs_ili = pd.Series(
      {
          loc: data['coef_ili']
          for loc, data in location_transformation_models.items()
      },
      name='coef_ili',
  )
  coefs_sin = pd.Series(
      {
          loc: data['coef_sin']
          for loc, data in location_transformation_models.items()
      },
      name='coef_sin',
  )
  coefs_cos = pd.Series(
      {
          loc: data['coef_cos']
          for loc, data in location_transformation_models.items()
      },
      name='coef_cos',
  )

  # Merge intercepts and coefficients onto the dataframe
  ilinet_synthetic_target_df = ilinet_synthetic_target_df.merge(
      intercepts.rename('intercept'),
      left_on='location',
      right_index=True,
      how='left',
  )
  ilinet_synthetic_target_df = ilinet_synthetic_target_df.merge(
      coefs_ili.rename('coef_ili'),
      left_on='location',
      right_index=True,
      how='left',
  )
  ilinet_synthetic_target_df = ilinet_synthetic_target_df.merge(
      coefs_sin.rename('coef_sin'),
      left_on='location',
      right_index=True,
      how='left',
  )
  ilinet_synthetic_target_df = ilinet_synthetic_target_df.merge(
      coefs_cos.rename('coef_cos'),
      left_on='location',
      right_index=True,
      how='left',
  )

  # Apply transformation using vectorized operations
  ilinet_synthetic_target_df[TARGET_STR_LOG] = (
      ilinet_synthetic_target_df['intercept']
      + ilinet_synthetic_target_df['coef_ili']
      * ilinet_synthetic_target_df['unweighted_ili']
      + ilinet_synthetic_target_df['coef_sin']
      * ilinet_synthetic_target_df['sin_week']
      + ilinet_synthetic_target_df['coef_cos']
      * ilinet_synthetic_target_df['cos_week']
  ).clip(lower=0)

  # Drop temporary columns
  ilinet_synthetic_target_df = ilinet_synthetic_target_df.drop(
      columns=['intercept', 'coef_ili', 'coef_sin', 'coef_cos']
  )

  # 4. Consolidate Full Historical Data (`df_full_history`)
  # Drop sin_week/cos_week from ilinet_synthetic_target_df before concat, they are re-calculated later in create_base_features
  df_synthetic_full = ilinet_synthetic_target_df[
      ['target_end_date', 'location', 'unweighted_ili', TARGET_STR_LOG]
  ].copy()

  # Merge national ILI into synthetic full history
  df_synthetic_full = pd.merge(
      df_synthetic_full,
      ilinet_national_processed[
          ['target_end_date', 'national_ili', 'week_of_year']
      ],
      on='target_end_date',
      how='left',
  )

  # Merge population from locations and hhs_region_name
  df_synthetic_full = pd.merge(
      df_synthetic_full,
      locations_extended[['location', 'population', 'hhs_region_name']],
      on='location',
      how='left',
  )

  # NEW: Merge HHS ILI into synthetic full history
  df_synthetic_full = pd.merge(
      df_synthetic_full,
      ilinet_hhs_processed[
          ['target_end_date', 'hhs_region_name', 'hhs_ili', 'week_of_year']
      ].drop(columns=['week_of_year']),
      on=['target_end_date', 'hhs_region_name'],
      how='left',
  )

  # Concatenate synthetic data with real data, prioritizing real observations
  df_full_history = pd.concat(
      [
          df_synthetic_full.drop(
              columns=['week_of_year'], errors='ignore'
          ),  # Drop temp woy from synthetic before concat
          df_train_actual[[
              'target_end_date',
              'location',
              'population',
              'hhs_region_name',
              TARGET_STR_LOG,
          ]],
      ],
      ignore_index=True,
  )

  df_full_history = (
      pd.DataFrame(df_full_history)
      .sort_values(by=['location', 'target_end_date'])
      .drop_duplicates(subset=['target_end_date', 'location'], keep='last')
  )

  # Add week_of_year to df_full_history for imputation and feature engineering
  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df_full_history['week_of_year'] = (
        df_full_history['target_end_date'].dt.isocalendar().week.astype(int)
    )

  # OPTIMIZATION: Vectorized ILI imputation for df_full_history
  # Merge seasonal averages
  df_full_history = pd.merge(
      df_full_history,
      location_seasonal_ili_avg_df,
      on=['location', 'week_of_year'],
      how='left',
  )
  df_full_history = pd.merge(
      df_full_history, global_weekly_ili_avg_df, on='week_of_year', how='left'
  )
  df_full_history = pd.merge(
      df_full_history,
      national_seasonal_ili_avg_df,
      on='week_of_year',
      how='left',
  )
  df_full_history = pd.merge(
      df_full_history,
      hhs_seasonal_ili_avg_df,
      on=['hhs_region_name', 'week_of_year'],
      how='left',
  )  # NEW

  # Impute unweighted_ili
  df_full_history['unweighted_ili'] = df_full_history['unweighted_ili'].fillna(
      df_full_history['unweighted_ili_seasonal_avg']
  )
  df_full_history['unweighted_ili'] = df_full_history['unweighted_ili'].fillna(
      df_full_history['unweighted_ili_global_weekly_avg']
  )
  df_full_history['unweighted_ili'] = df_full_history['unweighted_ili'].fillna(
      overall_ili_mean
  )

  # Impute national_ili
  df_full_history['national_ili'] = df_full_history['national_ili'].fillna(
      df_full_history['national_ili_seasonal_avg']
  )
  df_full_history['national_ili'] = df_full_history['national_ili'].fillna(
      overall_national_ili_mean
  )  # Fallback to national mean

  # NEW: Impute hhs_ili
  df_full_history['hhs_ili'] = df_full_history['hhs_ili'].fillna(
      df_full_history['hhs_ili_seasonal_avg']
  )
  df_full_history['hhs_ili'] = df_full_history['hhs_ili'].fillna(
      overall_hhs_ili_mean
  )

  # Drop temporary average columns
  df_full_history = df_full_history.drop(
      columns=[
          'unweighted_ili_seasonal_avg',
          'unweighted_ili_global_weekly_avg',
          'national_ili_seasonal_avg',
          'hhs_ili_seasonal_avg',  # NEW
      ],
      errors='ignore',
  )

  # Final check for any NaNs and drop
  df_full_history = df_full_history.dropna(
      subset=[
          TARGET_STR_LOG,
          'population',
          'unweighted_ili',
          'national_ili',
          'hhs_ili',
      ]
  )

  # 5. Feature Engineering for Training
  # Calculate min_target_end_date for days_since_start feature (New Trend Feature)
  min_target_end_date = df_full_history['target_end_date'].min()

  def create_base_features(
      df, max_pop_val, min_target_date
  ):
    df_out = df.copy()
    with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=FutureWarning)
      df_out['week_of_year'] = (
          df_out['target_end_date'].dt.isocalendar().week.astype(int)
      )
    df_out['sin_week'] = np.sin(2 * np.pi * df_out['week_of_year'] / 52)
    df_out['cos_week'] = np.cos(2 * np.pi * df_out['week_of_year'] / 52)
    df_out['population_scaled'] = df_out['population'] / max_pop_val
    # NEW: Add days_since_start to capture general temporal trends
    df_out['days_since_start'] = (
        df_out['target_end_date'] - min_target_date
    ).dt.days
    return df_out

  full_history_features = create_base_features(
      df_full_history, max_pop_overall, min_target_end_date
  )

  # OPTIMIZATION: Vectorize rest_of_us_target_log calculation (Population-Weighted)
  # Calculate total weighted log admissions and total population per target_end_date
  full_history_features['weighted_log_admissions'] = (
      full_history_features[TARGET_STR_LOG]
      * full_history_features['population']
  )
  full_history_features['total_weighted_log_admissions_by_date'] = (
      full_history_features.groupby('target_end_date')[
          'weighted_log_admissions'
      ].transform('sum')
  )
  full_history_features['total_population_by_date'] = (
      full_history_features.groupby('target_end_date')['population'].transform(
          'sum'
      )
  )

  # Calculate weighted sum and population for 'other' locations
  full_history_features['other_loc_weighted_log_admissions_sum'] = (
      full_history_features['total_weighted_log_admissions_by_date']
      - full_history_features['weighted_log_admissions']
  )
  full_history_features['other_loc_population_sum'] = (
      full_history_features['total_population_by_date']
      - full_history_features['population']
  )

  # Compute population-weighted average, handle division by zero
  full_history_features['rest_of_us_target_log'] = (
      full_history_features['other_loc_weighted_log_admissions_sum']
      / full_history_features['other_loc_population_sum']
  )
  full_history_features.loc[
      full_history_features['other_loc_population_sum'] == 0,
      'rest_of_us_target_log',
  ] = 0.0

  full_history_features = full_history_features.drop(
      columns=[
          'weighted_log_admissions',
          'total_weighted_log_admissions_by_date',
          'total_population_by_date',
          'other_loc_weighted_log_admissions_sum',
          'other_loc_population_sum',
      ]
  )

  # Create lagged features for training data (LAG_COUNT increased to 6, plus seasonal lags)
  full_history_features = full_history_features.sort_values(
      by=['location', 'target_end_date']
  ).copy()
  for i in range(1, LAG_COUNT + 1):
    full_history_features[f'{TARGET_STR_LOG}_lag_{i}'] = (
        full_history_features.groupby('location')[TARGET_STR_LOG].shift(i)
    )
    full_history_features[f'ili_lag_{i}'] = full_history_features.groupby(
        'location'
    )['unweighted_ili'].shift(i)
    full_history_features[f'national_ili_lag_{i}'] = (
        full_history_features.groupby('location')['national_ili'].shift(i)
    )
    full_history_features[f'hhs_ili_lag_{i}'] = full_history_features.groupby(
        'location'
    )['hhs_ili'].shift(i)
    full_history_features[f'rest_of_us_target_log_lag_{i}'] = (
        full_history_features.groupby('location')[
            'rest_of_us_target_log'
        ].shift(i)
    )

  # MODIFIED: Add seasonal lags 51, 52, 53 for ILI features, HHS ILI, and TARGET_STR_LOG
  for s_lag in [51, 52, 53]:
    full_history_features[f'ili_lag_{s_lag}'] = full_history_features.groupby(
        'location'
    )['unweighted_ili'].shift(s_lag)
    full_history_features[f'national_ili_lag_{s_lag}'] = (
        full_history_features.groupby('location')['national_ili'].shift(s_lag)
    )
    full_history_features[
        f'hhs_ili_lag_{s_lag}'
    ] = full_history_features.groupby('location')['hhs_ili'].shift(
        s_lag
    )  # NEW FEATURE
    full_history_features[f'{TARGET_STR_LOG}_lag_{s_lag}'] = (
        full_history_features.groupby('location')[TARGET_STR_LOG].shift(s_lag)
    )

  # One-hot encode locations
  all_locations = pd.concat(
      [full_history_features['location'], test_x['location']]
  ).unique()
  dummy_cols = [f'location_{loc}' for loc in all_locations]

  location_dummies_train = pd.get_dummies(
      full_history_features['location'], prefix='location_'
  )
  # Reindex to ensure all dummy columns are present, even if a location isn't in train
  location_dummies_train = location_dummies_train.reindex(
      columns=dummy_cols, fill_value=0
  )
  full_history_features = pd.concat(
      [full_history_features, location_dummies_train], axis=1
  )

  features = ['population_scaled', 'sin_week', 'cos_week', 'days_since_start']
  for i in range(1, LAG_COUNT + 1):
    features.append(f'{TARGET_STR_LOG}_lag_{i}')
    features.append(f'ili_lag_{i}')
    features.append(f'national_ili_lag_{i}')
    features.append(f'hhs_ili_lag_{i}')
    features.append(f'rest_of_us_target_log_lag_{i}')

  # MODIFIED: Add new seasonal lags to features list
  for s_lag in [51, 52, 53]:
    features.append(f'ili_lag_{s_lag}')
    features.append(f'national_ili_lag_{s_lag}')
    features.append(f'hhs_ili_lag_{s_lag}')  # NEW FEATURE
    features.append(f'{TARGET_STR_LOG}_lag_{s_lag}')

  features.extend([col for col in dummy_cols])

  # Drop rows with NaN lags for training
  lag_subset_cols = [
      f'{TARGET_STR_LOG}_lag_{LAG_COUNT}',
      f'ili_lag_{LAG_COUNT}',
      f'national_ili_lag_{LAG_COUNT}',
      f'hhs_ili_lag_{LAG_COUNT}',
      f'rest_of_us_target_log_lag_{LAG_COUNT}',
  ]
  # MODIFIED: Ensure new seasonal lags are included for dropna
  lag_subset_cols.extend([
      f'ili_lag_53',
      f'national_ili_lag_53',
      f'hhs_ili_lag_53',
      f'{TARGET_STR_LOG}_lag_53',
  ])  # Ensure longest lag for all key features

  # IMPROVEMENT: Perform dropna once for efficiency
  df_filtered_for_training = full_history_features.dropna(
      subset=lag_subset_cols
  )
  X_train = df_filtered_for_training[features]
  y_train = df_filtered_for_training[TARGET_STR_LOG]

  # Store means for filling NaNs in test features
  X_train_means = X_train.mean()

  # 6. Model Training
  # MODIFIED: Changed LinearRegression to BayesianRidge
  model = BayesianRidge()
  model.fit(X_train, y_train)

  # Calculate train residuals
  train_predictions = model.predict(X_train)
  train_residuals = y_train - train_predictions

  # Calculate location-specific residual standard deviations
  train_residuals_df = pd.DataFrame({
      'residuals': train_residuals,
      'location': full_history_features.loc[X_train.index, 'location'],
  })
  residual_std_per_location = train_residuals_df.groupby('location')[
      'residuals'
  ].std()

  # Handle locations with insufficient data or zero std
  min_residual_std = 0.1  # Minimum floor for std
  residual_std_per_location = residual_std_per_location.fillna(
      min_residual_std
  ).replace(0, min_residual_std)

  # global_fallback_std is now the mean of the *processed* location-specific STDs
  global_fallback_std = residual_std_per_location.mean()

  # 7. Recursive Prediction and Quantile Generation for Test Data
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  df_test_full = create_base_features(
      test_x, max_pop_overall, min_target_end_date
  )

  location_dummies_test = pd.get_dummies(
      df_test_full['location'], prefix='location_'
  )
  location_dummies_test = location_dummies_test.reindex(
      columns=dummy_cols, fill_value=0
  )
  df_test_full = pd.concat([df_test_full, location_dummies_test], axis=1)

  df_test_full = df_test_full.sort_values(by=['location', 'horizon'])

  all_forecast_locations = df_test_full['location'].unique()
  earliest_test_reference_date = df_test_full['reference_date'].min()
  max_horizon = df_test_full['horizon'].max()

  # NEW: Create a comprehensive history and imputation lookup
  min_date_for_lookup = min(
      df_full_history['target_end_date'].min()
      - pd.Timedelta(weeks=53),  # For earliest lags
      df_test_full['target_end_date'].min() - pd.Timedelta(weeks=53),
  )
  max_date_for_lookup = df_test_full['target_end_date'].max()

  all_dates_in_lookup_range = pd.date_range(
      min_date_for_lookup, max_date_for_lookup, freq='W-SAT'
  )

  full_lookup_scaffold = pd.MultiIndex.from_product(
      [all_dates_in_lookup_range, all_forecast_locations],
      names=['target_end_date', 'location'],
  ).to_frame(index=False)

  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      locations_extended[['location', 'population', 'hhs_region_name']],
      on='location',
      how='left',
  )

  with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    full_lookup_scaffold['week_of_year'] = (
        full_lookup_scaffold['target_end_date']
        .dt.isocalendar()
        .week.astype(int)
    )
  # NEW: Add sin_week and cos_week to lookup scaffold for transformation
  full_lookup_scaffold['sin_week'] = np.sin(
      2 * np.pi * full_lookup_scaffold['week_of_year'] / 52
  )
  full_lookup_scaffold['cos_week'] = np.cos(
      2 * np.pi * full_lookup_scaffold['week_of_year'] / 52
  )

  # Merge df_full_history onto the scaffold to get actual/synthetic values
  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      df_full_history[[
          'target_end_date',
          'location',
          TARGET_STR_LOG,
          'unweighted_ili',
          'national_ili',
          'hhs_ili',
      ]],
      on=['target_end_date', 'location'],
      how='left',
  )

  # Merge seasonal averages onto the scaffold for imputation
  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      location_seasonal_ili_avg_df,
      on=['location', 'week_of_year'],
      how='left',
  )
  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      global_weekly_ili_avg_df,
      on='week_of_year',
      how='left',
  )
  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      national_seasonal_ili_avg_df,
      on='week_of_year',
      how='left',
  )
  full_lookup_scaffold = pd.merge(
      full_lookup_scaffold,
      hhs_seasonal_ili_avg_df,
      on=['hhs_region_name', 'week_of_year'],
      how='left',
  )

  # Hierarchical fill for unweighted_ili
  full_lookup_scaffold['unweighted_ili'] = full_lookup_scaffold[
      'unweighted_ili'
  ].fillna(full_lookup_scaffold['unweighted_ili_seasonal_avg'])
  full_lookup_scaffold['unweighted_ili'] = full_lookup_scaffold[
      'unweighted_ili'
  ].fillna(full_lookup_scaffold['unweighted_ili_global_weekly_avg'])
  full_lookup_scaffold['unweighted_ili'] = full_lookup_scaffold[
      'unweighted_ili'
  ].fillna(overall_ili_mean)

  # Hierarchical fill for national_ili
  full_lookup_scaffold['national_ili'] = full_lookup_scaffold[
      'national_ili'
  ].fillna(full_lookup_scaffold['national_ili_seasonal_avg'])
  full_lookup_scaffold['national_ili'] = full_lookup_scaffold[
      'national_ili'
  ].fillna(overall_national_ili_mean)

  # Hierarchical fill for hhs_ili
  full_lookup_scaffold['hhs_ili'] = full_lookup_scaffold['hhs_ili'].fillna(
      full_lookup_scaffold['hhs_ili_seasonal_avg']
  )
  full_lookup_scaffold['hhs_ili'] = full_lookup_scaffold['hhs_ili'].fillna(
      overall_hhs_ili_mean
  )

  # MODIFIED: Impute TARGET_STR_LOG using the seasonally-aware transformation model and imputed ILI/seasonal features
  # First, ensure transformation params are available on the scaffold
  intercepts_series = pd.Series({
      loc: data['intercept']
      for loc, data in location_transformation_models.items()
  })
  coefs_ili_series = pd.Series({
      loc: data['coef_ili']
      for loc, data in location_transformation_models.items()
  })
  coefs_sin_series = pd.Series({
      loc: data['coef_sin']
      for loc, data in location_transformation_models.items()
  })
  coefs_cos_series = pd.Series({
      loc: data['coef_cos']
      for loc, data in location_transformation_models.items()
  })

  full_lookup_scaffold['transform_intercept'] = (
      full_lookup_scaffold['location']
      .map(intercepts_series)
      .fillna(global_intercept)
  )
  full_lookup_scaffold['transform_coef_ili'] = (
      full_lookup_scaffold['location']
      .map(coefs_ili_series)
      .fillna(global_coef_ili)
  )
  full_lookup_scaffold['transform_coef_sin'] = (
      full_lookup_scaffold['location']
      .map(coefs_sin_series)
      .fillna(global_coef_sin)
  )
  full_lookup_scaffold['transform_coef_cos'] = (
      full_lookup_scaffold['location']
      .map(coefs_cos_series)
      .fillna(global_coef_cos)
  )

  missing_target_log_mask = full_lookup_scaffold[TARGET_STR_LOG].isna()
  full_lookup_scaffold.loc[missing_target_log_mask, TARGET_STR_LOG] = (
      full_lookup_scaffold.loc[missing_target_log_mask, 'transform_intercept']
      + full_lookup_scaffold.loc[missing_target_log_mask, 'transform_coef_ili']
      * full_lookup_scaffold.loc[missing_target_log_mask, 'unweighted_ili']
      + full_lookup_scaffold.loc[missing_target_log_mask, 'transform_coef_sin']
      * full_lookup_scaffold.loc[missing_target_log_mask, 'sin_week']
      + full_lookup_scaffold.loc[missing_target_log_mask, 'transform_coef_cos']
      * full_lookup_scaffold.loc[missing_target_log_mask, 'cos_week']
  ).clip(lower=np.log1p(0))

  # Drop temporary columns and set multi-index for lookup
  full_lookup_scaffold = full_lookup_scaffold.drop(
      columns=[
          'week_of_year',
          'population',
          'hhs_region_name',
          'unweighted_ili_seasonal_avg',
          'unweighted_ili_global_weekly_avg',
          'national_ili_seasonal_avg',
          'hhs_ili_seasonal_avg',
          'transform_intercept',
          'transform_coef_ili',
          'transform_coef_sin',
          'transform_coef_cos',
          'sin_week',
          (  # Drop these as they are not stored in the lookup dict values, but recalculated from date
              'cos_week'
          ),
      ]
  )
  full_lookup_scaffold = full_lookup_scaffold.set_index(
      ['target_end_date', 'location']
  )
  history_imputation_lookup_dict = full_lookup_scaffold.to_dict('index')

  # Initialize current_lags_store for all relevant locations (no lag_51/52/53 lists here)
  current_lags_store = {}
  for loc in all_forecast_locations:
    loc_history = full_history_features[
        (full_history_features['location'] == loc)
        & (
            full_history_features['target_end_date']
            < earliest_test_reference_date
        )
    ].sort_values('target_end_date')

    # Pad with zeros if history is shorter than LAG_COUNT
    current_lags_store[loc] = {
        'target_log_lags': (
            np.pad(
                loc_history[TARGET_STR_LOG].tail(LAG_COUNT).values,
                (
                    LAG_COUNT
                    - len(loc_history[TARGET_STR_LOG].tail(LAG_COUNT)),
                    0,
                ),
                'constant',
                constant_values=0.0,
            ).tolist()
        ),
        'ili_lags': (
            np.pad(
                loc_history['unweighted_ili'].tail(LAG_COUNT).values,
                (
                    LAG_COUNT
                    - len(loc_history['unweighted_ili'].tail(LAG_COUNT)),
                    0,
                ),
                'constant',
                constant_values=0.0,
            ).tolist()
        ),
        'national_ili_lags': (
            np.pad(
                loc_history['national_ili'].tail(LAG_COUNT).values,
                (
                    LAG_COUNT
                    - len(loc_history['national_ili'].tail(LAG_COUNT)),
                    0,
                ),
                'constant',
                constant_values=0.0,
            ).tolist()
        ),
        'hhs_ili_lags': (
            np.pad(
                loc_history['hhs_ili'].tail(LAG_COUNT).values,
                (LAG_COUNT - len(loc_history['hhs_ili'].tail(LAG_COUNT)), 0),
                'constant',
                constant_values=0.0,
            ).tolist()
        ),
        'rest_of_us_target_log_lags': (
            np.pad(
                loc_history['rest_of_us_target_log'].tail(LAG_COUNT).values,
                (
                    LAG_COUNT
                    - len(loc_history['rest_of_us_target_log'].tail(LAG_COUNT)),
                    0,
                ),
                'constant',
                constant_values=0.0,
            ).tolist()
        ),
    }

  for horizon_idx in range(max_horizon + 1):
    horizon_rows_to_predict = df_test_full[
        df_test_full['horizon'] == horizon_idx
    ].copy()

    if horizon_rows_to_predict.empty:
      continue

    horizon_predictions_log = (
        {}
    )  # To store point predictions for this horizon for all locations
    horizon_populations = (
        {}
    )  # To store populations for this horizon for all locations
    horizon_unweighted_ili_imputed = {}
    horizon_national_ili_imputed = {}
    horizon_hhs_ili_imputed = {}

    # Sort by location to process consistently, although order within a horizon doesn't strictly matter for final output,
    # it helps for reproducible intermediate states.
    for original_idx, row in horizon_rows_to_predict.sort_values(
        'location'
    ).iterrows():
      loc = row['location']
      t_end_date = row[
          'target_end_date'
      ]  # Get target_end_date for current prediction

      current_test_row_features = pd.DataFrame(
          [row]
      )  # Convert series row to DataFrame

      # Populate short-term lag features using current_lags_store
      for i in range(1, LAG_COUNT + 1):
        current_test_row_features[f'{TARGET_STR_LOG}_lag_{i}'] = (
            current_lags_store[loc]['target_log_lags'][LAG_COUNT - i]
        )
        current_test_row_features[f'ili_lag_{i}'] = current_lags_store[loc][
            'ili_lags'
        ][LAG_COUNT - i]
        current_test_row_features[f'national_ili_lag_{i}'] = current_lags_store[
            loc
        ]['national_ili_lags'][LAG_COUNT - i]
        current_test_row_features[f'hhs_ili_lag_{i}'] = current_lags_store[loc][
            'hhs_ili_lags'
        ][LAG_COUNT - i]
        current_test_row_features[f'rest_of_us_target_log_lag_{i}'] = (
            current_lags_store[loc]['rest_of_us_target_log_lags'][LAG_COUNT - i]
        )

      # Dynamically compute lag_51, 52, 53 features by looking up from history_imputation_lookup_dict
      for s_lag in [51, 52, 53]:
        lag_s_date = t_end_date - pd.Timedelta(weeks=s_lag)

        # Get all relevant values from the comprehensive lookup
        lookup_values = history_imputation_lookup_dict.get((lag_s_date, loc))

        # Fallback defaults
        target_log_s_ago_val = np.log1p(0)
        ili_s_ago_val = overall_ili_mean
        national_ili_s_ago_val = overall_national_ili_mean
        hhs_ili_s_ago_val = overall_hhs_ili_mean

        if lookup_values is not None:
          target_log_s_ago_val = lookup_values.get(
              TARGET_STR_LOG, target_log_s_ago_val
          )
          ili_s_ago_val = lookup_values.get('unweighted_ili', ili_s_ago_val)
          national_ili_s_ago_val = lookup_values.get(
              'national_ili', national_ili_s_ago_val
          )
          hhs_ili_s_ago_val = lookup_values.get('hhs_ili', hhs_ili_s_ago_val)

        current_test_row_features[f'{TARGET_STR_LOG}_lag_{s_lag}'] = (
            target_log_s_ago_val
        )
        current_test_row_features[f'ili_lag_{s_lag}'] = ili_s_ago_val
        current_test_row_features[f'national_ili_lag_{s_lag}'] = (
            national_ili_s_ago_val
        )
        current_test_row_features[f'hhs_ili_lag_{s_lag}'] = (
            hhs_ili_s_ago_val  # NEW FEATURE
        )

      # Retrieve current step's imputed ILI values from the comprehensive lookup
      ili_imputed_values = history_imputation_lookup_dict.get(
          (t_end_date, loc),
          {
              'unweighted_ili': overall_ili_mean,
              'national_ili': overall_national_ili_mean,
              'hhs_ili': overall_hhs_ili_mean,
          },
      )

      imputed_unweighted_ili_current_step = ili_imputed_values['unweighted_ili']
      imputed_national_ili_current_step = ili_imputed_values['national_ili']
      imputed_hhs_ili_current_step = ili_imputed_values['hhs_ili']

      # Store imputed ILI values for this location and horizon
      horizon_unweighted_ili_imputed[loc] = imputed_unweighted_ili_current_step
      horizon_national_ili_imputed[loc] = imputed_national_ili_current_step
      horizon_hhs_ili_imputed[loc] = imputed_hhs_ili_current_step

      # Select relevant features
      X_test_row = current_test_row_features[features]
      X_test_row = X_test_row.fillna(X_train_means)

      # Get point prediction (on log scale)
      pred_mean_log = model.predict(X_test_row)[0]

      # Clip log prediction to ensure numerical stability before sampling
      pred_mean_log = np.clip(pred_mean_log, np.log1p(0), None)

      horizon_predictions_log[loc] = (
          pred_mean_log  # Store for rest-of-US calculation
      )
      horizon_populations[loc] = row[
          'population'
      ]  # Store population for weighted average

      # Retrieve location-specific residual standard deviation
      loc_residual_std = residual_std_per_location.get(loc, global_fallback_std)

      # Generate samples on log scale using location-specific std
      samples_log = norm.rvs(
          loc=pred_mean_log, scale=loc_residual_std, size=N_SAMPLES
      )

      # Explicitly clip log samples to ensure numerical stability before inverse transformation
      samples_log = np.clip(samples_log, np.log1p(0), None)

      # Inverse transform samples, clip at zero
      samples = np.expm1(samples_log)
      samples = np.maximum(
          0, samples
      )  # Keep as float for monotonicity enforcement

      # Calculate quantiles, enforce monotonicity, then round and convert to int
      row_quantiles = np.percentile(samples, [q * 100 for q in QUANTILES])
      row_quantiles = np.maximum.accumulate(
          row_quantiles
      )  # Enforce monotonicity on floats BEFORE rounding
      row_quantiles = row_quantiles.round().astype(int)  # Then round and cast

      test_y_hat_quantiles.loc[original_idx] = row_quantiles

    # After all locations are processed for the current horizon_idx, update lags for the next horizon
    if (
        horizon_predictions_log
    ):  # Only proceed if predictions were made for this horizon
      # 1. Create a temporary DataFrame for vectorized calculation for the current horizon
      current_horizon_locs_df = pd.DataFrame(
          {
              'pred_log': pd.Series(horizon_predictions_log),
              'population': pd.Series(horizon_populations),
          },
          index=all_forecast_locations,  # Use all locations as index to ensure consistent size
      ).fillna(
          0
      )  # Fill with 0 for locations not explicitly in horizon_predictions_log (shouldn't happen with correct loop)

      # 2. Vectorize calculation of `rest_of_us_target_log` for all locations in this horizon
      current_horizon_locs_df['weighted_pred_log'] = (
          current_horizon_locs_df['pred_log']
          * current_horizon_locs_df['population']
      )

      total_weighted_pred_log_horizon = current_horizon_locs_df[
          'weighted_pred_log'
      ].sum()
      total_population_horizon = current_horizon_locs_df['population'].sum()

      current_horizon_locs_df['other_loc_weighted_sum'] = (
          total_weighted_pred_log_horizon
          - current_horizon_locs_df['weighted_pred_log']
      )
      current_horizon_locs_df['other_loc_population_sum'] = (
          total_population_horizon - current_horizon_locs_df['population']
      )

      current_horizon_locs_df['rest_of_us_target_log_calculated'] = (
          current_horizon_locs_df['other_loc_weighted_sum']
          / current_horizon_locs_df['other_loc_population_sum']
      )
      current_horizon_locs_df.loc[
          current_horizon_locs_df['other_loc_population_sum'] == 0,
          'rest_of_us_target_log_calculated',
      ] = 0.0

      # 3. For each loc in all_forecast_locations, update current_lags_store
      for (
          loc
      ) in (
          all_forecast_locations
      ):  # Iterate through all_forecast_locations to ensure all are updated
        # Update target_log_lags
        current_lags_store[loc]['target_log_lags'].pop(0)
        current_lags_store[loc]['target_log_lags'].append(
            horizon_predictions_log.get(loc, np.log1p(0))
        )  # Use .get with default 0.0 for robustness

        # Update ili_lags
        current_lags_store[loc]['ili_lags'].pop(0)
        current_lags_store[loc]['ili_lags'].append(
            horizon_unweighted_ili_imputed.get(loc, overall_ili_mean)
        )

        # Update national_ili_lags
        current_lags_store[loc]['national_ili_lags'].pop(0)
        current_lags_store[loc]['national_ili_lags'].append(
            horizon_national_ili_imputed.get(loc, overall_national_ili_mean)
        )

        # NEW: Update hhs_ili_lags
        current_lags_store[loc]['hhs_ili_lags'].pop(0)
        current_lags_store[loc]['hhs_ili_lags'].append(
            horizon_hhs_ili_imputed.get(loc, overall_hhs_ili_mean)
        )

        # Update rest_of_us_target_log_lags using the vectorized result
        current_lags_store[loc]['rest_of_us_target_log_lags'].pop(0)
        current_lags_store[loc]['rest_of_us_target_log_lags'].append(
            current_horizon_locs_df.loc[loc, 'rest_of_us_target_log_calculated']
        )
        # Note: ili_lag_{51,52,53}, national_ili_lag_{51,52,53}, hhs_ili_lag_{51,52,53} and TARGET_STR_LOG_lag_{51,52,53} are *not* updated here as they are dynamically looked up from history_imputation_lookup_dict.

  # Ensure output is integer type as per sample_submission
  test_y_hat_quantiles = test_y_hat_quantiles.astype(int)

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
