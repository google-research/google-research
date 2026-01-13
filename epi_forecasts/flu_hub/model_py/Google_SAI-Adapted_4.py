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
MODEL_NAME = 'Google_SAI-Adapted_4'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import datetime
import warnings
from typing import List, Dict, Tuple, Union
from sklearn.linear_model import HuberRegressor

# Assuming QUANTILES and TARGET_STR are globally available from the notebook's setup
# Assuming locations, dataset, ilinet_state, ilinet are globally available and preprocessed as per preamble.

# IMPLEMENTATION PLAN.
# ## Core Principles Checklist
# 1. Matches seasonal growth rate trends against historic growth rate curves to identify the closest matches.
#    My code ensures that the database of historic growth rate curves is populated only with segments that exhibit meaningful dynamism, by filtering out historical trends where the standard deviation of normalized growth rates is negligible. This prevents uninformative flat historical periods from diluting the nearest neighbor search. The current observed trend, if it contains real data, is always used for matching to allow for forecasts even from flat-but-non-zero starting points, and its epidemiological phase is aligned using a circular season-week distance.
# 2. Makes forecasts based on nearest neighbor trajectories.
#    Once the `K_NEIGHBORS` historical segments are identified as closest matches, my code retrieves their associated future trajectories. Each historical trajectory is then scaled based on the ratio of the current observed value to the historical observed value at the end of the matching growth rate trend. These scaled trajectories are aggregated, and the specified quantiles are calculated for each future forecast horizon.
# 3. Calculate current seasonal growth rate trends from available data.
#    The `fit_and_predict_fn` extracts the most recent `LOOKBACK_WEEKS + 1` raw historical `Total Influenza Admissions` values (converted to per-capita after adding a pseudo-count) for the relevant period. These `LOOKBACK_WEEKS + 1` values are then log-transformed, and `LOOKBACK_WEEKS` week-over-week differences are calculated, forming the current seasonal growth rate trend for matching. The `season_week` corresponding to the end of this current trend is also determined using the `get_season_week_from_date` to inform the distance metric.

# ## Step-by-Step Logic
# 1.  **Define Constants and Globals:** Set `LOOKBACK_WEEKS`, `K_NEIGHBORS`, `PSEUDO_COUNT_FOR_ADMISSIONS` (for handling zero counts and stabilizing log transform input for admissions), `PSEUDO_RATE_FOR_ILI` (for ILI rates), `EPSILON_STABILITY` (for robust divisions and norm checks), `MIN_OVERLAP_FOR_HUBER` (for HuberRegressor training), `SEASON_WEEK_PENALTY_FACTOR` for KNN, `SEASON_LENGTH` (for circular season week distance), `MAX_STORED_FUTURE_TRAJECTORY_LENGTH` (for historical library, decoupled from MAX_HORIZON), and `MAX_SCALING_FACTOR` (to clamp trajectory scaling). Access `QUANTILES`, `TARGET_STR`, `locations`, `ilinet_state`, `ilinet` from the global scope. Define `NATIONAL_LOCATION_FIPS`.
# 2.  **Helper Functions (defined within `fit_and_predict_fn`):**
#     a.  `get_flu_season_id(date_val: datetime.date)`: Assigns a flu season ID.
#     b.  `convert_ilinet_season_to_id(season_str: str)`: Converts 'YYYY/YY' ILINet season format to 'YYYY-YYYY+1'.
#     c.  `get_season_week_from_date(date_val: datetime.date)`: Derives a consistent season week by calculating the number of weeks since a fixed season start anchor (September 28th) for the relevant epidemiological year, and adding an offset (10) to align with ILINet's season_week 10 for that date. This ensures consistent season week calculation across `dataset` and `ilinet_state_df`.
#     d.  `calculate_normalized_growth_rate(series_values: np.ndarray, epsilon: float)`: Calculates log-transformed differences and normalizes to unit L2 norm.
# 3.  **Data Preprocessing and Augmentation:**
#     a.  Combine `train_x` and `train_y` into `current_history_df`. Convert `target_end_date` to `datetime.date`. Map `population` directly from `location_population_map` for consistency and set as `float`. Add `season_id` using `get_flu_season_id` and `season_week` using `get_season_week_from_date`.
#     b.  Prepare `ilinet_state_df`: filter by `region_type=='States'`, rename columns, convert dates, merge with global `locations` DataFrame to add FIPS `location` codes. Explicitly fill `NaN` values in `ili_value` with 0. Derive `season_id` from the existing `season` column using `convert_ilinet_season_to_id`. Derive `season_week` using `get_season_week_from_date`. Ensure `population` is mapped as float.
#     c.  Prepare `national_ilinet_df`: filter by `region_type=='National'`, rename columns, convert dates. Explicitly fill `NaN` values in `ili_value` with 0. Assign `NATIONAL_LOCATION_FIPS` and `total_us_population`. Derive `season_id` and `season_week`.
#     d.  Aggregate `current_history_df` to `national_admissions_df`: sum `Total Influenza Admissions` and `population`, assign `NATIONAL_LOCATION_FIPS`. Derive `season_id` and `season_week`.
#     e.  Calculate `MAX_HORIZON` as the maximum *non-negative* horizon from `test_x`.
#     f.  Calculate `per_capita_admissions` for `current_history_df` and `national_admissions_df` (using `PSEUDO_COUNT_FOR_ADMISSIONS`).
#     g.  Calculate `per_capita_ili` for `ilinet_state_df` and `national_ilinet_df` (using `PSEUDO_RATE_FOR_ILI`).
#     h.  **Learn Transformation (Robust Linear Regression with HuberRegressor):**
#         i.  Combine state-level and national-level overlap data for training a global `HuberRegressor`.
#         ii. Initialize `global_median_ratio_fallback`.
#         iii. Train `global_huber_model`. If fitting fails (e.g., insufficient variance), explicitly set `global_huber_model.coef_` to `global_median_ratio_fallback` and `global_huber_model.intercept_` to 0.0.
#         iv. Initialize `location_transformation_map`. For each unique `location` in the *state-level* overlap data with sufficient data/variance, train and store a location-specific `HuberRegressor`.
#         v. Apply transformations: Apply `global_huber_model` to all `ilinet_state_df['per_capita_ili']` and `national_ilinet_df['per_capita_ili']` to create `synthetic_admissions_per_capita`. Then, for locations with specific models, overwrite their `synthetic_admissions_per_capita`. Ensure non-negativity.
#     i.  **Create Full Historical Library (`full_history_df`):**
#         i.  Create `actual_admissions_df` (state-level actual).
#         ii. Create `synthetic_admissions_df` (state-level synthetic).
#         iii. Create `national_actual_df_for_full_history` (national-level actual, with derived season data).
#         iv. Create `national_synthetic_df_for_full_history` (national-level synthetic, with derived season data).
#         v.  Concatenate these four dataframes.
#         vi. Apply `drop_duplicates(subset=['target_end_date', 'location'], keep='first')` to ensure actual admissions are prioritized where overlap occurs.
#         vii. Sort `full_history_df` by `location` and `target_end_date`.
#         viii. Ensure all `value_per_capita` entries in `full_history_df` are non-negative floats.
# 4.  **Build Historical Trajectory Library (`historical_library_by_location`):**
#     a.  Initialize an empty dictionary `historical_library_by_location`.
#     b.  Update `location_population_map` to include `NATIONAL_LOCATION_FIPS` mapping to `total_us_population`.
#     c.  Iterate through each unique `location` (including `NATIONAL_LOCATION_FIPS`) in `full_history_df`.
#     d.  For each `location`, iterate through each unique `season_id` within that location's data.
#     e.  For each season: Extract series, reindex, handle season_week, calculate growth rates, generate sliding windows (filtering flat trends), and store segments (`raw_per_capita_at_trend_end`, `season_week_at_trend_end`, `future_trajectory_per_capita`). Append to `historical_library_by_location[loc]`.
# 5.  **Generate Forecasts for `test_x`:**
#     a.  Initialize `test_y_hat_quantiles` DataFrame.
#     b.  Convert `reference_date` and `target_end_date` in `test_x` to `datetime.date`.
#     c.  Iterate through each unique `(reference_date, location)` group in `test_x`. (Note: `loc` here will only be state-level FIPS codes).
#     d.  Get current location's population.
#     e.  Handle `horizon = -1` by retrieving actual values.
#     f.  **Prepare Current Trend for Matching (for `horizon >= 0`):**
#         i.  Extract last `LOOKBACK_WEEKS + 1` of `per_capita_admissions` for current `location` up to `ref_date - 1 week`. Ensure weekly frequency.
#         ii. If current trend is empty, too short, or consists *only* of pseudo-counts/flat, assign 0 to all quantiles and continue.
#         iii. Store `last_observed_per_capita_value`.
#         iv. Calculate normalized log-transformed growth rate.
#         v.  Calculate `current_season_week`.
#     g.  **Find Nearest Neighbors (with season_week penalty using circular distance):**
#         i.  Retrieve `relevant_historical_curves` by combining both `historical_library_by_location.get(loc, [])` (state-specific) and `historical_library_by_location.get(NATIONAL_LOCATION_FIPS, [])` (national-level).
#         ii. Calculate L2 distance + season_week penalty.
#         iii. Sort by distance and select `K_NEIGHBORS`.
#     h.  **Scale and Aggregate Trajectories:**
#         i.  Initialize `forecast_trajectories_by_horizon_pc` dictionary.
#         ii. For each `neighbor` in `nearest_neighbors`: Calculate `scaling_factor_pc`, clamp it, and append scaled predictions to `forecast_trajectories_by_horizon_pc[h]`.
#     i.  **Calculate and Output Quantiles for Horizons >= 0:**
#         i.  For each `row` in current `test_x` group: Convert to `np.array`, ensure non-negativity, convert to absolute counts, calculate `np.percentile`, round to nearest integer, and enforce monotonicity.
#         ii. Assign final integer quantiles to `test_y_hat_quantiles`.
#         iii. Otherwise, assign 0 to all quantiles for that row.
# 6.  **Return:** Return the completed `test_y_hat_quantiles` DataFrame.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using the required method by modelling train_x to train_y.

  Returns quantile forecasts based on nearest neighbor growth rate trajectories.
  """

  # --- Custom Helper Functions for the KNN Logic (now nested as per plan) ---
  def get_flu_season_id(date_val):
    """Assigns a flu season ID (e.g., '2020-2021') based on the month."""
    year = date_val.year
    if (
        date_val.month >= 7
    ):  # July onwards counts towards the next flu season year
      return f'{year}-{year + 1}'
    else:  # January to June counts towards the current flu season year (which started last calendar year)
      return f'{year - 1}-{year}'

  def convert_ilinet_season_to_id(season_str):
    """Converts 'YYYY/YY' ILINet season format to 'YYYY-YYYY+1' format."""
    start_year_str = season_str.split('/')[0]
    start_year = int(start_year_str)
    return f'{start_year}-{start_year + 1}'

  def get_season_week_from_date(date_val):
    """Derives a consistent season_week number from a date, aligned with ILINet's season_week 10 convention.

    This function anchors the season week calculation to a fixed epidemiological
    start (September 28th) for each flu season, ensuring consistency across
    datasets.
    """
    # Determine the epidemiological year for the given date.
    # Flu season typically starts in the fall (Sept/Oct) and extends into the following calendar year.
    epi_year = date_val.year
    # If the date is before September, it belongs to the previous fall's season
    if date_val.month < 9:
      epi_year -= 1

    # Define the anchor date for the start of the season as September 28th of the epi_year.
    # ILINet data shows '1997-09-28' corresponds to season_week 10.
    season_anchor_date = datetime.date(epi_year, 9, 28)

    # Calculate the number of full weeks passed since the anchor date.
    # Add an offset of 10 to align with ILINet's season_week convention.
    season_week = (date_val - season_anchor_date).days // 7 + 10

    return season_week

  def calculate_normalized_growth_rate(
      series_values, epsilon
  ):
    """Calculates log-transformed differences (growth rates) on an array and normalizes to unit L2 norm.

    The input `series_values` should have at least two points to calculate one
    difference. `np.log1p` is used, assuming `series_values` are already
    positive due to pseudo-count.
    """
    if len(series_values) < 2:
      return np.array([])

    # Apply log1p transformation to make values more amenable to proportional change
    # Series values are already per-capita and > 0 due to pseudo-count
    log_transformed_series = np.log1p(series_values)

    # Calculate simple differences (growth rates) on log-transformed series.
    diff_array = np.diff(log_transformed_series)

    # Calculate L2 norm. Add a small epsilon to avoid division by zero if all diffs are zero.
    norm = np.linalg.norm(diff_array)

    if norm < epsilon:  # Use EPSILON_STABILITY here
      return np.zeros_like(diff_array, dtype=float)
    else:
      return diff_array / norm

  # Parameters for KNN - OPTIMIZED VALUES (as per original notebook's implicit optimization)
  LOOKBACK_WEEKS = 6
  K_NEIGHBORS = 50
  PSEUDO_COUNT_FOR_ADMISSIONS = (
      5.0  # pseudo-count for Total Influenza Admissions
  )
  PSEUDO_RATE_FOR_ILI = (
      0.01  # pseudo-rate for ILI values (already a rate/percentage)
  )
  EPSILON_STABILITY = 1e-6  # Small epsilon for robust scaling factor calculation, norm checks, and other divisions
  MIN_OVERLAP_FOR_HUBER = 10  # Minimum data points required to train location-specific HuberRegressor
  SEASON_WEEK_PENALTY_FACTOR = (
      0.05  # Parameter for season_week penalty in KNN distance
  )
  SEASON_LENGTH = (
      52  # Approximate number of weeks in a flu season for circular distance
  )
  MAX_STORED_FUTURE_TRAJECTORY_LENGTH = 20  # Max length of future trajectory stored in historical library (e.g., covers up to horizon 19)
  MAX_SCALING_FACTOR = (
      50.0  # Upper bound for scaling factor to prevent extreme forecasts
  )

  # TARGET_STR and QUANTILES are globally available from the helper functions cell.
  QUANTILES = globals()['QUANTILES']
  TARGET_STR = globals()['TARGET_STR']

  # Define MAX_HORIZON based only on non-negative horizons
  relevant_horizons_for_library = test_x[test_x['horizon'] >= 0]['horizon']
  MAX_HORIZON = (
      relevant_horizons_for_library.max()
      if not relevant_horizons_for_library.empty
      else 3
  )

  # Define a special FIPS for National level data
  NATIONAL_LOCATION_FIPS = -999

  # --- 1. Initialize and Prepare Data ---

  # Create location-to-population map directly from global locations for robustness
  location_population_map: Dict[int, float] = (
      globals()['locations']
      .set_index('location')['population']
      .astype(float)
      .to_dict()
  )
  total_us_population = globals()['locations']['population'].sum()
  # Add national population to the map
  location_population_map[NATIONAL_LOCATION_FIPS] = total_us_population

  # Combine train_x and train_y
  current_history_df = train_x.copy()
  current_history_df[TARGET_STR] = train_y.copy()
  current_history_df['target_end_date'] = pd.to_datetime(
      current_history_df['target_end_date']
  ).dt.date
  # Ensure population is float and available from location map for consistency
  current_history_df['population'] = (
      current_history_df['location'].map(location_population_map).astype(float)
  )
  current_history_df = current_history_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)
  current_history_df['season_id'] = current_history_df['target_end_date'].apply(
      get_flu_season_id
  )
  current_history_df['season_week'] = current_history_df[
      'target_end_date'
  ].apply(get_season_week_from_date)

  # Prepare ilinet_state_df (assuming it's globally available and cleaned as in preamble)
  ilinet_state_df = globals()['ilinet_state'].copy()
  ilinet_state_df = ilinet_state_df[
      ilinet_state_df['region_type'] == 'States'
  ].copy()
  ilinet_state_df = ilinet_state_df.rename(
      columns={
          'region': 'location_name',
          'week_start': 'target_end_date',
          'unweighted_ili': 'ili_value',
      }
  )
  ilinet_state_df['target_end_date'] = pd.to_datetime(
      ilinet_state_df['target_end_date']
  ).dt.date
  ilinet_state_df['ili_value'] = ilinet_state_df['ili_value'].fillna(
      0
  )  # Fill NaN ILI values with 0

  # Merge with location data to get FIPS and population, then drop rows that didn't merge
  ilinet_state_df = pd.merge(
      ilinet_state_df,
      globals()['locations'][['location_name', 'location']],
      on='location_name',
      how='left',
  )
  ilinet_state_df = ilinet_state_df.dropna(
      subset=['location']
  )  # Drop rows where location merge failed
  ilinet_state_df['location'] = ilinet_state_df['location'].astype(int)
  # Get population from the map
  ilinet_state_df['population'] = (
      ilinet_state_df['location'].map(location_population_map).astype(float)
  )
  ilinet_state_df = ilinet_state_df.dropna(
      subset=['population']
  )  # Drop rows if population lookup fails (shouldn't happen with filtered locations)

  # Derive season_id from the existing 'season' column for consistency.
  ilinet_state_df['season_id'] = ilinet_state_df['season'].apply(
      convert_ilinet_season_to_id
  )
  ilinet_state_df['season_week'] = ilinet_state_df['target_end_date'].apply(
      get_season_week_from_date
  )
  ilinet_state_df = ilinet_state_df[[
      'target_end_date',
      'location',
      'ili_value',
      'population',
      'season_week',
      'season_id',
  ]].copy()

  # --- Prepare National ILI Data (from ilinet) ---
  national_ilinet_df = globals()['ilinet'].copy()
  national_ilinet_df = national_ilinet_df[
      national_ilinet_df['region_type'] == 'National'
  ].copy()
  national_ilinet_df = national_ilinet_df.rename(
      columns={'week_start': 'target_end_date', 'unweighted_ili': 'ili_value'}
  )
  national_ilinet_df['target_end_date'] = pd.to_datetime(
      national_ilinet_df['target_end_date']
  ).dt.date
  national_ilinet_df['ili_value'] = national_ilinet_df['ili_value'].fillna(0)

  # Assign a dummy location ID for 'National' and total US population for consistency
  national_ilinet_df['location'] = NATIONAL_LOCATION_FIPS
  national_ilinet_df['population'] = (
      total_us_population  # Use the sum of all state populations
  )

  national_ilinet_df['season_id'] = national_ilinet_df['season'].apply(
      convert_ilinet_season_to_id
  )
  national_ilinet_df['season_week'] = national_ilinet_df[
      'target_end_date'
  ].apply(get_season_week_from_date)
  national_ilinet_df = national_ilinet_df[[
      'target_end_date',
      'location',
      'ili_value',
      'population',
      'season_week',
      'season_id',
  ]].copy()

  # --- Convert to Per-Capita Values using pseudo-count ---
  current_history_df['per_capita_admissions'] = (
      current_history_df[TARGET_STR].astype(float) + PSEUDO_COUNT_FOR_ADMISSIONS
  ) / current_history_df['population']

  # Use PSEUDO_RATE_FOR_ILI for ili_value (which is a rate/percentage)
  ilinet_state_df['per_capita_ili'] = (
      ilinet_state_df['ili_value'].astype(float) + PSEUDO_RATE_FOR_ILI
  )
  national_ilinet_df['per_capita_ili'] = (
      national_ilinet_df['ili_value'].astype(float) + PSEUDO_RATE_FOR_ILI
  )

  # --- Aggregate National Admissions Data (from current_history_df) ---
  national_admissions_df = (
      current_history_df.groupby('target_end_date')
      .agg({TARGET_STR: 'sum', 'population': 'sum'})
      .reset_index()
  )
  national_admissions_df['location'] = NATIONAL_LOCATION_FIPS
  national_admissions_df['per_capita_admissions'] = (
      national_admissions_df[TARGET_STR].astype(float)
      + PSEUDO_COUNT_FOR_ADMISSIONS
  ) / national_admissions_df['population']
  # Add season_id and season_week to national_admissions_df
  national_admissions_df['season_id'] = national_admissions_df[
      'target_end_date'
  ].apply(get_flu_season_id)
  national_admissions_df['season_week'] = national_admissions_df[
      'target_end_date'
  ].apply(get_season_week_from_date)
  national_admissions_df = national_admissions_df[[
      'target_end_date',
      'location',
      'per_capita_admissions',
      'season_id',
      'season_week',
  ]].copy()

  # --- Learn Transformation (Robust Linear Regression with HuberRegressor) ---

  # Combine state-level and national-level overlap data for training the GLOBAL Huber model
  combined_state_overlap_per_capita = pd.merge(
      current_history_df[
          ['target_end_date', 'location', 'per_capita_admissions']
      ],
      ilinet_state_df[['target_end_date', 'location', 'per_capita_ili']],
      on=['target_end_date', 'location'],
      how='inner',
  )

  national_overlap_per_capita = pd.merge(
      national_admissions_df[
          ['target_end_date', 'location', 'per_capita_admissions']
      ],
      national_ilinet_df[['target_end_date', 'location', 'per_capita_ili']],
      on=['target_end_date', 'location'],
      how='inner',
  )

  combined_overlap_for_global_huber = pd.concat(
      [combined_state_overlap_per_capita, national_overlap_per_capita],
      ignore_index=True,
  )

  # Calculate robust median ratio as a global fallback
  global_median_ratio_fallback = 1.0
  if not combined_overlap_for_global_huber.empty:
    # Calculate robust median ratio: only consider non-zero ILI values to avoid division by zero issues
    positive_ili_overlap = combined_overlap_for_global_huber[
        combined_overlap_for_global_huber['per_capita_ili'] > EPSILON_STABILITY
    ]
    if not positive_ili_overlap.empty:
      ratios = (
          positive_ili_overlap['per_capita_admissions']
          / positive_ili_overlap['per_capita_ili']
      )
      # Filter out infinite/NaN ratios if any arise from division
      ratios = ratios[np.isfinite(ratios)]
      if not ratios.empty:
        global_median_ratio_fallback = np.nanmedian(ratios)
    if pd.isna(global_median_ratio_fallback) or np.isinf(
        global_median_ratio_fallback
    ):
      global_median_ratio_fallback = (
          1.0  # Default to 1.0 if median is problematic
      )

  global_huber_model = HuberRegressor(epsilon=1.35, max_iter=1000)

  # Train global Huber model as a general fallback - Refactored for clarity
  huber_model_fitted = False
  X_global = combined_overlap_for_global_huber['per_capita_ili'].values.reshape(
      -1, 1
  )
  y_global = combined_overlap_for_global_huber['per_capita_admissions'].values

  if (
      X_global.shape[0] > 1
      and np.std(X_global) > EPSILON_STABILITY
      and np.std(y_global) > EPSILON_STABILITY
  ):
    try:
      global_huber_model.fit(X_global, y_global)
      huber_model_fitted = True
    except Exception:
      # If an error occurs during fitting, huber_model_fitted remains False
      pass

  if not huber_model_fitted:
    # Fallback to a simple proportional scaling if global HuberRegressor cannot be reliably fit or fails
    global_huber_model.coef_ = np.array([global_median_ratio_fallback])
    global_huber_model.intercept_ = 0.0

  location_transformation_map: Dict[int, HuberRegressor] = {}
  # Use only state-level overlap data for location-specific Huber training
  unique_locations_with_overlap = combined_state_overlap_per_capita[
      'location'
  ].unique()

  for loc in unique_locations_with_overlap:
    loc_overlap_df = combined_state_overlap_per_capita[
        combined_state_overlap_per_capita['location'] == loc
    ].copy()

    X_loc = loc_overlap_df['per_capita_ili'].values.reshape(-1, 1)
    y_loc = loc_overlap_df['per_capita_admissions'].values

    # Only attempt to train location-specific Huber if sufficient data AND variance
    if (
        X_loc.shape[0] >= MIN_OVERLAP_FOR_HUBER
        and np.std(X_loc) > EPSILON_STABILITY
        and np.std(y_loc) > EPSILON_STABILITY
    ):
      try:
        loc_huber_model = HuberRegressor(epsilon=1.35, max_iter=1000)
        loc_huber_model.fit(X_loc, y_loc)
        location_transformation_map[loc] = loc_huber_model
      except Exception:
        # If location-specific HuberRegressor fails, it won't be in map, falling back to global
        pass

  # Apply the learned transformations to create synthetic_admissions_per_capita for the entire ILINet history
  # First, apply the global Huber model to all ILINet state data
  ilinet_state_df['synthetic_admissions_per_capita'] = (
      global_huber_model.predict(
          ilinet_state_df['per_capita_ili'].values.reshape(-1, 1)
      )
  )

  # Also apply the global Huber model to national ILINet data
  national_ilinet_df['synthetic_admissions_per_capita'] = (
      global_huber_model.predict(
          national_ilinet_df['per_capita_ili'].values.reshape(-1, 1)
      )
  )

  # Then, iterate and overwrite for locations with specific Huber models (state-level only)
  for loc, huber_model in location_transformation_map.items():
    loc_mask = ilinet_state_df['location'] == loc
    if loc_mask.any():
      ilinet_state_df.loc[loc_mask, 'synthetic_admissions_per_capita'] = (
          huber_model.predict(
              ilinet_state_df.loc[loc_mask, 'per_capita_ili'].values.reshape(
                  -1, 1
              )
          )
      )

  ilinet_state_df['synthetic_admissions_per_capita'] = np.maximum(
      0.0, ilinet_state_df['synthetic_admissions_per_capita']
  )
  national_ilinet_df['synthetic_admissions_per_capita'] = np.maximum(
      0.0, national_ilinet_df['synthetic_admissions_per_capita']
  )

  # Create full_history_df using all available actual and synthetic per-capita values
  # Prioritize actual admissions where dates/locations overlap.
  actual_admissions_df = current_history_df[[
      'target_end_date',
      'location',
      'per_capita_admissions',
      'season_id',
      'season_week',
  ]].copy()
  actual_admissions_df = actual_admissions_df.rename(
      columns={'per_capita_admissions': 'value_per_capita'}
  )

  # Use ALL processed ilinet_state_df (which is inherently limited to ILINet's available dates)
  synthetic_admissions_df = ilinet_state_df[[
      'target_end_date',
      'location',
      'synthetic_admissions_per_capita',
      'season_id',
      'season_week',
  ]].copy()
  synthetic_admissions_df = synthetic_admissions_df.rename(
      columns={'synthetic_admissions_per_capita': 'value_per_capita'}
  )

  # Prepare national actual admissions for concatenation
  national_actual_df_for_full_history = national_admissions_df[[
      'target_end_date',
      'location',
      'per_capita_admissions',
      'season_id',
      'season_week',
  ]].copy()
  national_actual_df_for_full_history = (
      national_actual_df_for_full_history.rename(
          columns={'per_capita_admissions': 'value_per_capita'}
      )
  )

  # Prepare national synthetic admissions for concatenation
  national_synthetic_df_for_full_history = national_ilinet_df[[
      'target_end_date',
      'location',
      'synthetic_admissions_per_capita',
      'season_id',
      'season_week',
  ]].copy()
  national_synthetic_df_for_full_history = (
      national_synthetic_df_for_full_history.rename(
          columns={'synthetic_admissions_per_capita': 'value_per_capita'}
      )
  )

  # Concatenate all data, prioritizing actual over synthetic and state over national if FIPS codes overlap (though they won't due to NATIONAL_LOCATION_FIPS)
  full_history_df = pd.concat(
      [
          actual_admissions_df,
          synthetic_admissions_df,
          national_actual_df_for_full_history,
          national_synthetic_df_for_full_history,
      ],
      ignore_index=True,
  )
  full_history_df = full_history_df.drop_duplicates(
      subset=['target_end_date', 'location'], keep='first'
  )
  full_history_df = full_history_df.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  full_history_df['value_per_capita'] = full_history_df[
      'value_per_capita'
  ].apply(lambda x: max(0.0, float(x)))

  # --- 2. Generate Historical Trajectory Library (by location for efficiency) ---
  historical_library_by_location: Dict[int, List[Dict]] = {
      loc: [] for loc in full_history_df['location'].unique()
  }

  for loc in full_history_df['location'].unique():
    loc_df = full_history_df[full_history_df['location'] == loc].copy()

    current_loc_population = location_population_map.get(loc, 1.0)
    # Calculate the appropriate fill value for asfreq for this location
    fill_value_for_asfreq = PSEUDO_COUNT_FOR_ADMISSIONS / current_loc_population

    for season_id in loc_df['season_id'].unique():
      season_df = loc_df[loc_df['season_id'] == season_id].copy()

      # Resample both value_per_capita and season_week to ensure continuous weekly series
      # Use location-specific fill_value for consistent pseudo-counts
      season_series_pc = season_df.set_index('target_end_date').asfreq(
          'W-SAT', fill_value=fill_value_for_asfreq
      )

      # Robust season_week handling for dates introduced by asfreq
      # First, try ffill/bfill for existing season_week values
      season_series_pc['season_week'] = (
          season_series_pc['season_week'].ffill().bfill()
      )
      # For any remaining NaNs, derive season_week from the date index itself
      nan_season_week_mask = season_series_pc['season_week'].isna()
      if nan_season_week_mask.any():
        season_series_pc.loc[nan_season_week_mask, 'season_week'] = (
            season_series_pc.index[nan_season_week_mask]
            .to_series()
            .apply(get_season_week_from_date)
        )
      season_series_pc['season_week'] = season_series_pc['season_week'].astype(
          int
      )

      raw_series_pc = season_series_pc['value_per_capita']
      raw_season_week_series = season_series_pc['season_week']

      # We need LOOKBACK_WEEKS + 1 raw values for the trend and up to MAX_STORED_FUTURE_TRAJECTORY_LENGTH for future.
      # Total minimum length required for a trend segment and at least one future point is LOOKBACK_WEEKS + 2
      if len(raw_series_pc) < LOOKBACK_WEEKS + 2:
        continue

      # Calculate normalized log-transformed growth rates
      full_growth_rates_pc = calculate_normalized_growth_rate(
          raw_series_pc.values, EPSILON_STABILITY
      )

      # Loop through possible starting points of the raw series segments
      # 'i' is the starting index of the raw series window [v_i, ..., v_{i+LOOKBACK_WEEKS}]
      # which contributes to the growth rate trend [g_i, ..., g_{i+LOOKBACK_WEEKS-1}]
      # The range now ensures there are enough raw values for the trend and at least one subsequent future trajectory point.
      for i in range(len(raw_series_pc) - (LOOKBACK_WEEKS + 1)):
        # Extract LOOKBACK_WEEKS of normalized log-transformed growth rates
        trend_segment_growth_rates_pc = full_growth_rates_pc[
            i : i + LOOKBACK_WEEKS
        ]

        # --- CRUCIAL IMPROVEMENT: Filter out flat/uninformative historical trends ---
        # If the trend segment is too short or its standard deviation is negligible, skip it.
        # A very low standard deviation indicates a flat trend (e.g., all pseudo-counts or constant values)
        if (
            len(trend_segment_growth_rates_pc) < LOOKBACK_WEEKS
            or np.std(trend_segment_growth_rates_pc) < EPSILON_STABILITY
        ):
          continue

        # raw_per_capita_at_trend_end is the last known raw per-capita value that contributed to this trend.
        raw_per_capita_at_trend_end = raw_series_pc.iloc[i + LOOKBACK_WEEKS]

        # Get the season_week corresponding to the end of this trend
        season_week_at_trend_end = raw_season_week_series.iloc[
            i + LOOKBACK_WEEKS
        ]

        # Future trajectory includes the value for horizon 0 (current week) up to MAX_STORED_FUTURE_TRAJECTORY_LENGTH - 1.
        # It starts from the week *after* raw_per_capita_at_trend_end.
        # We store up to MAX_STORED_FUTURE_TRAJECTORY_LENGTH points.
        future_trajectory_per_capita = raw_series_pc.iloc[
            i
            + LOOKBACK_WEEKS
            + 1 : i
            + LOOKBACK_WEEKS
            + 1
            + MAX_STORED_FUTURE_TRAJECTORY_LENGTH
        ].values

        if (
            len(future_trajectory_per_capita) > 0
        ):  # Ensure at least one future point is available
          # Store in location-specific list
          historical_library_by_location[loc].append({
              'location': loc,
              'season_id': season_id,
              'start_date': (
                  raw_series_pc.index[i].date()
              ),  # Store as datetime.date (start of the raw values for the trend)
              'trend_growth_rates_pc': (
                  trend_segment_growth_rates_pc
              ),  # Already normalized
              'raw_per_capita_at_trend_end': raw_per_capita_at_trend_end,
              'future_trajectory_per_capita': (
                  future_trajectory_per_capita
              ),  # This now stores variable length
              'population': current_loc_population,
              'season_week_at_trend_end': season_week_at_trend_end,
          })

  # --- 3. Process test_x for Current Trends and Forecast ---

  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index,
      columns=[f'quantile_{q}' for q in QUANTILES],
      dtype=float,
  )

  # Function to assign zeros to all quantiles for a given set of row indices
  def assign_zeros(row_indices):
    for idx in row_indices:
      for q_col in test_y_hat_quantiles.columns:
        test_y_hat_quantiles.loc[idx, q_col] = 0

  # Ensure test_x dates are datetime.date
  test_x_processed = test_x.copy()
  test_x_processed['reference_date'] = pd.to_datetime(
      test_x_processed['reference_date']
  ).dt.date
  test_x_processed['target_end_date'] = pd.to_datetime(
      test_x_processed['target_end_date']
  ).dt.date

  for (ref_date, loc), group in test_x_processed.groupby(
      ['reference_date', 'location']
  ):
    current_loc_population = location_population_map.get(
        loc, 1.0
    )  # Fallback to 1.0, though locations should be covered
    fill_value_for_current_trend = (
        PSEUDO_COUNT_FOR_ADMISSIONS / current_loc_population
    )

    # Handle horizon -1 separately as its value is already known
    H_minus_1_rows = group[group['horizon'] == -1]
    if not H_minus_1_rows.empty:
      target_date_H_minus_1 = ref_date - datetime.timedelta(weeks=1)

      # Robustly check if actual_value exists
      actual_value_series = current_history_df[
          (current_history_df['location'] == loc)
          & (current_history_df['target_end_date'] == target_date_H_minus_1)
      ][TARGET_STR]

      actual_value = (
          actual_value_series.iloc[0] if not actual_value_series.empty else 0
      )

      for _, row_h_minus_1 in H_minus_1_rows.iterrows():
        for q_col in test_y_hat_quantiles.columns:
          test_y_hat_quantiles.loc[row_h_minus_1.name, q_col] = int(
              actual_value
          )

    # Filter group for horizons >= 0 for KNN forecasting
    group_for_knn = group[group['horizon'] >= 0]
    if group_for_knn.empty:
      continue  # All horizons for this group were -1, already processed

    # Extract current raw trend from train_df
    current_data_end = ref_date - datetime.timedelta(weeks=1)

    # Get historical data up to current_data_end for the specific location, then convert to per-capita
    current_loc_history_pc = current_history_df[
        (current_history_df['location'] == loc)
        & (current_history_df['target_end_date'] <= current_data_end)
    ].set_index('target_end_date')['per_capita_admissions']

    # Resample to ensure weekly frequency, filling missing with fill_value for consistency.
    # We need LOOKBACK_WEEKS + 1 raw values to get LOOKBACK_WEEKS differences.
    current_raw_trend_series_pc = current_loc_history_pc.asfreq(
        'W-SAT', fill_value=fill_value_for_current_trend
    ).iloc[-(LOOKBACK_WEEKS + 1) :]

    # Handle edge cases: no data or effectively all pseudo-counts in current trend for KNN
    expected_pseudo_count_sum_pc = fill_value_for_current_trend * (
        LOOKBACK_WEEKS + 1
    )
    # Check if the series is too short, empty, or consists only of pseudo-counts
    if (
        current_raw_trend_series_pc.empty
        or len(current_raw_trend_series_pc) < (LOOKBACK_WEEKS + 1)
        or np.isclose(
            current_raw_trend_series_pc.sum(), expected_pseudo_count_sum_pc
        )
    ):
      assign_zeros(group_for_knn.index)
      continue  # Move to next location/ref_date group

    last_observed_per_capita_value = current_raw_trend_series_pc.iloc[-1]

    # Calculate normalized log-transformed growth rates for the current trend
    current_growth_trend_pc = calculate_normalized_growth_rate(
        current_raw_trend_series_pc.values, EPSILON_STABILITY
    )

    # Get the season_week for the end of the current trend
    current_season_week = get_season_week_from_date(
        current_raw_trend_series_pc.index[-1].date()
    )

    # If current_growth_trend_pc is empty after calculation (e.g., less than 2 valid points), default to 0
    # REMOVED: The check `np.std(current_growth_trend_pc) < EPSILON_STABILITY` as it was too restrictive.
    # A non-zero but flat trend is still a signal to be matched.
    if (
        current_growth_trend_pc.size == 0
        or len(current_growth_trend_pc) < LOOKBACK_WEEKS
    ):
      assign_zeros(group_for_knn.index)
      continue

    # Find k nearest neighbors - now efficiently using the dict and combining state+national
    distances = []

    # Combine location-specific historical curves with national curves
    relevant_historical_curves = []
    # Add location-specific curves if available
    if loc in historical_library_by_location:
      relevant_historical_curves.extend(historical_library_by_location[loc])
    # Add national curves for any location (since test_x 'loc' will always be a state FIPS, and not NATIONAL_LOCATION_FIPS)
    if NATIONAL_LOCATION_FIPS in historical_library_by_location:
      relevant_historical_curves.extend(
          historical_library_by_location[NATIONAL_LOCATION_FIPS]
      )

    if (
        not relevant_historical_curves
    ):  # If no historical data for this location (or national), predict 0
      assign_zeros(group_for_knn.index)
      continue

    for h_curve in relevant_historical_curves:
      # Both current_growth_trend_pc and h_curve['trend_growth_rates_pc'] are already normalized arrays
      if (
          h_curve['trend_growth_rates_pc'].shape[0] == LOOKBACK_WEEKS
      ):  # Ensure historical trend length matches
        # Base distance from growth rates
        dist_growth = np.linalg.norm(
            current_growth_trend_pc - h_curve['trend_growth_rates_pc']
        )

        # Calculate circular season_week distance
        diff = abs(current_season_week - h_curve['season_week_at_trend_end'])
        # Assuming season_week values range roughly across 52 weeks (e.g., 10 to 61)
        circular_season_week_dist = min(diff, SEASON_LENGTH - diff)

        # Add season_week penalty using circular distance
        dist_season_week = (
            SEASON_WEEK_PENALTY_FACTOR * circular_season_week_dist
        )

        total_dist = dist_growth + dist_season_week
        distances.append((total_dist, h_curve))

    if (
        not distances
    ):  # Still no distances (e.g., historical library empty after filtering), predict 0
      assign_zeros(group_for_knn.index)
      continue

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [h_curve for dist, h_curve in distances[:K_NEIGHBORS]]

    # Aggregate future trajectories and calculate quantiles
    forecast_trajectories_by_horizon_pc: Dict[int, List[float]] = {
        h: [] for h in range(MAX_HORIZON + 1)
    }

    for neighbor in nearest_neighbors:
      historical_per_capita_at_trend_end = neighbor[
          'raw_per_capita_at_trend_end'
      ]

      # Robust scaling factor using EPSILON_STABILITY
      scaling_factor_pc = (
          last_observed_per_capita_value + EPSILON_STABILITY
      ) / (historical_per_capita_at_trend_end + EPSILON_STABILITY)

      # Clamp scaling factor to prevent extreme values, using EPSILON_STABILITY for lower bound consistency
      scaling_factor_pc = max(
          EPSILON_STABILITY, min(MAX_SCALING_FACTOR, scaling_factor_pc)
      )

      # Loop through horizons up to MAX_HORIZON, using stored trajectory
      for h in range(MAX_HORIZON + 1):
        # Only use the historical trajectory if it's long enough for the current horizon 'h'
        if h < len(neighbor['future_trajectory_per_capita']):
          scaled_prediction_pc = (
              neighbor['future_trajectory_per_capita'][h] * scaling_factor_pc
          )
          forecast_trajectories_by_horizon_pc[h].append(scaled_prediction_pc)

    for _, row in group_for_knn.iterrows():
      horizon = row['horizon']

      if (
          horizon in forecast_trajectories_by_horizon_pc
          and forecast_trajectories_by_horizon_pc[horizon]
      ):
        predictions_for_horizon_pc = np.array(
            forecast_trajectories_by_horizon_pc[horizon]
        )
        predictions_for_horizon_pc[predictions_for_horizon_pc < 0] = (
            0  # Ensure non-negative forecasts
        )

        # Convert back to absolute counts by multiplying by population, then subtract pseudo-count
        # Ensure the result is non-negative
        predictions_for_horizon_abs = np.maximum(
            0.0,
            predictions_for_horizon_pc * current_loc_population
            - PSEUDO_COUNT_FOR_ADMISSIONS,
        )

        # Calculate quantiles (floats first)
        current_quantiles_float = np.percentile(
            predictions_for_horizon_abs, [q * 100 for q in QUANTILES]
        )

        # Round to nearest integer (critical for strict monotonicity enforcement on integer output)
        current_quantiles_int = np.round(current_quantiles_float).astype(int)

        # Ensure non-negative
        current_quantiles_int[current_quantiles_int < 0] = 0

        # Ensure monotonicity on the integer values
        for i in range(1, len(current_quantiles_int)):
          if current_quantiles_int[i] < current_quantiles_int[i - 1]:
            current_quantiles_int[i] = current_quantiles_int[i - 1]

        for i, q_val in enumerate(QUANTILES):
          test_y_hat_quantiles.loc[row.name, f'quantile_{q_val}'] = (
              current_quantiles_int[i]
          )
      else:
        # If no predictions for this horizon (e.g., no matching historical trajectories extended far enough), default to 0
        assign_zeros([row.name])

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
