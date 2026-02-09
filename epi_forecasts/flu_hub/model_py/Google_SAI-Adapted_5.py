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
MODEL_NAME = 'Google_SAI-Adapted_5'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings

# Define a small epsilon for numerical stability, to prevent division by zero or near-zero
EPSILON = 1e-9
# Define a maximum cap for scaling factors to prevent overflow when denominators are very small,
# protecting against extreme ratios when ILI totals or percentages are near zero.
MAX_SCALING_FACTOR = 1_000_000
# Define the squared bandwidth for the exponential kernel used in weighting.
# Chosen to provide a reasonable decay for distances (e.g., a distance of 0.1 gives exp(-0.01/0.01) = exp(-1) ~ 0.36).
BANDWIDTH_SQUARED = 0.01

# IMPLEMENTATION PLAN
# ## Core Principles Checklist
# 1. Non-Negotiable Core Principle: "We produce a peak-aligned composite curve epidemic curves across past seasons and regions, normalizing the season for percent of cases each week, and then taking the median percentage across the resulting curves for each number of weeks from the peak, following the approach of Schanzer et al., Influenza and Other Respiratory Viruses 2010."
#    My code will satisfy this by: The `_build_composite_curve` function will calculate the total seasonal incidence for each location-season, normalize weekly counts to percentages, identify the peak week, align all season curves relative to their peaks, and then compute the median weekly percentage across all aligned curves for each relative week.
#
# 2. Non-Negotiable Core Principle: "Then, for each state, we compute the normalized case count for the preceding four weeks, and compare against the normalized four week windows of the composite curve, following the approach of Morel et al. PLOS Computational Biology, 2023."
#    My code will satisfy this by: The `_forecast_for_single_location_and_date` function will extract the most recent four weeks of observed `Total Influenza Admissions` for the target state, normalize these counts into percentages, and then iterate through sliding four-week windows on the composite curve, comparing their patterns using an exponential kernel for distance-based weighting.
#
# 3. Non-Negotiable Core Principle: "sample_method: “Each window along the curve is weighted based on its distance from the most recent window and these weights are used to select 100 windows (allowing repetition) along the curve. These windows are used to produce the 100 sample trajectories by applying the subsequent scaling factors to the observed data."
#    My code will satisfy this by: After calculating weights for each composite curve window based on similarity, `_forecast_for_single_location_and_date` will probabilistically sample 100 windows (with replacement). For each sampled window, a **"estimated total seasonal admissions"** will be determined by comparing the sum of the state's recent observed data to the sum of the sampled composite window's percentages. If the sum of composite window percentages is zero (or near-zero), the estimated total will be zero. This estimated total will then be used to project future weekly incidence from the composite curve to generate 100 distinct future trajectories.
#
# ## Step-by-Step Logic
# 1.  **Define Constants**: Define `EPSILON`, `MAX_SCALING_FACTOR`, `BANDWIDTH_SQUARED` for numerical stability and weight calculation.
# 2.  **`get_season` function**: Determine flu season from a date.
# 3.  **`_prepare_augmented_data` function**:
#     a.  **Date Standardization**: Convert all `target_end_date` and `week_start` columns to `pd.Timestamp`. Crucially, align `ilinet_state['week_start']` (Sunday) to `target_end_date` (Saturday) by adding 6 days.
#     b.  **FIPS Code Handling**: Convert `locations_df['location']` (int) to zero-padded strings to allow correct merging and filtering against `REQUIRED_CDC_LOCATIONS` (strings). Then convert back to `int` for consistency with `dataset`.
#     c.  **`ilitotal` NaN Handling**: Fill NaN values in `ilitotal` with 0 before any calculations.
#     d.  **Scaling Factor Calculation (Location-Season Specific)**:
#         i.  Merge `train_x`/`train_y` (actual admissions) and `ilinet_processed_df` (raw ILI) on overlapping dates and locations.
#         ii. For each `(location, season)` group in the overlap, calculate median `admissions / ilitotal` ratio for points where `ilitotal > EPSILON`.
#         iii. If `np.median(ratios)` results in `NaN` (no relevant data), the factor is set to `None` to enable fallback.
#         iv. Store these as `location_season_factors`.
#     e.  **Scaling Factor Calculation (Fallback)**:
#         i.  Calculate `location_fallback_factors` as the median of valid `location_season_factors` for each location.
#         ii. Calculate `global_final_scaling_factor` as the median of valid `location_fallback_factors`. If still no positive factor, default to `0.001` with a warning.
#     f.  **Apply Scaling**: Apply the most specific valid scaling factor (location-season > location-wide > global) to `ilinet_processed_df['ilitotal']` to create `scaled_ilitotal`.
#     g.  **Synthetic Admissions**: Convert `scaled_ilitotal` to non-negative integers as `synthetic_admissions`.
#     h.  **Combine Data**: Concatenate `actual_train_df` and `synthetic_df`, dropping duplicates to prioritize actual data where available.
# 4.  **`_build_composite_curve` function**:
#     a.  Iterate over `(location, season)` groups.
#     b.  For each group, calculate `total_seasonal_incidence`.
#     c.  If `total_seasonal_incidence` is zero, return an empty DataFrame for that group to avoid division by zero.
#     d.  Calculate `weekly_percentage` by dividing `target_str` by `total_seasonal_incidence + EPSILON`.
#     e.  Identify `peak_week_date` (highest `weekly_percentage`).
#     f.  Calculate `weeks_from_peak`.
#     g.  Aggregate all `weeks_from_peak` and `weekly_percentage` values and compute the median `weekly_percentage` for each `weeks_from_peak` to form the `composite_curve`.
#     h.  **Silence Pandas Warning**: Add `include_groups=False` to the `.apply()` call.
# 5.  **`_calculate_distance_for_composite_window` function**:
#     a.  Check if the 4-week `composite_window` is fully within the `composite_curve`'s valid index range. Return `np.inf` if not.
#     b.  Normalize the composite window percentages by their sum. Add `EPSILON` to the denominator for stability. Return `np.inf` if sum is near zero.
#     c.  Calculate Euclidean distance between the `current_window_normalized` and the `composite_window_normalized`.
# 6.  **`_forecast_for_single_location_and_date` function**:
#     a.  Extract the most recent four weeks of `Total Influenza Admissions` (`past_four_weeks`) *before* the `reference_date`.
#     b.  Handle Edge Cases: If `past_four_weeks` has less than 4 entries or its sum is 0, predict all zeros and return. Also, handle empty `possible_composite_window_starts` by predicting zeros.
#     c.  Normalize `current_window_actual_counts` to `current_window_normalized`.
#     d.  Iterate through `possible_composite_window_starts` on the composite curve, considering `temporal_shifts = [-1, 0, 1]`.
#     e.  For each potential start, calculate `min_distance` using `_calculate_distance_for_composite_window`.
#     f.  Calculate `weights` using an exponential kernel: `np.exp(-min_distance**2 / BANDWIDTH_SQUARED)`.
#     g.  **Normalize Weights**: If `np.sum(weights)` is zero (e.g., all distances were infinite), fallback to uniform weights and normalize. Otherwise, normalize existing weights.
#     h.  **Sample Trajectories**: Use `np.random.choice` with `p=weights` to sample `num_samples=100` `sampled_start_relative_weeks`.
#     i.  For each sampled `start_relative_week`:
#         i.  Determine the sum of percentages in the corresponding 4-week composite window: `sum_comp_window_perc`.
#         ii. If `sum_comp_window_perc` is less than `EPSILON`, set `estimated_total_seasonal_admissions = 0.0`.
#         iii. Otherwise, calculate `estimated_total_seasonal_admissions = np.clip(sum_current_window_actual_counts / sum_comp_window_perc, 0.0, MAX_SCALING_FACTOR)`.
#         iv. Project future weekly incidence by taking the sequence of median percentages from the composite curve, starting from the week immediately following the selected four-week window, and multiplying them by `estimated_total_seasonal_admissions`.
#     j.  **Generate Quantiles**: For each horizon, calculate `np.percentile` from the collected trajectories.
#     k.  **Enforce Monotonicity (Float)**: Apply `np.maximum.accumulate` to the float quantiles.
#     l.  **Round and Re-Enforce Monotonicity (Integer)**: Round the quantiles to `int`, then apply `np.maximum.accumulate` *again* to ensure integer monotonicity after rounding.
# 7.  **`fit_and_predict_fn` orchestration**:
#     a.  Call `_prepare_augmented_data`.
#     b.  Call `_build_composite_curve`. Handle an empty composite curve by predicting zeros.
#     c.  Convert `test_x['reference_date']` to `pd.Timestamp`.
#     d.  Group `test_x` by `location` and `reference_date_dt`.
#     e.  Loop through groups, calling `_forecast_for_single_location_and_date`.
#     f.  Collect results into `test_y_hat_quantiles`.
#     g.  Return `test_y_hat_quantiles`.


def get_season(date_obj):
  """Determines the flu season for a given date (e.g., '2020/2021').

  A season starts in October of the first year and ends in September of the
  second year. Accepts pd.Timestamp objects.
  """
  if date_obj.month >= 10:  # Oct-Dec
    return f'{date_obj.year}/{date_obj.year + 1}'
  else:  # Jan-Sep
    return f'{date_obj.year - 1}/{date_obj.year}'


def _prepare_augmented_data(
    train_x,
    train_y,
    ilinet_state_df_raw,
    locations_df_raw,
    required_locations,  # This list contains FIPS codes as strings, e.g., '01'
    target_str,
    ilinet_max_date,
):
  """Preprocesses and augments training data with scaled ILINet data.

  This function implements the "Learn a Transformation" strategy to make
  historical ILINet data comparable to 'Total Influenza Admissions' using
  a hierarchical scaling factor approach based on median ratios.
  """

  locations_df = locations_df_raw.copy()

  # 1. Preprocess ilinet_state for augmentation
  ilinet_processed_df = ilinet_state_df_raw[
      ilinet_state_df_raw['region_type'] == 'States'
  ].copy()

  # CRITICAL FIX: Align ilinet_state's 'week_start' (typically Sunday) to 'target_end_date' (Saturday).
  # This ensures consistency with the primary 'dataset' target_end_date.
  ilinet_processed_df['target_end_date'] = pd.to_datetime(
      ilinet_processed_df['week_start']
  ) + pd.Timedelta(days=6)

  ilinet_processed_df.rename(columns={'region': 'location_name'}, inplace=True)

  # Convert 'location' in locations_df (int, e.g., 1) to zero-padded string (e.g., '01')
  locations_df['location_fips_str'] = (
      locations_df['location'].astype(str).str.zfill(2)
  )

  # Merge with locations to get FIPS codes (as zero-padded strings).
  ilinet_processed_df = ilinet_processed_df.merge(
      locations_df[['location_name', 'location_fips_str']],
      on='location_name',
      how='left',
  )
  # Filter to only include the CDC specified locations (FIPS codes as strings, e.g., '01').
  ilinet_processed_df = ilinet_processed_df[
      ilinet_processed_df['location_fips_str'].isin(required_locations)
  ].copy()
  # Convert the filtered FIPS string back to int for the 'location' column, consistent with the main dataset.
  ilinet_processed_df['location'] = ilinet_processed_df[
      'location_fips_str'
  ].astype(int)

  # Fill NaN `ilitotal` values with 0. This is crucial before any sum/division operations.
  ilinet_processed_df['ilitotal'] = ilinet_processed_df['ilitotal'].fillna(0)

  # Assign flu season and filter ILINet data to be before its maximum available date (Oct 15, 2022).
  ilinet_processed_df['season'] = ilinet_processed_df['target_end_date'].apply(
      get_season
  )
  ilinet_processed_df = ilinet_processed_df[
      ilinet_processed_df['target_end_date'] < ilinet_max_date
  ].copy()
  ilinet_processed_df = ilinet_processed_df[
      ['target_end_date', 'location', 'location_name', 'ilitotal', 'season']
  ]

  # 2. Combine train_x and train_y into a single DataFrame for actual admissions.
  actual_train_df = pd.merge(
      train_x, train_y.rename(target_str), left_index=True, right_index=True
  )
  # Ensure target_end_date is datetime for consistency with other date columns.
  actual_train_df['target_end_date'] = pd.to_datetime(
      actual_train_df['target_end_date']
  )
  actual_train_df['season'] = actual_train_df['target_end_date'].apply(
      get_season
  )
  actual_train_df = actual_train_df[
      ['target_end_date', 'location', 'location_name', target_str, 'season']
  ]

  # 3. Learn Scaling Factors for ILINet Data using an overlap period via Median Ratio.
  min_actual_train_date = actual_train_df['target_end_date'].min()
  overlap_admissions = actual_train_df[
      (actual_train_df['target_end_date'] >= min_actual_train_date)
      & (actual_train_df['target_end_date'] < ilinet_max_date)
  ]
  overlap_ilitotal = ilinet_processed_df[
      (ilinet_processed_df['target_end_date'] >= min_actual_train_date)
      & (ilinet_processed_df['target_end_date'] < ilinet_max_date)
  ]

  overlap_combined_for_regression = pd.merge(
      overlap_admissions,
      overlap_ilitotal[['target_end_date', 'location', 'season', 'ilitotal']],
      on=['target_end_date', 'location', 'season'],
      how='inner',
  )

  # Initialize factors.
  location_season_factors = (
      {}
  )  # Stores (location, season) -> regression coefficient
  location_fallback_factors = (
      {}
  )  # Stores location -> median of season-specific coefficients

  # a. Calculate Location-Season Specific Factors using enhanced logic for robustness.
  for (loc, season), group in overlap_combined_for_regression.groupby(
      ['location', 'season']
  ):

    # Filter to relevant points for ratio calculation: ILI must be positive to form a meaningful ratio.
    relevant_data = group[
        group['ilitotal'] > EPSILON
    ]  # Use EPSILON for robustness against very small ILI values.

    current_factor = None  # Default factor to None

    if not relevant_data.empty:
      ratios = relevant_data[target_str] / relevant_data['ilitotal']
      median_ratio = np.median(ratios)

      if not np.isnan(median_ratio):
        current_factor = np.clip(
            np.maximum(0, median_ratio), 0.0, MAX_SCALING_FACTOR
        )

    # If no relevant data or median_ratio was NaN, current_factor remains None.
    location_season_factors[(loc, season)] = current_factor

  # b. Calculate Location-Wide Fallback Factors: Median of valid season-specific factors for that location.
  for loc_id in ilinet_processed_df['location'].unique():
    loc_season_valid_factors = [
        f
        for (l, s), f in location_season_factors.items()
        if l == loc_id
        and f is not None
        and f > 0  # Only consider positive factors for median.
    ]

    if loc_season_valid_factors:
      location_fallback_factors[loc_id] = np.median(loc_season_valid_factors)
    else:
      location_fallback_factors[loc_id] = (
          None  # No valid factors for this location.
      )

  # c. Calculate Global Fallback Factor.
  global_final_scaling_factor = np.nan
  all_valid_location_factors = [
      f for f in location_fallback_factors.values() if f is not None and f > 0
  ]

  if all_valid_location_factors:
    global_final_scaling_factor = np.median(all_valid_location_factors)

  if np.isnan(global_final_scaling_factor) or global_final_scaling_factor <= 0:
    # If still NaN or non-positive after all checks, assign a safe positive default.
    # This can happen if all learned factors are zero or None.
    global_final_scaling_factor = 0.001
    warnings.warn(
        'No valid positive scaling factors could be calculated for any location'
        ' or season. Using a global default scaling factor of 0.001 as a last'
        ' resort.'
    )

  # 4. Create Augmented Historical Data ('synthetic_admissions').
  # Apply the determined scaling factors to the historical ILITotal data.
  ilinet_processed_df['scaling_key'] = list(
      zip(ilinet_processed_df['location'], ilinet_processed_df['season'])
  )
  # Use map for efficient application of scaling factors.
  final_scaling_factors_map = {}
  for loc_id in ilinet_processed_df['location'].unique():
    for season_str in ilinet_processed_df['season'].unique():
      factor_key = (loc_id, season_str)

      # 1. Try location-season specific factor (from regression).
      factor = location_season_factors.get(factor_key)

      # 2. Fallback to location-wide factor if season-specific is None or <= 0.
      if factor is None or factor <= 0:
        factor = location_fallback_factors.get(loc_id)

      # 3. Fallback to global factor if location-wide is None or <= 0.
      if factor is None or factor <= 0:
        factor = global_final_scaling_factor

      # Clip the final factor to prevent extreme values and ensure non-negativity.
      final_scaling_factors_map[factor_key] = np.clip(
          factor, 0.0, MAX_SCALING_FACTOR
      )

  ilinet_processed_df['scaled_ilitotal'] = (
      ilinet_processed_df['scaling_key'].map(final_scaling_factors_map)
      * ilinet_processed_df['ilitotal']
  )

  # Ensure synthetic_admissions are non-negative and converted to integers, as admissions are counts.
  ilinet_processed_df['synthetic_admissions'] = np.maximum(
      0, ilinet_processed_df['scaled_ilitotal']
  ).astype(int)

  synthetic_df = ilinet_processed_df[[
      'target_end_date',
      'location',
      'location_name',
      'synthetic_admissions',
      'season',
  ]].rename(columns={'synthetic_admissions': target_str})

  # 5. Combine actual and synthetic data, prioritizing actual data where available.
  full_historical_data = pd.concat([actual_train_df, synthetic_df])

  full_historical_data = full_historical_data.sort_values(
      by=['target_end_date', 'location'], ascending=[False, True]
  )
  full_historical_data.drop_duplicates(
      subset=['target_end_date', 'location'], keep='first', inplace=True
  )
  full_historical_data = full_historical_data.sort_values(
      by=['location', 'target_end_date']
  ).reset_index(drop=True)

  return full_historical_data


def _build_composite_curve(
    full_historical_data, target_str
):
  """Constructs the peak-aligned composite curve from historical data.

  Refactored to use groupby().apply() for improved performance.
  """

  def _process_single_season_curve(group_df):
    """Helper function to process a single season's data for composite curve building."""
    total_seasonal_incidence = group_df[target_str].sum()

    # Skip seasons with zero total incidence to avoid distortion and division by zero.
    if total_seasonal_incidence == 0:
      return pd.DataFrame()  # Return empty DataFrame if no incidence

    # Calculate weekly percentage of total seasonal incidence, adding EPSILON for numerical stability.
    group_df['weekly_percentage'] = group_df[target_str] / (
        total_seasonal_incidence + EPSILON
    )

    # Identify the peak week based on the highest weekly percentage.
    # Use .idxmax() which handles multiple maximums by taking the first one.
    peak_week_date = group_df.loc[
        group_df['weekly_percentage'].idxmax(), 'target_end_date'
    ]

    # Calculate weeks relative to the peak week (0 at peak, negative before, positive after).
    group_df['weeks_from_peak'] = (
        (group_df['target_end_date'] - peak_week_date).dt.days // 7
    ).astype(int)

    return group_df[['weeks_from_peak', 'weekly_percentage']]

  # Apply the helper function to each (location, season) group.
  # `include_groups=False` suppresses a FutureWarning in pandas.
  season_curves_data = full_historical_data.groupby(
      ['location', 'season'], group_keys=False
  ).apply(_process_single_season_curve, include_groups=False)

  # If no valid season curves could be generated (e.g., all seasons had zero incidence), return an empty series.
  if season_curves_data.empty:
    return pd.Series(dtype=float)

  # Compute the median percentage for each 'weeks_from_peak' from all processed season curves.
  composite_curve = season_curves_data.groupby('weeks_from_peak')[
      'weekly_percentage'
  ].median()
  composite_curve = composite_curve.sort_index()

  return composite_curve


def _calculate_distance_for_composite_window(
    current_window_normalized,
    composite_curve,
    composite_window_start_idx,  # This is the start of the 4-week window on the composite curve
    window_length = 4,
):
  """Calculates Euclidean distance between a normalized current window and a normalized composite window."""
  min_comp_idx = composite_curve.index.min()
  max_comp_idx = composite_curve.index.max()

  # Check if the entire 4-week window is within the composite curve boundaries.
  if (
      composite_window_start_idx < min_comp_idx
      or composite_window_start_idx + window_length - 1 > max_comp_idx
  ):
    return np.inf  # Return infinity if the window is out of bounds.

  # Extract the composite window percentages.
  composite_window_percentages = composite_curve.loc[
      composite_window_start_idx : composite_window_start_idx
      + window_length
      - 1
  ]
  sum_composite_window_percentages = composite_window_percentages.sum()

  # Normalize the composite window if its sum is sufficiently non-zero.
  if (
      sum_composite_window_percentages > EPSILON
  ):  # Use EPSILON for stability to avoid division by near-zero.
    composite_window_normalized = (
        composite_window_percentages / sum_composite_window_percentages
    )
    # Calculate Euclidean distance between the two normalized windows.
    return np.sqrt(
        np.sum(
            (current_window_normalized - composite_window_normalized.values)
            ** 2
        )
    )
  return (
      np.inf
  )  # If composite window sums to zero or near-zero, it's not a good match for pattern comparison.


def _forecast_for_single_location_and_date(
    location,
    reference_date,
    test_x_group_for_location,
    full_historical_data,
    composite_curve,
    target_str,
    quantiles,
):
  """Generates quantile forecasts for a single location and reference date."""

  results = pd.DataFrame(
      index=test_x_group_for_location.index,
      columns=[f'quantile_{q}' for q in quantiles],
  )

  # Get Current State Data: the last 4 weeks of observed data *before* the reference_date.
  # The 'target_end_date' in full_historical_data is already pd.Timestamp.
  past_four_weeks = (
      full_historical_data[
          (full_historical_data['location'] == location)
          & (full_historical_data['target_end_date'] < reference_date)
      ]
      .sort_values('target_end_date')
      .tail(4)
  )

  current_window_actual_counts = past_four_weeks[target_str]
  sum_current_window_actual_counts = current_window_actual_counts.sum()

  # Handle insufficient (less than 4 weeks) or zero current state data by predicting zeros.
  if (
      len(current_window_actual_counts) < 4
      or sum_current_window_actual_counts == 0
  ):
    results.iloc[:] = 0
    return results

  # Normalize the current window's actual counts to get its percentage pattern.
  current_window_normalized = (
      current_window_actual_counts.values / sum_current_window_actual_counts
  )

  # Define the range of possible starting weeks on the composite curve for comparison.
  # A 4-week window ending at max_composite_week requires starting at max_composite_week - 3.
  min_composite_week = composite_curve.index.min()
  max_composite_week = composite_curve.index.max()

  possible_composite_window_starts = composite_curve.index[
      (composite_curve.index >= min_composite_week)
      & (
          composite_curve.index <= max_composite_week - 3
      )  # Ensure entire 4-week window fits
  ]

  # If no valid 4-week composite windows can be formed, predict zeros.
  if possible_composite_window_starts.empty:
    warnings.warn(
        'No valid 4-week composite curve windows can be formed for location'
        f' {location} from reference date {reference_date}. Predicting zeros.'
    )
    results.iloc[:] = 0
    return results

  weights = []
  # Temporal shifts for the composite window relative to its 'start_idx' to account for slight phase differences.
  temporal_shifts = [-1, 0, 1]

  # Compare current window to each possible composite window, considering temporal shifts.
  for s_comp in possible_composite_window_starts:
    min_distance_for_this_s_comp = np.inf

    for shift_amount in temporal_shifts:
      current_comp_window_start = s_comp + shift_amount

      distance_for_this_shift = _calculate_distance_for_composite_window(
          current_window_normalized, composite_curve, current_comp_window_start
      )
      min_distance_for_this_s_comp = min(
          min_distance_for_this_s_comp, distance_for_this_shift
      )

    # Calculate weight using an exponential kernel for smoother decay and better robustness.
    weights.append(
        np.exp(-(min_distance_for_this_s_comp**2) / BANDWIDTH_SQUARED)
    )

  weights = np.array(weights)
  # Handle the edge case where all distances are infinite or weights sum to zero (e.g., no good matches).
  # In this case, assign uniform weights as a fallback.
  if np.sum(weights) == 0:
    warnings.warn(
        f'All composite windows have infinite distance for location {location}'
        f' from reference date {reference_date}. Assigning uniform weights.'
    )
    weights = np.ones_like(
        possible_composite_window_starts.values, dtype=float
    )  # All weights are 1.0
    weights = weights / np.sum(weights)  # Normalize uniform weights
  else:
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1.

  # Sample Trajectories and Generate Forecasts.
  forecast_trajectories_by_horizon = {
      h: [] for h in test_x_group_for_location['horizon'].unique()
  }

  num_samples = 100
  # Sample starting weeks from the composite curve based on the calculated weights.
  # We sample indices first, then map them to the actual relative week values.
  # Explicitly convert possible_composite_window_starts to a NumPy array for np.random.choice robustness.
  sampled_s_comp_values = np.random.choice(
      possible_composite_window_starts.values,
      num_samples,
      p=weights,
      replace=True,
  )

  for start_relative_week in sampled_s_comp_values:
    # Get the composite curve percentages for the 4-week window that matched.
    composite_window_percentages_for_scaling = composite_curve.loc[
        start_relative_week : start_relative_week + 3
    ]

    sum_composite_window_percentages_for_scaling = (
        composite_window_percentages_for_scaling.sum()
    )

    # IMPROVED: Calculate estimated_total_seasonal_admissions based on the observed data and matched composite window.
    estimated_total_seasonal_admissions = (
        0.0  # Default to 0 if composite window shows no activity
    )
    if sum_composite_window_percentages_for_scaling > EPSILON:
      # If the matched composite window has activity, estimate the total seasonal admissions
      # by scaling the sum of current actual counts by the sum of composite percentages.
      estimated_total_seasonal_admissions = np.clip(
          sum_current_window_actual_counts
          / sum_composite_window_percentages_for_scaling,
          0.0,
          MAX_SCALING_FACTOR,
      )

    for horizon in test_x_group_for_location['horizon'].unique():
      # Calculate the relative week in the composite curve corresponding to the forecast horizon.
      # Based on the problem description:
      # -1: week preceding reference_date (corresponds to sampled_start_relative_week + 3)
      # 0: current epiweek (corresponds to sampled_start_relative_week + 4)
      # 1, 2, 3: next epiweeks
      forecast_relative_week = start_relative_week + 4 + horizon

      # Retrieve the percentage from the composite curve for the forecast week.
      # Use .get with a default value of 0.0 if the week is outside the composite curve's range.
      forecast_percentage = composite_curve.get(forecast_relative_week, 0.0)

      # Project the future value by multiplying the estimated total seasonal admissions
      # by the composite curve's percentage for that week.
      sampled_forecast_value = (
          estimated_total_seasonal_admissions * forecast_percentage
      )

      # Ensure forecasts are non-negative.
      forecast_trajectories_by_horizon[horizon].append(
          np.maximum(0, sampled_forecast_value)
      )

  # Calculate quantiles for each horizon from the generated trajectories and enforce monotonicity.
  for idx in test_x_group_for_location.index:
    horizon = test_x_group_for_location.loc[idx, 'horizon']
    if not forecast_trajectories_by_horizon[horizon]:
      # Safeguard if no trajectories were generated (should not happen with num_samples > 0).
      quantiles_for_horizon_float = np.array([0.0] * len(quantiles))
    else:
      quantiles_for_horizon_float = np.percentile(
          forecast_trajectories_by_horizon[horizon],
          [q * 100 for q in quantiles],
      )

    # Enforce monotonicity on float values first.
    quantiles_for_horizon_float = np.maximum.accumulate(
        quantiles_for_horizon_float
    )

    # Round forecasts to the nearest integer as admissions are counts.
    # Ensure that non-negative values are maintained after rounding.
    quantiles_for_horizon = np.round(
        np.maximum(0, quantiles_for_horizon_float)
    ).astype(int)

    # Re-enforce monotonicity after rounding to ensure integer monotonicity.
    quantiles_for_horizon = np.maximum.accumulate(quantiles_for_horizon)

    results.loc[idx] = quantiles_for_horizon

  return results


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Trains a model based on the provided method contract and generates quantile predictions.

  This function orchestrates the data augmentation, composite curve building,
  and forecasting steps.
  """

  # Constants from global scope (assumed to be available).
  # QUANTILES, locations, ilinet_state (raw), REQUIRED_CDC_LOCATIONS, TARGET_STR
  ILINET_MAX_DATE = pd.to_datetime('2022-10-15')

  # 1. Prepare Augmented Historical Data by scaling ILINet data to Admissions scale.
  # This leverages the "Learn a Transformation" strategy.
  full_historical_data = _prepare_augmented_data(
      train_x,
      train_y,
      ilinet_state.copy(),
      locations.copy(),
      REQUIRED_CDC_LOCATIONS,
      TARGET_STR,
      ILINET_MAX_DATE,
  )

  # 2. Construct Peak-Aligned Composite Curve from the augmented historical data.
  # This curve represents the typical shape of a flu season.
  composite_curve = _build_composite_curve(full_historical_data, TARGET_STR)

  # Handle case where composite curve could not be built (e.g., no historical data processed).
  if composite_curve.empty:
    warnings.warn(
        'Composite curve could not be built. Predicting zeros for all'
        ' forecasts.'
    )
    output_df = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return output_df.fillna(0).astype(int)

  # 3. Generate Quantile Predictions for each location and reference date in test_x.
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Convert reference_date in test_x to datetime once for efficiency during grouping and comparisons.
  test_x['reference_date_dt'] = pd.to_datetime(test_x['reference_date'])

  # Group test_x by location and reference date to generate forecasts for each unique forecast point.
  for (location, reference_date_dt), group in test_x.groupby(
      ['location', 'reference_date_dt']
  ):
    forecasts_for_group = _forecast_for_single_location_and_date(
        location=location,
        reference_date=reference_date_dt,
        test_x_group_for_location=group,
        full_historical_data=full_historical_data,
        composite_curve=composite_curve,
        target_str=TARGET_STR,
        quantiles=QUANTILES,
    )
    test_y_hat_quantiles.loc[group.index] = forecasts_for_group

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
