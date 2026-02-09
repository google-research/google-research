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
MODEL_NAME = 'Google_SAI-Adapted_13'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import functools  # Import functools for lru_cache

# TARGET_STR and QUANTILES are defined in the preamble and accessible globally.


# Helper function for cyclic week window with memoization
@functools.lru_cache(maxsize=None)
def get_cyclic_weeks(
    center_week, max_week, window_size = 7
):
  """Calculates a list of week numbers within a cyclic window around a center week,

  handling year boundaries and ISO week numbering (e.g., 52 or 53 weeks).

  Args:
      center_week: The ISO week number to center the window around.
      max_week: The maximum ISO week number for the specific historical year
        (e.g., 52 or 53).
      window_size: The total size of the window (e.g., 7 for +/- 3 weeks).

  Returns:
      A tuple of sorted unique week numbers within the cyclic window.
  """
  half_window = (window_size - 1) // 2
  weeks = set()
  for offset in range(-half_window, half_window + 1):
    candidate_week = center_week + offset
    if candidate_week < 1:
      # Wrap around to the end of the previous "year" (max_week)
      weeks.add(candidate_week + max_week)
    elif candidate_week > max_week:
      # Wrap around to the beginning of the next "year" (week 1)
      weeks.add(candidate_week - max_week)
    else:
      weeks.add(candidate_week)
  return tuple(sorted(list(weeks)))


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  # IMPLEMENTATION PLAN
  # ## Core Principles Checklist:
  # 1. Time Series Awareness: The model explicitly utilizes the target week's ISO week (`target_epiweek`) as the central point for a 7-week cyclic sampling window. This window is applied uniquely for each distinct `target_epiweek` present in the test data, capturing the historical climatology relevant to *that specific target week*. Dynamic `max_week` calculations for each historical ISO year ensure accurate cyclic week handling, adhering to the specified time-series aware approach for seasonality by matching the forecast's target period.
  # 2. Calibration: Probabilistic forecasts are derived directly from empirical historical distributions, rather than parametric assumptions. By calculating quantiles from both geo-specific and geo-aggregated historical observations within a robustly defined, target-week-specific sample window, and by handling cases of sparse or missing historical data gracefully (e.g., filling with 0, using `np.nanmean`), the model aims to produce well-calibrated predictions that reflect observed past variability for the *precise week being forecasted*. The final monotonicity constraint also contributes to well-behaved quantile forecasts.
  # 3. Logging: The model itself is designed to operate silently. No explicit verbose logging is enabled, and `verbose=0` or similar settings are implicitly maintained, fulfilling the requirement to be as quiet as possible during execution.

  # ## Step-by-Step Logic:
  # 1. Initial Data Preparation and Global Dependency Acknowledgment:
  #    - The `fit_and_predict_fn` will *not* use `train_x` and `train_y` for historical sampling. Instead, it will initialize `historical_data_base` by accessing a globally available `dataset` (assumed to be pre-loaded in the execution environment, containing all historical `Total Influenza Admissions`, `target_end_date`, `location`, and `population`). This is necessary to satisfy the "data from 2022 onwards" requirement, as `train_y` may not cover this period for all forecast folds.
  #    - Convert `target_end_date` in `test_x` and `historical_data_base` to datetime objects.
  #    - Add `target_epiweek` to `test_x` and `epiweek`, `year` columns to `historical_data_base` using `dt.isocalendar()`.
  #    - Filter `historical_data_base` to create `historical_data_2022_onwards`, including only records where `target_end_date` is on or after '2022-09-01'.
  #    - If `historical_data_2022_onwards` is empty, return a DataFrame of zeros for all predictions.
  #    - Calculate `rate_per_100k` for `historical_data_2022_onwards`, handling zero populations gracefully.
  #    - Pre-calculate `max_week_by_historical_year` for all unique ISO years in `historical_data_2022_onwards` to correctly handle cyclic weeks.
  #    - Create `historical_data_2022_onwards_indexed` with a MultiIndex `['year', 'epiweek']` for efficient filtering.
  # 2. Iterate through Test Data by `target_epiweek`:
  #    - Group `test_x` by the `target_epiweek` column.
  #    - For each `current_target_epiweek` group:
  #       a. Define Sample Window: Compile `all_sample_year_epiweeks`: a set of `(historical_year, sample_epiweek)` tuples by iterating through `unique_historical_iso_years`, retrieving `max_week` for each year, and using the memoized `get_cyclic_weeks` function.
  #       b. Filter `historical_data_2022_onwards_indexed` using these `(year, epiweek)` tuples to create `sample_window_data` (containing all locations for the relevant historical weeks).
  #       c. Calculate Geo-Aggregated Quantiles:
  #          - Collect `historical_rates_in_window` from `sample_window_data['rate_per_100k']`.
  #          - If empty, initialize `geo_aggregated_rate_quantiles_arr` with NaNs; otherwise, use `np.quantile`.
  #       d. Calculate Geo-Specific Quantiles:
  #          - Identify unique locations in the current `group_df`.
  #          - Filter `sample_window_data` for these specific locations.
  #          - Calculate `calculated_quantiles_per_location` by grouping this filtered data by `location` and applying `quantile(QUANTILES).unstack()` on `Total Influenza Admissions`. Rename columns appropriately.
  #          - Merge `group_df` with `calculated_quantiles_per_location` to create `geo_specific_quantiles_df`, aligning by `location` and preserving `group_df`'s index.
  #       e. Convert Geo-Aggregated Quantiles to Counts:
  #          - Create `geo_aggregated_count_quantiles_df` by vectorizing the conversion of `geo_aggregated_rate_quantiles_arr` to counts for each specific geographic location in `group_df` using its population, ensuring alignment with `group_df`'s index.
  #       f. Combine Forecasts and Post-process for the `target_epiweek` group:
  #          - For each quantile column, compute the `np.nanmean` of corresponding values from `geo_specific_quantiles_df` and `geo_aggregated_count_quantiles_df`.
  #          - Fill any resulting `NaN` values with `0.0`.
  #          - Apply `np.maximum.accumulate` across quantile columns for monotonicity.
  #          - Append the processed `combined_quantiles_for_group` to `all_quantile_preds`.
  # 3. Final Output Assembly:
  #    - Concatenate all `combined_quantiles_for_group` DataFrames.
  #    - If `all_quantile_preds` is empty, return an all-zeros DataFrame matching `test_x`'s index and quantile columns.
  #    - Round all quantile predictions to the nearest integer, convert to `int` type, and clip to non-negative values.
  #    - Ensure column names are correctly set.
  #    - Reindex the final output DataFrame to precisely match the original index order of the input `test_x` DataFrame.

  all_quantile_preds = []

  # Ensure target_end_date and reference_date columns are datetime in test_x
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])
  # Add target_epiweek to test_x for grouping based on the target week
  test_x['target_epiweek'] = (
      test_x['target_end_date'].dt.isocalendar().week.astype(int)
  )

  # Step 1: Data Selection for Historical Samples (Using global 'dataset' as per plan)
  # train_x and train_y are not used for climatological sampling as per method contract.
  historical_data_base = dataset[
      ['target_end_date', 'location', 'population', TARGET_STR]
  ].copy()
  historical_data_base['target_end_date'] = pd.to_datetime(
      historical_data_base['target_end_date']
  )
  historical_data_base['epiweek'] = (
      historical_data_base['target_end_date'].dt.isocalendar().week.astype(int)
  )
  historical_data_base['year'] = (
      historical_data_base['target_end_date'].dt.isocalendar().year.astype(int)
  )

  # Filter historical data from 2022-2023 flu season onwards
  historical_data_2022_onwards = historical_data_base[
      historical_data_base['target_end_date'] >= pd.Timestamp('2022-09-01')
  ].copy()

  # Handle cases where historical_data_2022_onwards might be empty (Robustness)
  if historical_data_2022_onwards.empty:
    preds_df = pd.DataFrame(
        0, index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )
    return preds_df

  # Pre-calculate rates for historical data
  historical_data_2022_onwards['rate_per_100k'] = np.nan
  valid_pop_mask = historical_data_2022_onwards['population'] > 0
  historical_data_2022_onwards.loc[valid_pop_mask, 'rate_per_100k'] = (
      historical_data_2022_onwards.loc[valid_pop_mask, TARGET_STR]
      / historical_data_2022_onwards.loc[valid_pop_mask, 'population']
      * 100000
  )

  # Pre-calculate max_week for each unique historical ISO year
  # FIX: Correctly determine the number of ISO weeks in a year (52 or 53)
  # by checking the ISO week of December 28th, which is always in the last ISO week of its year.
  unique_historical_iso_years = historical_data_2022_onwards['year'].unique()
  max_week_by_historical_year = {
      h_year: pd.Timestamp(year=h_year, month=12, day=28).isocalendar().week
      for h_year in unique_historical_iso_years
  }

  # Create a MultiIndex for efficient lookups on historical data
  historical_data_2022_onwards_indexed = historical_data_2022_onwards.set_index(
      ['year', 'epiweek']
  )

  quantile_cols = [f'quantile_{q}' for q in QUANTILES]

  # Iterate through test_x grouped by target_epiweek
  for current_target_epiweek, group_df in test_x.groupby('target_epiweek'):
    # Define Sample Window using current_target_epiweek
    all_sample_year_epiweeks = set()
    for h_year in unique_historical_iso_years:
      h_max_week = max_week_by_historical_year[h_year]
      sample_epiweeks = get_cyclic_weeks(current_target_epiweek, h_max_week)
      for week in sample_epiweeks:
        all_sample_year_epiweeks.add((h_year, week))

    # Filter historical data once using the combined set of (year, epiweek) tuples
    sample_multi_index = pd.MultiIndex.from_tuples(
        list(all_sample_year_epiweeks), names=['year', 'epiweek']
    )
    sample_window_data = (
        historical_data_2022_onwards_indexed[
            historical_data_2022_onwards_indexed.index.isin(sample_multi_index)
        ]
        .reset_index()
        .copy()
    )

    # Calculate geo-aggregated quantiles for this window once per target_epiweek group
    geo_aggregated_rate_quantiles_arr = np.full(len(QUANTILES), np.nan)
    historical_rates_in_window = (
        sample_window_data['rate_per_100k'].dropna().values
    )

    # Robustness: Calculate quantiles only if there's sufficient historical data
    if len(historical_rates_in_window) > 0:
      geo_aggregated_rate_quantiles_arr = np.quantile(
          historical_rates_in_window, QUANTILES
      )

    # Calculate Geo-Specific Quantiles (Vectorized)
    locations_in_group = group_df['location'].unique()

    # Filter sample_window_data only once for relevant locations within the current target_epiweek group
    sample_window_for_group_locations = sample_window_data[
        sample_window_data['location'].isin(locations_in_group)
    ]

    # Use groupby().quantile().unstack() for vectorized quantile calculation for *available* locations.
    calculated_quantiles_per_location = (
        sample_window_for_group_locations.groupby('location')[TARGET_STR]
        .quantile(QUANTILES)
        .unstack()
    )
    calculated_quantiles_per_location.columns = (
        quantile_cols  # Rename columns directly
    )

    # Merge calculated_quantiles_per_location with group_df to align predictions
    geo_specific_quantiles_df = (
        group_df[['location']]
        .merge(calculated_quantiles_per_location, on='location', how='left')
        .set_index(group_df.index)
    )

    # Ensure only quantile columns are kept, and in correct order
    geo_specific_quantiles_df = geo_specific_quantiles_df[quantile_cols]

    # Convert Geo-Aggregated Quantiles to Counts (Vectorized)
    geo_aggregated_count_quantiles_df = pd.DataFrame(
        np.outer(
            group_df['population'] / 100000, geo_aggregated_rate_quantiles_arr
        ),
        index=group_df.index,
        columns=quantile_cols,
    )

    # Combine Forecasts (Vectorized for all quantile columns at once)
    # np.nanmean on 2D arrays will take the element-wise mean, ignoring NaNs.
    combined_quantiles_for_group_values = np.nanmean(
        [
            geo_specific_quantiles_df.values,
            geo_aggregated_count_quantiles_df.values,
        ],
        axis=0,
    )
    combined_quantiles_for_group = pd.DataFrame(
        combined_quantiles_for_group_values,
        index=group_df.index,
        columns=quantile_cols,
    )

    # Fill any remaining NaNs (where both sources were NaN for a given location/quantile) with 0.0
    combined_quantiles_for_group = combined_quantiles_for_group.fillna(0.0)

    # Ensure monotonicity across quantiles for each row
    combined_quantiles_for_group = pd.DataFrame(
        np.maximum.accumulate(combined_quantiles_for_group.values, axis=1),
        columns=quantile_cols,
        index=group_df.index,
    )

    all_quantile_preds.append(combined_quantiles_for_group)

  # Create DataFrame from collected predictions
  if (
      not all_quantile_preds
  ):  # In case all groups were skipped or no forecasts generated
    return pd.DataFrame(0, index=test_x.index, columns=quantile_cols)

  test_y_hat_quantiles = pd.concat(all_quantile_preds)

  # Apply final post-processing: integer, non-negative
  test_y_hat_quantiles = test_y_hat_quantiles.round().astype(int)
  test_y_hat_quantiles = test_y_hat_quantiles.clip(lower=0)

  # Re-assert column names (already done above, but good for safety and explicit adherence)
  test_y_hat_quantiles.columns = quantile_cols

  # Ensure the output index matches the input test_x index order
  test_y_hat_quantiles = test_y_hat_quantiles.reindex(test_x.index)

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
