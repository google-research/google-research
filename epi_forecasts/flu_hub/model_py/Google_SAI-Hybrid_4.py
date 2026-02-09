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
MODEL_NAME = 'Google_SAI-Hybrid_4'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd


# Helper function to get lagged rates and growth factors robustly (NEW helper for this solution)
def _get_lagged_features_and_growth(
    df, rate_col, num_lags = 2
):
  """Calculates lagged rates and a 1-week growth factor for each location.

  Growth factor for week T is rate(T) / rate(T-1).
  """
  df_sorted = df.sort_values(['location', 'target_end_date']).copy()

  # Calculate current rate as 'lag_0' for clarity in growth calculation
  df_sorted[f'{rate_col}_lag_0'] = df_sorted[rate_col]

  # Calculate lagged rates
  for i in range(1, num_lags + 1):
    df_sorted[f'{rate_col}_lag_{i}'] = df_sorted.groupby('location')[
        rate_col
    ].shift(i)

  # Calculate 1-week growth factor: rate(T) / rate(T-1)
  # Robustly handle division by zero/NaN for growth factor
  # Large value for "infinite" growth from 0 to >0. This occurs when rate(T-1)=0 and rate(T)>0.
  LARGE_GROWTH_VAL = 1000.0

  df_sorted[f'{rate_col}_growth_1wk'] = df_sorted.apply(
      lambda r: LARGE_GROWTH_VAL
      if (pd.isna(r[f'{rate_col}_lag_1']) or r[f'{rate_col}_lag_1'] == 0)
      and r[f'{rate_col}_lag_0'] > 0
      else (
          1.0
          if (pd.isna(r[f'{rate_col}_lag_1']) or r[f'{rate_col}_lag_1'] == 0)
          and r[f'{rate_col}_lag_0'] == 0
          else (
              0.0
              if r[f'{rate_col}_lag_1'] > 0 and r[f'{rate_col}_lag_0'] == 0
              else r[f'{rate_col}_lag_0'] / r[f'{rate_col}_lag_1']
          )
      ),  # Standard ratio: rate(T)/rate(T-1)
      axis=1,
  )

  # Fill remaining NaNs from beginning of series for growth and lags
  # For initial weeks, lags might be NaN. Assume 0 rate. Growth 1.0 (no change).
  for i in range(num_lags + 1):  # Include lag_0 (current week's rate)
    df_sorted[f'{rate_col}_lag_{i}'] = df_sorted[f'{rate_col}_lag_{i}'].fillna(
        0
    )
  df_sorted[f'{rate_col}_growth_1wk'] = df_sorted[
      f'{rate_col}_growth_1wk'
  ].fillna(
      1.0
  )  # Assume no growth at start

  return df_sorted


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a hybrid strategy leveraging pattern matching."""

  # --- Configuration Constants for this function ---
  WINDOW_SIZE = 3  # Look +/- 3 weeks around the target epiweek for analogues
  MIN_SAMPLES_FOR_PREDICTION = (
      5  # Minimum samples to use history directly for quantile calculation
  )
  MIN_SAMPLES_FOR_LOCATION_SCALING = (
      5  # Minimum samples to calculate location-specific scaling factor
  )
  LEVEL_SIMILARITY_THRESHOLD = 0.5  # Relative difference for rate levels (50%)
  GROWTH_SIMILARITY_THRESHOLD = (
      0.75  # Relative difference for growth factors (75%)
  )
  _TARGET_RATE_COL = f'{TARGET_STR}_rate_per_100k'

  # Ensure date columns are datetime objects and add epiweek/year
  for df_name in ['train_x', 'test_x']:
    df = locals()[df_name]
    if 'target_end_date' in df.columns:
      df['target_end_date'] = pd.to_datetime(df['target_end_date'])
      df['epiweek'] = df['target_end_date'].dt.isocalendar().week.astype(int)
      df['year'] = df['target_end_date'].dt.isocalendar().year.astype(int)
    # test_x also has 'reference_date' which is already datetime (from _create_fold_scaffold)
    if 'reference_date' in df.columns and pd.api.types.is_object_dtype(
        df['reference_date']
    ):
      df['reference_date'] = pd.to_datetime(df['reference_date'])

  # Process ilinet_state globally available dataframe (assumed to be global)
  # Create a copy to avoid modifying global state directly.
  ilinet_state_copy = globals().get('ilinet_state', pd.DataFrame()).copy()

  # Also, locations dataframe is global
  locations_df = globals().get('locations', pd.DataFrame()).copy()

  if ilinet_state_copy.empty or locations_df.empty:
    # If ilinet_state or locations are not available (e.g. during local testing without all data loaded)
    warnings.warn(
        'ILINet or Locations data not found globally. Proceeding without ILINet'
        ' augmentation.'
    )

    # Fallback to a scenario where no ILINet data is used.
    # This will result in only actual_admissions_df being used for historical data.
    actual_admissions_df = train_x.copy()
    actual_admissions_df[TARGET_STR] = train_y
    actual_admissions_df[_TARGET_RATE_COL] = (
        (
            actual_admissions_df[TARGET_STR]
            / actual_admissions_df['population']
            * 100_000
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    full_historical_rates_df = actual_admissions_df.copy()

    # Pre-calculate lagged features for the full historical data (which is just actual in this fallback)
    full_historical_rates_df_with_lags = _get_lagged_features_and_growth(
        full_historical_rates_df, _TARGET_RATE_COL
    )
    full_historical_rates_df_with_lags = (
        full_historical_rates_df_with_lags.set_index(
            ['location', 'target_end_date']
        ).sort_index()
    )

    actual_admissions_df_with_lags = (
        full_historical_rates_df_with_lags  # They are the same in fallback
    )

  else:  # Normal path with ILINet augmentation
    ilinet_state_copy['week_start'] = pd.to_datetime(
        ilinet_state_copy['week_start']
    )
    ilinet_state_copy['epiweek'] = (
        ilinet_state_copy['week_start'].dt.isocalendar().week.astype(int)
    )
    ilinet_state_copy['year'] = (
        ilinet_state_copy['week_start'].dt.isocalendar().year.astype(int)
    )

    # --- 1. Data Preparation and Augmentation (Code 2 principles: rates, hierarchical scaling) ---

    # A. Process ILINet data (ilinet_state)
    ilinet_state_processed = ilinet_state_copy[
        ilinet_state_copy['region_type'] == 'States'
    ].copy()
    ilinet_state_processed = ilinet_state_processed.rename(
        columns={'region': 'location_name', 'week_start': 'target_end_date'}
    )

    ilinet_state_processed = ilinet_state_processed.merge(
        locations_df[['location', 'location_name', 'population']],
        on='location_name',
        how='left',
    )
    ilinet_state_processed = ilinet_state_processed.dropna(
        subset=['location', 'population']
    )

    # Calculate ILI rate per 100k. Handle potential division by zero for population.
    ilinet_state_processed['ilitotal_rate_per_100k'] = (
        (
            ilinet_state_processed['ilitotal']
            / ilinet_state_processed['population']
            * 100_000
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # B. Combine train_x and train_y for actual admissions
    actual_admissions_df = train_x.copy()
    actual_admissions_df[TARGET_STR] = train_y
    # Calculate actual admissions rate per 100k. Handle potential division by zero for population.
    actual_admissions_df[_TARGET_RATE_COL] = (
        (
            actual_admissions_df[TARGET_STR]
            / actual_admissions_df['population']
            * 100_000
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # C. Learn ILI to Admissions Rate Scaling Factors (Hierarchical)
    overlap_data = pd.merge(
        actual_admissions_df[['target_end_date', 'location', _TARGET_RATE_COL]],
        ilinet_state_processed[
            ['target_end_date', 'location', 'ilitotal_rate_per_100k']
        ],
        on=['target_end_date', 'location'],
        how='inner',
    )

    # Filter out rows where either rate is zero or NaN for ratio calculation
    overlap_data_valid = overlap_data[
        (overlap_data[_TARGET_RATE_COL] > 0)
        & (overlap_data['ilitotal_rate_per_100k'] > 0)
    ].copy()

    # Calculate ratio only where ilitotal_rate_per_100k is not zero to avoid division by zero
    overlap_data_valid['ili_to_admissions_rate_ratio'] = (
        overlap_data_valid[_TARGET_RATE_COL]
        / overlap_data_valid['ilitotal_rate_per_100k']
    )

    location_scaling_factors = {}
    global_median_ratio = (
        overlap_data_valid['ili_to_admissions_rate_ratio'].median()
        if not overlap_data_valid.empty
        else 1.0
    )
    if pd.isna(global_median_ratio) or global_median_ratio == 0:
      global_median_ratio = 1.0  # Ultimate fallback

    for loc in locations_df['location'].unique():
      loc_data = overlap_data_valid[overlap_data_valid['location'] == loc]
      if len(loc_data) >= MIN_SAMPLES_FOR_LOCATION_SCALING:
        loc_median_ratio = loc_data['ili_to_admissions_rate_ratio'].median()
        location_scaling_factors[loc] = (
            loc_median_ratio
            if pd.notna(loc_median_ratio) and loc_median_ratio > 0
            else global_median_ratio
        )
      else:
        location_scaling_factors[loc] = global_median_ratio

    # D. Generate Synthetic Admissions Rates for ALL ILINet history
    synthetic_admissions_df_full = ilinet_state_processed.copy()
    synthetic_admissions_df_full[_TARGET_RATE_COL] = 0.0  # Initialize

    for loc in ilinet_state_processed['location'].unique():
      factor = location_scaling_factors.get(loc, global_median_ratio)
      mask = synthetic_admissions_df_full['location'] == loc
      synthetic_admissions_df_full.loc[mask, _TARGET_RATE_COL] = (
          synthetic_admissions_df_full.loc[mask, 'ilitotal_rate_per_100k']
          * factor
      )

    # Ensure non-negative rates
    synthetic_admissions_df_full[_TARGET_RATE_COL] = np.maximum(
        0, synthetic_admissions_df_full[_TARGET_RATE_COL]
    )

    # E. Construct Full Historical Admissions Rates (Actual + Synthetic)
    # Prioritize actual admissions data where available
    full_historical_rates_df = pd.concat(
        [
            actual_admissions_df[[
                'target_end_date',
                'location',
                'location_name',
                'population',
                'epiweek',
                'year',
                _TARGET_RATE_COL,
            ]],
            synthetic_admissions_df_full[[
                'target_end_date',
                'location',
                'location_name',
                'population',
                'epiweek',
                'year',
                _TARGET_RATE_COL,
            ]],
        ],
        ignore_index=True,
    )

    # Drop duplicates based on date and location, keeping actual data (`keep='first'`)
    full_historical_rates_df = full_historical_rates_df.drop_duplicates(
        subset=['target_end_date', 'location'], keep='first'
    )

    # Pre-calculate lagged features for the full historical data
    full_historical_rates_df_with_lags = _get_lagged_features_and_growth(
        full_historical_rates_df, _TARGET_RATE_COL
    )
    full_historical_rates_df_with_lags = (
        full_historical_rates_df_with_lags.set_index(
            ['location', 'target_end_date']
        ).sort_index()
    )

    # Also pre-calculate lagged features for actual data for current state extraction
    actual_admissions_df_with_lags = _get_lagged_features_and_growth(
        actual_admissions_df, _TARGET_RATE_COL
    )
    actual_admissions_df_with_lags = actual_admissions_df_with_lags.set_index(
        ['location', 'target_end_date']
    ).sort_index()

  # --- 2. Forecast Generation (Hybrid: Pattern matching on rate data) ---

  test_y_hat_quantiles = pd.DataFrame(
      index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  # Get the globally available get_cyclic_weeks function
  _get_cyclic_weeks = globals().get('get_cyclic_weeks')
  if _get_cyclic_weeks is None:
    # Fallback: if get_cyclic_weeks is not globally available, define a basic one
    warnings.warn(
        'get_cyclic_weeks not found globally. Using a simplified local version.'
    )

    def _get_cyclic_weeks(
        current_epiweek, window_size = 3
    ):
      weeks = []
      for i in range(-window_size, window_size + 1):
        week = current_epiweek + i
        if week < 1:
          week += 52
        elif week > 52:
          week -= 52
        weeks.append(week)
      return sorted(list(set(weeks)))

  for idx, row in test_x.iterrows():
    location_code = row['location']
    forecast_target_date = row['target_end_date']
    forecast_population = row['population']
    reference_date = row['reference_date']
    horizon = row['horizon']
    target_epiweek = forecast_target_date.isocalendar().week

    relevant_epiweeks = _get_cyclic_weeks(target_epiweek, WINDOW_SIZE)

    # --- Determine Current State and Trend for Matching ---
    # Latest observed week in training data is reference_date - 1 week.
    # The growth factor ending at (reference_date - 1 week) is what we need.
    train_end_date_for_this_ref = reference_date - pd.Timedelta(weeks=1)

    current_level = 0.0
    current_growth = 1.0  # Default to stable if no data

    if (
        location_code,
        train_end_date_for_this_ref,
    ) in actual_admissions_df_with_lags.index:
      current_level = actual_admissions_df_with_lags.loc[
          (location_code, train_end_date_for_this_ref), _TARGET_RATE_COL
      ]
      current_growth = actual_admissions_df_with_lags.loc[
          (location_code, train_end_date_for_this_ref),
          f'{_TARGET_RATE_COL}_growth_1wk',
      ]

    # If current_level is very low, adjust growth factor to avoid misleadingly high values from small numbers
    if (
        current_level < 0.1 and current_growth > 5
    ):  # e.g. rate 0.01 -> 0.1, growth = 10.0
      current_growth = 1.0  # Treat very small numbers as stable if they just started growing from near 0
    # Ensure non-negative level
    current_level = max(0, current_level)

    # --- Collect Analogue Future Rates (Geo-Specific and National) ---
    geo_specific_analogue_future_rates = []
    national_analogue_future_rates = []

    # Filter historical data for relevant epiweeks and dates strictly before
    # the current forecast reference date, considering the horizon.
    # The analogue's "current state" (h_date) must be such that its future (h_date + horizon)
    # is strictly before the *current* forecast's reference_date.
    # So, h_date < reference_date - horizon.

    analogue_candidates_df = full_historical_rates_df_with_lags[
        (
            full_historical_rates_df_with_lags.index.get_level_values(
                'target_end_date'
            )
            < (reference_date - pd.Timedelta(weeks=horizon))
        )
        & (
            full_historical_rates_df_with_lags['epiweek'].isin(
                relevant_epiweeks
            )
        )
    ].copy()

    for h_loc, h_date in analogue_candidates_df.index:
      h_row = analogue_candidates_df.loc[(h_loc, h_date)]

      # h_date is the date of the analogue's "current level" and "growth"
      h_level = h_row[_TARGET_RATE_COL]  # Rate at h_date
      h_growth = h_row[
          f'{_TARGET_RATE_COL}_growth_1wk'
      ]  # Growth ending at h_date

      # Skip if lagged features are NaN (e.g., at very start of history)
      if pd.isna(h_level) or pd.isna(h_growth):
        continue

      # --- Similarity Check (Robust for near-zero values) ---
      is_level_similar = False
      is_growth_similar = False

      # Level similarity
      if (
          current_level < 0.5 and h_level < 0.5
      ):  # Both very low rates (near zero)
        is_level_similar = True
      elif current_level > 0.05 and h_level > 0.05:  # Both non-negligible rates
        # Relative difference for non-zero rates
        is_level_similar = (
            abs(h_level - current_level) / current_level
            <= LEVEL_SIMILARITY_THRESHOLD
        )
      # If one is near zero and the other is not, they are not similar in level (already handled by thresholds)

      # Growth similarity
      if (
          current_growth == 1.0 and h_growth == 1.0
      ):  # Both stable (e.g., both zero -> zero, or both constant)
        is_growth_similar = True
      elif (
          current_growth > 0.1 and h_growth > 0.1
      ):  # Both experiencing some non-negligible growth/decline
        is_growth_similar = (
            abs(h_growth - current_growth) / current_growth
            <= GROWTH_SIMILARITY_THRESHOLD
        )
      # If one growth is very high (from 0 to >0) and other is not, or one is 0 and other not, not similar

      if is_level_similar and is_growth_similar:
        # The 'future' value we want for this analogue is at h_date + horizon
        future_analogue_date = h_date + pd.Timedelta(weeks=horizon)

        # Fetch the actual rate at that future date for this analogue
        try:
          future_rate_val = full_historical_rates_df_with_lags.loc[
              (h_loc, future_analogue_date), _TARGET_RATE_COL
          ]
          if pd.notna(future_rate_val):
            if h_loc == location_code:
              geo_specific_analogue_future_rates.append(future_rate_val)
            national_analogue_future_rates.append(future_rate_val)
        except KeyError:
          # This historical "future" date might not exist in the DataFrame for this location
          pass

    # --- Calculate Quantiles from Analogue Pools and Blend ---

    # Initialize with NaN to allow nanmean to ignore missing categories
    geo_quantiles_rates = np.full(len(QUANTILES), np.nan)
    national_quantiles_rates = np.full(len(QUANTILES), np.nan)

    if len(geo_specific_analogue_future_rates) >= MIN_SAMPLES_FOR_PREDICTION:
      geo_quantiles_rates = np.quantile(
          geo_specific_analogue_future_rates, QUANTILES
      )

    if len(national_analogue_future_rates) >= MIN_SAMPLES_FOR_PREDICTION:
      national_quantiles_rates = np.quantile(
          national_analogue_future_rates, QUANTILES
      )

    # Blend using nanmean, which will pick the non-NaN array if one is NaN
    blended_predictions_rates = np.nanmean(
        [geo_quantiles_rates, national_quantiles_rates], axis=0
    )

    # Fallback if no sufficient analogues were found (blended_predictions_rates might still be all nan)
    if np.all(np.isnan(blended_predictions_rates)):
      # Use current_level as a crude forecast if no analogues
      blended_predictions_rates = np.full(len(QUANTILES), current_level)

    # Ensure non-negative rates
    blended_predictions_rates = np.maximum(0, blended_predictions_rates)

    # Convert blended rates back to counts
    predictions_counts = (
        blended_predictions_rates / 100_000
    ) * forecast_population

    # --- Post-processing: Ensure non-negativity, integer, and monotonicity ---
    predictions_counts = np.maximum(0, predictions_counts).round().astype(int)
    predictions_counts = np.maximum.accumulate(
        predictions_counts
    )  # Enforce monotonicity

    test_y_hat_quantiles.loc[idx] = predictions_counts

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
