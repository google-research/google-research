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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,reimported,missing-function-docstring

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
MODEL_NAME = 'Google_SAI-Adapted_24'
TARGET_STR = 'Total Influenza Admissions'
# Alias matching the global defined in the Kaggle notebooks the models were
# developed in; some fit_and_predict_fn implementations reference it directly.
horizons = HORIZONS

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Core Principles of the Method ---
# 1. Proxy Signal Calibration: Leverage historical ILINet data by learning a location-specific
#    scaling factor to approximate the target (Admissions), creating a "synthetic" history.
# 2. Trend-Aware Analog Ensemble: Forecast by projecting the current value using a weighted
#    blend of the recent observed trend (momentum) and historical seasonal growth rates.
# 3. Neighboring Week Sampling: Account for seasonal timing uncertainty by sampling historical
#    growth rates from a temporal window around the current week (smoothing).
# 4. Parametric Distribution Fitting: Generate quantiles using a Log-Normal distribution
#    parameterized by the moments of the ensemble projections.


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):

  # -------------------------------------------------------------------------
  # 1. Setup and Pre-computation
  # -------------------------------------------------------------------------
  # Merge target into train for easier handling
  train = train_x.copy()
  train[TARGET_STR] = train_y.values
  train['target_end_date'] = pd.to_datetime(train['target_end_date'])

  # Global constants for distribution fitting
  Z_SCORES = norm.ppf(QUANTILES)

  # Prepare ILI data for synthetic history
  # Using 'ilinet_state' from global scope
  ili_data = ilinet_state[ilinet_state['region_type'] == 'States'].copy()
  ili_data['unweighted_ili'] = pd.to_numeric(
      ili_data['unweighted_ili'], errors='coerce'
  ).fillna(0)
  # ILI week_start is Sunday. Target is Saturday (6 days later)
  ili_data['target_end_date'] = pd.to_datetime(
      ili_data['week_start']
  ) + pd.Timedelta(days=6)

  # -------------------------------------------------------------------------
  # 2. Learn Transformation (Proxy Signal Calibration)
  # -------------------------------------------------------------------------
  # Intersection of Train (Admissions) and ILI to calculate scaling ratio
  merged = pd.merge(
      train[['target_end_date', 'location_name', TARGET_STR]],
      ili_data[['target_end_date', 'region', 'unweighted_ili']],
      left_on=['target_end_date', 'location_name'],
      right_on=['target_end_date', 'region'],
      how='inner',
  )

  # Calculate Ratio: Admissions / ILI
  valid_mask = (merged['unweighted_ili'] > 0) & (merged[TARGET_STR] > 0)
  merged = merged[valid_mask]
  merged['ratio'] = merged[TARGET_STR] / merged['unweighted_ili']

  # Compute robust median ratio per State
  state_ratios = merged.groupby('location_name')['ratio'].median()
  global_ratio = merged['ratio'].median()
  if pd.isna(global_ratio):
    global_ratio = 1.0

  # Apply scaling to create Synthetic History
  ili_data['scaling_factor'] = (
      ili_data['region'].map(state_ratios).fillna(global_ratio)
  )
  ili_data['synthetic_admissions'] = (
      ili_data['unweighted_ili'] * ili_data['scaling_factor']
  )

  # Transform to log space: log(x + 1)
  ili_data['log_synthetic'] = np.log1p(ili_data['synthetic_admissions'])

  # Create History Matrix: Index=Season, Columns=Week, Values=LogSynthetic
  history_matrix = ili_data.pivot_table(
      index='season', columns='week', values='log_synthetic', aggfunc='first'
  )

  # -------------------------------------------------------------------------
  # 3. Build Enhanced Growth Rate Library
  # -------------------------------------------------------------------------
  # library[start_week][horizon] = [(growth_val, start_level_val), ...]
  library = {}
  all_weeks = range(1, 54)

  # We allow a window of +/- 1 week for sampling to handle seasonal jitter
  sampling_window = [-1, 0, 1]

  for start_week in all_weeks:
    library[start_week] = {}
    for h in horizons:  # [0, 1, 2, 3]
      steps = h + 1
      end_week = start_week + steps

      samples = []  # List of (growth, level)

      for offset in sampling_window:
        w_s = start_week + offset
        w_e = end_week + offset

        # Handle year wrap-around (approximate 53 weeks)
        if w_s > 53:
          w_s -= 53
        if w_s < 1:
          w_s += 53
        if w_e > 53:
          w_e -= 53

        if (w_s in history_matrix.columns) and (w_e in history_matrix.columns):
          s_vals = history_matrix[w_s]
          e_vals = history_matrix[w_e]

          valid_idx = s_vals.notna() & e_vals.notna()
          if valid_idx.any():
            diffs = e_vals[valid_idx] - s_vals[valid_idx]
            levels = s_vals[valid_idx]
            samples.extend(zip(diffs.values, levels.values))

      library[start_week][h] = samples

  # -------------------------------------------------------------------------
  # 4. Generate Forecasts
  # -------------------------------------------------------------------------
  results = []

  unique_groups = test_x[
      ['location', 'location_name', 'reference_date']
  ].drop_duplicates()

  for _, group_row in unique_groups.iterrows():
    loc_id = group_row['location']
    ref_date = group_row['reference_date']

    # Extract recent observations for this location
    loc_train = train[train['location'] == loc_id].sort_values(
        'target_end_date'
    )

    # --- A. Estimate Current State and Trend ---
    current_trend = 0.0
    current_log = 0.0

    if loc_train.empty:
      last_date = ref_date - pd.Timedelta(days=7)
    else:
      # Reindex to ensure we have consecutive weeks for trend calc
      last_obs_date = loc_train['target_end_date'].max()
      start_window = last_obs_date - pd.Timedelta(
          weeks=3
      )  # Get last 4 weeks total

      # Create a complete date range for the window to handle gaps
      window_dates = pd.date_range(
          start=start_window, end=last_obs_date, freq='W-SAT'
      )
      recent_df = loc_train[
          loc_train['target_end_date'].isin(window_dates)
      ].set_index('target_end_date')
      recent_df = recent_df.reindex(window_dates)

      # Fill gaps with interpolation (log space) or ffill
      recent_vals = recent_df[TARGET_STR].astype(float)
      recent_log = np.log1p(recent_vals)
      recent_log = (
          recent_log.interpolate(method='linear')
          .fillna(method='ffill')
          .fillna(0)
      )

      current_log = recent_log.iloc[-1]
      last_date = last_obs_date

      # Calculate weighted trend (momentum)
      # Give higher weight to the most recent change
      if len(recent_log) >= 2:
        diffs = np.diff(recent_log.values)
        # Weighted average of diffs: [0.33, 0.66] for 2 diffs, etc.
        if len(diffs) > 0:
          weights = np.linspace(1, 2, len(diffs))
          weights /= weights.sum()
          current_trend = np.sum(diffs * weights)

        # Clip trend to prevent explosive growth/decay extrapolation
        # +/- 0.6 log units per week is roughly doubling/halving in ~1.2 weeks
        max_trend = 0.6
        current_trend = np.clip(current_trend, -max_trend, max_trend)

    curr_week_idx = last_date.isocalendar().week

    # Get horizons for this group
    group_horizons = test_x[
        (test_x['location'] == loc_id) & (test_x['reference_date'] == ref_date)
    ]

    for _, row in group_horizons.iterrows():
      h = row['horizon']
      steps_ahead = h + 1

      # --- B. Level-Weighted Analog Ensemble ---
      lookup_week = curr_week_idx if curr_week_idx in library else 52
      raw_samples = library.get(lookup_week, {}).get(h, [])

      if not raw_samples:
        mu_growth, sigma_growth = 0.0, 0.5
      else:
        growth_rates, levels = zip(*raw_samples)
        growth_rates = np.array(growth_rates)
        levels = np.array(levels)

        # Weighting: Gaussian Kernel based on Log-Level similarity
        # Tighter bandwidth (0.5) to enforce finding strictly similar past seasons
        dist = np.abs(levels - current_log)
        weights = np.exp(-dist / 0.5)

        # Normalize
        w_sum = weights.sum()
        if w_sum < 1e-9:
          # If no similar seasons, fall back to uniform weights
          weights = np.ones_like(weights) / len(weights)
        else:
          weights /= w_sum

        # Weighted Moments
        weighted_mean = np.sum(growth_rates * weights)
        weighted_var = np.sum(weights * (growth_rates - weighted_mean) ** 2)
        weighted_std = np.sqrt(weighted_var)

        mu_growth = weighted_mean
        # Add a base uncertainty that scales with horizon to prevent overconfidence
        # Base floor 0.15, plus 0.05 per week ahead
        min_std = 0.15 + (0.05 * steps_ahead)
        sigma_growth = np.sqrt(weighted_std**2 + min_std**2)

      # --- C. Blend with Momentum ---
      trend_proj = current_trend * steps_ahead

      # Blend weight 'alpha' for the current trend
      # Decay faster than previous implementation to avoid over-extrapolating transient noise
      # 0.5^steps: (1wk: 0.5, 2wk: 0.25, 3wk: 0.125...)
      alpha = 0.5**steps_ahead

      final_mu_log_growth = (alpha * trend_proj) + ((1 - alpha) * mu_growth)
      mu_log = current_log + final_mu_log_growth

      # --- D. Distribution Fitting (Log-Normal) ---
      # sigma_growth already includes the horizon-based expansion
      q_log = mu_log + (sigma_growth * Z_SCORES)
      q_values = np.expm1(q_log)
      q_values = np.maximum(0, q_values)

      # Enforce monotonicity
      q_values = np.sort(q_values)

      res_row = {
          'reference_date': ref_date,
          'location': loc_id,
          'horizon': h,
      }
      for q_name, q_val in zip([f'quantile_{q}' for q in QUANTILES], q_values):
        res_row[q_name] = q_val

      results.append(res_row)

  # -------------------------------------------------------------------------
  # 5. Assemble Output
  # -------------------------------------------------------------------------
  if not results:
    return pd.DataFrame(
        columns=[f'quantile_{q}' for q in QUANTILES], index=test_x.index
    )

  pred_df = pd.DataFrame(results)

  # Merge back to ensure alignment with input test_x
  final_df = pd.merge(
      test_x[['reference_date', 'location', 'horizon']],
      pred_df,
      on=['reference_date', 'location', 'horizon'],
      how='left',
  )

  final_df = final_df.fillna(0)
  final_df.index = test_x.index

  return final_df[[f'quantile_{q}' for q in QUANTILES]]


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

  dataset[TARGET_STR] = (
      pd.to_numeric(dataset[TARGET_STR], errors='coerce')
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
      dataset.groupby('target_end_date')[TARGET_STR].sum().reset_index()
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
