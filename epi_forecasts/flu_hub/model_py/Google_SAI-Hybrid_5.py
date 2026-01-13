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
MODEL_NAME = 'Google_SAI-Hybrid_5'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression


# Helper function for cyclic weeks, moved here for clarity and reusability
def _get_cyclic_weeks_around(epiweek, window_size):
  candidate_epiweeks = []
  for i in range(-window_size, window_size + 1):
    current_week_cand = epiweek + i
    if current_week_cand < 1:
      candidate_epiweeks.append(current_week_cand + 52)
    elif current_week_cand > 52:
      candidate_epiweeks.append(current_week_cand - 52)
    else:
      candidate_epiweeks.append(current_week_cand)
  return sorted(list(set(candidate_epiweeks)))


# Helper function to sample from historical data with recency weighting
def _sample_from_historical(
    historical_df,
    target_year,
    n_samples,
    target_col,
    epsilon = 1e-6,
    recency_decay_rate = 0.2,
):
  if historical_df.empty or historical_df[target_col].sum() == 0:
    return np.zeros(n_samples)

  historical_df_copy = historical_df.copy()

  if 'year' not in historical_df_copy.columns:
    historical_df_copy['year'] = historical_df_copy['target_end_date'].dt.year

  # Corrected recency weighting: Favor more recent past years
  year_diffs = target_year - historical_df_copy['year']
  year_diffs = np.maximum(
      0, year_diffs
  )  # Ensure only past years contribute to decay
  historical_df_copy['recency_weight'] = np.exp(
      -year_diffs * recency_decay_rate
  )

  if historical_df_copy['recency_weight'].sum() == 0:
    sampling_weights = np.ones(len(historical_df_copy)) / len(
        historical_df_copy
    )
  else:
    sampling_weights = (
        historical_df_copy['recency_weight']
        / historical_df_copy['recency_weight'].sum()
    )

  if (
      historical_df_copy[target_col].empty
      or len(historical_df_copy[target_col].unique()) == 0
  ):
    return np.zeros(n_samples)

  samples = np.random.choice(
      historical_df_copy[target_col].values,
      size=n_samples,
      p=sampling_weights.values,
  )
  return samples


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a hybrid model combining climatology,

  dynamic adjustment (log-growth rate), and an autoregressive component based on
  recent observations.
  """
  # Suppress warnings that might clutter output, but do not suppress errors.
  warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
  warnings.filterwarnings(
      'ignore', category=RuntimeWarning
  )  # Suppress warnings from numpy, e.g., for empty slice median/mean

  # --- Configuration Constants for the Hybrid Model ---
  N_ENSEMBLE = 2000
  CYCLIC_WINDOW_SIZE = 7
  EPSILON = 1e-6  # Used for division to prevent inf/nan
  RECENCY_DECAY_RATE = 0.2
  LOCAL_BLEND_WEIGHT = 0.6
  MIN_LOCAL_SAMPLES_FOR_FULL_WEIGHT = 10

  MIN_NOISE_FLOOR_RAW = 2.0
  MIN_LOG_NOISE_FLOOR = 0.05
  MIN_OVERLAP_FOR_LOCAL_REGRESSION = 30

  RECENT_WEEKS_FOR_DYNAMIC_CALCS = 7
  MAX_ILINET_DATE = pd.to_datetime('2022-10-15')

  # --- NEW HYBRID-SPECIFIC HYPERPARAMETERS (Refined for log-space noise) ---
  DYNAMIC_GROWTH_DECAY = 0.85
  DYNAMIC_BLEND_DECAY = 0.7
  LAST_OBS_PERTURB_FACTOR = 0.3
  GROWTH_RATE_PERTURB_FACTOR = 0.6
  BASE_STEP_LOG_NOISE_SCALING_FACTOR = 0.1
  DYNAMIC_PROJECTION_MIN_SEED = 1.0

  # --- 0. Initial Data Preparation ---
  train_x_copy = train_x.copy()
  train_x_copy['target_end_date'] = pd.to_datetime(
      train_x_copy['target_end_date']
  )
  test_x_copy = test_x.copy()
  test_x_copy['target_end_date'] = pd.to_datetime(
      test_x_copy['target_end_date']
  )
  test_x_copy['reference_date'] = pd.to_datetime(test_x_copy['reference_date'])

  train_actual_admissions = train_x_copy.copy()
  train_actual_admissions[TARGET_STR] = train_y.copy()
  train_actual_admissions['epiweek'] = (
      train_actual_admissions['target_end_date']
      .dt.isocalendar()
      .week.astype(int)
  )
  train_actual_admissions['year'] = train_actual_admissions[
      'target_end_date'
  ].dt.year.astype(int)
  train_actual_admissions['source'] = 'actual'

  all_locations = test_x_copy['location'].unique()
  locations_df_filtered = locations[
      locations['location'].isin(all_locations)
  ].copy()

  # --- 1. ILINet Data Preprocessing for Augmentation (Strategy 2 Principle) ---
  ilinet_state_processed = ilinet_state.copy()
  ilinet_state_processed['target_end_date'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  ) + pd.Timedelta(days=6)

  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] < MAX_ILINET_DATE
  ].copy()

  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_processed = ilinet_state_processed[
      ['target_end_date', 'location_name', 'unweighted_ili']
  ].dropna(subset=['unweighted_ili'])

  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations_df_filtered[['location_name', 'location']],
      on='location_name',
      how='left',
  ).dropna(subset=['location'])
  ilinet_state_processed['location'] = ilinet_state_processed[
      'location'
  ].astype(int)

  # --- 2. Learn ILI-to-Admissions Transformation Parameters (Robust Log-Linear HuberRegressor) ---
  from collections import defaultdict

  overlap_data = pd.merge(
      train_actual_admissions,
      ilinet_state_processed,
      on=['target_end_date', 'location'],
      how='inner',
      suffixes=('_actual', '_ili'),
  )

  # Removed + EPSILON from log1p for counts/ILI
  overlap_data['log_ili'] = np.log1p(overlap_data['unweighted_ili'])
  overlap_data['log_admissions'] = np.log1p(overlap_data[TARGET_STR])

  default_national_mean_log_admissions = (
      overlap_data['log_admissions'].mean() if not overlap_data.empty else 0.0
  )
  national_ili_model_params = {
      'slope': 0.0,
      'intercept': default_national_mean_log_admissions,
      'residual_std_log_for_synth': MIN_LOG_NOISE_FLOOR,
  }

  # Train national model on log-transformed data
  if len(overlap_data) >= MIN_OVERLAP_FOR_LOCAL_REGRESSION:
    X_nat = overlap_data[['log_ili']]
    y_nat = overlap_data['log_admissions']

    ili_has_variance = X_nat['log_ili'].nunique() > 1
    y_has_variance = y_nat.nunique() > 1

    if y_has_variance:
      if ili_has_variance:
        huber_nat = HuberRegressor(max_iter=1000, alpha=0.1)
        try:
          huber_nat.fit(X_nat, y_nat)
          national_ili_model_params = {
              'slope': max(0.0, huber_nat.coef_[0]),
              'intercept': huber_nat.intercept_,
          }
          residuals_nat = y_nat - huber_nat.predict(X_nat)
          national_ili_model_params['residual_std_log_for_synth'] = max(
              MIN_LOG_NOISE_FLOOR, np.std(residuals_nat)
          )
        except Exception:
          try:
            lin_reg_nat = LinearRegression()
            lin_reg_nat.fit(X_nat, y_nat)
            national_ili_model_params = {
                'slope': max(0.0, lin_reg_nat.coef_[0]),
                'intercept': lin_reg_nat.intercept_,
            }
            residuals_nat = y_nat - lin_reg_nat.predict(X_nat)
            national_ili_model_params['residual_std_log_for_synth'] = max(
                MIN_LOG_NOISE_FLOOR, np.std(residuals_nat)
            )
          except Exception:
            pass
      else:
        national_ili_model_params = {
            'slope': 0.0,
            'intercept': y_nat.mean(),
            'residual_std_log_for_synth': max(
                MIN_LOG_NOISE_FLOOR, np.std(y_nat)
            ),
        }
    else:
      national_ili_model_params = {
          'slope': 0.0,
          'intercept': y_nat.iloc[0],
          'residual_std_log_for_synth': MIN_LOG_NOISE_FLOOR,
      }  # If constant, std is 0, use floor
  elif not overlap_data.empty:
    national_ili_model_params = {
        'slope': 0.0,
        'intercept': default_national_mean_log_admissions,
        'residual_std_log_for_synth': max(
            MIN_LOG_NOISE_FLOOR, np.std(overlap_data['log_admissions'])
        ),
    }

  location_ili_models = defaultdict(lambda: national_ili_model_params)

  for loc in all_locations:
    loc_overlap_data = overlap_data[overlap_data['location'] == loc]
    loc_model_params = (
        national_ili_model_params.copy()
    )  # Start with national defaults
    if len(loc_overlap_data) >= MIN_OVERLAP_FOR_LOCAL_REGRESSION:
      X_loc = loc_overlap_data[['log_ili']]
      y_loc = loc_overlap_data['log_admissions']

      ili_has_variance = X_loc['log_ili'].nunique() > 1
      y_has_variance = y_loc.nunique() > 1

      if y_has_variance:
        if ili_has_variance:
          huber_loc = HuberRegressor(max_iter=1000, alpha=0.1)
          try:
            huber_loc.fit(X_loc, y_loc)
            loc_model_params = {
                'slope': max(0.0, huber_loc.coef_[0]),
                'intercept': huber_loc.intercept_,
            }
            residuals_loc = y_loc - huber_loc.predict(X_loc)
            loc_model_params['residual_std_log_for_synth'] = max(
                MIN_LOG_NOISE_FLOOR, np.std(residuals_loc)
            )
          except Exception:
            try:
              lin_reg_loc = LinearRegression()
              lin_reg_loc.fit(X_loc, y_loc)
              loc_model_params = {
                  'slope': max(0.0, lin_reg_loc.coef_[0]),
                  'intercept': lin_reg_loc.intercept_,
              }
              residuals_loc = y_loc - lin_reg_loc.predict(X_loc)
              loc_model_params['residual_std_log_for_synth'] = max(
                  MIN_LOG_NOISE_FLOOR, np.std(residuals_loc)
              )
            except Exception:
              pass
        else:
          loc_model_params = {
              'slope': 0.0,
              'intercept': y_loc.mean(),
              'residual_std_log_for_synth': max(
                  MIN_LOG_NOISE_FLOOR, np.std(y_loc)
              ),
          }
      else:
        loc_model_params = {
            'slope': 0.0,
            'intercept': y_loc.iloc[0],
            'residual_std_log_for_synth': MIN_LOG_NOISE_FLOOR,
        }  # If constant, std is 0, use floor
    elif (
        not loc_overlap_data.empty
    ):  # Less than threshold, but some data: fallback to mean log admissions with its std
      loc_model_params = {
          'slope': 0.0,
          'intercept': loc_overlap_data['log_admissions'].mean(),
          'residual_std_log_for_synth': max(
              MIN_LOG_NOISE_FLOOR, np.std(loc_overlap_data['log_admissions'])
          ),
      }

    location_ili_models[loc] = loc_model_params

  # --- 3. Synthesize Historical Admissions and Create Augmented Dataset ---
  synthetic_admissions_df = ilinet_state_processed.copy()
  synthetic_admissions_df['log_ili'] = np.log1p(
      synthetic_admissions_df['unweighted_ili']
  )

  # Added stochasticity to synthetic admissions generation
  synthetic_admissions_df['admissions_log_transformed'] = (
      synthetic_admissions_df.apply(
          lambda row: (
              location_ili_models[row['location']]['slope'] * row['log_ili']
              + location_ili_models[row['location']]['intercept']
              + np.random.normal(
                  0,
                  location_ili_models[row['location']][
                      'residual_std_log_for_synth'
                  ],
              )
          ),
          axis=1,
      )
  )
  synthetic_admissions_df['admissions'] = (
      np.expm1(synthetic_admissions_df['admissions_log_transformed'])
      .clip(lower=0)
      .round()
      .astype(int)
  )
  synthetic_admissions_df['source'] = 'synthetic'

  synthetic_admissions_for_concat = synthetic_admissions_df[
      ['target_end_date', 'location', 'location_name', 'admissions', 'source']
  ]

  actual_admissions_for_concat = train_actual_admissions[
      ['target_end_date', 'location', 'location_name', TARGET_STR, 'source']
  ].rename(columns={TARGET_STR: 'admissions'})

  augmented_historical_data = pd.concat(
      [actual_admissions_for_concat, synthetic_admissions_for_concat],
      ignore_index=True,
  )

  augmented_historical_data['target_end_date'] = pd.to_datetime(
      augmented_historical_data['target_end_date']
  )

  augmented_historical_data = pd.merge(
      augmented_historical_data.drop(columns='population', errors='ignore'),
      locations_df_filtered[
          ['location', 'population', 'location_name']
      ].drop_duplicates(subset=['location']),
      on=['location', 'location_name'],
      how='left',
  )

  augmented_historical_data = augmented_historical_data.sort_values(
      by=['target_end_date', 'location', 'source'], ascending=[True, True, True]
  ).drop_duplicates(subset=['target_end_date', 'location'], keep='first')

  augmented_historical_data['epiweek'] = (
      augmented_historical_data['target_end_date']
      .dt.isocalendar()
      .week.astype(int)
  )
  augmented_historical_data['year'] = augmented_historical_data[
      'target_end_date'
  ].dt.year.astype(int)

  # --- 4. Pre-compute Dynamic Adjustment & Noise Level for Each Location ---
  reference_date = test_x_copy['reference_date'].iloc[0]
  last_obs_date = reference_date - pd.Timedelta(weeks=1)

  latest_obs_df = train_actual_admissions[
      (train_actual_admissions['target_end_date'] == last_obs_date)
      & (train_actual_admissions['location'].isin(all_locations))
  ].set_index('location')[TARGET_STR]

  recent_log_growth_rate_per_location = pd.Series(0.0, index=all_locations)
  std_log_growth_rate_per_location = pd.Series(
      MIN_LOG_NOISE_FLOOR / 5.0, index=all_locations
  )
  train_actual_admissions_filtered = train_actual_admissions[
      train_actual_admissions['location'].isin(all_locations)
  ].copy()

  for loc in all_locations:
    loc_recent_data = []
    last_valid_obs = DYNAMIC_PROJECTION_MIN_SEED

    for week_offset in range(RECENT_WEEKS_FOR_DYNAMIC_CALCS):
      current_obs_date_iter = last_obs_date - pd.Timedelta(weeks=week_offset)
      actual_obs_at_date = train_actual_admissions_filtered[
          (train_actual_admissions_filtered['location'] == loc)
          & (
              train_actual_admissions_filtered['target_end_date']
              == current_obs_date_iter
          )
      ][TARGET_STR]

      if not actual_obs_at_date.empty:
        current_obs_val = actual_obs_at_date.iloc[0]
        loc_recent_data.append(current_obs_val)
        last_valid_obs = current_obs_val
      else:
        loc_recent_data.append(last_valid_obs)

    loc_recent_data.reverse()

    if len(loc_recent_data) >= 2:
      # Removed + EPSILON from log1p for counts
      log1p_vals = np.log1p(np.array(loc_recent_data))
      log_diffs = np.diff(log1p_vals)

      if len(log_diffs) > 0:
        weights = np.array([
            DYNAMIC_GROWTH_DECAY**j for j in range(len(log_diffs) - 1, -1, -1)
        ])
        weights = weights / weights.sum()

        recent_log_growth_rate_per_location[loc] = np.average(
            log_diffs, weights=weights
        )
        std_log_growth_rate_per_location[loc] = np.std(log_diffs)
      else:
        recent_log_growth_rate_per_location[loc] = 0.0
        std_log_growth_rate_per_location[loc] = 0.0

    std_log_growth_rate_per_location[loc] = max(
        MIN_LOG_NOISE_FLOOR, std_log_growth_rate_per_location[loc]
    )

  national_augmented_historical_data = augmented_historical_data.groupby(
      ['target_end_date', 'epiweek', 'year'], as_index=False
  )['admissions'].sum()
  national_augmented_historical_data['population'] = locations_df_filtered[
      'population'
  ].sum()
  national_augmented_historical_data['rate_per_100k'] = (
      national_augmented_historical_data['admissions']
      / (national_augmented_historical_data['population'] + EPSILON)
      * 100_000
  )

  overall_noise_raw_std_per_location = pd.Series(
      MIN_NOISE_FLOOR_RAW * 5, index=all_locations
  )
  overall_noise_log_std_per_location = pd.Series(
      MIN_LOG_NOISE_FLOOR * 5, index=all_locations
  )

  for loc in all_locations:
    forecast_epiweek_for_std = reference_date.isocalendar().week
    candidate_epiweeks_for_std = _get_cyclic_weeks_around(
        forecast_epiweek_for_std, CYCLIC_WINDOW_SIZE
    )

    historical_loc_epiweek_admissions = augmented_historical_data[
        (augmented_historical_data['location'] == loc)
        & (
            augmented_historical_data['epiweek'].isin(
                candidate_epiweeks_for_std
            )
        )
    ]['admissions'].values

    if historical_loc_epiweek_admissions.size > 1:
      loc_raw_std = np.std(historical_loc_epiweek_admissions)
      overall_noise_raw_std_per_location[loc] = max(
          MIN_NOISE_FLOOR_RAW, loc_raw_std
      )

      # Removed + EPSILON from log1p for counts
      log1p_admissions = np.log1p(historical_loc_epiweek_admissions)
      loc_log_std = np.std(log1p_admissions)
      overall_noise_log_std_per_location[loc] = max(
          MIN_LOG_NOISE_FLOOR, loc_log_std
      )
    else:
      national_climat_admissions_for_epiweek = (
          national_augmented_historical_data[(
              national_augmented_historical_data['epiweek'].isin(
                  candidate_epiweeks_for_std
              )
          )]['admissions'].values
      )

      national_climat_raw_std_for_epiweek = MIN_NOISE_FLOOR_RAW * 2
      national_climat_log_std_for_epiweek = MIN_LOG_NOISE_FLOOR * 2

      if national_climat_admissions_for_epiweek.size > 1:
        national_climat_raw_std_for_epiweek = np.std(
            national_climat_admissions_for_epiweek
        )
        # Removed + EPSILON from log1p for counts
        log1p_national_admissions = np.log1p(
            national_climat_admissions_for_epiweek
        )
        national_climat_log_std_for_epiweek = np.std(log1p_national_admissions)

      loc_pop = locations_df_filtered[locations_df_filtered['location'] == loc][
          'population'
      ].iloc[0]
      national_pop = locations_df_filtered['population'].sum()
      if national_pop > EPSILON:
        pop_scale_factor = np.sqrt(loc_pop / national_pop)
        overall_noise_raw_std_per_location[loc] = max(
            MIN_NOISE_FLOOR_RAW,
            national_climat_raw_std_for_epiweek * pop_scale_factor,
        )
        overall_noise_log_std_per_location[loc] = max(
            MIN_LOG_NOISE_FLOOR,
            national_climat_log_std_for_epiweek * pop_scale_factor,
        )
      else:
        overall_noise_raw_std_per_location[loc] = max(
            MIN_NOISE_FLOOR_RAW, national_climat_raw_std_for_epiweek
        )
        overall_noise_log_std_per_location[loc] = max(
            MIN_LOG_NOISE_FLOOR, national_climat_log_std_for_epiweek
        )

  # --- 5. Generate Forecasts for test_x (Hybrid Model) ---
  test_y_hat_quantiles = pd.DataFrame(
      index=test_x_copy.index, columns=[f'quantile_{q}' for q in QUANTILES]
  )

  for idx, target_row in test_x_copy.iterrows():
    forecast_location = target_row['location']
    forecast_population = target_row['population']
    forecast_target_date = target_row['target_end_date']
    forecast_epiweek = forecast_target_date.isocalendar().week
    forecast_horizon = target_row['horizon']

    steps_ahead = forecast_horizon + 1

    candidate_epiweeks = _get_cyclic_weeks_around(
        forecast_epiweek, CYCLIC_WINDOW_SIZE
    )

    # Retrieve pre-computed dynamic parameters
    last_obs_for_loc = latest_obs_df.get(forecast_location, 0.0)
    log_growth_rate_for_loc = recent_log_growth_rate_per_location.get(
        forecast_location, 0.0
    )
    std_log_growth_rate_for_loc = std_log_growth_rate_per_location.get(
        forecast_location, MIN_LOG_NOISE_FLOOR
    )
    overall_noise_raw_std_for_loc = overall_noise_raw_std_per_location.get(
        forecast_location, MIN_NOISE_FLOOR_RAW * 5
    )
    overall_noise_log_std_for_loc = overall_noise_log_std_per_location.get(
        forecast_location, MIN_LOG_NOISE_FLOOR * 5
    )

    # Prepare Climatological Samples (vectorized for N_ENSEMBLE)
    historical_loc_data = augmented_historical_data[
        (augmented_historical_data['location'] == forecast_location)
        & (augmented_historical_data['epiweek'].isin(candidate_epiweeks))
    ]
    local_sample_arr = _sample_from_historical(
        historical_loc_data,
        forecast_target_date.year,
        N_ENSEMBLE,
        'admissions',
        EPSILON,
        RECENCY_DECAY_RATE,
    )

    historical_national_data = national_augmented_historical_data[
        national_augmented_historical_data['epiweek'].isin(candidate_epiweeks)
    ]
    geo_rate_sample_arr = _sample_from_historical(
        historical_national_data,
        forecast_target_date.year,
        N_ENSEMBLE,
        'rate_per_100k',
        EPSILON,
        RECENCY_DECAY_RATE,
    )
    geo_val_sample_arr = geo_rate_sample_arr / 100_000 * forecast_population

    # Adaptive Blending Weight Calculation (perturbed for each ensemble member)
    base_local_blend_weights_arr = np.clip(
        LOCAL_BLEND_WEIGHT + np.random.normal(0, 0.05, N_ENSEMBLE), 0.0, 1.0
    )
    if (
        historical_loc_data['admissions'].size
        < MIN_LOCAL_SAMPLES_FOR_FULL_WEIGHT
    ):
      sparsity_factor = (
          historical_loc_data['admissions'].size
          / MIN_LOCAL_SAMPLES_FOR_FULL_WEIGHT
      ) ** 0.5
      actual_local_blend_weights_arr = (
          base_local_blend_weights_arr * sparsity_factor
      )
    else:
      actual_local_blend_weights_arr = base_local_blend_weights_arr

    # Blended Base Climatological Prediction (vectorized)
    blended_base_pred_arr = np.zeros(N_ENSEMBLE)
    if historical_loc_data.empty and historical_national_data.empty:
      blended_base_pred_arr = np.zeros(N_ENSEMBLE)
    elif historical_loc_data.empty:
      blended_base_pred_arr = geo_val_sample_arr
    elif historical_national_data.empty:
      blended_base_pred_arr = local_sample_arr
    else:
      blended_base_pred_arr = (
          actual_local_blend_weights_arr * local_sample_arr
          + (1 - actual_local_blend_weights_arr) * geo_val_sample_arr
      )
    blended_base_pred_arr = np.maximum(0.0, blended_base_pred_arr)

    # --- Hybrid Dynamic Component: Step-by-Step Projection with Log-Multiplicative Noise ---
    initial_dynamic_state_arr = np.maximum(
        DYNAMIC_PROJECTION_MIN_SEED,
        last_obs_for_loc
        + np.random.normal(
            0,
            overall_noise_raw_std_for_loc * LAST_OBS_PERTURB_FACTOR,
            N_ENSEMBLE,
        ),
    )
    initial_dynamic_state_arr = np.maximum(0, initial_dynamic_state_arr)

    # Removed + EPSILON from log1p for counts
    ensemble_dynamic_predictions_log1p = np.log1p(initial_dynamic_state_arr)

    perturbed_initial_log_growth_rate_arr = (
        log_growth_rate_for_loc
        + np.random.normal(
            0,
            std_log_growth_rate_for_loc * GROWTH_RATE_PERTURB_FACTOR,
            N_ENSEMBLE,
        )
    )

    per_step_log_noise_std = (
        overall_noise_log_std_for_loc * BASE_STEP_LOG_NOISE_SCALING_FACTOR
    )
    per_step_log_noise_std = max(
        MIN_LOG_NOISE_FLOOR / 2.0, per_step_log_noise_std
    )

    for s in range(1, steps_ahead + 1):
      current_step_growth_rate = perturbed_initial_log_growth_rate_arr * (
          DYNAMIC_GROWTH_DECAY ** (s - 1)
      )

      ensemble_dynamic_predictions_log1p = (
          ensemble_dynamic_predictions_log1p
          + current_step_growth_rate
          + np.random.normal(0, per_step_log_noise_std, N_ENSEMBLE)
      )

      # Ensure log1p values do not become too small (e.g., negative log of a positive number)
      ensemble_dynamic_predictions_log1p = np.maximum(
          np.log1p(EPSILON), ensemble_dynamic_predictions_log1p
      )

    ensemble_dynamic_predictions = np.expm1(ensemble_dynamic_predictions_log1p)
    ensemble_dynamic_predictions = np.maximum(0, ensemble_dynamic_predictions)

    # --- Blending Dynamic Projection with Climatological Samples ---
    dynamic_blend_weight_arr = DYNAMIC_BLEND_DECAY**steps_ahead
    dynamic_blend_weight_arr = np.clip(
        dynamic_blend_weight_arr + np.random.normal(0, 0.05, N_ENSEMBLE),
        0.0,
        1.0,
    )

    final_ensemble_member_pred_arr = (
        dynamic_blend_weight_arr * ensemble_dynamic_predictions
    ) + ((1 - dynamic_blend_weight_arr) * blended_base_pred_arr)

    final_ensemble_member_pred_arr = np.maximum(
        0, np.round(final_ensemble_member_pred_arr)
    ).astype(int)

    final_quantiles = np.percentile(
        final_ensemble_member_pred_arr, [q * 100 for q in QUANTILES]
    )

    final_quantiles = np.maximum(0, final_quantiles)
    final_quantiles = np.maximum.accumulate(final_quantiles)

    test_y_hat_quantiles.loc[idx] = final_quantiles

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
