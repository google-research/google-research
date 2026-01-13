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
MODEL_NAME = 'Google_SAI-Hybrid_2'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist  # Optimized for distance calculations

# Constants for robust calculations (defined within this rewrite cell's scope)
# These constants are critical for the hybrid model's robustness and parameterization.
MIN_TOTAL_ADMISSIONS_FLOOR = 1  # Minimum count for total admissions before per-capita conversion for log transforms.
PSEUDO_RATE_FOR_ILI = 1e-7  # Pseudo-rate floor for ILI data per capita to avoid log(0) and division by zero.
EPSILON_STABILITY = 1e-6  # Small constant for numerical stability in log transforms and divisions.
CYCLIC_WINDOW_SIZE = (
    2  # Weeks on either side for climatological lookups (Code 1 influence).
)
MIN_OVERLAP_FOR_LOCATION_MODEL = (
    20  # Minimum overlap points to train a location-specific Huber model.
)
LOOKBACK_WEEKS = 4  # Number of weeks to look back for current level and trend for state vector (Code 2 principle).
NUM_ANALOGS = 25  # Number of nearest neighbors to use for ensemble (increased for robustness).
MAX_SCALING_FACTOR = (
    5.0  # Max factor to scale historical trajectories (Code 2 constraint).
)
MIN_ANALOGS_FOR_DYNAMIC_BLEND = 10  # Minimum analogs required to start blending dynamic forecast with climatology.
MIN_TREND_STDEV_FOR_ANALOG = 0.01  # Minimum standard deviation of log growth rates for a historical period to be considered a dynamic analog. Helps filter 'flat' trends.

# TARGET_STR and QUANTILES are defined in the global preamble and are accessible.


# Helper functions for epiweek, season and cyclic week window
def get_epiweek(date):
  """Returns the ISO calendar week number (1-52 or 53)."""
  # Ensure date is a pandas Timestamp for .isocalendar()
  if not isinstance(date, pd.Timestamp):
    date = pd.to_datetime(date)
  return date.isocalendar()[1]


def get_season(date):
  """Defines the flu season (e.g., '2020/21') based on epiweek."""
  if not isinstance(date, pd.Timestamp):
    date = pd.to_datetime(date)
  year = date.year
  week = date.isocalendar()[1]
  # Flu season typically starts in week 40 of a year and ends in week 39 of the next year.
  if week >= 40:
    return f'{year}/{year+1}'
  else:
    return f'{year-1}/{year}'


def get_cyclic_weeks(epiweek):
  """Returns a list of epiweeks within a cyclic window around the given epiweek."""
  all_weeks = []
  # Use the constant CYCLIC_WINDOW_SIZE
  for offset in range(-CYCLIC_WINDOW_SIZE, CYCLIC_WINDOW_SIZE + 1):
    current_week = epiweek + offset
    if current_week > 52:
      current_week -= 52
    elif current_week < 1:
      current_week += 52
    all_weeks.append(current_week)
  return sorted(list(set(all_weeks)))


def get_location_pseudo_pc_rate(
    location_fips, populations_map
):
  """Helper function for location-specific pseudo per-capita rate using MIN_TOTAL_ADMISSIONS_FLOOR."""
  pop = populations_map.get(
      location_fips, 1
  )  # Default to 1 to avoid division by zero
  # Use the constant MIN_TOTAL_ADMISSIONS_FLOOR
  return MIN_TOTAL_ADMISSIONS_FLOOR / pop


# New helper function for national pseudo rate
def get_national_pseudo_pc_rate(national_population):
  """Helper function for national pseudo per-capita rate using MIN_TOTAL_ADMISSIONS_FLOOR."""
  return MIN_TOTAL_ADMISSIONS_FLOOR / national_population


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make predictions for test_x using a hybrid model combining augmented climatology

  with a current-level and trend-based adjustment, blended according to dynamic
  confidence
  and the current level of flu activity. Includes location-specific
  transformation
  models for ILI data and a global analog fallback for locations with sparse
  data.
  This version enhances the dynamic forecasting component by incorporating
  national-level
  ILI data as additional features in the state vector for analog matching.
  """
  # Suppress specific sklearn warnings about convergence if max_iter is hit
  warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
  warnings.filterwarnings(
      'ignore',
      category=RuntimeWarning,
      message='invalid value encountered in log',
  )
  warnings.filterwarnings(
      'ignore',
      category=RuntimeWarning,
      message='invalid value encountered in divide',
  )
  warnings.filterwarnings(
      'ignore', category=RuntimeWarning, message='Mean of empty slice'
  )

  # Ensure date columns are datetime objects at the start for consistent handling
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])
  test_x['reference_date'] = pd.to_datetime(test_x['reference_date'])

  # Pre-cache location populations for efficiency (assuming 'locations' is globally available)
  loc_populations_map = locations.set_index('location')['population'].to_dict()
  national_population = locations[
      'population'
  ].sum()  # Sum of all state populations
  national_pseudo_pc_rate = get_national_pseudo_pc_rate(national_population)

  # Determine the latest date available in the actual training data for this fold
  max_train_date = train_x['target_end_date'].max()
  # Determine the end date of the training window for current_pc_admissions_history and query vector
  # This is consistent with rolling evaluation: reference_date - 1 week.
  current_fold_train_end_date = test_x['reference_date'].iloc[0] - pd.Timedelta(
      weeks=1
  )

  # --- 1. Data Preprocessing and Augmentation ---

  # 1.1. Prepare primary training data (actual admissions)
  historical_admissions_actual = train_x.copy()
  historical_admissions_actual[TARGET_STR] = train_y
  historical_admissions_actual['epiweek'] = historical_admissions_actual[
      'target_end_date'
  ].apply(get_epiweek)
  historical_admissions_actual['season'] = historical_admissions_actual[
      'target_end_date'
  ].apply(get_season)
  historical_admissions_actual['population'] = historical_admissions_actual[
      'location'
  ].map(loc_populations_map)
  historical_admissions_actual['per_capita_admissions'] = (
      historical_admissions_actual[TARGET_STR]
      / historical_admissions_actual['population']
  )
  # Apply pseudo-rate floor for per-capita admissions to ensure non-zero for log transforms.
  historical_admissions_actual['per_capita_admissions'] = (
      historical_admissions_actual.apply(
          lambda r: np.maximum(
              get_location_pseudo_pc_rate(r['location'], loc_populations_map),
              r['per_capita_admissions'],
          ),
          axis=1,
      )
  )

  # 1.2. Prepare ILINet data for augmentation (State-level and National)
  # State-level ILINet
  ilinet_state_processed = ilinet_state.copy()
  ilinet_state_processed['week_start'] = pd.to_datetime(
      ilinet_state_processed['week_start']
  )
  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'region': 'location_name'}
  )
  ilinet_state_processed = ilinet_state_processed[
      ['week_start', 'location_name', 'unweighted_ili']
  ]
  ilinet_state_processed = pd.merge(
      ilinet_state_processed,
      locations[['location', 'location_name']],
      on='location_name',
      how='left',
  )
  ilinet_state_processed['population'] = ilinet_state_processed['location'].map(
      loc_populations_map
  )
  ilinet_state_processed['per_capita_ili'] = np.maximum(
      PSEUDO_RATE_FOR_ILI,
      (
          ilinet_state_processed['unweighted_ili']
          / ilinet_state_processed['population']
      ).fillna(0),
  )
  ilinet_state_processed = ilinet_state_processed.rename(
      columns={'week_start': 'target_end_date'}
  )
  ilinet_state_processed['epiweek'] = ilinet_state_processed[
      'target_end_date'
  ].apply(get_epiweek)
  ilinet_state_processed['season'] = ilinet_state_processed[
      'target_end_date'
  ].apply(get_season)
  ilinet_state_processed = ilinet_state_processed[
      ilinet_state_processed['target_end_date'] <= max_train_date
  ]

  # National-level ILINet (NEW: Process `ilinet` for national features)
  ilinet_national_processed = ilinet[ilinet['region'] == 'National'].copy()
  ilinet_national_processed['week_start'] = pd.to_datetime(
      ilinet_national_processed['week_start']
  )
  ilinet_national_processed = ilinet_national_processed[
      ['week_start', 'unweighted_ili']
  ]
  ilinet_national_processed['population'] = (
      national_population  # Assign national population
  )
  ilinet_national_processed['per_capita_ili'] = np.maximum(
      PSEUDO_RATE_FOR_ILI,
      (
          ilinet_national_processed['unweighted_ili']
          / ilinet_national_processed['population']
      ).fillna(0),
  )
  ilinet_national_processed = ilinet_national_processed.rename(
      columns={'week_start': 'target_end_date'}
  )
  ilinet_national_processed['epiweek'] = ilinet_national_processed[
      'target_end_date'
  ].apply(get_epiweek)
  ilinet_national_processed['season'] = ilinet_national_processed[
      'target_end_date'
  ].apply(get_season)
  ilinet_national_processed = ilinet_national_processed[
      ilinet_national_processed['target_end_date'] <= max_train_date
  ]

  # 1.3. Learn transformations (Global and Location-Specific HuberRegressor)
  # This part remains mostly the same, as the Huber model learns the ILI->Admissions mapping.
  overlap_data = pd.merge(
      historical_admissions_actual[
          ['target_end_date', 'location', 'per_capita_admissions']
      ],
      ilinet_state_processed[['target_end_date', 'location', 'per_capita_ili']],
      on=['target_end_date', 'location'],
      how='inner',
  )
  overlap_data = overlap_data.dropna(
      subset=['per_capita_admissions', 'per_capita_ili']
  )

  scaler_ili = None
  scaler_admissions = None
  huber_model_global = None
  location_huber_models = {}

  if not overlap_data.empty and len(overlap_data) > 1:
    X_global = overlap_data[['per_capita_ili']]
    y_global = overlap_data['per_capita_admissions']

    try:
      scaler_ili = StandardScaler()
      scaler_admissions = StandardScaler()
      X_global_scaled = np.nan_to_num(scaler_ili.fit_transform(X_global))
      y_global_scaled = np.nan_to_num(
          scaler_admissions.fit_transform(y_global.values.reshape(-1, 1))
      ).flatten()

      huber_model_global = HuberRegressor(epsilon=1.35, max_iter=500, tol=1e-5)
      huber_model_global.fit(X_global_scaled, y_global_scaled)

      for loc_fips in overlap_data['location'].unique():
        loc_overlap_data = overlap_data[overlap_data['location'] == loc_fips]
        if len(loc_overlap_data) >= MIN_OVERLAP_FOR_LOCATION_MODEL:
          X_loc = loc_overlap_data[['per_capita_ili']]
          y_loc = loc_overlap_data['per_capita_admissions']
          X_loc_scaled = np.nan_to_num(scaler_ili.transform(X_loc))
          y_loc_scaled = np.nan_to_num(
              scaler_admissions.transform(y_loc.values.reshape(-1, 1))
          ).flatten()

          huber_model_loc = HuberRegressor(epsilon=1.35, max_iter=500, tol=1e-5)
          huber_model_loc.fit(X_loc_scaled, y_loc_scaled)
          location_huber_models[loc_fips] = huber_model_loc
    except Exception as e:
      huber_model_global = None
      warnings.warn(
          f'HuberRegressor training failed: {e}. Synthetic admissions will not'
          ' be generated or will use global fallback if local model failed.',
          UserWarning,
      )
  else:
    warnings.warn(
        'Not enough overlap data to train HuberRegressor models. Synthetic'
        ' admissions will not be generated.',
        UserWarning,
    )

  # 1.4. Generate Synthetic Admissions (using global or location-specific Huber models)
  # This section remains largely the same, generating state-level synthetic data.
  synthetic_admissions_list = []
  if (
      huber_model_global is not None
      and scaler_ili is not None
      and scaler_admissions is not None
  ):
    all_locations_with_ili = ilinet_state_processed['location'].unique()
    actual_data_dates_per_loc = (
        historical_admissions_actual.groupby('location')['target_end_date']
        .apply(lambda x: set(x.dt.date))
        .to_dict()
    )

    for loc_fips in all_locations_with_ili:
      loc_ilinet_data = ilinet_state_processed[
          ilinet_state_processed['location'] == loc_fips
      ].copy()
      if loc_ilinet_data.empty:
        continue

      actual_dates_for_loc = actual_data_dates_per_loc.get(loc_fips, set())
      dates_to_synthesize = loc_ilinet_data[
          ~loc_ilinet_data['target_end_date'].dt.date.isin(actual_dates_for_loc)
      ].copy()
      if dates_to_synthesize.empty:
        continue

      X_pred = dates_to_synthesize[['per_capita_ili']]
      X_pred_scaled = np.nan_to_num(scaler_ili.transform(X_pred))
      model_to_use = location_huber_models.get(loc_fips, huber_model_global)
      y_pred_scaled = model_to_use.predict(X_pred_scaled)
      predicted_per_capita_admissions = scaler_admissions.inverse_transform(
          y_pred_scaled.reshape(-1, 1)
      ).flatten()

      dates_to_synthesize['synthetic_per_capita_admissions'] = (
          predicted_per_capita_admissions
      )
      dates_to_synthesize['synthetic_per_capita_admissions'] = (
          dates_to_synthesize.apply(
              lambda r: np.maximum(
                  get_location_pseudo_pc_rate(
                      r['location'], loc_populations_map
                  ),
                  r['synthetic_per_capita_admissions'],
              ),
              axis=1,
          )
      )
      dates_to_synthesize['synthetic_total_admissions'] = (
          dates_to_synthesize['synthetic_per_capita_admissions']
          * dates_to_synthesize['population']
      )
      synthetic_admissions_list.append(dates_to_synthesize)

    if synthetic_admissions_list:
      synthetic_admissions_df = pd.concat(
          synthetic_admissions_list, ignore_index=True
      )
      synthetic_admissions_df = synthetic_admissions_df[[
          'target_end_date',
          'location',
          'population',
          'epiweek',
          'season',
          'synthetic_total_admissions',
          'synthetic_per_capita_admissions',
      ]]
      synthetic_admissions_df = synthetic_admissions_df.rename(
          columns={
              'synthetic_total_admissions': TARGET_STR,
              'synthetic_per_capita_admissions': 'per_capita_admissions',
          }
      )
    else:
      synthetic_admissions_df = pd.DataFrame(
          columns=[
              'target_end_date',
              'location',
              'population',
              'epiweek',
              'season',
              TARGET_STR,
              'per_capita_admissions',
          ]
      )
  else:
    synthetic_admissions_df = pd.DataFrame(
        columns=[
            'target_end_date',
            'location',
            'population',
            'epiweek',
            'season',
            TARGET_STR,
            'per_capita_admissions',
        ]
    )

  # 1.5. Combine Actual and Synthetic Data to create full_history_df (state-level)
  full_history_df = pd.concat([
      historical_admissions_actual[[
          'target_end_date',
          'location',
          'population',
          'epiweek',
          'season',
          TARGET_STR,
          'per_capita_admissions',
      ]],
      synthetic_admissions_df[[
          'target_end_date',
          'location',
          'population',
          'epiweek',
          'season',
          TARGET_STR,
          'per_capita_admissions',
      ]],
  ]).reset_index(drop=True)
  full_history_df = full_history_df.drop_duplicates(
      subset=['target_end_date', 'location'], keep='first'
  )
  full_history_df[TARGET_STR] = (
      np.maximum(0, full_history_df[TARGET_STR]).round().astype(int)
  )

  # --- NEW: Prepare National-level Historical Admissions (Actual + Synthetic) for features ---
  # Aggregate actual state admissions to national
  national_admissions_actual = (
      historical_admissions_actual.groupby('target_end_date')
      .agg({TARGET_STR: 'sum', 'population': 'sum'})
      .reset_index()
  )
  national_admissions_actual['per_capita_admissions'] = (
      national_admissions_actual[TARGET_STR]
      / national_admissions_actual['population']
  )
  national_admissions_actual['per_capita_admissions'] = np.maximum(
      national_pseudo_pc_rate,
      national_admissions_actual['per_capita_admissions'],
  )

  # Generate synthetic national admissions using the global Huber model on national ILI
  synthetic_national_admissions_list = []
  if (
      huber_model_global is not None
      and scaler_ili is not None
      and scaler_admissions is not None
  ):
    actual_national_dates = set(
        national_admissions_actual['target_end_date'].dt.date
    )
    national_ili_to_synthesize = ilinet_national_processed[
        ~ilinet_national_processed['target_end_date'].dt.date.isin(
            actual_national_dates
        )
    ].copy()

    if not national_ili_to_synthesize.empty:
      X_pred_national = national_ili_to_synthesize[['per_capita_ili']]
      X_pred_national_scaled = np.nan_to_num(
          scaler_ili.transform(X_pred_national)
      )
      y_pred_national_scaled = huber_model_global.predict(
          X_pred_national_scaled
      )  # Use global model
      predicted_per_capita_national_admissions = (
          scaler_admissions.inverse_transform(
              y_pred_national_scaled.reshape(-1, 1)
          ).flatten()
      )

      national_ili_to_synthesize['synthetic_per_capita_admissions'] = (
          predicted_per_capita_national_admissions
      )
      national_ili_to_synthesize['synthetic_per_capita_admissions'] = (
          np.maximum(
              national_pseudo_pc_rate,
              national_ili_to_synthesize['synthetic_per_capita_admissions'],
          )
      )
      national_ili_to_synthesize['synthetic_total_admissions'] = (
          national_ili_to_synthesize['synthetic_per_capita_admissions']
          * national_population
      )
      synthetic_national_admissions_list.append(national_ili_to_synthesize)

    if synthetic_national_admissions_list:
      synthetic_national_admissions_df = pd.concat(
          synthetic_national_admissions_list, ignore_index=True
      )
      synthetic_national_admissions_df = synthetic_national_admissions_df[[
          'target_end_date',
          'population',
          'synthetic_total_admissions',
          'synthetic_per_capita_admissions',
      ]].rename(
          columns={
              'synthetic_total_admissions': TARGET_STR,
              'synthetic_per_capita_admissions': 'per_capita_admissions',
          }
      )
    else:
      synthetic_national_admissions_df = pd.DataFrame(
          columns=[
              'target_end_date',
              'population',
              TARGET_STR,
              'per_capita_admissions',
          ]
      )
  else:
    synthetic_national_admissions_df = pd.DataFrame(
        columns=[
            'target_end_date',
            'population',
            TARGET_STR,
            'per_capita_admissions',
        ]
    )

  # Combine actual and synthetic national admissions
  full_national_history_df = pd.concat([
      national_admissions_actual[
          ['target_end_date', 'population', TARGET_STR, 'per_capita_admissions']
      ],
      synthetic_national_admissions_df[
          ['target_end_date', 'population', TARGET_STR, 'per_capita_admissions']
      ],
  ]).reset_index(drop=True)
  full_national_history_df = full_national_history_df.drop_duplicates(
      subset=['target_end_date'], keep='first'
  )
  full_national_history_df = full_national_history_df.sort_values(
      'target_end_date'
  ).reset_index(drop=True)
  full_national_history_df['epiweek'] = full_national_history_df[
      'target_end_date'
  ].apply(get_epiweek)
  full_national_history_df[TARGET_STR] = (
      np.maximum(0, full_national_history_df[TARGET_STR]).round().astype(int)
  )

  # Pre-compute national climatological medians for national deviation feature
  national_climatology_medians_lookup = {}
  ew_to_cyclic_weeks = {
      ew: get_cyclic_weeks(ew) for ew in range(1, 53)
  }  # Re-using for national
  for ew in range(1, 53):
    cyclic_ews = ew_to_cyclic_weeks[ew]
    national_clim_data_pc = full_national_history_df[
        full_national_history_df['epiweek'].isin(cyclic_ews)
    ]['per_capita_admissions']
    median_val = (
        national_clim_data_pc.median()
        if not national_clim_data_pc.empty
        else national_pseudo_pc_rate
    )
    national_climatology_medians_lookup[ew] = np.maximum(
        national_pseudo_pc_rate, median_val
    )

  # Create a lookup for national full history for efficiency
  national_full_history_lookup = full_national_history_df.set_index(
      'target_end_date'
  )['per_capita_admissions']
  # End NEW National Data Preparation

  # --- OPTIMIZATION: Pre-compute Climatology Lookups (Code 1 principles) ---
  # This section remains the same, calculating state-level climatology.
  climatology_base_df = full_history_df.copy()
  climatology_quantiles_lookup = {}
  climatology_medians_lookup = {}
  climatology_pc_percentiles_lookup = {}

  for loc_code, loc_data in climatology_base_df.groupby('location'):
    pseudo_per_capita_rate_for_loc = get_location_pseudo_pc_rate(
        loc_code, loc_populations_map
    )
    for ew in range(1, 53):
      cyclic_ews_for_median = ew_to_cyclic_weeks[
          ew
      ]  # Re-using pre-computed cyclic weeks
      clim_data_for_median_pc = loc_data[
          loc_data['epiweek'].isin(cyclic_ews_for_median)
      ]['per_capita_admissions']
      if clim_data_for_median_pc.empty:
        median_val = pseudo_per_capita_rate_for_loc
      else:
        median_val = clim_data_for_median_pc.median()
      climatology_medians_lookup[(loc_code, ew)] = np.maximum(
          pseudo_per_capita_rate_for_loc, median_val
      )

      p10_pc = (
          np.percentile(clim_data_for_median_pc, 10)
          if not clim_data_for_median_pc.empty
          else pseudo_per_capita_rate_for_loc
      )
      p50_pc = (
          np.percentile(clim_data_for_median_pc, 50)
          if not clim_data_for_median_pc.empty
          else pseudo_per_capita_rate_for_loc + (2 * EPSILON_STABILITY)
      )
      p10_pc = np.maximum(pseudo_per_capita_rate_for_loc, p10_pc)
      p50_pc = np.maximum(p10_pc + EPSILON_STABILITY, p50_pc)
      climatology_pc_percentiles_lookup[(loc_code, ew)] = {
          'p10': p10_pc,
          'p50': p50_pc,
      }

      cyclic_ews = ew_to_cyclic_weeks[ew]
      clim_data_for_quantiles = loc_data[loc_data['epiweek'].isin(cyclic_ews)][
          TARGET_STR
      ]
      if clim_data_for_quantiles.empty or clim_data_for_quantiles.sum() == 0:
        climatological_q_array = np.full(len(QUANTILES), 0)
      else:
        climatological_q_array = np.percentile(
            clim_data_for_quantiles.values, [q * 100 for q in QUANTILES]
        )
      climatological_q_array = np.maximum(0, climatological_q_array)
      climatological_q_array = np.maximum.accumulate(climatological_q_array)
      climatology_quantiles_lookup[(loc_code, ew)] = climatological_q_array

  # Pre-compute state vectors for relevant history for efficient lookup and scaler fitting (Code 2 principles)
  relevant_full_history_for_analogs = (
      full_history_df.copy()
  )  # State-level history

  history_state_vectors_list = []
  history_metadata_list = []

  for loc_code in relevant_full_history_for_analogs['location'].unique():
    loc_data = relevant_full_history_for_analogs[
        relevant_full_history_for_analogs['location'] == loc_code
    ].sort_values('target_end_date')

    if len(loc_data) >= LOOKBACK_WEEKS + 1:
      log_transformed_pc_series = np.log(
          loc_data['per_capita_admissions'] + EPSILON_STABILITY
      )

      for i in range(LOOKBACK_WEEKS, len(loc_data)):
        current_date_for_analog = loc_data.iloc[i]['target_end_date']

        # --- Local features ---
        recent_log_transformed_pc = log_transformed_pc_series.iloc[
            i - LOOKBACK_WEEKS : i + 1
        ].values
        log_growth_rates_local = np.diff(recent_log_transformed_pc)
        if np.std(log_growth_rates_local) < MIN_TREND_STDEV_FOR_ANALOG:
          continue  # Skip if local trend is too flat

        current_pc_admission_for_deviation = loc_data.iloc[i][
            'per_capita_admissions'
        ]
        current_epiweek_history = loc_data.iloc[i]['epiweek']
        climatological_median_pc_history = climatology_medians_lookup.get(
            (loc_code, current_epiweek_history),
            get_location_pseudo_pc_rate(loc_code, loc_populations_map),
        )
        climatological_deviation_local = np.log(
            current_pc_admission_for_deviation + EPSILON_STABILITY
        ) - np.log(climatological_median_pc_history + EPSILON_STABILITY)
        sin_ew = np.sin(2 * np.pi * current_epiweek_history / 52)
        cos_ew = np.cos(2 * np.pi * current_epiweek_history / 52)

        # --- National features (NEW) ---
        # Get national data corresponding to current_date_for_analog's lookback window
        national_recent_dates = pd.date_range(
            end=current_date_for_analog,
            periods=LOOKBACK_WEEKS + 1,
            freq='W-SAT',
        )
        national_pc_admissions_history = []
        for d in national_recent_dates:
          val = national_full_history_lookup.get(
              d, national_pseudo_pc_rate
          )  # Fallback to national pseudo rate
          national_pc_admissions_history.append(val)
        national_pc_admissions_history = np.array(
            national_pc_admissions_history
        )

        log_transformed_pc_national = np.log(
            national_pc_admissions_history + EPSILON_STABILITY
        )
        log_growth_rates_national = np.diff(log_transformed_pc_national)

        # National climatological deviation for this date
        national_current_epiweek = get_epiweek(current_date_for_analog)
        national_median_pc = national_climatology_medians_lookup.get(
            national_current_epiweek, national_pseudo_pc_rate
        )
        national_deviation = np.log(
            national_pc_admissions_history[-1] + EPSILON_STABILITY
        ) - np.log(national_median_pc + EPSILON_STABILITY)

        # Combine local and national features into a single state vector
        state_vector = np.concatenate([
            log_growth_rates_local,  # Local recent week-over-week growth rates
            [
                recent_log_transformed_pc[-1]
            ],  # Local current log-transformed level
            [
                climatological_deviation_local
            ],  # Local deviation from climatological median
            [sin_ew],  # Cyclical seasonal component (sine)
            [cos_ew],  # Cyclical seasonal component (cosine)
            log_growth_rates_national,  # NEW: National recent week-over-week growth rates
            [
                log_transformed_pc_national[-1]
            ],  # NEW: National current log-transformed level
            [
                national_deviation
            ],  # NEW: National deviation from climatological median
        ])

        history_state_vectors_list.append(state_vector)
        history_metadata_list.append({
            'location': loc_code,
            'target_end_date': current_date_for_analog,
            'last_pc_admission': loc_data.iloc[i]['per_capita_admissions'],
        })

  history_state_df = pd.DataFrame(history_metadata_list)
  historical_state_vectors_np = np.array(history_state_vectors_list)

  # Robust StandardScaler for state vectors (Code 2 principle)
  state_scaler = None
  has_variance_features_mask = None
  historical_state_vectors_scaled = historical_state_vectors_np.copy()

  if (
      historical_state_vectors_np.shape[0] > MIN_ANALOGS_FOR_DYNAMIC_BLEND
      and historical_state_vectors_np.shape[0] > 1
  ):
    stds = np.std(historical_state_vectors_np, axis=0)
    has_variance_features_mask = stds > EPSILON_STABILITY
    if np.any(has_variance_features_mask):
      try:
        state_scaler = StandardScaler()
        state_scaler.fit(
            historical_state_vectors_np[:, has_variance_features_mask]
        )
        historical_state_vectors_scaled[:, has_variance_features_mask] = (
            state_scaler.transform(
                historical_state_vectors_np[:, has_variance_features_mask]
            )
        )
      except Exception as e:
        state_scaler = None
        warnings.warn(
            f'State vector StandardScaler fitting failed: {e}. Dynamic forecast'
            ' may rely on unscaled vectors for these features.',
            UserWarning,
        )
    else:
      warnings.warn(
          'Historical state vectors have no variance in any feature; cannot fit'
          ' StandardScaler. Relying heavily on climatology for dynamic blend.',
          UserWarning,
      )
  else:
    warnings.warn(
        'Not enough historical state vectors to fit StandardScaler or perform'
        ' dynamic forecast. Will rely heavily on climatology.',
        UserWarning,
    )

  # Pre-filter and pre-group historical states for current fold
  history_mask_for_current_fold = (
      history_state_df['target_end_date'] < current_fold_train_end_date
  )
  filtered_history_state_df_for_fold = history_state_df[
      history_mask_for_current_fold
  ].reset_index(drop=True)
  filtered_historical_state_vectors_scaled_for_fold = (
      historical_state_vectors_scaled[history_mask_for_current_fold]
  )

  grouped_filtered_historical_states = {}
  if not filtered_history_state_df_for_fold.empty:
    for loc_code in filtered_history_state_df_for_fold['location'].unique():
      loc_mask = filtered_history_state_df_for_fold['location'] == loc_code
      grouped_filtered_historical_states[loc_code] = {
          'metadata': filtered_history_state_df_for_fold[loc_mask],
          'vectors_scaled': filtered_historical_state_vectors_scaled_for_fold[
              loc_mask
          ],
      }

  # --- 2. Prepare test_x for predictions ---
  test_x['epiweek'] = test_x['target_end_date'].apply(get_epiweek)
  test_x['population'] = test_x['location'].map(loc_populations_map)

  # --- New: Define horizon-dependent blending factors ---
  horizon_dynamic_boost = {
      0: 1.2,
      1: 1.1,
      2: 0.9,
      3: 0.8,
  }

  # --- 3. Core Forecasting Logic (Hybrid: Blending Trajectory Matching with Climatology) ---

  predictions_list = []

  full_history_lookup = full_history_df.set_index(
      ['target_end_date', 'location']
  )['per_capita_admissions']
  # national_full_history_lookup is already defined above for national features

  for idx, row in test_x.iterrows():
    ref_date = row['reference_date']
    loc = row['location']
    target_end_date_forecast = row['target_end_date']
    target_epiweek_forecast = row['epiweek']
    horizon = row['horizon']
    location_population = row['population']

    pseudo_per_capita_rate_for_loc = get_location_pseudo_pc_rate(
        loc, loc_populations_map
    )

    climatological_forecast_quantiles_total = climatology_quantiles_lookup.get(
        (loc, target_epiweek_forecast), np.full(len(QUANTILES), 0)
    )
    dynamic_forecast_quantiles_total = np.copy(
        climatological_forecast_quantiles_total
    )

    weight_from_analogs = 0.0
    weight_from_activity = 0.0

    # A. Get Current Season's Recent History for query vector (Code 2 principle)

    # --- Local features for query vector ---
    recent_dates_for_query = pd.date_range(
        end=current_fold_train_end_date,
        periods=LOOKBACK_WEEKS + 1,
        freq='W-SAT',
    )
    current_pc_admissions_history = []
    for d in recent_dates_for_query:
      val = full_history_lookup.get((d, loc), np.nan)
      current_pc_admissions_history.append(val)
    current_pc_admissions_history = np.array(current_pc_admissions_history)

    climatological_recent_per_capita = []
    for d in recent_dates_for_query:
      clim_val = climatology_medians_lookup.get(
          (loc, get_epiweek(d)), pseudo_per_capita_rate_for_loc
      )
      climatological_recent_per_capita.append(clim_val)

    current_pc_admissions_history_filled = np.where(
        pd.isna(current_pc_admissions_history),
        climatological_recent_per_capita,
        current_pc_admissions_history,
    )
    current_pc_admissions_history_filled = np.maximum(
        pseudo_per_capita_rate_for_loc, current_pc_admissions_history_filled
    )

    last_pc_admission_current = current_pc_admissions_history_filled[-1]

    log_transformed_pc_current = np.log(
        current_pc_admissions_history_filled + EPSILON_STABILITY
    )
    log_growth_rates_current_local = np.diff(log_transformed_pc_current)

    current_pc_admission_for_query_deviation = last_pc_admission_current
    current_epiweek_for_query = get_epiweek(current_fold_train_end_date)
    climatological_median_pc_query = climatology_medians_lookup.get(
        (loc, current_epiweek_for_query), pseudo_per_capita_rate_for_loc
    )
    climatological_deviation_current_local = np.log(
        current_pc_admission_for_query_deviation + EPSILON_STABILITY
    ) - np.log(climatological_median_pc_query + EPSILON_STABILITY)
    sin_ew_query = np.sin(2 * np.pi * current_epiweek_for_query / 52)
    cos_ew_query = np.cos(2 * np.pi * current_epiweek_for_query / 52)

    # --- National features for query vector (NEW) ---
    national_pc_admissions_query_history = []
    for d in recent_dates_for_query:  # Use the same recent_dates_for_query
      val_national = national_full_history_lookup.get(
          d, national_pseudo_pc_rate
      )
      national_pc_admissions_query_history.append(val_national)
    national_pc_admissions_query_history = np.array(
        national_pc_admissions_query_history
    )
    national_pc_admissions_query_history = np.maximum(
        national_pseudo_pc_rate, national_pc_admissions_query_history
    )  # Ensure floor

    log_transformed_pc_national_query = np.log(
        national_pc_admissions_query_history + EPSILON_STABILITY
    )
    log_growth_rates_current_national = np.diff(
        log_transformed_pc_national_query
    )

    national_current_epiweek_query = get_epiweek(current_fold_train_end_date)
    national_median_pc_query = national_climatology_medians_lookup.get(
        national_current_epiweek_query, national_pseudo_pc_rate
    )
    national_deviation_query = np.log(
        national_pc_admissions_query_history[-1] + EPSILON_STABILITY
    ) - np.log(national_median_pc_query + EPSILON_STABILITY)

    # Construct the query vector representing the current state (Local + National features)
    query_vector = np.concatenate([
        log_growth_rates_current_local,
        [log_transformed_pc_current[-1]],
        [climatological_deviation_current_local],
        [sin_ew_query],
        [cos_ew_query],
        log_growth_rates_current_national,  # NEW: National growth rates
        [log_transformed_pc_national_query[-1]],  # NEW: National current level
        [national_deviation_query],  # NEW: National climatological deviation
    ])

    # Robust query_vector scaling
    query_vector_to_use = query_vector.copy()
    if (
        state_scaler is not None
        and has_variance_features_mask is not None
        and np.any(has_variance_features_mask)
    ):
      query_vector_to_use[has_variance_features_mask] = state_scaler.transform(
          query_vector_to_use[has_variance_features_mask].reshape(1, -1)
      ).flatten()

    # B. Find Nearest Neighbors in History (combining local/global, Code 2 principle)
    final_neighbors_list = []
    processed_neighbor_keys = set()

    if (
        state_scaler is not None
        and query_vector_to_use.size > 0
        and has_variance_features_mask is not None
        and np.any(has_variance_features_mask)
    ):

      loc_history_data_for_query = grouped_filtered_historical_states.get(
          loc, None
      )

      if (
          loc_history_data_for_query
          and not loc_history_data_for_query['metadata'].empty
      ):
        loc_vectors_scaled = loc_history_data_for_query['vectors_scaled']
        if loc_vectors_scaled.shape[0] > 0:
          dists_local = cdist(
              query_vector_to_use.reshape(1, -1),
              loc_vectors_scaled,
              metric='euclidean',
          ).flatten()
          for k in np.argsort(dists_local):
            meta_row = loc_history_data_for_query['metadata'].iloc[k]
            key = (meta_row['target_end_date'], meta_row['location'])
            if key not in processed_neighbor_keys:
              final_neighbors_list.append((dists_local[k], meta_row))
              processed_neighbor_keys.add(key)
            if len(final_neighbors_list) >= NUM_ANALOGS:
              break

      if (
          len(final_neighbors_list) < NUM_ANALOGS
          and not filtered_history_state_df_for_fold.empty
      ):
        if filtered_historical_state_vectors_scaled_for_fold.shape[0] > 0:
          dists_global = cdist(
              query_vector_to_use.reshape(1, -1),
              filtered_historical_state_vectors_scaled_for_fold,
              metric='euclidean',
          ).flatten()
          for k in np.argsort(dists_global):
            meta_row = filtered_history_state_df_for_fold.iloc[k]
            key = (meta_row['target_end_date'], meta_row['location'])
            if key not in processed_neighbor_keys:
              final_neighbors_list.append((dists_global[k], meta_row))
              processed_neighbor_keys.add(key)
            if len(final_neighbors_list) >= NUM_ANALOGS:
              break

    final_neighbors_list.sort(key=lambda x: x[0])
    neighbors_metadata = pd.DataFrame(
        [n[1] for n in final_neighbors_list[:NUM_ANALOGS]]
    )

    num_neighbors_found = len(neighbors_metadata)

    if num_neighbors_found >= MIN_ANALOGS_FOR_DYNAMIC_BLEND:
      ensemble_forecast_for_current_horizon = []

      for _, neighbor_row in neighbors_metadata.iterrows():
        analog_end_date = neighbor_row['target_end_date']
        last_pc_admission_analog = np.maximum(
            pseudo_per_capita_rate_for_loc, neighbor_row['last_pc_admission']
        )

        scaling_factor = last_pc_admission_current / last_pc_admission_analog
        scaling_factor = np.clip(
            scaling_factor, 1 / MAX_SCALING_FACTOR, MAX_SCALING_FACTOR
        )

        forecast_date_for_analog = analog_end_date + pd.Timedelta(
            weeks=horizon + 1
        )

        analog_loc = neighbor_row['location']
        pc_admissions_at_horizon = full_history_lookup.get(
            (forecast_date_for_analog, analog_loc), np.nan
        )

        if pd.isna(pc_admissions_at_horizon):
          pc_admissions_at_horizon = climatology_medians_lookup.get(
              (analog_loc, get_epiweek(forecast_date_for_analog)),
              get_location_pseudo_pc_rate(analog_loc, loc_populations_map),
          )
        pc_admissions_at_horizon = np.maximum(
            get_location_pseudo_pc_rate(analog_loc, loc_populations_map),
            pc_admissions_at_horizon,
        )

        scaled_total_admissions_at_horizon = (
            pc_admissions_at_horizon * scaling_factor * location_population
        )
        ensemble_forecast_for_current_horizon.append(
            scaled_total_admissions_at_horizon
        )

      ensemble_forecast_for_current_horizon_clean = np.array(
          ensemble_forecast_for_current_horizon
      )
      ensemble_forecast_for_current_horizon_clean = np.nan_to_num(
          ensemble_forecast_for_current_horizon_clean, nan=0.0
      )

      if ensemble_forecast_for_current_horizon_clean.size > 0 and not np.all(
          ensemble_forecast_for_current_horizon_clean == 0
      ):
        dynamic_forecast_quantiles_total = np.percentile(
            ensemble_forecast_for_current_horizon_clean,
            [q * 100 for q in QUANTILES],
        )
        dynamic_forecast_quantiles_total = np.maximum(
            0, dynamic_forecast_quantiles_total
        )
        dynamic_forecast_quantiles_total = np.maximum.accumulate(
            dynamic_forecast_quantiles_total
        )

    # C. Blending Logic - This remains consistent with the original hybrid approach.
    denominator_for_analogs = (
        NUM_ANALOGS - MIN_ANALOGS_FOR_DYNAMIC_BLEND + EPSILON_STABILITY
    )

    if num_neighbors_found < MIN_ANALOGS_FOR_DYNAMIC_BLEND:
      weight_from_analogs = 0.0
    elif num_neighbors_found >= NUM_ANALOGS:
      weight_from_analogs = 1.0
    else:
      weight_from_analogs = (
          num_neighbors_found - MIN_ANALOGS_FOR_DYNAMIC_BLEND
      ) / denominator_for_analogs

    pc_thresholds = climatology_pc_percentiles_lookup.get(
        (loc, current_epiweek_for_query),
        {
            'p10': pseudo_per_capita_rate_for_loc,
            'p50': pseudo_per_capita_rate_for_loc + (2 * EPSILON_STABILITY),
        },
    )
    activity_ramp_start = pc_thresholds['p10']
    activity_ramp_end = pc_thresholds['p50']

    if activity_ramp_end <= activity_ramp_start:
      activity_ramp_end = activity_ramp_start + EPSILON_STABILITY

    if last_pc_admission_current <= activity_ramp_start:
      weight_from_activity = 0.0
    elif last_pc_admission_current >= activity_ramp_end:
      weight_from_activity = 1.0
    else:
      weight_from_activity = (
          last_pc_admission_current - activity_ramp_start
      ) / (activity_ramp_end - activity_ramp_start)

    initial_dynamic_weight = (weight_from_analogs + weight_from_activity) / 2.0

    current_horizon_boost = horizon_dynamic_boost.get(horizon, 1.0)
    effective_dynamic_weight = np.clip(
        initial_dynamic_weight * current_horizon_boost, 0.0, 1.0
    )

    blended_quantiles = (
        effective_dynamic_weight * dynamic_forecast_quantiles_total
    ) + (
        (1 - effective_dynamic_weight) * climatological_forecast_quantiles_total
    )

    final_quantiles_float = np.maximum.accumulate(blended_quantiles)
    final_quantiles_rounded = (
        np.maximum(0, final_quantiles_float).round().astype(int)
    )
    final_quantiles = np.maximum.accumulate(final_quantiles_rounded)

    predictions_list.append(final_quantiles)

  # --- 4. Format Output ---
  test_y_hat_quantiles = pd.DataFrame(
      predictions_list,
      columns=[f'quantile_{q}' for q in QUANTILES],
      index=test_x.index,
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
