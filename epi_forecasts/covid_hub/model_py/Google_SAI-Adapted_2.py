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

# pylint: disable=g-bad-import-order,missing-module-docstring,unused-import,g-import-not-at-top,g-line-too-long,unused-variable,used-before-assignment,redefined-outer-name,pointless-statement,unnecessary-pass,invalid-name,g-bare-generic,reimported,g-importing-member,g-doc-return-or-yield,g-doc-args,g-short-docstring-punctuation,cell-var-from-loop,f-string-without-interpolation,g-long-ternary

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
MODEL_NAME = 'Google_SAI-Adapted_2'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Described Method:
# 1. Stochastic Age-structured Metapopulation Model: Simulates disease transmission across geographic patches and spatial spread via real-world air travel and commuting networks.
#    -> Proxy in this model: The model uses 'location' and 'population' to define geographic patches. National-level aggregate signals ('MODEL_INPUT_NATIONAL_AVG', its lags and EWMAs) statistically represent spatial spread/metapopulation dynamics. Local deviations from national trends ('local_vs_national_diff') capture specific location behaviors. To further enhance this proxy, new interaction features (e.g., 'population_x_...', 'horizon_x_...', 'national_ewma_short_x_week_sin_lag1') are introduced. Crucially, features like 'national_ewma_diff_short_long_lag_1' and its interactions provide a statistical measure of *metapopulation momentum and its sensitivity to local population size*, better reflecting the dynamic influence of mobility networks. These statistically approximate how subpopulation characteristics (e.g., size, seasonality sensitivity) or forecast horizon modulate these dynamics, offering a more refined proxy for 'age-structured' components and 'mobility network' influence without explicit simulation.
# 2. SLIR Compartmental Definition: Utilizes an age-structured Susceptible-Latent-Infectious-Removed model within each subpopulation to track disease status.
#    -> Proxy in this model: This is statistically approximated by autoregressive features derived from the log-transformed target variable ('MODEL_INPUT_LOG_TARGET_MEDIAN', 'Q025', 'Q075'). These include various lags, rates of change (e.g., '_lag_diff', '_ewma_diff'), and uncertainty bounds (e.g., '_iqr'). New features like '_acceleration' (second-order difference) directly model the momentum and turning points of the epidemic curve, providing a stronger statistical signal of changes within SLIR compartments. 'local_dev_from_short_ewma_lag_1' captures immediate departures from trends, and 'log_target_iqr_to_median_ratio_lag_1' proxies the relative variability of the disease state. Further enhancements like 'iqr_lag_1_x_acceleration_lag_1' and 'log_target_iqr_to_median_ratio_ewma_short_lag_1' explicitly model the *dynamics of uncertainty itself*, which is a statistical representation of how local prevalence and contact matrices might influence the spread and predictability of disease progression within compartments. The monotonicity constraint on quantiles helps maintain consistency in the probabilistic description of disease state.
# 3. Stochastic Simulation: Initialize agents and simulate time steps using a Monte Carlo approach: update agent locations based on mobility fluxes, then update infection status based on local prevalence and contact matrices.
#    -> Proxy in this model: While not directly simulating agents, mobility, or contact matrices, this principle is statistically approximated by introducing stochasticity via bootstrapped residuals. The model generates multiple plausible future trajectories by adding randomly sampled residuals (from historical training errors, conditionally sampled based on prediction magnitude) to the median point forecasts. This simulates the *uncertainty in future outcomes* that would arise from agent-level transitions and mobility fluxes. The *richer feature set* (described in Principles 1 & 2's proxies) directly informs the median and quantile models, allowing them to learn how historical mobility proxies, prevalence, and uncertainty dynamics influence the overall predicted distribution and its spread, thereby indirectly reflecting the *effects* of the underlying stochastic processes.
# 4. Ensemble Generation: Execute independent simulation runs to generate a distribution of possible outcomes.
#    -> Adherence: The model directly adheres to this by generating N_SIMULATIONS independent trajectories for each location and forecast horizon. These simulations are then used to compute the required quantile forecasts for each state, ensuring a full probabilistic distribution is provided, thereby completing the ensemble generation and aggregation step. A crucial improvement scales the spread of these simulations to match the statistically learned IQR from the quantile models, ensuring better calibration of the generated ensemble distribution in line with the model's learned uncertainty.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import deque  # For efficient historical buffer

TARGET_STR = 'Total COVID-19 Admissions'
TARGET_PER_POP_STR = f'{TARGET_STR}_per_100k'
LOG_TARGET_PER_POP_STR = f'log_{TARGET_PER_POP_STR}'

QUANTILES = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]

N_SIMULATIONS = 500
EWMA_SPAN_SHORT = 4
EWMA_SPAN_LONG = 8
LAGS = [1, 2, 3, 4]
DIFF_LAGS = [1, 2, 3]  # Lags for rate of change features
EPSILON = 1e-6  # For avoiding division by zero in ratios for certain features

# Feature prefixes for easier management
MODEL_INPUT_LOG_TARGET_MEDIAN = '_model_input_log_target_median'
MODEL_INPUT_LOG_TARGET_Q025 = '_model_input_log_target_q025'
MODEL_INPUT_LOG_TARGET_Q075 = '_model_input_log_target_q075'
MODEL_INPUT_NATIONAL_AVG = '_model_input_national_avg'
# New prefixes for storing current EWMA states incrementally
CURRENT_EWMA_MEDIAN_PREFIX = '_current_ewma_median_span'
CURRENT_EWMA_NATIONAL_PREFIX = '_current_ewma_national_span'


def create_static_features(
    df, min_date_overall
):
  """Creates static and time-based features that don't depend on historical target values.

  Aligns with "Network Construction" by providing location attributes.
  """
  df_copy = df.copy()
  df_copy['week_of_year'] = (
      df_copy['target_end_date'].dt.isocalendar().week.astype(int)
  )
  df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['week_of_year'] / 52)
  df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['week_of_year'] / 52)
  df_copy['log_population'] = np.log1p(df_copy['population'])
  df_copy['time_idx'] = (
      df_copy['target_end_date'] - min_date_overall
  ).dt.days // 7
  # 'horizon' is already present and handled by fit_and_predict_fn for consistency
  return df_copy


# This function is now specifically for batch feature generation for training data.
# It will NOT be used within the prediction loop in the refactored solution for efficiency.
def _compute_dynamic_features_for_training(df):
  """Creates lagged and EWMA features based on current and historical values of

  _model_input_log_target_median, _model_input_q025, _model_input_q075, and
  _model_input_national_avg.
  This function uses `groupby().shift()` and `ewm()` for batch processing (e.g.,
  for training
  or a consolidated history slice in prediction). These features proxy
  "Compartmental Definition"
  and "Metapopulation" dynamics, with added interaction features to enhance
  proxies.
  """
  df_copy = df.copy()

  # Lagged features for _model_input_log_target_median (proxy for SLIR dynamics)
  for lag in LAGS:
    df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_LOG_TARGET_MEDIAN].shift(lag)
    df_copy[f'{MODEL_INPUT_LOG_TARGET_Q025}_lag_{lag}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_LOG_TARGET_Q025].shift(lag)
    df_copy[f'{MODEL_INPUT_LOG_TARGET_Q075}_lag_{lag}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_LOG_TARGET_Q075].shift(lag)
    # Ensure IQR is non-negative, proxy for variability in compartmental states
    df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_iqr_lag_{lag}'] = np.maximum(
        0,
        df_copy[f'{MODEL_INPUT_LOG_TARGET_Q075}_lag_{lag}']
        - df_copy[f'{MODEL_INPUT_LOG_TARGET_Q025}_lag_{lag}'],
    )

  # Add "rate of change" features (proxy for acceleration/deceleration in compartmental states)
  for lag_diff in DIFF_LAGS:
    df_copy[
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag_diff}_diff'
    ] = df_copy.groupby('location')[MODEL_INPUT_LOG_TARGET_MEDIAN].shift(
        lag_diff
    ) - df_copy.groupby(
        'location'
    )[
        MODEL_INPUT_LOG_TARGET_MEDIAN
    ].shift(
        lag_diff + 1
    )
  # Longer-term rate of change (e.g., current lag_1 vs lag_3) - NEW FEATURE for more robust metapopulation proxy
  df_copy[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_vs_lag_3_diff'
  ] = df_copy.groupby('location')[MODEL_INPUT_LOG_TARGET_MEDIAN].shift(
      1
  ) - df_copy.groupby(
      'location'
  )[
      MODEL_INPUT_LOG_TARGET_MEDIAN
  ].shift(
      3
  )

  # Exponential Weighted Moving Average (EWMA) of the log-transformed target for smoothed trends
  for span in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]:
    # Compute EWMA for each location
    df_copy[f'{CURRENT_EWMA_MEDIAN_PREFIX}_{span}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_LOG_TARGET_MEDIAN].transform(
        lambda x: x.ewm(span=span, adjust=False, min_periods=1).mean()
    )
    # Shift EWMA by 1 to use as a feature for the current week (lagged EWMA)
    df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{span}_lag_1'] = (
        df_copy.groupby('location')[
            f'{CURRENT_EWMA_MEDIAN_PREFIX}_{span}'
        ].shift(1)
    )

  # Difference between short and long EWMA (smoothed rate of change/momentum proxy for compartmental states)
  df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_diff_short_long_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_SHORT}_lag_1']
      - df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_LONG}_lag_1']
  )

  # National trend/metapopulation proxy: lagged national average admissions
  for lag in LAGS:
    df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_NATIONAL_AVG].shift(lag)

  # Add "rate of change" features for national average (proxy for acceleration/deceleration in metapopulation spread)
  for lag_diff in DIFF_LAGS:
    df_copy[
        f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag_diff}_diff'
    ] = df_copy.groupby('location')[MODEL_INPUT_NATIONAL_AVG].shift(
        lag_diff
    ) - df_copy.groupby(
        'location'
    )[
        MODEL_INPUT_NATIONAL_AVG
    ].shift(
        lag_diff + 1
    )

  # Exponential Weighted Moving Average (EWMA) of national_admissions_avg for a smoothed national trend
  for span in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]:
    df_copy[f'{CURRENT_EWMA_NATIONAL_PREFIX}_{span}'] = df_copy.groupby(
        'location'
    )[MODEL_INPUT_NATIONAL_AVG].transform(
        lambda x: x.ewm(span=span, adjust=False, min_periods=1).mean()
    )
    df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{span}_lag_1'] = df_copy.groupby(
        'location'
    )[f'{CURRENT_EWMA_NATIONAL_PREFIX}_{span}'].shift(1)

  # NEW MINOR IMPROVEMENT: National EWMA diff short-long (momentum of national trend)
  df_copy[f'national_ewma_diff_short_long_lag_1'] = (
      df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_SHORT}_lag_1']
      - df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_LONG}_lag_1']
  )

  # Local deviation from national trend (proxy for relative metapopulation dynamics)
  df_copy['local_vs_national_lag_1_diff'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1']
      - df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_1']
  )

  # Local vs national EWMA differences (trend-based metapopulation interaction)
  df_copy['local_vs_national_ewma_short_diff_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_SHORT}_lag_1']
      - df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_SHORT}_lag_1']
  )
  df_copy['local_vs_national_ewma_long_diff_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_LONG}_lag_1']
      - df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_LONG}_lag_1']
  )

  # NEW MINOR IMPROVEMENT: Local vs National EWMA diff of short-long differences
  df_copy['local_vs_national_ewma_diff_short_long_lag_1'] = (
      df_copy['local_vs_national_ewma_short_diff_lag_1']
      - df_copy['local_vs_national_ewma_long_diff_lag_1']
  )

  # --- NEW INTERACTION FEATURES (Minor Improvement for Metapopulation/Age-Structured Proxy) ---
  # These features aim to capture how population characteristics interact with disease dynamics.
  df_copy['population_x_log_target_lag1'] = (
      df_copy['log_population']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1']
  )
  df_copy['population_x_log_target_ewma_short_lag1'] = (
      df_copy['log_population']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_SHORT}_lag_1']
  )
  df_copy['population_x_local_vs_national_lag1_diff'] = (
      df_copy['log_population'] * df_copy['local_vs_national_lag_1_diff']
  )

  # Further refined interaction features (added for minor improvement)
  df_copy['log_population_x_week_sin'] = (
      df_copy['log_population'] * df_copy['week_sin']
  )
  df_copy['log_population_x_week_cos'] = (
      df_copy['log_population'] * df_copy['week_cos']
  )
  # Note: MODEL_INPUT_LOG_TARGET_MEDIAN_lag_1_diff needs to be available before this calculation.
  df_copy[f'log_population_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'] = (
      df_copy['log_population']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff']
  )
  df_copy[f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_1'] = (
      df_copy['log_population'] * df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_1']
  )

  # New minor improvements:
  # 1. Acceleration (local target): difference of differences
  df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff']
      - df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_2_diff']
  )
  # 2. Acceleration (national average): difference of differences
  df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_acceleration_lag_1'] = (
      df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_1_diff']
      - df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_2_diff']
  )
  # 3. Local deviation from short-term EWMA: captures immediate departure from smoothed trend
  df_copy['local_dev_from_short_ewma_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1']
      - df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_SHORT}_lag_1']
  )

  # 4. Interaction: log_population * local_acceleration (how population size affects local speed-up/slow-down)
  df_copy['population_x_local_acceleration_lag1'] = (
      df_copy['log_population']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1']
  )
  # 5. Interaction: log_population * national_average_lag_2 (how population interacts with a slightly older national signal)
  df_copy[f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_2'] = (
      df_copy['log_population'] * df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_lag_2']
  )
  # 6. Interaction: national_ewma_short_x_week_sin (Interaction between national trend and seasonality)
  df_copy['national_ewma_short_x_week_sin_lag1'] = (
      df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_SHORT}_lag_1']
      * df_copy['week_sin']
  )
  # 7. Lagged IQR ratio (IQR/Median): relative spread of log target
  # Guard against division by zero by adding a small epsilon
  df_copy['log_target_iqr_to_median_ratio_lag_1'] = df_copy[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_iqr_lag_1'
  ] / (df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1'] + EPSILON)

  # NEW MINOR IMPROVEMENT: Horizon interaction features
  df_copy['horizon_x_log_population'] = (
      df_copy['horizon'] * df_copy['log_population']
  )
  df_copy['horizon_x_week_sin'] = df_copy['horizon'] * df_copy['week_sin']
  df_copy['horizon_x_week_cos'] = df_copy['horizon'] * df_copy['week_cos']

  # NEW MINOR IMPROVEMENT: Additional Horizon interaction features
  df_copy[f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'] = (
      df_copy['horizon']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff']
  )
  df_copy[f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1'] = (
      df_copy['horizon']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1']
  )

  # NEW MINOR IMPROVEMENT: Population x national EWMA momentum
  df_copy['population_x_national_ewma_diff_short_long_lag_1'] = (
      df_copy['log_population'] * df_copy['national_ewma_diff_short_long_lag_1']
  )

  # NEW MINOR IMPROVEMENT: Population x national short EWMA
  df_copy['log_population_x_national_ewma_short_lag1'] = (
      df_copy['log_population']
      * df_copy[f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_SHORT}_lag_1']
  )

  # NEW MINOR IMPROVEMENT: IQR x Acceleration (proxy for uncertainty dynamics)
  df_copy[f'iqr_lag_1_x_acceleration_lag_1'] = (
      df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_iqr_lag_1']
      * df_copy[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1']
  )

  # NEW MINOR IMPROVEMENT: Smoothed relative IQR (proxy for evolving contact matrices/transmissibility)
  df_copy[
      f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}'
  ] = df_copy.groupby('location')[
      'log_target_iqr_to_median_ratio_lag_1'
  ].transform(
      lambda x: x.ewm(span=EWMA_SPAN_SHORT, adjust=False, min_periods=1).mean()
  )
  df_copy[f'log_target_iqr_to_median_ratio_ewma_short_lag_1'] = df_copy.groupby(
      'location'
  )[f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}'].shift(1)

  # NEW MINOR IMPROVEMENT: Horizon x local deviation from short EWMA
  df_copy['horizon_x_local_dev_from_short_ewma_lag_1'] = (
      df_copy['horizon'] * df_copy['local_dev_from_short_ewma_lag_1']
  )

  # NEW MINOR IMPROVEMENT: Horizon x relative IQR
  df_copy['horizon_x_log_target_iqr_to_median_ratio_lag_1'] = (
      df_copy['horizon'] * df_copy['log_target_iqr_to_median_ratio_lag_1']
  )

  # --- END NEW INTERACTION FEATURES ---

  # Fill NaNs for initial lags and EWMA features. These will appear for early time points where history is insufficient.
  # Filling with 0.0 is common for tree-based models and implies 'no historical signal'.
  feature_cols_to_fill = [
      col
      for col in df_copy.columns
      if any(
          x in col
          for x in [
              '_lag_',
              '_diff',
              '_ewma',
              'population_x_',
              'local_dev',
              'acceleration',
              'iqr_to_median_ratio',
              'national_ewma_short_x_week_sin',
              'horizon_x_',
          ]
      )
      and (
          MODEL_INPUT_LOG_TARGET_MEDIAN in col
          or MODEL_INPUT_NATIONAL_AVG in col
          or 'local_vs_national' in col
          or 'log_population' in col
          or 'week_sin' in col
          or 'week_cos' in col
          or 'horizon' in col
          or 'iqr_ratio' in col
      )
  ]
  df_copy[feature_cols_to_fill] = df_copy[feature_cols_to_fill].fillna(0.0)

  return df_copy


# Helper function for incremental EWMA calculation
def _calculate_incremental_ewma(current_value, previous_ewma, span):
  """Calculates the next EWMA value incrementally, consistent with pandas ewm(adjust=False, min_periods=1)."""
  alpha = 2 / (span + 1)

  # If no previous EWMA (e.g., first value in series), current value is the start
  if pd.isna(previous_ewma) or np.isnan(
      previous_ewma
  ):  # Added np.isnan for robustness
    return current_value

  # current_value should not be NaN in prediction loop, but if it were,
  # pandas ewm typically propagates the last valid EWMA until a new value appears.
  # For predictions, we assume current_value is always available.
  return alpha * current_value + (1 - alpha) * previous_ewma


def _extract_and_compute_features_for_step(
    current_row_static_features,
    loc_hist_deque,
):
  """Extracts and computes dynamic features for a single location and forecast step

  from its historical deque and current static features.
  Refactored for improved readability and reduced redundant dictionary lookups.
  """
  current_step_features = current_row_static_features.to_dict()

  def get_hist_val(deque_obj, lag, col_name, default=0.0):
    if len(deque_obj) >= lag:
      return deque_obj[-lag].get(col_name, default)
    return default

  # --- Extract static features to local variables for convenience ---
  log_population_val = current_row_static_features.get('log_population', 0.0)
  horizon_val = current_row_static_features.get('horizon', 0.0)
  week_sin_val = current_row_static_features.get('week_sin', 0.0)
  week_cos_val = current_row_static_features.get('week_cos', 0.0)

  # --- Extract all necessary base lagged values from history into local variables and update current_step_features ---
  # Using dictionaries to store lagged values for different lags
  median_lag = {}
  q025_lag = {}
  q075_lag = {}
  national_avg_lag = {}

  for lag in LAGS:
    median_lag[lag] = get_hist_val(
        loc_hist_deque, lag, MODEL_INPUT_LOG_TARGET_MEDIAN
    )
    q025_lag[lag] = get_hist_val(
        loc_hist_deque, lag, MODEL_INPUT_LOG_TARGET_Q025
    )
    q075_lag[lag] = get_hist_val(
        loc_hist_deque, lag, MODEL_INPUT_LOG_TARGET_Q075
    )
    national_avg_lag[lag] = get_hist_val(
        loc_hist_deque, lag, MODEL_INPUT_NATIONAL_AVG
    )

    current_step_features[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag}'] = (
        median_lag[lag]
    )
    current_step_features[f'{MODEL_INPUT_LOG_TARGET_Q025}_lag_{lag}'] = (
        q025_lag[lag]
    )
    current_step_features[f'{MODEL_INPUT_LOG_TARGET_Q075}_lag_{lag}'] = (
        q075_lag[lag]
    )
    current_step_features[f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag}'] = (
        national_avg_lag[lag]
    )

  # Extract EWMA values (lag 1 of the current EWMA state)
  median_ewma_short_lag1 = get_hist_val(
      loc_hist_deque, 1, f'{CURRENT_EWMA_MEDIAN_PREFIX}_{EWMA_SPAN_SHORT}'
  )
  median_ewma_long_lag1 = get_hist_val(
      loc_hist_deque, 1, f'{CURRENT_EWMA_MEDIAN_PREFIX}_{EWMA_SPAN_LONG}'
  )
  national_ewma_short_lag1 = get_hist_val(
      loc_hist_deque, 1, f'{CURRENT_EWMA_NATIONAL_PREFIX}_{EWMA_SPAN_SHORT}'
  )
  national_ewma_long_lag1 = get_hist_val(
      loc_hist_deque, 1, f'{CURRENT_EWMA_NATIONAL_PREFIX}_{EWMA_SPAN_LONG}'
  )
  iqr_ratio_ewma_short_lag1 = get_hist_val(
      loc_hist_deque,
      1,
      f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}',
  )

  current_step_features[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_SHORT}_lag_1'
  ] = median_ewma_short_lag1
  current_step_features[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{EWMA_SPAN_LONG}_lag_1'
  ] = median_ewma_long_lag1
  current_step_features[
      f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_SHORT}_lag_1'
  ] = national_ewma_short_lag1
  current_step_features[
      f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{EWMA_SPAN_LONG}_lag_1'
  ] = national_ewma_long_lag1
  current_step_features[f'log_target_iqr_to_median_ratio_ewma_short_lag_1'] = (
      iqr_ratio_ewma_short_lag1
  )

  # --- Compute derived features using the local variables ---

  # IQR features (proxy for variability in compartmental states)
  iqr_lag1 = 0.0  # Initialize for interaction features
  for lag in LAGS:
    iqr_val = np.maximum(0, q075_lag.get(lag, 0.0) - q025_lag.get(lag, 0.0))
    current_step_features[f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_iqr_lag_{lag}'] = (
        iqr_val
    )
    if lag == 1:
      iqr_lag1 = iqr_val  # Store for later use

  # Diff features (proxy for acceleration/deceleration in compartmental states and metapopulation spread)
  median_lag_diff = {}
  national_avg_lag_diff = {}
  for lag_diff in DIFF_LAGS:
    median_diff_val = median_lag.get(lag_diff, 0.0) - median_lag.get(
        lag_diff + 1, 0.0
    )
    national_diff_val = national_avg_lag.get(
        lag_diff, 0.0
    ) - national_avg_lag.get(lag_diff + 1, 0.0)
    current_step_features[
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag_diff}_diff'
    ] = median_diff_val
    current_step_features[f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag_diff}_diff'] = (
        national_diff_val
    )
    median_lag_diff[lag_diff] = median_diff_val
    national_avg_lag_diff[lag_diff] = national_diff_val

  # Longer-term rate of change
  current_step_features[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_vs_lag_3_diff'
  ] = median_lag.get(1, 0.0) - median_lag.get(3, 0.0)

  # EWMA diff and local vs national diffs
  median_ewma_diff_short_long_lag1 = (
      median_ewma_short_lag1 - median_ewma_long_lag1
  )
  current_step_features[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_diff_short_long_lag_1'
  ] = median_ewma_diff_short_long_lag1

  # NEW MINOR IMPROVEMENT: National EWMA diff short-long (momentum of national trend)
  national_ewma_diff_short_long_lag1 = (
      national_ewma_short_lag1 - national_ewma_long_lag1
  )
  current_step_features[f'national_ewma_diff_short_long_lag_1'] = (
      national_ewma_diff_short_long_lag1
  )

  local_vs_national_lag1_diff = median_lag.get(1, 0.0) - national_avg_lag.get(
      1, 0.0
  )
  current_step_features['local_vs_national_lag_1_diff'] = (
      local_vs_national_lag1_diff
  )

  local_vs_national_ewma_short_diff_lag1 = (
      median_ewma_short_lag1 - national_ewma_short_lag1
  )
  current_step_features['local_vs_national_ewma_short_diff_lag_1'] = (
      local_vs_national_ewma_short_diff_lag1
  )

  local_vs_national_ewma_long_diff_lag1 = (
      median_ewma_long_lag1 - national_ewma_long_lag1
  )
  current_step_features['local_vs_national_ewma_long_diff_lag_1'] = (
      local_vs_national_ewma_long_diff_lag1
  )

  # NEW MINOR IMPROVEMENT: Local vs National EWMA diff of short-long differences
  local_vs_national_ewma_diff_short_long_lag1 = (
      local_vs_national_ewma_short_diff_lag1
      - local_vs_national_ewma_long_diff_lag1
  )
  current_step_features['local_vs_national_ewma_diff_short_long_lag_1'] = (
      local_vs_national_ewma_diff_short_long_lag1
  )

  # --- Calculate NEW INTERACTION FEATURES incrementally ---
  current_step_features['population_x_log_target_lag1'] = (
      log_population_val * median_lag.get(1, 0.0)
  )
  current_step_features['population_x_log_target_ewma_short_lag1'] = (
      log_population_val * median_ewma_short_lag1
  )
  current_step_features['population_x_local_vs_national_lag1_diff'] = (
      log_population_val * local_vs_national_lag1_diff
  )

  current_step_features['log_population_x_week_sin'] = (
      log_population_val * week_sin_val
  )
  current_step_features['log_population_x_week_cos'] = (
      log_population_val * week_cos_val
  )
  current_step_features[
      f'log_population_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'
  ] = log_population_val * median_lag_diff.get(1, 0.0)
  current_step_features[
      f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_1'
  ] = log_population_val * national_avg_lag.get(1, 0.0)

  # New minor improvements:
  # 1. Acceleration (local target)
  local_acceleration_lag1 = median_lag_diff.get(1, 0.0) - median_lag_diff.get(
      2, 0.0
  )
  current_step_features[
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1'
  ] = local_acceleration_lag1

  # 2. Acceleration (national average)
  national_acceleration_lag1 = national_avg_lag_diff.get(
      1, 0.0
  ) - national_avg_lag_diff.get(2, 0.0)
  current_step_features[f'{MODEL_INPUT_NATIONAL_AVG}_acceleration_lag_1'] = (
      national_acceleration_lag1
  )

  # 3. Local deviation from short-term EWMA
  local_dev_from_short_ewma_lag1 = (
      median_lag.get(1, 0.0) - median_ewma_short_lag1
  )
  current_step_features['local_dev_from_short_ewma_lag_1'] = (
      local_dev_from_short_ewma_lag1
  )

  # 4. Interaction: log_population * local_acceleration
  current_step_features['population_x_local_acceleration_lag1'] = (
      log_population_val * local_acceleration_lag1
  )

  # 5. Interaction: log_population * national_average_lag_2
  current_step_features[
      f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_2'
  ] = log_population_val * national_avg_lag.get(2, 0.0)

  # 6. Interaction: national_ewma_short_x_week_sin (National trend seasonality interaction)
  current_step_features['national_ewma_short_x_week_sin_lag1'] = (
      national_ewma_short_lag1 * week_sin_val
  )

  # 7. Lagged IQR ratio (IQR/Median)
  median_lag1 = median_lag.get(
      1, 0.0
  )  # Ensure median_lag1 is explicitly retrieved
  log_target_iqr_to_median_ratio_lag1 = (
      iqr_lag1 / (median_lag1 + EPSILON)
      if (median_lag1 + EPSILON) != 0
      else 0.0
  )
  current_step_features['log_target_iqr_to_median_ratio_lag_1'] = (
      log_target_iqr_to_median_ratio_lag1
  )

  # NEW MINOR IMPROVEMENT: Horizon interaction features
  current_step_features['horizon_x_log_population'] = (
      horizon_val * log_population_val
  )
  current_step_features['horizon_x_week_sin'] = horizon_val * week_sin_val
  current_step_features['horizon_x_week_cos'] = horizon_val * week_cos_val

  # NEW MINOR IMPROVEMENT: Additional Horizon interaction features
  current_step_features[
      f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'
  ] = horizon_val * median_lag_diff.get(1, 0.0)
  current_step_features[
      f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1'
  ] = (horizon_val * local_acceleration_lag1)

  # NEW MINOR IMPROVEMENT: Population x national EWMA momentum
  current_step_features['population_x_national_ewma_diff_short_long_lag_1'] = (
      log_population_val * national_ewma_diff_short_long_lag1
  )

  # NEW MINOR IMPROVEMENT: Population x national short EWMA
  current_step_features['log_population_x_national_ewma_short_lag1'] = (
      log_population_val * national_ewma_short_lag1
  )

  # NEW MINOR IMPROVEMENT: IQR x Acceleration (proxy for uncertainty dynamics)
  current_step_features[f'iqr_lag_1_x_acceleration_lag_1'] = (
      iqr_lag1 * local_acceleration_lag1
  )

  # NEW MINOR IMPROVEMENT: Smoothed relative IQR (proxy for evolving contact matrices/transmissibility)
  current_step_features[f'log_target_iqr_to_median_ratio_ewma_short_lag_1'] = (
      iqr_ratio_ewma_short_lag1
  )

  # NEW MINOR IMPROVEMENT: Horizon x local deviation from short EWMA
  current_step_features['horizon_x_local_dev_from_short_ewma_lag_1'] = (
      horizon_val * local_dev_from_short_ewma_lag1
  )

  # NEW MINOR IMPROVEMENT: Horizon x relative IQR
  current_step_features['horizon_x_log_target_iqr_to_median_ratio_lag_1'] = (
      horizon_val * log_target_iqr_to_median_ratio_lag1
  )
  # --- END NEW INTERACTION FEATURES ---

  return current_step_features


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Make probabilistic quantile predictions for test_x using a stochastic,

  autoregressive LightGBM model that approximates metapopulation dynamics.
  This model generates multiple simulations per forecast horizon by adding
  bootstrapped residuals to point predictions, directly addressing the
  'stochastic simulation' principle. Lagged features mimic compartmental
  dynamics, and national trends proxy metapopulation dynamics. Enhanced
  with interaction features for improved proxy of age-structured dynamics.

  Args:
      train_x: DataFrame of historical features.
      train_y: Series of historical target values.
      test_x: DataFrame of future features for prediction.

  Returns:
      pd.DataFrame: A DataFrame containing quantile predictions for test_x.
  """
  warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

  # 1. Initial Data Preparation and Merging
  train_df_with_target = train_x.copy()
  train_df_with_target[TARGET_STR] = train_y

  # Ensure 'target_end_date' is datetime
  train_df_with_target['target_end_date'] = pd.to_datetime(
      train_df_with_target['target_end_date']
  )
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  # NEW IMPROVEMENT: Add horizon=0 to training data, so 'horizon' is a complete feature
  train_df_with_target['horizon'] = 0

  # Store the original test_x index for the output DataFrame
  original_test_x_index = test_x.index

  # Combine historical and future data into a single DataFrame for consistent processing
  recursive_full_data = (
      pd.concat([
          train_df_with_target[[
              'target_end_date',
              'location',
              'location_name',
              'population',
              'horizon',
              TARGET_STR,
          ]],
          test_x[[
              'target_end_date',
              'location',
              'location_name',
              'population',
              'horizon',
          ]].assign(
              **{TARGET_STR: np.nan}
          ),  # Add dummy target for test
      ])
      .sort_values(by=['location', 'target_end_date'])
      .reset_index(drop=True)
  )

  # Calculate target per 100k population for historical data
  recursive_full_data[TARGET_PER_POP_STR] = (
      recursive_full_data[TARGET_STR]
      / recursive_full_data['population']
      * 100000
  )
  recursive_full_data.loc[
      recursive_full_data['population'] == 0, TARGET_PER_POP_STR
  ] = 0.0
  recursive_full_data.loc[
      recursive_full_data['population'].isna(), TARGET_PER_POP_STR
  ] = 0.0  # Handle NaN populations if any

  # Apply log1p transformation to the target variable for training
  recursive_full_data[LOG_TARGET_PER_POP_STR] = np.log1p(
      recursive_full_data[TARGET_PER_POP_STR]
  )

  # Calculate `min_date_overall` once for the entire dataset for `time_idx`
  min_date_overall = recursive_full_data['target_end_date'].min()
  recursive_full_data = create_static_features(
      recursive_full_data, min_date_overall
  )

  # Initialize 'current state' columns which will be used for creating lagged features.
  train_end_date = train_df_with_target['target_end_date'].max()
  historical_mask = recursive_full_data['target_end_date'] <= train_end_date

  # Use actual historical values for initial median, Q025, Q075 inputs
  recursive_full_data[MODEL_INPUT_LOG_TARGET_MEDIAN] = np.nan
  recursive_full_data.loc[historical_mask, MODEL_INPUT_LOG_TARGET_MEDIAN] = (
      recursive_full_data.loc[historical_mask, LOG_TARGET_PER_POP_STR]
  )

  # Initialize Q025 and Q075 for historical data
  recursive_full_data[MODEL_INPUT_LOG_TARGET_Q025] = np.nan
  recursive_full_data.loc[historical_mask, MODEL_INPUT_LOG_TARGET_Q025] = (
      recursive_full_data.loc[historical_mask, LOG_TARGET_PER_POP_STR]
  )
  recursive_full_data[MODEL_INPUT_LOG_TARGET_Q075] = np.nan
  recursive_full_data.loc[historical_mask, MODEL_INPUT_LOG_TARGET_Q075] = (
      recursive_full_data.loc[historical_mask, LOG_TARGET_PER_POP_STR]
  )

  recursive_full_data[MODEL_INPUT_NATIONAL_AVG] = np.nan
  national_historical_stats = (
      recursive_full_data[historical_mask]
      .groupby('target_end_date')
      .agg(
          national_admissions_sum=(TARGET_STR, 'sum'),
          national_population_sum=('population', 'sum'),
      )
  )
  national_historical_stats[MODEL_INPUT_NATIONAL_AVG] = (
      national_historical_stats.apply(
          lambda row: (
              row['national_admissions_sum']
              / row['national_population_sum']
              * 100000
          )
          if row['national_population_sum'] > 0
          else 0,
          axis=1,
      )
  )
  # Map national average back to the recursive_full_data for historical entries
  recursive_full_data.loc[historical_mask, MODEL_INPUT_NATIONAL_AVG] = (
      recursive_full_data.loc[historical_mask, 'target_end_date'].map(
          national_historical_stats[MODEL_INPUT_NATIONAL_AVG]
      )
  )

  # 2. Pre-calculate historical features (lags and EWMA) for training and initial state
  # This will populate all MODEL_INPUT_*_lag_X and CURRENT_EWMA_* columns for historical data
  train_data_full_features = _compute_dynamic_features_for_training(
      recursive_full_data[historical_mask].copy()
  )

  # Define feature columns to be used in the model
  # Static features
  static_feature_cols = [
      'location',
      'log_population',
      'week_of_year',
      'week_sin',
      'week_cos',
      'time_idx',
      'horizon',  # Added horizon
  ]
  feature_cols = list(
      static_feature_cols
  )  # Start feature_cols with static features

  # Dynamically generated lagged/EWMA features
  for lag in LAGS:
    feature_cols.extend([
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag}',
        f'{MODEL_INPUT_LOG_TARGET_Q025}_lag_{lag}',
        f'{MODEL_INPUT_LOG_TARGET_Q075}_lag_{lag}',
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_iqr_lag_{lag}',
    ])
    feature_cols.extend([f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag}'])

  for lag_diff in DIFF_LAGS:
    feature_cols.extend([
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_{lag_diff}_diff',
        f'{MODEL_INPUT_NATIONAL_AVG}_lag_{lag_diff}_diff',
    ])
  feature_cols.append(f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_vs_lag_3_diff')

  for span in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]:
    feature_cols.extend([
        f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_{span}_lag_1',
        f'{MODEL_INPUT_NATIONAL_AVG}_ewma_{span}_lag_1',
    ])
  feature_cols.append(
      f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_ewma_diff_short_long_lag_1'
  )

  # NEW IMPROVEMENT: Add national EWMA diff short-long
  feature_cols.append(f'national_ewma_diff_short_long_lag_1')

  feature_cols.append('local_vs_national_lag_1_diff')
  feature_cols.append('local_vs_national_ewma_short_diff_lag_1')
  feature_cols.append('local_vs_national_ewma_long_diff_lag_1')

  # NEW IMPROVEMENT: Add local_vs_national_ewma_diff_short_long_lag_1
  feature_cols.append('local_vs_national_ewma_diff_short_long_lag_1')

  # --- Add NEW INTERACTION FEATURES to feature_cols ---
  feature_cols.append('population_x_log_target_lag1')
  feature_cols.append('population_x_log_target_ewma_short_lag1')
  feature_cols.append('population_x_local_vs_national_lag1_diff')
  # Add newly defined interaction features
  feature_cols.append('log_population_x_week_sin')
  feature_cols.append('log_population_x_week_cos')
  feature_cols.append(
      f'log_population_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'
  )
  feature_cols.append(f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_1')

  feature_cols.append(f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1')
  feature_cols.append(f'{MODEL_INPUT_NATIONAL_AVG}_acceleration_lag_1')
  feature_cols.append('local_dev_from_short_ewma_lag_1')
  feature_cols.append('population_x_local_acceleration_lag1')
  feature_cols.append(f'log_population_x_{MODEL_INPUT_NATIONAL_AVG}_lag_2')
  feature_cols.append('national_ewma_short_x_week_sin_lag1')
  feature_cols.append('log_target_iqr_to_median_ratio_lag_1')

  # NEW: Add horizon interaction features to feature_cols
  feature_cols.append('horizon_x_log_population')
  feature_cols.append('horizon_x_week_sin')
  feature_cols.append('horizon_x_week_cos')
  feature_cols.append(
      f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1_diff'
  )  # Added new interaction feature
  feature_cols.append(
      f'horizon_x_{MODEL_INPUT_LOG_TARGET_MEDIAN}_acceleration_lag_1'
  )  # Added new interaction feature

  # NEW IMPROVEMENT: Population x national EWMA momentum
  feature_cols.append('population_x_national_ewma_diff_short_long_lag_1')

  # NEW IMPROVEMENT: Population x national short EWMA
  feature_cols.append('log_population_x_national_ewma_short_lag1')

  # NEW IMPROVEMENT: IQR x Acceleration
  feature_cols.append(f'iqr_lag_1_x_acceleration_lag_1')

  # NEW IMPROVEMENT: Smoothed relative IQR
  feature_cols.append(f'log_target_iqr_to_median_ratio_ewma_short_lag_1')

  # NEW IMPROVEMENT: Horizon x local deviation from short EWMA
  feature_cols.append('horizon_x_local_dev_from_short_ewma_lag_1')

  # NEW IMPROVEMENT: Horizon x relative IQR
  feature_cols.append('horizon_x_log_target_iqr_to_median_ratio_lag_1')

  # --- END NEW INTERACTION FEATURES ---

  # Filter to ensure only existing columns are used (e.g., if history is too short for some lags)
  feature_cols = [
      col for col in feature_cols if col in train_data_full_features.columns
  ]

  # Training data for models
  X_train = train_data_full_features.dropna(
      subset=[MODEL_INPUT_LOG_TARGET_MEDIAN]
      + [
          f'{MODEL_INPUT_LOG_TARGET_MEDIAN}_lag_1',
          f'{MODEL_INPUT_NATIONAL_AVG}_lag_1',
      ],
      how='any',
  )[feature_cols]
  y_train = train_data_full_features.loc[
      X_train.index, MODEL_INPUT_LOG_TARGET_MEDIAN
  ]

  categorical_features = ['location', 'week_of_year']
  # 'horizon' can also be treated as categorical or numeric, for now it remains numeric
  categorical_features = [
      col for col in categorical_features if col in X_train.columns
  ]

  # 3. Model Training
  lgb_common_params = {
      'metric': 'mae',
      'n_estimators': 700,
      'learning_rate': 0.01,
      'num_leaves': 31,
      'max_depth': -1,
      'min_child_samples': 20,
      'subsample': 0.7,
      'colsample_bytree': 0.7,
      'random_state': 42,
      'n_jobs': -1,
      'verbose': -1,
      'boosting_type': 'gbdt',
  }

  median_model = lgb.LGBMRegressor(
      objective='regression_l1', **lgb_common_params
  )
  median_model.fit(X_train, y_train, categorical_feature=categorical_features)

  q025_model = lgb.LGBMRegressor(
      objective='quantile', alpha=0.25, **lgb_common_params
  )
  q025_model.fit(X_train, y_train, categorical_feature=categorical_features)

  q075_model = lgb.LGBMRegressor(
      objective='quantile', alpha=0.75, **lgb_common_params
  )
  q075_model.fit(X_train, y_train, categorical_feature=categorical_features)

  # Calculate residuals on the training set for stochastic simulation (from median model)
  train_preds_median = median_model.predict(X_train)
  train_residuals = y_train - train_preds_median

  # Conditional Residual Sampling for Stochastic Simulation
  global_residuals = train_residuals.dropna().values

  low_bound, high_bound = np.nan, np.nan
  residuals_low, residuals_med, residuals_high = (
      global_residuals,
      global_residuals,
      global_residuals,
  )

  if len(train_preds_median) >= 3 and len(global_residuals) > 0:
    # Using terciles for conditional sampling (conditioned on predicted median for consistency in prediction)
    low_bound = np.percentile(train_preds_median, 33)
    high_bound = np.percentile(train_preds_median, 67)

    residuals_low_candidates = (
        train_residuals[train_preds_median <= low_bound].dropna().values
    )
    residuals_med_candidates = (
        train_residuals[
            (train_preds_median > low_bound)
            & (train_preds_median <= high_bound)
        ]
        .dropna()
        .values
    )
    residuals_high_candidates = (
        train_residuals[train_preds_median > high_bound].dropna().values
    )

    if len(residuals_low_candidates) > 0:
      residuals_low = residuals_low_candidates
    if len(residuals_med_candidates) > 0:
      residuals_med = residuals_med_candidates
    if len(residuals_high_candidates) > 0:
      residuals_high = residuals_high_candidates
  else:
    warnings.warn(
        'Not enough training predictions or residuals to create conditional'
        ' residual bins. Using global residuals for all bins.'
    )

  def get_residuals_for_prediction(pred_value):
    if not np.isnan(low_bound) and pred_value <= low_bound:
      return residuals_low
    elif (
        not np.isnan(low_bound)
        and pred_value > low_bound
        and pred_value <= high_bound
    ):
      return residuals_med
    else:  # Default to high or global if bounds are nan or pred_value > high_bound
      return residuals_high

  # 4. Initialize location history for recursive prediction
  # Store relevant historical data for each location in a deque for efficient access
  # This deque will be updated with predictions in each forecasting step.
  max_lag_required = max(
      LAGS + [l + 1 for l in DIFF_LAGS] + [1]
  )  # Max lag for features + 1 for EWMA_lag_1, and other diffs
  max_history_buffer_len = max(
      max_lag_required + 1, EWMA_SPAN_LONG + 1
  )  # Keep enough for all lags and EWMA history + current week

  location_history_buffers = {}
  history_cols_to_store = (
      [
          'target_end_date',
          MODEL_INPUT_LOG_TARGET_MEDIAN,
          MODEL_INPUT_LOG_TARGET_Q025,
          MODEL_INPUT_LOG_TARGET_Q075,
          MODEL_INPUT_NATIONAL_AVG,
          'log_population',
          'week_sin',
          'week_cos',
          'horizon',  # Added horizon
          (  # Store this for incremental EWMA calculation later
              'log_target_iqr_to_median_ratio_lag_1'
          ),
      ]
      + [
          f'{CURRENT_EWMA_MEDIAN_PREFIX}_{s}'
          for s in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]
      ]
      + [
          f'{CURRENT_EWMA_NATIONAL_PREFIX}_{s}'
          for s in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]
      ]
      + [f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}']
  )  # For incremental relative IQR EWMA

  # Fill initial history buffers from pre-calculated train_data_full_features
  # We slice by location to get the correct EWMA states initialized per location
  for loc_code in train_data_full_features['location'].unique():
    loc_hist_df = train_data_full_features[
        train_data_full_features['location'] == loc_code
    ].sort_values('target_end_date')

    # Ensure all history_cols_to_store are present, filling missing with NaN (especially for early EWMAs)
    for col in history_cols_to_store:
      if col not in loc_hist_df.columns:
        loc_hist_df[col] = np.nan

    # Convert to list of dicts for deque
    location_history_buffers[loc_code] = deque(
        loc_hist_df[history_cols_to_store].to_dict('records'),
        maxlen=max_history_buffer_len,
    )

  # 5. Recursive Prediction Generation for Test Data
  future_dates = sorted(test_x['target_end_date'].unique())
  predictions_list = []

  # Pre-compute population for test locations to use in scaling
  test_populations = test_x.set_index(['location', 'target_end_date'])[
      'population'
  ]

  for current_forecast_date in future_dates:
    current_week_test_x_slice = test_x[
        test_x['target_end_date'] == current_forecast_date
    ].copy()
    current_week_test_x_slice = create_static_features(
        current_week_test_x_slice, min_date_overall
    )
    current_week_test_x_slice.set_index(
        original_test_x_index.intersection(current_week_test_x_slice.index),
        inplace=True,
    )  # Reindex to match original test_x index

    if current_week_test_x_slice.empty:
      continue

    rows_for_prediction_data = (
        []
    )  # To build a new DataFrame for prediction, containing all features
    for original_idx, row_data in current_week_test_x_slice.iterrows():
      loc = row_data['location']
      loc_hist_deque = location_history_buffers.get(
          loc, deque([], maxlen=max_history_buffer_len)
      )

      # Use the new helper function to extract and compute features for the current step
      current_row_features = _extract_and_compute_features_for_step(
          row_data, loc_hist_deque
      )  # Pass full row_data to get static features
      current_row_features['original_index'] = (
          original_idx  # Preserve original index
      )
      rows_for_prediction_data.append(current_row_features)

    # Convert list of dicts to DataFrame for batch prediction
    X_predict_features_df = pd.DataFrame(rows_for_prediction_data).set_index(
        'original_index'
    )
    # Ensure correct columns are present and fill any remaining NaNs (e.g., from initial history, though get_hist_val uses 0.0)
    X_predict_features_df = X_predict_features_df[feature_cols].fillna(0.0)

    # Make predictions from the three models (in log('per 100k') scale)
    median_preds_log_per_100k = median_model.predict(X_predict_features_df)
    q025_preds_log_per_100k = q025_model.predict(X_predict_features_df)
    q075_preds_log_per_100k = q075_model.predict(X_predict_features_df)

    # Ensure base predictions are monotonic for consistency (Principle 2)
    # Simplified monotonicity enforcement using np.clip
    q025_preds_log_per_100k = np.minimum(
        q025_preds_log_per_100k, median_preds_log_per_100k
    )
    q075_preds_log_per_100k = np.maximum(
        q075_preds_log_per_100k, median_preds_log_per_100k
    )
    median_preds_log_per_100k = np.clip(
        median_preds_log_per_100k,
        q025_preds_log_per_100k,
        q075_preds_log_per_100k,
    )

    # Stochastic Simulation: Generate N_SIMULATIONS trajectories around the median prediction (Principle 3 & 4)
    simulated_log_per_100k_values = np.zeros(
        (len(median_preds_log_per_100k), N_SIMULATIONS)
    )

    if len(global_residuals) > 0:
      for i in range(len(median_preds_log_per_100k)):
        current_residuals_pool = get_residuals_for_prediction(
            median_preds_log_per_100k[i]
        )
        if len(current_residuals_pool) > 0:
          sampled_residuals = np.random.choice(
              current_residuals_pool, size=N_SIMULATIONS, replace=True
          )
          simulated_log_per_100k_values[i, :] = (
              median_preds_log_per_100k[i] + sampled_residuals
          )
        else:  # Fallback if a specific bin is empty
          sampled_residuals = np.random.choice(
              global_residuals, size=N_SIMULATIONS, replace=True
          )
          simulated_log_per_100k_values[i, :] = (
              median_preds_log_per_100k[i] + sampled_residuals
          )
    else:
      simulated_log_per_100k_values = median_preds_log_per_100k[
          :, np.newaxis
      ] * np.ones((1, N_SIMULATIONS))

    # --- START MINOR IMPROVEMENT: Adjust spread of simulations based on learned Q025/Q075 ---
    for i in range(len(median_preds_log_per_100k)):
      sims = simulated_log_per_100k_values[i, :]

      # Predicted IQR from quantile models
      predicted_iqr = q075_preds_log_per_100k[i] - q025_preds_log_per_100k[i]

      # IQR from current simulations
      simulated_q25 = np.percentile(sims, 25)
      simulated_q75 = np.percentile(sims, 75)
      simulated_iqr = simulated_q75 - simulated_q25

      # Only adjust if the predicted IQR is wider than the simulated IQR
      # This ensures we respect the quantile models' learned spread
      if (
          predicted_iqr > simulated_iqr + EPSILON
      ):  # Add epsilon for numerical stability
        sims_median = np.median(sims)
        sims_centered = sims - sims_median

        # Calculate scaling factor to stretch the simulations
        # Ensure simulated_iqr is not zero to avoid division by zero
        scale_factor = predicted_iqr / (simulated_iqr + EPSILON)

        sims_scaled = sims_centered * scale_factor
        # Re-center scaled simulations around the median prediction
        simulated_log_per_100k_values[i, :] = (
            sims_scaled + median_preds_log_per_100k[i]
        )
    # --- END MINOR IMPROVEMENT ---

    # Inverse transform simulated values and scale to absolute counts
    simulated_per_100k_values = np.expm1(simulated_log_per_100k_values)
    simulated_per_100k_values = np.maximum(0, simulated_per_100k_values)

    # Retrieve populations for scaling, matching the order of predictions
    # Use current_week_test_x_slice's index to get populations correctly
    populations_for_current_week = current_week_test_x_slice.loc[
        X_predict_features_df.index, 'population'
    ].values
    scale_factor = np.divide(
        populations_for_current_week,
        100000,
        out=np.zeros_like(populations_for_current_week, dtype=float),
        where=(populations_for_current_week != 0)
        & (~pd.isna(populations_for_current_week)),
    )
    simulated_absolute_values = (
        simulated_per_100k_values * scale_factor[:, np.newaxis]
    )
    simulated_absolute_values = np.round(
        np.maximum(0, simulated_absolute_values)
    ).astype(int)

    # Aggregate simulated_absolute_values by location and compute quantiles for output
    current_date_predictions_output = pd.DataFrame(
        index=X_predict_features_df.index,  # Use the index from the prediction DataFrame
        columns=[f'quantile_{q}' for q in QUANTILES],
    )

    # Calculate and update national average for the current week *before* EWMA updates
    # Use median of the N_SIMULATIONS for each location, then sum to national level
    predicted_national_admissions_sum = np.sum(
        np.median(simulated_absolute_values, axis=1)
    )
    predicted_national_population_sum = current_week_test_x_slice[
        'population'
    ].sum()
    current_national_avg_per_100k = (
        (
            predicted_national_admissions_sum
            / predicted_national_population_sum
            * 100000
        )
        if predicted_national_population_sum > 0
        else 0
    )

    # Update location_history_buffers with current week's predictions and new EWMA states
    for i, original_idx in enumerate(X_predict_features_df.index):
      loc = current_week_test_x_slice.loc[original_idx, 'location']
      loc_hist_deque = location_history_buffers.get(
          loc, deque([], maxlen=max_history_buffer_len)
      )

      # Create a new historical entry for the current week based on predictions
      new_hist_entry = {
          'target_end_date': current_forecast_date,
          MODEL_INPUT_LOG_TARGET_MEDIAN: median_preds_log_per_100k[i],
          MODEL_INPUT_LOG_TARGET_Q025: q025_preds_log_per_100k[
              i
          ],  # Use predicted Q025
          MODEL_INPUT_LOG_TARGET_Q075: q075_preds_log_per_100k[
              i
          ],  # Use predicted Q075
          MODEL_INPUT_NATIONAL_AVG: current_national_avg_per_100k,
          'log_population': current_week_test_x_slice.loc[
              original_idx, 'log_population'
          ],
          'week_sin': current_week_test_x_slice.loc[original_idx, 'week_sin'],
          'week_cos': current_week_test_x_slice.loc[original_idx, 'week_cos'],
          'horizon': current_week_test_x_slice.loc[
              original_idx, 'horizon'
          ],  # Store horizon for next step
      }
      # Also calculate and store the lagged IQR ratio for the current step (based on current predictions)
      iqr_current_pred = np.maximum(
          0,
          new_hist_entry[MODEL_INPUT_LOG_TARGET_Q075]
          - new_hist_entry[MODEL_INPUT_LOG_TARGET_Q025],
      )
      median_current_pred = new_hist_entry[MODEL_INPUT_LOG_TARGET_MEDIAN]
      new_hist_entry['log_target_iqr_to_median_ratio_lag_1'] = (
          iqr_current_pred / (median_current_pred + EPSILON)
          if (median_current_pred + EPSILON) != 0
          else 0.0
      )

      # Calculate and store new EWMA states for this location using `_calculate_incremental_ewma`
      last_hist_entry = loc_hist_deque[-1] if loc_hist_deque else {}
      for span in [EWMA_SPAN_SHORT, EWMA_SPAN_LONG]:
        # Local Median EWMA
        prev_ewma_median = last_hist_entry.get(
            f'{CURRENT_EWMA_MEDIAN_PREFIX}_{span}', np.nan
        )
        new_hist_entry[f'{CURRENT_EWMA_MEDIAN_PREFIX}_{span}'] = (
            _calculate_incremental_ewma(
                median_preds_log_per_100k[i], prev_ewma_median, span
            )
        )
        # National Average EWMA
        prev_national_ewma = last_hist_entry.get(
            f'{CURRENT_EWMA_NATIONAL_PREFIX}_{span}', np.nan
        )
        new_hist_entry[f'{CURRENT_EWMA_NATIONAL_PREFIX}_{span}'] = (
            _calculate_incremental_ewma(
                current_national_avg_per_100k, prev_national_ewma, span
            )
        )

      # NEW IMPROVEMENT: Incremental EWMA for relative IQR
      prev_ewma_iqr_ratio = last_hist_entry.get(
          f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}',
          np.nan,
      )
      new_hist_entry[
          f'{CURRENT_EWMA_MEDIAN_PREFIX}_iqr_ratio_span{EWMA_SPAN_SHORT}'
      ] = _calculate_incremental_ewma(
          new_hist_entry['log_target_iqr_to_median_ratio_lag_1'],
          prev_ewma_iqr_ratio,
          EWMA_SPAN_SHORT,
      )

      loc_hist_deque.append(new_hist_entry)
      location_history_buffers[loc] = (
          loc_hist_deque  # Ensure the updated deque is stored back
      )

      # Compute quantiles for final output
      for q_val in QUANTILES:
        current_date_predictions_output.loc[
            original_idx, f'quantile_{q_val}'
        ] = np.quantile(simulated_absolute_values[i, :], q_val)

    predictions_list.append(current_date_predictions_output)

  if not predictions_list:
    return pd.DataFrame(
        index=original_test_x_index,
        columns=[f'quantile_{q}' for q in QUANTILES],
    )

  final_predictions_df = pd.concat(predictions_list).sort_index()

  # 6. Post-processing: Ensure monotonicity and non-negativity (Principle 4: Aggregation)
  quantile_cols = [f'quantile_{q}' for q in QUANTILES]

  final_predictions_df[quantile_cols] = np.maximum(
      0, final_predictions_df[quantile_cols]
  )
  final_predictions_df[quantile_cols] = final_predictions_df[
      quantile_cols
  ].apply(
      lambda x: pd.Series(np.maximum.accumulate(x.values), index=x.index),
      axis=1,
  )

  final_predictions_df = final_predictions_df.reindex(original_test_x_index)[
      quantile_cols
  ]

  return final_predictions_df


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
    cdc_submission = format_for_cdc(test_forecasts, 'wk inc covid hosp')
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
