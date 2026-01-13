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
MODEL_NAME = 'Google_SAI-Novel_2'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core Principles of the Monte Carlo Forecasting Method:
# 1. Monte Carlo Simulation for Uncertainty: The model quantifies unconditional uncertainty
#    by executing N Monte Carlo simulation runs, generating diverse future outcome trajectories.
# 2. Scenario-Based Driver Trajectory Sampling: In each simulation run, key uncertain future
#    drivers (e.g., new variant emergence, mobility changes, seasonal fluctuations) are
#    probabilistically sampled, evolving over the forecast horizon to create a unique,
#    plausible future scenario. These trajectories are now generated using vectorized random walks
#    for improved efficiency.
# 3. Pillar 1 Mock Model Conditioning: A pre-trained base forecasting model (LightGBM
#    Regressor serving as a mock for "Pillar 1") generates a single median forecast. This median
#    forecast is then modulated by the sampled scenario driver trajectories and further
#    perturbed by stochastic noise (including overdispersion) to produce a scenario-conditioned outcome.
# 4. Empirical Quantile Aggregation: Final quantile forecasts (e.g., 0.01 to 0.99) are
#    derived as the empirical quantiles calculated across the entire collection of N
#    simulated outcome trajectories for each specific forecast date.

import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import deque
from typing import Any, Tuple
import warnings  # Added for improved error handling

# Use the globally defined QUANTILES list, no need to redefine here.


def _engineer_features(
    df,
    max_lags,
    pop_normalization_factor,
    is_train = True,
    target_series = None,
):
  """Engineers common features for COVID-19 hospitalization data."""
  df_processed = df.copy()
  df_processed['target_end_date'] = pd.to_datetime(
      df_processed['target_end_date']
  )
  df_processed['location'] = (
      pd.to_numeric(df_processed['location'], errors='coerce')
      .fillna(-1)
      .astype(int)
  )

  # Population handling
  df_processed['population'] = pd.to_numeric(
      df_processed['population'], errors='coerce'
  )
  median_population = (
      df_processed['population'].median()
      if not df_processed['population'].empty
      else pop_normalization_factor
  )
  df_processed['population'] = df_processed['population'].fillna(
      median_population
  )
  df_processed['population'] = np.maximum(
      1.0, df_processed['population']
  )  # Ensure population is always at least 1.0 for division

  # Target variable processing for training
  if is_train:
    df_processed['Total COVID-19 Admissions'] = target_series.copy()
    df_processed['normalized_admissions'] = df_processed[
        'Total COVID-19 Admissions'
    ] / (df_processed['population'] / pop_normalization_factor)
    df_processed['normalized_admissions'] = np.maximum(
        0.0, df_processed['normalized_admissions'].fillna(0)
    )
    df_processed = df_processed.sort_values(
        by=['location', 'target_end_date']
    ).reset_index(drop=True)

    # FIX 2: Add horizon feature for training data, set to 0.
    # This allows the LGBM model to be trained with 'horizon' as a feature,
    # which is present in test_x.
    df_processed['horizon'] = 0

    # Generate lagged features for normalized admissions for training data
    for i in range(1, max_lags + 1):
      df_processed[f'normalized_admissions_lag_{i}'] = df_processed.groupby(
          'location', observed=False
      )['normalized_admissions'].shift(i)
      # FIX 1: Fill NaNs with np.nan instead of 0.0 to avoid overwriting true zero values.
      # These NaNs will be filled with global mean later in fit_and_predict_fn.
      df_processed[f'normalized_admissions_lag_{i}'] = df_processed[
          f'normalized_admissions_lag_{i}'
      ].fillna(np.nan)
  else:
    # For test data, ensure horizon is integer. This horizon feature is kept in test_x.
    df_processed['horizon'] = df_processed['horizon'].astype(int)

  # Extract temporal features
  df_processed['year'] = (
      df_processed['target_end_date'].dt.isocalendar().year.astype(int)
  )
  df_processed['week'] = (
      df_processed['target_end_date'].dt.isocalendar().week.astype(int)
  )
  df_processed['day_of_year'] = df_processed['target_end_date'].dt.dayofyear
  df_processed['month'] = df_processed['target_end_date'].dt.month
  df_processed['week_in_month'] = (
      df_processed['target_end_date'].dt.day - 1
  ) // 7 + 1

  # Add sine/cosine transforms for cyclical features (e.g., seasonality)
  df_processed['week_sin'] = np.sin(2 * np.pi * df_processed['week'] / 52.1775)
  df_processed['week_cos'] = np.cos(2 * np.pi * df_processed['week'] / 52.1775)
  df_processed['dayofyear_sin'] = np.sin(
      2 * np.pi * df_processed['day_of_year'] / 365.25
  )
  df_processed['dayofyear_cos'] = np.cos(
      2 * np.pi * df_processed['day_of_year'] / 365.25
  )

  return df_processed


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Implements an unconditional uncertainty quantification module using Monte Carlo simulation

  for COVID-19 hospital admissions forecasting.

  This function operates as follows:
  1.  **Pillar 1 Mock Model Training**: A LightGBM Regressor is used as a mock
  for the
      "Causal Spatio-Temporal Graph Transformer (Pillar 1)". This model is
      trained
      on historical data to provide a baseline median forecast. It incorporates
      time-based features, location information, population, and lagged
      admissions
      to capture spatio-temporal dynamics. The 'regression_l1' (MAE) objective
      is used to target the median, aligning with the "median forecast
      trajectory"
      description.
  2.  **Simulation Loop**: Executes N Monte Carlo simulation runs.
      In each run, a complete, coherent future trajectory for all defined
      drivers is sampled:
      a.  For "New Variant Emergence", an initial impact deviation is sampled,
      and its
          impact evolves as a random walk across the forecast horizon, creating
          a time-varying
          trajectory of variant impact. The volatility of this random walk can
          depend on
          the variant type (e.g., more severe variants have more volatile
          impact).
          The random walk is generated using a vectorized approach for
          efficiency.
      b.  For "Holiday Mobility Change" (interpreted as general mobility
      shifts),
          an initial shock is sampled, and its impact evolves as a random walk
          across the forecast horizon, creating a time-varying trajectory.
          The random walk is generated using a vectorized approach for
          efficiency.
      c.  For "Seasonal Fluctuations", an initial seasonal deviation is sampled,
      and its
          impact evolves as a random walk across the forecast horizon, creating
          a time-varying
          trajectory of seasonal uncertainty.
          The random walk is generated using a vectorized approach for
          efficiency.
      d.  The combined scenario trajectory is then applied as a multiplier to
      the
          base forecast generated by the LightGBM model.
      e.  Poisson noise is added, *preceded by a multiplicative Gaussian noise
      term*
          to account for overdispersion, to reflect the count nature of the data
          and
          further introduce stochasticity.
  3.  **Aggregation**: Collects the N resulting forecast trajectories.
  4.  **Quantile Calculation**: The final submitted forecast quantiles (0.01 to
  0.99)
      are the empirical quantiles calculated across this aggregated distribution
      of
      N trajectories for each forecast date. Predictions are ensured to be
      non-negative
      and monotonically increasing across quantiles, and rounded to integers.

  Args:
      train_x (pd.DataFrame): Training features, containing historical data.
      train_y (pd.Series): Training target values (Total COVID-19 Admissions).
      test_x (pd.DataFrame): Test features for future time periods for which
        predictions are required.

  Returns:
      pd.DataFrame: A DataFrame with quantile predictions for each row in
      test_x.
                    Columns are named 'quantile_0.XXX' (e.g., 'quantile_0.01').
                    The DataFrame's index must match test_x's index.
  """
  # --- Configuration and Hyperparameters ---
  config = {
      'n_simulations': (
          10000
      ),  # Increased simulations for better quantile stability
      'max_lags': (
          3
      ),  # Moderate lags for stability, preventing too much reliance on distant past
      'pop_normalization_factor': (
          100000.0
      ),  # Standardize admissions per 100,000 people
      'lgbm_params': {
          'objective': (
              'regression_l1'
          ),  # MAE objective, good for median and robustness to outliers
          'metric': 'mae',
          'n_estimators': (
              800
          ),  # Slightly increased n_estimators for better robustness and capacity
          'learning_rate': 0.015,  # Robust learning rate for more stability
          # MODIFICATION TO NUM_LEAVES: Increased for slightly higher model capacity.
          'num_leaves': 50,  # Adjusted from 45 to 50 for more model capacity
          'feature_fraction': 0.7,  # Feature subsampling rate
          'bagging_fraction': 0.7,  # Data subsampling rate
          'bagging_freq': 1,  # Perform bagging at every iteration
          'lambda_l1': 0.1,  # L1 regularization for sparsity
          'lambda_l2': 0.1,  # L2 regularization for preventing large weights
          'verbose': -1,  # Suppress verbose output
          'n_jobs': -1,  # Use all available CPU cores
          'seed': 789,  # Seed for reproducibility of LGBM training
      },
      # MODIFICATION TO RANDOM_WALK_STEP_STD: Increased for more diverse scenarios
      'mobility_dist_params': {
          'loc': 0.0,
          'scale': 0.05,
          'random_walk_step_std': 0.01,
      },  # Adjusted from 0.008 to 0.01
      'mobility_clip': {'min': 0.7, 'max': 1.3},
      # MODIFICATION TO RANDOM_WALK_STEP_STD for each variant type
      'variant_scenarios': {
          'None': {
              'prob': 0.6,
              'initial_deviation_mean': 0.0,
              'initial_deviation_std': 0.01,
              'random_walk_step_std': 0.007,
          },  # Adjusted from 0.005 to 0.007
          'Mild': {
              'prob': 0.3,
              'initial_deviation_mean': 0.15,
              'initial_deviation_std': 0.04,
              'random_walk_step_std': 0.015,
          },  # Adjusted from 0.012 to 0.015
          'Severe': {
              'prob': 0.1,
              'initial_deviation_mean': 0.4,
              'initial_deviation_std': 0.08,
              'random_walk_step_std': 0.03,
          },  # Adjusted from 0.025 to 0.03
      },
      'variant_impact_clip': {
          'min': 0.5,
          'max': 2.5,
      },  # Clipping for overall variant multiplier
      'seasonality_uncertainty_scale': (
          0.05
      ),  # Std dev for initial seasonal deviation
      # MODIFICATION TO RANDOM_WALK_STEP_STD: Increased for more diverse scenarios
      'seasonal_random_walk_step_std': 0.01,  # Adjusted from 0.008 to 0.01
      'seasonality_noise_clip': {
          'min': 0.5,
          'max': 1.5,
      },  # Clipping for overall seasonality multiplier
      'min_poisson_lam_for_zeros': (
          0.05
      ),  # FIX 7: Added for robustness: minimum lambda for Poisson noise when base prediction is zero.
      # This helps ensure some uncertainty even if the median forecast is 0.
      # MODIFICATION TO OVERDISPERSION_NOISE_CLIP: Widened for broader uncertainty
      'overdispersion_multiplicative_noise_std': (
          0.05
      ),  # Standard deviation for multiplicative noise to simulate overdispersion
      'overdispersion_noise_clip': {
          'min': 0.7,
          'max': 1.3,
      },  # Adjusted from 0.8/1.2 to 0.7/1.3 for wider overdispersion
      'mc_seed': 9101,  # Seed for Monte Carlo simulations
  }
  N_SIMULATIONS = config['n_simulations']
  MAX_LAGS = config['max_lags']
  POP_NORMALIZATION_FACTOR = config['pop_normalization_factor']
  lgbm_params = config['lgbm_params']
  MIN_POISSON_LAM_FOR_ZEROS = config[
      'min_poisson_lam_for_zeros'
  ]  # Retrieve the new config parameter
  OVERDISPERSION_MULT_NOISE_STD = config[
      'overdispersion_multiplicative_noise_std'
  ]  # New parameter
  OVERDISPERSION_NOISE_CLIP = config[
      'overdispersion_noise_clip'
  ]  # New parameter

  # --- Scenario Definition: Key Uncertain Future Drivers and their Distributions ---
  variant_scenarios = config['variant_scenarios']
  variant_choices = list(variant_scenarios.keys())
  variant_probs = [s['prob'] for s in variant_scenarios.values()]
  variant_impact_clip = config['variant_impact_clip']

  mobility_dist_params = config['mobility_dist_params']
  mobility_multiplier_clip = config['mobility_clip']

  seasonality_uncertainty_scale = config['seasonality_uncertainty_scale']
  seasonal_random_walk_step_std = config['seasonal_random_walk_step_std']
  seasonality_noise_clip = config['seasonality_noise_clip']

  # Random number generator for Monte Carlo simulations
  rng = np.random.default_rng(config['mc_seed'])

  # --- Preprocessing Data for "Pillar 1" Mock Model ---
  test_x_original_index = test_x.index

  # Engineer features for training data
  train_df_engineered = _engineer_features(
      train_x,
      MAX_LAGS,
      POP_NORMALIZATION_FACTOR,
      is_train=True,
      target_series=train_y,
  )

  # FIX 3: Calculate global_norm_admissions_mean AFTER filtering for valid normalized admissions
  # and ensure it's robustly positive.
  valid_normalized_admissions = train_df_engineered[
      'normalized_admissions'
  ].dropna()
  global_norm_admissions_mean = (
      valid_normalized_admissions.mean()
      if not valid_normalized_admissions.empty
      else 1.0
  )
  global_norm_admissions_mean = np.maximum(1e-6, global_norm_admissions_mean)

  # FIX 1: Fill NaNs in lag features with global mean, AFTER it's calculated
  for i in range(1, MAX_LAGS + 1):
    train_df_engineered[f'normalized_admissions_lag_{i}'] = train_df_engineered[
        f'normalized_admissions_lag_{i}'
    ].fillna(global_norm_admissions_mean)

  # Define features to be used by the LGBM model
  # MODIFICATION 1: Added 'horizon' to categorical features for proper interpretation
  categorical_cols_for_lgbm = ['location', 'horizon']
  features_for_lgbm = [
      'year',
      'week',
      'day_of_year',
      'month',
      'week_in_month',
      'week_sin',
      'week_cos',
      'dayofyear_sin',
      'dayofyear_cos',
      'population',
      'location',
      'horizon',
  ]
  lag_feature_names = [
      f'normalized_admissions_lag_{i}' for i in range(1, MAX_LAGS + 1)
  ]
  features_for_lgbm.extend(lag_feature_names)

  processed_train_x = train_df_engineered[features_for_lgbm]
  processed_train_y = np.log1p(train_df_engineered['normalized_admissions'])

  # Engineer features for test data (without lags, which will be filled iteratively during prediction)
  test_df_processed = _engineer_features(
      test_x, MAX_LAGS, POP_NORMALIZATION_FACTOR, is_train=False
  )

  # --- "Pillar 1" Mock Model Training ---
  model = None
  # Filter out rows with NaN in target or features after lag generation for training
  valid_train_indices = processed_train_x.dropna(subset=features_for_lgbm).index
  processed_train_x = processed_train_x.loc[valid_train_indices]
  processed_train_y = processed_train_y.loc[valid_train_indices]

  if (
      not processed_train_x.empty
      and not processed_train_y.empty
      and len(processed_train_x) > 0
      and processed_train_y.count() > 0
  ):
    try:
      model = lgb.LGBMRegressor(**lgbm_params)
      lgbm_cat_features_present = [
          f for f in categorical_cols_for_lgbm if f in processed_train_x.columns
      ]

      for col in lgbm_cat_features_present:
        processed_train_x[col] = processed_train_x[col].astype(
            'category'
        )  # Use 'category' dtype for LGBM

      model.fit(
          processed_train_x,
          processed_train_y,
          categorical_feature=lgbm_cat_features_present,
      )
    except Exception as e:
      warnings.warn(
          f'LGBM model training failed: {e}. Model will not be used for'
          ' predictions.'
      )
      model = None
  else:
    warnings.warn(
        'Insufficient or invalid data for LGBM model training. Skipping model'
        ' training.'
    )

  # --- Generate Base Predictions for Test Data (Iterative for Lags) ---
  last_known_admissions_by_loc = {}

  # Initialize deque for all training locations with their last known normalized values
  for loc_code in train_df_engineered['location'].unique():
    loc_train_data = train_df_engineered[
        train_df_engineered['location'] == loc_code
    ]
    history_values = (
        loc_train_data['normalized_admissions'].tail(MAX_LAGS).tolist()
    )

    padding_needed = MAX_LAGS - len(history_values)
    if padding_needed > 0:
      history_values = [
          global_norm_admissions_mean
      ] * padding_needed + history_values

    last_known_admissions_by_loc[loc_code] = deque(
        history_values, maxlen=MAX_LAGS
    )

  base_predictions_list = []
  base_predictions_indices = []

  # Sort test_df_processed by location and date to ensure correct iterative lag updates
  test_df_sorted = test_df_processed.sort_values(
      by=['location', 'target_end_date']
  ).copy()

  for original_idx, row in test_df_sorted.iterrows():
    current_loc_code = row['location']

    loc_history_deque = last_known_admissions_by_loc.get(current_loc_code)
    if loc_history_deque is None:
      # If location not seen in training, initialize with global mean
      loc_history_deque = deque(
          [global_norm_admissions_mean] * MAX_LAGS, maxlen=MAX_LAGS
      )
      last_known_admissions_by_loc[current_loc_code] = (
          loc_history_deque  # Store new deque
      )

    current_features_dict = {
        col: row[col]
        for col in features_for_lgbm
        if col not in lag_feature_names
    }

    for i in range(1, MAX_LAGS + 1):
      current_features_dict[f'normalized_admissions_lag_{i}'] = (
          loc_history_deque[-i]
      )

    features_for_prediction_df = pd.DataFrame(
        [current_features_dict], index=[original_idx]
    )[features_for_lgbm]

    for col in categorical_cols_for_lgbm:
      if col in features_for_prediction_df.columns:
        features_for_prediction_df[col] = features_for_prediction_df[
            col
        ].astype('category')

    pred_norm = global_norm_admissions_mean  # Default prediction if model is not available or fails
    if model:
      try:
        pred_log_norm = model.predict(features_for_prediction_df)
        pred_norm = np.expm1(pred_log_norm[0])
      except Exception as e:
        # FIX 5: Use warnings.warn instead of suppressed print, adhering to instruction.
        warnings.warn(
            f'LGBM prediction failed for row {original_idx} (Location:'
            f" {current_loc_code}, Date: {row['target_end_date']}): {e}. Using"
            ' global mean as fallback.'
        )
        pass  # Continue with default pred_norm
    pred_norm = np.maximum(
        0.0, pred_norm
    )  # Ensure non-negative normalized prediction

    current_population = row['population']
    pred_val = pred_norm * (current_population / POP_NORMALIZATION_FACTOR)
    pred_val = np.maximum(0.0, pred_val)

    base_predictions_list.append(pred_val)
    base_predictions_indices.append(original_idx)

    loc_history_deque.append(pred_norm)

  base_predictions = pd.Series(
      base_predictions_list, index=base_predictions_indices
  ).loc[test_x_original_index]

  if base_predictions.empty:
    warnings.warn('No base predictions were made. Returning zero quantiles.')
    test_y_hat_quantiles = pd.DataFrame(
        np.zeros((len(test_x), len(QUANTILES))),
        columns=[f'quantile_{q}' for q in QUANTILES],
        index=test_x_original_index,
    )
    # FIX 6: Ensure integer type for the empty DataFrame as well
    return test_y_hat_quantiles.astype(int)

  base_predictions = base_predictions.fillna(0.0)

  # --- Simulation Loop: Execute N Monte Carlo Runs ---
  num_test_rows = len(base_predictions)
  all_simulated_forecasts = np.zeros((N_SIMULATIONS, num_test_rows))

  for i in range(N_SIMULATIONS):
    # 1. Coherent Variant Impact Trajectory (vectorized random walk for deviation)
    sampled_variant_key = rng.choice(variant_choices, p=variant_probs)
    variant_config = variant_scenarios[sampled_variant_key]
    initial_variant_deviation = rng.normal(
        loc=variant_config['initial_deviation_mean'],
        scale=variant_config['initial_deviation_std'],
    )
    # MODIFICATION 2: Use specific random_walk_step_std for each variant type
    variant_rw_step_std = variant_config['random_walk_step_std']

    variant_deviation_trajectory = (
        initial_variant_deviation
        + rng.normal(0, variant_rw_step_std, size=num_test_rows).cumsum()
    )  # Starts effectively from initial_deviation, cumsum for trajectory
    variant_multiplier_trajectory = 1.0 + variant_deviation_trajectory
    variant_multiplier_trajectory = np.clip(
        variant_multiplier_trajectory,
        variant_impact_clip['min'],
        variant_impact_clip['max'],
    )

    # 2. Coherent Mobility Change Trajectory (vectorized random walk)
    initial_mobility_shock = rng.normal(
        loc=mobility_dist_params['loc'], scale=mobility_dist_params['scale']
    )

    # MODIFICATION 1: Use slightly increased random_walk_step_std from config
    mobility_rw_step_std = mobility_dist_params['random_walk_step_std']
    mobility_shocks_trajectory = (
        initial_mobility_shock
        + rng.normal(0, mobility_rw_step_std, size=num_test_rows).cumsum()
    )  # Starts effectively from initial_shock
    mobility_multiplier_trajectory = 1.0 + mobility_shocks_trajectory
    mobility_multiplier_trajectory = np.clip(
        mobility_multiplier_trajectory,
        mobility_multiplier_clip['min'],
        mobility_multiplier_clip['max'],
    )

    # 3. Coherent Seasonal Fluctuations Trajectory (vectorized random walk for deviation)
    initial_seasonal_deviation = rng.normal(
        loc=0.0, scale=seasonality_uncertainty_scale
    )

    # MODIFICATION 1: Use slightly increased random_walk_step_std from config
    seasonal_rw_step_std = seasonal_random_walk_step_std
    seasonal_deviation_trajectory = (
        initial_seasonal_deviation
        + rng.normal(0, seasonal_rw_step_std, size=num_test_rows).cumsum()
    )  # Starts effectively from initial_deviation
    seasonal_multiplier_trajectory = 1.0 + seasonal_deviation_trajectory
    seasonal_multiplier_trajectory = np.clip(
        seasonal_multiplier_trajectory,
        seasonality_noise_clip['min'],
        seasonality_noise_clip['max'],
    )

    # Combine all scenario impacts into a single trajectory multiplier
    scenario_multiplier_trajectory = (
        variant_multiplier_trajectory
        * mobility_multiplier_trajectory
        * seasonal_multiplier_trajectory
    )

    current_scenario_forecast_multiplied = (
        base_predictions.values * scenario_multiplier_trajectory
    )
    current_scenario_forecast_multiplied = np.maximum(
        0.0, current_scenario_forecast_multiplied
    )

    # MODIFICATION 3: Add multiplicative Gaussian noise to simulate overdispersion
    overdispersion_noise = 1 + rng.normal(
        0, OVERDISPERSION_MULT_NOISE_STD, size=num_test_rows
    )
    overdispersion_noise = np.clip(
        overdispersion_noise,
        OVERDISPERSION_NOISE_CLIP['min'],
        OVERDISPERSION_NOISE_CLIP['max'],
    )
    current_scenario_forecast_multiplied = (
        current_scenario_forecast_multiplied * overdispersion_noise
    )
    current_scenario_forecast_multiplied = np.maximum(
        0.0, current_scenario_forecast_multiplied
    )

    # FIX 7: Apply minimum Poisson lambda for robustness, especially when base prediction is zero.
    # This allows for a small amount of "uncertainty" (e.g., occasional 1 or 2 admissions)
    # even if the median forecast is exactly zero, improving calibration for low counts.
    lam_for_poisson = np.maximum(
        current_scenario_forecast_multiplied, MIN_POISSON_LAM_FOR_ZEROS
    )
    current_scenario_forecast = rng.poisson(lam=lam_for_poisson)

    all_simulated_forecasts[i, :] = current_scenario_forecast

  # --- Aggregation: Calculate Empirical Quantiles ---
  predicted_quantiles_raw = np.percentile(
      all_simulated_forecasts, q=[q * 100 for q in QUANTILES], axis=0
  )

  predicted_quantiles = np.maximum(
      0.0, predicted_quantiles_raw
  )  # Ensure non-negativity

  # FIX 4: Enforce monotonicity across quantiles BEFORE rounding to integer
  predicted_quantiles = np.maximum.accumulate(predicted_quantiles, axis=0)

  # Round to integer counts
  predicted_quantiles = np.round(predicted_quantiles)

  # FIX 4: Re-apply monotonicity enforcement AFTER rounding to ensure integer monotonicity.
  # Rounding can sometimes subtly break monotonicity for values that were very close.
  predicted_quantiles = np.maximum.accumulate(predicted_quantiles, axis=0)

  quantile_col_names = [f'quantile_{q}' for q in QUANTILES]

  test_y_hat_quantiles = pd.DataFrame(
      predicted_quantiles.T, columns=quantile_col_names
  )
  test_y_hat_quantiles.index = test_x_original_index

  # Explicitly cast quantile columns to integer type as per sample_submission_df
  for col in quantile_col_names:
    test_y_hat_quantiles[col] = test_y_hat_quantiles[col].astype(int)

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
