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
MODEL_NAME = 'Google_SAI-Novel_1'
TARGET_STR = 'Total COVID-19 Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

# Core principles of the method:
# 1. Automated Bayesian Change Point (BCP) detection on hospitalization growth rate.
# 2. Trigger regime switch alert if BCP posterior probability exceeds a threshold.
# 3. Upon alert, retrain models with temporal decay (stronger for pre-switch data).
# 4. Upon alert, widen prediction intervals by a multiplicative factor.

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import logging
from typing import Any

# Configure logging to suppress verbose output from LightGBM and potential FutureWarnings
logging.getLogger('lightgbm').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings(
    'ignore', category=UserWarning
)  # For common pandas warnings not critical for execution


class RegimeSwitchModel:
  """A class encapsulating the regime-switching detection and LightGBM forecasting logic.

  This helps to manage state and organize the complex pipeline steps.
  """

  def __init__(self, config):
    self.config = config

    # Configuration parameters for the Detection Module:
    self.bcp_analysis_window_weeks = config.get(
        'bcp_analysis_window_weeks', 78
    )  # Total weeks of data to consider for BCP (e.g., 1.5 years)
    self.bcp_search_window_weeks = config.get(
        'bcp_search_window_weeks', 12
    )  # Weeks from the end of analysis window to search for CPT
    self.min_data_for_bcp_segment = config.get(
        'min_data_for_bcp_segment', 5
    )  # Minimum data points required for pre/post change segments
    self.change_detection_confidence_threshold = config.get(
        'change_detection_confidence_threshold', 0.95
    )  # Posterior probability threshold for alert
    self.change_point_prior_probability = config.get(
        'change_point_prior_probability', 0.01
    )  # Prior probability of a change occurring at any given point
    self.min_std_epsilon = (
        1e-6  # Epsilon to prevent division by zero or log(0) for std dev in BCP
    )
    self.growth_rate_diff_period = config.get(
        'growth_rate_diff_period', 2
    )  # Period for calculating hospitalization growth rate (e.g., diff(1) or diff(2))

    # Configuration parameters for the Response Protocol actions
    self.base_temporal_decay_factor = config.get(
        'base_temporal_decay_factor', 0.03
    )  # Base exponential decay rate for sample weights
    self.regime_switch_decay_multiplier = config.get(
        'regime_switch_decay_multiplier', 1.25
    )  # Multiplier for decay if regime switch detected
    self.widen_prediction_factor = config.get(
        'widen_prediction_factor', 1.15
    )  # Factor to widen prediction intervals

    # LightGBM model hyperparameters. Objective is always 'quantile'.
    self.lgbm_params = config.get(
        'lgbm_params',
        {
            'n_estimators': 500,
            'learning_rate': 0.015,
            'num_leaves': 63,
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'verbose': -1,  # Suppress verbose output during training
            'importance_type': (
                'gain'
            ),  # Good for feature importance if needed later
            'colsample_bytree': 0.8,  # Feature subsampling for regularization
            'subsample': 0.8,  # Data subsampling for regularization
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 0.1,  # L2 regularization
            'min_child_samples': 20,  # Minimum data in a child leaf
        },
    ).copy()
    self.lgbm_params['objective'] = 'quantile'  # Enforce quantile objective

    # Feature Engineering parameters
    self.lag_feature_list_numbers = config.get(
        'lag_feature_list_numbers', [1, 2, 3, 4, 8, 13, 26, 52, 104]
    )  # Lags of target variable
    self.rolling_window_sizes = config.get(
        'rolling_window_sizes', [4, 8, 13, 26]
    )  # Rolling window sizes for mean/std
    self.target_scale_factor = config.get(
        'target_scale_factor', 100000
    )  # For admissions per capita (e.g., per 100,000 people)
    self.growth_rate_feature_name = (  # Standardized name for growth rate feature
        'log_admissions_per_capita_growth_rate_lag1'
    )

    # Model components and state variables to be learned/stored during fit
    self.quantile_models = (
        {}
    )  # Dictionary to store a LightGBM model for each quantile
    self.le = LabelEncoder()  # LabelEncoder for location names
    self.base_features = (
        []
    )  # List of non-lag/rolling features (e.g., calendar features, population)
    self.lag_features = []  # List of names for generated lag features
    self.rolling_features = []  # List of names for generated rolling features
    self.model_features = (
        []
    )  # Combined list of all features used by the LightGBM models
    self.regime_switch_detected_locations = (
        {}
    )  # Stores detection flags for each location {location_name: bool}
    self.loc_name_to_encoded = (
        {}
    )  # Mapping from location name to its integer encoding
    self.max_population = (
        1.0  # Max population for scaling, learned from training data
    )

  def _create_common_features(self, df_in):
    """Adds time-based and other common features to the DataFrame."""
    df = df_in.copy()
    df['week_of_year'] = df['target_end_date'].dt.isocalendar().week.astype(int)
    df['month'] = df['target_end_date'].dt.month
    # df['year'] = df['target_end_date'].dt.year # Removed to reduce features and redundancy

    df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Add log1p of population as a feature
    df['log1p_population'] = np.log1p(df['population'])

    return df

  def _detect_regime_switches(
      self,
      data_for_detection,
      current_train_end_date,
  ):
    """[Detection Module]: Applies an improved Bayesian Change Point (BCP) detection mechanism

    to the log growth rate of admissions per capita for each location. It
    searches
    for a change point within a recent 'search window' of the
    `bcp_analysis_window_weeks` data.

    [Trigger Mechanism]: A 'regime switch' alert is triggered if the calculated
    posterior
    probability of change at any candidate change point exceeds
    `self.change_detection_confidence_threshold`.
    """
    regime_switch_flags = {}
    unique_locations = data_for_detection['location_name'].unique()

    # Minimum data points needed for `log_growth_rate_per_capita`
    min_required_history_for_analysis = (
        self.bcp_analysis_window_weeks + self.growth_rate_diff_period
    )
    min_required_for_segments = (
        2 * self.min_data_for_bcp_segment
    )  # Ensure pre and post have enough points

    for loc_name in unique_locations:
      loc_data = data_for_detection[
          data_for_detection['location_name'] == loc_name
      ].copy()
      loc_data = loc_data.sort_values(
          'target_end_date'
      )  # Ensure data is sorted chronologically

      # Filter data to the BCP analysis window ending at current_train_end_date
      analysis_window_start_date = current_train_end_date - pd.Timedelta(
          weeks=self.bcp_analysis_window_weeks - 1
      )
      loc_data_in_window = loc_data[
          (loc_data['target_end_date'] >= analysis_window_start_date)
          & (loc_data['target_end_date'] <= current_train_end_date)
      ]

      if len(loc_data_in_window) < min_required_history_for_analysis:
        regime_switch_flags[loc_name] = False
        continue

      admissions_series = np.maximum(
          0, loc_data_in_window['Total COVID-19 Admissions'].values
      )
      population_series = (
          loc_data_in_window['population'].values + 1e-9
      )  # Add epsilon to avoid division by zero

      admissions_per_capita = (
          admissions_series / population_series
      ) * self.target_scale_factor
      log_admissions_per_capita = np.log1p(admissions_per_capita)

      # Calculate Key Indicator Time Series for BCP: Hospitalization Growth Rate
      if len(log_admissions_per_capita) < self.growth_rate_diff_period + 1:
        regime_switch_flags[loc_name] = False
        continue

      log_growth_rate_per_capita_raw = (
          pd.Series(log_admissions_per_capita)
          .diff(self.growth_rate_diff_period)
          .dropna()
          .values
      )
      log_growth_rate_per_capita = log_growth_rate_per_capita_raw[
          np.isfinite(log_growth_rate_per_capita_raw)
      ]

      num_growth_rate_points = len(log_growth_rate_per_capita)

      if num_growth_rate_points < min_required_for_segments:
        regime_switch_flags[loc_name] = False
        continue

      # Calculate parameters for the "no change" model over the entire analysis window
      mu_entire = np.mean(log_growth_rate_per_capita)
      sigma_entire = np.std(log_growth_rate_per_capita)
      sigma_entire = np.maximum(
          sigma_entire, self.min_std_epsilon
      )  # Clamp std dev

      try:
        log_likelihood_entire_no_change = np.sum(
            norm.logpdf(
                log_growth_rate_per_capita, loc=mu_entire, scale=sigma_entire
            )
        )
      except RuntimeWarning as e:
        logging.warning(
            f'RuntimeWarning during norm.logpdf for {loc_name} (entire data):'
            f' {e}. Skipping BCP for this location.'
        )
        regime_switch_flags[loc_name] = False
        continue

      max_posterior_prob_change_k = 0.0

      # Define the window to search for change points
      effective_search_window_len = min(
          self.bcp_search_window_weeks,
          num_growth_rate_points - self.min_data_for_bcp_segment,
      )
      search_start_idx = num_growth_rate_points - effective_search_window_len
      search_start_idx = max(self.min_data_for_bcp_segment, search_start_idx)
      search_end_idx = num_growth_rate_points - self.min_data_for_bcp_segment

      if search_start_idx >= search_end_idx:
        regime_switch_flags[loc_name] = False
        continue

      for cp_idx in range(search_start_idx, search_end_idx + 1):
        pre_change_data = log_growth_rate_per_capita[:cp_idx]
        post_change_data = log_growth_rate_per_capita[cp_idx:]

        if (
            len(pre_change_data) < self.min_data_for_bcp_segment
            or len(post_change_data) < self.min_data_for_bcp_segment
        ):
          continue

        mu_pre = np.mean(pre_change_data)
        sigma_pre = np.std(pre_change_data)
        sigma_pre = np.maximum(sigma_pre, self.min_std_epsilon)

        mu_post = np.mean(post_change_data)
        sigma_post = np.std(post_change_data)
        sigma_post = np.maximum(sigma_post, self.min_std_epsilon)

        try:
          # Log-likelihood of data under a change point at cp_idx
          log_likelihood_change_at_k = np.sum(
              norm.logpdf(pre_change_data, loc=mu_pre, scale=sigma_pre)
          ) + np.sum(
              norm.logpdf(post_change_data, loc=mu_post, scale=sigma_post)
          )
        except RuntimeWarning as e:
          logging.warning(
              f'RuntimeWarning during norm.logpdf for {loc_name} at cp_idx'
              f' {cp_idx} (change model): {e}. Skipping this change point.'
          )
          continue

        # Bayesian model comparison using log evidences
        log_evidence_change_k = log_likelihood_change_at_k + np.log(
            self.change_point_prior_probability
        )
        log_evidence_no_change = log_likelihood_entire_no_change + np.log(
            1.0 - self.change_point_prior_probability
        )

        # Compute posterior probability of change at k using log-sum-exp trick
        max_log_evidence = max(log_evidence_change_k, log_evidence_no_change)

        try:
          term_change = np.exp(log_evidence_change_k - max_log_evidence)
          term_no_change = np.exp(log_evidence_no_change - max_log_evidence)

          if (
              not np.isfinite(term_change)
              or not np.isfinite(term_no_change)
              or (term_change + term_no_change) == 0
          ):
            posterior_prob_change_k = 0.0
          else:
            posterior_prob_change_k = term_change / (
                term_change + term_no_change
            )
        except RuntimeWarning as e:
          logging.warning(
              f'RuntimeWarning during exp/sum for posterior prob for {loc_name}'
              f' at cp_idx {cp_idx}: {e}. Setting to 0.0.'
          )
          posterior_prob_change_k = 0.0

        max_posterior_prob_change_k = max(
            max_posterior_prob_change_k, posterior_prob_change_k
        )

      is_regime_switch = (
          max_posterior_prob_change_k
          > self.change_detection_confidence_threshold
      )
      regime_switch_flags[loc_name] = is_regime_switch

    return regime_switch_flags

  def _prepare_data_and_features(
      self,
      df_in,
      is_training,
      current_target_date = None,
  ):
    """Helper to create features for both training and prediction."""
    df = self._create_common_features(df_in)

    # Ensure target_end_date is datetime before sorting
    if not pd.api.types.is_datetime64_any_dtype(df['target_end_date']):
      df['target_end_date'] = pd.to_datetime(df['target_end_date'])

    # Sort for correct lag/rolling calculations
    df = df.sort_values(by=['location_name', 'target_end_date']).reset_index(
        drop=True
    )

    if is_training:
      # During training, learn label encoding and max population
      self.max_population = (
          df['population'].max() if not df['population'].empty else 1.0
      )
      self.le = LabelEncoder()
      self.le.fit(df['location_name'])
      self.loc_name_to_encoded = dict(
          zip(self.le.classes_, self.le.transform(self.le.classes_))
      )

    df['population_scaled'] = df['population'] / self.max_population
    df['location_encoded'] = df['location_name'].map(self.loc_name_to_encoded)
    df['location_encoded'] = (
        df['location_encoded'].fillna(-1).astype(int)
    )  # Handle unseen locations in prediction if any

    df['Total COVID-19 Admissions'] = pd.to_numeric(
        df['Total COVID-19 Admissions'], errors='coerce'
    )
    df['admissions_per_capita'] = (
        df['Total COVID-19 Admissions'] / (df['population'] + 1e-9)
    ) * self.target_scale_factor
    df['log_admissions_per_capita'] = np.log1p(df['admissions_per_capita'])

    # Define model features (ensure order consistency)
    self.base_features = [
        'population_scaled',
        'log1p_population',
        'week_of_year',
        'month',
        'week_of_year_sin',
        'week_of_year_cos',
        'location_encoded',
    ]
    self.lag_features = [
        f'lag_log_admissions_per_capita_{lag}'
        for lag in self.lag_feature_list_numbers
    ]
    self.rolling_features = [
        f'rolling_mean_log_admissions_per_capita_{window_size}_wk'
        for window_size in self.rolling_window_sizes
    ] + [
        f'rolling_std_log_admissions_per_capita_{window_size}_wk'
        for window_size in self.rolling_window_sizes
    ]

    self.model_features = (
        self.base_features
        + self.lag_features
        + self.rolling_features
        + [self.growth_rate_feature_name]
    )

    # Generate lag, rolling, and growth rate features for all data
    # Fill NaNs created by lags/rolling for model features
    for loc_code in df['location_encoded'].unique():
      loc_mask = df['location_encoded'] == loc_code

      # Generate lag features
      for lag in self.lag_feature_list_numbers:
        df.loc[loc_mask, f'lag_log_admissions_per_capita_{lag}'] = df.loc[
            loc_mask, 'log_admissions_per_capita'
        ].shift(lag)

      # Generate rolling features
      for window_size in self.rolling_window_sizes:
        df.loc[
            loc_mask, f'rolling_mean_log_admissions_per_capita_{window_size}_wk'
        ] = (
            df.loc[loc_mask, 'log_admissions_per_capita']
            .rolling(window=window_size, min_periods=1)
            .mean()
            .shift(1)
        )
        df.loc[
            loc_mask, f'rolling_std_log_admissions_per_capita_{window_size}_wk'
        ] = (
            df.loc[loc_mask, 'log_admissions_per_capita']
            .rolling(window=window_size, min_periods=2)
            .std(ddof=1)
            .shift(1)
        )

      # Generate growth rate feature
      df.loc[loc_mask, self.growth_rate_feature_name] = (
          df.loc[loc_mask, 'log_admissions_per_capita']
          .diff(self.growth_rate_diff_period)
          .shift(1)
      )

      # Fill NaNs created by lags/rolling for model features (forward fill, then fill remaining with 0.0)
      for col in self.model_features:
        if col in df.columns:
          df.loc[loc_mask, col] = (
              df.loc[loc_mask, col].ffill().fillna(0.0)
          )  # ffill then fill remaining NaNs (at start of series)

    return df

  def fit(self, train_x, train_y):
    """Trains the LightGBM quantile models.

    This method implements elements of the Response Protocol, specifically: 1.
    Force immediate retraining of core predictive models; 2. Apply temporal
    decay weight to the loss function during retraining.
    """
    train_df = train_x.copy()
    train_df['Total COVID-19 Admissions'] = train_y

    combined_df_train = self._prepare_data_and_features(
        train_df, is_training=True
    )

    # Latest date in the training data for decay weighting reference
    latest_overall_train_date = combined_df_train['target_end_date'].max()

    # Perform regime switch detection on the processed training data
    self.regime_switch_detected_locations = self._detect_regime_switches(
        combined_df_train, latest_overall_train_date
    )

    X_train_for_model = combined_df_train[self.model_features]
    y_train_for_model = combined_df_train['log_admissions_per_capita']

    # Calculate base temporal decay weights
    time_diff_train = (
        latest_overall_train_date - combined_df_train['target_end_date']
    ).dt.days / 7
    base_sample_weights = np.exp(
        -self.base_temporal_decay_factor * time_diff_train
    )
    base_sample_weights.index = combined_df_train.index

    self.quantile_models = {}
    for q in QUANTILES:
      model = lgb.LGBMRegressor(alpha=q, **self.lgbm_params)

      current_sample_weights = base_sample_weights.copy()

      # Apply regime-switch specific decay
      for loc_name, is_switch in self.regime_switch_detected_locations.items():
        if is_switch:
          loc_encoded_val = self.loc_name_to_encoded.get(loc_name)
          if loc_encoded_val is not None:
            loc_mask = combined_df_train['location_encoded'] == loc_encoded_val
            effective_decay_factor = (
                self.base_temporal_decay_factor
                * self.regime_switch_decay_multiplier
            )
            time_diff_loc = (
                latest_overall_train_date
                - combined_df_train.loc[loc_mask, 'target_end_date']
            ).dt.days / 7
            current_sample_weights.loc[loc_mask] = np.exp(
                -effective_decay_factor * time_diff_loc
            )

      current_sample_weights = np.clip(
          current_sample_weights, 1e-6, None
      )  # Ensure weights are not zero or negative

      # Combine data, target, and weights for final training
      aligned_data = pd.concat(
          [
              X_train_for_model,
              y_train_for_model.rename('y'),
              current_sample_weights.rename('weights'),
          ],
          axis=1,
      )
      final_train_df = aligned_data.dropna(
          subset=self.model_features + ['y', 'weights']
      )

      if not final_train_df.empty:
        model.fit(
            final_train_df[self.model_features],
            final_train_df['y'],
            sample_weight=final_train_df['weights'],
        )
        self.quantile_models[q] = model
      else:
        self.quantile_models[q] = (
            None  # No data to train, store None or raise error
        )

  def predict(
      self, test_x, train_x, train_y
  ):
    """Generates quantile predictions for test_x, iteratively computing features

    and applying prediction interval widening where regime switches were
    detected.

    This method implements remaining parts of the Response Protocol:
    3. Signal the meta-ensemble to adjust weights. (Addressed through decay and
    widening)
    4. Apply a multiplicative factor to widen final prediction intervals.
    """
    # Ensure date columns are datetime types
    train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
    test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

    # Prepare combined data for feature generation
    train_df_for_pred = train_x.copy()
    train_df_for_pred['Total COVID-19 Admissions'] = train_y
    train_df_for_pred['is_test'] = False

    test_df_copy = test_x.copy()
    test_df_copy['Total COVID-19 Admissions'] = (
        np.nan
    )  # Target is unknown for test
    test_df_copy['is_test'] = True

    # Store original indices to map back predictions
    train_df_for_pred['original_index'] = train_df_for_pred.index
    test_df_copy['original_index'] = test_df_copy.index

    combined_df_for_pred = pd.concat(
        [train_df_for_pred, test_df_copy], ignore_index=True
    )

    # Calculate features once for the entire combined dataset.
    combined_df_processed = self._prepare_data_and_features(
        combined_df_for_pred, is_training=False
    )

    all_predictions_list = []
    unique_test_target_dates = (
        combined_df_processed[combined_df_processed['is_test']][
            'target_end_date'
        ]
        .sort_values()
        .unique()
    )

    # Iterate through unique test prediction dates in chronological order
    for current_prediction_date in unique_test_target_dates:
      # Select the rows that need prediction for the current date
      current_test_rows = combined_df_processed[
          (combined_df_processed['is_test'])
          & (
              combined_df_processed['target_end_date']
              == current_prediction_date
          )
      ]

      if current_test_rows.empty:
        continue

      # Extract features for prediction
      # Fill any remaining NaNs in test features, although _prepare_data_and_features should have handled most.
      X_test_current_date = current_test_rows[self.model_features].fillna(0.0)

      # Store log-scaled quantile predictions
      quantile_preds_log_scaled = pd.DataFrame(index=current_test_rows.index)
      median_preds_log_scaled = pd.Series(
          index=current_test_rows.index, dtype=float
      )

      for q in QUANTILES:
        model = self.quantile_models.get(q)
        if model is not None:
          pred = model.predict(X_test_current_date)
        else:
          pred = np.full(len(X_test_current_date), np.log1p(0.0))  # Fallback

        quantile_preds_log_scaled[f'q_{q}'] = pred
        if q == 0.5:
          median_preds_log_scaled = pd.Series(
              pred, index=current_test_rows.index
          )

      # Apply prediction interval widening for detected regime switches
      for loc_name in current_test_rows['location_name'].unique():
        if self.regime_switch_detected_locations.get(loc_name, False):
          loc_mask = current_test_rows['location_name'] == loc_name

          loc_median_pred_log = median_preds_log_scaled.loc[loc_mask]

          if (
              loc_median_pred_log.empty
          ):  # Should not happen if loc_mask is valid
            continue

          for q in QUANTILES:
            q_col = f'q_{q}'
            loc_q_preds_log = quantile_preds_log_scaled.loc[loc_mask, q_col]

            if not loc_q_preds_log.empty:
              if q <= 0.5:
                widened_val = (
                    loc_median_pred_log
                    - (loc_median_pred_log - loc_q_preds_log)
                    * self.widen_prediction_factor
                )
              else:
                widened_val = (
                    loc_median_pred_log
                    + (loc_q_preds_log - loc_median_pred_log)
                    * self.widen_prediction_factor
                )
              quantile_preds_log_scaled.loc[loc_mask, q_col] = widened_val

      # Convert log-scaled predictions back to raw counts
      final_predictions_for_date_float = pd.DataFrame(
          index=current_test_rows.index
      )
      current_populations = current_test_rows['population'] + 1e-9

      for q_idx, q in enumerate(QUANTILES):
        pred_log_scaled = quantile_preds_log_scaled[f'q_{q}']
        pred_admissions_per_capita = np.expm1(pred_log_scaled)
        raw_count_pred = (
            pred_admissions_per_capita / self.target_scale_factor
        ) * current_populations
        final_predictions_for_date_float[f'quantile_{q}'] = raw_count_pred

      # Ensure non-negativity and then monotonicity for float predictions (before rounding)
      final_predictions_for_date_float = np.maximum(
          0, final_predictions_for_date_float
      )
      final_predictions_for_date_float = final_predictions_for_date_float.apply(
          lambda row: np.maximum.accumulate(row), axis=1
      )

      # Round to integer after ensuring non-negativity and monotonicity
      final_predictions_for_date = np.round(
          final_predictions_for_date_float
      ).astype(int)

      # Collect predictions for the current date
      for original_idx_in_temp_df, row in final_predictions_for_date.iterrows():
        row_predictions_dict = {
            'original_index': current_test_rows.loc[
                original_idx_in_temp_df, 'original_index'
            ]
        }
        for q in QUANTILES:
          row_predictions_dict[f'quantile_{q}'] = row[f'quantile_{q}']
        all_predictions_list.append(row_predictions_dict)

      # Update `combined_df_processed` with median predictions for next iteration's feature calculations
      median_q_col_name = f'quantile_{0.5}'
      if median_q_col_name in final_predictions_for_date.columns:
        median_pred_values = final_predictions_for_date[
            median_q_col_name
        ].astype(float)

        indices_to_update = current_test_rows.index
        combined_df_processed.loc[
            indices_to_update, 'Total COVID-19 Admissions'
        ] = median_pred_values.values

        combined_df_processed.loc[
            indices_to_update, 'admissions_per_capita'
        ] = (
            combined_df_processed.loc[
                indices_to_update, 'Total COVID-19 Admissions'
            ]
            / (
                combined_df_processed.loc[indices_to_update, 'population']
                + 1e-9
            )
        ) * self.target_scale_factor
        combined_df_processed.loc[
            indices_to_update, 'log_admissions_per_capita'
        ] = np.log1p(
            combined_df_processed.loc[
                indices_to_update, 'admissions_per_capita'
            ]
        )

        # Efficiently update ONLY the relevant future lagged/rolling features
        # The minimum `lag` or `window_size` determines how far back we need to start recalculating.
        max_lag_or_window = max(
            max(self.lag_feature_list_numbers),
            max(self.rolling_window_sizes, default=0),
            self.growth_rate_diff_period,
        )
        recalc_start_date = current_prediction_date - pd.Timedelta(
            weeks=max_lag_or_window
        )

        # Identify rows from the start of the recalculation window up to the end of the combined_df
        # This ensures rolling/lag calculations correctly pick up updated values
        rows_for_feature_recalc_mask = (
            combined_df_processed['target_end_date'] >= recalc_start_date
        ) & (
            combined_df_processed['location_encoded'].isin(
                current_test_rows['location_encoded'].unique()
            )
        )

        # Recompute lags, rolling stats, and growth rate for the affected window
        for loc_code in current_test_rows['location_encoded'].unique():
          loc_recalc_mask = rows_for_feature_recalc_mask & (
              combined_df_processed['location_encoded'] == loc_code
          )
          loc_data_for_recalc = combined_df_processed.loc[
              loc_recalc_mask,
              [
                  'target_end_date',
                  'log_admissions_per_capita',
                  'location_encoded',
              ],
          ].sort_values('target_end_date')

          if loc_data_for_recalc.empty:
            continue

          # Apply feature generation specifically to this slice and then update combined_df_processed
          # This ensures features for *future* prediction dates (in `combined_df_processed`) are correctly updated
          # based on the new median predictions for `current_prediction_date`.

          # Lag features
          for lag in self.lag_feature_list_numbers:
            combined_df_processed.loc[
                loc_recalc_mask, f'lag_log_admissions_per_capita_{lag}'
            ] = (
                loc_data_for_recalc['log_admissions_per_capita']
                .shift(lag)
                .values
            )

          # Rolling features
          for window_size in self.rolling_window_sizes:
            combined_df_processed.loc[
                loc_recalc_mask,
                f'rolling_mean_log_admissions_per_capita_{window_size}_wk',
            ] = (
                loc_data_for_recalc['log_admissions_per_capita']
                .rolling(window=window_size, min_periods=1)
                .mean()
                .shift(1)
                .values
            )
            combined_df_processed.loc[
                loc_recalc_mask,
                f'rolling_std_log_admissions_per_capita_{window_size}_wk',
            ] = (
                loc_data_for_recalc['log_admissions_per_capita']
                .rolling(window=window_size, min_periods=2)
                .std(ddof=1)
                .shift(1)
                .values
            )

          # Growth rate feature
          combined_df_processed.loc[
              loc_recalc_mask, self.growth_rate_feature_name
          ] = (
              loc_data_for_recalc['log_admissions_per_capita']
              .diff(self.growth_rate_diff_period)
              .shift(1)
              .values
          )

          # Fill NaNs for the recalculated window
          for col in (
              self.lag_features
              + self.rolling_features
              + [self.growth_rate_feature_name]
          ):
            if col in combined_df_processed.columns:
              combined_df_processed.loc[loc_recalc_mask, col] = (
                  combined_df_processed.loc[loc_recalc_mask, col]
                  .ffill()
                  .fillna(0.0)
              )

    test_y_hat_quantiles = pd.DataFrame(all_predictions_list)

    if not test_y_hat_quantiles.empty:
      test_y_hat_quantiles = test_y_hat_quantiles.set_index('original_index')
      test_y_hat_quantiles = test_y_hat_quantiles.reindex(test_x.index)
      # Ensure final output types
      for q_col in [f'quantile_{q}' for q in QUANTILES]:
        if q_col in test_y_hat_quantiles.columns:
          test_y_hat_quantiles[q_col] = test_y_hat_quantiles[q_col].astype(int)
    else:
      quantile_cols = [f'quantile_{q}' for q in QUANTILES]
      test_y_hat_quantiles = pd.DataFrame(
          columns=quantile_cols, index=test_x.index
      )
      for col in quantile_cols:
        test_y_hat_quantiles[col] = 0

    return test_y_hat_quantiles


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  """Implements an automated regime-switching detection and response system for

  forecasting COVID-19 hospital admissions, adhering to the specified protocol.

  This function initializes and utilizes the `RegimeSwitchModel` class to
  encapsulate
  the complex logic of detection, retraining, weighted loss, and prediction
  interval adjustment.
  """
  # Ensure date columns are datetime types right at the start
  train_x['target_end_date'] = pd.to_datetime(train_x['target_end_date'])
  test_x['target_end_date'] = pd.to_datetime(test_x['target_end_date'])

  # Adjusted configuration constants for potentially better performance.
  config = {
      # Tuned configuration for improved performance and robustness
      'bcp_analysis_window_weeks': 104,  # Longer window for BCP (2 years)
      'bcp_search_window_weeks': (
          16
      ),  # Search for change points within the most recent 16 weeks
      'min_data_for_bcp_segment': (
          8
      ),  # Minimum data points for pre/post segments (increased for robustness)
      'change_detection_confidence_threshold': (
          0.95
      ),  # As per prompt example, for more active detection
      'change_point_prior_probability': 0.01,  # As per prompt example
      'growth_rate_diff_period': (
          2
      ),  # Use 2-week difference for growth rate for robustness
      'base_temporal_decay_factor': (
          0.02
      ),  # Slightly slower base decay for older data
      'regime_switch_decay_multiplier': (
          1.3
      ),  # Moderate additional decay for switched regimes
      'widen_prediction_factor': (
          1.2
      ),  # Moderate widening to balance calibration and sharpness
      'lag_feature_list_numbers': [
          1,
          2,
          3,
          4,
          8,
          13,
          26,
          52,
          104,
      ],  # Extended lags to capture yearly/bi-yearly patterns
      'rolling_window_sizes': [
          4,
          8,
          13,
          26,
      ],  # Adjusted rolling windows for various temporal summaries
      'target_scale_factor': 100000,  # Scale target to per 100,000 population
      'lgbm_params': {
          'n_estimators': (
              750
          ),  # Increased estimators for potentially better fit
          'learning_rate': 0.01,  # Lower learning rate for more stable training
          'num_leaves': (
              63
          ),  # Reduced num_leaves for better generalization, balanced by estimators
          'random_state': 42,
          'n_jobs': -1,
          'verbose': -1,
          'colsample_bytree': 0.7,  # Feature subsampling for regularization
          'subsample': 0.7,  # Data subsampling for regularization
          'reg_alpha': 0.15,  # L1 regularization
          'reg_lambda': 0.15,  # L2 regularization
          'min_child_samples': 25,  # Minimum data in a child leaf
          'max_depth': 10,  # Limiting tree depth for regularization
      },
  }

  model = RegimeSwitchModel(config)
  model.fit(train_x, train_y)
  predictions_df = model.predict(test_x, train_x, train_y)

  return predictions_df


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
