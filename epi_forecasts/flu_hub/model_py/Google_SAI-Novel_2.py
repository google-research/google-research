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
MODEL_NAME = 'Google_SAI-Novel_2'
TARGET_STR = 'Total Influenza Admissions'

ilinet_hhs = pd.read_csv(f'{INPUT_DIR}/ilinet_hhs_before_20221015.csv')
ilinet = pd.read_csv(f'{INPUT_DIR}/ilinet_before_20221015.csv')
ilinet_state = pd.read_csv(f'{INPUT_DIR}/ilinet_state_before_20221015.csv')
locations = pd.read_csv(f'{INPUT_DIR}/locations.csv')

import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import copy

# --- Configuration Constants ---

# Define a standard season length (e.g., 52 weeks from epiweek 40 to 39 next year)
SEASON_WEEKS_ORDERED = list(range(40, 53)) + list(
    range(1, 40)
)  # 52 weeks total
SERIES_LENGTH = len(SEASON_WEEKS_ORDERED)  # 52
MASK_LENGTH = 4  # Mask out last 4 weeks for pre-training

# Map horizons to lead times from the end of the input series
# Input series ends at `t`.
# Horizon -1: predict `t` (lead_time 0)
# Horizon 0:  predict `t+1` (lead_time 1)
# Horizon 1:  predict `t+2` (lead_time 2)
# Horizon 2:  predict `t+3` (lead_time 3)
# Horizon 3:  predict `t+4` (lead_time 4)
HORIZON_TO_LEAD_TIME_IDX = {h: h + 1 for h in HORIZONS}
# Total number of future points to predict, starting from the end of the input series
NUM_PRED_POINTS = (
    max(HORIZON_TO_LEAD_TIME_IDX.values()) + 1
)  # Should be 5 (0,1,2,3,4)

PRETRAINED_MODEL_PATH = '/working/pretrained_vision_encoder.pth'  # Check path is correct for your setup


# --- Helper Functions ---
def get_epiweek_from_date(date_obj):
  return pd.to_datetime(date_obj).isocalendar().week


def get_season_year_from_date(date_obj):
  date_obj = pd.to_datetime(date_obj)
  if date_obj.isocalendar().week >= 40:
    return date_obj.year
  else:
    return date_obj.year - 1


# --- Robust Scaler Implementation ---
class RobustMinMaxScaler:
  """A wrapper around MinMaxScaler that handles constant (zero-variance) data gracefully.

  - If fitted on constant zero data, it maps all inputs to zero. - If fitted on
  constant non-zero data, it maps all inputs to 0.5 (within a [0,1] range). -
  Otherwise, it behaves like a standard MinMaxScaler.
  """

  def __init__(self, feature_range=(0, 1)):
    self._scaler = MinMaxScaler(feature_range=feature_range)
    self._is_fitted = False
    self._is_constant = False
    self._constant_value = 0.0
    self._scaled_constant_value = 0.0
    self.feature_range = feature_range

  def fit(self, X):
    if X.ndim == 1:
      X = X.reshape(-1, 1)

    if X.shape[0] == 0:
      warnings.warn(
          'Fitting RobustMinMaxScaler on empty data. Will default to constant'
          ' zero behavior.'
      )
      self._is_constant = True
      self._constant_value = 0.0
      self._scaled_constant_value = self.feature_range[
          0
      ]  # Maps to min of feature range
    else:
      unique_values = np.unique(X)
      if len(unique_values) == 1:
        self._is_constant = True
        self._constant_value = unique_values[0]
        # Map 0 to min_range, other constants to midpoint of feature_range
        if self._constant_value == 0:
          self._scaled_constant_value = self.feature_range[0]
        else:
          self._scaled_constant_value = (
              self.feature_range[0] + self.feature_range[1]
          ) / 2
      else:
        self._is_constant = False
        self._scaler.fit(X)
    self._is_fitted = True
    return self

  def transform(self, X):
    if not self._is_fitted:
      raise RuntimeError('Scaler not fitted yet!')

    if X.ndim == 1:
      X = X.reshape(-1, 1)

    if self._is_constant:
      if self._constant_value == 0:
        return np.zeros_like(X, dtype=float) + self.feature_range[0]
      else:
        return np.full_like(X, self._scaled_constant_value, dtype=float)
    else:
      return self._scaler.transform(X)

  def inverse_transform(self, X_scaled):
    if not self._is_fitted:
      raise RuntimeError('Scaler not fitted yet!')

    if X_scaled.ndim == 1:
      X_scaled = X_scaled.reshape(-1, 1)

    if self._is_constant:
      return np.full_like(X_scaled, self._constant_value, dtype=float)
    else:
      return self._scaler.inverse_transform(X_scaled)


# --- Time Series Processing Helper ---
def series_to_fixed_vector_and_scale(
    df_group, value_col, season_year, location_fips, scaler=None
):
  """Extracts a series for a given season and location, pads it to SERIES_LENGTH,

  and scales it using the provided scaler.
  """
  season_template_df = pd.DataFrame({
      'epiweek': SEASON_WEEKS_ORDERED,
      'season_year': [season_year] * SERIES_LENGTH,
      'location': [location_fips] * SERIES_LENGTH,
  })

  df_group_filtered = df_group[
      ['location', 'season_year', 'epiweek', value_col]
  ].copy()

  merged_df = pd.merge(
      season_template_df,
      df_group_filtered,
      on=['season_year', 'epiweek', 'location'],
      how='left',
  )

  series_values = merged_df[value_col].fillna(0).values

  if scaler is not None:
    if np.max(series_values) == np.min(series_values):  # Series is constant
      if series_values[0] == 0:
        # If all zeros, scaled should be all zeros (or min of feature range)
        return np.full(SERIES_LENGTH, scaler.feature_range[0])
      else:
        # Constant non-zero. Transform the single value, then fill array.
        scaled_val = scaler.transform(np.array([[series_values[0]]]))[0, 0]
        return np.full(SERIES_LENGTH, scaled_val)
    else:  # Variable series, normal transform
      return scaler.transform(series_values.reshape(-1, 1)).flatten()
  else:  # No scaler provided, return unscaled
    return series_values


# --- Model Architecture (1D Time Series Encoder, interpreting as 1D "vision model") ---
INPUT_SIZE_SERIES = SERIES_LENGTH  # Length of the time series vector
HIDDEN_SIZE = 128
NUM_QUANTILE_OUTPUTS = len(QUANTILES)


class TimeSeriesEncoder1D(nn.Module):

  def __init__(self, series_length, hidden_size):
    super().__init__()
    self.series_length = series_length

    self.conv_layers = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.zeros(1, 1, self.series_length).to(device)
    with torch.no_grad():
      dummy_output = self.conv_layers(dummy_input)
      flattened_size = dummy_output.view(1, -1).size(1)

    self.flatten = nn.Flatten()
    self.fc = nn.Linear(flattened_size, hidden_size)

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.conv_layers(x)
    x = self.flatten(x)
    x = F.relu(self.fc(x))
    return x


class MaskedPredictionHead(nn.Module):

  def __init__(
      self, hidden_size, output_size
  ):  # output_size here is MASK_LENGTH
    super().__init__()
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    return self.fc(x)


class QuantileRegressionHead(nn.Module):

  def __init__(self, hidden_size, num_pred_points, num_quantiles):
    super().__init__()
    self.num_pred_points = num_pred_points
    self.num_quantiles = num_quantiles
    self.fc = nn.Linear(hidden_size, num_pred_points * num_quantiles)

  def forward(self, x):
    predictions = self.fc(x)
    return predictions.view(-1, self.num_pred_points, self.num_quantiles)


class FluForecastModel(nn.Module):

  def __init__(self, encoder, pretrain_head, quantile_head):
    super().__init__()
    self.encoder = encoder
    self.pretrain_head = pretrain_head
    self.quantile_head = quantile_head

  def forward_pretrain(self, x):
    features = self.encoder(x)
    return self.pretrain_head(features)

  def forward_finetune(self, x):
    features = self.encoder(x)
    return self.quantile_head(features)


# Pinball Loss Implementation for multi-horizon, multi-quantile targets, handling NaNs
def pinball_loss_multi_horizon(y_pred, y_true, quantiles_tensor):
  # y_pred: batch_size x num_pred_points x num_quantiles
  # y_true: batch_size x num_pred_points (can contain NaNs)
  # quantiles_tensor: 1 x num_quantiles

  not_nan_mask = ~torch.isnan(y_true)  # batch_size x num_pred_points

  # Expand y_true to match y_pred's dimensions for error calculation
  y_true_expanded = y_true.unsqueeze(2)  # batch_size x num_pred_points x 1

  # Calculate error for each quantile and each horizon
  error = (
      y_true_expanded - y_pred
  )  # batch_size x num_pred_points x num_quantiles

  # Calculate loss component for each quantile and horizon
  loss = torch.max(quantiles_tensor * error, (quantiles_tensor - 1) * error)

  # Apply the mask: set loss to 0 where target is NaN
  not_nan_mask_expanded = not_nan_mask.unsqueeze(2).expand_as(loss)
  loss = loss * not_nan_mask_expanded.float()

  total_loss = torch.sum(loss)
  num_valid_predictions = torch.sum(not_nan_mask_expanded.float())

  if num_valid_predictions == 0:
    return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
  else:
    return total_loss / num_valid_predictions


# --- FluForecaster Class (encapsulates all model logic) ---
class FluForecaster:

  def __init__(self, locations_df, raw_dataset_df, raw_ilinet_state_df):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store raw dataframes
    self._locations = locations_df
    self._raw_dataset = raw_dataset_df
    self._raw_ilinet_state = raw_ilinet_state_df

    self._processed_dataset = None
    self._processed_ilinet_state = None

    self.target_global_scaler = RobustMinMaxScaler()
    self.ilinet_per_location_scalers = defaultdict(RobustMinMaxScaler)

    self._ilinet_to_target_transformers_fold = (
        {}
    )  # Stores transformers for the current fold

    # Initialize models (encoder is pre-trained, heads are randomly initialized)
    self.encoder = TimeSeriesEncoder1D(INPUT_SIZE_SERIES, HIDDEN_SIZE).to(
        self.device
    )
    self.pretrain_head = MaskedPredictionHead(HIDDEN_SIZE, MASK_LENGTH).to(
        self.device
    )
    self.quantile_head = QuantileRegressionHead(
        HIDDEN_SIZE, NUM_PRED_POINTS, NUM_QUANTILE_OUTPUTS
    ).to(self.device)
    self.model = FluForecastModel(
        self.encoder, self.pretrain_head, self.quantile_head
    ).to(self.device)

    self.is_initialized = False
    self.is_pretrained = False

    self._initial_setup()

  def _initial_setup(self):
    """Performs one-time data processing and scaler fitting."""
    print(
        'Performing initial FluForecaster setup (data preprocessing and scaler'
        ' fitting)...'
    )

    # Process `dataset`
    self._processed_dataset = self._raw_dataset.copy()
    self._processed_dataset['target_end_date'] = pd.to_datetime(
        self._processed_dataset['target_end_date']
    )
    self._processed_dataset['season_year'] = self._processed_dataset[
        'target_end_date'
    ].apply(get_season_year_from_date)
    self._processed_dataset['epiweek'] = self._processed_dataset[
        'target_end_date'
    ].apply(get_epiweek_from_date)

    # Process `ilinet_state`
    self._processed_ilinet_state = self._raw_ilinet_state.copy()
    location_name_to_fips = dict(
        zip(self._locations['location_name'], self._locations['location'])
    )
    self._processed_ilinet_state = self._processed_ilinet_state[
        self._processed_ilinet_state['region_type'] == 'States'
    ].copy()
    self._processed_ilinet_state['location'] = self._processed_ilinet_state[
        'region'
    ].map(location_name_to_fips)
    self._processed_ilinet_state = self._processed_ilinet_state.dropna(
        subset=['location']
    )
    self._processed_ilinet_state['location'] = self._processed_ilinet_state[
        'location'
    ].astype(int)
    self._processed_ilinet_state['week_start'] = pd.to_datetime(
        self._processed_ilinet_state['week_start']
    )
    self._processed_ilinet_state['season_year'] = self._processed_ilinet_state[
        'week_start'
    ].apply(get_season_year_from_date)
    self._processed_ilinet_state['epiweek'] = self._processed_ilinet_state[
        'week_start'
    ].apply(get_epiweek_from_date)

    # Fit TARGET_GLOBAL_SCALER
    print('Fitting global scaler for Total Influenza Admissions...')
    all_targets = self._processed_dataset[TARGET_STR].dropna().values
    self.target_global_scaler.fit(all_targets)
    print('Global scaler fitted.')

    # Fit ILINET_PER_LOCATION_SCALERS
    print('Fitting per-location scalers for ILINet data...')
    ilinet_grouped_by_loc_season = self._processed_ilinet_state.groupby(
        ['location', 'season_year']
    )

    ilinet_values_per_location = defaultdict(list)
    for (loc, s_year), group in ilinet_grouped_by_loc_season:
      series_values_for_season = series_to_fixed_vector_and_scale(
          group, 'ilitotal', s_year, loc, scaler=None
      )
      ilinet_values_per_location[loc].extend(series_values_for_season)

    for loc, values in ilinet_values_per_location.items():
      if len(values) > 0:
        self.ilinet_per_location_scalers[loc].fit(
            np.array(values).reshape(-1, 1)
        )
      else:  # If a location has no ILINet data, fit with dummy values
        self.ilinet_per_location_scalers[loc].fit(np.array([[0.0], [1.0]]))
    print('ILINet per-location scalers fitted.')

    self.is_initialized = True
    print('FluForecaster initial setup complete.')

  def _pretrain(self):
    """Pre-trains the time series encoder on ILINet data if not already done."""
    if self.is_pretrained:
      if os.path.exists(PRETRAINED_MODEL_PATH):
        self.encoder.load_state_dict(
            torch.load(PRETRAINED_MODEL_PATH, map_location=self.device)
        )
        print('Pre-trained encoder weights loaded.')
      else:
        warnings.warn(
            'Pre-trained model path specified, but file not found. Pre-training'
            ' again or starting fresh.'
        )
      return

    print('Pre-training time series encoder on ILINet data...')
    pretrain_data_list = []

    ilinet_grouped = self._processed_ilinet_state.groupby(
        ['location', 'season_year']
    )

    for (loc, s_year), group in ilinet_grouped:
      scaler = self.ilinet_per_location_scalers.get(loc)
      series_vector = series_to_fixed_vector_and_scale(
          group, 'ilitotal', s_year, loc, scaler=scaler
      )

      if len(series_vector) == SERIES_LENGTH and np.sum(series_vector) > 0:
        pretrain_data_list.append(series_vector)

    if not pretrain_data_list:
      warnings.warn(
          'No sufficient ILINet data for pre-training. Skipping pre-training.'
      )
      self.is_pretrained = True  # Mark as done to prevent re-attempting
      return

    pretrain_inputs_raw = torch.tensor(
        np.array(pretrain_data_list), dtype=torch.float32
    ).to(self.device)

    targets_reconstruction = pretrain_inputs_raw[
        :, SERIES_LENGTH - MASK_LENGTH :
    ].clone()
    masked_inputs = pretrain_inputs_raw.clone()
    masked_inputs[:, SERIES_LENGTH - MASK_LENGTH :] = (
        0  # Mask the end by setting to 0
    )

    pretrain_dataset = TensorDataset(masked_inputs, targets_reconstruction)
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=32, shuffle=True
    )

    pretrain_optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    pretrain_epochs = 10

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      for epoch in range(pretrain_epochs):
        for batch_idx, (inputs, targets) in enumerate(pretrain_dataloader):
          pretrain_optimizer.zero_grad()
          outputs = self.model.forward_pretrain(inputs)
          loss = F.mse_loss(outputs, targets)
          loss.backward()
          pretrain_optimizer.step()

    torch.save(self.encoder.state_dict(), PRETRAINED_MODEL_PATH)
    self.is_pretrained = True
    print('Pre-training complete. Encoder weights saved.')

  def fit(self, train_x, train_y):
    """Fits the model for the current fold."""
    if not self.is_initialized:
      raise RuntimeError(
          'FluForecaster not initialized. Call _initial_setup first (done in'
          ' constructor).'
      )

    self._pretrain()  # Ensure pre-training is done

    # --- Learn Transformation (Runs for each fold using current train_x, train_y) ---
    print(f'Learning ILINet to Target transformation for current fold...')
    self._ilinet_to_target_transformers_fold = {}

    train_df_current_fold = train_x.copy()
    train_df_current_fold[TARGET_STR] = train_y

    train_df_current_fold['target_end_date'] = pd.to_datetime(
        train_df_current_fold['target_end_date']
    )
    train_df_current_fold['season_year'] = train_df_current_fold[
        'target_end_date'
    ].apply(get_season_year_from_date)
    train_df_current_fold['epiweek'] = train_df_current_fold[
        'target_end_date'
    ].apply(get_epiweek_from_date)

    active_locations = set(train_df_current_fold['location'].unique())

    for loc in active_locations:
      loc_ilinet_df = self._processed_ilinet_state[
          self._processed_ilinet_state['location'] == loc
      ]
      loc_train_df = train_df_current_fold[
          train_df_current_fold['location'] == loc
      ]

      overlap_df = pd.merge(
          loc_ilinet_df[['week_start', 'epiweek', 'season_year', 'ilitotal']],
          loc_train_df[
              ['target_end_date', 'epiweek', 'season_year', TARGET_STR]
          ],
          left_on=['week_start', 'epiweek', 'season_year'],
          right_on=['target_end_date', 'epiweek', 'season_year'],
          how='inner',
      )

      if not overlap_df.empty:
        ilinet_scaler = self.ilinet_per_location_scalers.get(loc)
        scaled_ilitotal = ilinet_scaler.transform(
            overlap_df['ilitotal'].values.reshape(-1, 1)
        ).flatten()
        scaled_target_admissions = self.target_global_scaler.transform(
            overlap_df[TARGET_STR].values.reshape(-1, 1)
        ).flatten()

        # Sufficient data and variance for Linear Regression
        if (
            len(scaled_ilitotal) > 1
            and np.std(scaled_ilitotal) > 1e-6
            and np.std(scaled_target_admissions) > 1e-6
        ):
          regressor = LinearRegression()
          regressor.fit(
              scaled_ilitotal.reshape(-1, 1), scaled_target_admissions
          )
          self._ilinet_to_target_transformers_fold[loc] = copy.deepcopy(
              regressor
          )
        else:  # Fallback for insufficient or constant data
          mean_target = (
              scaled_target_admissions.mean()
              if len(scaled_target_admissions) > 0
              else 0.0
          )
          mean_ilitotal = (
              scaled_ilitotal.mean() if len(scaled_ilitotal) > 0 else 0.0
          )

          if mean_target == 0 and mean_ilitotal > 0:
            self._ilinet_to_target_transformers_fold[loc] = {
                'type': 'fixed_zero'
            }
          elif mean_target > 0 and mean_ilitotal > 0:
            ratio = mean_target / mean_ilitotal
            self._ilinet_to_target_transformers_fold[loc] = {
                'type': 'ratio',
                'ratio': ratio,
            }
          else:  # Default to predicting zero if no good basis for transformation
            self._ilinet_to_target_transformers_fold[loc] = {
                'type': 'fixed_zero'
            }
      else:
        self._ilinet_to_target_transformers_fold[loc] = {'type': 'fixed_zero'}
    print('ILINet to Target transformation learned for current fold.')

    # --- Fine-tuning Logic ---
    # Load pre-trained encoder weights
    if os.path.exists(PRETRAINED_MODEL_PATH):
      self.encoder.load_state_dict(
          torch.load(PRETRAINED_MODEL_PATH, map_location=self.device)
      )
    else:
      warnings.warn(
          'Pre-trained encoder not found for fine-tuning. Initializing encoder'
          ' with random weights.'
      )

    # Re-initialize the quantile head if needed, or ensure it's on the correct device
    self.quantile_head = QuantileRegressionHead(
        HIDDEN_SIZE, NUM_PRED_POINTS, NUM_QUANTILE_OUTPUTS
    ).to(self.device)
    self.model = FluForecastModel(
        self.encoder, self.pretrain_head, self.quantile_head
    ).to(self.device)

    optimizer = optim.Adam([
        {
            'params': self.model.encoder.parameters(),
            'lr': 1e-4,
        },  # Lower LR for pre-trained encoder
        {
            'params': self.model.quantile_head.parameters(),
            'lr': 1e-3,
        },  # Higher LR for new head
    ])

    finetune_inputs = []
    finetune_targets_scaled = []

    min_data_for_sample = SERIES_LENGTH + NUM_PRED_POINTS - 1

    # 1. Add real target data for fine-tuning from current fold's train_df
    for loc in active_locations:
      loc_df = (
          train_df_current_fold[train_df_current_fold['location'] == loc]
          .copy()
          .sort_values('target_end_date')
      )

      if len(loc_df) < SERIES_LENGTH:
        continue

      for i in range(len(loc_df) - SERIES_LENGTH + 1):
        input_series_values = loc_df.iloc[i : i + SERIES_LENGTH][
            TARGET_STR
        ].values

        # Check if input series is all zeros or has too little sum
        if (
            np.sum(input_series_values) < 1.0
        ):  # Filter out input series that are effectively all zeros
          continue

        scaled_input_series = self.target_global_scaler.transform(
            input_series_values.reshape(-1, 1)
        ).flatten()

        target_values_for_all_horizons = np.full(NUM_PRED_POINTS, np.nan)
        for h_idx in range(NUM_PRED_POINTS):
          target_idx_abs = i + SERIES_LENGTH - 1 + h_idx
          if target_idx_abs < len(loc_df):
            target_value = loc_df.iloc[target_idx_abs][TARGET_STR]
            scaled_target_value = self.target_global_scaler.transform(
                np.array([[target_value]])
            ).flatten()[0]
            target_values_for_all_horizons[h_idx] = scaled_target_value

        # We now allow NaNs in targets, relying on loss function to handle them
        finetune_inputs.append(scaled_input_series)
        finetune_targets_scaled.append(target_values_for_all_horizons)

    # 2. Add synthetic target data for fine-tuning from ILINet (using fold-specific transformers)
    ilinet_grouped_for_synthetic = self._processed_ilinet_state.groupby(
        ['location', 'season_year']
    )

    for (loc, s_year), group in ilinet_grouped_for_synthetic:
      if loc not in self._ilinet_to_target_transformers_fold:
        continue

      ilinet_scaler = self.ilinet_per_location_scalers.get(loc)
      scaled_ilinet_series = series_to_fixed_vector_and_scale(
          group, 'ilitotal', s_year, loc, scaler=ilinet_scaler
      )

      if np.sum(scaled_ilinet_series) == 0:
        continue

      transformer = self._ilinet_to_target_transformers_fold.get(loc)
      synthetic_scaled_series = np.zeros_like(scaled_ilinet_series)

      if isinstance(transformer, LinearRegression):
        synthetic_scaled_series = transformer.predict(
            scaled_ilinet_series.reshape(-1, 1)
        ).flatten()
      elif isinstance(transformer, dict):
        if transformer['type'] == 'ratio':
          synthetic_scaled_series = scaled_ilinet_series * transformer['ratio']
        elif transformer['type'] == 'fixed_zero':
          synthetic_scaled_series = np.zeros_like(scaled_ilinet_series)

      synthetic_scaled_series = np.clip(
          synthetic_scaled_series,
          self.target_global_scaler.feature_range[0],
          self.target_global_scaler.feature_range[1],
      )

      if len(synthetic_scaled_series) >= SERIES_LENGTH:
        for i in range(len(synthetic_scaled_series) - SERIES_LENGTH + 1):
          input_window = synthetic_scaled_series[i : i + SERIES_LENGTH]

          target_values_for_all_horizons = np.full(NUM_PRED_POINTS, np.nan)
          for h_idx in range(NUM_PRED_POINTS):
            target_idx_abs = i + SERIES_LENGTH - 1 + h_idx
            if target_idx_abs < len(synthetic_scaled_series):
              target_values_for_all_horizons[h_idx] = synthetic_scaled_series[
                  target_idx_abs
              ]

          # Only add if at least one target is valid (not NaN)
          if not np.all(np.isnan(target_values_for_all_horizons)):
            finetune_inputs.append(input_window)
            finetune_targets_scaled.append(target_values_for_all_horizons)

    if not finetune_inputs:
      warnings.warn(
          'No sufficient target (real or synthetic) data for fine-tuning in'
          ' this fold.'
      )
      self.model = None  # Indicate that fine-tuning failed
      return

    finetune_inputs_tensor = torch.tensor(
        np.array(finetune_inputs), dtype=torch.float32
    ).to(self.device)
    finetune_targets_tensor = torch.tensor(
        np.array(finetune_targets_scaled), dtype=torch.float32
    ).to(self.device)
    quantiles_tensor = (
        torch.tensor(QUANTILES, dtype=torch.float32)
        .unsqueeze(0)
        .to(self.device)
    )

    finetune_dataset = TensorDataset(
        finetune_inputs_tensor, finetune_targets_tensor
    )
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=32, shuffle=True
    )

    finetune_epochs = 20

    self.model.train()  # Set model to training mode
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      for epoch in range(finetune_epochs):
        for batch_idx, (inputs, targets) in enumerate(finetune_dataloader):
          optimizer.zero_grad()
          predictions = self.model.forward_finetune(inputs)
          loss = pinball_loss_multi_horizon(
              predictions, targets, quantiles_tensor
          )
          loss.backward()
          optimizer.step()
    self.model.eval()  # Set model to evaluation mode

  def predict(self, test_x):
    """Generates quantile predictions for the given test_x DataFrame."""
    output_df = pd.DataFrame(
        index=test_x.index, columns=[f'quantile_{q}' for q in QUANTILES]
    )

    if self.model is None:  # Fallback if fine-tuning failed
      for q_col in output_df.columns:
        output_df[q_col] = 0
      return output_df

    # Combine _raw_dataset with current train_x/train_y if needed for history context
    # In a rolling window, train_x/train_y contains the actual historical data up to reference_date-1.
    # So we use the latest _processed_dataset which implicitly contains this.
    current_fold_history = self._processed_dataset.copy()
    current_fold_history['target_end_date'] = pd.to_datetime(
        current_fold_history['target_end_date']
    )
    current_fold_history = current_fold_history.sort_values(
        ['location', 'target_end_date']
    )

    with torch.no_grad():
      test_x_grouped = test_x.groupby(['reference_date', 'location'])

      for (ref_date_str, loc), group_df in test_x_grouped:
        ref_date = pd.to_datetime(ref_date_str)
        input_end_date_for_pred = ref_date - pd.Timedelta(weeks=1)

        history_for_prediction = current_fold_history[
            (current_fold_history['location'] == loc)
            & (
                current_fold_history['target_end_date']
                <= input_end_date_for_pred
            )
        ]

        all_predictions_for_group_scaled = np.zeros(
            (NUM_PRED_POINTS, NUM_QUANTILE_OUTPUTS)
        )

        if (
            not history_for_prediction.empty
            and len(history_for_prediction) >= SERIES_LENGTH
        ):
          input_series_values = history_for_prediction.tail(SERIES_LENGTH)[
              TARGET_STR
          ].values

          if np.sum(input_series_values) < 1.0:
            warnings.warn(
                f'Input series for prediction at {ref_date}, location {loc} is'
                ' all zeros. Predicting zeros for all horizons.'
            )
          else:
            scaled_input_series = self.target_global_scaler.transform(
                input_series_values.reshape(-1, 1)
            ).flatten()
            test_input_tensor = (
                torch.tensor(scaled_input_series, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            all_predictions_for_group_scaled = (
                self.model.forward_finetune(test_input_tensor)
                .cpu()
                .numpy()
                .squeeze(0)
            )
        else:
          warnings.warn(
              f'Insufficient history for prediction at {ref_date}, location'
              f' {loc}. Predicting zeros for all horizons.'
          )

        for idx, row in group_df.iterrows():
          horizon_val = row['horizon']
          lead_time_idx = HORIZON_TO_LEAD_TIME_IDX[horizon_val]

          predictions_for_this_horizon = all_predictions_for_group_scaled[
              lead_time_idx, :
          ]

          predictions = self.target_global_scaler.inverse_transform(
              predictions_for_this_horizon.reshape(-1, 1)
          ).flatten()

          predictions[predictions < 0] = 0
          predictions = np.maximum.accumulate(
              predictions
          )  # Enforce monotonicity
          predictions = np.round(predictions).astype(int)

          for i, q in enumerate(QUANTILES):
            output_df.loc[idx, f'quantile_{q}'] = predictions[i]
    return output_df


# The wrapper function to manage the singleton instance
_flu_forecaster_instance = None  # Singleton instance


def fit_and_predict_fn(
    train_x,
    train_y,
    test_x,
):
  global _flu_forecaster_instance
  # Access global dataframes from the preamble once, on first call.
  global locations, dataset, ilinet_state, ilinet_hhs, ilinet

  # Initialize the model instance only once across all calls to fit_and_predict_fn
  if _flu_forecaster_instance is None:
    print('Initializing FluForecaster model...')
    # Pass raw dataframes to the FluForecaster constructor
    _flu_forecaster_instance = FluForecaster(
        locations.copy(), dataset.copy(), ilinet_state.copy()
    )
    print('FluForecaster model initialized.')

  # Fit the model for the current fold (learn transformation and fine-tune)
  ref_date = test_x['reference_date'].iloc[0] if not test_x.empty else 'N/A'
  print(f'Fitting FluForecaster model for reference_date: {ref_date}...')
  _flu_forecaster_instance.fit(train_x.copy(), train_y.copy())
  print('FluForecaster model fit complete.')

  # Make predictions for the current test set
  print('Generating predictions...')
  predictions_df = _flu_forecaster_instance.predict(test_x.copy())
  print('Predictions generated.')

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
