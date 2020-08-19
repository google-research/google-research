# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Custom formatting functions for Favorita dataset.

Defines dataset specific column definitions and data transformations.
"""
import random

from libs.data_utils import read_csv, write_csv

random.seed(42)

import data_formatters.base
import libs.utils as utils
import numpy as np
np.random.seed(42)
import pandas as pd
import sklearn.preprocessing

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class CGFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the Favorita dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('campaign_id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('date', DataTypes.DATE, InputTypes.TIME),
      ('log1p_conversions_campaign', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('log1p_spend_campaign', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log1p_impressions_campaign', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log1p_clicks_campaign', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('num_adsets', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('campaign_age', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('is_missing', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('present', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('dow_jones_index', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('data_connector_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def _split_series(self, s, time_steps, horizon):
    # train + valid + test
    if len(s) >= time_steps + 2 * horizon:
      return {
        'train': s.iloc[:-2 * horizon, :],
        'valid': s.iloc[-horizon - time_steps: -horizon, :],
        'test': s.iloc[-time_steps:, :],
      }
    # train + valid / test
    elif len(s) >= time_steps + horizon:
      key = 'test' if random.random() > 0.5 else 'valid'
      return {
        'train': s.iloc[: -horizon, :],
        key: s.iloc[-time_steps:, :],
      }
    # train
    elif len(s) >= time_steps:
      return {
        'train': s,
      }
    else:
      return {}

  def _split_series_1(self, s, time_steps, horizon):
    if len(s) >= time_steps + horizon:
      if random.random() > 0.5:
        return self._split_series(s, time_steps, horizon)
    key = np.random.choice(['train', 'valid', 'test'], p=[0.7, 0.15, 0.15])
    return {
      key: s,
    }

  def _train_valid_test_split(self, df, time_steps, forecast_horizon):
    df_lists = {'train': [], 'valid': [], 'test': []}
    for _, sliced in df.groupby('campaign_id'):
      sliced = sliced.sort_values('date')
      parts = self._split_series_1(sliced, time_steps, forecast_horizon)
      for k, v in parts.items():
        df_lists[k].append(v)

    for k, v in df_lists.items():
      print(f"{k} size in {len(v)}")

    dfs = {}
    for k in df_lists:
      sub_df = pd.concat(df_lists[k], axis=0).reset_index(drop=True)
      campaigns = sub_df[['campaign_id', 'date']].drop_duplicates()
      dfs[k] = campaigns

    return dfs

  def _load_or_perform_split(self, df, time_steps, forecast_horizon, config):
    filenames = {
      'train': 'train_campaigns.csv',
      'valid': 'valid_campaigns.csv',
      'test': 'test_campaigns.csv',
    }
    train_campaigns = read_csv(filenames['train'], config)
    if train_campaigns is None:
      campaigns = self._train_valid_test_split(df, time_steps, forecast_horizon)
      for k in campaigns:
        write_csv(campaigns[k], filenames[k], config, to_csv_kwargs={'index': False})
    else:
      campaigns = {
        k: read_csv(v, config)
        for k, v in filenames.items()
      }
    for k, v in campaigns.items():
      print(f"{k} size is {len(v)} with {v['campaign_id'].nunique()} campaigns")
    return campaigns

  def split_data(self, df, config=None, valid_n_days=15, test_n_days=15):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    fixed_params = self.get_fixed_params()
    time_steps = fixed_params['total_time_steps']
    lookback = fixed_params['num_encoder_steps']
    forecast_horizon = time_steps - lookback

    df['date'] = pd.to_datetime(df['date'])

    campaigns = self._load_or_perform_split(df, time_steps, forecast_horizon, config)
    train = df.merge(campaigns['train'], on=['campaign_id', 'date'])
    valid = df.merge(campaigns['valid'], on=['campaign_id', 'date'])
    test = df.merge(campaigns['test'], on=['campaign_id', 'date'])

    self.set_scalers(train, set_real=True)

    # Use all data for label encoding  to handle labels not present in training.
    self.set_scalers(df, set_real=False)

    # # Filter out identifiers not present in training (i.e. cold-started items).
    # def filter_ids(frame):
    #   identifiers = set(self.identifiers)
    #   index = frame['series_id']
    #   return frame.loc[index.apply(lambda x: x in identifiers)]
    #
    # valid = filter_ids(dfs['valid'])
    # test = filter_ids(dfs['test'])

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df, set_real=True):
    """Calibrates scalers using the data supplied.

    Label encoding is applied to the entire dataset (i.e. including test),
    so that unseen labels can be handled at run-time.

    Args:
      df: Data to use to calibrate scalers.
      set_real: Whether to fit set real-valued or categorical scalers
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    if set_real:

      # Extract identifiers in case required
      self.identifiers = list(df[id_column].unique())

      present_df = df[df["present"] == 1]

      # Format real scalers
      self._real_scalers = {}
      for col in [
          'log1p_conversions_campaign',
          'log1p_spend_campaign',
          'log1p_impressions_campaign',
          'log1p_clicks_campaign',
      ]:
        self._real_scalers[col] = (present_df[col].mean(), present_df[col].std())
      for col in [
        'dow_jones_index',
      ]:
        self._real_scalers[col] = (df[col].mean(), df[col].std())

      self._target_scaler = (present_df[target_column].mean(), present_df[target_column].std())

    else:
      # Format categorical scalers
      categorical_inputs = utils.extract_cols_from_data_type(
          DataTypes.CATEGORICAL, column_definitions,
          {InputTypes.ID, InputTypes.TIME})

      categorical_scalers = {}
      num_classes = []
      if self.identifiers is None:
        raise ValueError('Scale real-valued inputs first!')
      id_set = set(self.identifiers)
      valid_idx = df['campaign_id'].apply(lambda x: x in id_set)
      for col in categorical_inputs:
        # Set all to str so that we don't have mixed integer/string columns
        srs = df[col].apply(str)#.loc[valid_idx]
        categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
            srs.values)

        num_classes.append(srs.nunique())

      # Set categorical scaler outputs
      self._cat_scalers = categorical_scalers
      self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    for col in [
        'log1p_conversions_campaign',
        'log1p_spend_campaign',
        'log1p_impressions_campaign',
        'log1p_clicks_campaign',
        'dow_jones_index',
    ]:
      mean, std = self._real_scalers[col]
      output[col] = (df[col] - mean) / std

      if col == 'log1p_conversions_campaign':
        output.loc[output["present"] == 0, col] = -1

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def reverse_scale(self, df):
    # Format real inputs
    for col in [
      'log1p_conversions_campaign',
      'log1p_spend_campaign',
      'log1p_impressions_campaign',
      'log1p_clicks_campaign',
      'dow_jones_index',
    ]:
      mean, std = self._real_scalers[col]
      df[col] = df[col] * std + mean

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns
    mean, std = self._target_scaler
    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = (predictions[col] * std) + mean

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 30,
        'num_encoder_steps': 15,
        'num_epochs': 20,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 4
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 240,
        'learning_rate': 0.001,
        'minibatch_size': 128,
        'max_gradient_norm': 100.,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1
