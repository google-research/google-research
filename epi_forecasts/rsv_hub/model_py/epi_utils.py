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


"""Utilities functions for epi models.

Includes functions for date utilities, evaluation, formatting, and plotting,
as well as relevant constant definitions.
"""

import datetime
from typing import Callable, List, Set, Tuple
import warnings

import constant_defs
import numpy as np
import pandas as pd
from sklearn import isotonic


QUANTILES = constant_defs.QUANTILES
TARGET_STR = constant_defs.TARGET_STR
Timedelta = datetime.timedelta
Date = datetime.date


### DATE UTILITIES ###


def get_saturdays_between_dates(
    start_date_str, end_date_str
):
  """Generates a list of Saturday dates within a given date range."""
  date_range = pd.date_range(
      start=start_date_str, end=end_date_str, freq='D', inclusive='both'
  )
  saturdays = date_range[date_range.weekday == 5]
  return [d.date() for d in saturdays]


def get_next_saturday_date_str():
  """Calculates date of next upcoming Saturday, including today if relevant.

  Returns:
      str: The date of the next upcoming Saturday in 'YYYY-MM-DD' format.
  """
  today = Date.today()
  days_to_add = (5 - today.weekday()) % 7
  next_saturday = today + Timedelta(days=days_to_add)
  return next_saturday.strftime('%Y-%m-%d')


def get_most_recent_saturday_date_str():
  """Calculates date of most recent Saturday, including today if relevant.

  Returns:
      str: The date of the most recent Saturday in 'YYYY-MM-DD' format.
  """
  today = Date.today()
  days_to_subtract = (today.weekday() - 5) % 7
  most_recent_saturday = today - Timedelta(days=days_to_subtract)
  return most_recent_saturday.strftime('%Y-%m-%d')


### EVALUATION FUNCTIONS ###


def interval_score(
    observed, lower, upper, alpha
):
  if observed < lower:
    score = (upper - lower) + (2 / alpha) * (lower - observed)
  elif observed > upper:
    score = (upper - lower) + (2 / alpha) * (observed - upper)
  else:
    score = upper - lower
  return score


def weighted_interval_score(
    predicted,
    observed,
    quantile_level = QUANTILES,
    count_median_twice = True,
):
  """WIS is a proper scoring rule to evaluate forecasts in an interval or quantile format.

  Smaller values are better. (See Bracher et al., 2021)

  Args:
      predicted: Numeric nxN dataframe of predictive quantiles with n forecasts
        (same as number of observed values) and N quantiles per forecast. If
        `observed` is a single number, then predicted can just be an N-vector.
      observed: Numeric data of n observed values.
      quantile_level (np.array or list): Vector of size N with the quantile
        levels for which predictions were made.
      count_median_twice (bool, optional): If True, count the median twice in
        the score. Defaults to False.

  Returns:
      np.ndarray: a numeric vector with WIS values of size n (one per
      observation)
  """

  # If 'observed' is a single value (median) (WIS simplifies to absolute error)
  if observed.size == 1:
    predicted = predicted.reshape(1, -1)

  first_half_quantiles = sorted(quantile_level)[: len(quantile_level) // 2]
  k = len(first_half_quantiles)
  # WIS for each observation
  wis_scores = []
  for i in range(len(observed)):
    y = observed.iloc[i]
    predictions = predicted.iloc[i]
    weighted_interval_scores = []
    median_forecast = predictions.iloc[predictions.shape[0] // 2]
    weighted_interval_scores.append(0.5 * abs(y - median_forecast))
    if count_median_twice:
      weighted_interval_scores.append(0.5 * abs(y - median_forecast))

    for l_q in first_half_quantiles:
      # l: the lower bound of the interval
      # u: the upper bound of the interval
      u_q = round(1 - l_q, 3)
      alpha_i = 2 * l_q
      weighted_interval_scores.append(
          (alpha_i / 2)
          * (
              interval_score(
                  y,
                  predictions[f'quantile_{l_q}'],
                  predictions[f'quantile_{u_q}'],
                  alpha_i,
              )
          )
      )
    # Weighted sum to get WIS score for observation
    wis_scores.append(np.sum(np.array(weighted_interval_scores)))
  return 1 / (k + 0.5) * np.array(wis_scores)


def _create_fold_scaffold(
    reference_date,
    horizons,
    location_codes,
    locations_df,
):
  """Creates the feature scaffold for a single forecast date."""
  all_combinations = pd.MultiIndex.from_product(
      [[reference_date], horizons, location_codes],
      names=['reference_date', 'horizon', 'location'],
  )
  scaffold = pd.DataFrame(index=all_combinations).reset_index()
  scaffold['target_end_date'] = scaffold.apply(
      lambda row: row['reference_date'] + pd.Timedelta(weeks=row['horizon']),
      axis=1,
  )
  return scaffold.merge(locations_df, on='location', how='left')


def _process_single_fold(
    reference_date,
    observed_values,
    fit_and_predict_fn,
    horizons,
    location_codes,
    locations_df,
    encountered_warnings,
):
  """Processes a single fold: prepares data, runs prediction, and returns the forecast."""
  train_end_date = reference_date - pd.Timedelta(weeks=1)
  train = observed_values[observed_values['target_end_date'] <= train_end_date]
  train_x = train.drop(columns=[TARGET_STR])
  train_y = train[TARGET_STR]

  if train_x.empty or train_y.empty:
    print(
        f'Skipping fold for reference_date {reference_date} due to empty'
        ' training data.'
    )
    return pd.DataFrame()  # Return empty DF to signify a skipped fold

  test_x_i = _create_fold_scaffold(
      reference_date, horizons, location_codes, locations_df
  )

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    predicted_quantiles = fit_and_predict_fn(
        train_x.copy(), train_y.copy(), test_x_i.copy()
    )
    for warning_message in w:
      msg = str(warning_message.message)
      if msg not in encountered_warnings:
        print(f'Encountered new warning: {msg}')
        encountered_warnings.add(msg)

  forecast_df = test_x_i.copy()
  for col in predicted_quantiles.columns:
    forecast_df[col] = predicted_quantiles[col]

  return forecast_df


def compute_rolling_evaluation(
    observed_values,
    reference_dates,
    fit_and_predict_fn,
    horizons,
    location_codes,
    locations_df,
):
  """Generates forecasts and computes scores for a set of reference dates.

  Always returns a complete forecast DataFrame. The score will be NaN if no
  ground truth is available for scoring.

  Args:
      observed_values: DataFrame containing observed target values.
      reference_dates: A list of dates from which forecasts are made.
      fit_and_predict_fn: A callable function that takes training data (features
        and target) and test features, and returns predicted quantiles.
      horizons: A list of integer horizons (in weeks) to forecast.
      location_codes: An array of location codes to include in the forecasts.
      locations_df: DataFrame containing location information, used to merge
        with the scaffold.

  Returns:
      A tuple containing:
      - pd.DataFrame: A DataFrame with all generated forecasts.
      - float: The mean weighted interval score across all scorable folds, or
        NaN if no folds could be scored.
  """
  all_forecasts = []
  fold_scores = []
  encountered_warnings = set()

  for ref_date in reference_dates:
    forecast_df = _process_single_fold(
        ref_date,
        observed_values,
        fit_and_predict_fn,
        horizons,
        location_codes,
        locations_df,
        encountered_warnings,
    )
    if forecast_df.empty:
      continue

    all_forecasts.append(forecast_df)

    # Try to score by merging with ground truth
    data_for_scoring = pd.merge(
        forecast_df,
        observed_values[['target_end_date', 'location', TARGET_STR]],
        how='inner',  # Inner join filters to only rows with ground truth
        on=['target_end_date', 'location'],
    )

    if not data_for_scoring.empty:
      y_observed = data_for_scoring[TARGET_STR]
      y_predicted = data_for_scoring[[f'quantile_{q}' for q in QUANTILES]]
      scores_in_fold = weighted_interval_score(y_predicted, y_observed)
      fold_scores.append(np.mean(scores_in_fold))

  if not all_forecasts:
    print('No forecasts were generated. Returning an empty DataFrame.')
    return pd.DataFrame(), np.nan

  # Calculate final score, handling the case where no folds could be scored
  final_score = np.mean(fold_scores) if fold_scores else np.nan

  # Combine all forecast dataframes
  final_forecasts_df = pd.concat(all_forecasts, ignore_index=True)

  return final_forecasts_df, final_score


### FORMATTING FUNCTIONS ###


def ensure_monotonicity(df):
  """Ensures that the values in the 'value' column are monotonically increasing with respect to the 'output_type_id' column.

  Args:
      df (pd.DataFrame): DataFrame with 'output_type_id' and 'value' columns.

  Returns:
      pd.DataFrame: DataFrame with 'value' column adjusted to ensure
      monotonicity.
  """
  df = df.sort_values(by='output_type_id')
  y_ = df['value'].values
  quantiles_numeric = df['output_type_id'].values

  # Ensure non-negative, increasing
  ir = isotonic.IsotonicRegression(
      y_min=0, increasing=True, out_of_bounds='clip'
  )
  df['value'] = ir.fit_transform(quantiles_numeric, y_)
  return df


def format_for_cdc(
    submission,
    target_col = 'wk inc flu hosp',
):
  """Formats a forecast DataFrame into the CDC's required format.

  The CDC requires a DataFrame with separate rows for each quantile. the
  following columns:
  - reference_date: The date of the forecast.
  - horizon: The horizon of the forecast.
  - location: The location of the forecast.
  - target_end_date: The date of the target.
  - output_type: The type of the output.
  - output_type_id: The id of the output type.
  - value: The value of the output.

  Args:
      submission: DataFrame with forecast data.
      target_col: The name of the target column.

  Returns:
      DataFrame: CDC-formatted forecast DataFrame.
  """
  id_vars = [
      'reference_date',
      'horizon',
      'location',
      'target_end_date',
  ]
  value_vars = [f'quantile_{q}' for q in QUANTILES]

  cdc_df = pd.melt(
      submission,
      id_vars=id_vars,
      value_vars=value_vars,
      var_name='output_type_id',
      value_name='value',
  )

  cdc_df = cdc_df.rename(columns={'abbreviation': 'location'})
  cdc_df['target'] = target_col
  cdc_df['output_type'] = 'quantile'
  cdc_df['output_type_id'] = cdc_df['output_type_id'].str.replace(
      'quantile_', ''
  )
  cdc_df['output_type_id'] = cdc_df['output_type_id'].astype(float).round(3)
  cdc_df['location'] = cdc_df['location'].astype(str).str.zfill(2)
  cdc_df['value'] = pd.to_numeric(cdc_df['value']).clip(lower=0)

  cdc_monotonic_df = (
      cdc_df.groupby(
          ['reference_date', 'target', 'horizon', 'target_end_date', 'location']
      )
      .apply(ensure_monotonicity)
      .reset_index(drop=True)
  )
  diff_df = pd.merge(
      cdc_df,
      cdc_monotonic_df,
      on=[
          'reference_date',
          'target',
          'horizon',
          'target_end_date',
          'location',
          'output_type',
          'output_type_id',
      ],
      how='inner',
      suffixes=('_cdc', '_monotonic'),
  )
  diff_df['value_diff'] = np.abs(
      diff_df['value_monotonic'] - diff_df['value_cdc']
  )
  print(
      'difference between original forecasts and monotonic version:'
      f' {diff_df.value_diff.sum()}'
  )

  return cdc_monotonic_df[[
      'reference_date',
      'target',
      'horizon',
      'target_end_date',
      'location',
      'output_type',
      'output_type_id',
      'value',
  ]]
