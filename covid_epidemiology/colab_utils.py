# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Common utilities for colab notebooks."""

import collections
import datetime
import functools
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# pylint: disable=g-import-not-at-top
# Only import version_utils if we are in a colab
if 'google.colab' in sys.modules:
  from cloud_covid19_forecasting.etl.tools import version_utils
# pylint: enable=g-import-not-at-top

BQ_TABLES = {
    'japan': 'covid_prod_features.jp_prefecture_ts',
    'state': 'covid_prod_features.us_state_ts',
    'county': 'covid_prod_features.us_county_ts',
}

# pylint: disable=line-too-long
GT_FIELD_NAMES = {
    'japan':
        'kaz_deaths,open_gt_jp_deaths,kaz_confirmed_cases,open_gt_jp_confirmed_cases',
    'state':
        'jhu_state_confirmed_cases,jhu_state_deaths',
    'county':
        'jhu_county_confirmed_cases,jhu_county_deaths',
}
# pylint: enable=line-too-long

LOC_NAME = {
    'japan': 'pref',
    'state': 'state_code',
    'county': 'geo_id',
}


def calculate_mape_apply_fn(row,
                            average_type,
                            expected_num_locations,
                            min_count=None,
                            min_mae=None,
                            time_horizon=27,
                            debug=False,
                            value_type='cumulative'):
  """Calculate MAPE, depending on various flags.

  From
  'https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin':
  "A macro-average will compute the metric independently for each class and then
  take the average (hence treating all classes equally), whereas a micro-average
  will aggregate the contributions of all classes to compute the average
  metric."

  Args:
    row: A pd.Series. Should all have the same date.
    average_type: Python string. Either 'micro' or 'macro'.
    expected_num_locations: Expected number of locations.
    min_count: If not `None`, ignore values with max(gt) < min_count. Should be
      `None` if `average_type='micro'`. Only one of `min_count` and `min_mae`
      should be not `None.
    min_mae: If not `None`, ignore values with `mae < min_mae`. Should be `None`
      if `average_type='micro'`. Only one of `min_count` and `min_mae` should be
      not `None.
    time_horizon: The time horizon value to compare cumulative metrics on. The
      end of the forecast window is usually 28 days, but Reichlab uses 27 days,
      so we use that as the default. Note that `time_horizon` is 1-indexed,
      while Python arrays are 0-indexed.
    debug: Python bool. Whether to print verbose debug info.
    value_type: Python string. Describes the values to use for MAPE calculation.

  Returns:
    The appropriate MAPE score.
  """
  assert average_type in ['micro', 'macro']
  assert expected_num_locations in [47, 51]
  assert value_type in ['cumulative', '4week', 'weekly']
  if average_type == 'micro':
    assert min_count is None, min_count
    assert min_mae is None, min_mae
  assert (min_count is None) or (min_mae is None), (min_count, min_mae)
  if 'forecast_date' in row:
    assert len(row.forecast_date.unique()) == 1, row
    cur_forecast_date = row.forecast_date.unique()[0]
  row_dat = row[['location_name', 0]]
  assert len(row_dat) == expected_num_locations, len(row_dat)

  # For macro average.
  mapes = []
  # For micro average.
  total_pred = None
  total_gt = None

  for _, loc_dat in row_dat.iterrows():
    cur_location_name = loc_dat.location_name
    cur_dat_dict = loc_dat[0]
    assert isinstance(cur_location_name, str), cur_location_name
    if not isinstance(cur_dat_dict, dict):
      assert np.isnan(cur_dat_dict), cur_dat_dict
      print(f'{cur_forecast_date} {cur_location_name} had no data, so '
            'continuing with date...')
      continue
    # Time horizon is 1-indexed, but arrays are 0-index, so subtract 1.
    assert time_horizon <= len(cur_dat_dict['predictions'])
    if value_type == '4week':
      assert 'day_zero_gt' in cur_dat_dict, cur_dat_dict
      base_gt = cur_dat_dict['day_zero_gt']
      assert base_gt is not None, cur_dat_dict

      # The total increase over the last time window is
      # `dat[-1] - day_0_gt`, so we use this value in the "4week" case.
      # We use `day_0_gt` (rather than `gt[0]` in the forecast window) so that
      # incident cases on every day of the forecast window is possibly non-zero.
      pred_ys = (
          cur_dat_dict['predictions'][time_horizon - 1:time_horizon] - base_gt)
      gtru_ys = (
          cur_dat_dict['ground_truth'][time_horizon - 1:time_horizon] - base_gt)
    elif value_type == 'cumulative':
      # Since we are computing MAPE over cumulative values, we only care about
      # the values at the end of the forecast period.
      pred_ys = cur_dat_dict['predictions'][time_horizon - 1:time_horizon]
      gtru_ys = cur_dat_dict['ground_truth'][time_horizon - 1:time_horizon]
    else:
      assert value_type == 'weekly'
      # TODO(joelshor): We ignore `time_horizon`. Consider whether to use it.
      # Weekly is i=0,1,2,3, `dat[i*7 + 6] - dat[i*7]`, averaged over i.
      # However, since we only have 27 days into the future for GT, we modify
      # the last day.
      preds = cur_dat_dict['predictions']
      pred_ys = np.array([preds[7 * i + 6] - preds[7 * i] for i in range(3)] +
                         [preds[7 * 3 + 5] - preds[7 * 3]])
      gtrs = cur_dat_dict['ground_truth']
      gtru_ys = np.array([gtrs[7 * i + 6] - gtrs[7 * i] for i in range(3)] +
                         [gtrs[7 * 3 + 5] - gtrs[7 * 3]])
    if debug:
      print(f'{cur_forecast_date} {cur_location_name}: '
            f'gt {gtru_ys} pred {pred_ys}')

    if average_type == 'micro':
      if total_pred is None and total_gt is None:
        total_pred = np.zeros_like(pred_ys)
        total_gt = np.zeros_like(gtru_ys)
      assert isinstance(total_pred, np.ndarray)
      assert isinstance(total_gt, np.ndarray)
      total_pred += pred_ys
      total_gt += gtru_ys
      continue

    assert average_type == 'macro'
    assert isinstance(pred_ys, np.ndarray), type(pred_ys)
    assert isinstance(gtru_ys, np.ndarray), type(gtru_ys)
    assert pred_ys.size == (4 if value_type == 'weekly' else 1), pred_ys.size
    assert gtru_ys.size, gtru_ys.size

    cur_mape = calculate_mape(
        predictions=pred_ys.tolist(), ground_truths=gtru_ys.tolist())
    if debug:
      print(f'{cur_forecast_date} {cur_location_name}: MAPE: {cur_mape}')

    if min_count and max(gtru_ys) < min_count:
      mapes.append(np.nan)
    elif min_mae:
      cur_mae = calculate_mae(
          predictions=pred_ys.tolist(), ground_truth=gtru_ys.tolist())
      if cur_mae < min_mae:
        mapes.append(np.nan)
      else:
        mapes.append(cur_mape)
    else:
      mapes.append(cur_mape)

  if average_type == 'micro':
    if total_pred is None:
      return np.nan
    else:
      assert isinstance(total_pred, np.ndarray), total_pred
      assert isinstance(total_gt, np.ndarray), total_gt
      # We are only looking at the end of the forecast window, so it should be
      # size 1 if not weekly..
      assert total_pred.size == (4 if value_type == 'weekly' else
                                 1), total_pred.size
      assert total_gt.size == total_pred.size, total_gt.size
      avg_pred = total_pred / expected_num_locations
      avg_gt = total_gt / expected_num_locations
      cur_mape = calculate_mape(
          predictions=avg_pred.tolist(), ground_truths=avg_gt.tolist())
      return cur_mape
  else:
    if len(mapes) == 0:  # pylint:disable=g-explicit-length-test
      return np.nan
    else:
      assert average_type == 'macro', average_type
      # With low death counts, some prefectures are dropped, so this assert
      # isn't always helpful.
      # assert len(mapes) == expected_num_locations, len(mapes)
      return np.nanmean(mapes)


def get_gt(loc,
           start_date,
           end_date,
           feature_name,
           locale,
           bq_client,
           version = None,
           capitalize = True):
  """Get ground truth in a colab."""
  assert locale in ['japan', 'state', 'county'], locale

  assert feature_name in GT_FIELD_NAMES[locale], \
      f'{feature_name} vs {GT_FIELD_NAMES[locale]}'

  bq_table = BQ_TABLES[locale]
  # Get the proper version.
  q = f'select DISTINCT(version) from `{bq_table}`'
  versions = bq_client.query(q).to_dataframe()
  if version:
    if version not in versions:
      raise ValueError(f'Version not found: {version} vs {versions}')
  else:
    # Get latest GT data ex "2020-07-18 19:59:42 UTC"
    version = versions.max()[0].strftime('%Y-%m-%d %H:%M:%S %Z')

  loc_field = LOC_NAME[locale]
  if capitalize:
    loc = loc.capitalize()
  q = (f'select dt,{loc_field},feature_name,feature_value '
       f'from `{bq_table}` '
       f'where dt >= "{start_date}" and dt <= "{end_date}" '
       f'and {loc_field} = "{loc}"'
       f'and feature_name = "{feature_name}" '
       f'and version = "{version}"')
  gt_pd = bq_client.query(q).to_dataframe()
  assert gt_pd.size > 0, q
  xs, ys = [], []
  for d, v in gt_pd.sort_values(by='dt')[['dt', 'feature_value']].values:
    xs.append(d)
    ys.append(v)
  return xs, ys


def get_all_gt(start_date,
               end_date,
               locale,
               bq_client,
               version = None,
               feature_keys = None):
  """Return all ground truth during a certain date."""
  assert locale in ['japan', 'state', 'county'], locale

  bq_table = BQ_TABLES[locale]
  # Get the proper version.If `version` is `None`, select latest.
  # If `version` is a string, use that version.
  # If `version` is a list, get all those versions.
  q = f'select DISTINCT(version) from `{bq_table}`'
  versions = bq_client.query(q).to_dataframe()
  if version is None:
    # Get latest GT data ex "2020-07-18 19:59:42 UTC"
    version = versions.max()[0].strftime('%Y-%m-%d %H:%M:%S %Z')
  elif isinstance(version, str):
    if not np.any(str(version) == versions):
      raise ValueError(f'Version not found: {version} vs {versions.to_numpy()}')
  else:
    assert isinstance(version, list)

  loc_field = LOC_NAME[locale]
  if isinstance(version, list):
    v_str = ','.join([f'"{x}"' for x in version])
    version_q = f'and version IN ({v_str})'
  else:
    version_q = f'and version = "{version}"'
  if feature_keys is not None:
    feat_str = ','.join([f'"{x}"' for x in feature_keys])
    feature_q = f' and feature_name IN ({feat_str})'
  else:
    feature_q = ''
  q = (f'select dt,{loc_field},feature_name,feature_value,version '
       f'from `{bq_table}` '
       f'where dt >= "{start_date}" and dt <= "{end_date}" '
       f' {version_q}{feature_q}')
  print(f'Querying GT with: "{q}"')
  gt_pd = bq_client.query(q).to_dataframe()
  assert gt_pd.size > 0, q

  if locale == 'japan':
    # Change data to standard spelling eg "Hyōgo" -> "Hyogo". The prefecture
    # names changed when we switched Japan data sources, so both names are
    # present in the GT tables.
    gt_pd = gt_pd.replace({loc_field: open_covid_locations_to_kaz_map()})

  return gt_pd


def get_gt_over_dates_and_versions(dates_and_gt_versions,
                                   locale,
                                   client,
                                   gt_feature_name,
                                   duration_days=28):
  """Gets a single GT dataframe with the dates and versions requested."""
  assert locale in ['japan', 'state'], locale

  # Minimize the number of reads by grouping reads of the same version.
  def _end_date_from_str(start_date):
    return (datetime.datetime.strptime(start_date, '%Y-%m-%d').date() +
            datetime.timedelta(days=duration_days)).isoformat()

  version_map = collections.defaultdict(list)
  for date, version in dates_and_gt_versions:
    version_map[version].append(date)
  version_map_list = [(v, min(dates), _end_date_from_str(max(dates)))
                      for v, dates in version_map.items()]

  gt_dfs = []
  for version, start_date, end_date in version_map_list:
    # Read GT slice.
    print(f'Reading GT from {start_date} to {end_date}, version {version}...')
    gt_df = get_all_gt(
        start_date=start_date,
        end_date=end_date,
        locale=locale,
        bq_client=client,
        feature_keys=[gt_feature_name],
        version=version)
    gt_dfs.append(gt_df)
  gt_df = pd.concat(gt_dfs, axis=0).drop_duplicates()
  return gt_df


def get_gt_version_names(dates, version_locale, min_version, use_latest_gt,
                         client):
  """Returns either latest version or T+28 data version."""
  assert isinstance(dates, list), type(dates)
  if use_latest_gt:
    gt_version = datetime.datetime.strptime(
        get_latest_version(client).name, '%Y-%m-%d %H:%M:%S %Z')
    gt_versions = [gt_version for _ in range(len(dates))]
  else:
    if isinstance(dates[0], str):
      dates = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    gt_versions = data_version_for_retrospective(
        dates,
        min_version=min_version,
        locale={
            'japan': 'jp_prefecture',
            'state': 'us_state'
        }[version_locale],
        client=client)
  return [
      gt_version.strftime('%Y-%m-%d %H:%M:%S+00:00')
      for gt_version in gt_versions
  ]


def get_public_forecast(
    forecast_pd,
    loc,
    cur_date,
    feature_key,
    loc_key,
    expeced_forecast_len = 28,
    prediction_date_key = 'prediction_date'):
  """Extracts a single prediction from the historical forecasts table."""
  assert loc_key in forecast_pd.columns, (loc_key, forecast_pd.columns)
  assert prediction_date_key in forecast_pd.columns, forecast_pd.columns
  assert feature_key in forecast_pd.columns, (feature_key, forecast_pd.columns)

  forecast = forecast_pd[(forecast_pd[loc_key] == loc)
                         & (forecast_pd.forecast_date == cur_date) &
                         (forecast_pd[prediction_date_key] > cur_date)]
  assert forecast.size > 0, (loc, cur_date, feature_key, loc_key)
  forecast_vals = forecast[[prediction_date_key,
                            feature_key]].sort_values(by=prediction_date_key)
  xs = forecast_vals[prediction_date_key].to_numpy()
  ys = forecast_vals[feature_key].astype(np.float32).to_numpy()
  assert isinstance(xs[0], datetime.date), type(xs[0])
  assert xs.size == expeced_forecast_len, f'xs size: {xs.size}'
  assert ys.size == expeced_forecast_len, f'xs size: {ys.size}'
  return xs, ys


def calculate_mape(predictions, ground_truths):
  """Calculate MAPE in a colab friendly way."""
  if not predictions or len(predictions) != len(ground_truths):
    raise ValueError(
        'Predictions and Ground Truth should be of equal length and non-empty')

  error = 0.
  num_nonzero_ground_truth = 0
  for prediction, ground_truth in zip(predictions, ground_truths):
    if ground_truth != 0:
      error += abs((prediction - ground_truth) / ground_truth)
      num_nonzero_ground_truth += 1

  if num_nonzero_ground_truth == 0:
    return float('nan')

  return 100 * error / num_nonzero_ground_truth


def calculate_mae(predictions, ground_truth):
  """Calculate MAE in a colab friendly way."""
  if not predictions or len(predictions) != len(ground_truth):
    raise ValueError(
        'Predictions and Ground Truth should be of equal length and non-empty')

  error = 0.
  for i in range(len(predictions)):
    error += abs(predictions[i] - ground_truth[i])

  error /= len(predictions)

  return error


LOCALES = ['country', 'us_county', 'us_state', 'jp_prefecture']


@functools.lru_cache(maxsize=32)
def _read_data_versions(bq_client, locale):
  """Wrapper around version table for easy caching."""
  return version_utils.Client(bq_client).read_data_versions(
      dataset_name='covid_prod_features', table_name=f'{locale}_ts')


def get_closest_versions(ds,
                         locale,
                         bq_client = None,
                         buffer_days = 0):
  """Returns last first version stricly after given date + buffer days."""
  if locale not in LOCALES:
    raise ValueError(f'Locale not recognized: {locale} not in {LOCALES}')
  vs = _read_data_versions(bq_client, locale)
  vs = [(version_utils.version_name_to_datetime(v).date(), v) for v in vs]

  vs = sorted(vs)
  rets = []
  for d in ds:
    if d < vs[0][0]:
      rets.append(vs[0][1])
      continue
    if d >= vs[-1][0]:
      rets.append(vs[-1][1])
      continue

    for i in range(len(vs)):
      # Don't break on equality, since we want strictly less than.
      if vs[i][0] > d + datetime.timedelta(days=buffer_days or 0):
        break
    rets.append(vs[i][1])

  return rets


def get_latest_version(bq_client):
  return version_utils.Client(bq_client).get_latest_version()


def version_or_min_version(version, min_version):
  """Pick version or min version."""
  if (version_utils.version_name_to_datetime(version) <=
      version_utils.version_name_to_datetime(min_version)):
    return min_version
  else:
    return version


def data_version_for_retrospective(dates,
                                   min_version,
                                   locale,
                                   client,
                                   days_ahead = 28):
  """Get the GT data version associated with a particular time period.

  Args:
    dates: List of dates to fetch values for. This is parallelized for speed.
    min_version: Japan has a data minimum version, so use that if the requested
      version is too early.
    locale: String for locale.
    client: BQ Client for reading tables.
    days_ahead: Minimum number of days to include. If, for example, we want 28
      days ahead, we should look ahead (28-1) = 27 days. Some days in the
      prospective period actually require 28 day lookahead for some reason,
      however, so we rely on the automatic increment mechanism later in the
      code.

  Returns:
    List of version datetimes.
  """
  version_dates = [d + datetime.timedelta(days=days_ahead) for d in dates]
  version_strs = get_closest_versions(version_dates, locale, client)
  if min_version:
    version_strs = [
        version_or_min_version(v_str, min_version=min_version)
        for v_str in version_strs
    ]
  return [
      version_utils.version_name_to_datetime(v_str) for v_str in version_strs
  ]


def trial_name_to_df(trial_name,
                     predicted_metric,
                     client,
                     num_locs=47,
                     forecast_len=28):
  """Reads a trial name forecasts. Used for fetching retrospective results."""
  assert predicted_metric in ['death', 'confirmed']

  # Read from the table and make sure the output is correct.
  all_pd = client.query(f'select * from `eval.{trial_name}`').to_dataframe()
  cur_pd = all_pd[all_pd.predicted_metric == predicted_metric][[
      'location_name', 'time_horizon', 'point_prediction',
      'target_prediction_date'
  ]]
  assert len(cur_pd) == num_locs * forecast_len, (
      len(cur_pd), all_pd.predicted_metric.unique())

  def _sort(x):
    x = x.sort_values(by='time_horizon')
    return {
        'predictions': x.point_prediction.values,
        'prediction_date': x.target_prediction_date.dt.date.values,
    }

  preds_pd = cur_pd.groupby('location_name').apply(_sort)
  assert len(preds_pd) == num_locs, len(preds_pd)

  target_prediction_dates = sorted(cur_pd['target_prediction_date'].unique())
  assert len(target_prediction_dates) == forecast_len

  return preds_pd, target_prediction_dates


def gather_data_from_prospective_row(row,
                                     gt_df,
                                     locale,
                                     available_versions,
                                     expected_forecast_len = 28,
                                     expected_gt_len = 27,
                                     debug = False):
  """Pandas `apply` fn to gather GT and preds from data.

  We use this function to parallelize using the pandas groupby.
  With this optimization (compared to a for-loop), this function goes from
  1.2 min -> 0.2 on Japan.
  2.0 min -> XXX on US.

  Args:
    row:
    gt_df:
    locale:
    available_versions: A numpy array of available versions in `gt_df`. The
      difference between precomputing this and doing it on-the-fly for each row
      is about 10 minutes of compute over the prospective window, so we pass
      this in precomputed as a speedup.
    expected_forecast_len: Expected length of the forecast.
    expected_gt_len: Expected length of GT. Doesn't have to match
      `expected_forecast_len` due to how forecasts are measured.
    debug:

  Returns:
    {'dates', 'predictions', 'ground_truth'}
  """
  assert 'predictions' in row.columns, row.columns
  assert 'prediction_date' in row.columns, row.colums
  assert 'gt_version' in row.columns, row.columns
  assert len(row.gt_version.unique()) == 1
  gt_version = row.gt_version.unique()[0]
  date, loc = row.name
  row = row.sort_values(by='prediction_date')
  pred_xs = row.prediction_date.values
  pred_ys = row.predictions.values

  return _row_gather_helper(loc, date, gt_version, available_versions, locale,
                            pred_xs, pred_ys, expected_forecast_len,
                            expected_gt_len, gt_df, debug)


def gather_data_from_retrospective_row(row,
                                       gt_df,
                                       locale,
                                       available_versions,
                                       expected_forecast_len = 28,
                                       expected_gt_len = 27,
                                       set_missing_values_to_zero = False,
                                       debug = False):
  """Pandas `apply` fn to gather GT and preds from data.

  We use this function to parallelize using the pandas groupby.
  With this optimization (compared to a for-loop), this function goes from
  1.2 min -> 0.2 on Japan.
  2.0 min -> XXX on US.

  Args:
    row:
    gt_df:
    locale:
    available_versions: A numpy array of available versions in `gt_df`. The
      difference between precomputing this and doing it on-the-fly for each row
      is about 10 minutes of compute over the prospective window, so we pass
      this in precomputed as a speedup.
    expected_forecast_len: Expected length of the forecast.
    expected_gt_len: Expected length of GT. Doesn't have to match
      `expected_forecast_len` due to how forecasts are measured.
    set_missing_values_to_zero: If `True`, assuming missing GT values are 0.
      This should only be used for Japan deaths, during the retrospective
      period.
    debug:

  Returns:
    {'dates', 'predictions', 'ground_truth'}
  """
  assert 'data' in row.index, row.index
  assert 'gt_version' in row.index, row.index
  date, loc = row.name
  date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
  pred_xs = row.data['prediction_date']
  pred_ys = row.data['predictions']
  gt_version = row.gt_version

  return _row_gather_helper(loc, date, gt_version, available_versions, locale,
                            pred_xs, pred_ys, expected_forecast_len,
                            expected_gt_len, gt_df, debug,
                            set_missing_values_to_zero)


def get_next_gt_version(cur_gt_version, all_gt_versions):
  """Get the next gt version."""
  assert cur_gt_version in all_gt_versions, (cur_gt_version, all_gt_versions)
  index = np.where(all_gt_versions == cur_gt_version)[0]
  assert len(index) == 1, index
  index = int(index[0])
  assert isinstance(index, int), (index, type(index))

  next_gt_version = all_gt_versions[index + 1]
  assert isinstance(next_gt_version, str), type(next_gt_version)

  return next_gt_version


def _day_before(day_date):
  assert isinstance(day_date, datetime.date)
  return day_date - datetime.timedelta(days=1)


def _row_gather_helper(loc,
                       date,
                       gt_version,
                       available_versions,
                       locale,
                       pred_xs,
                       pred_ys,
                       expected_forecast_len,
                       expected_gt_len,
                       gt_df,
                       debug,
                       set_missing_values_to_zero=False,
                       get_day_zero_gt=True):
  """Common code for prospective / retrospective row gathers."""
  assert len(pred_xs) == len(pred_ys)

  if len(pred_ys) != expected_forecast_len:
    print(f'No good predictions for {loc} {date}: '
          f'{len(pred_ys)} vs {expected_forecast_len}')
    return np.nan

  assert isinstance(pred_xs, np.ndarray), type(pred_xs)
  assert isinstance(pred_ys, np.ndarray), type(pred_ys)

  loc_key = LOC_NAME[locale]
  assert isinstance(date, datetime.date)

  # Change the location name, if necessary.
  gt_loc = loc.title()
  if locale == 'state':
    if gt_loc not in STATE_TO_CODE_MAP_:
      print(f'Skipping state {gt_loc}, since not found.')
      return np.nan
    gt_loc = STATE_TO_CODE_MAP_[gt_loc]

  # Get the right GT slice for this prediction window.
  # We start with `gt_version`, and increment until we have enough data to cover
  # `expected_gt_len` days into the future. This matches how the evaluator does
  # it.
  # We might have multiple features, so pick one that's the right length.
  feature_names = gt_df.feature_name.unique()
  cur_gt_version, cur_gt, day_zero_gt = None, pd.DataFrame(), None
  while cur_gt.size == 0 or len(cur_gt.dt.unique()) < expected_gt_len:
    if cur_gt_version is None:
      cur_gt_version = gt_version
    else:
      print(f'Date {date} {gt_loc} failed: Found {len(cur_gt.dt.unique())}, '
            f'expected {expected_gt_len} days with version {cur_gt_version}, '
            'so incrementing...')
      cur_gt_version = get_next_gt_version(cur_gt_version, available_versions)
    for feature_name in feature_names:
      min_gt = _day_before(min(pred_xs)) if get_day_zero_gt else min(pred_xs)
      assert isinstance(min_gt, datetime.date)
      cur_gt = gt_df[(gt_df.version == cur_gt_version)
                     & (gt_df.dt >= min_gt) & (gt_df.dt <= max(pred_xs)) &
                     (gt_df[loc_key] == gt_loc) &
                     (gt_df.feature_name == feature_name)][[
                         'dt', 'feature_value'
                     ]]
      expected_len = expected_gt_len + 1 if get_day_zero_gt else expected_gt_len
      if len(cur_gt.dt.unique()) >= expected_len:
        if get_day_zero_gt:
          day_zero_gt = cur_gt[cur_gt.dt == min_gt]
          if len(day_zero_gt) == 0 and set_missing_values_to_zero:  # pylint:disable=g-explicit-length-test
            day_zero_gt = 0
          else:
            day_zero_gt = day_zero_gt.feature_value.values[0]
            cur_gt = cur_gt[~(cur_gt.dt == min_gt)]
        break
    # Japan deaths are often empty, but we can safely assume that they're 0. So
    # skip the sanity check, and just assume that missing values are 0 later.
    if set_missing_values_to_zero:
      break
  assert len(cur_gt.dt.unique()) >= expected_gt_len
  assert expected_gt_len <= expected_forecast_len
  if set_missing_values_to_zero:
    cur_dates = cur_gt.dt.unique()
    expected_dates = pd.date_range(start=min(pred_xs), end=max(pred_xs))
    missing_dates = expected_dates.difference(cur_dates)
    if missing_dates.size > 0:
      print(f'Found missing dates for {date} {gt_loc}: {missing_dates}')
      cur_gt = cur_gt.append([{
          'dt': dt.date(),
          'feature_value': 0.0
      } for dt in missing_dates],
                             ignore_index=True)
    assert expected_dates.difference(cur_gt.dt.unique()).size == 0
  assert np.all(
      sorted(cur_gt.dt.unique())[:expected_gt_len] == pred_xs[:expected_gt_len])

  cur_gt = cur_gt.sort_values(by='dt')
  gtru_xs = cur_gt.dt.to_numpy()
  gtru_ys = cur_gt.feature_value.to_numpy()
  if len(gtru_ys) < expected_gt_len:
    print(f'Length of gt wrong: {len(gtru_ys)} {loc} {date}')
    return np.nan

  if debug:
    print(f'Finish a row: {loc} {date}')

  ret = {
      'dates': gtru_xs,
      'predictions': pred_ys,
      'ground_truth': gtru_ys,
  }
  if get_day_zero_gt:
    ret['day_zero_gt'] = day_zero_gt if day_zero_gt else 0.0

  return ret


def mean_confidence_interval(data, confidence=0.95, ignore_nan=False):
  a = 1.0 * np.array(data)
  n = len(a)
  if ignore_nan:
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
  else:
    m, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return m, m - h, m + h


def kaz_locations_to_open_covid_map():
  return {
      'Hyogo': 'Hyōgo',
      'Kochi': 'Kōchi',
      'Oita': 'Ōita',
  }


def open_covid_locations_to_kaz_map():
  return {v: k for (k, v) in kaz_locations_to_open_covid_map().items()}


def state_name_to_code_map():
  """State name to code map."""
  abbreviation = 'abbreviation'
  name = 'name'
  state_map = [
      {
          abbreviation: 'AL',
          name: 'Alabama'
      },
      {
          abbreviation: 'AK',
          name: 'Alaska'
      },
      {
          abbreviation: 'AZ',
          name: 'Arizona'
      },
      {
          abbreviation: 'AR',
          name: 'Arkansas'
      },
      {
          abbreviation: 'CA',
          name: 'California'
      },
      {
          abbreviation: 'CO',
          name: 'Colorado'
      },
      {
          abbreviation: 'CT',
          name: 'Connecticut'
      },
      {
          abbreviation: 'DE',
          name: 'Delaware'
      },
      {
          abbreviation: 'DC',
          name: 'District Of Columbia'
      },
      {
          abbreviation: 'FL',
          name: 'Florida'
      },
      {
          abbreviation: 'GA',
          name: 'Georgia'
      },
      {
          abbreviation: 'HI',
          name: 'Hawaii'
      },
      {
          abbreviation: 'ID',
          name: 'Idaho'
      },
      {
          abbreviation: 'IL',
          name: 'Illinois'
      },
      {
          abbreviation: 'IN',
          name: 'Indiana'
      },
      {
          abbreviation: 'IA',
          name: 'Iowa'
      },
      {
          abbreviation: 'KS',
          name: 'Kansas'
      },
      {
          abbreviation: 'KY',
          name: 'Kentucky'
      },
      {
          abbreviation: 'LA',
          name: 'Louisiana'
      },
      {
          abbreviation: 'ME',
          name: 'Maine'
      },
      {
          abbreviation: 'MD',
          name: 'Maryland'
      },
      {
          abbreviation: 'MA',
          name: 'Massachusetts'
      },
      {
          abbreviation: 'MI',
          name: 'Michigan'
      },
      {
          abbreviation: 'MN',
          name: 'Minnesota'
      },
      {
          abbreviation: 'MS',
          name: 'Mississippi'
      },
      {
          abbreviation: 'MO',
          name: 'Missouri'
      },
      {
          abbreviation: 'MT',
          name: 'Montana'
      },
      {
          abbreviation: 'NE',
          name: 'Nebraska'
      },
      {
          abbreviation: 'NV',
          name: 'Nevada'
      },
      {
          abbreviation: 'NH',
          name: 'New Hampshire'
      },
      {
          abbreviation: 'NJ',
          name: 'New Jersey'
      },
      {
          abbreviation: 'NM',
          name: 'New Mexico'
      },
      {
          abbreviation: 'NY',
          name: 'New York'
      },
      {
          abbreviation: 'NC',
          name: 'North Carolina'
      },
      {
          abbreviation: 'ND',
          name: 'North Dakota'
      },
      {
          abbreviation: 'OH',
          name: 'Ohio'
      },
      {
          abbreviation: 'OK',
          name: 'Oklahoma'
      },
      {
          abbreviation: 'OR',
          name: 'Oregon'
      },
      {
          abbreviation: 'PA',
          name: 'Pennsylvania'
      },
      {
          abbreviation: 'RI',
          name: 'Rhode Island'
      },
      {
          abbreviation: 'SC',
          name: 'South Carolina'
      },
      {
          abbreviation: 'SD',
          name: 'South Dakota'
      },
      {
          abbreviation: 'TN',
          name: 'Tennessee'
      },
      {
          abbreviation: 'TX',
          name: 'Texas'
      },
      {
          abbreviation: 'UT',
          name: 'Utah'
      },
      {
          abbreviation: 'VT',
          name: 'Vermont'
      },
      {
          abbreviation: 'VA',
          name: 'Virginia'
      },
      {
          abbreviation: 'WA',
          name: 'Washington'
      },
      {
          abbreviation: 'WV',
          name: 'West Virginia'
      },
      {
          abbreviation: 'WI',
          name: 'Wisconsin'
      },
      {
          abbreviation: 'WY',
          name: 'Wyoming'
      },
  ]
  return {d[name]: d[abbreviation] for d in state_map}


STATE_TO_CODE_MAP_ = state_name_to_code_map()


def code_to_state_name_map():
  return {v: k for (k, v) in state_name_to_code_map().items()}
