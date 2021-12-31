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

# Lint as: python3
"""Define your static and time-series features based on input data."""
import collections
import functools
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from covid_epidemiology.src import constants
from covid_epidemiology.src import feature_preprocessing as preprocessing


def get_gt_source_features(location_granularity,
                           gt_source):
  """Return 2-tuple of gt feature keys for deaths and confirmed cases.

  Note that the keys point to features that have the keyword 'state' in them.
  This is by design - the features were named such that they could be used
  interchangeably.

  Args:
    location_granularity: string. Geographic resolution, from
      `constants.LOCATION_GRANULARITY_LIST` (ex. "STATE").
    gt_source: string. Source of ground truth, from one of
      `constants.GT_SOURCE_LIST` (ex. "JHU").

  Returns:
    2-tuple of type string, containing the death feature key and confirmed cases
    features key, in that order.
  """
  if location_granularity not in constants.LOCATION_GRANULARITY_LIST:
    raise ValueError(
        "`location_granularity` should be one of "
        f"{constants.LOCATION_GRANULARITY_LIST}: {location_granularity}")
  if gt_source not in constants.GT_SOURCE_LIST:
    raise ValueError(
        f"`gt_source` should be one of {constants.GT_SOURCE_LIST}: {gt_source}")

  gt_source = gt_source.upper()
  location = location_granularity.lower()
  if gt_source == constants.GT_SOURCE_JHU:
    gt_keys = (constants.JHU_DEATH_FEATURE_KEY.replace("state", location),
               constants.JHU_CONFIRMED_FEATURE_KEY.replace("state", location))
  elif gt_source == constants.GT_SOURCE_NYT:
    gt_keys = (constants.NYT_DEATH_FEATURE_KEY.replace("state", location),
               constants.NYT_CONFIRMED_FEATURE_KEY.replace("state", location))
  elif gt_source == constants.GT_SOURCE_USAFACTS:
    gt_keys = (constants.USAFACTS_DEATH_FEATURE_KEY,
               constants.USAFACTS_CONFIRMED_FEATURE_KEY)
  elif gt_source == constants.GT_SOURCE_JAPAN:
    gt_keys = (constants.JAPAN_PREFECTURE_DEATH_FEATURE_KEY,
               constants.JAPAN_PREFECTURE_CONFIRMED_FEATURE_KEY)
  else:
    raise ValueError(
        "Unknown ground truth source {}.".format(gt_source),
        "Must be one of ('" + constants.GT_SOURCE_JHU + "', '" +
        constants.GT_SOURCE_NYT + "','" + constants.GT_SOURCE_USAFACTS +
        "', '" + constants.GT_SOURCE_JAPAN + "').")

  return gt_keys


def filter_to_location(data, location,
                       location_granularity):
  if location_granularity == constants.LOCATION_GRANULARITY_COUNTRY:
    return data[data[constants.COUNTRY_COLUMN] == location]
  else:
    return data[data[constants.GEO_ID_COLUMN] == location]


def splice_data(from_df, to_df,
                feature_map, date_list,
                location_list):
  """Splice a subset of one dataframe into another.

  Copy a subset of one dataframe into another, overwriting any existing
  features.

  Args:
    from_df: Pandas dataframe containing source data
    to_df: Pandas dataframe, optional. Datadframe where the data is to be
      spliced into, in the same schema as 'from_df'.
    feature_map: Dictionary. Map of old features to new features.
    date_list: List, optional. List of times for which feature values are to be
      copied.
    location_list: List, optional. List of locations for which feature values
      are to be copied.

  Returns:
    Pandas dataframe that contains the target dataframe with data replaced with
    the spliced data.

  Raises:
    AssertionError if all source features are not in source dataframe.
    AssertionError if all target features are not in target dataframe.
  """

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.NOTSET)  # inherit level from root logger

  if to_df is None:
    to_df = from_df.copy()

  # checks
  from_df_features = from_df[constants.FEATURE_NAME_COLUMN].unique().tolist()
  assert all(e in from_df_features for e in feature_map.keys()), (
      "All {} source features not in {} ".format(feature_map.keys(),
                                                 from_df_features))
  to_df_features = to_df[constants.FEATURE_NAME_COLUMN].unique().tolist()
  assert all(e in to_df_features for e in feature_map.values()), (
      "All {} target features not in {} ".format(feature_map.values(),
                                                 to_df_features))

  logging.info("Splicing data by mapping features.")

  # define masks to isolate source and target features
  from_df_mask = from_df[constants.FEATURE_NAME_COLUMN].isin(feature_map.keys())
  to_df_mask = to_df[constants.FEATURE_NAME_COLUMN].isin(feature_map.values())
  if location_list is not None:
    from_df_loc_mask = from_df[constants.GEO_ID_COLUMN].isin(location_list)
    to_df_loc_mask = to_df[constants.GEO_ID_COLUMN].isin(location_list)
  else:
    from_df_loc_mask = True
    to_df_loc_mask = True
  if date_list is not None:
    from_df_date_mask = from_df[constants.DATE_COLUMN].isin(date_list)
    to_df_date_mask = to_df[constants.DATE_COLUMN].isin(date_list)
  else:
    from_df_date_mask = True
    to_df_date_mask = True

  # drop target features
  to_df = to_df.drop(to_df[to_df_mask & to_df_loc_mask & to_df_date_mask].index)

  # then pull out and append source features
  slice_df = from_df[from_df_mask & from_df_loc_mask & from_df_date_mask].copy()
  slice_df[constants.FEATURE_NAME_COLUMN] = slice_df[
      constants.FEATURE_NAME_COLUMN].map(feature_map).fillna(
          slice_df[constants.FEATURE_NAME_COLUMN])
  to_df = to_df.append(slice_df)
  return to_df.sort_values([
      constants.GEO_ID_COLUMN, constants.DATE_COLUMN,
      constants.FEATURE_NAME_COLUMN
  ],
                           axis=0).reset_index(drop=True)


US_GT_SOURCES = frozenset((constants.GT_SOURCE_JHU, constants.GT_SOURCE_NYT,
                           constants.GT_SOURCE_USAFACTS))


def overwrite_counties_gt(
    ts_data,
    gt_source,
    overwrite_locations = ("36005", "36047", "36061", "36081",
                                          "36085"),
    overwrite_source = constants.GT_SOURCE_USAFACTS,
    overwrite_location_granularity = constants.LOCATION_GRANULARITY_COUNTY,
    min_overwrite_date = None,
    max_overwrite_date = None,
):
  """Splice GT data from a different source for the listed locations.

  NYC counties are not separately represented in the JHU or NYT ts datasets.
  Replace the value for the specified GT for the county location granularity
  with the values from the feature_name returned by get_gt_source_features
  with the specified GT source and location_granularity.

  Duke and Nantucket counties are not separately represented in the JHU
  dataset. Copy the data from the overwrite_source dataset into the appropriate
  locations in the input feature rows.

  This function will overwrite ground-truth data, so use with care.

  Args:
    ts_data: Pandas dataframe. Dataframe containing timeseries data.
    gt_source: string. Source of ground-truth, one of 'USAFACTS', 'JHU' or
      'NYT'.
    overwrite_locations: fips codes for overwriting counties.
    overwrite_source: The GT-source to use to overwrite the data.
    overwrite_location_granularity: The location granularity to use to overwrite
      the data. The input granularity is assumed to be 'COUNTY'.
    min_overwrite_date: The first date to overwrite. Inclusive.
    max_overwrite_date: The most recent date to overwrite. Inclusive.

  Returns:
    Pandas dataframe containing source data with NYC county ground truth copied
    from the specified dataset and granularity.

  Raises:
    ValueError if gt_source is not one of the accepted strings.
  """

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.NOTSET)  # inherit level from root logger

  if gt_source not in US_GT_SOURCES:
    raise ValueError(("Ground-truth source mut be in ",
                      "{}, received {}".format(US_GT_SOURCES, gt_source)))
  if overwrite_source not in US_GT_SOURCES:
    raise ValueError(("Overwrite source mut be in ",
                      "{}, received {}".format(US_GT_SOURCES, gt_source)))
  if overwrite_location_granularity not in {
      constants.LOCATION_GRANULARITY_COUNTY,
      constants.LOCATION_GRANULARITY_STATE
  }:
    raise ValueError(("Overwrite location granularity mut one of ",
                      "['COUNTY', 'STATE'], received {}".format(
                          overwrite_location_granularity)))

  if (gt_source == overwrite_source and
      constants.LOCATION_GRANULARITY_COUNTY == overwrite_location_granularity):
    # Nothing to do if source matches the overwrite and it is county.
    return ts_data

  gt_features = get_gt_source_features("COUNTY", gt_source)
  overwrite_features = get_gt_source_features(overwrite_location_granularity,
                                              overwrite_source)
  features_map = dict(zip(overwrite_features, gt_features))
  # For New York City Counties:
  # The Bronx is Bronx County (ANSI / FIPS 36005)
  # Brooklyn is Kings County (ANSI / FIPS 36047)
  # Manhattan is New York County (ANSI / FIPS 36061)
  # Queens is Queens County (ANSI / FIPS 36081)
  # Staten Island is Richmond County (ANSI / FIPS 36085)
  # For Duke/Nantucket Counties:
  # The Duke County (FIPS 25007)
  # The Nantucket County (FIPS 25019)
  loc_mask = ts_data[constants.GEO_ID_COLUMN].isin(overwrite_locations)
  gt_mask = ts_data[constants.FEATURE_NAME_COLUMN].isin(overwrite_features)
  date_mask = pd.Series(
      np.ones((ts_data.shape[0]), dtype=bool), index=ts_data.index)
  if min_overwrite_date:
    date_mask &= pd.to_datetime(
        ts_data[constants.DATE_COLUMN]) >= min_overwrite_date
  if max_overwrite_date:
    date_mask &= pd.to_datetime(
        ts_data[constants.DATE_COLUMN]) <= max_overwrite_date
  full_mask = loc_mask & gt_mask & date_mask
  if not full_mask.any():
    return ts_data
  gt_times = ts_data[full_mask][constants.DATE_COLUMN].unique()
  new_data = splice_data(ts_data, None, features_map, gt_times,
                         overwrite_locations)
  return new_data


def get_feature_ts(data, feature_name):
  """Returns time series feature data for feature with feature_name.

  Args:
    data: Pandas dataframe to get data from
    feature_name: The str name of the feature to return
  """
  np_results = data[data[constants.FEATURE_NAME_COLUMN] == feature_name][
      constants.FEATURE_VALUE_COLUMN]
  if np_results.size == 1:
    return np_results.item()
  # Dedup if needed - return non-zero/non-nan if present, else return first.
  elif np_results.size > 1:
    for np_result in np_results:
      if np_result and not math.isnan(np_result):
        return np_result
    return np_results.iloc[0]
  # Entry not populated, returning nan to handle downstream.
  else:
    return float("nan")


def extract_ts_overrides(
    ts_data, locations,
    ts_categorical_features):
  """Extract time-series overrides.

  Args:
    ts_data: A dataframe that contains all time-series overrides.
    locations: A list of locations for which the features are needed.
    ts_categorical_features: A list of names of categorical features.

  Returns:
    A mapping from the feature name to its value, the value of each feature
    is map from location to np.ndarray.
  """

  all_dates = preprocessing.get_all_valid_dates(ts_data)
  all_feature_names = ts_data["feature_name"].unique().tolist()
  ts_features = collections.defaultdict(
      functools.partial(
          collections.defaultdict,
          functools.partial(np.zeros, shape=(len(all_dates)), dtype="float32")))

  # 3 level dictionary with defaults representing feature_name, location and
  # date_index.
  # pylint: disable=g-long-lambda
  dct = collections.defaultdict(lambda: collections.defaultdict(
      lambda: collections.defaultdict(lambda: 1.0)))
  # Default value for ts categorical overrides must be -1.0 to be no-op.
  for feature_name in ts_categorical_features:
    for location in locations:
      for date_index, _ in enumerate(all_dates):
        dct[feature_name][location][date_index] = -1.0
  dt_index = {
      pd.Timestamp(dt).to_pydatetime(): idx for idx, dt in enumerate(all_dates)
  }
  for _, row in ts_data.iterrows():
    if row[constants.DATE_COLUMN] not in dt_index:
      continue
    dct[row[constants.FEATURE_NAME_COLUMN]][row[constants.GEO_ID_COLUMN]][
        dt_index[row[constants.DATE_COLUMN]]] = row[
            constants.FEATURE_VALUE_COLUMN]

  for feature_key in all_feature_names:
    for location in locations:
      for date_index, _ in enumerate(all_dates):
        ts_features[feature_key][location][date_index] = dct[feature_key][
            location][date_index]

      key_list = list(ts_features.keys())
      if key_list:
        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        preprocessing._assert_feature_lengths_for_location(
            ts_features, location, reference_key=key_list[0])
        # pylint: enable=protected-access

  return ts_features


def extract_static_override_features(
    static_overrides):
  """Extract static feature override values.

  Args:
    static_overrides: A dataframe that contains the value for static overrides
      to be passed to the GAM Encoders.

  Returns:
    A mapping from feature name to location and then to the override value.
    This is a two-level dictionary of the format: {feature: {location: value}}
  """
  static_overrides_features = dict()
  for feature in set(static_overrides[constants.FEATURE_NAME_COLUMN]):
    static_overrides_features[feature] = dict()
    override_slice = static_overrides.loc[static_overrides[
        constants.FEATURE_NAME_COLUMN] == feature]
    for location in set(override_slice[constants.GEO_ID_COLUMN]):
      override_sub_slice = override_slice.loc[override_slice[
          constants.GEO_ID_COLUMN] == location]
      static_overrides_features[feature][location] = override_sub_slice[
          constants.FEATURE_MODIFIER_COLUMN].to_numpy()[0]
  return static_overrides_features
