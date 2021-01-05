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

"""SIGTYP I/O routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import re

from absl import logging
import constants as const
import numpy as np
import pandas as pd

# Number of columns in the original distributed data.
NUM_COLUMNS = 8

# Feature/values list separator.
_FEATURE_LIST_KEY_VALUE_SEPARATOR = "|"

# Single feature/value separator.
_FEATURE_VALUE_SEPARATOR = "="

# Sub-dictionary key for values.
_VALUES_KEY = "values"

# Sub-dictionary key for stats.
_STATS_KEY = "stats"


def _custom_parser(filename):
  """Custom parser generator: the feature values may have tabs in them."""
  with open(filename, "r", encoding=const.ENCODING) as f:
    for line in f:
      # Maximum number of splits has to be specified explicitly because the
      # feature-value lists contain spurious tab characters which we replace
      # with spaces at a later stage.
      yield re.split(r"\t", line.strip(), maxsplit=NUM_COLUMNS - 1)


def _value_sorter(feature_value):
  """Custom sorting function taking into account the structure of values.

  Given a feature value extracts its numeric portion, e.g.
    "18 ObligDoubleNeg" -> "18".

  Args:
    feature_value: (string) WALS feature value.

  Returns:
    An integer corresponding to the first portion of the string.
  """
  toks = feature_value.split(" ", maxsplit=1)
  assert len(toks) == 2, "%s: Invalid feature value" % feature_value
  val = int(toks[0])
  assert val > 0, "%s: Feature value should be positive. Found %d" % (
      feature_value, val)
  return val


def get_feature_values(df_row):
  """Returns a list of feature-value pairs from a SIGTYP encoding."""
  kv_list = df_row[NUM_COLUMNS - 1].strip().replace("\t", "").split(
      _FEATURE_LIST_KEY_VALUE_SEPARATOR)
  assert kv_list, "Empty key-value list!"
  feature_values = [kv.split(
      _FEATURE_VALUE_SEPARATOR, maxsplit=1) for kv in kv_list]
  for kv in feature_values:
    assert len(kv) == 2, "%s: Invalid feature-value" % kv
  assert feature_values, "Empty feature/value list!"
  feature_values = [(kv[0].strip(), kv[1].strip()) for kv in feature_values]
  return feature_values


def _get_data_info(df, verbose=True):
  """Given the data frames extracts the feature value mapping.

  Iterates through the data frame and parse the features. The features are
  represented as a list of key-value pairs separated by "|". Within each pair,
  the key and value are separated by "=".

  Args:
    df: (object) Pandas dataframe representing the corpus.
    verbose: (boolean) Flag indicating whether to print verbose debugging info.

  Returns:
    Dictionary containing various per-feature, per-value and per-language
    mappings.
  """
  # Following dictionary contains multiple "sections" that contain miscellaneous
  # mappings between different items of interest and their information, such as
  # values, counts and so on.
  data_info = {}
  data_info[const.DATA_KEY_FEATURES] = {}
  features = data_info[const.DATA_KEY_FEATURES]
  data_info[const.DATA_KEY_VALUES] = {}
  data_info[const.DATA_KEY_VALUES]["counts"] = collections.defaultdict(int)
  value_counts = data_info[const.DATA_KEY_VALUES]["counts"]
  data_info[const.DATA_KEY_LANGUAGES] = collections.defaultdict(
      lambda: collections.defaultdict(str))
  language_features = data_info[const.DATA_KEY_LANGUAGES]
  data_info[const.DATA_KEY_GENERA] = {}
  data_info[const.DATA_KEY_GENERA][_VALUES_KEY] = []
  genera = data_info[const.DATA_KEY_GENERA]
  data_info[const.DATA_KEY_FAMILIES] = {}
  data_info[const.DATA_KEY_FAMILIES][_VALUES_KEY] = []
  families = data_info[const.DATA_KEY_FAMILIES]
  data_info[const.DATA_KEY_FEATURES_TO_PREDICT] = collections.defaultdict(list)

  # Process the data frame and fill in the data mappings.
  num_total_values = 0
  num_unique_values = 0
  for _, row in df.iterrows():
    wals_code = row[0]
    # Language genus and family.
    genus = row[4]
    if genus not in genera[_VALUES_KEY]:
      genera[_VALUES_KEY].append(genus)
    family = row[5]
    if family not in families[_VALUES_KEY]:
      families[_VALUES_KEY].append(family)

    # Parse feature/value list.
    feature_values = get_feature_values(row)
    language_code = row[0]
    current_language_features = language_features[language_code]
    for name, value in feature_values:
      # Omit the unknown feature values from the final database.
      if value == const.UNKNOWN_FEATURE_VALUE:
        if name not in data_info[const.DATA_KEY_FEATURES_TO_PREDICT][wals_code]:
          data_info[const.DATA_KEY_FEATURES_TO_PREDICT][wals_code].append(name)
        continue

      # Extract key and value.
      current_language_features[name] = value

      # Update global feature mapping.
      if name not in features:
        features[name] = {}
        features[name][_VALUES_KEY] = [value]
        features[name][_STATS_KEY] = {}
        features[name][_STATS_KEY]["total_count"] = 0
        num_unique_values += 1
      else:
        if value not in features[name][_VALUES_KEY]:
          features[name][_VALUES_KEY].append(value)
          num_unique_values += 1
      features[name][_STATS_KEY]["total_count"] += 1
      num_total_values += 1
      value_counts[value] += 1

  if verbose:
    logging.info("Observed %d features and %d total feature values "
                 "(%d unique).", len(features), num_total_values,
                 num_unique_values)

  # Post-processing: Sort the individual feature values. We rely on the order of
  # values for categorical feature conversion.
  genera[_VALUES_KEY].sort()
  families[_VALUES_KEY].sort()
  if verbose:
    logging.info("Collected %d unique genera from %d unique families.",
                 len(genera[_VALUES_KEY]), len(families[_VALUES_KEY]))
  for feature in features.keys():
    features[feature][_VALUES_KEY] = sorted(
        features[feature][_VALUES_KEY], key=_value_sorter)
  return data_info


def read(filename, categorical_as_ints=False, verbose=True):
  """Reads dataset from the supplied directory.

  By default the feature values will be returned as strings. If
  `categorical_as_ints` is enabled, the values will be mapped to integers.

  Args:
    filename: (string) Filename (in csv format) to read from.
    categorical_as_ints: (boolean) Convert categorical string values to ints.
    verbose: (boolean) Log debugging information.

  Returns:
    A tuple consisting the original data frame (with minimal modifications),
    the new dataframe (massaged into shape, with every feature having its own
    dedicated column) and the data info mappings.
  """
  # Read the file into a data frame.
  logging.info("Reading \"%s\" ...", filename)
  parser = _custom_parser(filename)
  header = next(parser)
  if len(header) != NUM_COLUMNS:
    logging.warning("Header does not have %d tab-separated columns: "
                    "the SIGTYP organizers have been sloppy.", len(header))
    if len(header) == 1:
      header = header[0].split()  # Assume it's spaces
      if len(header) != NUM_COLUMNS:
        raise ValueError("Invalid number of header columns: %d" % len(header))
    else:  # Give up
      raise ValueError("Invalid number of header columns: %d" % len(header))
  vanilla_df = pd.DataFrame(parser, columns=header)
  logging.info("Read %d entries", vanilla_df.shape[0])
  if verbose:
    print("Summary: {}".format(vanilla_df))

  # Latitude and longitude are floats.
  geo_cols = ["latitude", "longitude"]
  for col in geo_cols:
    vanilla_df[col] = pd.to_numeric(vanilla_df[col], errors="coerce")
  if verbose:
    print("Types: {}".format(vanilla_df.dtypes))

  # Massage the supplied data into some acceptable shape and
  # create a new dataframe with original encoded feature column removed.
  data_info = _get_data_info(vanilla_df, verbose=verbose)
  df = vanilla_df.copy()
  df = df.drop(["features"], axis=1)
  num_columns = NUM_COLUMNS - 1
  assert df.shape[1] == num_columns

  # Fill in explicit feature columns.
  feature_values = data_info[const.DATA_KEY_FEATURES]
  language_features = data_info[const.DATA_KEY_LANGUAGES]

  if verbose:
    logging.info("Preparing data ...")
  for feature_name in sorted(feature_values.keys()):
    df.insert(num_columns, feature_name,
              np.NaN if categorical_as_ints else None)
    num_columns += 1
  genera = data_info[const.DATA_KEY_GENERA]
  families = data_info[const.DATA_KEY_FAMILIES]
  for index, row in df.iterrows():
    # Convert genus and family to categorical ints, if needed.
    if categorical_as_ints:
      genus = genera[_VALUES_KEY].index(row[4]) + 1
      df.at[index, "genus"] = genus
      family = families[_VALUES_KEY].index(row[5]) + 1
      df.at[index, "family"] = family

    # Fill individual WALS features as separate columns.
    language_code = row[0]
    current_features = language_features[language_code]
    for feature_name in sorted(current_features.keys()):
      value = current_features[feature_name]
      if value == const.UNKNOWN_FEATURE_VALUE:
        value = np.NaN
      else:
        assert feature_name in feature_values
        if categorical_as_ints:
          value = feature_values[feature_name][_VALUES_KEY].index(value) + 1
      df.at[index, feature_name] = value
  if verbose:
    print("{}".format(df))
  return vanilla_df, df, data_info
