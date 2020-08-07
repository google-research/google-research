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

r"""Utilities for loading vanilla WALS distribution.

The WALS files are distributed here:
  https://github.com/cldf-datasets/wals

Example:
--------
 Clone the GitHub WALS data to WALS_DIR. Then run:

 > WALS_DIR=...
 > python3 vanilla_reader_main.py \
     --wals_dir ${WALS_DIR} \
     --output_dir ${OUTPUT_DIR}

 The above will create "wals.csv" files converted from the CLDF format
 provided by WALS. Our models should be able to injest these csv files.
 By default this exports everything as strings. To change this behavior
 and exports categorical variables as ints, please pass
 --categorical_as_ints flag.

 Note: It looks like the "countrycodes" information is not provided by WALS,
 possibly coming from other sources, such as Glottolog. Leaving this column
 empty for now.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os

from absl import app
from absl import flags
from absl import logging

import constants as const
import data_info as data_lib
import numpy as np
import pandas as pd

flags.DEFINE_string(
    "wals_dir", "", "Root directory of WALS distribution.")

flags.DEFINE_string(
    "output_dir", "",
    "Output directory for preprocessed files.")

flags.DEFINE_bool(
    "categorical_as_ints", False,
    "Encode all the categorical features as ints.")

FLAGS = flags.FLAGS

# Names of the data files, also serving as keys into the global data dictionary.
_DATA_LANGUAGE = "languages"
_DATA_CODES = "codes"
_DATA_VALUES = "values"
_DATA_PARAMS = "parameters"

# Language column names.
_LANGUAGE_COLUMN_NAMES = [
    "ISO639P3code", "Glottocode", "Name", "Latitude", "Longitude", "Genus",
    "Family", "Subfamily", "Macroarea", "ISO_codes"
]

# Parameter ID key used to denote feature name ID.
_PARAM_ID = "Parameter_ID"

# Name of the output dataset.
_DATASET_NAME = "wals"


def _cldf_dir(wals_dir):
  """Returns directory where the cldf files reside."""
  return os.path.join(wals_dir, "cldf")


def _build_df(wals_dir, filename):
  """Reads data file in csv format into a dictionary."""
  path = os.path.join(_cldf_dir(wals_dir), filename)
  logging.info("Reading \"%s\" ...", path)
  df = pd.read_csv(path, encoding=const.ENCODING)
  df = df.to_dict("index")
  logging.info("Read %d elements.", len(df))
  return df


def _get_languages(data):
  """Builds dictionary of languages."""
  languages = collections.defaultdict(lambda: collections.defaultdict(int))
  for lang_id in range(len(data[_DATA_LANGUAGE])):
    for col_id in _LANGUAGE_COLUMN_NAMES:
      lang_data = data[_DATA_LANGUAGE][lang_id]
      languages[lang_data["ID"]][col_id] = lang_data[col_id]
  logging.info("Collected %d languages.", len(languages))
  return languages


def _get_feature_names(codes):
  """Builds feature names dictionary from the codes table.

  Runs over all the feature value types and their names and collects a
  mapping from all the feature IDs to their possible values. The
  corresponding code table entries in data look as follows:

     ID,Parameter_ID,Name,Description,Number,icon
     ...
     5A-3,5A,Missing /p/,Missing /p/,3,cdd0000
     5A-4,5A,Missing /g/,Missing /g/,4,c0000dd
     ...

  Resulting dictionary:
     ...
     144Q -> {'n_values': 4, 'Names': [
        'NoDoubleNeg', 'OptDoubleNeg', 'OnlyWithAnotherNeg', 'No SNegOV']}
     ...

  Args:
    codes: (list) List of language codes (strings).

  Returns:
    Dictionary containing the mapping of features codes to their values.
  """
  feat2values = collections.defaultdict(lambda: collections.defaultdict(int))
  current_feature = codes[0][_PARAM_ID]  # 1A
  current_feature_names = []
  current_value_codes = []
  num_values = 0
  for code_id in range(len(codes)):
    code_info = codes[code_id]
    feature_value = code_info["Name"]
    if code_info[_PARAM_ID] == current_feature:
      current_feature_names.append(feature_value)
      current_value_codes.append(code_info["ID"])
      feat2values[code_info[_PARAM_ID]]["num_values"] += 1
      num_values += 1
    else:
      prev_feature = feat2values[codes[code_id - 1][_PARAM_ID]]
      prev_feature["Names"] = current_feature_names
      prev_feature["Codes"] = current_value_codes
      current_feature_names = [feature_value]
      current_value_codes = [code_info["ID"]]
      feat2values[code_info[_PARAM_ID]]["num_values"] += 1
      num_values += 1

    if code_id == len(codes) - 1:
      # Last feature value.
      feature = feat2values[code_info[_PARAM_ID]]
      feature["Names"] = current_feature_names
      feature["Codes"] = current_value_codes
    current_feature = code_info[_PARAM_ID]
  logging.info("Collected %d feature names (%d supported values).",
               len(feat2values), num_values)
  return feat2values


def _fill_feature_values(data_values, feature_names, languages):
  """Adds feature values to languages dictionary."""
  logging.info("Filling feature values for languages ...")
  for value_id in range(len(data_values)):
    values = data_values[value_id]
    feature_name = values[_PARAM_ID]
    cur_value_code = feature_name + "-" + str(values["Value"])
    if cur_value_code != values["Code_ID"]:
      raise ValueError("Invalid value code: %s" % cur_value_code)
    val = feature_names[feature_name]["Codes"].index(cur_value_code) + 1
    if val != values["Value"]:
      raise ValueError("Invalid value: %s" % val)
    languages[values["Language_ID"]][feature_name] = val


def _get_areas(params, feature_names):
  """Returns mapping between areas and constituent feature types."""
  areas = {}
  for feature_id in range(len(params)):
    feature_info = params[feature_id]
    feature = feature_info["ID"]
    area = feature_info["Area"]
    if area not in areas:
      areas[area] = [feature]
    else:
      areas[area].append(feature)

    # Update the mapping from ID to name.
    feature_names[feature]["Name"] = feature_info["Name"]

  logging.info("Collected %d areas (%s).", len(areas), ",".join(areas.keys()))
  return areas


def _fill_feature_stats(languages, feature_names, areas):
  """Fills in basic feature statistics given the features and their areas."""
  total_num_features = len(feature_names)
  for lang_id in languages:
    language = languages[lang_id]
    # Measure of inverse feature sparsity: percentage of populated features.
    language["%"] = ((len(language) - len(_LANGUAGE_COLUMN_NAMES)) *
                     100.0 / total_num_features)

    # Fill in area statistics.
    for area in areas:
      count = 0
      features = areas[area]
      for feature in features:
        # Following will insert a 0-valued feature if it is not found.
        if language[feature] != 0:
          count += 1
      language[area] = count / len(features)


def _get_feature_types(areas, feature_names):
  """Fills in some statistics on feature types (aka areas)."""
  types = collections.defaultdict(lambda: collections.defaultdict(int))
  for area in areas:
    features = areas[area]
    feature_type = types[area]
    feature_type["num_features"] = len(features)
    for feature in features:
      feature_info = feature_names[feature]
      feature_type["num_values"] += feature_info["num_values"]
      feature_info["emb_dim"] = max(1, int(max(1, np.floor(
          (feature_info["num_values"] + 1) / 10.0))))
      feature_type["total_dim"] += feature_info["emb_dim"]
  return types


def _read(wals_dir):
  """Read vanilla WALS dataset from the supplied directory."""
  logging.info("Reading WALS from a root directory \"%s\" ...", wals_dir)
  # Read in all the relevant data files.
  datafile_names = [_DATA_LANGUAGE, _DATA_CODES, _DATA_VALUES, _DATA_PARAMS]
  data = {}
  for name in datafile_names:
    data[name] = _build_df(wals_dir, name + ".csv")

  # Build dictionaries.
  languages = _get_languages(data)
  feature_names = _get_feature_names(data[_DATA_CODES])
  areas = _get_areas(data[_DATA_PARAMS], feature_names)
  feature_types = _get_feature_types(areas, feature_names)
  _fill_feature_values(data[_DATA_VALUES], feature_names, languages)
  _fill_feature_stats(languages, feature_names, areas)
  return data, languages, areas, feature_names, feature_types


def _prepare_data_info(mappings):
  """Prepares data info mappings."""
  # Prepare the container.
  _, languages, _, feature_names_dict, _ = mappings
  data_info = {}
  data_info[const.DATA_KEY_FEATURES] = {}
  features = data_info[const.DATA_KEY_FEATURES]
  data_info[const.DATA_KEY_GENERA] = []
  genera = data_info[const.DATA_KEY_GENERA]
  data_info[const.DATA_KEY_FAMILIES] = []
  families = data_info[const.DATA_KEY_FAMILIES]

  # Fill in data mappings.
  for lang_id in languages.keys():
    # Genera and families.
    language = languages[lang_id]
    if language["Genus"] not in genera:
      genera.append(language["Genus"])
    if language["Family"] not in families:
      families.append(language["Family"])

  # Actual features.
  feature_ids = list(feature_names_dict.keys())
  feature_names = [
      feature_names_dict[name]["Name"] for name in feature_ids]
  feature_names = [name.replace(" ", "_") for name in feature_names]
  for i in range(len(feature_ids)):
    feature_id = feature_ids[i]
    name = feature_names[i]
    features[name] = feature_names_dict[feature_id]["Names"]
    features[name].sort(key=str)

  # Postprocess.
  genera.sort(key=str)
  families.sort(key=str)

  return data_info


def _make_df(mappings, categorical_as_ints=False):
  """Converts WALS mappings to data frame."""
  # Prepare the core columns.
  _, languages, _, feature_names_dict, _ = mappings
  data_info = _prepare_data_info(mappings)

  wals_codes = sorted(languages.keys(), key=str)
  names = [languages[code]["Name"] for code in wals_codes]
  latitudes = [languages[code]["Latitude"] for code in wals_codes]
  longitudes = [languages[code]["Longitude"] for code in wals_codes]
  countries = [np.nan for code in wals_codes]

  genera = [languages[code]["Genus"] for code in wals_codes]
  if categorical_as_ints:
    genera_names = data_info[const.DATA_KEY_GENERA]
    genera = [genera_names.index(name) + 1 for name in genera]
  families = [languages[code]["Family"] for code in wals_codes]
  if categorical_as_ints:
    families_names = data_info[const.DATA_KEY_FAMILIES]
    families = [families_names.index(name) + 1 for name in families]

  # Prepare feature columns.
  feature_ids = list(feature_names_dict.keys())
  feature_names = [
      feature_names_dict[name]["Name"] for name in feature_ids]
  feature_names = [name.replace(" ", "_") for name in feature_names]
  all_feature_values = []
  for i in range(len(feature_ids)):
    feature_id = feature_ids[i]
    feature_name = feature_names[i]
    feature_values = []
    data_info_feature_vals = data_info[const.DATA_KEY_FEATURES][feature_name]
    for code in wals_codes:
      language = languages[code]
      assert feature_id in language
      val = language[feature_id]
      if val == 0:  # Missing marker.
        value_name = np.nan
      else:
        value_name = feature_names_dict[feature_id]["Names"][val - 1]
        if categorical_as_ints:
          value_name = data_info_feature_vals.index(value_name) + 1
      feature_values.append(value_name)
    all_feature_values.append(feature_values)

  # Create dataframe.
  columns = ["wals_code", "name", "latitude", "longitude", "genus", "family",
             "countrycodes"]
  for feature_id in range(len(feature_ids)):
    columns.append(feature_names[feature_id])

  data = {
      "wals_code": wals_codes, "name": names, "latitude": latitudes,
      "longitude": longitudes, "genus": genera, "family": families,
      "countrycodes": countries
  }
  for feature_id in range(len(feature_ids)):
    data[feature_names[feature_id]] = all_feature_values[feature_id]

  return pd.DataFrame(data, columns=columns), data_info


def main(unused_argv):
  if not FLAGS.wals_dir:
    raise ValueError("Specify --wals_dir!")
  if not FLAGS.output_dir:
    raise ValueError("Specify --output_dir!")

  logging.info("Preparing dataset ...")
  df, data_info = _make_df(_read(FLAGS.wals_dir),
                           categorical_as_ints=FLAGS.categorical_as_ints)

  output_file = os.path.join(FLAGS.output_dir, _DATASET_NAME + ".csv")
  logging.info("Saving dataset to \"%s\" ...", output_file)
  df.to_csv(output_file, sep="|", index=False, float_format="%g")
  logging.info("Saved %d languages.", len(df))

  output_file = os.path.join(
      FLAGS.output_dir,
      const.DATA_INFO_FILENAME + "_" + _DATASET_NAME +
      data_lib.FILE_EXTENSION)
  data_lib.write_data_info(output_file, data_info)


if __name__ == "__main__":
  app.run(main)
