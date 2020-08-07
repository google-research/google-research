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

"""Feature matrix builder.

Builds a feature matrix for one feature suitable for feeding into a
classifier.

Output is in the form of rows:

   language latitude longitude feature value genus_feature_triple
     family_feature_triple neighbor_feature_triple implicational_quintuples,

where each triple is of the form:

   (majority value, probability estimate, count)

and each quintuple in implicationals is

   (majority value, probability estimate, count, probability estimate, count).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import csv
import os

from absl import flags
from absl import logging
import constants as const
import data_info as data_lib
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS

_UNKNOWN_STR = "NA"
_SMALL_PROB = 1E-6

# We set default probability to be a small value rather than zero
# to allow potential conversion to logprob domain.
_DUMMY_TRIPLE = (_UNKNOWN_STR, 1E-6, 0)
_DUMMY_QUINTUPLE = (_UNKNOWN_STR, 1E-6, 0, 1E-5, 0)


def _get_associations_path(filename):
  return os.path.join(FLAGS.association_dir, filename)


class FeatureMaker:
  """Constructs feature representations suitable for building classifiers."""

  def __init__(self,
               training_data,
               dev_data,
               data_info_file):
    self._genus_values = collections.defaultdict(
        lambda: collections.defaultdict(tuple))
    self._family_values = collections.defaultdict(
        lambda: collections.defaultdict(tuple))
    self._neighborhood_values = collections.defaultdict(
        lambda: collections.defaultdict(tuple))
    self._implicational_values = collections.defaultdict(
        lambda: collections.defaultdict(tuple))
    logging.info("Reading data info from \"%s\" ...", data_info_file)
    self._data_info = data_lib.load_data_info(data_info_file)
    self._columns = []
    logging.info("Reading training_data from \"%s\" ...", training_data)
    self._training_df = pd.read_csv(training_data, delimiter="|",
                                    encoding=const.ENCODING)
    logging.info("Reading dev data from \"%s\" ...", dev_data)
    self._dev_df = pd.read_csv(dev_data, delimiter="|",
                               encoding=const.ENCODING)
    self._per_feature_dfs = {}

    genus_data = _get_associations_path(FLAGS.genus_filename)
    family_data = _get_associations_path(FLAGS.family_filename)
    neighborhood_data = _get_associations_path(FLAGS.neighborhood_filename)
    implicational_data = _get_associations_path(FLAGS.implicational_filename)
    self._load_stats(genus_data, family_data, neighborhood_data,
                     implicational_data)

  def _load_stats(self,
                  genus_data,
                  family_data,
                  neighborhood_data,
                  implicational_data):
    """Loads the association mappings from files.

    Args:
      genus_data: (string) Genera associations file.
      family_data: (string) Family associations file.
      neighborhood_data: (string) Areal associations file.
      implicational_data: (string) Implicational associations file.
    """
    logging.info("Reading genus_data from \"%s\" ...", genus_data)
    with open(genus_data, encoding=const.ENCODING) as stream:
      reader = csv.reader(stream, delimiter="|", quotechar='"')
      for row in reader:
        try:
          genus, feature, value, prob, count = row
          prob = float(prob)
          count = int(count)
          self._genus_values[genus][feature] = value, prob, count
        except ValueError:
          continue
    logging.info("Reading family_data from \"%s\" ...", family_data)
    with open(family_data, encoding=const.ENCODING) as stream:
      reader = csv.reader(stream, delimiter="|", quotechar='"')
      for row in reader:
        try:
          family, feature, value, prob, count = row
          prob = float(prob)
          count = int(count)
          self._family_values[family][feature] = value, prob, count
        except ValueError:
          continue
    logging.info("Reading neighborhood_data from \"%s\" ...", neighborhood_data)
    with open(neighborhood_data, encoding=const.ENCODING) as stream:
      reader = csv.reader(stream, delimiter="|", quotechar='"')
      for row in reader:
        try:
          neighborhood, feature, value, prob, count = row
          prob = float(prob)
          count = int(count)
          self._neighborhood_values[neighborhood][feature] = value, prob, count
        except ValueError:
          continue
    logging.info("Reading implicational_data from \"%s\" ...",
                 implicational_data)
    with open(implicational_data, encoding=const.ENCODING) as stream:
      reader = csv.reader(stream, delimiter="|", quotechar='"')
      for row in reader:
        try:
          f1, v1, f2, v2, prob, count, pprob, pcount = row
          prob = float(prob)
          pprob = float(pprob)
          count = int(count)
          pcount = int(pcount)
          self._implicational_values[f1, v1][f2] = (v2, prob, count, pprob,
                                                    pcount)
        except ValueError:
          continue
    # Creates the initial column headers
    self._mandatory_columns = [
        "wals_code", "latitude", "longitude", "target_feature", "target_value"]
    self._basic_columns = [
        "genus_majval", "genus_prob", "genus_count",
        "family_majval", "family_prob", "family_count",
        "neighbor_majval", "neighbor_prob", "neighbor_count"]
    self._columns = self._mandatory_columns + self._basic_columns
    self._categorical_features = set(
        ["target_value", "genus_majval", "family_majval", "neighbor_majval"])
    self._prob_features = set(["genus_prob", "family_prob", "neighbor_prob"])
    self._count_features = set(["genus_count", "family_count",
                                "neighbor_count"])
    self._implicational_features = set()
    for (f1, v1) in sorted(self._implicational_values):
      for f2 in sorted(self._implicational_values[f1, v1]):
        quintuple = self._bundle_implicationals(f1, v1, f2)
        for t in quintuple:
          self._implicational_features.add(t)
        self._categorical_features.add(quintuple[0])
        self._prob_features.add(quintuple[1])
        self._count_features.add(quintuple[2])
        self._prob_features.add(quintuple[3])
        self._count_features.add(quintuple[4])

  def _bundle_implicationals(self, f1, v1, f2):
    bundle = "{}@{}@{}".format(f1, v1, f2)
    return ("{}_majval".format(bundle),
            "{}_prob".format(bundle),
            "{}_count".format(bundle),
            "{}_pprob".format(bundle),
            "{}_pcount".format(bundle))

  def _adjust_counts_and_probs(self, tup, target_value, is_dev,
                               is_quint=False):
    """Given the tuple adjusts its count and probability estimates.

    Args:
      tup: (tuple) A triple or quintuple.
      target_value: (string) Value of the WALS feature.
      is_dev: (boolean) True if the tuple comes from the development data.
      is_quint: (boolean) True if `tup` is a quintuple, otherwise triple is
                assumed.
    Returns:
      Adjusted tuple.
    """
    if is_quint:
      try:
        v, p, n, pp, pn = tup
      except ValueError:
        return _DUMMY_QUINTUPLE
    else:
      try:
        v, p, n = tup
      except ValueError:
        return _DUMMY_TRIPLE
    # No need to adjust if we are running on the dev data:
    if is_dev or not target_value:
      return tup
    new_n = n - 1 if v == target_value else n
    if is_quint:
      new_pn = pn - 1 if v == target_value else pn
      try:
        return (v, new_n / (n / p - 1), new_n,
                new_pn / (pn / pp - 1), new_pn)
      except ZeroDivisionError:
        return _DUMMY_QUINTUPLE
    else:
      try:
        return v, new_n / (n / p - 1), new_n
      except ZeroDivisionError:
        return _DUMMY_TRIPLE

  def _process_language(self,
                        language_df,
                        target_feature,
                        target_value,
                        features,
                        implicational_features_for_languages,
                        is_dev=False):
    """Processes a single language."""
    wals_code = language_df["wals_code"]
    latitude = language_df["latitude"]
    longitude = language_df["longitude"]
    target_value_name = target_value if target_value else _UNKNOWN_STR
    language = [wals_code, latitude, longitude, target_feature,
                target_value_name]
    # Add genus triple.
    genus = language_df["genus"]
    if isinstance(genus, np.int64):
      logging.fatal("The data needs genera encoded as strings. "
                    "Found genus: \"%d\"", genus)
    try:
      genus_triple = self._adjust_counts_and_probs(
          self._genus_values[genus][target_feature], target_value, is_dev)
    except KeyError:
      genus_triple = _DUMMY_TRIPLE
    language.extend(genus_triple)
    family = language_df["family"]
    if isinstance(family, np.int64):
      logging.fatal("The data needs families encoded as strings. "
                    "Found family: \"%d\"", family)
    try:
      family_triple = self._adjust_counts_and_probs(
          self._family_values[family][target_feature], target_value, is_dev)
    except KeyError:
      family_triple = _DUMMY_TRIPLE
    language.extend(family_triple)

    neighborhood = "{},{}".format(latitude, longitude)
    try:
      neighborhood_triple = self._adjust_counts_and_probs(
          self._neighborhood_values[neighborhood][target_feature],
          target_value, is_dev)
    except KeyError:
      neighborhood_triple = _DUMMY_TRIPLE
    language.extend(neighborhood_triple)

    all_featvals = set()
    for f in features:
      v = language_df[f]
      all_featvals.add((f, v))
    all_featvals = sorted(list(all_featvals))
    implicational_features_for_language = {}
    f = target_feature
    for f1, v1 in all_featvals:
      if f1 == f:
        continue
      if (f1, v1) in self._implicational_values:
        if f in self._implicational_values[f1, v1]:
          implicational_quintuple = self._adjust_counts_and_probs(
              self._implicational_values[f1, v1][f], target_value, is_dev,
              is_quint=True)
          if implicational_quintuple == _DUMMY_QUINTUPLE:
            continue
          implicational_features_for_language[f1, v1, f] = (
              implicational_quintuple)
          implicational_features_for_languages.add((f1, v1, f))
    language.append(implicational_features_for_language)
    return language

  def _massage_df(self, df, feature_values):
    """Performs some very basic but costly conversions."""
    # Convert categorical columns to integers.
    columns = [col for col in df.columns if col in self._categorical_features]
    values = df[columns].values
    values_shape = values.shape
    values = [0 if val == "NA" or pd.isnull(val) else
              feature_values.index(val) + 1 for val in values.flatten()]
    df[columns] = np.reshape(values, values_shape)

    # Convert probs to logprobs.
    columns = [col for col in df.columns if col in self._prob_features]
    df[columns] = np.log(df[columns].values)

  def process_data(self, target_feature, force_recompute=False,
                   prediction_mode=False):
    """Processes the data for the supplied target feature.

    If `force_recompute` is disabled, the previously computed train and dev
    features are returned from the cache. Otherwise the features are always
    recomputed.

    If `prediction_mode` is disabled, an evaluation mode is assumed. In
    evaluation mode, we only consider features with valid (existing) feature
    values in the development set. This means that some of the languages in the
    development set may be ignored because no truth values for the feature in
    question are available. The proper test mode is enabled by enabling the
    `prediction_mode` flag. In this mode we always have to produce the features
    for every language in the development set.

    Args:
      target_feature: (string) Target WALS feature for which the data is needed.
      force_recompute: (boolean) Recomputes the data, if enabled, otherwise
                       returns cached data.
      prediction_mode: (boolean) If enabled, run in prediction mode, otherwise
                       evaluation mode is assumed.
    Returns:
      Pandas dataframes for features and labels for the given WALS feature.
    """
    if not force_recompute and target_feature in self._per_feature_dfs:
      return self._per_feature_dfs[target_feature]
    logging.info("Processing \"%s\" ...", target_feature)
    # Cumulate implicational features for this target feature for all languages
    # in train and dev.
    implicational_features_for_languages = set()
    training_languages = []
    training_feature_names = self._training_df.columns[7:]
    training = self._training_df.to_dict(orient="row")
    for language_df in training:
      if target_feature not in language_df:
        logging.error("Feature \"%s\" not in the training data", target_feature)
      target_value = language_df[target_feature]
      if pd.isnull(target_value):  # Nothing to predict.
        continue
      training_languages.append(self._process_language(
          language_df,
          target_feature,
          target_value,
          training_feature_names,
          implicational_features_for_languages,
          is_dev=False))
    dev_languages = []
    dev_feature_names = self._dev_df.columns[7:]
    dev = self._dev_df.to_dict(orient="row")
    for language_df in dev:
      if target_feature not in language_df:  # Nothing to predict.
        if not prediction_mode:  # Evaluation mode
          continue
        else:
          target_value = None  # Target value unknown.
      else:
        target_value = language_df[target_feature]
      if pd.isnull(target_value) and not prediction_mode:
        # Nothing to predict (feature empty): In evaluation mode simply skip
        # this language.
        continue
      dev_languages.append(self._process_language(
          language_df,
          target_feature,
          target_value,
          dev_feature_names,
          implicational_features_for_languages,
          is_dev=True))
    final_training_languages = []
    for language in training_languages:
      implicational_features_for_language = language[-1]
      new_language = language[:-1]
      for (f1, v1, f) in sorted(implicational_features_for_languages):
        if (f1, v1, f) in implicational_features_for_language:
          new_language.extend(implicational_features_for_language[f1, v1, f])
        else:
          new_language.extend(_DUMMY_QUINTUPLE)
      final_training_languages.append(new_language)
    final_dev_languages = []
    for language in dev_languages:
      implicational_features_for_language = language[-1]
      new_language = language[:-1]
      for (f1, v1, f) in sorted(implicational_features_for_languages):
        if (f1, v1, f) in implicational_features_for_language:
          new_language.extend(implicational_features_for_language[f1, v1, f])
        else:
          new_language.extend(_DUMMY_QUINTUPLE)
      final_dev_languages.append(new_language)
    columns = self._columns.copy()
    for (f1, v1, f) in sorted(implicational_features_for_languages):
      columns.extend(self._bundle_implicationals(f1, v1, f))
    training_df = pd.DataFrame(final_training_languages, columns=columns)
    dev_df = pd.DataFrame(final_dev_languages, columns=columns)
    feature_values = self._data_info[
        const.DATA_KEY_FEATURES][target_feature]["values"]
    if not feature_values:
      raise ValueError("No feature values found for \"%s\"" % target_feature)
    self._massage_df(training_df, feature_values)
    self._massage_df(dev_df, feature_values)
    self._per_feature_dfs[target_feature] = training_df, dev_df
    return training_df, dev_df

  def select_columns(self, df,
                     discard_counts=False,
                     discard_implicationals=False):
    """Given a data frame selectively updates the returned columns."""
    # TODO(rws): Maybe add other controls upon request.
    columns = df.columns
    keep_columns = self._mandatory_columns
    other_columns = columns[len(keep_columns):]
    for c in other_columns:
      if discard_counts and c in self._count_features:
        continue
      if discard_implicationals and c in self._implicational_features:
        continue
      keep_columns.append(c)
    return df[keep_columns]

  @property
  def categorical_features(self):
    return self._categorical_features

  @property
  def prob_features(self):
    return self._prob_features

  @property
  def count_features(self):
    return self._count_features

  @property
  def genus_values(self):
    return self._genus_values

  @property
  def family_values(self):
    return self._family_values

  @property
  def neighborhood_values(self):
    return self._neighborhood_values

  @property
  def implicational_values(self):
    return self._implicational_values
