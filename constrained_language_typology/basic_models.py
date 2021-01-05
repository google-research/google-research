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

"""Collection of basic algorithms.

This module provides the following simple algorithms:

[1] `BasicMajorityClass`: Simplest possible predictor providing estimates based
    on the global majority class counts.

[2] `BasicHaversineKNN`: Predictor based on the information provided by the
    closest (geographically) language.

[3] `BasicCladeMajorityClass`: Majority class predictor where the predictions
    are retrieved from the same language genus (if available), family or global
    majority class predictions. Country-based clades are supported as well.

[4] `BasicHaversineKNNWithClades`: Predictor that combines strategies [2] and
    [3]: When making predictions follow the predictions from the clades that are
    closest geographically.

[5] `BasicExperimentalNemo`: Simple predictor utilizing basic ensembling with
    majority voting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os

from absl import logging
import build_feature_matrix as feature_lib
import constants as const
import data_info as data_info_lib
import pandas as pd
import utils

# pylint: disable=g-long-lambda


def load_training_data(algo_name, training_data_dir, train_set_name):
  """Loads preprocessed training data into a data frame."""
  input_file = os.path.join(training_data_dir, train_set_name + ".csv")
  logging.info("[%s] Reading training data in \"%s\" ...",
               algo_name, input_file)
  df = pd.read_csv(input_file, sep="|", encoding=const.ENCODING)
  logging.info("[%s] Read %d languages.", algo_name, len(df))
  return df


def make_feature_maker(algo_name, training_data_dir,
                       train_set_name, dev_set_name):
  logging.info("[%s] Initializing feature maker ...", algo_name)
  train_path = os.path.join(training_data_dir, train_set_name + ".csv")
  dev_path = os.path.join(training_data_dir, dev_set_name + ".csv")
  data_info_path = data_info_lib.data_info_path_for_testing(training_data_dir)
  return feature_lib.FeatureMaker(train_path, dev_path, data_info_path)


def _names_with_biggest_counts(counts_dict):
  counts = [(name, count) for name, count in counts_dict.items()]
  names = [name for name, count in sorted(
      counts, key=lambda x: x[1], reverse=True)]
  return names


def collect_majority_stats(algo_name, df):
  """Collects majority class stats for several types of sources.

  Majority class counts are collected and returned for the following types:
    - All features globally,
    - Per language family,
    - Per language genus,
    - Per all the countries where the language is spoken.

  Args:
    algo_name: (string) Algorithm name.
    df: (pandas) Dataframe representing the corpus.

  Returns:
    A four-tuple with majority class mappings for different types (global,
    families, genera and countries).
  """
  global_value_counts = collections.defaultdict(
      lambda: collections.defaultdict(int))
  family_value_counts = collections.defaultdict(
      lambda: collections.defaultdict(
          lambda: collections.defaultdict(int)))
  genus_value_counts = collections.defaultdict(
      lambda: collections.defaultdict(
          lambda: collections.defaultdict(int)))
  country_value_counts = collections.defaultdict(
      lambda: collections.defaultdict(
          lambda: collections.defaultdict(int)))

  # Accumulate counts.
  logging.info("[%s] Initializing ...", algo_name)
  feature_offset = 7  # This is where the features start.
  feature_names = df.columns[feature_offset:]
  languages = df.to_dict(orient="row")
  for language_df in languages:
    for feature in feature_names:
      value = language_df[feature]
      if not pd.isnull(value):
        global_value_counts[feature][value] += 1
        family_value_counts[language_df["family"]][feature][value] += 1
        genus_value_counts[language_df["genus"]][feature][value] += 1
        if not pd.isnull(language_df["countrycodes"]):
          for country_code in language_df["countrycodes"].split(" "):
            country_value_counts[country_code][feature][value] += 1

  # Compute majority classes.
  global_majority_class = {}
  for feature in feature_names:
    names = _names_with_biggest_counts(global_value_counts[feature])
    global_majority_class[feature] = names[0]

  family_majority_class = collections.defaultdict(
      lambda: collections.defaultdict(str))
  for family in family_value_counts:
    for feature in feature_names:
      names = _names_with_biggest_counts(family_value_counts[family][feature])
      if names:
        family_majority_class[family][feature] = names[0]

  genus_majority_class = collections.defaultdict(
      lambda: collections.defaultdict(str))
  for genus in genus_value_counts:
    for feature in feature_names:
      names = _names_with_biggest_counts(genus_value_counts[genus][feature])
      if names:
        genus_majority_class[genus][feature] = names[0]

  country_majority_class = collections.defaultdict(
      lambda: collections.defaultdict(str))
  for country in country_value_counts:
    for feature in feature_names:
      names = _names_with_biggest_counts(country_value_counts[country][feature])
      if names:
        country_majority_class[country][feature] = names[0]

  return (global_majority_class, family_majority_class, genus_majority_class,
          country_majority_class)


class BasicHaversineKNN(object):
  """Basic Haversine-distance based search.

  This is a very silly little algorithm. We assume that each language is fully
  described by a single lat/long coordinate.

  Important note: The algorithm assumes that the training data contains string
  feature values rather than ints.
  """

  def __init__(self, num_best):
    self._name = "BasicHaversineKNN"
    self._df = None
    self._num_best = num_best
    self._distance_cache = {}
    self._language_distances = {}

  def _compute_or_fetch(self, code_one, loc_one, code_two, loc_two):
    """Computes the distance between two languages or fetches it."""
    key_one = (code_one, code_two)
    key_two = (code_two, code_one)
    if key_one in self._distance_cache or key_two in self._distance_cache:
      return self._distance_cache[key_one]
    else:
      dist = utils.haversine_distance(loc_one, loc_two)
      self._distance_cache[key_one] = dist
      self._distance_cache[key_two] = dist
      return dist

  def init(self, training_data_dir, train_set_name):
    """Initializes the model."""
    self._df = load_training_data(self._name, training_data_dir,
                                  train_set_name)

  def prepare_target(self, target_df):
    """Precomputes the distances given the target."""
    target_code = target_df["wals_code"]
    target_lat = target_df["latitude"]
    target_long = target_df["longitude"]
    distances = []
    languages = self._df.to_dict(orient="row")
    for index, language_df in enumerate(languages):
      lang_code = language_df["wals_code"]
      dist = self._compute_or_fetch(
          lang_code, (language_df["latitude"], language_df["longitude"]),
          target_code, (target_lat, target_long))
      distances.append((index, dist))
    distances = sorted(distances, key=lambda x: float(x[1]))
    distances = [index for index, dist in distances]
    self._language_distances[target_code] = distances

  def predict(self, target_df, context_features, feature):
    """Predicts the feature given the context provided by target_df.

    The (valid) context features are provided in the "context_features"
    mapping from feature names to their valid values for this language.

    If the best feature happens to be missing, we move on to the next best
    languages, which may be very suboptimal.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature: (string) Feature name.

    Returns:
      N-best list.
    """
    del context_features  # Don't use context.

    target_code = target_df["wals_code"]
    best_indexes = self._language_distances[target_code]
    values = []
    for index in best_indexes:
      value = self._df.at[index, feature]
      if not pd.isnull(value):  # Only fill in valid features.
        values.append(value)
      if len(values) == self._num_best:
        break
    if not values:
      # Back-off plan for the case where the feature is completely sparse.
      # Not likely to happen, but select the first language "just in case".
      values.append(0)
    return values


class BasicMajorityClass(object):
  """Basic majority class estimator.

  Another very silly algorithm where for an unknown value we always return
  the majority value for that class computed over all the features globally.
  """

  def __init__(self):
    self._name = "BasicMajorityClass"
    self._df = None

  def init(self, training_data_dir, train_set_name):
    """Initializes the model."""
    self._df = load_training_data(self._name, training_data_dir,
                                  train_set_name)
    self._global_majority_class, _, _, _ = collect_majority_stats(
        self._name, self._df)

  def prepare_target(self, target_df):
    """Precomputes the distances given the target."""
    pass

  def predict(self, target_df, context_features, feature):
    """Predicts the feature given the context provided by target_df.

    The (valid) context features are provided in the "context_features"
    mapping from feature names to their valid values for this language.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature: (string) Feature name.

    Returns:
      This estimator always returns single-best list.
    """
    del target_df, context_features  # Use global counts only.

    return [self._global_majority_class[feature]]


class BasicCladeMajorityClass(object):
  """Basic majority class estimator using clades.

  Another very silly algorithm where for an unknown value we always return
  the majority value for the clade in question (genus, if present or family).
  """

  def __init__(self, include_countries=True):
    self._name = "BasicCladeMajorityClass"
    self._df = None
    self._include_countries = include_countries

  def init(self, training_data_dir, train_set_name):
    """Initializes the model."""
    self._df = load_training_data(self._name, training_data_dir,
                                  train_set_name)
    (self._global_majority_class, self._family_majority_class,
     self._genus_majority_class,
     self._country_majority_class) = collect_majority_stats(
         self._name, self._df)

  def prepare_target(self, target_df):
    """Precomputes the distances given the target."""
    pass

  def predict(self, target_df, context_features, feature):
    """Predicts the feature given the context provided by target_df.

    The (valid) context features are provided in the "context_features"
    mapping from feature names to their valid values for this language.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature: (string) Feature name.

    Returns:
      This estimator always returns single-best list.
    """
    del context_features  # Don't use context.

    # Most frequent value for a feature per family.
    family_value = None
    family = target_df.family
    if feature in self._family_majority_class[family]:
      family_value = self._family_majority_class[family][feature]

    # Most frequent value for a feature per genus.
    genus_value = None
    genus = target_df.genus
    if feature in self._genus_majority_class[genus]:
      genus_value = self._genus_majority_class[genus][feature]

    # Value and its frequency computed over multiple countries.
    countries = []
    if not pd.isnull(target_df.countrycodes):
      countries = target_df.countrycodes.split(" ")
    country_values = collections.defaultdict(int)
    for country in countries:
      if feature in self._country_majority_class[country]:
        value = self._country_majority_class[country][feature]
        country_values[value] += 1
    country_values = list(country_values.items())
    country_values.sort(key=lambda x: x[1], reverse=True)

    if not self._include_countries:
      # Don't include country information.
      if not genus_value and not family_value:
        return [self._global_majority_class[feature]]
      elif not genus_value and family_value:
        return [family_value]
      elif genus_value:
        return [genus_value]
    else:
      # Include country information.
      if not country_values and not genus_value and not family_value:
        # No clade and country information whatsoever.
        return [self._global_majority_class[feature]]
      elif country_values and not genus_value and not family_value:
        # Only country information available. Select the most frequent value
        # among all the countries.
        return [country_values[0][0]]
      elif not genus_value and family_value:
        # Family is available but no genus. Include country codes, if available.
        country_values.append((family_value, 1))
        country_values.sort(key=lambda x: x[1], reverse=True)
        return [country_values[0][0]]
      elif genus_value:
        # Give more weight to genus than the family. Include country codes,
        # if available.
        if family_value:
          country_values.append((family_value, 1))
        country_values.append((genus_value, 2))
        country_values.sort(key=lambda x: x[1], reverse=True)
        return [country_values[0][0]]
      else:
        return [family_value]


class BasicHaversineKNNWithClades(BasicHaversineKNN):
  """Basic Haversine-distance based search with clade (genus/family) info..

  We assume that each language is fully described by a single lat/long
  coordinate and the clade information (genera and families). The clades
  provide information for filtering the best candidates.
  """

  def __init__(self, num_best):
    super().__init__(num_best)
    self._name = "BasicHaversineKNNWithClades"

  def init(self, training_data_dir, train_set_name):
    """Initializes the model."""
    super().init(training_data_dir, train_set_name)
    self._same_genus_distances = {}
    self._same_family_distances = {}

  def prepare_target(self, target_df):
    """Precomputes the distances given the target."""
    super().prepare_target(target_df)
    target_code = target_df["wals_code"]
    target_genus = target_df["genus"]
    target_family = target_df["family"]
    distances = self._language_distances[target_code]

    # Prune the distances by target genus.
    same_genus_distances = []
    for index in distances:
      if self._df.at[index, "genus"] == target_genus:
        same_genus_distances.append(index)
    self._same_genus_distances[target_code] = same_genus_distances

    # Prune the distances by target family.
    same_family_distances = []
    for index in distances:
      if self._df.at[index, "family"] == target_family:
        same_family_distances.append(index)
    self._same_family_distances[target_code] = same_family_distances

  def predict(self, target_df, context_features, feature):
    """Predicts the feature given the context provided by target_df.

    The (valid) context features are provided in the "context_features"
    mapping from feature names to their valid values for this language.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature: (string) Feature name.

    Returns:
      This estimator always returns single-best list.
    """
    target_code = target_df["wals_code"]

    # Search within genera first.
    values = []
    genus_indexes = self._same_genus_distances[target_code]
    for index in genus_indexes:
      value = self._df.at[index, feature]
      if not pd.isnull(value):  # Only fill in valid features.
        values.append(value)
      if len(values) == self._num_best:
        break
    if len(values) == self._num_best:
      return values

    # Next search within families.
    family_indexes = self._same_family_distances[target_code]
    for index in family_indexes:
      if index in values:
        continue
      value = self._df.at[index, feature]
      if not pd.isnull(value):  # Only fill in valid features.
        values.append(value)
      if len(values) == self._num_best:
        break
    if len(values) == self._num_best:
      return values

    # Fall back to global distance list.
    best_indexes = self._language_distances[target_code]
    for index in best_indexes:
      if index in values:
        continue
      value = self._df.at[index, feature]
      if not pd.isnull(value):  # Only fill in valid features.
        values.append(value)
      if len(values) == self._num_best:
        break

    if not values:
      # Back-off plan for the case where the feature is completely sparse.
      # Not likely to happen, but select the first language "just in case".
      values.append(0)
    return values


class BasicExperimentalNemo(BasicHaversineKNNWithClades):
  """Very experimental NEMO model based on multiple associations."""

  def __init__(self, num_best):
    super().__init__(num_best)
    self._name = "BasicExperimentalNemo"

  def init(self, training_data_dir, train_set_name, dev_set_name):
    """Initializes the model."""
    super().init(training_data_dir, train_set_name)
    logging.info("Initializing feature maker ...")
    self._feature_maker = make_feature_maker(
        self._name, training_data_dir, train_set_name, dev_set_name)
    (self._global_majority_class, self._family_majority_class,
     self._genus_majority_class,
     self._country_majority_class) = collect_majority_stats(
         self._name, self._df)

  def prepare_target(self, target_df):
    """Precomputes the distances given the target."""
    # pylint: disable=useless-super-delegation
    super().prepare_target(target_df)
    # pylint: enable=useless-super-delegation

  def predict(self, target_df, context_features, feature):
    """Predicts the feature given the context provided by target_df.

    The (valid) context features are provided in the "context_features"
    mapping from feature names to their valid values for this language.

    Args:
      target_df: (pandas) Dataframe representing language.
      context_features: (dict) Contextual features.
      feature: (string) Feature name.

    Returns:
      This estimator always returns single-best list.
    """
    # Get the candidate for the best neighbor.
    target_code = target_df["wals_code"]
    closest_indexes = self._language_distances[target_code]
    neighborhoods = self._feature_maker.neighborhood_values
    target_loc = "%s,%s" % (target_df["latitude"], target_df["longitude"])
    n_cand = (None, 0.0, -1)
    if target_loc in neighborhoods:
      if feature in neighborhoods[target_loc]:
        n_cand = neighborhoods[target_loc][feature]
    if not n_cand[0]:  # No neighborhood information.
      best_index = closest_indexes[0]
      loc = "%s,%s" % (self._df.at[best_index, "latitude"],
                       self._df.at[best_index, "longitude"])
      if loc in neighborhoods:
        if feature in neighborhoods[loc]:
          n_cand = neighborhoods[loc][feature]

    # Direct distance prediction from the parent.
    d_cand = (super().predict(target_df, context_features, feature)[0], 0.0, 1)
    assert d_cand

    # Look at genus.
    target_genus = target_df["genus"]
    genera = self._feature_maker.genus_values
    g_cand = (None, 0.0, -1)
    if target_genus in genera:
      if feature in genera[target_genus]:
        g_cand = genera[target_genus][feature]

    # Look at family.
    target_family = target_df["family"]
    families = self._feature_maker.family_values
    f_cand = (None, 0.0, -1)
    if target_family in families:
      if feature in families[target_family]:
        f_cand = families[target_family][feature]

    # Use simple majority voting, ignore empties for now.
    cands = [("neighborhood", n_cand), ("genus", g_cand), ("family", f_cand)]
    cand_values = [cand[1][0] for cand in cands]
    best_count = collections.Counter(cand_values).most_common(2)[0]
    if best_count[0] and best_count[1] > 1:
      return [best_count[0]]  # At least two estimates agree.

    # TODO(agutkin): The following should be reviewed and improved.

    # Special case when nothing is available. Fallback to majority class.
    if not g_cand[0] and not f_cand[0] and not n_cand[0]:
      return [self._global_majority_class[feature]]

    # Silly heuristics: Sort by counts.
    cands.sort(key=lambda x: x[1][2], reverse=True)
    print(cands)
    top_cand = cands[0][1][0]
    return [top_cand]
