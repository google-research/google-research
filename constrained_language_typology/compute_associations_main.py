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

r"""Computation of feature associations.

Computes:

 1) Most likely value for a given feature given the clade (genetic
    preference) for genera and families.

 2) Most likely value2 for a given feature2 given feature1, value1
    (implicational feature preference).

Usage, e.g.:

First run:

  python3 sigtyp_reader_main.py \
    --sigtyp_dir ~/ST2020-master/data \
    --output_dir=/var/tmp/sigtyp

Then, using the defaults:

  python3 compute_associations_main.py \
    --training_data=/var/tmp/sigtyp/train.csv \
    --dev_data=/var/tmp/sigtyp/dev.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import os

from absl import app
from absl import flags

import compute_associations  # pylint: disable=[unused-import]
import constants as const
import pandas as pd
import utils

# pylint: disable=g-long-lambda

flags.DEFINE_string(
    "training_data", "",
    "Training data in CSV file format with the `|` column separator. "
    "This format is produced by `sigtyp_reader_main.py`.")

flags.DEFINE_string(
    "dev_data", None,
    "Development data in CSV file format with the `|` column separator. "
    "This format is produced by `sigtyp_reader_main.py`.")

flags.DEFINE_float(
    "close_enough", 2500,
    "Distance in kilometers between two languages to count as 'close enough' "
    "to be in the same neighborhood")

FLAGS = flags.FLAGS


def write_neighborhoods(path, neighborhoods):
  """Writes neighbourhood associations to a file."""
  with open(path, "w") as stream:
    # Write out for max value
    #
    # Lat,Lng for language
    # Feature
    # Value
    # Probability of value given feature and neighborhood
    # Total counts for feature+value+neighborhood
    stream.write("{}|{}|{}|{}|{}\n".format(
        "lat,lng", "f", "v", '"p(v|f, c)"', "n(f, c)"))
    for latlng in neighborhoods:
      for f in neighborhoods[latlng]:
        tot = 0
        max_c = 0
        for v in neighborhoods[latlng][f]:
          if neighborhoods[latlng][f][v] > max_c:
            max_c = neighborhoods[latlng][f][v]
            max_v = v
          tot += neighborhoods[latlng][f][v]
        stream.write("{},{}|{}|{}|{:0.3f}|{}\n".format(
            latlng[0], latlng[1], f, max_v, max_c / tot, tot))


def write_implicational(path, implicational, implicational_prior):
  """Writes implicational associations to a file."""
  with open(path, "w") as stream:
    # Write out for max value
    #
    # Feature1
    # Value1
    # Feature2
    # Value2
    # Probabilty of Value2 given Feature1, Value1 and Feature2
    # Total counts for Feature1, Value1 and Feature2
    # Probability of Value2 given Feature2
    # Total counts for Value2, Feature2
    stream.write("{}|{}|{}|{}|{}|{}|{}|{}\n".format(
        "f1", "v1", "f2", "v2", '"p(v2|f1, v1, f2)"', "n(f1, v1, f2)",
        '"p(v2|f2)"', "n(f2, v2)"))
    for (f1, v1) in implicational:
      for f2 in implicational[f1, v1]:
        tot = 0
        max_c = 0
        for v2 in implicational[f1, v1][f2]:
          if implicational[f1, v1][f2][v2] > max_c:
            max_c = implicational[f1, v1][f2][v2]
            max_v2 = v2
          tot += implicational[f1, v1][f2][v2]
        tot_f2 = 0
        for v2 in implicational_prior[f2]:
          tot_f2 += implicational_prior[f2][v2]
        prior_v2_prob = implicational_prior[f2][max_v2] / tot_f2
        stream.write("{}|{}|{}|{}|{:0.3f}|{}|{:0.3f}|{}\n".format(
            f1, v1, f2, max_v2, max_c / tot, tot, prior_v2_prob, tot_f2))


def write_clades(path, clades):
  """Writes glade information to a file."""
  with open(path, "w") as stream:
    # Write out for max value
    #
    # Clade
    # Feature
    # Value
    # Probability of value given feature and clade
    # Total counts for feature+clade
    stream.write("{}|{}|{}|{}|{}\n".format(
        "clade", "f", "v", '"p(v|f, c)"', "n(f, c)"))
    for clade in clades:
      for f in clades[clade]:
        tot = 0
        max_c = 0
        for v in clades[clade][f]:
          if clades[clade][f][v] > max_c:
            max_c = clades[clade][f][v]
            max_v = v
          tot += clades[clade][f][v]
        stream.write("{}|{}|{}|{:0.3f}|{}\n".format(
            clade, f, max_v, max_c / tot, tot))


def find_close_languages(lat1, lng1, languages, distance_cache):
  """Given latitude/longitude coordinates finds the nearest language."""
  close_language_indices = []
  for i, language in enumerate(languages):
    lat2 = language["latitude"]
    lng2 = language["longitude"]
    loc1 = (float(lat1), float(lng1))
    loc2 = (float(lat2), float(lng2))
    if (loc1, loc2) not in distance_cache:
      dist = utils.haversine_distance((float(lat1), float(lng1)),
                                      (float(lat2), float(lng2)))
      distance_cache[(loc1, loc2)] = dist
      distance_cache[(loc2, loc1)] = dist
    else:
      dist = distance_cache[(loc1, loc2)]
    if dist < FLAGS.close_enough:
      close_language_indices.append(i)
  return close_language_indices


def correlate_features_for_training():
  """Computes all the feature associations required for training a model."""
  training = pd.read_csv(FLAGS.training_data, delimiter="|",
                         encoding=const.ENCODING)
  features = training.columns[7:]
  clades = collections.defaultdict(
      lambda:
      collections.defaultdict(
          lambda:
          collections.defaultdict(
              lambda:
              collections.defaultdict(int))))
  implicational = collections.defaultdict(
      lambda:
      collections.defaultdict(
          lambda:
          collections.defaultdict(int)))
  # Whereas the implicationals collect the conditional probability of v2, given
  # f1,v1 and f2, this just collects the conditional probability of v2 given
  # v1. If the latter is also high, then the fact that the former is high is
  # probably of less interest.
  implicational_prior = collections.defaultdict(
      lambda:
      collections.defaultdict(int))
  neighborhoods = collections.defaultdict(
      lambda:
      collections.defaultdict(
          lambda:
          collections.defaultdict(int)))

  feature_frequency = collections.defaultdict(int)
  distance_cache = {}
  training_list = training.to_dict(orient="row")
  for language_df in training_list:
    genus = language_df["genus"]
    family = language_df["family"]
    for f1 in features:
      v1 = language_df[f1]
      if pd.isnull(v1):
        continue
      clades["genus"][genus][f1][v1] += 1
      clades["family"][family][f1][v1] += 1
      feature_frequency[f1, v1] += 1
      for f2 in features:
        if f1 == f2:
          continue
        v2 = language_df[f2]
        if pd.isnull(v2):
          continue
        implicational[f1, v1][f2][v2] += 1
    for f2 in features:
      v2 = language_df[f2]
      if pd.isnull(v2):
        continue
      implicational_prior[f2][v2] += 1
    # Find nearby languages
    lat1 = language_df["latitude"]
    lng1 = language_df["longitude"]
    close_language_indices = find_close_languages(
        lat1, lng1, training_list, distance_cache)
    if len(close_language_indices) == 1:
      continue
    for f1 in features:
      for k in close_language_indices:
        v1 = training_list[k][f1]
        if pd.isnull(v1):
          continue
        neighborhoods[lat1, lng1][f1][v1] += 1

  if FLAGS.dev_data:
    # If we are also processing the development data, make sure that we also
    # provide neighborhoods for the lat,lng for each language in the development
    # data --- of course only actually using data from training.
    development = pd.read_csv(FLAGS.dev_data, delimiter="|",
                              encoding=const.ENCODING)
    development_list = development.to_dict(orient="row")
    for language_df in development_list:
      lat1 = language_df["latitude"]
      lng1 = language_df["longitude"]
      close_language_indices = find_close_languages(
          lat1, lng1, training_list, distance_cache)
      if len(close_language_indices) == 1:
        continue
      for f1 in features:
        for k in close_language_indices:
          v1 = training_list[k][f1]
          if pd.isnull(v1):
            continue
          neighborhoods[lat1, lng1][f1][v1] += 1

  clade_types = [("genus", FLAGS.genus_filename),
                 ("family", FLAGS.family_filename)]
  for clade_type, clade_filename in clade_types:
    write_clades(os.path.join(FLAGS.association_dir, clade_filename),
                 clades[clade_type])
  write_neighborhoods(os.path.join(
      FLAGS.association_dir, FLAGS.neighborhood_filename), neighborhoods)
  write_implicational(os.path.join(
      FLAGS.association_dir, FLAGS.implicational_filename),
                      implicational, implicational_prior)


def main(unused_argv):
  correlate_features_for_training()


if __name__ == "__main__":
  app.run(main)
