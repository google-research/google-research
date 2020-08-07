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

r"""Checks that we provide values for all the requested features.

Sample usage:
-------------
  python3 sanity_check_results_main.py \
    --test_features_to_predict /var/tmp/sigtyp/test_features_to_predict.json \
    --submission data/results/NEMO_system1_assoc-train-dev.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import app
from absl import flags

flags.DEFINE_string(
    "test_features_to_predict", "",
    "JSON of features needed for each language.")

flags.DEFINE_string(
    "submission", "",
    "File in SIGTYP format that contains the predictions.")

FLAGS = flags.FLAGS


def run_check():
  """Runs the sanity check."""

  def lfix(wals_code):
    if wals_code == "nan":
      return "nxn"
    return wals_code

  def lunfix(wals_code):
    if wals_code == "nxn":
      return "nan"
    return wals_code

  warning = False
  with open(FLAGS.test_features_to_predict) as s:
    feature_table = eval(s.read())  # pylint: disable=eval-used
  with open(FLAGS.submission) as s:
    hdr = True
    for line in s:
      if hdr:
        hdr = False
        continue
      line = line.strip().split("\t")
      wals_code = lfix(line[0])
      language = line[1]
      features = line[7]
      needed_feats = feature_table[wals_code]
      seen_feats = set()
      for featval in features.split("|"):
        feat, val = featval.split("=", 1)
        if feat in needed_feats:
          print("{} ({}): {} -> {}".format(
              lunfix(wals_code), language, feat, val))
          seen_feats.add(feat)
      for feat in needed_feats:
        if feat not in seen_feats:
          warning = True
          print("Warning: {} ({}) has no value for {}".format(
              lunfix(wals_code), language, feat))
  assert not warning


def main(unused_argv):
  run_check()

if __name__ == "__main__":
  app.run(main)
