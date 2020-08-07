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

"""Simple example demonstrating the use of feature matrix builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import app
from absl import flags

import build_feature_matrix as lib
import compute_associations  # pylint: disable=[unused-import]

# These should all point to wherever you computed the
# initial data.
flags.DEFINE_string(
    "dev_data", "/tmp/dev.csv",
    "Development data file.")

flags.DEFINE_string(
    "training_data", "/tmp/train.csv",
    "Training data file.")

flags.DEFINE_string(
    "data_info", "/tmp/data_info_train.json.gz",
    "Training info file.")

FLAGS = flags.FLAGS


def main(unused_argv):
  feature_maker = lib.FeatureMaker(
      FLAGS.training_data,
      FLAGS.dev_data,
      FLAGS.data_info)
  training_df, dev_df = feature_maker.process_data(
      "Order_of_Subject,_Object_and_Verb")
  long_implicational = (
      "The_Position_of_Negative_Morphemes_in_SOV_Languages"
      "@18 SV&OV&NegV@Order_of_Subject,_Object_and_Verb_majval")
  assert "family_majval" in dev_df.columns
  assert "family_count" in dev_df.columns
  assert long_implicational in dev_df.columns
  assert long_implicational in training_df.columns
  non_zeroes = []
  for fname in training_df.columns:
    if ("majval" in fname and
        "genus" not in fname and
        "family" not in fname and
        "neighborhood" not in fname):
      for i in training_df[fname]:
        if i:
          non_zeroes.append(i)
  # Show that there are non-zero (non-NA) entries for implicationals:
  assert non_zeroes
  non_zeroes = []
  for fname in dev_df.columns:
    if ("majval" in fname and
        "genus" not in fname and
        "family" not in fname and
        "neighborhood" not in fname):
      for i in dev_df[fname]:
        if i:
          non_zeroes.append(i)
  # Show that there are non-zero (non-NA) entries for implicationals:
  assert non_zeroes

  # Remove some of the columns.
  #
  # Obviously if you use this make sure you do the same thing to both training
  # and dev.
  smaller_dev_df = feature_maker.select_columns(
      dev_df, discard_counts=True, discard_implicationals=True)
  assert "wals_code" in smaller_dev_df.columns
  assert "target_value" in smaller_dev_df.columns
  assert long_implicational not in smaller_dev_df.columns
  assert "family_count" not in smaller_dev_df.columns
  # Remove some different columns
  smaller_dev_df = feature_maker.select_columns(
      dev_df, discard_counts=True, discard_implicationals=False)
  assert "wals_code" in smaller_dev_df.columns
  assert "target_value" in smaller_dev_df.columns
  assert long_implicational in smaller_dev_df.columns
  assert "family_count" not in smaller_dev_df.columns

  # Try another feature
  training_df, dev_df = feature_maker.process_data("Hand_and_Arm")
  assert "family_majval" in dev_df.columns
  assert "family_count" in dev_df.columns
  long_implicational = (
      "Number_of_Cases@9 Exclusively borderline case-marking"
      "@Hand_and_Arm_majval")
  assert long_implicational in dev_df.columns
  for c in dev_df.columns:
    assert c in training_df.columns


if __name__ == "__main__":
  app.run(main)
