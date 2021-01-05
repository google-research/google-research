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

r"""Replaces random feature values with "?"/null.

Sample usage:
-------------
  python3 randomly_question_features_main.py \
    --input /var/tmp/sigtyp/dev.csv \
    --output /tmp/dev.csv \
    --output_nulles /tmp/dev_nulled.csv \
    --json /tmp/dev.json \
    --proportion 0.2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import random

from absl import app
from absl import flags

import constants as const

flags.DEFINE_string(
    "input", "",
    "Input CSV file with specified features.")

flags.DEFINE_string(
    "output", "",
    "Output CSV file with some known features replaced with `?`.")

flags.DEFINE_string(
    "output_nulled", "",
    "Output CSV file with some known features replaced with NULL.")

flags.DEFINE_string(
    "json", "",
    "Output JSON file with table of features to predict per language.")

flags.DEFINE_float(
    "proportion", 0.05,
    "Rough proportion of specified features to replace with `?`.")

FLAGS = flags.FLAGS


def randomly_question_features():
  """Randomly fills in valid feature values with the unknown markers."""
  features_to_predict = {}
  nulled_rows = []
  rows = []
  with open(FLAGS.input, "r", encoding=const.ENCODING) as s:
    reader = csv.reader(s, delimiter="|", quotechar='"')
    begin = True
    for row in reader:
      if begin:
        hdr = row
        begin = False
        nulled_rows.append(hdr)
        rows.append(hdr)
        continue
      language = row[0]
      fixed = row[:7]
      variables = row[7:]
      new_nulled_variables = []
      new_variables = []
      i = 7
      for v in variables:
        if not v:
          new_nulled_variables.append(v)
          new_variables.append(v)
        elif random.random() < FLAGS.proportion:
          new_nulled_variables.append("")
          new_variables.append("?")
          if language not in features_to_predict:
            features_to_predict[language] = []
          features_to_predict[language].append(hdr[i])
        else:
          new_nulled_variables.append(v)
          new_variables.append(v)
        i += 1
      nulled_rows.append(fixed + new_nulled_variables)
      rows.append(fixed + new_variables)
  with open(FLAGS.output_nulled, "w") as s:
    writer = csv.writer(s, delimiter="|", quotechar='"')
    for row in nulled_rows:
      writer.writerow(row)
  with open(FLAGS.output, "w") as s:
    s.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        "wals_code", "name",
        "latitude", "longitude",
        "genus", "family",
        "countrycodes", "features"))
    hdr = rows[0]
    for row in rows[1:]:
      s.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t".format(*(row[:7])))
      rest = []
      i = 7
      for val in row[7:]:
        if val:
          rest.append("{}={}".format(hdr[i], val))
        i += 1
      s.write("|".join(rest))
      s.write("\n")
  with open(FLAGS.json, "w") as s:
    s.write("{}".format(features_to_predict))


def main(unused_argv):
  randomly_question_features()


if __name__ == "__main__":
  app.run(main)
