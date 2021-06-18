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

r"""Tool for computing lexical scores (aka L-scores) from neural training data.

The data needs to be in the format required for the neural model training. The
format of the output tsv file is tab-separated rows corresponding to each
language. The rows consist of four columns: corpus name, number of unique
pronunciations, number of unique spellings and L-score.

Example:
--------
For biblical training data generated with Epitran using `bible_epitran_data.py`
in ${BIBLE_DIR}, run:

  > python3 neural_data_lscores_main.py \
      --training_data_dir ${BIBLE_DIR} \
      --output_tsv /tmp/lscores.tsv

Dependencies:
-------------
  absl-py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string(
    "training_data_dir", "",
    "Directory containing training/test data files in csv format for "
    "individual languages, one file per language.")

flags.DEFINE_string(
    "output_tsv", "",
    "Tab-separated output file with the results.")

flags.DEFINE_bool(
    "only_use_training", True,
    "Only use the `train` designated lines of the training files.")

flags.DEFINE_bool(
    "case_free", False,
    "If enabled, always ignores case in orthography.")

FLAGS = flags.FLAGS

_ENCODING = "utf8"

_NULL_PRON = "**NULLPRON**"


def _process_file(filename):
  """Processes csv (tab-separated) file for a single language."""
  path = os.path.join(FLAGS.training_data_dir, filename)
  logging.info("Processing \"%s\" ...", path)

  # Collect pronunciation->spelling mappings.
  pron_spellings = collections.defaultdict(set)
  pron_counts = collections.defaultdict(int)
  unique_spellings = set()
  row_id = 0
  with open(path, encoding=_ENCODING) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
      row_id += 1
      if len(row) != 2:
        raise ValueError("%s: [row %d] Invalid row: %s" % (
            filename, row_id, row))
      if FLAGS.only_use_training and "train_" not in row[0]:
        continue
      tokens = [token for token in row[1].split(" ") if token]
      for token in tokens:
        parts = token.split("/")
        if len(parts) != 2:
          logging.warning("%s: Invalid token: %s", filename, token)
          continue
        spelling, pron = parts
        if FLAGS.case_free:
          spelling = spelling.lower()
        if pron and pron != _NULL_PRON:
          # For OOVs, the pron may me empty (e.g., Swedish, English).
          unique_spellings.add(spelling)
          pron_spellings[pron].add(spelling)
          pron_counts[pron] += 1

  # Compute token-based L-score.
  spell_sum = 0.0
  total_prons = 0
  for pron in pron_counts:
    total_prons += pron_counts[pron]
    spell_sum += pron_counts[pron] * len(pron_spellings[pron])
  lscore_token = spell_sum / total_prons

  # Compute type-based L-score.
  spell_sum = 0.0
  for spellings in pron_spellings.values():
    spell_sum += len(spellings)
  num_unique_prons = len(pron_spellings)
  lscore_type = spell_sum / num_unique_prons
  return (filename.rstrip(".tsv"), num_unique_prons, len(unique_spellings),
          lscore_type, lscore_token)


def main(unused_argv):
  if not FLAGS.training_data_dir:
    raise ValueError("Specify --training_data_dir!")
  if not FLAGS.output_tsv:
    raise ValueError("Supply --output_tsv!")

  results = []
  for filename in os.listdir(FLAGS.training_data_dir):
    if not filename.endswith(".tsv"):
      continue
    results.append(_process_file(filename))
  if not results:
    raise ValueError("No files with `.tsv` extension found!")
  logging.info("Processed %d data files.", len(results))

  logging.info("Writing \"%s\" ...", FLAGS.output_tsv)
  with open(FLAGS.output_tsv, mode="w", encoding=_ENCODING) as f:
    for result in results:
      f.write("%s\t%d\t%d\t%f\t%f\n" % (result[0], result[1], result[2],
                                        result[3], result[4]))


if __name__ == "__main__":
  app.run(main)
