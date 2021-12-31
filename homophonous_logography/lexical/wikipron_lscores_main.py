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

r"""Simple tool for computing $L$-scores from Wikipron data.

Wikipron contains multilingual pronunciation data mined from Wiktionary.
The project can be found here: https://github.com/kylebgorman/wikipron.

Produces tab-separated file with the following format:
  - Language code
  - Language name.
  - Type of the lexicon: phonemic or phonetic
  - Number of unique pronunciations.
  - Number of unique spellings.
  - $L$-score.

Sample output:
--------------
  bul     Bulgarian       phonetic        239     233     1.000000

Example:
--------
Given the Wikipron installation in ${WIKIPRON_DIR}:

 > python3 wikipron_lscores_main.py \
     --input_dir ${WIKIPRON_DIR}/data/wikipron/tsv \
     --output_tsv lscores.tsv \
     --sort_by score

Dependencies:
-------------
  absl-py
  pycountry
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from absl import app
from absl import flags
from absl import logging
import pycountry as pyc

# Separator used in the filename.
_NAME_SEPARATOR = "_"

flags.DEFINE_string(
    "input_dir", "",
    "Directory containing Wikipron lexicons in tsv format.")

flags.DEFINE_string(
    "output_tsv", "",
    "Tab-separated output file with the results.")

flags.DEFINE_string(
    "sort_by", "language",
    "Sorting criterion. One of: \"language\", \"score\".")

FLAGS = flags.FLAGS


def _compute_stats(filename):
  """Accumulates the statistics and computes the (type) L-score."""
  pron_spellings = {}
  unique_spellings = set()
  with open(os.path.join(FLAGS.input_dir, filename),
            mode="r", encoding="utf8") as f:
    reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
      pron = row[1]
      spelling = row[0]
      unique_spellings.add(spelling)
      if pron not in pron_spellings:
        pron_spellings[pron] = set()
      pron_spellings[pron].add(spelling)

  spell_sum = 0.0
  for spellings in pron_spellings.values():
    spell_sum += len(spellings)
  num_unique_prons = len(pron_spellings)
  lscore = spell_sum / num_unique_prons
  return num_unique_prons, len(unique_spellings), lscore


def _process_lexicon_file(filename):
  """Processes single lexicon file in Wikipron format."""
  basename = os.path.basename(filename)
  toks = os.path.splitext(basename)[0].split(_NAME_SEPARATOR)
  if len(toks) < 2:
    logging.error("%s: Expected at least two components in name", basename)
    return
  if len(toks) > 3:
    logging.error("%s: Too many components in name!", basename)
    return
  language = toks[0]
  if len(toks) == 3:
    language += "_" + toks[1]
    lexicon_type = toks[2]
  else:
    lexicon_type = toks[1]
  logging.info("Processing %s (%s)...", language, lexicon_type)
  stats = _compute_stats(filename)
  return language, lexicon_type, stats


def _sort(results):
  if FLAGS.sort_by == "language":
    return sorted(results, key=lambda tup: tup[0])  # Language name.
  elif FLAGS.sort_by == "score":
    return sorted(results, key=lambda tup: tup[2][2])  # L-score.
  else:
    return None


def main(unused_argv):
  if not FLAGS.input_dir:
    raise ValueError("Supply --input_dir")
  if not FLAGS.output_tsv:
    raise ValueError("Supply --output_tsv!")
  if FLAGS.sort_by != "language" and FLAGS.sort_by != "score":
    raise ValueError("Invalid sorting critertion: %s" % FLAGS.sort_by)

  results = []
  for filename in os.listdir(FLAGS.input_dir):
    if not filename.endswith(".tsv"):
      continue
    results.append(_process_lexicon_file(filename))
  if not results:
    raise ValueError("No files with .tsv extension found!")
  results = _sort(results)
  logging.info("Processed %d lexicons.", len(results))

  with open(FLAGS.output_tsv, mode="w", encoding="utf8") as f:
    for result in results:
      language_code = result[0]
      language_alpha_3 = language_code.split(_NAME_SEPARATOR)[0]
      language_info = pyc.languages.get(alpha_3=language_alpha_3)
      language_full = "-"
      if language_info:
        language_full = language_info.name
      else:
        bib_info = pyc.languages.get(bibliographic=language_alpha_3)
        if bib_info:
          language_full = bib_info.name
      lexicon_type = result[1]
      stats = result[2]
      f.write("%s\t%s\t%s\t%d\t%d\t%f\n" % (
          language_code, language_full, lexicon_type,
          stats[0], stats[1], stats[2]))


if __name__ == "__main__":
  app.run(main)
