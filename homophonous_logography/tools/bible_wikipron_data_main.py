# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""A tool for generating training data from the Bibles using WikiPron.

This tool processes a multilingual parallel corpus created from translations of
the Bible available from https://github.com/christos-c/bible-corpus:

  Christos Christodoulopoulos and Mark Steedman (2015): "A massively parallel
  corpus: the Bible in 100 languages", Proc. of Language Resources and
  Evaluation (LREC), pp. 375–395, European Language Resources Association
  (ELRA).

The pronunciations are taken from the WikiPron project which is a command-line
tool and Python API for mining multilingual pronunciation data from Wiktionary,
as well as a database of pronunciation dictionaries mined using this tool. The
project is available from https://github.com/kylebgorman/wikipron. Please see

  Jackson L. Lee, Lucas F.E. Ashby, M. Elizabeth Garza, Yeonju Lee-Sikka,
  Sean Miller, Alan Wong, Arya D. McCarthy, and Kyle Gorman (2020): "Massively
  multilingual pronunciation mining with WikiPron.", In Proc. of LREC,
  pp. 4223--4228, Marseille, France, European Language Resources Association
  (ELRA).

Examples:
---------
Download WikiPron from GitHub:

  > git clone https://github.com/kylebgorman/wikipron ${WIKIPRON_DIR}

Assuming the Bibles are in ${BIBLE_DIR},

  > python3 -m homophonous_logography.tools.bible_wikipron_data_main \
     --input_xml_file ${BIBLE_DIR}/bibles/Greek.xml \
     --test_set_ids_file bible_test_ids.txt \
     --input_wikipron_file ${WIKIPRON_DIR}/data/tsv/gre_phonemic.tsv \
     --output_data_file modern_greek_data.tsv

Please see `bible_reader.py` library for the main flags.

Dependencies:
-------------
  absl-py
  progress
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv

from absl import app
from absl import flags
from absl import logging

import homophonous_logography.tools.bible_reader as bible

flags.DEFINE_string(
    "input_wikipron_file", "",
    "Path to the WikiPron dump in tab-separated csv format. These files are "
    "located in ${WIKIPRON_DIR}/data/tsv subdirectory.")

FLAGS = flags.FLAGS


class BibleWikiPronReader(bible.BibleReader):
  """Bible reader using WikiPron pronunciations."""

  def __init__(self):
    super().__init__()
    self._lexicon = self._load_lexicon()
    self._total_num_tokens = 0
    self._total_num_oov_tokens = 0
    self._oov_tokens = set()

  def _load_lexicon(self):
    """Loads WikiPron lexicon."""
    logging.info("Loading WikiPron lexicon from \"%s\" ...",
                 FLAGS.input_wikipron_file)
    num_spellings = 0
    num_prons = 0
    lexicon = collections.defaultdict(list)
    with open(FLAGS.input_wikipron_file, mode="r", encoding="utf-8") as f:
      reader = csv.reader(f, delimiter="\t")
      for row in reader:
        if len(row) < 2 or len(row) > 3:
          raise ValueError("Unexpected number of entries: {}: \"{}\"".format(
              len(row), " ".join(row)))
        pron = row[1]
        spelling = row[0].lower()  # Lower-casing is redundant, just in case.
        lexicon[spelling].append(pron)
        num_spellings += 1
        num_prons += 1
    logging.info("Loaded %d spellings and %d pronunciations.", num_spellings,
                 num_prons)
    return lexicon

  def _process_sentence(self, text):
    """Processes the sentence returning the word/pronunciation pairs."""
    word_tokens = text.split()
    word_prons = []
    bad_prons = False
    for word_token in word_tokens:
      if not word_token:
        continue
      self._total_num_tokens += 1
      word_token = word_token.lower()
      if word_token not in self._lexicon:
        bad_prons = True
        phonemes_list = []
        self._total_num_oov_tokens += 1
        self._oov_tokens.add(word_token)
      else:
        # Please note no homograph resolution here.
        phonemes_list = self._lexicon[word_token][0].split()
      phonemes = "_".join(phonemes_list)
      word_prons.append("%s/%s" % (word_token, phonemes))
    return word_prons, bad_prons

  @property
  def total_num_tokens(self):
    return self._total_num_tokens

  @property
  def total_num_oov_tokens(self):
    return self._total_num_oov_tokens

  @property
  def lexicon(self):
    return self._lexicon

  @property
  def oov_tokens(self):
    return self._oov_tokens


def main(unused_argv):
  if not FLAGS.input_wikipron_file:
    raise ValueError("Specify --input_wikipron_file!")

  logging.info("Processing Bible ...")
  reader = BibleWikiPronReader()
  reader.read()
  logging.info("Processed total %d tokens (%d unique tokens in lexicon). "
               "Found %d OOVs (%d unique OOVs): %f%%.",
               reader.total_num_tokens, len(reader.lexicon),
               reader.total_num_oov_tokens, len(reader.oov_tokens),
               reader.total_num_oov_tokens / reader.total_num_tokens * 100.0)


if __name__ == "__main__":
  app.run(main)
