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

r"""A tool for generating training data from the Bibles using Epitran.

This tool processes a multilingual parallel corpus created from translations of
the Bible available from https://github.com/christos-c/bible-corpus:

  Christos Christodoulopoulos and Mark Steedman (2015): "A massively parallel
  corpus: the Bible in 100 languages", Proc. of Language Resources and
  Evaluation (LREC), pp. 375–395, European Language Resources Association
  (ELRA).

The language of the Bible has to be supported by the Epitran G2P, which is a
library and tool for transliterating orthographic text as IPA (International
Phonetic Alphabet) available from: https://github.com/dmort27/epitran. For more
information please see

  David R. Mortensen, Siddharth Dalmia, and Patrick Littell (2018): "Epitran:
  Precision G2P for many languages.", In Proc. of the 11th International
  Conference on Language Resources and Evaluation (LREC 2018), Paris, France.
  European Language Resources Association (ELRA).

The tool generates a tab-separated file where each line consists of two columns:

  - Sentence ID, e.g. "b.GEN.1.1".
  - Word orthography/pronunciation pairs, separated by whitespace, e.g.
      "In/ɪ_n het/h_ɛ_t begin/b_eː_ɣ_ɪ_n heeft/h_eː_f_t ...".

Please see `bible_reader.py` library for the main flags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import bible_reader as bible
import epitran

flags.DEFINE_string(
    "language_id", "",
    ("The ISO 639-3 code of the language plus a hyphen plus a four letter code "
     "for the script (e.g. 'Latn' for Latin script). Example: 'uig-Arab'."))

FLAGS = flags.FLAGS


class BibleEpitranReader(bible.BibleReader):
  """Bible reader using Epitran pronunciations."""

  def __init__(self, epi):
    super().__init__()
    self._epi = epi

  def _process_sentence(self, text):
    """Processes the sentence returning the word/pronunciation pairs."""
    word_tokens = text.split()
    word_prons = []
    bad_prons = False
    for word_token in word_tokens:
      if not word_token:
        continue
      phonemes_list = self._epi.trans_list(word_token.lower())
      if not phonemes_list:
        # Note: We never hit this point. The "trans_list" API above always
        # succeeds. If we want to track down invalid graphemes on the input
        # side, we'll have to switch to "word_to_tuples" API.
        bad_prons = True
      phonemes = "_".join(phonemes_list)
      word_prons.append("%s/%s" % (word_token, phonemes))
    return word_prons, bad_prons


def main(unused_argv):
  if not FLAGS.language_id:
    raise ValueError("Specify --language_id!")

  logging.info("Initializing Epitran for \"%s\" ...", FLAGS.language_id)
  epi = epitran.Epitran(FLAGS.language_id)

  logging.info("Processing Bible ...")
  reader = BibleEpitranReader(epi)
  reader.read()


if __name__ == "__main__":
  app.run(main)
