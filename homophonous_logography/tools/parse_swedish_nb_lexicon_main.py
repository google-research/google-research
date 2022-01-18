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

r"""Utility for parsing the Swedish pronunciation lexicon.

The lexicon is hosted by the Norwegian National Library (Nasjonalbiblioteket).
The lexicon is in public domain. It was originally developed by NST. There are
927,167 entries. Please see the original documentation (in Norwegian) at:
  https://www.nb.no/sbfil/dok/nst_leksdat_se.pdf

This tool will parse the lexicon converting it to WikiPron format that can be
used to generate the training data with `bible_wikipron_data_main.py`.

Other sources:
--------------
Please also see lexicon at http://www.openslr.org/resources/29/lexicon-sv.tgz.
It may have possibly been derived from the lexicon above but only has 822,747
entries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import re

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string(
    "input_lexicon_file", "",
    "Path to the `;`-separated csv file containing the original lexicon.")

flags.DEFINE_string(
    "output_wikipron_lexicon", "",
    "Output lexicon in WikiPron tab-separated format: spelling, followed by a "
    "tab and a space-separated list of phonemes.")

FLAGS = flags.FLAGS

# Overall there are 51 fields in the original lexicon.
_NUM_FIELDS = 51

# Syllable marker.
_SYLLABLE_MARKER = "$"

# Phoneme set. Sorted by the longest phoneme names first.
_PHONEMES = [
    "%a*U", "\"a*U", "\"E*U", "a*U", "E*U", "%}:", "%y:", "%u:", "%u0", "%t`",
    "%s`", "%s'", "%o:", "%n`", "%l`", "%i:", "%e:", "%d`", "%E:", "%A:", "%2:",
    "%\"j", "\"}:", "\"y:", "\"u:", "\"u0", "\"t`", "\"s`", "\"s'", "\"o:",
    "\"n`", "\"l`", "\"i:", "\"e:", "\"d`", "\"E:", "\"A:", "\"2:", "}:", "y:",
    "u:", "u0", "t`", "s`", "s'", "o:", "n`", "l`", "i:", "e:", "d`", "E:",
    "A:", "2:", "%v", "%t", "%s", "%r", "%p", "%n", "%m", "%l", "%k", "%j",
    "%h", "%g", "%f", "%e", "%d", "%b", "%a", "%Y", "%U", "%S", "%O", "%I",
    "%E", "%9", "\"v", "\"t", "\"s", "\"r", "\"p", "\"n", "\"m", "\"l", "\"k",
    "\"j", "\"h", "\"g", "\"f", "\"e", "\"d", "\"b", "\"a", "\"Y", "\"U", "\"S",
    "\"O", "\"I" "\"E", "\"9", "v", "t", "s", "r", "p", "n", "m", "l", "k", "j",
    "h", "g", "f", "e", "d", "b", "a", "Y", "U", "S", "O", "N", "I", "E", "9",
    # Manually added.
    "%x", "x", "\"x",  # This is a very wierd one mostly found in foreign words.
    "\"E", "\"I",  # Not clear whether this is valid.
]

# Maximum length of a phoneme (in characters).
_MAX_PHONEME_LENGTH = 4

# Advanced normalization.
_CLEANUP_RE = r"[_¤\\]"


def _split_phonemes(syllable):
  """Split syllable by the existing phonemes preferring the longer ones."""
  phonemes = []
  i = 0
  while i < len(syllable):
    found = False
    for j in reversed(range(_MAX_PHONEME_LENGTH)):
      candidate = syllable[i:i+j+1]
      for phoneme in _PHONEMES:
        if candidate == phoneme:
          phonemes.append(phoneme)
          i += len(phoneme)
          found = True
          break
      if found:
        break
    if not found:
      raise ValueError("No phonemes found for syllable: %s "
                       "(last candidate: %s)" % (syllable, candidate))
  return phonemes


def _process_pronunciation(pron_string):
  """Processes a single entry in the original lexicon returning phonemes."""
  phonemes = []
  pron_string = pron_string.replace("%$", "$")  # `%` does not appear alone.
  syls = pron_string.split(_SYLLABLE_MARKER)
  for syl in syls:
    syl = syl.replace("\"\"", "\"")  # Typo.
    syl = re.sub(_CLEANUP_RE, "", syl)
    syl_phonemes = _split_phonemes(syl)
    phonemes.extend(syl_phonemes)
  return phonemes


def main(unused_argv):
  if not FLAGS.input_lexicon_file:
    raise ValueError("Specify --input_lexicon_file!")
  if not FLAGS.output_wikipron_lexicon:
    raise ValueError("Specify --output_wikipron_lexicon!")

  logging.info("Processing \"%s\" ...", FLAGS.input_lexicon_file)
  num_entries = 0
  with open(FLAGS.input_lexicon_file, mode="r", encoding="utf-8") as i_f:
    reader = csv.reader(i_f, delimiter=";", quoting=csv.QUOTE_NONE)
    with open(FLAGS.output_wikipron_lexicon, mode="w", encoding="utf-8") as o_f:
      for row in reader:
        if len(row) != _NUM_FIELDS:
          raise ValueError("Unexpected number of fields: %d" % len(row))
        spelling = row[0]
        pron_string = row[11]
        phonemes = _process_pronunciation(pron_string)
        line = "%s\t%s\n" % (spelling, " ".join(phonemes))
        # Replace \" marker with "!".
        line = line.replace("\"", "!")
        o_f.write(line)
        num_entries += 1
  logging.info("Wrote %d entries to \"%s\".", num_entries,
               FLAGS.output_wikipron_lexicon)


if __name__ == "__main__":
  app.run(main)
