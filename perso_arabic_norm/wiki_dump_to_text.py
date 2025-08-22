# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

r"""Converts XML Wiki dump to newline-separated text file.

Notes:
------
  - We currently don't control for sentence length.
  - It's not entirely clear how to sentence split, if at all required for the
    downstream applications. If required, for Arabic for example, we'll need
    to split additionally by the Arabic-specific punctuation symbols.

Example:
--------
  python wiki_dump_to_text.py \
    --xml_dump_file ${DATA_DIR}/wiki/kswiki-latest-pages-meta-current.xml.bz2 \
    --prob_threshold 0.6 \
    --output_file ks.txt.bz2

Dependencies:
-------------
  absl
  mwparserfromhell
  mwtypes
  mwxml
  unicodedataplus
"""

from typing import Optional, Sequence, Tuple

import collections
import logging
import operator
import re

from absl import app
from absl import flags

import mwparserfromhell
import mwtypes
import mwxml
import unicodedataplus

import utils

flags.DEFINE_string(
    "xml_dump_file", "", "Path to the XML dump.")

flags.DEFINE_string(
    "script_name", "Arabic",
    "Case-sensitive Unicode script name (rather than ISO 15924 code) to select "
    "(e.g., `Bengali`)")

flags.DEFINE_float(
    "prob_threshold", 0.6,
    "Drop all the lines that are determined to be in script `script_name` "
    "and have probability lower than `prob_threshold`.")

flags.DEFINE_string(
    "output_file", "",
    "Output text (or compressed text) file containing the text in the "
    "requested script. Compression will depend on file extension (`bz2` or "
    "`gz`).")

FLAGS = flags.FLAGS

# Wikipedia <ref>...</ref> tags for footnotes.
_REGEX_REF_TAGS = re.compile("<ref>.*?</ref>")

# Whitespace regex.
_REGEX_WHITESPACE = re.compile("[, \t\-!?:]+")


def _remove_footnote_refs(text):
  """Removes footnotes marked with Wikipedia <ref>/</ref> tags."""
  # This is too basic, perhaps something more sophisticated is needed here.
  return _REGEX_REF_TAGS.sub("", text)


def _detect_best_script_name(
    word,
    strict = False,
):
    """Returns the most likely script name (rather than ISO 15924 code) the
    word belongs to along with the corresponding confidence expressed as a
    maximum likelihood estimate computed over the `word` sample. If `strict`
    is enabled, then all the characters must belong to the same script and
    `None` is returned on failure.
    Example: "ژۇرنال" -> ("Arabic", 1.0).
    """
    script_counts: DefaultDict[
        str,
        float,
    ] = collections.defaultdict(float)
    for char in word:
        script_counts[unicodedataplus.script(char)] += 1.0
    script_probs = [
        (
            s,
            script_counts[s] / len(word),
        )
        for s in script_counts
    ]
    script_probs.sort(
        key=operator.itemgetter(1),
        reverse=True,
    )
    if strict and len(script_probs) != 1:
        return None
    else:
        # The script names in Unicode data tables have underscores instead of
        # whitespace to enable parsing. See:
        # https://www.unicode.org/Public/13.0.0/ucd/Scripts.txt
        return script_probs[0]


def _parse_article(text):
  """Parses a single article."""
  wikicode = mwparserfromhell.parse(_remove_footnote_refs(text))
  for line in wikicode.strip_code().split("\n"):
    line = line.strip()
    if not line:
      continue

    # Before detecting the script remove all the whitespace.
    without_whitespace="".join(_REGEX_WHITESPACE.split(line))
    if not without_whitespace:
      continue
    best_script, best_prob = _detect_best_script_name(without_whitespace)

    # Decide whether to proceed with this line.
    if best_script != FLAGS.script_name or best_prob < FLAGS.prob_threshold:
      continue
    yield line


def _process_dump(dump, output_file):
  """Processes XML dump file."""
  for page in dump:
    logging.info(f"Processing page {page.id} ...")
    num_revisions = 0
    for revision in page:
      if num_revisions == 1:
        raise ValueError("Only supports single revision per page")
      if not revision.text:
        continue
      for line in _parse_article(revision.text):
        output_file.write(line + "\n")
      num_revisions += 1


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.xml_dump_file:
    raise app.UsageError("Specify --xml_dump_file [FILE]!")
  if not FLAGS.output_file:
    raise app.UsageError("Specify --output_file [FILE]!")

  path = mwtypes.files.reader(FLAGS.xml_dump_file)
  dump = mwxml.Dump.from_file(path)
  logging.info(f"Saving text dump to {FLAGS.output_file} ...")
  with utils.open_file(FLAGS.output_file, "w") as output_f:
    _process_dump(dump, output_f)


if __name__ == "__main__":
  app.run(main)
