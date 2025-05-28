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

r"""Applies specified finite-state normalization grammars to a text corpus.

Notes:
------
  - At the moment this tool only processes the input strings token-by-token.
    This is because for Perso-Arabic we don't accept non-Perso-Arabic
    characters.
  - Tokens containing square brackets characters ('[', ']') will fail because
    these have special meaning in Pynini. Transduction failures imply that the
    tokens are kept as is, unchanged.

Dependencies:
-------------
  absl
  pynini
"""

from typing import Sequence

import csv
import collections
import logging
import pickle

from absl import app
from absl import flags

import pynini
import utils

flags.DEFINE_string(
    "corpus_file", "",
    "Path to the input corpus file in text format or bzip2/gzip format.")

flags.DEFINE_string(
    "far_file", "",
    "Path to the input FST archive (FAR) file containing the grammars.")

flags.DEFINE_string(
    "grammar_name", "",
    "Name of the FST normalization grammar to load from the supplied FAR.")

flags.DEFINE_string(
    "output_file", "",
    "Path to the normalized corpus file in text format or bzip2/gzip format.")

flags.DEFINE_string(
    "output_token_diffs_file", "",
    "Tab-separated file containing actual token diffs (source and target "
    "tokens) and their frequencies.")

flags.DEFINE_string(
    "output_line_diffs_file", "",
    "Binary pickle file containing line IDs that have diffs.")

FLAGS = flags.FLAGS


def _process_corpus(input_f, normalizer_fst):
  """Processes input corpus line-by-line, token-by-token."""
  for input_line in input_f.readlines():
    input_line = input_line.rstrip("\n")
    output_line = []
    token_diffs = []
    line_diff = False
    for tok in filter(None, input_line.split()):
      try:
        output_tok_fst = pynini.accep(tok) @ normalizer_fst
        output_tok = None
        if output_tok_fst.start() != pynini.NO_STATE_ID:
          output_tok = output_tok_fst.string()
        if output_tok:
          if output_tok != tok:
            token_diffs.append((tok, output_tok))
            line_diff = True
          output_line.append(output_tok)
        else:
          output_line.append(tok)
      except (pynini.FstStringCompilationError, pynini.FstOpError):
        output_line.append(tok)
    yield " ".join(output_line), token_diffs, line_diff


def _open_and_process_corpus():
  """Opens the corpus and prepares the normalizer FST."""
  # Process corpus. When token diffs are recorded, for each modified source
  # token a corresponding rewritten token and its frequency is stored.
  token_diffs = collections.defaultdict(collections.Counter)
  line_id_diffs = []
  line_id = 0
  with pynini.default_token_type("byte"):
    with pynini.Far(FLAGS.far_file, "r") as far:
      normalizer_fst = far[FLAGS.grammar_name]
      with utils.open_file(FLAGS.corpus_file) as input_f:
        with utils.open_file(FLAGS.output_file, mode="w") as output_f:
          for line, line_token_diffs, line_diff in _process_corpus(
              input_f, normalizer_fst):
            output_f.write(line + "\n")
            if FLAGS.output_line_diffs_file and line_diff:
              line_id_diffs.append(line_id)
            line_id += 1

            # Update token diffs dictionary.
            for source_tok, new_tok in line_token_diffs:
                token_diffs[source_tok].update([new_tok])

  # Save token diffs.
  if FLAGS.output_token_diffs_file:
    with utils.open_file(FLAGS.output_token_diffs_file, mode="w") as f:
      token_diff_writer = csv.writer(f, delimiter="\t")
      for source_tok, new_tok_counter in sorted(token_diffs.items()):
        new_tok, count = new_tok_counter.most_common(1)[0]
        token_diff_writer.writerow([source_tok, new_tok, count])
    logging.info(f"Wrote {len(token_diffs)} unique token diffs to "
                 f"{FLAGS.output_token_diffs_file}")

  # Save line diffs.
  if FLAGS.output_line_diffs_file:
    logging.info(f"Saving {len(line_id_diffs)} line diffs to "
                 f"{FLAGS.output_line_diffs_file} ...")
    with open(FLAGS.output_line_diffs_file, "wb") as f:
      pickle.dump(line_id_diffs, f)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not FLAGS.corpus_file:
    raise app.UsageError("Specify --corpus_file [FILE]!")
  if not FLAGS.far_file:
    raise app.UsageError("Specify --far_file [FILE]!")
  if not FLAGS.grammar_name:
    raise app.UsageError("Specify --grammar_name [NAME]!")
  if not FLAGS.output_file:
    raise app.UsageError("Specify --output_file [FILE]!")

  _open_and_process_corpus()


if __name__ == "__main__":
  app.run(main)
