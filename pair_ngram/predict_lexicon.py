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

"""Predicts best outputs using pair LM and lexicon filter."""

import argparse
import functools
import multiprocessing

from typing import Iterator

import pynini
from pynini.lib import pynutil
from pynini.lib import rewrite


def _parse_token_type(token_type: str) -> pynini.TokenType:
  """Parses token type string.

  Args:
    token_type: token type, or the path to a symbol table.

  Returns:
    A token type string, or a symbol table.
  """
  if token_type in ["byte", "utf8"]:
    return token_type
  return pynini.SymbolTable.read_text(token_type)


def _lexicon(path: str, token_type: pynini.TokenType) -> pynini.Fst:
  """Compiles lexicon FST.

  Args:
    path: path to input file.
    token_type: token type, or the path to a symbol table.

  Returns:
    A lexicon FST.
  """
  words = pynini.string_file(
      path, input_token_type=token_type, output_token_type=token_type)
  return pynutil.join(words, " ").optimize()


class _LexiconRewriter:
  """Lexicon-based rewriting helper."""

  def __init__(
      self,
      rule: pynini.Fst,
      lexicon: pynini.Fst,
      input_token_type: pynini.TokenType,
      output_token_type: pynini.TokenType,
  ):
    self._rewrite_s1 = functools.partial(
        rewrite.rewrite_lattice, rule=rule, token_type=input_token_type)
    self._lexicon = lexicon
    self._rewrite_s2 = functools.partial(
        rewrite.lattice_to_top_string, token_type=output_token_type)

  def __call__(self, string: str) -> str:
    try:
      lattice = self._rewrite_s1(string)
    except rewrite.Error:
      return "<composition failure>"
    filtered = pynini.intersect(lattice, self._lexicon)
    # If intersection fails, take the top string from the original lattice.
    # But if it succeeds, take the top string from the filtered lattice.
    if filtered.start() == pynini.NO_STATE_ID:
      return self._rewrite_s2(lattice)
    else:
      return self._rewrite_s2(filtered)


def _reader(path: str) -> Iterator[str]:
  """Reads strings from a single-column filepath.

  Args:
    path: path to input file.

  Yields:
     Stripped lines.
  """
  with open(path, "r") as source:
    for line in source:
      yield line.rstrip()


def main(args: argparse.Namespace) -> None:
  rule = pynini.Fst.read(args.rule)
  input_token_type = _parse_token_type(args.input_token_type)
  output_token_type = _parse_token_type(args.output_token_type)
  lexicon = _lexicon(args.lexicon, output_token_type)
  rewriter = _LexiconRewriter(rule, lexicon, input_token_type,
                              output_token_type)
  reader = _reader(args.input)
  with multiprocessing.Pool(args.processes) as pool:
    with open(args.output, "w") as sink:
      for output in pool.map(rewriter, reader):
        print(output, file=sink)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="path to the input data")
  parser.add_argument(
      "--output", required=True, help="path for the output data")
  parser.add_argument("--rule", help="path to the input FST rule")
  parser.add_argument("--lexicon", help="path to the input lexicon file")
  parser.add_argument(
      "--input_token_type",
      default="utf8",
      help="input side token type, or the path to the input symbol table "
      "(default: %(default)s)",
  )
  parser.add_argument(
      "--output_token_type",
      default="utf8",
      help="output side token type, or the path to the output symbol table "
      "(default: %(default)s)",
  )
  parser.add_argument(
      "--processes",
      type=int,
      default=multiprocessing.cpu_count(),
      help="maximum number of concurrent processes (default: %(default)s)",
  )
  main(parser.parse_args())
