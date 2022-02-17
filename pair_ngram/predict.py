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

"""Predicts best outputs using pair LM."""

import argparse
import functools
import multiprocessing

from typing import Iterator

import pynini
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


class _Rewriter:
  """Rewriting helper."""

  def __init__(
      self,
      rule: pynini.Fst,
      input_token_type: pynini.TokenType,
      output_token_type: pynini.TokenType,
  ):
    self._rewrite = functools.partial(
        rewrite.top_rewrite,
        rule=rule,
        input_token_type=input_token_type,
        output_token_type=output_token_type,
    )

  def __call__(self, string: str) -> str:
    try:
      return self._rewrite(string)
    except rewrite.Error:
      return "<composition failure>"


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
  rewriter = _Rewriter(
      pynini.Fst.read(args.rule),
      _parse_token_type(args.input_token_type),
      _parse_token_type(args.output_token_type),
  )
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
