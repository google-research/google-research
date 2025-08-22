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

"""Well-formed-ness grammar checker for TCD-DSL.

Parses text according to the grammar rules for the
"tensor calculus dependencies domain specific language" TCD-DSL.
"""

import bisect
import glob
import re

import lark


_DEFAULT_GRAMMAR = 'tcd_dsl_grammar.lark'


def get_parser(grammar_filename=_DEFAULT_GRAMMAR):
  """Returns a parser."""
  with open(grammar_filename, 'rt') as h_grammar:
    grammar_text = h_grammar.read()
  return lark.Lark(grammar_text)


def check_file(filename, show=True, parser=None):
  """Tries to parse a file's contents according to TCD-DSL."""
  with open(filename, 'rt') as h:
    tcd = h.read()
  # By default, we create a new parser at every call, to allow iterating fast
  # on grammar tweaks.
  if parser is None:
    parser = get_parser()
  parsed = parser.parse(tcd)
  if show:
    print(parser.parse(tcd).pretty())
  return parsed


def check_files(fileglob, parser=None, detail=1):
  """Checks if all files matching the glob parse, returning failures."""
  if parser is None:
    parser = get_parser()
  failed = []
  for tcd_file in sorted(glob.glob(fileglob)):
    print(f'### {tcd_file}', end='')
    try:
      parsed = check_file(tcd_file, show=False, parser=None)
      if detail >= 1:
        print(': OK')
      if detail >= 2:
        print(parsed.pretty())
    # Here, we indeed do want to surface any exception whatsoever.
    except Exception as exn:  # pylint:disable=broad-except
      if detail >= 1:
        print(f': FAILED {exn!r}')
      failed.append(tcd_file)
  return failed


def check_latex(
    tex_filepath,
    parser=None,
    detail=3,
    marker_re='%%%% TCD-DSL-EQUATION(?P<unchecked>:unchecked)?',
    report=print):
  """Checks a LaTeX file.

  Args:
    tex_filepath: path to the .tex file to be checked.
    parser: Optional parser object to use for grammar-checking.
      If `None`, then `get_parser()` will be used to create a parser.
    detail: detail level for the report, 1-3.
    marker_re: Regular expression for recognizing the formula-to-be-checked
      marker. Must have a named group `unchecked` for identifying
      recognize-but-do-not-check formulas.
    report: Callable to process to-be-reported message strings.
      Defaults to `print`.

  Returns:
    Sequence of parse-errors, each being a tuple
    `(line_number, code, parser_exception)`.
  """
  if parser is None:
    parser = get_parser()
  parse_errors = []
  re_marker = re.compile(
      r'(?ms)^%s\s*\\begin\{verbatim\}(?P<body>.*?)\\end\{verbatim\}' % (
          marker_re,))
  with open(tex_filepath, 'rt') as h_in:
    tex_text = h_in.read()
  tex_lines = tex_text.split('\n')
  line_char_offsets = [0]
  for line in tex_lines:
    line_char_offsets.append(line_char_offsets[-1] + len(line) + 1)
  for the_match in re_marker.finditer(tex_text):
    start_pos, end_pos = the_match.span()
    del end_pos  # Unused.
    tcd = the_match['body']
    lineno = 1 + bisect.bisect_left(line_char_offsets, start_pos)
    parse_error = None
    parse_message = 'OK'
    try:
      if the_match['unchecked']:
        parse_message = 'UNCHECKED'
      else:
        _ = parser.parse(tcd)
    except lark.LarkError as exn:
      parse_error = exn
      parse_errors.append((lineno, tcd, exn))
      parse_message = f'ERROR: {parse_error!r}'
    report(f'### Line {lineno:5d}  {parse_message}')
    if detail >= 2:
      report('\n' + the_match['body'])
    if parse_error and detail >= 3:
      print(str(parse_error))
      print('###\n')
  return parse_errors


if __name__ == '__main__':
  check_files('*.tcd')
