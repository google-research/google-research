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

"""Util for normalizing math answers."""

import re


def normalize_math(math_text):
  """Taken from the Minerva appendix: https://arxiv.org/abs/2206.14858.

  Args:
    math_text: the text to normalize.

  Returns:
    text: the normalized text
  """

  substitutions = [
      ('an ', ''),
      ('a ', ''),
      ('.$', '$'),
      ('\\$', ''),
      (r'\ ', ''),
      (' ', ''),
      ('mbox', 'text'),
      (',\\text{and}', ','),
      ('\\text{and}', ','),
      ('\\text{m}', '\\text{}'),
  ]

  removed_expressions = [
      'square',
      'ways',
      'integers',
      'dollars',
      'mph',
      'inches',
      'ft',
      'hours',
      'km',
      'units',
      '\\ldots',
      'sue',
      'points',
      'feet',
      'minutes',
      'digits',
      'cents',
      'degrees',
      'cm',
      'gm',
      'pounds',
      'meters',
      'meals',
      'edges',
      'students',
      'childrentickets',
      'multiples',
      '\\text{s}',
      '\\text{.}',
      '\\text{\ns}',
      '\\text{}^2',
      '\\text{}^3',
      '\\text{\n}',
      '\\text{}',
      r'\mathrm{th}',
      r'^\circ',
      r'^{\circ}',
      r'\;',
      r',\!',
      '{,}',
      '"',
      '\\dots',
      '%',
  ]

  math_text = math_text.split('=')[-1]

  for before, after in substitutions:
    math_text = math_text.replace(before, after)
  for expr in removed_expressions:
    math_text = math_text.replace(expr, '')

  # Extract answer that is in LaTeX math, is bold,
  # is surrounded by a box, etc.
  math_text = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', math_text)
  math_text = re.sub(r'(\\text\{)(.*?)(\})', '\\2', math_text)
  math_text = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', math_text)
  math_text = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', math_text)
  math_text = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', math_text)

  # Normalize shorthand TeX:
  # \fracab -> \frac{a}{b}
  # \frac{abc}{bef} -> \frac{abc}{bef}
  # \fracabc -> \frac{a}{b}c
  # \sqrta -> \sqrt{a}
  # \sqrtab -> sqrt{a}b
  math_text = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', math_text)
  math_text = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', math_text)
  math_text = math_text.replace('$', '')

  # Normalize 100,000 -> 100000
  if math_text.replace(',', '').isdigit():
    math_text = math_text.replace(',', '')

  return math_text


def last_boxed_only_string(string):
  """Copied from github.com/hendrycks/math."""
  idx = string.rfind('\\boxed')
  if idx < 0:
    idx = string.rfind('\\fbox')
    if idx < 0:
      return None

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == '{':
      num_left_braces_open += 1
    if string[i] == '}':
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if right_brace_idx is None:
    retval = None
  else:
    retval = string[idx : right_brace_idx + 1]

  return retval
