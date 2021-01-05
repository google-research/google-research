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

"""Utilities functions for grammar."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from neural_guided_symbolic_regression.utils import arithmetic_grammar


def load_grammar(grammar_path):
  """Loads context-free grammar from file.

  The grammar used for symbolic regularization has specific setup. The padding
  production rule is the 0-th index production rule. And an unique starting
  production rule O -> S is added as 1-st index production rule. The production
  rules in grammar_path are added after this two rules.

  Args:
    grammar_path: String, the path to the grammar file.

  Returns:
    arithmetic_grammar.Grammar or TokenGrammar object.
  """
  grammar = arithmetic_grammar.Grammar(
      arithmetic_grammar.read_grammar_from_file(
          filename=grammar_path, return_list=True),
      padding_at_end=False,
      add_unique_production_rule_to_start=True)
  return grammar


def _get_num_expressions(
    s_to_s_t, s_to_t, t_to_s, t_to_terminal, max_num_production_rules):
  """Helper function for get_num_expressions().

  It returns the entire n_s and n_t arrays.

  Args:
    s_to_s_t: Integer, number of production rules S -> S T
    s_to_t: Integer, number of production rules S -> T
    t_to_s: Integer, number of production rules T -> S
    t_to_terminal: Integer, number of production rules T -> terminal symbols
    max_num_production_rules: Integer, the maximum number of production rules
        in sequence.

  Returns:
    n_s: Float numpy array with shape [max_num_production_rules + 1,],
        the number of expressions starting with non-terminal symbol S.
    n_t: Float numpy array with shape [max_num_production_rules + 1,]
        the number of expressions starting with non-terminal symbol T.
  """
  n_s = np.zeros(max_num_production_rules + 1)
  n_t = np.zeros(max_num_production_rules + 1)
  # Set ground case values for one production rule.
  n_s[0] = 1
  n_t[0] = 1
  for num_production_rules in range(1, max_num_production_rules + 1):
    num_s_to_s_t = 0
    for p in range(num_production_rules):
      num_s_to_s_t += n_s[p] * n_t[num_production_rules - 1 - p]
    n_s[num_production_rules] = (
        s_to_s_t * num_s_to_s_t + s_to_t * n_t[num_production_rules - 1])
    n_t[num_production_rules] = (
        t_to_s * n_s[num_production_rules - 1] + t_to_terminal)
  return n_s, n_t


def get_num_expressions(
    s_to_s_t, s_to_t, t_to_s, t_to_terminal, max_num_production_rules):
  r"""Gets the number of expressions within maximum number of production rules.

  The context-free grammar has two non-terminal symbols, S and T.
  When the left hand side symbol is S, the right hand side symbols can be
  non-terminal symbols (S, T) or T.
  When the left hand side symbol is T, the right hand side symbols can be
  non-terminal symbol S or terminal symbols.

  We denote
  s_to_s_t: number of production rules S -> S T
  s_to_t: number of production rules S -> T
  t_to_s: number of production rules T -> S
  t_to_terminal: number of production rules T -> terminal symbols

  For example, for grammar
  S -> S '+' T
  S -> S '-' T
  S -> S '*' T
  S -> S '/' T
  S -> T
  T -> '(' S ')'
  T -> 'x'
  T -> '1'

  s_to_s_t = 4
  s_to_t = 1
  t_to_s = 1
  t_to_terminal = 2

  We define n_s[i] and n_t[i] as the number of expressions for a production rule
  sequence with maximum length i starting with S and T respectively.

  Here is the recursive relation:
  n_s[i] = s_to_s_t * \sum_{p=0,...,i - 1} n_s[p] * n_t[i - 1 - p]
           + s_to_t * n_t[i - 1]
  n_t[i] = t_to_s * n_s[i - 1] + t_to_terminal

  Ground case:
  n_s[0] = 1
  n_t[0] = 1

  This function will return n_s[max_num_production_rules].

  Note this function also counts the non-terminal expressions.

  Args:
    s_to_s_t: Integer, number of production rules S -> S T
    s_to_t: Integer, number of production rules S -> T
    t_to_s: Integer, number of production rules T -> S
    t_to_terminal: Integer, number of production rules T -> terminal symbols
    max_num_production_rules: Integer, the maximum number of production rules
        in sequence.

  Returns:
    Float, the number of expressions within maximum number of production rules
    starting with non-terminal symbol S.
  """
  n_s, _ = _get_num_expressions(
      s_to_s_t, s_to_t, t_to_s, t_to_terminal, max_num_production_rules)
  return n_s[max_num_production_rules]
