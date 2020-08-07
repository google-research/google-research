# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Define arithmetic context-free grammars for expressions.

Define basic arithmetic context-free grammars for expressions using
Natural Language Toolkit (NLTK).
http://www.nltk.org/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
from nltk.parse import generate
import numpy as np
import six
from six.moves import map
from six.moves import range
from tensorflow.compat.v1 import gfile

from neural_guided_symbolic_regression.utils import constants


def all_terminal(symbols):
  """Whether the input list of symbols are all terminal symbols.

  Note this function will return True for symbols=[].

  Args:
    symbols: List of context-free grammar symbols.

  Returns:
    Boolean whether the input list of symbols are all terminal symbols.
  """
  return all(nltk.grammar.is_terminal(symbol) for symbol in symbols)


def read_grammar_from_file(filename, return_list=False):
  """Reads grammar from file.

  Any lines start with '#' will be considered as comments and be ignored.

  Args:
    filename: String, the filename of txt file containing the grammar production
        rules.
    return_list: Boolean, whether to return a nltk.grammar.CFG object or a list
        of grammar production rule strings.

  Returns:
    A nltk.grammar.CFG object or a list of grammar production rule strings.
  """
  production_rule_strings = []
  with gfile.Open(filename, 'r') as f:
    for line in f.read().strip().split('\n'):
      if not line.startswith('#'):
        production_rule_strings.append(line)
  if return_list:
    return production_rule_strings
  else:
    return nltk.CFG.fromstring(production_rule_strings)


class Grammar(object):
  """Grammar object initialized by strings of context-free grammar.

  Attributes:
    delimiter: String, delimiter of the tokens in the expression string.
    prod_rules: List of nltk.grammar.Production objects.
    num_production_rules: Integer, number of unique production rules defined in
        context-free grammar.
    padding_rule_index: Integer, the index of the padding rule.
    padding_at_end: Boolean, if True, the padding rule is at the end of
        production rules. Otherwise, the padding rule is the first production
        rule in grammar.
    start_index: The left hand side (lhs) object of the first production rule.
    start_rule: nltk.grammar.Production object, the starting rule.
    prod_rule_to_index: A dict mapping from production rule to its index in
        prod_rules.
    prod_rule_lhs: List of string, left hand side (lhs) symbols of each
        production rule in prod_rules.
    unique_lhs: List of unique left hand side (lhs) objects sorted by symbol.
    num_unique_lhs: Integer, number of unique left hand side (lhs).
    lhs_to_index: A dict mapping from left hand side (lhs) to its index in
        unique_lhs.
    masks: Numpy array with shape [num_unique_lhs, num_production_rules]. Used
        as a mask for production rules with given left hand side (lhs).
    prod_rule_index_to_lhs_index: Numpy array with shape
        [num_production_rules,]. For the i-th production rule in prod_rules,
        prod_rule_index_to_lhs_index[i] gives the index of its left hand side in
        unique_lhs.
    prod_rule_rhs_indices: List of list of integers, the indices of right hand
        side (rhs) symbols in unique_lhs for each production rule.
    max_rhs_indices_size: Integer, the maximum size of the indice list in
        prod_rule_rhs_indices.
    parser: A nltk.ChartParser object for parsing the expression string.
  """

  def __init__(
      self,
      grammar_rules,
      delimiter=' ',
      padding_at_end=True,
      add_unique_production_rule_to_start=False):
    """Constructs the context-free grammar.

    Args:
      grammar_rules: A list of strings, each string is a grammar production
          rule.
          Each string should have one left hand side (lhs) symbol, an arrow
          '->', and any number of right hand side (rhs) symbols, including zero.
          A space is required between lhs and arrow. All the symbols should be
          separated by single space.
      delimiter: String, delimiter of the tokens in the expression string.
      padding_at_end: Boolean. If True, the padding production rule will be the
          last production rule in the grammar. If False, it will be the first.
      add_unique_production_rule_to_start: Boolean. If the start symbol of
          grammar is S, there are usually more than one production rules with
          lhs S. If True, a production rule O -> S will be added to the grammar
          and O will be set as the start symbol. There will be only one unique
          production rule with lhs start symbol (O). Default False.
    """
    self.delimiter = delimiter
    self.parser = nltk.ChartParser(
        self._set_context_free_grammar(
            grammar_rules,
            padding_at_end,
            add_unique_production_rule_to_start))

  def _set_context_free_grammar(
      self,
      grammar_rules,
      padding_at_end,
      add_unique_production_rule_to_start):
    """Sets context-free grammar and useful attributes.

    This method sets a self._cfg attribute containing a nltk.grammar.CFG object
    for context-free grammar.

    Args:
      grammar_rules: A list of strings, each string is a grammar production
          rule.
          Each string should have one left hand side (lhs) symbol, an arrow
          '->', and any number of right hand side (rhs) symbols, including zero.
          A space is required between lhs and arrow. All the symbols should be
          separated by single space.
      padding_at_end: Boolean. If True, the padding production rule will be the
          last production rule in the grammar. If False, it will be the first.
      add_unique_production_rule_to_start: Boolean. If the start symbol of
          grammar is S, there are usually more than one production rules with
          lhs S. If True, a production rule O -> S will be added to the grammar
          and O will be set as the start symbol. There will be only one unique
          production rule with lhs start symbol (O).

    Returns:
      A nltk.grammar.CFG object.

    Raises:
      ValueError: If the input grammar_rules is not list.
      ValueError: If the last production rule is not the dummy production rule
          defined in this module.
      ValueError: If add_unique_production_rule_to_start is True but symbol O
          has already been used in the input grammar rules.
    """
    if not isinstance(grammar_rules, list):
      raise ValueError('The input grammar_rules should be list.')

    if add_unique_production_rule_to_start:
      original_start_symbol_string = nltk.grammar.standard_nonterm_parser(
          grammar_rules[0], pos=0)[0].symbol()
      for grammar_rule in grammar_rules:
        if 'O' in grammar_rule.split():
          raise ValueError(
              'add_unique_production_rule_to_start=True '
              'so O -> %s will be added to grammar rules. '
              'But symbol O has already been used in %s'
              % (original_start_symbol_string, grammar_rule))
      grammar_rules = ['O -> %s' % original_start_symbol_string] + grammar_rules

    self.padding_at_end = padding_at_end
    if self.padding_at_end:
      # Add DUMMY_PRODUCTION_RULE for padding in production rule sequence. Avoid
      # append to prevent changing the input grammar_rules list.
      grammar_rules = grammar_rules + [constants.DUMMY_PRODUCTION_RULE]
    else:
      grammar_rules = [constants.DUMMY_PRODUCTION_RULE] + grammar_rules
    if len(set(grammar_rules)) != len(grammar_rules):
      raise ValueError('The grammar production rules are not unique.')

    if self.padding_at_end:
      self._cfg = nltk.CFG.fromstring(grammar_rules)
    else:
      # NOTE(leeley): By default, nltk.CFG.fromstring will set the lhs of the
      # first production rule as start symbol. This causes problem if this
      # grammar is used in parser when padding_at_end is False.
      # We need to set the start symbol as the lhs of the second production rule
      # in this situation.
      _, productions = nltk.grammar.read_grammar(
          grammar_rules, nltk.grammar.standard_nonterm_parser, encoding=None)
      start = productions[1].lhs()
      self._cfg = nltk.grammar.CFG(start, productions)

    self.prod_rules = self._cfg.productions()
    self.num_production_rules = len(self.prod_rules)

    if self.padding_at_end:
      self.padding_rule_index = self.num_production_rules - 1
      self.start_index = self.prod_rules[0].lhs()
      self.start_rule = self.prod_rules[0]
    else:
      self.padding_rule_index = 0
      self.start_index = self.prod_rules[1].lhs()
      self.start_rule = self.prod_rules[1]

    # Map from production rule to its index in prod_rules
    self.prod_rule_to_index = {
        str(prod_rule): i
        for i, prod_rule in enumerate(self.prod_rules)
    }

    # Left hand side (LHS) symbols of each production rule.
    self.prod_rule_lhs = [
        prod_rule.lhs().symbol() for prod_rule in self.prod_rules
    ]

    # List of unique lhs as a lookup table of index and symbol.
    self.unique_lhs = sorted(list(set(self.prod_rule_lhs)))
    self.num_unique_lhs = len(self.unique_lhs)

    # Map from unique lhs to its index in unique_lhs
    self.lhs_to_index = {
        lhs_sym: i
        for i, lhs_sym in enumerate(self.unique_lhs)
    }

    # For each lhs symbol, which production rules should be masked.
    masks = np.zeros((self.num_unique_lhs, self.num_production_rules))
    for i, sym in enumerate(self.unique_lhs):
      masks[i] = np.array([lhs_sym == sym for lhs_sym in self.prod_rule_lhs])
    self.masks = masks

    # From the index of production rule to the index of its lhs
    # in list of unique lhs
    prod_rule_index_to_lhs_index = np.zeros(self.num_production_rules,
                                            dtype=int)
    for i in range(self.num_production_rules):
      prod_idx = self.prod_rule_lhs[i]
      prod_rule_index_to_lhs_index[i] = self.unique_lhs.index(prod_idx)
    self.prod_rule_index_to_lhs_index = prod_rule_index_to_lhs_index

    # The indices of rhs symbols in unique_lhs for each production rule.
    prod_rule_rhs_indices = []
    for prod_rule in self.prod_rules:
      rhs_indices = []
      for sym in prod_rule.rhs():
        # Use six.string_types for python 2/3 compatibility
        if not isinstance(sym, six.string_types):
          s = sym.symbol()
          rhs_indices.extend(
              [i for i, lhs_sym in enumerate(self.unique_lhs) if lhs_sym == s])
      prod_rule_rhs_indices.append(rhs_indices)
    self.prod_rule_rhs_indices = prod_rule_rhs_indices

    # The maximum size of the indices list in prod_rule_rhs_indices
    self.max_rhs_indices_size = max(list(map(len, self.prod_rule_rhs_indices)))
    return self._cfg

  def get_grammar(self):
    """Gets nltk.grammar.CFG object of context-free grammar.

    Returns:
      nltk.grammar.CFG object for context-free grammar.
    """
    return self._cfg

  def grammar_to_string(self, indent=True):
    r"""Display grammar production rules as a string.

    If the grammar contains production rules
    Nothing -> None
    S -> T
    T -> '1'

    The output string will be
    "\t0: Nothing -> None\n\t1: S -> T\n\t2:T -> '1'\n"

    The order of production rules is the order in prod_rules attribute.

    Args:
      indent: Boolean, whether to add indent before each line. Indent is
          required for markdown format.

    Returns:
      String.
    """
    output = []
    for i, production_rule in enumerate(self.prod_rules):
      output.append('%d: %s\n' % (i, str(production_rule)))
    if indent:
      output = ['\t%s' % line for line in output]
    return ''.join(output)

  def parse_expressions_to_indices_sequences(
      self, expression_strings, max_length):
    """Parses a list of expression strings to sequences of indices of rules.

    Args:
      expression_strings: A list of strings, each string is an expression
          string.
      max_length: Int, the maximum length of the production rule sequence.

    Returns:
      expression_tensor: Numpy array with shape
          [num_expression, max_length]. The max_length is the
          maximum length of the production rule sequence. Each element is an
          index of a production rule in grammar. For sequence shorter than
          max_length, it will be padded by dummy production rule
          (index = padding_rule_index) at the end.

    Raises:
      ValueError: If the number of production rules to required to represent an
          expression is greater than max_length or expression cannot be parsed;
          the expression_strings is not list, tuple or np.ndarray; or the
          expression_strings can not be parsed to production rules.
    """
    if not isinstance(expression_strings, (list, tuple, np.ndarray)):
      raise ValueError('expression_strings is expected to be list, but got %s.'
                       % type(expression_strings))
    indices = []
    for expression_string in expression_strings:
      try:
        # Parse by context-free grammar tree.
        # Get the production rules by preorder traversal.
        # NOTE(leeley): nltk parser will return several ways to parse the
        # expression string. self.parser.parse() iterator returns those parsing
        # results from most likely to least likely. So we choose the first
        # parsing result (most likely).
        prod_rules = next(self.parser.parse(
            # Tokenize the expression strings.
            expression_string.split(self.delimiter))).productions()
      except StopIteration:
        raise ValueError('%s cannot be parsed to production rules.' %
                         expression_string)
      num_prod_rules = len(prod_rules)
      if num_prod_rules > max_length:
        raise ValueError('The number of production rules to parse expression '
                         '%s is %d, which can not be greater than max_length '
                         '%d.' % (expression_string, num_prod_rules, max_length)
                        )
      prod_rule_indices = [self.prod_rule_to_index[str(prod_rule)]
                           for prod_rule in prod_rules]
      prod_rule_indices += [self.padding_rule_index] * (
          max_length - len(prod_rule_indices))
      indices.append(prod_rule_indices)
    return np.asarray(indices, dtype=np.int32)

  def parse_expressions_to_tensor(self, expression_strings, max_length):
    """Parses a list of expression strings to an expression tensor.

    Args:
      expression_strings: A list of strings, each string is an expression
          string.
      max_length: Int, the maximum length of the production rule sequence.

    Returns:
      expression_tensor: Numpy array with shape
          [num_expression, max_length, vector_size]. The max_length is the
          maximum length of the production rule sequence. The vector_size is the
          total number of production rules.
    """
    indices = self.parse_expressions_to_indices_sequences(
        expression_strings, max_length)
    expression_tensor = np.zeros(
        (len(indices), max_length, self.num_production_rules), dtype=np.float32)
    for i, prod_rule_indices in enumerate(indices):
      expression_tensor[i, np.arange(max_length), prod_rule_indices] = 1.
    return expression_tensor

  def generate(self, tree_depth, num_expressions):
    """Generates expression strings from context-free grammar.

    Args:
      tree_depth: Integer, depth of the grammar parsing tree.
      num_expressions: Integer, maximum number of expressions to generate.

    Yields:
      List of token strings for an expression string.
    """
    for token_list in generate.generate(self._cfg,
                                        depth=tree_depth,
                                        n=num_expressions):
      yield token_list
