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

r"""Generates empirical distribution dataframe of next production rule.

There are two different empirical distribution dataframes we can generate. One
is the empirical distribution of the next production rule for a partial
sequence, and the other is the empirical distribution of the next production
rule for the tail of a partial sequence. "Tail" means that we just use the last
several production rules from the partial sequence. It uses limited history (or
called partial information) to determine the next production rule.

Each row of the dataframe contains a string of partial sequence indices (or tail
of it), and the conditions of the complete sequence (such as leading power at
zero and leading power at inf). These are placed as multi-indices of the
dataframe. The columns of the dataframe are the probabilities of the next
production rule.

Here is an example of the first kind of dataframe:
partial_sequence_indices  leading_at_0  leading_at_inf  0  1  2   ...
        1_4_3_5                -1            -1         0  0  0.5 ...

Here is an example of the second kind of dataframe with tail_length=2:
tail_partial_sequence_indices  leading_at_0  leading_at_inf  0  1  2   ...
            3_5                     -1            -1         0  0  0.5 ...

How is the dataframe computed:
For each given string of partial sequence indices (or tail of it) and
conditions, find all the matches from the training set. This gives a number of
next production rules. Compute the empirical distribution of these production
rules. Note that for the tail partial sequence, the next production rule mask of
the entire sequence is needed to ensure the validity of the proposed next
production rule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import pandas as pd

from neural_guided_symbolic_regression.utils import evaluators
from neural_guided_symbolic_regression.utils import postprocessor


def load_examples(sstable_pattern, features):
  """Loads expression examples from sstables.

  Args:
    sstable_pattern: String, pattern of input sstables.
    features: List of strings, list of features to extract from sstables.

  Returns:
    A Pandas dataframe, each row of which gives an expression and its property
    values.
  """
  data = evaluators.load_examples_from_sstable(sstable_pattern, features)
  expression_df = pd.DataFrame.from_dict(
      data, orient='index').reset_index(drop=True)
  return expression_df


def get_number_valid_next_step(prod_rules_sequence_indices, grammar):
  """Gets number of valid next production rules.

  Args:
    prod_rules_sequence_indices: A 1-D Numpy array of indices of a production
        rule sequence.
    grammar: A grammar object.

  Returns:
    An integer representing the number of valid next production rules.
  """
  prod_rules_sequence = [
      grammar.prod_rules[i] for i in prod_rules_sequence_indices
  ]
  stack = postprocessor.production_rules_sequence_to_stack(prod_rules_sequence)
  return int(sum(grammar.masks[grammar.lhs_to_index[stack.peek()]]))


def get_partial_sequence_df(expression_df, grammar, max_length):
  """Gets partial sequence dataframe.

  Args:
    expression_df: A Pandas dataframe, each row of which gives an expression and
        its property values.
    grammar: A grammar object.
    max_length: Integer, the maximum length of a production rule sequence.

  Returns:
    A partial sequence dataframe obtained from the expression dataframe. It has
    columns expression_string, partial_sequence_indices,
    partial_sequence_length, number_valid_next_step, next_step_index,
    sequence_length, remain_sequence_length and property values.
  """
  expression_strings = expression_df['expression_string'].values
  expression_tensor = grammar.parse_expressions_to_indices_sequences(
      expression_strings, max_length=max_length)
  data_dict = collections.defaultdict(list)
  for (expression_string, expression_array_indices) in zip(
      expression_strings, expression_tensor):
    sequence_length = len(expression_array_indices[
        expression_array_indices != grammar.padding_rule_index])
    for partial_sequence_length in range(1, sequence_length):
      data_dict['expression_string'].append(expression_string)
      # Convert partial sequence to string so it is hashable.
      data_dict['partial_sequence_indices'].append('_'.join(
          map(str, expression_array_indices[:partial_sequence_length])))
      data_dict['partial_sequence_length'].append(partial_sequence_length)
      data_dict['number_valid_next_step'].append(
          get_number_valid_next_step(
              expression_array_indices[:partial_sequence_length], grammar))
      data_dict['next_step_index'].append(
          expression_array_indices[partial_sequence_length])
      data_dict['sequence_length'].append(sequence_length)
      data_dict['remain_sequence_length'].append(sequence_length -
                                                 partial_sequence_length)
  partial_sequence_df = pd.DataFrame(data_dict)
  return pd.merge(
      partial_sequence_df, expression_df, on='expression_string', how='left')


def extract_tail_partial_sequence(partial_sequence_string, tail_length):
  """Extracts the tail of a partial sequence.

  If the specified tail length is larger than the partial sequence length, get
  the entire partial sequence. For example, when partial_sequence_string is
  '1_6' and tail_length is 4, it should give '1_6'.

  Args:
    partial_sequence_string: String, a partial sequence string, such as
        '1_6_7_6'.
    tail_length: Integer, length of the tail partial sequence to extract.

  Returns:
    The extracted tail partial sequence. When partial_sequence_string is
    '1_6_7_6' and tail_length is 2, it should give '7_6'.
  """
  partial_sequence_indices = partial_sequence_string.split('_')
  return '_'.join(partial_sequence_indices[-tail_length:])


def get_empirical_distribution_df(partial_sequence_df,
                                  properties,
                                  num_production_rules,
                                  tail_length):
  """Gets dataframe of empirical probabilities for next production rule.

  Args:
    partial_sequence_df: A partial sequence dataframe obtained from the
        expression dataframe. It has columns expression_string,
        partial_sequence_indices, partial_sequence_length,
        number_valid_next_step, next_step_index, sequence_length,
        remain_sequence_length and property values.
    properties: List of symbolic properties contained in the empirical
        probability dataframe. It can be an empty list, then the empirical
        distribution dataframe will only have partial_sequence (or tail of it)
        as the single index.
    num_production_rules: Integer, the number of production rules in grammar.
    tail_length: Integer, length of the tail partial sequence. If None, use the
        entire partial sequence.

  Returns:
    Pandas dataframe recording the empirical probability distribution of the
    next production rule under various settings of partial_sequence_indices and
    conditions. Each row gives the probability distribution of the next
    production rule corresponding to one particular partial_sequence (or tail of
    it), and conditions such as leading_at_0 and leading_at_inf.
    The partial_sequence (or tail of it), and conditions are placed in the
    dataframe as multi-indices. The columns are the probabilities of the next
    production rule (the rules are represented by indices), e.g.:
    partial_sequence_indices  leading_at_0  leading_at_inf  0  1  2   ...
            1_4_3_5                -1            -1         0  0  0.5 ...
  """
  if tail_length is None:
    groupby_cols = ['partial_sequence_indices'] + properties
  else:
    partial_sequence_df['tail_partial_sequence_indices'] = (
        partial_sequence_df['partial_sequence_indices'].apply(
            lambda x: extract_tail_partial_sequence(x, tail_length)))
    groupby_cols = ['tail_partial_sequence_indices'] + properties
  empirical_distribution_df = partial_sequence_df.groupby(
      by=groupby_cols)['next_step_index'].value_counts(
          normalize=True).to_frame('probability')
  empirical_distribution_df = pd.pivot_table(
      empirical_distribution_df,
      values='probability',
      index=groupby_cols,
      columns=['next_step_index'],
      fill_value=0)
  # Add missing columns.
  missing_cols = list(
      set(range(num_production_rules)).difference(
          empirical_distribution_df.columns))
  for col in missing_cols:
    empirical_distribution_df[col] = 0
  # Reorder columns.
  return empirical_distribution_df[range(num_production_rules)]
