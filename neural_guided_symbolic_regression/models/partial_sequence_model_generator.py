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

"""Generates full sequences of expressions by trained partial sequence model.

Model trained in run_partial_sequence_model.py predicts the next production rule
from partial sequence and conditions.

To generate expressions, the partial sequence model is loaded from a given
checkpoint. The generation starts from partial sequence of length 1, predicts
the next production rule. The next production rule will be appended to the
current partial sequence. The new partial sequence of length 2 will be used as
an new input, and the model will predict the next production rule. The next
production rule will be appended to the current partial sequence to form a new
partial sequence of length 3. This process will be repeated until all symbols
are terminal.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import six
from six.moves import map
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.mcts import states


def load_model_to_session(export_dir, sess=None):
  """Loads a SavedModel to session.

  Args:
    export_dir: String, the directory in which the SavedModel protocol buffer
        and variables to be loaded are located.
        For example, "/path/to/dir/latest_exported_model/1538797255/"
    sess: tf.Session, the TensorFlow session to restore the variables. Default
        None, a new session with its own graph will be created.

  Returns:
    A tf.Session with all the model variables restored.
  """
  with tf.Graph().as_default():
    sess = sess or tf.Session()
    tf.saved_model.loader.load(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        export_dir=export_dir)
    return sess


def generate_next_production_rule_randomly(
    num_production_rules, next_production_rule_distribution, random_state):
  """Generates next production rule randomly from grammarly valid rules.

  Args:
    num_production_rules: Integer, the number of production rules in grammar.
    next_production_rule_distribution: Float numpy array with shape
        [num_production_rules], the (possibly unnormalized) probability
        distribution of production rules. Invalid production rules have a
        probability of zero.
    random_state: Numpy RandomState. Default None.

  Returns:
    Integer. The index of the next production rule.
  """
  if random_state is None:
    random_state = np.random.RandomState()
  return random_state.choice(
      num_production_rules,
      # next_production_rule_distribution is float numpy array.
      # divide np.sum(next_production_rule_mask) to normalize the
      # probability anyway.
      p=next_production_rule_distribution /
      np.sum(next_production_rule_distribution))


def get_masked_probabilities_from_model(
    sess,
    max_length,
    partial_sequence,
    next_production_rule_mask,
    conditions=None,
    input_variable_scope='serving_input'):
  """Gets masked probabilities from model.

  Args:
    sess: tf.Session, the session contains the trained model to predict next
        production rule from input partial sequence.
    max_length: Integer, the max length of production rule sequence.
    partial_sequence: Integer numpy array with shape [partial_sequence_length].
        It will be padded into max_length.
    next_production_rule_mask: Float numpy array with shape
        [num_production_rules], the mask of valid production rule.
    conditions: Dict of numpy arrays.
    input_variable_scope: String, the variable scope for the tensor in input
        features. Default 'serving_input'. Used when sess is not None.

  Returns:
    Float numpy array with shape [num_production_rules].
  """
  partial_sequence_length = len(partial_sequence)

  if partial_sequence_length > max_length:
    raise ValueError(
        'The length of partial_sequence (%d) cannot be greater than '
        'max_length (%d).' % (partial_sequence_length, max_length))

  features = {
      'partial_sequence': np.array([
          np.pad(
              partial_sequence,
              pad_width=(0, max_length - partial_sequence_length),
              mode='constant',
              constant_values=0)], dtype=np.int32),
      'partial_sequence_length': np.array(
          [partial_sequence_length], dtype=np.int32),
      'next_production_rule_mask': np.array([next_production_rule_mask]),
  }

  if conditions:
    # Check whether the corresponding tensor exist in the graph in the input
    # session.
    graph = sess.graph
    for condition_key, condition_array in six.iteritems(conditions):
      tensor_name = '%s/%s:0' % (input_variable_scope, condition_key)
      try:
        graph.get_tensor_by_name(tensor_name)
      except KeyError:
        logging.warning('%s does not exist in graph.', tensor_name)
        continue
      features[condition_key] = condition_array

  return sess.run(
      'predictions/masked_probabilities:0',
      feed_dict={
          # Complete tensor name in features for serving.
          '%s/%s:0' % (input_variable_scope, key): value
          for key, value in six.iteritems(features)
      })[0]  # input batch size is 1, use [0] to get the element.


def generate_next_production_rule_from_model(
    sess,
    max_length,
    partial_sequence,
    next_production_rule_mask,
    conditions=None,
    sampling=False,
    random_state=None,
    input_variable_scope='serving_input'):
  """Generates next production rule from trained model in sess.

  Args:
    sess: tf.Session, the session contains the trained model to predict next
        production rule from input partial sequence.
    max_length: Integer, the max length of production rule sequence.
    partial_sequence: Integer numpy array with shape [partial_sequence_length].
        It will be padded into max_length.
    next_production_rule_mask: Float numpy array with shape
        [num_production_rules], the mask of valid production rule.
    conditions: Dict of numpy arrays.
    sampling: Boolean, whether to do sampling. If True, the next production rule
        will be sampled from the probabilities predicted by the partial sequence
        model. If False, the generator deterministically chooses the next
        production rule with highest probability at each step.
    random_state: Numpy RandomState. Default None. This is used when sampling
        is True.
    input_variable_scope: String, the variable scope for the tensor in input
        features. Default 'serving_input'. Used when sess is not None.

  Returns:
    Integer. The index of the next production rule.
  """
  masked_probabilities = get_masked_probabilities_from_model(
      sess=sess,
      max_length=max_length,
      partial_sequence=partial_sequence,
      next_production_rule_mask=next_production_rule_mask,
      conditions=conditions,
      input_variable_scope=input_variable_scope)

  if sampling:
    if random_state is None:
      random_state = np.random.RandomState()
    next_production_rule = random_state.choice(
        len(masked_probabilities), p=masked_probabilities)
  else:
    next_production_rule = np.argmax(masked_probabilities)

  return next_production_rule


def get_next_production_rule_distribution(
    empirical_distribution_df,
    tail_length,
    current_partial_sequence_indices,
    symbolic_properties_dict,
    next_production_rule_mask):
  """Gets next production rule probabilities from empirical distribution df.

  If there are more than one empirical probability distributions available given
  the condition in the empirical distribution dataframe, we simply take their
  average.

  Args:
    empirical_distribution_df: Pandas dataframe recording the empirical
        probability distribution of the next production rule under various
        settings of partial_sequence_indices and conditions. Each row gives the
        probability distribution of the next production rule corresponding to
        one particular partial_sequence and conditions such as leading_at_0 and
        leading_at_inf. The partial_sequence and conditions are placed in the
        dataframe as multi-indices. The columns are the probabilities of the
        next production rule (the rules are represented by indices), e.g.:
        partial_sequence_indices  leading_at_0  leading_at_inf  0  1  2   ...
                1_4_3_5                -1            -1         0  0  0.5 ...
    tail_length: Integer, length of the tail partial sequence used for
        generating the empirical distribution dataframe. If None, the entire
        partial sequence is used.
    current_partial_sequence_indices: String, current partial sequence indices
        represented as sequence indices concatenated by the underscore sign.
        E.g.: 1_5_6_8_7_3_5_6_9_8.
    symbolic_properties_dict: Dict, the keys are the symbolic properties used as
        conditions. Values are the corresponding desired values of the symbolic
        properties.
    next_production_rule_mask: Float numpy array with shape
        [num_production_rules], the mask of valid production rule.

  Returns:
    Float numpy array with shape [num_production_rules], probability
    distribution of the next production rule. If there is no rule found, return
    None.
  """
  if tail_length is None:
    level_name = 'partial_sequence_indices'
    current_entire_or_tail_partial_sequence_indices = (
        current_partial_sequence_indices)
    effective_next_production_rule_mask = np.ones(
        len(next_production_rule_mask))
  else:
    level_name = 'tail_partial_sequence_indices'
    current_entire_or_tail_partial_sequence_indices = (
        '_'.join(current_partial_sequence_indices.split('_')[-tail_length:]))
    effective_next_production_rule_mask = next_production_rule_mask
  entire_or_tail_partial_sequence_indices = (
      set(empirical_distribution_df.index.get_level_values(
          level=level_name).values))

  if current_entire_or_tail_partial_sequence_indices not in (
      entire_or_tail_partial_sequence_indices):
    return None
  # Subset the empirical distribution dataframe by matching the (tail) partial
  # sequence.
  empirical_distribution_partial_sequence = empirical_distribution_df[
      empirical_distribution_df.index.get_level_values(level=level_name).values
      == current_entire_or_tail_partial_sequence_indices]

  # Obtain the condition according to what is available in the empirical
  # distribution dataframe.
  symbolic_properties = list(
      set(empirical_distribution_partial_sequence.index.names).intersection(
          symbolic_properties_dict))
  # Subset the empirical distribution dataframe by matching each symbolic
  # property available in the empirical distribution dataframe.
  for symbolic_property in symbolic_properties:
    values = empirical_distribution_partial_sequence.index.get_level_values(
        level=symbolic_property).unique()
    if symbolic_properties_dict[symbolic_property] in values:
      empirical_distribution_partial_sequence = (
          empirical_distribution_partial_sequence[(
              empirical_distribution_partial_sequence.index.get_level_values(
                  level=symbolic_property).values
              == symbolic_properties_dict[symbolic_property])])
    else:
      return None
  # We simply take the average if there are more than one probability
  # distributions available given the condition.
  next_production_rule_distribution = (
      empirical_distribution_partial_sequence.mean(axis=0).values)
  # Apply the mask.
  next_production_rule_distribution *= effective_next_production_rule_mask
  return next_production_rule_distribution


def _get_starting_partial_sequence(partial_sequence, grammar, random_state):
  """Gets the starting partial sequence for generation.

  Args:
    partial_sequence: List of integers, the partial sequence to start the
        generation. If None, the generation will start from scratch.
    grammar: arithmetic_grammar.Grammar object.
    random_state: Numpy RandomState.

  Returns:
    List of integers.
  """
  if partial_sequence is None:
    # NOTE(leeley): The input partial sequence to the partial sequence model
    # should at least have length 1.
    # Randomly select the first production rule.
    valid_first_production_rule_indices = np.arange(
        grammar.num_production_rules)[
            grammar.masks[
                grammar.lhs_to_index[grammar.start_index.symbol()]
            ].astype(bool)]
    first_production_rule_index = random_state.choice(
        valid_first_production_rule_indices)
    partial_sequence = [first_production_rule_index]
  else:
    # NOTE(leeley): Make a copy since partial_sequence will be modified in
    # place in this function. The input list shouldn't be affected.
    partial_sequence = partial_sequence[:]
  return partial_sequence


def generate_expression(sess,
                        grammar,
                        max_length,
                        symbolic_properties_dict=None,
                        numerical_values=None,
                        clip_value_min=None,
                        clip_value_max=None,
                        random_state=None,
                        sampling=False,
                        empirical_distribution_df=None,
                        tail_length=None,
                        partial_sequence=None,
                        input_variable_scope='serving_input'):
  """Generates an expression by a trained partial sequence model.

  Args:
    sess: tf.Session, the session contains the trained model to predict next
        production rule from input partial sequence. If None, each step will be
        selected randomly.
    grammar: arithmetic_grammar.Grammar object.
    max_length: Integer, the max length of production rule sequence.
    symbolic_properties_dict: Dict, the keys are the symbolic properties used as
        conditions. Values are the corresponding desired values of the symbolic
        properties.
    numerical_values: Float numpy array with shape [num_numerical_points]. The
        value of expression evaluated on points.
    clip_value_min: Float, the minimum value to clip by.
    clip_value_max: Float, the maximum value to clip by.
    random_state: Numpy RandomState. Default None.
    sampling: Boolean, whether to do sampling. If True, the next production rule
        will be sampled from the probabilities predicted by the partial sequence
        model. If False, the generator deterministically chooses the next
        production rule with highest probability at each step.
    empirical_distribution_df: Pandas dataframe recording the empirical
        probability distribution of the next production rule under various
        settings of partial_sequence_indices and conditions. Each row gives the
        probability distribution of the next production rule corresponding to
        one particular partial_sequence (or a tail of it), and conditions such
        as leading_at_0 and leading_at_inf. The partial_sequence (or a tail of
        it) and conditions are placed in the dataframe as multi-indices. The
        columns are the probabilities of the next production rule (the rules are
        represented by indices), e.g.:
        partial_sequence_indices  leading_at_0  leading_at_inf  0  1  2   ...
                1_4_3_5                -1            -1         0  0  0.5 ...
    tail_length: Integer, length of the tail partial sequence used for
        generating the empirical distribution dataframe. If None, the entire
        partial sequence is used.
    partial_sequence: List of integers, the partial sequence to start the
        generation. Default None, the generation will start from scratch.
    input_variable_scope: String, the variable scope for the tensor in input
        features. Default 'serving_input'. Used when sess is not None.

  Returns:
    Dict with the following keys:
      * 'expression_string': String.
      * 'is_terminal': Boolean, whether all the symbols in the generated
            expression are terminal.
      * 'production_rule_sequence': List of integers, the indices of generated
            sequence of production rules in grammar.
      * 'history': List of strings, the history of expression generation.

  Raises:
    ValueError: The proposed probability distribution of the next production
        rule is invalid.
  """
  if sess is None:
    logging.info('Input sess is None, '
                 'each step in the generator will be selected randomly.')

  if random_state is None:
    random_state = np.random.RandomState()

  conditions = {}
  if symbolic_properties_dict is not None:
    conditions.update({
        key: np.array([value], dtype=np.float32)
        for key, value in six.iteritems(symbolic_properties_dict)
    })
  if numerical_values is not None:
    conditions['numerical_values'] = np.atleast_2d(
        np.clip(numerical_values, clip_value_min, clip_value_max)
        ).astype(np.float32)

  partial_sequence = _get_starting_partial_sequence(
      partial_sequence=partial_sequence,
      grammar=grammar,
      random_state=random_state)

  # NOTE(leeley): ProductionRulesState records (partial) expression by
  # non-terminal symbol stack and sequence of production rule objects.
  # partial_sequence is used to record the indices of production rules in
  # grammar instead of production rule objects.
  state = states.ProductionRulesState(
      production_rules_sequence=[
          grammar.prod_rules[production_rule_index]
          for production_rule_index in partial_sequence
      ])

  while len(partial_sequence) < max_length and not state.is_terminal():
    next_production_rule_mask = grammar.masks[
        grammar.lhs_to_index[state.stack_peek()]]

    if sess is None:
      if empirical_distribution_df is None:
        next_production_rule_distribution = next_production_rule_mask
      else:
        current_partial_sequence_indices = '_'.join(map(str, partial_sequence))
        next_production_rule_distribution = (
            get_next_production_rule_distribution(
                empirical_distribution_df,
                tail_length,
                current_partial_sequence_indices,
                symbolic_properties_dict,
                next_production_rule_mask))
        logging.info('Current partial sequence indices: %s.',
                     current_partial_sequence_indices)
        logging.info('Symbolic properties dict: %s.', symbolic_properties_dict)
        logging.info('Next production rule probabilities: %s.',
                     next_production_rule_distribution)
        # If there is no rule found, leave the sequence unterminated.
        if next_production_rule_distribution is None:
          break
      next_production_rule = generate_next_production_rule_randomly(
          num_production_rules=grammar.num_production_rules,
          next_production_rule_distribution=next_production_rule_distribution,
          random_state=random_state)
    else:
      next_production_rule = generate_next_production_rule_from_model(
          sess=sess,
          max_length=max_length,
          partial_sequence=partial_sequence,
          next_production_rule_mask=next_production_rule_mask,
          conditions=conditions,
          sampling=sampling,
          random_state=random_state,
          input_variable_scope=input_variable_scope)

    # Update the partial sequence as input features for next round.
    partial_sequence.append(next_production_rule)
    # Update the expression state.
    state.append_production_rule(grammar.prod_rules[next_production_rule])

  return {
      'expression_string': state.get_expression(),
      'is_terminal': state.is_terminal(),
      'production_rule_sequence': partial_sequence,
      'history': state.generate_history(),
  }
