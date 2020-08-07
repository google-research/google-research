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

"""Metrics for tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging
import numpy as np
from six.moves import range
import sympy
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import partial_sequence_model_generator
from neural_guided_symbolic_regression.utils import evaluators
from neural_guided_symbolic_regression.utils import postprocessor
from neural_guided_symbolic_regression.utils import symbolic_properties
from neural_guided_symbolic_regression.utils import timeout
from tensorflow.contrib import metrics as contrib_metrics


# pylint: disable=unbalanced-tuple-unpacking


def evaluate_expression(expression_string, grids, symbol):
  """Evaluates expression.

  Args:
    expression_string: String. The univariate expression, for example
        'x * x + 1 / x'.
    grids: Numpy array with shape [num_grid_points], the points to evaluate
        expression.
    symbol: String. Symbol of variable in expression.

  Returns:
    Numpy array with shape [num_grid_points].
  """
  try:
    expression_on_grids = evaluators.numpy_array_eval(
        str(sympy.simplify(expression_string)), arguments={symbol: grids})
  except SyntaxError as error:
    # NOTE(leeley): In some rare cases, after sympy.simplify(),
    # expression_string will contain symbols which can not be parsed,
    # for example 'zoo'. If this occurs, evaluate expression without
    # simplification.
    logging.warning(error)
    logging.warning('SyntaxError occurs after sympy.simplify(), '
                    'evaluate %s directly without simplification.',
                    expression_string)
    expression_on_grids = evaluators.numpy_array_eval(
        expression_string, arguments={symbol: grids})

  if np.asarray(expression_on_grids).size == 1:
    expression_on_grids = expression_on_grids * np.ones_like(grids)
  return expression_on_grids


def compute_rmse(
    expression_string_1,
    expression_string_2,
    values):
  """Computes rmse of two expressions on given values.

  Args:
    expression_string_1: String, an expression.
    expression_string_2: String, the other expression.
    values: Numpy array with shape [num_values]. The values to evaluate the
        difference between two expressions.

  Returns:
    Float.
  """
  output_values_1 = evaluate_expression(
      expression_string=expression_string_1, grids=values, symbol='x')
  output_values_2 = evaluate_expression(
      expression_string=expression_string_2, grids=values, symbol='x')
  return np.sqrt(np.mean((output_values_1 - output_values_2) ** 2))


def evaluate_leading_powers_at_0_inf(expression_string, symbol):
  """Evaluates leading powers at 0 and inf.

  Args:
    expression_string: String. The univariate expression, for example
        'x * x + 1 / x'.
    symbol: String. Symbol of variable in expression.

  Returns:
    leading_at_0: Float, leading power at 0.
    leading_at_inf: Float, leading power at inf.
  """
  try:
    leading_at_0 = timeout.RunWithTimeout(
        functools.partial(
            symbolic_properties.get_leading_power,
            x0='0',
            symbol=symbol,
            coefficients=None),
        args=(expression_string,),
        name='symbolic_properties.get_leading_power_at_0').run(
            time_limit_seconds=30)
    leading_at_inf = timeout.RunWithTimeout(
        functools.partial(
            symbolic_properties.get_leading_power,
            x0='inf',
            symbol=symbol,
            coefficients=None),
        args=(expression_string,),
        name='symbolic_properties.get_leading_power_at_inf').run(
            time_limit_seconds=30)
    return leading_at_0, leading_at_inf
  except (timeout.FunctionTimeoutError, ValueError):
    logging.info('Fail to compute leading power for %s', expression_string)
    return np.nan, np.nan


def probabilities_info_string(probabilities, next_production_rule, grammar):
  """Generates string of softmax logtis information.

  Args:
    probabilities: Float numpy array with shape [num_production_rules].
    next_production_rule: Integer. The index of the next production rule.
    grammar: arithmetic_grammar.Grammar object.

  Returns:
    A list of string.
  """
  output_info = []
  argmax_index = np.argmax(probabilities)
  output_info.append(
      '%s, probability: %4.2f'
      % (str(grammar.prod_rules[argmax_index]), probabilities[argmax_index]))
  probabilities_list = []
  for i, value in enumerate(probabilities):
    if i == next_production_rule or i == argmax_index:
      probabilities_list.append('*%4.2f*' % value)
    else:
      probabilities_list.append(' %4.2f ' % value)
  output_info.append('|'.join(probabilities_list))
  return output_info


def next_production_rule_info(
    expression_string,
    partial_sequence,
    partial_sequence_length,
    next_production_rule,
    unmasked_probabilities,
    masked_probabilities,
    grammar):
  """Converts information of next production rule prediction to a string.

  Args:
    expression_string: String. Expression where the partial sequence is sampled
        from.
    partial_sequence: Integer numpy array with shape [max_length].
    partial_sequence_length: Integer. The length of partial sequence. The input
        partial_sequence has padding at the end.
        partial_sequence[:partial_sequence_length] is the actual partial
        sequence.
    next_production_rule: Integer. The index of the next production rule.
    unmasked_probabilities: Float numpy array with shape
        [num_production_rules]. The probabilities from the model prediction
        without valid production rule mask.
    masked_probabilities: Float numpy array with shape
        [num_production_rules]. The probabilities from the model prediction
        after applied valid production rule mask.
    grammar: arithmetic_grammar.Grammar object.

  Returns:
    String. The information of next production rule prediction.
  """
  output_info = ['expression string:', expression_string]

  prod_rules_sequence = [
      grammar.prod_rules[index]
      for index in partial_sequence[:partial_sequence_length]]
  output_info.append('partial expression:')
  output_info.append(
      postprocessor.production_rules_sequence_to_expression_string(
          prod_rules_sequence=prod_rules_sequence, delimiter=' '))

  output_info.append('true next production rule:')
  output_info.append(str(grammar.prod_rules[next_production_rule]))

  output_info.append('unmasked prediction next production rule:')
  output_info.extend(
      probabilities_info_string(
          probabilities=unmasked_probabilities,
          next_production_rule=next_production_rule,
          grammar=grammar))

  output_info.append('masked prediction next production rule:')
  output_info.extend(
      probabilities_info_string(
          probabilities=masked_probabilities,
          next_production_rule=next_production_rule,
          grammar=grammar))

  # Add '\t' for markdown display in tensorboard.
  return '\n'.join(['\t' + line for line in output_info])


def next_production_rule_info_batch(
    expression_strings,
    partial_sequences,
    partial_sequence_lengths,
    next_production_rules,
    unmasked_probabilities_batch,
    masked_probabilities_batch,
    grammar):
  """Converts information of a batch next production rule prediction to strings.

  Args:
    expression_strings: String numpy array with shape [batch_size].
    partial_sequences: Integer numpy array with shape [batch_size, max_length].
    partial_sequence_lengths: Integer numpy array with shape [batch_size].
    next_production_rules: Integer numpy array with shape [batch_size]. The
        indice of the next production rules.
    unmasked_probabilities_batch: Float numpy array with shape
        [batch_size, num_production_rules]. The probabilities from the model
        prediction without valid production rule mask.
    masked_probabilities_batch: Boolean numpy array with shape
        [batch_size, num_production_rules]. The probabilities from the model
        prediction after applied valid production rule mask.
    grammar: arithmetic_grammar.Grammar object.

  Returns:
    String numpy array with shape [batch_size]. The information strings of next
    production rule prediction.
  """
  output_info = []
  for i in range(len(expression_strings)):
    output_info.append(next_production_rule_info(
        expression_string=expression_strings[i],
        partial_sequence=partial_sequences[i],
        partial_sequence_length=partial_sequence_lengths[i],
        next_production_rule=next_production_rules[i],
        unmasked_probabilities=unmasked_probabilities_batch[i],
        masked_probabilities=masked_probabilities_batch[i],
        grammar=grammar))
  return np.asarray(output_info, dtype=np.unicode_)


def next_production_rule_info_batch_text_summary(
    expression_strings,
    partial_sequences,
    partial_sequence_lengths,
    next_production_rules,
    unmasked_probabilities_batch,
    masked_probabilities_batch,
    grammar,
    target_length=None):
  """Ceates text summary for a batch next production rule prediction.

  Args:
    expression_strings: String tensor with shape [batch_size].
    partial_sequences: Integer tensor with shape [batch_size, max_length].
    partial_sequence_lengths: Integer tensor with shape [batch_size].
    next_production_rules: Integer tensor with shape [batch_size]. The
        indice of the next production rules.
    unmasked_probabilities_batch: Float tensor with shape
        [batch_size, num_production_rules]. The probabilities from the model
        prediction without valid production rule mask.
    masked_probabilities_batch: Boolean tensor with shape
        [batch_size, num_production_rules]. The probabilities from the model
        prediction after applied valid production rule mask.
    grammar: arithmetic_grammar.Grammar object.
    target_length: Integer. Only examples with partial sequence length equal to
        target_length will be used. If None (the default), all examples in
        batch will be used.

  Returns:
    summary: String Tensor containing a Summary proto.
    update_op: Op that updates summary (and the underlying stream).
  """
  if target_length is not None:
    (expression_strings,
     partial_sequences,
     partial_sequence_lengths,
     next_production_rules,
     unmasked_probabilities_batch,
     masked_probabilities_batch) = mask_by_partial_sequence_length(
         tensors=(
             expression_strings,
             partial_sequences,
             partial_sequence_lengths,
             next_production_rules,
             unmasked_probabilities_batch,
             masked_probabilities_batch),
         partial_sequence_lengths=partial_sequence_lengths,
         target_length=target_length)
    suffix = '/length_%d' % target_length
  else:
    suffix = ''

  info = tf.py_func(
      functools.partial(next_production_rule_info_batch, grammar=grammar),
      [expression_strings,
       partial_sequences,
       partial_sequence_lengths,
       next_production_rules,
       unmasked_probabilities_batch,
       masked_probabilities_batch],
      tf.string,
      name='py_func-next_production_rule_info_batch_text_summary' + suffix)
  info.set_shape([expression_strings.shape[0]])
  value, update_op = contrib_metrics.streaming_concat(info)
  value = tf.random_shuffle(value)  # So we see different summaries.
  summary = tf.summary.text('next_production_rule_info' + suffix, value[:10])
  return summary, update_op


def mask_by_partial_sequence_length(
    tensors,
    partial_sequence_lengths=None,
    target_length=None):
  """Selects examples with partial sequence length equal to target_length.

  Args:
    tensors: Tuple of tensors to mask.
    partial_sequence_lengths: Integer tensor with shape [batch_size].
        Default None.
    target_length: Integer. Only examples with partial sequence length equal to
        target_length will be used. If None (the default), all examples in
        batch will be used.

  Returns:
    A tuple of masked tensors.

  Raises:
    ValueError: if partial_sequence_lengths is None when target_length is not
        None.
  """
  if target_length is not None:
    if partial_sequence_lengths is None:
      raise ValueError(
          'partial_sequence_lengths is expected '
          'when target_length is not None.')
    # A mask on batch_size dimension.
    partial_sequence_length_mask = tf.equal(
        partial_sequence_lengths, target_length)
    masked_tensors = []
    for tensor in tensors:
      masked_tensors.append(
          tf.boolean_mask(tensor, partial_sequence_length_mask))
    return tuple(masked_tensors)
  else:
    return tensors


def next_production_rule_valid_ratio(
    unmasked_probabilities_batch,
    next_production_rule_masks,
    partial_sequence_lengths=None,
    target_length=None):
  """Computes the mean valid ratio of next production rule.

  For each production rule prediction, if it is grammarly valid as the next
  production rule of the partial sequence, it is 1. Otherwise, 0. The validness
  of the partial sequence is represented by next_production_rule_masks.

  Args:
    unmasked_probabilities_batch: Float tensor with shape
        [batch_size, num_production_rules]. The probabilities from the model
        prediction without valid production rule mask.
    next_production_rule_masks: Boolean tensor with shape
        [batch_size, num_production_rules]. Mask of the grammarly allowed
        choices of next production rules.
    partial_sequence_lengths: Integer tensor with shape [batch_size].
        Default None.
    target_length: Integer. Only examples with partial sequence length equal to
        target_length will be used. If None (the default), all examples in
        batch will be used.

  Returns:
    value: Float scalar tensor of valid_ratio.
    update_op: Op that updates value.
  """
  unmasked_probabilities_batch, next_production_rule_masks = (
      mask_by_partial_sequence_length(
          tensors=(unmasked_probabilities_batch, next_production_rule_masks),
          partial_sequence_lengths=partial_sequence_lengths,
          target_length=target_length))

  argmax_indices = tf.argmax(unmasked_probabilities_batch, axis=1)
  indices = tf.transpose(tf.stack([
      tf.cast(
          tf.range(tf.shape(unmasked_probabilities_batch)[0]),
          argmax_indices.dtype),
      argmax_indices]))
  is_valid = tf.gather_nd(next_production_rule_masks, indices)
  return tf.metrics.mean(tf.cast(is_valid, tf.float32))


def next_production_rule_accuracy(
    next_production_rules,
    predict_next_production_rules,
    partial_sequence_lengths=None,
    target_length=None):
  """Computes the accuracy of next production rule prediction.

  Args:
    next_production_rules: Integer tensor with shape [batch_size]. The
        indice of the next production rules.
    predict_next_production_rules: Integer tensor with shape [batch_size]. The
        prediction indice of the next production rules.
    partial_sequence_lengths: Integer tensor with shape [batch_size].
        Default None.
    target_length: Integer. Only examples with partial sequence length equal to
        target_length will be used. If None (the default), all examples in
        batch will be used.

  Returns:
    value: Float scalar tensor of accuracy.
    update_op: Op that updates value.
  """
  next_production_rules, predict_next_production_rules = (
      mask_by_partial_sequence_length(
          tensors=(next_production_rules, predict_next_production_rules),
          partial_sequence_lengths=partial_sequence_lengths,
          target_length=target_length))
  return tf.metrics.accuracy(
      labels=next_production_rules,
      predictions=predict_next_production_rules)


def get_leading_powers(leading_powers_abs_sum):
  """Gets leading powers pairs summing to leading_powers_abs_sum.

  Get (leading_at_0, leading_at_inf) pairs that
  abs(leading_at_0) + abs(leading_at_inf) = leading_powers_abs_sum

  For example, (leading_at_0, leading_at_inf) pairs for
  leading_powers_abs_sum = 2:
  (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0), (1, 1)

  Args:
    leading_powers_abs_sum: Positive integer, the sum of absolute value of
        leading powers.

  Yields:
    leading_at_0: Integer, leading power at 0.
    leading_at_inf: Integer, leading power at inf.
  """
  for abs_leading_at_0 in range(leading_powers_abs_sum + 1):
    abs_leading_at_inf = leading_powers_abs_sum - abs_leading_at_0
    pairs = set([
        (abs_leading_at_0, abs_leading_at_inf),
        (-abs_leading_at_0, abs_leading_at_inf),
        (abs_leading_at_0, -abs_leading_at_inf),
        (-abs_leading_at_0, -abs_leading_at_inf),
    ])
    for leading_at_0, leading_at_inf in pairs:
      yield leading_at_0, leading_at_inf


class GenerationWithLeadingPowersHook(tf.train.SessionRunHook):
  """SessionRunHook that generates expressions condition on leading powers.
  """

  def __init__(
      self,
      generation_leading_powers_abs_sums,
      num_expressions_per_condition,
      max_length,
      grammar):
    """Initializer.

    Args:
      generation_leading_powers_abs_sums: List of integers, the sum of the
          absolute values of leading power at 0 and at inf, defining the
          condition in generation.
          For example, if generation_leading_powers_abs_sums = [1, 2],
          expressions will be generated with
          the following conditions (leading_at_0, leading_at_inf):
          (0, 1), (-1, 0), (0, -1), (1, 0)
          (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0), (1, 1)
      num_expressions_per_condition: Integer, the number of expressions to
          generate for each condition.
      max_length: Integer, the max length of production rule sequence.
      grammar: arithmetic_grammar.Grammar object.
    """
    self._generation_leading_powers_abs_sums = (
        generation_leading_powers_abs_sums)
    self._num_expressions_per_condition = num_expressions_per_condition
    self._max_length = max_length
    self._grammar = grammar
    self._symbolic_property_functions = (
        symbolic_properties.get_symbolic_property_functions(symbol='x'))

  def after_create_session(self, session, coord):
    """Resets _finished_generation after session is created.

    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    """
    del session, coord
    self._finished_generation = False

  def before_run(self, run_context):
    """Generates expressions condition on symbolic properties.

    Args:
      run_context: tf.train.SessionRunContext.

    Returns:
      tf.train.SessionRunArgs used to update the pending call to session.run().
    """
    if self._finished_generation:
      feed_match_ratio = []
      feed_fail_ratio = []
    else:
      total_count = 0
      # Count the number of expressions generated with leading powers match the
      # conditions.
      match_count = 0
      # Count the number of sympy fail. Sympy may fail for complicated
      # expressions. That doesn't mean the leading powers do match the
      # conditions.
      # It is just when the expression hits some corner cases in sympy.
      fail_count = 0

      for leading_powers_abs_sum in self._generation_leading_powers_abs_sums:
        for leading_at_0, leading_at_inf in get_leading_powers(
            leading_powers_abs_sum):
          for i in range(self._num_expressions_per_condition):
            logging.info(
                'generate sample %d / %d',
                i, self._num_expressions_per_condition)
            result = partial_sequence_model_generator.generate_expression(
                sess=run_context.session,
                grammar=self._grammar,
                max_length=self._max_length,
                symbolic_properties_dict={
                    'leading_at_0': leading_at_0,
                    'leading_at_inf': leading_at_inf},
                numerical_values=None,
                clip_value_min=None,
                clip_value_max=None,
                random_state=None,
                sampling=True,
                input_variable_scope='features')
            logging.info('generate expression %s', result['expression_string'])
            logging.info('is_terminal %s', result['is_terminal'])

            total_count += 1
            try:
              leading_at_0_value = timeout.RunWithTimeout(
                  self._symbolic_property_functions['leading_at_0'],
                  args=(result['expression_string'],),
                  name='leading_at_0').run(time_limit_seconds=30)
              leading_at_inf_value = timeout.RunWithTimeout(
                  self._symbolic_property_functions['leading_at_inf'],
                  args=(result['expression_string'],),
                  name='leading_at_inf').run(time_limit_seconds=30)
            except (timeout.FunctionTimeoutError, ValueError):
              leading_at_0_value = np.nan
              leading_at_inf_value = np.nan
              fail_count += 1

            if (np.isclose(leading_at_0_value, leading_at_0) and
                np.isclose(leading_at_inf_value, leading_at_inf)):
              match_count += 1
      if total_count == 0:
        match_ratio = 0.
        fail_ratio = 0.
      else:
        match_ratio = float(match_count) / total_count
        fail_ratio = float(fail_count) / total_count

      self._finished_generation = True
      feed_match_ratio = [match_ratio]
      feed_fail_ratio = [fail_ratio]

    return tf.train.SessionRunArgs(
        fetches={
            'match_ratio': 'conditional_generation/match_ratio:0',
            'fail_ratio': 'conditional_generation/fail_ratio:0',
        },
        feed_dict={
            'conditional_generation/match_ratio:0': feed_match_ratio,
            'conditional_generation/fail_ratio:0': feed_fail_ratio,
        })
