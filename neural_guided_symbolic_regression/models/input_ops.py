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

"""Helper functions for model input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile

from neural_guided_symbolic_regression.models import core
from neural_guided_symbolic_regression.utils import evaluators
from neural_guided_symbolic_regression.utils import postprocessor
from tensorflow.contrib import training as contrib_training


def get_hparams(**kwargs):
  """Creates a set of default hyperparameters.

  Model hyperparameters:
    train_pattern: String, input pattern for training set (glob).
    tune_pattern: String, input pattern for tune set (glob).
    test_pattern: String, input pattern for test set (glob).
    symbol: String, the symbol of variable in univariate expression.
    symbolic_properties: List of strings, symbolic properties to concatenate on
        embedding as conditions.
    numerical_points: List of floats, points to evaluate expression values.
    clip_value_min: Float, the minimum value to clip by. Used only when
        numerical_points is not empty.
    clip_value_max: Float, the maximum value to clip by. Used only when
        numerical_points is not empty.
    batch_size: Integer, batch size.
    max_length: Integer, the max length of production rule sequence.
    label_key: String, the key in features to be used as label.
    reset_batch: Boolean, whether to reset batch size to batch_size in case some
        data are removed during preprocessing.
    cache_dataset: Boolean, whether to cache dataset after preprocessing.
    num_parallel_calls: Integer, the number of elements to process in paralled
        in dataset.map(). If not specified, elements will be processed
        sequentially.
    shuffle_buffer_size: Integer, number of examples in shuffle buffer.
    prefetch_buffer_size: Integer, number of examples to preprocess.

  Args:
    **kwargs: Dict of parameter overrides.

  Returns:
    HParams.
  """
  hparams = contrib_training.HParams(
      train_pattern=None,
      tune_pattern=None,
      test_pattern=None,
      symbol='x',
      symbolic_properties=core.HPARAMS_EMPTY_LIST_STRING,
      numerical_points=core.HPARAMS_EMPTY_LIST_FLOAT,
      clip_value_min=-10.,
      clip_value_max=10.,
      batch_size=1,
      max_length=25,
      label_key='next_production_rule',
      cache_dataset=True,
      num_parallel_calls=None,
      shuffle_buffer_size=1000,
      prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
  )
  return hparams.override_from_dict(kwargs)


def parse_example_batch(
    examples,
    symbolic_properties):
  """Parses a batch of tf.Examples.

  Args:
    examples: String tensor with shape [batch_size]. A batch of serialized
        tf.Example protos.
    symbolic_properties: List of strings, symbolic properties to concatenate on
        embedding as conditions. Those symbolic_properties will be read from
        input data.

  Returns:
    A feature dict. It contains key 'expression_string' with a string tensor of
    expressions with shape [batch_size]. It also contain keys in
    symbolic_properties with float tensors.
  """
  features_to_extract = {
      'expression_string': tf.FixedLenFeature([], tf.string),
  }
  for symbolic_property in symbolic_properties:
    features_to_extract[symbolic_property] = tf.FixedLenFeature([], tf.float32)

  features = tf.parse_example(examples, features=features_to_extract)

  return features


def evaluate_expression_numerically_batch(
    features, numerical_points, clip_value_min, clip_value_max, symbol='x'):
  """Evaluates expressions numerically at certain points.

  Args:
    features: Dict of tensors. This dict need to have key 'expression_string',
        the corresponding value is a string tensor with shape [batch_size].
    numerical_points: Float numpy array with shape [num_numerical_points]. The
        points to evaluate expression values.
    clip_value_min: Float, the minimum value to clip by.
    clip_value_max: Float, the maximum value to clip by.
    symbol: String. Symbol of variable in expression.

  Returns:
    A feature dict. Key 'numerical_values' are added to the dict. It is a
    float32 tensor with shape [batch_size, num_numerical_points]
  """
  expression_strings = features['expression_string']
  numerical_values = tf.py_func(
      functools.partial(
          evaluators.evaluate_expression_strings_1d_grid,
          num_samples=len(numerical_points),
          num_grids=1,
          arguments={
              symbol: numerical_points.reshape((-1, 1))}),
      [expression_strings],
      tf.float32,
      name='py_func-evaluate_expression_numerically_batch')
  numerical_values.set_shape(
      [expression_strings.shape[0], len(numerical_points), 1])

  features['numerical_values'] = tf.divide(
      tf.clip_by_value(
          tf.squeeze(numerical_values, axis=2),
          clip_value_min=clip_value_min,
          clip_value_max=clip_value_max),
      clip_value_max - clip_value_min)

  return features


def parse_production_rule_sequence_batch(features, max_length, grammar):
  """Parses a batch of expressions to sequences of production rules.

  Args:
    features: Dict of tensors. This dict need to have key 'expression_string',
        the corresponding value is a string tensor with shape [batch_size].
    max_length: Integer. The maximum length of the production rule sequence.
    grammar: arithmetic_grammar.Grammar.

  Returns:
    A feature dict. Key 'expression_sequence', 'expression_sequence_mask' are
    added to the dict.
    * 'expression_sequence': an int32 tensor with shape
          [batch_size, max_length].
    * 'expression_sequence_mask': a boolean tensor with shape
          [batch_size, max_length].
  """
  def _parse_expressions_to_indices_sequences(expression_strings):
    return grammar.parse_expressions_to_indices_sequences(
        expression_strings=[
            expression_string.decode('utf-8')
            for expression_string in expression_strings],
        max_length=max_length)

  production_rule_sequences = tf.py_func(
      _parse_expressions_to_indices_sequences,
      [features['expression_string']],
      tf.int32,
      name='py_func-parse_production_rule_sequence_batch')
  production_rule_sequences.set_shape(
      (features['expression_string'].shape[0], max_length))
  features['expression_sequence'] = production_rule_sequences
  features['expression_sequence_mask'] = tf.not_equal(
      production_rule_sequences, grammar.padding_rule_index)
  return features


def sample_partial_sequence(expression_sequence_and_mask, constant_values=0):
  """Samples partial sequence from expression sequence.

  A partial sequence of expression sequence is a sequence of production rules
  from the first production rule to an arbitrary production rule in the
  expression sequence.

  For example, for expression_sequence [2, 1, 3, 0, 0] and
  expression_sequence_mask [True, True, True, False, False],
  The partial sequence can be [2] or [2, 1] before padding.

  Args:
    expression_sequence_and_mask:
        Tuple (expression_sequence, expression_sequence_mask).
        * expression_sequence: Production rule sequence tensor for one
              expression with shape [max_length].
        * expression_sequence_mask: Mask tensor of production rule sequence
              where the padding sequence is False.
    constant_values: Integer. The value to pad at the end of partial sequence
        to the same length of expression sequence.

  Returns:
    partial_sequence: Integer tensor with shape [max_length]. Partial sequence
        of the expression sequence.
    partial_sequence_mask: Boolean tensor with shape [max_length]. Mask out the
        padding.
    partial_sequence_length: Integer scalar tensor with shape. The length
        of partial sequence.
    next_production_rule: Integer scalar tensor. The index of the next
        production rule of the partial sequence.
  """
  expression_sequence, expression_sequence_mask = expression_sequence_and_mask

  maxval = tf.reduce_sum(tf.cast(expression_sequence_mask, tf.int32))
  partial_sequence_length = tf.random_uniform(
      [], minval=1, maxval=maxval, dtype=tf.int32)
  padding_size = tf.shape(expression_sequence)[0] - partial_sequence_length

  partial_sequence = tf.pad(
      expression_sequence[:partial_sequence_length],
      [[0, padding_size]],  # padding at the end.
      mode='CONSTANT',
      constant_values=constant_values)
  partial_sequence.set_shape([expression_sequence.shape[0]])

  partial_sequence_mask = tf.cast(
      tf.concat(
          [tf.ones(partial_sequence_length), tf.zeros(padding_size)], axis=0),
      tf.bool)
  partial_sequence_mask.set_shape([expression_sequence.shape[0]])

  next_production_rule = expression_sequence[partial_sequence_length]
  return (
      partial_sequence,
      partial_sequence_mask,
      partial_sequence_length,
      next_production_rule)


def sample_partial_sequence_batch(features, constant_values=0):
  """Samples partial sequences from a batch of expression sequences.

  Args:
    features: Dict of tensors. This dict need to have:
        * 'expression_sequence': an int32 tensor with shape
              [batch_size, max_length].
        * 'expression_sequence_mask': an boolean tensor with shape
              [batch_size, max_length].
    constant_values: Integer. The value to pad at the end of partial sequence
        to the same length of expression sequence.

  Returns:
    A feature dict. The following keys are added to the dict.
    * 'partial_sequence': an int32 tensor with shape [batch_size, max_length].
    * 'partial_sequence_mask': a boolean tensor with shape
          [batch_size, max_length].
    * 'partial_sequence_length': an int32 tensor with shape [batch_size].
    * 'next_production_rule': an int32 tensor with shape [batch_size].
  """
  (partial_sequences,
   partial_sequence_masks,
   partial_sequence_lengths,
   next_production_rules
  ) = tf.map_fn(
      functools.partial(
          sample_partial_sequence, constant_values=constant_values),
      (features['expression_sequence'], features['expression_sequence_mask']),
      dtype=(tf.int32, tf.bool, tf.int32, tf.int32))
  features['partial_sequence'] = partial_sequences
  features['partial_sequence_mask'] = partial_sequence_masks
  features['partial_sequence_length'] = partial_sequence_lengths
  features['next_production_rule'] = next_production_rules
  return features


def _get_next_production_rule_mask_batch(
    partial_sequences, partial_sequence_lengths, grammar):
  """Gets masks of next production rule for a batch of partial sequences.

  Args:
    partial_sequences: Integer numpy array with shape [batch_size, max_length].
        Batch of partial sequences of the expression sequences.
    partial_sequence_lengths: Integer numpy array with shape [batch_size]. The
        actual length of partial sequences without padding.
    grammar: arithmetic_grammar.Grammar.

  Returns:
    Boolean numpy array with shape [batch_size, num_production_rules].
    num_production_rules is the number of production rules in grammar.
  """
  next_production_rule_masks = np.zeros(
      (len(partial_sequences), grammar.num_production_rules), dtype=bool)
  for i, (partial_sequence, partial_sequence_length) in enumerate(
      zip(partial_sequences, partial_sequence_lengths)):
    stack = postprocessor.production_rules_sequence_to_stack(
        [grammar.prod_rules[index]
         for index in partial_sequence[:partial_sequence_length]])
    next_production_rule_masks[i] = grammar.masks[
        grammar.lhs_to_index[stack.pop()]]
  return next_production_rule_masks


def get_next_production_rule_mask_batch(features, grammar):
  """Gets masks of next production rule for a batch of partial sequences.

  Args:
    features: Dict of tensors. This dict need to have:
        * 'partial_sequence': an int32 tensor with shape
              [batch_size, max_length].
        * 'partial_sequence_length': an int32 tensor with shape [batch_size].
    grammar: arithmetic_grammar.Grammar.

  Returns:
    A feature dict. The following key is added to the dict.
    * 'next_production_rule_mask': a boolean tensor with shape
          [batch_size, num_production_rules]. num_production_rules is the
          number of production rules in grammar.
  """
  partial_sequences = features['partial_sequence']

  next_production_rule_masks = tf.py_func(
      functools.partial(
          _get_next_production_rule_mask_batch, grammar=grammar),
      [partial_sequences, features['partial_sequence_length']],
      tf.bool,
      name='py_func-get_next_production_rule_mask_batch')
  next_production_rule_masks.set_shape(
      [partial_sequences.shape[0], grammar.num_production_rules])

  features['next_production_rule_mask'] = next_production_rule_masks
  return features


def split_features_labels(features, label_key):
  """Splits labels from features.

  Args:
    features: Dict of tensors. This dict need to have label_key.
    label_key: String. The key of label in features dict.

  Returns:
    features: Dict of tensors without label_key.
    labels: features[label_key] tensor.
  """
  labels = features.pop(label_key)
  return features, labels


def parse_examples_fn(examples, params, grammar):
  """Parses examples.

  This function will be used in dataset.map(). It creates features from the
  expression string which will be used for later processing. The steps in this
  function should be deterministic to allow dataset.cache() for speeding up.

  Args:
    examples: String tensor with shape [batch_size]. A batch of serialized
        tf.Example protos.
    params: HParams object containing model hyperparameters.
    grammar: arithmetic_grammar.Grammar.

  Returns:
    A feature dict with items:
    * 'expression_string': a string tensor with shape [batch_size].
    * keys in params.symbolic_properties: Each is a float32 tensor with shape
          [batch_size].
    * 'numerical_values': if numerical_values is not empty, a float32 tensor
          with shape [batch_size, num_numerical_points].
    * 'expression_sequence': an int32 tensor with shape
          [batch_size, max_length].
    * 'expression_sequence_mask': a boolean tensor with shape
          [batch_size, max_length].
  """
  features = parse_example_batch(
      examples=examples,
      symbolic_properties=core.hparams_list_value(params.symbolic_properties))

  numerical_points = core.hparams_list_value(params.numerical_points)
  if numerical_points:
    features = evaluate_expression_numerically_batch(
        features,
        numerical_points=np.asarray(numerical_points),
        clip_value_min=params.clip_value_min,
        clip_value_max=params.clip_value_max,
        symbol=params.symbol)

  features = parse_production_rule_sequence_batch(
      features,
      max_length=params.max_length,
      grammar=grammar)

  return features


def process_dataset_fn(features, params, grammar):
  """Processes dataset.

  This function will be used in dataset.map(). It processes the features in
  dataset. Steps generate different values in each batch (for example, sampling)
  should be include in this function.

  Finally, labels will be splited from features if params.label_key is not None.

  Args:
    features: Dict of tensors. This dict need to have:
        * 'expression_sequence': an int32 tensor with shape
              [batch_size, max_length].
        * 'expression_sequence_mask': an boolean tensor with shape
              [batch_size, max_length].
        * label_key: If params.label_key is not None.
    params: HParams object containing model hyperparameters.
    grammar: arithmetic_grammar.Grammar.

  Returns:
    features: Dict of tensors.
    labels: features[label_key] tensor if params.label_key is not None.
  """
  features = sample_partial_sequence_batch(features, constant_values=0)
  features = get_next_production_rule_mask_batch(features, grammar)
  features, labels = split_features_labels(features, label_key=params.label_key)
  return features, labels


def input_fn(input_pattern, mode, params, grammar):
  """Creates input features and labels tensor dicts.

  Args:
    input_pattern: String, input path.
    mode: tf.estimator.ModeKeys execution mode.
    params: HParams object containing model hyperparameters.
    grammar: arithmetic_grammar.Grammar.

  Returns:
    features: Dict containing input tensors.
    labels: label tensor.
  """
  if mode == tf.estimator.ModeKeys.TRAIN:
    randomize = True
    num_epochs = None
  else:
    randomize = False
    num_epochs = 1

  filenames = gfile.Glob(input_pattern)
  num_files = len(filenames)
  filename_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(filenames))
  if randomize:
    filename_dataset = filename_dataset.shuffle(num_files)
  dataset = filename_dataset.interleave(
      tf.data.TFRecordDataset,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      cycle_length=num_files,
      block_length=1)
  if randomize:
    dataset = dataset.shuffle(params.shuffle_buffer_size or
                              1000 * params.batch_size)

  dataset = dataset.batch(params.batch_size)

  dataset = dataset.map(
      functools.partial(parse_examples_fn, params=params, grammar=grammar),
      num_parallel_calls=params.num_parallel_calls)

  if params.cache_dataset:
    # Cache the expensive read and parsing from file system.
    dataset = dataset.cache()

  dataset = dataset.map(
      functools.partial(process_dataset_fn, params=params, grammar=grammar),
      num_parallel_calls=params.num_parallel_calls)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.prefetch(
      params.prefetch_buffer_size or 1000 * params.batch_size)

  features, labels = dataset.make_one_shot_iterator().get_next()
  return features, labels


def serving_input_receiver_fn(params, num_production_rules):
  """An input receiver for serving trained partial sequence model.

  Args:
    params: HParams object containing model hyperparameters.
    num_production_rules: Integer, number of production rules defined in
        context-free grammar to predict by model.

  Returns:
    Returns of tf.estimator.export.ServingInputReceiver.
  """
  with tf.variable_scope('serving_input'):
    features = {
        'partial_sequence': tf.placeholder(
            dtype=tf.int32,
            shape=[None, params.max_length],
            name='partial_sequence'),
        'partial_sequence_length': tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='partial_sequence_length'),
        'next_production_rule_mask': tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_production_rules],
            name='next_production_rule_mask'),
    }

    symbolic_properties = core.hparams_list_value(params.symbolic_properties)
    for symbolic_property in symbolic_properties:
      features[symbolic_property] = tf.placeholder(
          dtype=tf.float32, shape=[None], name=symbolic_property)

    numerical_points = core.hparams_list_value(params.numerical_points)
    if numerical_points:
      features['numerical_values'] = tf.placeholder(
          dtype=tf.float32,
          shape=[None, len(numerical_points)],
          name='numerical_values')

  return tf.estimator.export.ServingInputReceiver(
      features=features, receiver_tensors=features)
