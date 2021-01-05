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

r"""Runs partial sequence model.

This is a supervised learning model. This model use the partial sequence of
production rules to predict next production rule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags
import six
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile

from neural_guided_symbolic_regression.models import core
from neural_guided_symbolic_regression.models import grammar_utils
from neural_guided_symbolic_regression.models import input_ops
from neural_guided_symbolic_regression.models import metrics
from neural_guided_symbolic_regression.models import networks
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import training as contrib_training


flags.DEFINE_string('model_dir', None,
                    'The directory where the model will be stored.')
flags.DEFINE_string('hparams', None, 'Filename for serialized HParams.')
flags.DEFINE_bool('is_chief', False,
                  'If True, write hparams and geninfo to model_dir.')


FLAGS = flags.FLAGS


def mask_logits(logits, mask):
  """Masks logits and writes them to predictions dict.

  Args:
    logits: Float tensor with shape [batch_size, num_classes].
    mask: Boolean tensor with shape [batch_size, num_classes].

  Returns:
    Dict of tensors. It contains keys
    'unmasked_probabilities' and 'masked_probabilities'. Both of them are float
    tensor with shape [batch_size, num_classes].
  """
  with tf.variable_scope('mask_logits'):
    unmasked_probabilities = tf.nn.softmax(
        logits, name='unmasked_probabilities')
    unnormalized_masked_probabilities = tf.multiply(
        unmasked_probabilities,
        tf.cast(mask, unmasked_probabilities.dtype))
    masked_probabilities = tf.divide(
        unnormalized_masked_probabilities,
        tf.reduce_sum(unnormalized_masked_probabilities, axis=1, keepdims=True),
        name='masked_probabilities')
    return {
        'unmasked_probabilities': unmasked_probabilities,
        'masked_probabilities': masked_probabilities,
    }


def model_fn(features, labels, mode, params, grammar):
  """Builds the model graph.

  Args:
    features: Dict of tensors.
    labels: Dict of tensors, or None if mode == INFER.
    mode: tf.estimator.ModeKeys execution mode.
    params: HParams object containing model hyperparameters.
    grammar: arithmetic_grammar.Grammar object.

  Returns:
    A ModelFnOps object defining predictions, loss, and train_op.
  """
  if mode != tf.estimator.ModeKeys.PREDICT:
    tf.summary.text('expression_string', features['expression_string'][:10])
  tf.summary.text('production_rules', tf.constant(grammar.grammar_to_string()))

  # Make features easier to look up.
  with tf.variable_scope('features'):
    features = {
        key: tf.identity(value, name=key)
        for key, value in six.iteritems(features)
    }

  embedding_layer = networks.partial_sequence_encoder(
      features=features,
      symbolic_properties=core.hparams_list_value(params.symbolic_properties),
      numerical_points=core.hparams_list_value(params.numerical_points),
      num_production_rules=grammar.num_production_rules,
      embedding_size=params.embedding_size)

  logits = networks.build_stacked_gru_model(
      embedding_layer=embedding_layer,
      partial_sequence_length=features['partial_sequence_length'],
      gru_hidden_sizes=params.gru_hidden_sizes,
      num_output_features=grammar.num_production_rules,
      bidirectional=params.bidirectional)

  predictions = {'logits': tf.identity(logits, name='predictions/logits')}
  predictions.update({
      name: tf.identity(tensor, name='predictions/%s' % name)
      for name, tensor in six.iteritems(
          mask_logits(logits, features['next_production_rule_mask']))
  })
  predictions['next_production_rule'] = tf.argmax(
      predictions['masked_probabilities'],
      axis=1,
      name='predictions/next_production_rule')

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # NOTE(leeley): The mask cannot be applied directly on logits. Because 0
  # logit is still corresponding to a positive probability. Since
  # tf.losses.sparse_softmax_cross_entropy() only works for logits rather than
  # probabilities, I convert probabilities back to logits by tf.log(). Since
  # the probabilities for grammarly invalid production rules are 0, to avoid
  # numerical issue of log(0), I added a small number 1e-10.
  loss = tf.losses.sparse_softmax_cross_entropy(
      labels, tf.log(predictions['masked_probabilities'] + 1e-10))

  # Configure the training op for TRAIN mode.
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = contrib_layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=core.learning_rate_decay(
            initial_learning_rate=params.learning_rate,
            decay_steps=params.learning_rate_decay_steps,
            decay_rate=params.learning_rate_decay_rate),
        optimizer=params.optimizer,
        summaries=contrib_layers.OPTIMIZER_SUMMARIES)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

  # Add evaluation metrics for EVAL mode.
  eval_metric_ops = {
      'eval_loss':
          tf.metrics.mean(loss),
      'count':
          contrib_metrics.count(labels),
      'next_production_rule_valid_ratio':
          metrics.next_production_rule_valid_ratio(
              unmasked_probabilities_batch=predictions[
                  'unmasked_probabilities'],
              next_production_rule_masks=features['next_production_rule_mask']),
      'next_production_rule_accuracy':
          metrics.next_production_rule_accuracy(
              next_production_rules=labels,
              predict_next_production_rules=predictions['next_production_rule']
          ),
  }

  for target_length in range(1, params.max_length + 1):
    eval_metric_ops[
        'next_production_rule_info/length_%d' % target_length
    ] = metrics.next_production_rule_info_batch_text_summary(
        expression_strings=features['expression_string'],
        partial_sequences=features['partial_sequence'],
        partial_sequence_lengths=features['partial_sequence_length'],
        next_production_rules=labels,
        unmasked_probabilities_batch=predictions[
            'unmasked_probabilities'],
        masked_probabilities_batch=predictions['masked_probabilities'],
        grammar=grammar,
        target_length=target_length)

    eval_metric_ops[
        'next_production_rule_valid_ratio/length_%d' % target_length
    ] = metrics.next_production_rule_valid_ratio(
        unmasked_probabilities_batch=predictions[
            'unmasked_probabilities'],
        next_production_rule_masks=features['next_production_rule_mask'],
        partial_sequence_lengths=features['partial_sequence_length'],
        target_length=target_length)

    eval_metric_ops[
        'next_production_rule_accuracy/length_%d' % target_length
    ] = metrics.next_production_rule_accuracy(
        next_production_rules=labels,
        predict_next_production_rules=predictions['next_production_rule'],
        partial_sequence_lengths=features['partial_sequence_length'],
        target_length=target_length)

  if params.num_expressions_per_condition > 0:
    with tf.variable_scope('conditional_generation'):
      match_ratio = tf.placeholder(tf.float32, shape=[None], name='match_ratio')
      fail_ratio = tf.placeholder(tf.float32, shape=[None], name='fail_ratio')

    eval_metric_ops.update({
        'generation_match_ratio': tf.metrics.mean(match_ratio),
        'generation_fail_ratio': tf.metrics.mean(fail_ratio),
    })

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_hparams(**kwargs):
  """Creates a set of default hyperparameters.

  Note that in addition to the hyperparameters described below, the full set of
  hyperparameters includes input_ops.get_hparams() for specifying the input data
  pipeline (see that function for input_ops hyperparameter descriptions).

  Model hyperparameters:
    grammar_path: String, the filename of txt file containing the grammar
        production rules. Expressions will be parsed by this grammar.
    learning_rate: Float, learning rate.
    learning_rate_decay_rate: Float, decay rate for tf.train.exponential_decay.
    learning_rate_decay_step: Integer, decay steps for
        tf.train.exponential_decay.
    optimizer: String, optimizer name. Must be one of
        tf.contrib.layers.OPTIMIZER_CLS_NAMES.
    save_checkpoints_secs: Integer, number of seconds between model checkpoints.
    keep_checkpoint_max: Integer, the maximum number of recent checkpoint files
        to keep. As new files are created, older files are deleted.
        If None or 0, all checkpoint files are kept.
    start_delay_secs: Integer, number of seconds to wait before starting
        evaluations.
    throttle_secs: Integer, number of seconds between evaluations.
    train_steps: Integer, maximum number of training steps. Set to None to train
        forever.
    eval_steps: Integer, number of steps for each evaluation. Set to None to
        evaluate the entire tune/test set.
    embedding_size: Integer, the size of production rule embedding.
    symbolic_properties: List of strings, symbolic properties to concatenate on
        embedding as conditions.
    numerical_points: List of floats, points to evaluate expression values.
    gru_hidden_sizes: List of integers, number of units for each GRU layer.
    bidirectional: Boolean, whether to use bidirectional RNN.
    generation_leading_powers_abs_sums: List of integers, the sum of leading
        power at 0 and at inf, defining the condition in generation.
        For example, if generation_leading_powers_abs_sums = [1, 2],
        expressions will be generated with
        the following conditions (leading_at_0, leading_at_inf):
        (0, 1), (-1, 0), (0, -1), (1, 0)
        (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0), (1, 1)
        This is used for eval.
    num_expressions_per_condition: Integer, the number of expressions to
        generate for each condition. This is used for eval. Default 0, no
        generation in eval.
    exports_to_keep: Integer, the number of latest exported model to keep.

  Args:
    **kwargs: Dict of parameter overrides.

  Returns:
    HParams.
  """
  hparams = contrib_training.HParams(
      grammar_path=None,
      learning_rate=0.01,
      learning_rate_decay_rate=1.0,
      learning_rate_decay_steps=100000,
      optimizer='Adagrad',
      save_checkpoints_secs=600,
      keep_checkpoint_max=20,
      start_delay_secs=300,
      throttle_secs=300,
      train_steps=None,
      eval_steps=None,
      embedding_size=10,
      symbolic_properties=core.HPARAMS_EMPTY_LIST_STRING,
      numerical_points=core.HPARAMS_EMPTY_LIST_FLOAT,
      gru_hidden_sizes=[100],
      bidirectional=False,
      generation_leading_powers_abs_sums=core.HPARAMS_EMPTY_LIST_INT,
      num_expressions_per_condition=0,
      exports_to_keep=50)

  # Add hparams from input_ops.
  # Using add_hparam ensures there are no duplicated parameters.
  for key, value in six.iteritems(input_ops.get_hparams().values()):
    if key in hparams.values():
      continue  # Skip duplicated parameters.
    hparams.add_hparam(key, value)
  return hparams.override_from_dict(kwargs)


def run():
  """Runs train_and_evaluate."""
  hparams_filename = os.path.join(FLAGS.model_dir, 'hparams.json')
  if FLAGS.is_chief:
    gfile.MakeDirs(FLAGS.model_dir)
    hparams = core.read_hparams(FLAGS.hparams, get_hparams())
    core.write_hparams(hparams, hparams_filename)

  # Always load HParams from model_dir.
  hparams = core.wait_for_hparams(hparams_filename, get_hparams())

  grammar = grammar_utils.load_grammar(grammar_path=hparams.grammar_path)

  estimator = tf.estimator.Estimator(
      model_fn=functools.partial(model_fn, grammar=grammar),
      params=hparams,
      config=tf.estimator.RunConfig(
          save_checkpoints_secs=hparams.save_checkpoints_secs,
          keep_checkpoint_max=hparams.keep_checkpoint_max))

  train_spec = tf.estimator.TrainSpec(
      input_fn=functools.partial(
          input_ops.input_fn,
          input_pattern=hparams.train_pattern,
          grammar=grammar),
      max_steps=hparams.train_steps)

  # NOTE(leeley): The SavedModel will be stored under the
  # tf.saved_model.tag_constants.SERVING tag.
  latest_exporter = tf.estimator.LatestExporter(
      name='latest_exported_model',
      serving_input_receiver_fn=functools.partial(
          input_ops.serving_input_receiver_fn,
          params=hparams,
          num_production_rules=grammar.num_production_rules),
      exports_to_keep=hparams.exports_to_keep)

  eval_hooks = []
  if hparams.num_expressions_per_condition > 0:
    eval_hooks.append(
        metrics.GenerationWithLeadingPowersHook(
            generation_leading_powers_abs_sums=core.hparams_list_value(
                hparams.generation_leading_powers_abs_sums),
            num_expressions_per_condition=hparams.num_expressions_per_condition,
            max_length=hparams.max_length,
            grammar=grammar))

  eval_spec = tf.estimator.EvalSpec(
      input_fn=functools.partial(
          input_ops.input_fn,
          input_pattern=hparams.tune_pattern,
          grammar=grammar),
      steps=hparams.eval_steps,
      exporters=latest_exporter,
      start_delay_secs=hparams.start_delay_secs,
      throttle_secs=hparams.throttle_secs,
      hooks=eval_hooks)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(argv):
  del argv  # Unused.
  run()


if __name__ == '__main__':
  app.run(main)
