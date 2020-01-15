# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Defines helper methods for dealing with SQuAD data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
import tensorflow.compat.v1 as tf

from qanet.util import evaluator_util
from qanet.util import tokenizer_util

__all__ = ['get_answer_op', 'get_eval_metric_ops']


def get_answer_op(context, context_words, answer_start, answer_end, has_answer):
  return tf.py_func(
      _enum_fn(partial(
          tokenizer_util.get_answer,
          is_byte=True,
      )), [context, context_words, answer_start, answer_end, has_answer],
      'string')


def get_eval_metric_ops(targets, predictions):
  """Get a dictionary of eval metrics for `Experiment` object.

  Args:
    targets: `targets` that go into `model_fn` of `Experiment`.
    predictions: Dictionary of predictions, output of `get_preds`.
  Returns:
    A dictionary of eval metrics.
  """
  # TODO(seominjoon): yp should also consider no answer case.
  yp1 = tf.expand_dims(predictions['yp1'], -1)
  yp2 = tf.expand_dims(predictions['yp2'], -1)
  answer_mask = tf.sequence_mask(targets['num_answers'])
  start_correct = tf.reduce_any(
      tf.equal(targets['word_answer_starts'], yp1) & answer_mask, 1)
  end_correct = tf.reduce_any(
      tf.equal(targets['word_answer_ends'], yp2) & answer_mask, 1)
  correct = start_correct & end_correct

  em = tf.py_func(
      _enum_fn(partial(evaluator_util.compute_exact,
                       is_byte=True), dtype='float32'),
      [predictions['a'], targets['answers'], answer_mask],
      'float32')
  f1 = tf.py_func(
      _enum_fn(partial(evaluator_util.compute_f1,
                       is_byte=True), dtype='float32'),
      [predictions['a'], targets['answers'], answer_mask],
      'float32')

  eval_metric_ops = {
      'acc1': tf.metrics.mean(tf.cast(start_correct, 'float')),
      'acc2': tf.metrics.mean(tf.cast(end_correct, 'float')),
      'acc': tf.metrics.mean(tf.cast(correct, 'float')),
      'em': tf.metrics.mean(em),
      'f1': tf.metrics.mean(f1),
  }

  if 'plausible' in targets:  # SQuAD2
    # targets['plausible'] has values 1.0 for plausible (no-answer) examples
    #   and 0.0 otherwise.
    eval_metric_ops['f1_no_answer'] = tf.metrics.mean(
        f1 * targets['plausible'])
    eval_metric_ops['em_no_answer'] = tf.metrics.mean(
        em * targets['plausible'])

  return eval_metric_ops


def _enum_fn(fn, dtype='object'):

  def new_fn(*args):
    return np.array([fn(*each_args) for each_args in zip(*args)], dtype=dtype)

  return new_fn
