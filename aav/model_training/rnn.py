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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""RNN model for learning the packaging phenotype from mutant sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_model_fn(features, labels, mode, params):
  """RNN tf.estimator.Estimator model_fn definition.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    labels: (Tensor) The labels tensor provided by the input_fn.
    mode: (tf.estimator.ModeKeys) The invocation mode of the model.
    params: (dict) Model configuration parameters.
  Returns:
    (tf.estimator.EstimatorSpec) Model specification.
  """
  # Support both dict-based and HParams-based params.
  if not isinstance(params, dict):
    params = params.values()

  logits_train = build_rnn_inference_subgraph(
      features, reuse=False, params=params)
  logits_test = build_rnn_inference_subgraph(
      features, reuse=True, params=params)

  pred_labels = tf.argmax(logits_test, axis=1)
  pred_probas = tf.nn.softmax(logits_test)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'label': pred_labels,
            'proba': pred_probas,
        },
    )

  # Note: labels=None when mode==PREDICT (see tf.estimator API).
  one_hot_labels = tf.one_hot(labels, params['num_classes'])

  if mode == tf.estimator.ModeKeys.TRAIN:
    loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_train, labels=one_hot_labels))
    tf.summary.scalar('loss_train', loss_train)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss_train, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss_train,
    )

  accuracy = tf.metrics.accuracy(
      labels=labels,
      predictions=pred_labels)
  precision = tf.metrics.precision(
      labels=labels,
      predictions=pred_labels)
  recall = tf.metrics.recall(
      labels=labels,
      predictions=pred_labels)
  loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits_test, labels=one_hot_labels))
  tf.summary.scalar('loss_test', loss_test)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss_test,
      eval_metric_ops={
          'accuracy': accuracy,
          'precision': precision,
          'recall': recall,
      }
  )


def build_rnn_inference_subgraph(features, reuse, params):
  """Builds the inference subgraph for the RNN.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    reuse: (bool) Should the variables declared be reused?
    params: (dict) Model configuration parameters.
  Returns:
    (Tensor) A reference to the logits tensor for the inference subgraph.
  """
  with tf.variable_scope('inference', reuse=reuse):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(
            num_units=params['num_units'],
            state_is_tuple=True) for _ in range(params['num_layers'])
    ])

    initial_state = cell.zero_state(params['batch_size'], tf.float32)

    outputs, unused_state = tf.nn.dynamic_rnn(
        cell,
        features['sequence'],
        sequence_length=features['sequence_length'],
        initial_state=initial_state,
        dtype=tf.float32)
    output = tf.reduce_mean(outputs, axis=1)
    logits = tf.layers.dense(
        output, params['num_classes'], activation=None, use_bias=True)

    return logits
