# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""CNN model for learning packaging viability phenotype from sequence."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import estimator as tf_estimator


def cnn_model_fn(features, labels, mode, params, refs=None):
  """Adds tf ops for a 1-d CNN classifier model.

  Note: implements the tf.estimator.Estimator model_fn API.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    labels: (Tensor) The labels tensor provided by the input_fn.
    mode: (tf.estimator.ModeKeys) The invocation mode of the model.
    params: (dict) Model configuration parameters.
    refs: (dict) A dict to be populated with references to tensors that have
      been added to the graph by the model_fn.
  Returns:
    A tf.estimator.EstimatorSpec instance describing the specified mode.
  """
  if refs is None:
    refs = {}

  # Support both dict-based and HParams-based params.
  if not isinstance(params, dict):
    params = params.values()

  logits_train = build_nn_inference_subgraph(
      features,
      reuse=False,
      is_training=True,
      params=params,
      refs=refs)
  logits_test = build_nn_inference_subgraph(
      features,
      reuse=True,
      is_training=False,
      params=params,
      refs=refs)

  pred_classes = tf.argmax(logits_test, axis=1)
  pred_probas = tf.nn.softmax(logits_test)

  if mode == tf_estimator.ModeKeys.PREDICT:
    return tf_estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'label': pred_classes,
            'proba': pred_probas,
        },
    )

  one_hot_labels = tf.one_hot(labels, params['num_classes'])
  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits_train, labels=one_hot_labels))
  tf.summary.scalar('loss', loss_op)
  optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

  # To enable learning of the moving mean/variance for batch norm,
  # need to add the following control deps on the UPDATE_OPS collection; see:
  # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(
        loss_op, global_step=tf.train.get_global_step())

  acc_op = tf.metrics.accuracy(
      labels=labels,
      predictions=pred_classes)
  precision_op = tf.metrics.precision(
      labels=labels,
      predictions=pred_classes)
  recall_op = tf.metrics.recall(
      labels=labels,
      predictions=pred_classes)

  return tf_estimator.EstimatorSpec(
      mode=mode,
      train_op=train_op,
      loss=loss_op,
      eval_metric_ops={
          'accuracy': acc_op,
          'precision': precision_op,
          'recall': recall_op,
      },
  )


def build_nn_inference_subgraph(features, reuse, is_training, params, refs):
  """Builds the inference subgraph for the neural network.

  Args:
    features: ({str: Tensor}) The feature tensors provided by the input_fn.
    reuse: (bool) Should the variables declared be reused?
    is_training: (bool) Should the subgraph include ops used during training?
    params: (dict) Model configuration parameters.
    refs: (dict) A dict to be populated with references to tensors that have
      been added to the graph.
  Returns:
    A reference to the logits tensor for the inference subgraph
  """
  # Define a scope for reusing the variables
  with tf.variable_scope('inference', reuse=reuse):
    # Convert the fully flattened examples into [sequence, depth]
    # feature vectors (1-d with multiple features at each sequence position).
    features_flat = tf.reshape(features['sequence'], [-1])
    features_seq = tf.reshape(features_flat, shape=[
        -1, params['seq_encoding_length'], params['residue_encoding_size']])
    if is_training:
      refs['features_seq'] = features_seq

    conv1 = tf.layers.conv1d(
        features_seq,
        filters=params['conv_depth'],
        kernel_size=params['conv_width'],
        padding='same',
        activation=tf.nn.relu)
    conv1_bn = tf.layers.batch_normalization(
        conv1,
        axis=params['feature_axis'],
        training=is_training)

    if is_training:
      refs['conv1_bn'] = conv1_bn

    pool1 = tf.layers.max_pooling1d(
        conv1_bn,
        pool_size=params['pool_width'],
        strides=params['pool_width'],
        padding='valid')
    if is_training:
      refs['pool1'] = pool1

    conv2 = tf.layers.conv1d(
        pool1,
        filters=params['conv_depth'] * params['conv_depth_multiplier'],
        kernel_size=params['conv_width'],
        padding='same',
        activation=tf.nn.relu)
    conv2_bn = tf.layers.batch_normalization(
        conv2,
        axis=params['feature_axis'],
        training=is_training)
    if is_training:
      refs['conv2_bn'] = conv2_bn

    pool2 = tf.layers.max_pooling1d(
        conv2_bn,
        pool_size=params['pool_width'],
        strides=params['pool_width'],
        padding='valid')
    if is_training:
      refs['pool2'] = pool2

    post_conv_flat = tf.contrib.layers.flatten(pool2)

    fc1 = tf.layers.dense(
        post_conv_flat,
        units=params['fc_size'],
        activation=tf.nn.relu)
    fc1_bn = tf.layers.batch_normalization(
        fc1,
        axis=params['feature_axis'],
        training=is_training)
    if is_training:
      refs['fc1_bn'] = fc1_bn

    fc2 = tf.layers.dense(
        fc1_bn,
        units=int(params['fc_size'] * params['fc_size_multiplier']),
        activation=tf.nn.relu)
    fc2_bn = tf.layers.batch_normalization(
        fc2,
        axis=params['feature_axis'],
        training=is_training)
    if is_training:
      refs['fc2_bn'] = fc2_bn

    logits = tf.layers.dense(
        fc2_bn,
        units=params['num_classes'],
        activation=None)
    if is_training:
      refs['logits'] = logits

  return logits
