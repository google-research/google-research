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

"""Implementations of different metrics."""

import tensorflow.compat.v1 as tf


def masked_softmax_cross_entropy(preds, labels, mask):
  """Softmax cross-entropy loss with masking."""
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
  mask = tf.cast(mask, dtype=tf.float32)
  mask /= tf.reduce_mean(mask)
  loss *= mask
  return tf.reduce_mean(loss)


def masked_sigmoid_cross_entropy(preds, labels, mask):
  """Sigmoid cross-entropy loss with masking."""
  loss_all = tf.nn.sigmoid_cross_entropy_with_logits(
      logits=preds, labels=labels)
  mask = tf.cast(mask, dtype=tf.float32)
  mask /= tf.reduce_mean(mask)
  loss = tf.multiply(loss_all, mask[:, tf.newaxis])
  return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
  """Accuracy with masking."""
  correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
  accuracy_all = tf.cast(correct_prediction, tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  mask /= tf.reduce_mean(mask)
  accuracy_all *= mask
  return tf.reduce_mean(accuracy_all)


def masked_accuracy_multilabel(preds, labels, mask):
  """Multilabel accuracy with masking."""
  preds = preds > 0
  labels = labels > 0.5
  correct_prediction = tf.equal(preds, labels)
  accuracy_all = tf.cast(correct_prediction, tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  mask /= tf.reduce_mean(mask)
  accuracy_all = tf.multiply(accuracy_all, mask[:, tf.newaxis])
  return tf.reduce_mean(accuracy_all)
