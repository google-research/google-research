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

"""losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf


def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
  print('margin loss')
  logits = raw_logits - 0.5
  positive_cost = labels * tf.cast(tf.less(logits, margin),
                                   tf.float32) * tf.pow(logits - margin, 2)
  negative_cost = (1 - labels) * tf.cast(
      tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
  return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def order_loss(labels, logits, margin=0.2):
  label_act = tf.reduce_sum(labels * logits, axis=-1, keep_dims=True)
  negative_cost = (1 - labels) * tf.cast(
      tf.greater(logits, label_act - margin), tf.float32) * tf.pow(
          logits + margin - label_act, 2)
  return negative_cost


def optimizer(logits, labels, multi, scope, softmax, rate=1.0, step=0.0):
  """Calculate loss and metrics."""
  with tf.name_scope('loss'):
    if softmax:
      diff = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
    else:
      margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 50000.0 - 4))

      print('why softmax is on?!')
      diff = order_loss(labels=labels, logits=logits, margin=margin)
      print('what changed then?!')
      # diff = margin_loss(labels=labels, raw_logits=logits)

    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
      tf.add_to_collection('losses', rate * cross_entropy)
  tf.summary.scalar('batch_cross_entropy', cross_entropy)

  # cross entropy plus all of the regularizers.
  losses = tf.add_n(tf.get_collection('losses', scope), name='total_loss')
  tf.summary.scalar('total_loss', losses)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      _, classes = tf.nn.top_k(labels, k=2 if multi else 1)
      _, preds = tf.nn.top_k(logits, k=2 if multi else 1)
      wrong = tf.sets.size(tf.sets.difference(classes, preds))
      correct_prediction = tf.equal(wrong, 0)
      almost_correct = tf.less(wrong, 2)
      correct_prediction_sum = tf.reduce_sum(
          tf.cast(correct_prediction, tf.float32))
      almost_correct_sum = tf.reduce_sum(tf.cast(almost_correct, tf.float32))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('correct_prediction_batch', correct_prediction_sum)
  tf.summary.scalar('almost_correct_batch', almost_correct_sum)
  return losses, correct_prediction_sum, almost_correct_sum
