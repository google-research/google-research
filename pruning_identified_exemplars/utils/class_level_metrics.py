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

"""Class level test-set accuracy for all ImageNet classes."""
from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def create_eval_metrics(labels, logits, human_labels, params):
  """Creates the evaluation metrics for the model."""

  eval_metrics = {}
  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
  eval_metrics['top_5_eval_accuracy'] = tf.metrics.mean(in_top_5)
  eval_metrics['eval_accuracy'] = tf.metrics.accuracy(
      labels=labels, predictions=predictions)
  num_label_classes = params['num_label_classes']
  log_class_level_summaries = params['log_class_level_summaries']

  if log_class_level_summaries:
    labels = tf.cast(labels, tf.int64)

    with tf.name_scope('class_level_summaries') as scope:

      for i in range(num_label_classes):
        name = scope + '/{}_{}'.format(human_labels[i], i)
        eval_metrics['precision/{}_{}'.format(human_labels[i],
                                              i)] = tf.metrics.precision_at_k(
                                                  labels=labels,
                                                  predictions=logits,
                                                  class_id=i,
                                                  k=1,
                                                  name=name)
        eval_metrics['recall/{}_{}'.format(human_labels[i],
                                           i)] = tf.metrics.recall_at_k(
                                               labels=labels,
                                               predictions=logits,
                                               class_id=i,
                                               k=1,
                                               name=name)

  return eval_metrics
