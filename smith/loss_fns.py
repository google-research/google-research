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

"""Loss functions used in dual encoder SMITH model."""
import tensorflow.compat.v1 as tf


def _pointwise_cosine(encodings_1, encodings_2):
  """Pointwise version of cosine similarity function.

  Args:
    encodings_1: A 2-D tensor of (left) encodings with shape [batch size,
      encoding dim].
    encodings_2: A 2-D tensor of (right) encodings with shape [batch size,
      encoding dim].

  Returns:
    A 1-D tensor of cosine similarities with shape [batch size].
  """
  similarities = tf.reduce_sum(tf.multiply(encodings_1, encodings_2), 1)
  return similarities


def get_prediction_loss_cosine(input_tensor_1,
                               input_tensor_2,
                               labels,
                               similarity_score_amplifier=6.0,
                               neg_to_pos_example_ratio=1.0):
  """Get prediction based on pointwise version of cosine similarity function.

  Compute the model predictions and losses based on cosine similarity functions.
  This setting is useful for the binary classification task or regression task.

  Args:
    input_tensor_1: The Tensor with shape [batch_size, embed_size] to denote the
      left input text.
    input_tensor_2:  The Tensor with shape [batch_size, embed_size] to denote
      the right input text.
    labels: Float tensor with shape [batch_size]. The ground truth labels to
      denote whether two documents are matched.
    similarity_score_amplifier: The amplifier to increase the logits value, so
      that sigmoid(logits) is closer to 0 or 1. The default value is 6.0.
    neg_to_pos_example_ratio: The ratio to compensate when we have more negative
      examples.

  Returns:
    The loss, per example loss and similarities of two input texts.
  """
  with tf.variable_scope("loss/text_pair_matching"):
    logits = _pointwise_cosine(input_tensor_1, input_tensor_2)

    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.float32)
    # To compensate when we have way more neg examples than pos examples and
    # to compensate for larger masked lm loss.
    # Note that we use weights_2 to make sure the weight for neg examples is
    # 1, not 0.
    weights_1 = tf.multiply(labels, neg_to_pos_example_ratio)
    weights_2 = tf.add(tf.ones_like(labels), tf.negative(labels))
    # When neg_to_pos_example_ratio = 1.0, weights will be all ones.
    weights = tf.add(weights_1, weights_2)
    logits *= similarity_score_amplifier
    per_example_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels,
        logits=logits,
        weights=weights,
        reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits)
