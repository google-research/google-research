# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Loss functions from https://openreview.net/forum?id=8twKpG5s8Qh.

Loss functions have the following args (in addition to hyperparameters):
  labels: (num_examples, num_classes) matrix of one-hot labels.
  final_layer: (num_examples, num_classes) matrix of final-layer outputs.
  weights: Scalar or vector of weights for individual examples. In standard
    usage, this argument should be 1 or, when evaluating with batches that,
    may be padded, a vector containing 1s for real examples and 0 for padding.

Each loss function returns:
  loss: The value of the loss.
  outputs: (num_examples, num_classes) matrix of outputs. For loss functions
    that normalize or scale outputs before computing softmax cross-entropy,
    these outputs incorporate that normalization/scaling. For other loss
    functions, these outputs are identical to `final_layer`.
"""

import math
import tensorflow.compat.v1 as tf


def softmax(labels, final_layer, weights):
  return (tf.losses.softmax_cross_entropy(labels, final_layer, weights),
          final_layer)


def label_smoothing(labels, final_layer, weights, alpha):
  return (
      tf.losses.softmax_cross_entropy(
          labels, final_layer, weights * 1/(1 - alpha), label_smoothing=alpha),
      final_layer)


# Penultimate layer dropout implemented in resnet_model.py and uses ordinary
# softmax.
dropout = softmax


def extra_final_layer_l2(labels, final_layer, weights, lambda_,
                         final_layer_weights_variable_name='dense/kernel'):
  xent = tf.losses.softmax_cross_entropy(labels, final_layer, weights)
  final_layer_weights = tf.trainable_variables(
      final_layer_weights_variable_name)[0]
  l2 = lambda_ * tf.nn.l2_loss(final_layer_weights)
  tf.losses.add_loss(l2)
  return xent + l2, final_layer


def logit_penalty(labels, final_layer, weights, beta):
  xent = tf.losses.softmax_cross_entropy(labels, final_layer, weights)
  penalty = tf.losses.compute_weighted_loss(
      beta * tf.reduce_sum(final_layer ** 2, -1) / 2, weights)
  return xent + penalty, final_layer


def logit_normalization(labels, final_layer, weights, tau):
  logits = tf.nn.l2_normalize(final_layer, axis=-1) / tau
  return tf.losses.softmax_cross_entropy(labels, logits, weights), logits


def cosine_softmax(labels, final_layer, weights, tau):
  logits = final_layer / tau
  return tf.losses.softmax_cross_entropy(labels, logits, weights), logits


def sigmoid(labels, final_layer, weights):
  logits = final_layer - math.log(int(final_layer.shape[1]))
  return tf.losses.sigmoid_cross_entropy(
      labels, logits, weights * int(final_layer.shape[1])), logits


def squared_error(labels, final_layer, weights, kappa, m, loss_scale):
  correct_class_loss = tf.squared_difference(
      tf.reduce_sum(final_layer * labels, -1), m)
  other_class_loss = tf.reduce_sum(tf.square(final_layer * (1.0 - labels)), -1)
  return tf.losses.compute_weighted_loss(
      loss_scale * (kappa * correct_class_loss + other_class_loss),
      weights), final_layer
