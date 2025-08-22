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

# coding=utf-8
"""Defines the loss functions."""

import math

import tensorflow as tf


def get_weighted_binary_cross_entropy_fn(
    hparams):
  """Returns weighted binary cross entropy function."""
  if hparams.class_imbalance == 1.0:
    # We don't use keras BinaryCrossentropy since it does not preserve dims.
    def loss_fn(labels, logits):
      labels = tf.cast(labels, tf.float32)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
      return loss

    return loss_fn
  else:
    if hparams.loss_weight_method == 'linear':
      w = hparams.class_imbalance
    elif hparams.loss_weight_method == 'sqrt':
      w = math.sqrt(hparams.class_imbalance)
    else:
      raise ValueError(
          f'Wrong loss_weight_method: {hparams.loss_weight_method}')
    # 2.0 is introduced to make the sum of coefs to be 2.0
    pos_weight = 2.0 * w / (1.0 + w)
    neg_weight = 2.0 / (1.0 + w)

    def loss_fn(labels, logits):
      labels = tf.cast(labels, tf.float32)
      weights = labels * pos_weight + (1.0 - labels) * neg_weight
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits) * weights
      return loss

    return loss_fn
