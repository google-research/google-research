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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions for CF with support for optional negative sampling."""

import abc
import tensorflow.compat.v2 as tf


class LossFn(abc.ABC):
  """Abstract loss function for CF embeddings."""

  def __init__(self, sizes, neg_sample_size, double_neg, margin):
    """Initialize CF loss function.

    Args:
      sizes: Tuple of size 2 containing (n_users, n_items).
      neg_sample_size: Integer indicating the number of negative samples to use.
      double_neg: Bool indicating whether or not to use double negative
        sampling.
      margin: Float indicating the margin between ascore for positive and
        negative examples.
    """
    self.n_users = sizes[0]
    self.n_items = sizes[1]
    self.neg_sample_size = neg_sample_size
    self.double_neg = double_neg
    self.use_neg_sampling = neg_sample_size > 0
    self.gamma = tf.Variable(
        self.neg_sample_size * tf.keras.backend.ones(1) / self.n_items,
        trainable=False)
    self.margin = tf.Variable(
        margin * tf.keras.backend.ones(1),
        trainable=False)

  @abc.abstractmethod
  def loss_from_logits(self, logits, full_labels, labels):
    """Computes CF loss.

    Args:
      logits: Tensor of size batch_size x n_items containing predictions.
      full_labels: Tensor of size batch_size x n_items containing one-hot
        labels.
      labels: Tensor of size batch_size x 1 containing sparse labels (index of
        correct item).

    Returns:
      Average loss within batch.
    """
    pass

  @abc.abstractmethod
  def get_neg_sample_mask(self, logits, full_labels):
    """Generates negative sampling mask.

    Args:
      logits: Tensor of size batch_size x n_items containing predictions.
      full_labels: Tensor of size batch_size x n_items containing one-hot
        labels.

    Returns:
      neg_sample_mask: Tensor of size batch_size x n_items.
    """
    pass

  @abc.abstractmethod
  def calculate_loss(self, model, input_batch):
    """Computes loss with or without negative sampling.

    Args:
      model: tf.keras.Model CF embedding model.
      input_batch: Tensor of size batch_size x 2 containing input pairs.

    Returns:
      Average loss within the input_batch.
    """
    pass


class ExpLossFn(LossFn):
  """Exponent based losses."""

  def get_neg_sample_mask(self, logits, full_labels):
    """Generates negative sampling mask on logits for exp-based losses.

    Args:
      logits: Tensor of size batch_size x n_items containing predictions.
      full_labels: Tensor of size batch_size x n_items containing one-hot
        labels.

    Returns:
      neg_sample_mask: Tensor of size batch_size x n_items with -1e6 and
                       zeros (-1e6 indicates that the corresonding example
                       is masked).
    """
    neg_sample_mask = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    neg_sample_mask = tf.cast(neg_sample_mask > self.gamma, logits.dtype)
    neg_sample_mask = -1e6 * tf.maximum(neg_sample_mask - full_labels, 0)
    return neg_sample_mask

  def calculate_loss(self, model, input_batch):
    labels = input_batch[:, 1]
    logits = model(input_batch, eval_mode=True)
    full_labels = tf.one_hot(labels, depth=self.n_items, dtype=logits.dtype)
    if self.use_neg_sampling:
      # mask some values for negative sampling
      neg_sample_mask = self.get_neg_sample_mask(logits, full_labels)
      # mask logits to only keep target and negative examples' scores
      logits = logits + neg_sample_mask
    return self.loss_from_logits(logits, full_labels, labels)


class SigmoidCrossEntropy(ExpLossFn):
  """Sigmoid cross entropy loss."""

  def loss_from_logits(self, logits, full_labels, labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(full_labels, logits))


class SoftmaxCrossEntropy(ExpLossFn):
  """Softmax cross entropy loss."""

  def loss_from_logits(self, logits, full_labels, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))


class PairwiseHingeFn(LossFn):
  """Pairwise ranking hinge loss."""

  def get_neg_sample_mask(self, logits, full_labels):
    """Generates negative sampling mask.

    Args:
      logits: Tensor of size batch_size x n_items containing predictions.
      full_labels: Tensor of size batch_size x n_items containing one-hot
        labels.

    Returns:
      neg_sample_mask: Tensor of size batch_size x n_items with ones and
                       zeros (zero indicates that the corresonding example
                       is masked).
    """
    neg_sample_mask = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    neg_sample_mask = tf.cast(neg_sample_mask < self.gamma, logits.dtype)
    neg_sample_mask = tf.maximum(neg_sample_mask, full_labels)
    return neg_sample_mask

  def loss_from_logits(self, logits, full_labels, labels):
    signed_logits = (1.0 - 2.0 * full_labels) * logits
    return tf.reduce_mean(
        tf.nn.relu(
            self.margin + tf.reduce_sum(signed_logits, 1)))

  def calculate_loss(self, model, input_batch):
    labels = input_batch[:, 1]
    logits = model(input_batch, eval_mode=True)
    full_labels = tf.one_hot(labels, depth=self.n_items, dtype=logits.dtype)
    if self.use_neg_sampling:
      # mask some values for negative sampling
      neg_sample_mask = self.get_neg_sample_mask(logits, full_labels)
      # mask logits to only keep target and negative examples' scores
      logits = logits * neg_sample_mask
    return self.loss_from_logits(logits, full_labels, labels)
