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

"""Keras Layers for homology detection from local sequence alignments."""

from typing import Tuple, Type

import gin
import numpy as np
from scipy import optimize
import tensorflow as tf
import tensorflow_datasets as tfds

from dedal import pairs as pairs_lib
from dedal.data import builder
from dedal.models import aligners


@gin.configurable
class UncorrectedLogits(tf.keras.layers.Layer):
  """Computes homology detection logits.

  Logits are computed as:
    logits = b + lambda S.
  """

  def __init__(self,
               bias_init=tf.initializers.Zeros(),
               log_lambda_init=tf.initializers.Constant(0.0),
               **kwargs):
    super().__init__(**kwargs)
    self.b = self.add_weight(
        shape=(), initializer=bias_init, name='homology_bias')
    self.log_l = self.add_weight(
        shape=(), initializer=log_lambda_init, name='homology_log_lambda')

  def call(self, alignments, mask=None):
    """Computes homology detection logits from SW scores and seq lengths.

    Args:
      alignments: a 2-tuple of scores and paths for the batch.
      mask: a single tf.Tensor<float>[batch, 2, len], corresponding to the
        paddings masks for the two sequences.

    Returns:
      A tf.Tensor<float>[batch, 1] with the logits for each example in the
      batch.
    """
    scores = alignments[0]
    logits = self.b + tf.exp(self.log_l) * scores
    return logits[:, tf.newaxis]


@gin.configurable
class LogCorrectedLogits(tf.keras.layers.Layer):
  """Computes homology detection logits with length correction.

  Logits are computed as
    logits = b + lambda S - K log(len1 * len2).
  """

  def __init__(self,
               bias_init=tf.initializers.Zeros(),
               log_lambda_init=tf.initializers.Constant(-1.45),
               log_k_init=tf.initializers.Constant(-3.03),
               **kwargs):
    super().__init__(**kwargs)
    self.b = self.add_weight(
        shape=(), initializer=bias_init, name='homology_bias')
    self.log_l = self.add_weight(
        shape=(), initializer=log_lambda_init, name='homology_log_lambda')
    self.log_k = self.add_weight(
        shape=(), initializer=log_k_init, name='homology_log_k')

  def call(self, alignments, mask=None):
    """Computes homology detection logits from SW scores and seq lengths.

    Args:
      alignments: a 2-tuple of scores and paths for the batch.
      mask: a single tf.Tensor<float>[batch, 2, len], corresponding to the
        paddings masks for the two sequences.

    Returns:
      A tf.Tensor<float>[batch, 1] with the logits for each example in the
      batch.
    """
    scores = alignments[0]
    length_fn = lambda x: tf.reduce_sum(x, axis=1)
    masks = tf.cast(mask, tf.float32)
    mn = tf.cast(length_fn(masks[:, 0]) * length_fn(masks[:, 1]), scores.dtype)
    logits = (self.b + tf.exp(self.log_l) * scores -
              tf.exp(self.log_k) * tf.math.log(mn))
    return logits[:, tf.newaxis]


@gin.configurable
class GumbelCorrectedLogits(tf.keras.layers.Layer):
  """Computes homology detection logits from SW scores and sequence lengths.

  Logits are computed as
    logits = b + log (K_2 lambda_2) - log(K_1 lambda_1)
               + (lambda_1 - lambda_2) * S
               - (len1 * len2) * (K_2 exp(-lambda_2 S) - K_1 exp(-lambda_1 S)).
  """

  def __init__(self,
               bias_init=tf.initializers.Zeros(),
               log_lambda_init1=tf.initializers.Constant(-1.5),
               log_k_init1=tf.initializers.Constant(-3.0),
               log_lambda_init2=tf.initializers.Constant(-1.5),
               log_k_init2=tf.initializers.Constant(-3.0),
               **kwargs):
    super().__init__(**kwargs)
    self.b = self.add_weight(
        shape=(), initializer=bias_init, name='homology_bias')
    self.log_l1 = self.add_weight(
        shape=(), initializer=log_lambda_init1, name='homology_log_lambda1')
    self.log_k1 = self.add_weight(
        shape=(), initializer=log_k_init1, name='homology_log_k1')
    self.log_l2 = self.add_weight(
        shape=(), initializer=log_lambda_init2, name='homology_log_lambda2')
    self.log_k2 = self.add_weight(
        shape=(), initializer=log_k_init2, name='homology_log_k2')

  def call(self, alignments, mask=None):
    """Computes homology detection logits from SW scores and seq lengths.

    Args:
      alignments: a 2-tuple of scores and paths for the batch.
      mask: a single tf.Tensor<float>[batch, 2, len], corresponding to the
        paddings masks for the two sequences.

    Returns:
      A tf.Tensor<float>[batch, 1] with the logits for each example in the
      batch.
    """
    scores = alignments[0]
    masks = tf.cast(mask, tf.float32)
    length_fn = lambda x: tf.math.reduce_sum(x, axis=1)
    bias = self.b + self.log_k2 - self.log_k1 + self.log_l2 - self.log_l1
    lin = (tf.exp(self.log_l1) - tf.exp(self.log_l2)) * scores
    mn = tf.cast(length_fn(masks[:, 0]) * length_fn(masks[:, 1]), scores.dtype)
    exp = mn * (tf.exp(self.log_k2 - tf.exp(self.log_l2) * scores) -
                tf.exp(self.log_k1 - tf.exp(self.log_l1) * scores))
    logits = bias + lin - exp
    return logits[:, tf.newaxis]


@gin.configurable
def finetune_homology_head(loop,
                           head_cls,
                           x0,
                           n_steps = 500,
                           alignment_idx = 0,
                           homology_idx = 1):
  """(Re)-fits homology head using SciPy's minimize method prior to eval."""
  dummy_head = head_cls()
  loss = tf.losses.BinaryCrossentropy(from_logits=True)

  # Adds support for multi-input mode.
  multi_input_idx = None
  if isinstance(loop._dataset_builder, builder.MultiDatasetBuilder):  # pylint: disable=protected-access
    alignment_input_idx = loop.model.switch.alignments[alignment_idx]
    homology_input_idx = loop.model.switch.alignments[homology_idx]
    if alignment_input_idx != homology_input_idx:
      raise ValueError(
          'Alignment and homology output heads must be run on the same input.')
    multi_input_idx = alignment_input_idx

  def set_head_params(head, x):
    for var, value in zip(head.trainable_variables, x):
      var.assign(value)

  def length_fn(x):
    masks = loop.model.encoder.compute_mask(x)
    seq_lens = tf.reduce_sum(tf.cast(masks, tf.int32), 1)
    pos_indices = pairs_lib.consecutive_indices(x)
    seq_lens_pos = tf.gather(seq_lens, pos_indices)
    seq_lens_neg = tf.gather(seq_lens, pairs_lib.roll_indices(pos_indices))
    seq_lens = tf.concat([seq_lens_pos, seq_lens_neg], 0)
    return seq_lens

  @tf.function
  def step_fn(iterator):

    def fwd_fn(x, y_true):
      """Optimizes execution in multi-input mode ignoring unneeded heads."""
      if multi_input_idx is not None:
        selector = loop.model.switch.get_selector(multi_input_idx)
        model_output = loop.model.forward(x, selector=selector, training=False)
      else:
        model_output = loop.model(x, training=False)
      y_true = y_true[f'alignments/{homology_idx}']
      y_pred = model_output.flatten()[f'alignments/{alignment_idx}'][0]
      seq_lens = length_fn(x)
      return y_true, y_pred, seq_lens

    x, y_true, _, _ = next(iterator)
    x = x if multi_input_idx is None else x[multi_input_idx]
    return loop.strategy.run(fwd_fn, args=(x, y_true))

  # Builds a "dataset" of (homology labels, similarity scores, sequence lengths)
  # triplets.
  iterator = iter(loop.make_ds(tfds.Split.TRAIN))
  y_true, y_pred, seq_lens = [], [], []
  for _ in range(n_steps):
    y_true_i, y_pred_i, seq_lens_i = tf.nest.map_structure(
        lambda x: loop.strategy.gather(x, 0), step_fn(iterator))
    y_true.append(y_true_i)
    y_pred.append(y_pred_i)
    seq_lens.append(seq_lens_i)
  y_true = tf.concat(y_true, 0)
  y_pred = tf.squeeze(tf.concat(y_pred, 0))
  seq_lens = tf.concat(seq_lens, 0)

  @tf.function
  def tf_value_and_grad_fn():
    with tf.GradientTape() as tape:
      logits = dummy_head((y_pred,), seq_lens[Ellipsis, tf.newaxis])
      loss_val = loss(y_true, logits)
    grads = tape.gradient(loss_val, dummy_head.trainable_variables)
    return loss_val, tf.stack(grads)

  def value_and_grad_fn(x):
    set_head_params(dummy_head, x)
    return tf.nest.map_structure(lambda t: t.numpy(), tf_value_and_grad_fn())

  res = optimize.minimize(value_and_grad_fn, x0, jac=True)
  set_head_params(loop.model.heads['alignments'][homology_idx], res.x)
