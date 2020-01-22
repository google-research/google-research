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

r"""TCN loss for unsupervised training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.algos.algorithm import Algorithm
from tcc.config import CONFIG
from tcc.utils import get_cnn_feats
from tcc.utils import set_learning_phase


def _npairs_loss(labels, embeddings_anchor, embeddings_positive, reg_lambda):
  """Returns n-pairs metric loss."""
  reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_anchor), 1))
  reg_positive = tf.reduce_mean(tf.reduce_sum(
      tf.square(embeddings_positive), 1))
  l2loss = 0.25 * reg_lambda * (reg_anchor + reg_positive)

  # Get per pair similarities.
  similarity_matrix = tf.matmul(
      embeddings_anchor, embeddings_positive, transpose_a=False,
      transpose_b=True)

  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = tf.shape(labels)
  assert lshape.shape == 1
  labels = tf.reshape(labels, [lshape[0], 1])

  labels_remapped = tf.cast(
      tf.equal(labels, tf.transpose(labels)), tf.float32)
  labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

  # Add the softmax loss.
  xent_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=similarity_matrix, labels=labels_remapped)
  xent_loss = tf.reduce_mean(xent_loss)

  return l2loss + xent_loss


def single_sequence_loss(embs, num_steps):
  """Returns n-pairs loss for a single sequence."""

  labels = tf.range(num_steps)
  labels = tf.stop_gradient(labels)
  embeddings_anchor = embs[0::2]
  embeddings_positive = embs[1::2]
  loss = _npairs_loss(labels, embeddings_anchor, embeddings_positive,
                      reg_lambda=CONFIG.TCN.REG_LAMBDA)
  return loss


class TCN(Algorithm):
  """Time-contrastive Network."""

  @set_learning_phase
  def call(self, data, steps, seq_lens, training):
    """One pass through the model."""
    cnn = self.model['cnn']
    emb = self.model['emb']

    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES * CONFIG.DATA.NUM_STEPS
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES * CONFIG.DATA.NUM_STEPS

    # Number of steps is doubled due to sampling of positives and anchors.
    cnn_feats = get_cnn_feats(cnn, data, training, 2 * num_steps)

    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES

    embs = emb(cnn_feats, 2 * num_steps)
    embs = tf.stack(tf.split(embs, 2 * num_steps, axis=0), axis=1)

    return embs

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):
    if training:
      num_steps = CONFIG.TRAIN.NUM_FRAMES
      batch_size = CONFIG.TRAIN.BATCH_SIZE
    else:
      num_steps = CONFIG.EVAL.NUM_FRAMES
      batch_size = CONFIG.EVAL.BATCH_SIZE
    losses = []
    for i in xrange(batch_size):
      losses.append(single_sequence_loss(embs[i], num_steps))
    loss = tf.reduce_mean(tf.stack(losses))
    return loss
