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

r"""Shuffle and Learn loss for unsupervised training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.algos.algorithm import Algorithm
from tcc.config import CONFIG
from tcc.models import Classifier
from tcc.utils import random_choice_noreplace


def randomly_reverse_indices(indices):
  """Randomly reverse the indices."""
  return tf.cond(tf.random.uniform(()) < 0.5,
                 lambda: indices,
                 lambda: indices[:, ::-1])


def get_shuffled_indices_and_labels(batch_size, num_samples, shuffle_fraction,
                                    num_steps):
  """Produce possibly shuffled indices and labels."""
  total_num_samples = batch_size * num_samples
  num_shuffled_examples = int(shuffle_fraction * total_num_samples)

  shuffle_labels = tf.random.shuffle(tf.cast(
      num_shuffled_examples*[1] +
      (total_num_samples - num_shuffled_examples) * [0], tf.int32))
  indices = tf.sort(random_choice_noreplace(
      total_num_samples, num_steps)[:, :5], axis=1)
  indices = randomly_reverse_indices(indices)
  shuffled_samples = tf.where(
      tf.less_equal(tf.random.uniform((total_num_samples, 1)), 0.5),
      tf.gather(indices, [1, 0, 3], axis=1),
      tf.gather(indices, [1, 4, 3], axis=1))
  ordered_samples = tf.gather(indices, [1, 2, 3], axis=1)
  indices = tf.where(tf.equal(tf.expand_dims(shuffle_labels, axis=-1), 1),
                     shuffled_samples, ordered_samples)

  return indices, shuffle_labels


def sample_batch(embs, batch_size, num_steps):
  """Returns concatenated features and shuffle labels."""
  shuffle_fraction = CONFIG.SAL.SHUFFLE_FRACTION
  num_samples = CONFIG.SAL.NUM_SAMPLES
  indices, labels = get_shuffled_indices_and_labels(batch_size,
                                                    num_samples,
                                                    shuffle_fraction,
                                                    num_steps)
  labels = tf.one_hot(labels, 2)
  labels = tf.stop_gradient(labels)
  indices = tf.stop_gradient(indices)
  embs = tf.tile(embs, [num_samples, 1, 1])
  embs = tf.gather(embs, indices, batch_dims=-1)
  concat_embs = tf.squeeze(tf.concat(tf.split(embs, 3, axis=1), axis=-1),
                           axis=1)
  return concat_embs, labels


class SaL(Algorithm):
  """Shuffle and Learn algorithm (https://arxiv.org/abs/1603.08561) ."""

  def __init__(self, model=None):
    super(SaL, self).__init__(model)
    if CONFIG.SAL.FC_LAYERS[-1][0] != 2:
      raise ValueError('Shuffle and Learn classifier has only 2 classes:'
                       'correct order or incorrect order. Ensure last layer in '
                       'config.sal.fc_layers is 2.')

    sal_classifier = Classifier(CONFIG.SAL.FC_LAYERS, CONFIG.SAL.DROPOUT_RATE)
    self.model['sal_classifier'] = sal_classifier

  def get_algo_variables(self):
    return self.model['sal_classifier'].variables

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):
    if training:
      batch_size = CONFIG.TRAIN.BATCH_SIZE
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      batch_size = CONFIG.EVAL.BATCH_SIZE
      num_steps = CONFIG.EVAL.NUM_FRAMES

    concat_embs, labels = sample_batch(embs, batch_size, num_steps)
    logits = self.model['sal_classifier'](concat_embs)

    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True,
            label_smoothing=CONFIG.SAL.LABEL_SMOOTHING))

    return loss
