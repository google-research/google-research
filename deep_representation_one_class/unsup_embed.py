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

# Lint as: python3
"""Unsupervised Embedding module."""

import tensorflow as tf

from deep_representation_one_class.util.train import BaseTrain


class UnsupEmbed(BaseTrain):
  """UnsupEmbed."""

  def __init__(self, hparams):
    super(UnsupEmbed, self).__init__(hparams=hparams)

    assert self.latent_dim == len(self.aug_list), \
        'latent_dim should be set to {}'.format(len(self.aug_list))

  def set_metrics(self):
    # Metrics
    self.list_of_metrics = ['loss.train', 'loss.xe', 'loss.L2', 'acc.train']
    self.list_of_eval_metrics = [
        'logit.auc',
        'dscore.auc',
        'embed.auc',
        'embed.kocsvm',
        'embed.locsvm',
        'embed.kde',
        'embed.gde',
        'pool.auc',
        'pool.kocsvm',
        'pool.locsvm',
        'pool.kde',
        'pool.gde',
    ]
    self.metric_of_interest = [
        'logit.auc',
        'dscore.auc',
        'embed.auc',
        'embed.kocsvm',
        'embed.locsvm',
        'embed.kde',
        'embed.gde',
        'pool.auc',
        'pool.kocsvm',
        'pool.locsvm',
        'pool.kde',
        'pool.gde',
    ]
    assert all([
        m in self.list_of_eval_metrics for m in self.metric_of_interest
    ]), 'Some metric does not exist'

  @tf.function
  def train_step(self, iterator):
    """Train step."""

    def step_fn(input_data):
      replica_context = tf.distribute.get_replica_context()
      x, num_aug = input_data[:-2], self.latent_dim
      y = [
          tf.scalar_mul(i, tf.ones(x[i].shape[0], dtype=tf.int32))
          for i in range(num_aug)
      ]
      x = tf.concat(x, axis=0)
      y = tf.one_hot(tf.concat(y, axis=0), num_aug)
      with tf.GradientTape() as tape:
        logits = self.model(x, training=True)['logits']
        loss_xe = tf.keras.losses.categorical_crossentropy(
            y, logits, from_logits=True)
        loss_xe = tf.divide(
            tf.reduce_sum(loss_xe),
            self.cross_replica_concat(loss_xe,
                                      replica_context=replica_context).shape[0])
        loss_l2 = self.loss_l2(self.model.trainable_weights)
        loss = loss_xe + self.weight_decay * loss_l2
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
      # monitor
      self.metrics['loss.train'].update_state(loss)
      self.metrics['loss.xe'].update_state(loss_xe)
      self.metrics['loss.L2'].update_state(loss_l2)
      self.metrics['acc.train'].update_state(
          tf.argmax(y, axis=1), tf.argmax(logits, axis=1))

    # Call one step
    self.strategy.run(step_fn, args=(next(iterator),))
