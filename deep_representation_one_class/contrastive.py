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

# Lint as: python3
"""Contrastive learning module."""

import tensorflow as tf

from deep_representation_one_class.util.train import BaseTrain


class Contrastive(BaseTrain):
  """Contrastive learning."""

  def __init__(self, hparams):
    super(Contrastive, self).__init__(hparams=hparams)

  def set_hparams(self, hparams):
    # Algorithm-specific parameter
    self.temperature = hparams.temperature

    # File suffix
    self.file_suffix = 'temp{:g}'.format(self.temperature)

  def set_metrics(self):
    # Metrics
    self.list_of_metrics = ['loss.train', 'loss.xe', 'loss.L2', 'acc.train']
    self.list_of_eval_metrics = [
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

  def get_target_labels(self, x, is_onehot=True, replica_context=None):
    x_concat = self.cross_replica_concat(x, replica_context=replica_context)
    replica_idx = replica_context.replica_id_in_sync_group
    global_batch_size = x_concat.shape[0]
    num_per_replica = tf.math.floordiv(global_batch_size,
                                       replica_context.num_replicas_in_sync)
    target_labels = tf.range(replica_idx * num_per_replica,
                             (replica_idx + 1) * num_per_replica)
    if is_onehot:
      target_labels = tf.one_hot(target_labels, global_batch_size)
    return target_labels

  @tf.function
  def train_step(self, iterator):
    """Train step."""

    def step_fn(data):
      x1, x2 = data[0], data[1]
      replica_context = tf.distribute.get_replica_context()
      y = self.get_target_labels(
          x1, is_onehot=True, replica_context=replica_context)
      with tf.GradientTape() as tape:
        xc = tf.concat((x1, x2), axis=0)
        embeds = self.model(xc, training=True)['embeds']
        embeds = tf.nn.l2_normalize(embeds, axis=1)
        embeds1, embeds2 = tf.split(embeds, 2)
        embeds2_concat = self.cross_replica_concat(
            embeds2, replica_context=replica_context)
        ip = tf.matmul(embeds1, embeds2_concat, transpose_b=True)
        loss_xe = tf.keras.losses.categorical_crossentropy(
            y, tf.divide(ip, self.temperature), from_logits=True)
        loss_xe = self.global_reduce_mean(loss_xe)
        loss_l2 = self.loss_l2(self.model.trainable_weights)
        loss = loss_xe + self.weight_decay * loss_l2
      grad = tape.gradient(loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
      # monitor
      self.metrics['loss.train'].update_state(loss)
      self.metrics['loss.xe'].update_state(loss_xe)
      self.metrics['loss.L2'].update_state(loss_l2)
      self.metrics['acc.train'].update_state(
          tf.argmax(y, axis=1), tf.argmax(ip, axis=1))

    # Call one step
    self.strategy.run(step_fn, args=(next(iterator),))

  def global_reduce_mean(self, tensor, axis=None, replica_context=None):
    """Return global mean across multiple replica."""
    return tf.divide(
        tf.reduce_sum(tensor, axis=axis),
        self.cross_replica_concat(tensor,
                                  replica_context=replica_context).shape[0])
