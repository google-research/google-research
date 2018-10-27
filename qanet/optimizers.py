# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Model optimizers and initializers."""
import numpy as np

import tensorflow as tf

from qanet.util import configurable
from qanet.util import misc_util


class Decay(configurable.Configurable):
  """Base class for learning rate decays."""

  @staticmethod
  def _config():
    return {'warmup_steps': 0, 'warmup_type': 'inv_decay'}

  def __call__(self, learning_rate, global_step, train_steps):
    decayed = self._decay(learning_rate, global_step, train_steps)

    warmup_steps = self.config['warmup_steps']
    if self.config['warmup_type'] == 'inv_decay':
      inv_base = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = inv_base**tf.to_float(warmup_steps - global_step)
      warmup_lr = inv_decay * learning_rate
    elif self.config['warmup_type'] == 'constant':
      warmup_lr = learning_rate
    else:
      raise ValueError('Unknown warmup type %s' % self.config['warmup_type'])
    decayed_lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr,
                         lambda: decayed)
    tf.summary.scalar('learning_rate', decayed_lr)
    return decayed_lr


class NoDecay(Decay):
  """Constant learning rate."""

  def _decay(self, learning_rate, global_step, train_steps):
    return learning_rate


class CosineDecay(Decay):
  """Cosine learning rate decay."""

  def _decay(self, learning_rate, global_step, train_steps):
    decay_end = train_steps
    return 0.5 * learning_rate * (
        1 + tf.cos(np.pi *
                   (tf.to_float(global_step - self.config['warmup_steps'])) /
                   (decay_end - self.config['warmup_steps'])))


class LinearDecay(Decay):
  """Linearly decayed learning rate."""

  def _decay(self, learning_rate, global_step, train_steps):
    decay_end = train_steps

    def decayed():
      steps = (decay_end - tf.to_float(global_step) - self.config['warmup_steps'])
      total = (decay_end - self.config['warmup_steps'])
      return learning_rate * (steps / total)

    return tf.cond(
        global_step > decay_end,
        lambda: 0.0,
        lambda: decayed,
        name='exponential_decay_step_cond')


class ExponentialDecay(Decay):
  """Exponentially decayed learning rate."""

  @staticmethod
  def _config():
    config = Decay._config()
    config.update({
        'warmup_type': 'inv_decay',
        'rate': 0.1,
        'steps': 5000,
        'staircase': False,
    })
    return config

  def _decay(self, learning_rate, global_step, train_steps):
    # TODO(ddohan): Make this epoch based
    return tf.cond(
        global_step < self.config['warmup_steps'],
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(  # pylint: disable=g-long-lambda
            learning_rate=learning_rate,
            global_step=global_step - self.config['warmup_steps'],
            decay_steps=self.config['steps'],
            decay_rate=self.config['rate'],
            staircase=self.config['staircase']),
        name='exponential_decay_step_cond')



class Initializer(configurable.Configurable):
  pass


class XavierInit(Initializer):

  def __call__(self):
    return tf.contrib.layers.xavier_initializer()


class Optimizer(configurable.Configurable):
  """Base type for optimizers that are usable inside Model.

  Subclasses should setup the optimizer in _fn.
  """

  def __init__(self, config=None):
    super(Optimizer, self).__init__(config)
    self._ema = None

  @staticmethod
  def _config():
    return {
        'learning_rate': 0.001,
        'gradient_clipping_norm': 10.0,
        'decay': NoDecay,
        'l2_reg': 0.0,
        'ema_decay': 0.9999,  # Exponential Moving Average decay
        'normalized_grad': False,  # Normalized gradient
        'grad_noise': 0.0,
        'nograd_var': 'pretrained_encoder',
        'keepgrad_var': '',  # keepgrad_var is considered first, then nograd_var
        'alpha': 0.2
    }

  def _fn(self, lr):
    """Return a callable Optimizer object."""
    raise NotImplementedError

  @property
  def exponential_moving_average(self):
    """Return the EMA object for the optimizer."""
    if self._ema is None:
      self._ema = tf.train.ExponentialMovingAverage(
          decay=self.config['ema_decay'])
    return self._ema

  def __call__(self, loss, train_steps):

    # Regularize
    if self.config['l2_reg']:
      tf.logging.info('Applying l2 regularization of %s', self.config['l2_reg'])
      decay_costs = []
      for var in tf.trainable_variables():
        decay_costs.append(tf.nn.l2_loss(var))
      loss += tf.multiply(self.config['l2_reg'], tf.add_n(decay_costs))

    # Clipping
    if self.config['normalized_grad']:
      tf.logging.info('Applying normalized gradient, alpha %f' %
                      self.config['alpha'])
      def normalize_grad_fn(grads_and_vars):
        normalized_grads = []
        for grad, var in grads_and_vars:
          normalized_grads += [(grad / (tf.norm(grad) + tf.constant(1e-10)) *
                                tf.norm(var) * self.config['alpha'], var)]
        return normalized_grads
      clip_norm = normalize_grad_fn
    else:
      tf.logging.info('Applying clip norm %s' %
                      self.config['gradient_clipping_norm'])
      clip_norm = self.config['gradient_clipping_norm'] or None

    # Decay
    decay = configurable.Configurable.initialize(self.config['decay'])
    def decay_fn(learning_rate, global_step):
      return decay(learning_rate, global_step, train_steps)

    # Trainable variables
    trainable_vars, _, _ = misc_util.get_trainable_vars(
        keep_pattern=self.config['keepgrad_var'],
        exclude_pattern=self.config['nograd_var'])

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_or_create_global_step(),
        learning_rate=self.config['learning_rate'],
        learning_rate_decay_fn=decay_fn,
        clip_gradients=clip_norm,
        optimizer=self._fn,
        variables=trainable_vars,
        gradient_noise_scale=self.config['grad_noise'],
        summaries=['learning_rate', 'loss'],
        colocate_gradients_with_ops=True)

    if self.config['ema_decay'] < 1.0:
      # Keep track of an exponential moving average during training
      # TODO(ddohan): Allow tracking multiple values
      ema = self.exponential_moving_average
      maintain_average_op = ema.apply(trainable_vars)
      with tf.control_dependencies([train_op]):
        train_op = tf.group(maintain_average_op)
    return train_op


class SGDOptimizer(Optimizer):

  def _fn(self, lr):
    return tf.train.GradientDescentOptimizer(lr)


class AdamOptimizer(Optimizer):

  @staticmethod
  def _config():
    config = Optimizer._config()
    config.update({'epsilon': 1e-8, 'beta1': 0.9, 'beta2': 0.999})
    return config

  def _fn(self, lr):
    return tf.train.AdamOptimizer(
        lr,
        epsilon=self.config['epsilon'],
        beta1=self.config['beta1'],
        beta2=self.config['beta2'])
