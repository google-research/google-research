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

"""An implementation of GAIL with WGAN discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
from tensorflow.contrib.gan.python.losses.python import losses_impl as contrib_gan_python_losses_python_losses_impl


class Discriminator(tf.keras.Model):
  """Implementation of a discriminator network."""

  def __init__(self, input_dim):
    """Initializes a discriminator.

    Args:
       input_dim: size of the input space.
    """
    super(Discriminator, self).__init__()
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

    self.main = tf.keras.Sequential([
        tf.layers.Dense(
            units=256,
            input_shape=(input_dim,),
            activation='tanh',
            kernel_initializer=kernel_init),
        tf.layers.Dense(
            units=256, activation='tanh', kernel_initializer=kernel_init),
        tf.layers.Dense(units=1, kernel_initializer=kernel_init)
    ])

  def call(self, inputs):
    """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).

    Returns:
      Values of observations.
    """
    return self.main(inputs)


class GAIL(object):
  """Implementation of GAIL (https://arxiv.org/abs/1606.03476).

  Instead of the original GAN, it uses WGAN (https://arxiv.org/pdf/1704.00028).
  """

  def __init__(self, input_dim, subsampling_rate, lambd=10.0, gail_loss='airl'):
    """Initializes actor, critic, target networks and optimizers.

    Args:
       input_dim: size of the observation space.
       subsampling_rate: subsampling rate that was used for expert trajectories.
       lambd: gradient penalty coefficient for wgan.
       gail_loss: gail loss to use.
    """

    self.subsampling_rate = subsampling_rate
    self.lambd = lambd
    self.gail_loss = gail_loss

    with tf.variable_scope('discriminator'):
      self.disc_step = contrib_eager_python_tfe.Variable(
          0, dtype=tf.int64, name='step')
      self.discriminator = Discriminator(input_dim)
      self.discriminator_optimizer = tf.train.AdamOptimizer()
      self.discriminator_optimizer._create_slots(self.discriminator.variables)  # pylint: disable=protected-access

  def update(self, batch, expert_batch):
    """Updates the WGAN potential function or GAN discriminator.

    Args:
       batch: A batch from training policy.
       expert_batch: A batch from the expert.
    """
    obs = contrib_eager_python_tfe.Variable(
        np.stack(batch.obs).astype('float32'))
    expert_obs = contrib_eager_python_tfe.Variable(
        np.stack(expert_batch.obs).astype('float32'))

    expert_mask = contrib_eager_python_tfe.Variable(
        np.stack(expert_batch.mask).astype('float32'))

    # Since expert trajectories were resampled but no absorbing state,
    # statistics of the states changes, we need to adjust weights accordingly.
    expert_mask = tf.maximum(0, -expert_mask)
    expert_weight = expert_mask / self.subsampling_rate + (1 - expert_mask)

    action = contrib_eager_python_tfe.Variable(
        np.stack(batch.action).astype('float32'))
    expert_action = contrib_eager_python_tfe.Variable(
        np.stack(expert_batch.action).astype('float32'))

    inputs = tf.concat([obs, action], -1)
    expert_inputs = tf.concat([expert_obs, expert_action], -1)

    # Avoid using tensorflow random functions since it's impossible to get
    # the state of the random number generator used by TensorFlow.
    alpha = np.random.uniform(size=(inputs.get_shape()[0], 1))
    alpha = contrib_eager_python_tfe.Variable(alpha.astype('float32'))
    inter = alpha * inputs + (1 - alpha) * expert_inputs

    with tf.GradientTape() as tape:
      output = self.discriminator(inputs)
      expert_output = self.discriminator(expert_inputs)

      with contrib_summary.record_summaries_every_n_global_steps(
          100, self.disc_step):
        gan_loss = contrib_gan_python_losses_python_losses_impl.modified_discriminator_loss(
            expert_output,
            output,
            label_smoothing=0.0,
            real_weights=expert_weight)
        contrib_summary.scalar(
            'discriminator/expert_output',
            tf.reduce_mean(expert_output),
            step=self.disc_step)
        contrib_summary.scalar(
            'discriminator/policy_output',
            tf.reduce_mean(output),
            step=self.disc_step)

      with tf.GradientTape() as tape2:
        tape2.watch(inter)
        output = self.discriminator(inter)
        grad = tape2.gradient(output, [inter])[0]

      grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))

      loss = gan_loss + self.lambd * grad_penalty

    with contrib_summary.record_summaries_every_n_global_steps(
        100, self.disc_step):
      contrib_summary.scalar(
          'discriminator/grad_penalty', grad_penalty, step=self.disc_step)

    with contrib_summary.record_summaries_every_n_global_steps(
        100, self.disc_step):
      contrib_summary.scalar(
          'discriminator/loss', gan_loss, step=self.disc_step)

    grads = tape.gradient(loss, self.discriminator.variables)

    self.discriminator_optimizer.apply_gradients(
        zip(grads, self.discriminator.variables), global_step=self.disc_step)

  def get_reward(self, obs, action, next_obs):  # pylint: disable=unused-argument
    if self.gail_loss == 'airl':
      inputs = tf.concat([obs, action], -1)
      return self.discriminator(inputs)
    else:
      inputs = tf.concat([obs, action], -1)
      return -tf.log(1 - tf.nn.sigmoid(self.discriminator(inputs)) + 1e-8)

  @property
  def variables(self):
    """Returns all variables including optimizer variables.

    Returns:
      A dictionary of all variables that are defined in the model.
      variables.
    """
    disc_vars = (
        self.discriminator.variables + self.discriminator_optimizer.variables()
        + [self.disc_step])

    return disc_vars
