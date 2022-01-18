# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Utilities for RPC."""

from absl import logging
import gin
import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class ActorNet(actor_distribution_network.ActorDistributionNetwork):
  """An actor network for the RPC agent.

  The parent ActorDistributionClass already has an attribute self._encoder,
  which is different from the z encoder used here.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layers,
               encoder,
               predictor,
               initial_log_kl=np.log(1e-6)):
    self._s_spec = input_tensor_spec
    self._latent_dim = encoder.layers[-1]._event_shape
    self._z_spec = tensor_spec.TensorSpec(
        shape=(self._latent_dim,),
        dtype=tf.float32)
    super(ActorNet, self).__init__(
        input_tensor_spec=self._z_spec,
        output_tensor_spec=output_tensor_spec,
        fc_layer_params=fc_layers,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)  # pylint: disable=line-too-long
    self._input_tensor_spec = input_tensor_spec
    self._z_encoder = encoder
    self._predictor = predictor
    self._log_kl_coefficient = tf.Variable(initial_log_kl, dtype=tf.float32)

  @property
  def trainable_variables(self):
    extra_vars = [self._log_kl_coefficient]
    # The parent class already includes the encoder variables.
    return (super(ActorNet, self).trainable_variables + extra_vars
            + self._predictor.trainable_variables)

  def call(self, observations, step_type=(), network_state=(), training=False):
    z = self._z_encoder(observations, training=training)
    z = z.sample()
    self._input_tensor_spec = self._z_spec
    output = super(ActorNet, self).call(
        z, step_type=step_type, network_state=network_state, training=training)
    self._input_tensor_spec = self._s_spec
    return output


@gin.configurable
class CriticNet(critic_network.CriticNetwork):
  """A critic network for the RPC agent."""

  def __init__(self,
               input_tensor_spec,
               observation_fc_layer_params,
               action_fc_layer_params,
               joint_fc_layer_params,
               kernel_initializer,
               last_kernel_initializer,
               name='CriticNetwork'):
    super(CriticNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        observation_fc_layer_params=observation_fc_layer_params,
        action_fc_layer_params=action_fc_layer_params,
        joint_fc_layer_params=joint_fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=last_kernel_initializer,
        name=name,
    )

  def call(self, inputs, step_type=(), network_state=(), training=False):
    observation, action = inputs
    observation_spec, _ = self.input_tensor_spec
    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               observation_spec)
    has_time_dim = num_outer_dims == 2

    if has_time_dim:
      batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
      # Flatten: [B, T, ...] -> [BxT, ...]
      observation = batch_squash.flatten(observation)
      action = batch_squash.flatten(action)
    q_value, network_state = super(CriticNet, self).call(
        (observation, action),
        step_type=step_type,
        network_state=network_state,
        training=training)
    if has_time_dim:
      q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
    return q_value, network_state


@gin.configurable
class RecurrentActorNet(actor_distribution_rnn_network.ActorDistributionRnnNetwork):  # pylint: disable=line-too-long
  """A recurrent actor network for the RPC agent."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layers,
               encoder,
               predictor):
    self._s_spec = input_tensor_spec
    self._latent_dim = encoder.layers[-1]._event_shape
    self._z_spec = tensor_spec.TensorSpec(
        shape=(self._latent_dim,),
        dtype=tf.float32)
    super(RecurrentActorNet, self).__init__(
        input_tensor_spec=self._z_spec,
        output_tensor_spec=output_tensor_spec,
        input_fc_layer_params=(),  # No layers between encoder and LSTM.
        lstm_size=(40,),
        output_fc_layer_params=fc_layers,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)  # pylint: disable=line-too-long
    self._input_tensor_spec = input_tensor_spec
    self._z_encoder = encoder
    self._predictor = predictor
    self._log_kl_coefficient = tf.Variable(0.0)

  @property
  def trainable_variables(self):
    extra_vars = [self._log_kl_coefficient]
    # The parent class already includes the encoder variables.
    return (super(RecurrentActorNet, self).trainable_variables + extra_vars
            + self._predictor.trainable_variables)

  def call(self, observations, step_type=(), network_state=(), training=False):
    num_outer_dims = nest_utils.get_outer_rank(observations,
                                               self.input_tensor_spec)
    has_time_dim = num_outer_dims == 2
    if has_time_dim:
      batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
      # Flattening: [B, T, ...] -> [BxT, ...]
      observations = batch_squash.flatten(observations)
    z = self._z_encoder(observations, training=training)
    z = z.sample()
    if has_time_dim:
      z = batch_squash.unflatten(z)
    self._input_tensor_spec = self._z_spec
    output = super(RecurrentActorNet, self).call(
        z, step_type=step_type, network_state=network_state, training=training)
    self._input_tensor_spec = self._s_spec
    return output


class AverageKLMetric(tf_metric.TFMultiMetricStepMetric):
  """A metric for computing the number of bits used by the policy."""

  def __init__(self,
               encoder,
               predictor,
               name='AverageKL',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=1000):
    super(AverageKLMetric, self).__init__(name=name, prefix=prefix,
                                          metric_names=('KL', 'MSE'))
    self._kl_buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._mse_buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._encoder = encoder
    self._predictor = predictor

  @common.function(autograph=True)
  def call(self, inputs):
    time_step, policy_step, next_time_step = inputs
    action = policy_step.action
    prior = self._predictor((time_step.observation, action), training=False)
    z_next = self._encoder(next_time_step.observation, training=False)
    # Note that kl is a vector of size batch_size.
    kl = tfp.distributions.kl_divergence(z_next, prior)

    mse = tf.keras.losses.MSE(z_next.mean(), prior.mean())
    self._kl_buffer.extend(kl)
    self._mse_buffer.extend(mse)
    return inputs

  def result(self):
    return [self._kl_buffer.mean(), self._mse_buffer.mean()]

  @common.function
  def reset(self):
    self._kl_buffer.clear()
    self._mse_buffer.clear()


@tf.function
def squash_to_range(t, low=-np.inf, high=np.inf):
  """Squashes an input to the range [low, high]."""
  # assert low < 0 < high
  if low == -np.inf:
    t_low = t
  else:
    t_low = -low * tf.nn.tanh(t / (-low))
  if high == np.inf:
    t_high = t
  else:
    t_high = high * tf.nn.tanh(t / high)
  return tf.where(t < 0, t_low, t_high)


def eval_dropout_fn(tf_env, actor_net, global_step, prob_dropout):
  """Evaluates policy when observations are randomly dropped."""
  eval_helper_fn(
      tf_env,
      actor_net,
      prob_dropout=prob_dropout,
      prefix='dropout_%.3f' % prob_dropout,
      global_step=global_step)


@gin.configurable
def eval_helper_fn(tf_env, actor_net, prob_dropout, cutoff=10,
                   num_eval_episodes=10, stochastic_z=False, prefix='',
                   global_step=None):
  """Helper function for performing rollouts to evalaute robustness."""
  assert global_step is not None
  actor_net._input_tensor_spec = actor_net._z_spec  # pylint: disable=protected-access
  assert cutoff >= 1  # We must use the observation to initialize z at the first
  # time step.
  r_vec = []
  r_cutoff_vec = []
  t_vec = []
  x_vec = []
  x_cutoff_vec = []
  network_state = ()
  def _zero_array(spec):
    shape = (1,) + spec.shape
    return tf.zeros(shape, spec.dtype)
  network_state = tf.nest.map_structure(_zero_array, actor_net.state_spec)

  def _get_x(tf_env):
    if (hasattr(tf_env.envs[0], 'gym') and hasattr(tf_env.envs[0].gym, 'data')
        and hasattr(tf_env.envs[0].gym.data, 'qpos')):
      return tf_env.envs[0].gym.data.qpos.flatten()[0]
    elif hasattr(tf_env.envs[0], 'physics'):
      return tf_env.envs[0].physics.data.qpos.flatten()[1]
    else:
      return 0.0

  @tf.function
  def _get_a(z, step_type, network_state):
    a, network_state = super(actor_net.__class__, actor_net).call(
        z, step_type=step_type, network_state=network_state, training=False)
    a = a.sample()
    return a, network_state

  for _ in range(num_eval_episodes):
    ts = tf_env.reset()
    total_r = 0.0
    z = None
    for t in range(1000):
      if t == cutoff:
        x_cutoff_vec.append(_get_x(tf_env))
        r_cutoff_vec.append(tf.identity(total_r))
      if t >= cutoff and np.random.random() < prob_dropout:
        assert z is not None
        # Generate Z by prediction
        z = actor_net._predictor.layers[-1](  # pylint: disable=protected-access
            tf.concat([z, a], axis=1), training=False)
        if stochastic_z:
          z = z.sample()
        else:
          z = z.mean()
      else:
        # Generate Z using the current observation
        z = actor_net._z_encoder(ts.observation, training=False)  # pylint: disable=protected-access
      a, network_state = _get_a(
          z, step_type=ts.step_type, network_state=network_state)
      ts = tf_env.step(a)
      total_r += ts.reward
      if ts.is_last():
        break
    r_vec.append(total_r)
    t_vec.append(t)
    x_vec.append(_get_x(tf_env))
  actor_net._input_tensor_spec = actor_net._s_spec  # pylint: disable=protected-access
  avg_r = tf.reduce_mean(r_vec)
  avg_t = tf.reduce_mean(t_vec)
  avg_x = tf.reduce_mean(x_vec)
  avg_cutoff_r = tf.reduce_mean(r_cutoff_vec)
  avg_cutoff_x = tf.reduce_mean(x_cutoff_vec)

  logging.info('(%s) Return = %.3f', prefix, avg_r)
  logging.info('(%s) Duration = %.3f', prefix, avg_t)
  logging.info('(%s) Final X = %.3f', prefix, avg_x)
  logging.info('(%s) Cutoff Return = %.3f', prefix, avg_cutoff_r)
  logging.info('(%s) Cutoff X = %.3f', prefix, avg_cutoff_x)
  tf.compat.v2.summary.scalar(
      name='%s_return' % prefix, data=avg_r, step=global_step)
  tf.compat.v2.summary.scalar(
      name='%s_duration' % prefix, data=avg_t, step=global_step)
  tf.compat.v2.summary.scalar(
      name='%s_final_x' % prefix, data=avg_x, step=global_step)
  tf.compat.v2.summary.scalar(
      name='%s_cutoff_return' % prefix, data=avg_cutoff_r, step=global_step)
  tf.compat.v2.summary.scalar(
      name='%s_cutoff_x' % prefix, data=avg_cutoff_x, step=global_step)
  return avg_r, avg_t, avg_x, avg_cutoff_r, avg_cutoff_x
