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

# Lint as: python3
"""Tests for drq_sac_agent. Implementation by Ilya Kostrikov."""
import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import spec_utils
from tf_agents.utils import test_utils

from pse.dm_control.agents import drq_sac_agent
from pse.dm_control.agents import pse_drq_agent
from pse.dm_control.utils import env_utils
from pse.dm_control.utils import networks


class DrqSacAgentTest(test_utils.TestCase):

  def setup_agent(self, target_update_tau=0.01, target_update_period=2):
    env = env_utils.load_dm_env_for_training(
        'ball_in_cup-catch',
        frame_shape=(84, 84, 3),
        action_repeat=4,
        frame_stack=3)
    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))

    image_encoder = networks.ImageEncoder(observation_tensor_spec)

    actor_net = networks.Actor(
        observation_tensor_spec,
        action_tensor_spec,
        image_encoder=image_encoder,
        fc_layers=(1024, 1024))

    critic_net_1 = networks.Critic(
        (observation_tensor_spec, action_tensor_spec),
        image_encoder=image_encoder,
        joint_fc_layers=(1024, 1024))
    critic_net_2 = networks.Critic(
        (observation_tensor_spec, action_tensor_spec),
        image_encoder=image_encoder,
        joint_fc_layers=(1024, 1024))

    target_image_encoder = networks.ImageEncoder(observation_tensor_spec)
    target_critic_net_1 = networks.Critic(
        (observation_tensor_spec, action_tensor_spec),
        image_encoder=target_image_encoder)
    target_critic_net_2 = networks.Critic(
        (observation_tensor_spec, action_tensor_spec),
        image_encoder=target_image_encoder)

    train_step = tf.Variable(
        1,
        trainable=False,
        dtype=tf.int64,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        shape=())

    agent = drq_sac_agent.DrQSacAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_network=actor_net,
        critic_network=critic_net_1,
        critic_network_2=critic_net_2,
        target_critic_network=target_critic_net_1,
        target_critic_network_2=target_critic_net_2,
        actor_update_frequency=2,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=1.0,
        use_log_alpha_in_alpha_loss=False,
        gradient_clipping=None,
        train_step_counter=train_step)

    self._env = env
    self._observation_tensor_spec = observation_tensor_spec
    self._action_tensor_spec = action_tensor_spec
    self._time_step_tensor_spec = time_step_tensor_spec

    self._image_encoder = image_encoder
    self._actor_net = actor_net
    self._critic_net_1 = critic_net_1
    self._critic_net_2 = critic_net_2

    self._target_image_encoder = target_image_encoder
    self._target_critic_net_1 = target_critic_net_1
    self._target_critic_net_2 = target_critic_net_2

    self._agent = agent

  def test_network_variables_shared_correctly(self):
    self.setup_agent()
    encoder_variables = self._image_encoder.variables
    num_encoder_variables = len(encoder_variables)

    self.assertEqual(encoder_variables,
                     self._actor_net.variables[:num_encoder_variables])
    self.assertEqual(encoder_variables,
                     self._critic_net_1.variables[:num_encoder_variables])
    self.assertEqual(encoder_variables,
                     self._critic_net_2.variables[:num_encoder_variables])

    target_encoder_variables = self._target_image_encoder.variables

    self.assertEqual(
        target_encoder_variables,
        self._target_critic_net_1.variables[:num_encoder_variables])
    self.assertEqual(
        target_encoder_variables,
        self._target_critic_net_2.variables[:num_encoder_variables])

    # Different instances.
    self.assertTrue(
        all([
            v1 is not v2
            for v1, v2 in zip(encoder_variables, target_encoder_variables)
        ]))

    # Different values.
    self.assertTrue(
        any([
            tf.reduce_all(v1 != v2)
            for v1, v2 in zip(encoder_variables, target_encoder_variables)
        ]))

    # Same values after initialize
    self._agent.initialize()
    self.assertTrue(
        all([
            tf.reduce_all(v1 == v2)
            for v1, v2 in zip(encoder_variables, target_encoder_variables)
        ]))

  def test_actor_updated_on_second_train(self):
    self.setup_agent()
    experience_spec = self._agent.collect_data_spec

    def _bound_specs(s):
      if s.dtype != tf.float32:
        return s
      return tensor_spec.BoundedTensorSpec(
          dtype=s.dtype, shape=s.shape, minimum=-1, maximum=1)

    experience_spec = tf.nest.map_structure(_bound_specs, experience_spec)

    sample_experience_1 = tensor_spec.sample_spec_nest(
        experience_spec, outer_dims=(2,))
    sample_experience_2 = tensor_spec.sample_spec_nest(
        experience_spec, outer_dims=(2,))

    augmented_sample_1 = pse_drq_agent.image_aug(
        sample_experience_1, (),
        img_pad=4,
        num_augmentations=2)

    augmented_sample_2 = pse_drq_agent.image_aug(
        sample_experience_2, (),
        img_pad=4,
        num_augmentations=2)

    augmented_sample = tf.nest.map_structure(
        # pylint: disable=g-long-lambda
        lambda t1, t2: tf.concat([tf.expand_dims(t1, 0),
                                  tf.expand_dims(t2, 0)],
                                 axis=0),
        augmented_sample_1,
        augmented_sample_2)[0]

    augmented_experience = augmented_sample.pop('experience')
    sample_train_kwargs = augmented_sample

    self._agent.initialize()

    encoder_variables = self._image_encoder.variables
    num_encoder_variables = len(encoder_variables)

    # Evaluate here to get a copy of the values.
    actor_variables = self.evaluate([v for v in self._actor_net.variables])

    self._agent.train(augmented_experience, **sample_train_kwargs)

    updated_actor_variables = self.evaluate(
        [v for v in self._actor_net.variables])

    for v1, v2 in zip(
        actor_variables[num_encoder_variables:],
        updated_actor_variables[num_encoder_variables:]):
      np.testing.assert_equal(v1, v2)

    # Second call now variables should differ.
    self._agent.train(augmented_experience, **sample_train_kwargs)

    updated_actor_variables = self.evaluate(
        [v for v in self._actor_net.variables])

    with self.assertRaises(AssertionError):
      for v1, v2 in zip(
          actor_variables[num_encoder_variables:],
          updated_actor_variables[num_encoder_variables:]):
        np.testing.assert_equal(v1, v2)


if __name__ == '__main__':
  test_utils.main()
