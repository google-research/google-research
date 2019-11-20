# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Various functions used for TD3 and DDPG implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import Mask
import numpy as np
from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe


def soft_update(vars_, target_vars, tau=1.0):
  """Performs soft updates of the target networks.

  Args:
    vars_: a list of parameters of a source network ([tf.Variable]).
    target_vars: a list of parameters of a target network ([tf.Variable]).
    tau: update parameter.
  """
  for var, var_target in zip(vars_, target_vars):
    var_target.assign((1 - tau) * var_target + tau * var)


def do_rollout(env,
               actor,
               replay_buffer,
               noise_scale=0.1,
               num_trajectories=1,
               rand_actions=0,
               sample_random=False,
               add_absorbing_state=False):
  """Do N rollout.

  Args:
      env: environment to train on.
      actor: policy to take actions.
      replay_buffer: replay buffer to collect samples.
      noise_scale: std of gaussian noise added to a policy output.
      num_trajectories: number of trajectories to collect.
      rand_actions: number of random actions before using policy.
      sample_random: whether to sample a random trajectory or not.
      add_absorbing_state: whether to add an absorbing state.
  Returns:
    An episode reward and a number of episode steps.
  """
  total_reward = 0
  total_timesteps = 0

  for _ in range(num_trajectories):
    obs = env.reset()
    episode_timesteps = 0
    while True:
      if (replay_buffer is not None and
          len(replay_buffer) < rand_actions) or sample_random:
        action = env.action_space.sample()
      else:
        tfe_obs = contrib_eager_python_tfe.Variable([obs.astype('float32')])
        action = actor(tfe_obs).numpy()[0]
        if noise_scale > 0:
          action += np.random.normal(size=action.shape) * noise_scale
        action = action.clip(-1, 1)

      next_obs, reward, done, _ = env.step(action)
      # Extremely important, otherwise Q function is not stationary!
      # Taken from: https://github.com/sfujim/TD3/blob/master/main.py#L123
      if not done or episode_timesteps + 1 == env._max_episode_steps:  # pylint: disable=protected-access
        done_mask = Mask.NOT_DONE.value
      else:
        done_mask = Mask.DONE.value

      total_reward += reward
      episode_timesteps += 1
      total_timesteps += 1

      if replay_buffer is not None:
        if (add_absorbing_state and done and
            episode_timesteps < env._max_episode_steps):  # pylint: disable=protected-access
          next_obs = env.get_absorbing_state()
        replay_buffer.push_back(obs, action, next_obs, [reward], [done_mask],
                                done)

      if done:
        break

      obs = next_obs

    # Add an absorbing state that is extremely important for GAIL.
    if add_absorbing_state and (replay_buffer is not None and
                                episode_timesteps < env._max_episode_steps):  # pylint: disable=protected-access
      action = np.zeros(env.action_space.shape)
      absorbing_state = env.get_absorbing_state()

      # done=False is set to the absorbing state because it corresponds to
      # a state where gym environments stopped an episode.
      replay_buffer.push_back(absorbing_state, action, absorbing_state, [0.0],
                              [Mask.ABSORBING.value], False)
  return total_reward / num_trajectories, total_timesteps // num_trajectories
