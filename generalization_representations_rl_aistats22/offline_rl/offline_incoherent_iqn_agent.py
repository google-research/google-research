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

"""Offline Implicit Quantile agent with incoherent / orthogonality loss."""

from absl import logging
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
import gin
import numpy as onp

from generalization_representations_rl_aistats22.atari import incoherent_implicit_quantile_agent
from generalization_representations_rl_aistats22.offline_rl import fixed_replay


@gin.configurable
class OfflineIncoherentImplicitQuantileAgent(
    incoherent_implicit_quantile_agent.IncoherentImplicitQuantileAgent):
  """Implicit Quantile Agent with the Coherence loss."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               replay_suffix=None,
               replay_start_index=0,
               summary_writer=None,
               replay_buffer_builder=None):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      replay_start_index: int, Starting index for loading the data from files
        in `replay_data_dir`.
      summary_writer: SummaryWriter object for outputting training statistics
      replay_buffer_builder: Callable object that takes "self" as an argument
        and returns a replay buffer to use for training offline. If None,
        it will use the default FixedReplayBuffer.
    """
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t replay directory: %s', replay_data_dir)
    logging.info('\t replay_suffix %s', replay_suffix)
    logging.info('\t replay_suffix %d', replay_start_index)
    self.replay_data_dir = replay_data_dir
    self.replay_suffix = replay_suffix
    self.replay_start_index = replay_start_index
    if replay_buffer_builder is not None:
      self._build_replay_buffer = replay_buffer_builder

    # update_period=1 is a sane default for offline RL. However, this
    # can still be overridden with gin config.
    super().__init__(num_actions, summary_writer=summary_writer)

  def _build_replay_buffer(self):
    """Creates the fixed replay buffer used by the agent."""
    return fixed_replay.FixedReplayBuffer(
        data_dir=self.replay_data_dir,
        replay_suffix=self.replay_suffix,
        replay_start_index=self.replay_start_index,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)
    self._rng, self.action = implicit_quantile_agent.select_action(
        self.network_def, self.online_params, self.state, self._rng,
        self.num_quantile_samples, self.num_actions, self.eval_mode,
        self.epsilon_eval, self.epsilon_train, self.epsilon_decay_period,
        self.training_steps, self.min_replay_history, self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

  def end_episode(self, reward, terminal=True):
    """Override the function to do nothing."""
    pass

  def train_step(self):
    """Exposes the train step for offline learning."""
    self._train_step()
