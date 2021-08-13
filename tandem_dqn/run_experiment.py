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

"""Run training with Tandem-RL: one active and one passive agent."""

import sys

from absl import logging
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
import gin
import numpy as np
import tensorflow as tf
from tandem_dqn import minatar_env
from tandem_dqn import tandem_dqn_agent


@gin.configurable
def create_tandem_agents_and_checkpoints(sess, environment, agent_name='dqn',
                                         summary_writer=None, debug_mode=False):
  """Creates a tandem agent.

  Args:
    sess: TF session, unused since we are in JAX.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, unused.

  Returns:
    An active and passive agent.
  """
  assert agent_name is not None
  del sess
  del debug_mode
  if agent_name == 'dqn':
    return tandem_dqn_agent.TandemDQNAgent(
        num_actions=environment.action_space.n, summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
class TandemRunner(run_experiment.Runner):
  """Runner class to run Tandem RL experiments."""

  def __init__(self,
               base_dir,
               create_agent_fn,
               suite='atari'):
    if suite == 'atari':
      create_environment_fn = atari_lib.create_atari_environment
    elif suite == 'classic':
      create_environment_fn = gym_lib.create_gym_environment
    elif suite == 'minatar':
      create_environment_fn = minatar_env.create_minatar_env
    else:
      raise ValueError(f'Unknown suite: {suite}')
    super().__init__(base_dir, create_agent_fn,
                     create_environment_fn=create_environment_fn)

  def _initialize_episode(self, agent_type='active'):
    """Initialization for a new episode.

    Args:
      agent_type: str, the type of agent to run.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(agent_type, initial_observation)

  def _run_one_episode(self, agent_type='active'):
    """Executes a full trajectory of the agent interacting with the environment.

    Args:
      agent_type: str, the type of agent to run.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode(agent_type)
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        if agent_type == 'active':
          self._agent.end_episode(reward)
        action = self._agent.begin_episode(agent_type, observation)
      else:
        action = self._agent.step(agent_type, reward, observation)

    self._end_episode(reward)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str,
                     agent_type='active'):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.
      agent_type: str, the type of agent to run.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode(agent_type)
      statistics.append({
          '{}_{}_episode_lengths'.format(run_mode_str,
                                         agent_type): episode_length,
          '{}_{}_episode_returns'.format(run_mode_str,
                                         agent_type): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of logging so as to flush frequently
      # without generating a line break.
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_eval_phase(self, statistics, agent_type='active'):
    """Run evaluation phase on both passive and active learners.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.
      agent_type: str, the type of agent to run.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval', agent_type)
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per eval episode (%s): %.2f',
                 agent_type, average_return)
    statistics.append({f'{agent_type}_eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    active_num_episodes_eval, active_average_reward_eval = self._run_eval_phase(
        statistics, 'active')
    passive_num_episodes_eval, passive_average_reward_eval = (
        self._run_eval_phase(statistics, 'passive'))

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train,
                                     active_num_episodes_eval,
                                     active_average_reward_eval,
                                     passive_num_episodes_eval,
                                     passive_average_reward_eval,
                                     average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  active_num_episodes_eval,
                                  active_average_reward_eval,
                                  passive_num_episodes_eval,
                                  passive_average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      active_num_episodes_eval: int, number of active evaluation episodes run.
      active_average_reward_eval: float, The average active evaluation reward.
      passive_num_episodes_eval: int, number of passive evaluation episodes run.
      passive_average_reward_eval: float, The average passive evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    summary = tf.compat.v1.Summary(value=[
        tf.compat.v1.Summary.Value(
            tag='Train/NumEpisodes', simple_value=num_episodes_train),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward_train),
        tf.compat.v1.Summary.Value(
            tag='Train/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
        tf.compat.v1.Summary.Value(
            tag='Eval/ActiveNumEpisodes',
            simple_value=active_num_episodes_eval),
        tf.compat.v1.Summary.Value(
            tag='Eval/ActiveAverageReturns',
            simple_value=active_average_reward_eval),
        tf.compat.v1.Summary.Value(
            tag='Eval/PassiveNumEpisodes',
            simple_value=passive_num_episodes_eval),
        tf.compat.v1.Summary.Value(
            tag='Eval/PassiveAverageReturns',
            simple_value=passive_average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)
    self._summary_writer.flush()
