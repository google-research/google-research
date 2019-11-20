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

"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
import gin
import tensorflow as tf
from bisimulation_aaai2020.dopamine import agent_visualizer
from bisimulation_aaai2020.dopamine import rainbow_agent


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 evaluate_metric_only=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    evaluate_metric_only: bool, if set, will evaluate the loaded metric
      approximant.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if agent_name == 'rainbow':
    return rainbow_agent.BisimulationRainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer,
        evaluate_metric_only=evaluate_metric_only)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, trained_agent_checkpoint_path):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    trained_agent_checkpoint_path: str, the path to the checkpoint to reload.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  assert trained_agent_checkpoint_path is not None
  return BisimulationRunner(base_dir, trained_agent_checkpoint_path,
                            create_agent)


@gin.configurable
class BisimulationRunner(run_experiment.Runner):
  """Object that handles filling a replay buffer.
  """

  def __init__(self,
               base_dir,
               trained_agent_checkpoint_path,
               create_agent_fn,
               source_state_step=0):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      trained_agent_checkpoint_path: str, the path to the checkpoint to reload
        which contains the weights for the Q-network.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      source_state_step: int, step number to use for source state (only used
        when calling visualize().
    """
    assert base_dir is not None
    self._trained_agent_checkpoint_path = trained_agent_checkpoint_path
    super(BisimulationRunner, self).__init__(base_dir, create_agent_fn)
    self.source_state_step = source_state_step

  def _create_directories(self):
    super(BisimulationRunner, self)._create_directories()
    self._visualize_dir = os.path.join(self._base_dir, 'visualization')
    tf.gfile.MakeDirs(self._visualize_dir)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(
        self._trained_agent_checkpoint_path)
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0

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
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes, average_reward = self._run_eval_phase(statistics)

    self._save_tensorboard_summaries(iteration, num_episodes, average_reward)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes: int, number of episodes run.
      average_reward: float, The average reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Play/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(tag='Play/AverageReturns',
                         simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)

  @gin.configurable
  def visualize(self, num_global_steps=2000):
    self._agent.eval_mode = True
    visualizer = agent_visualizer.AgentVisualizer(
        record_path=self._visualize_dir)
    global_step = 0
    while global_step < num_global_steps:
      initial_observation = self._environment.reset()
      action = self._agent.begin_episode(initial_observation)
      total_reward = 0.
      local_step = 0
      start_state = True
      while True:
        set_source_state = local_step == self.source_state_step
        observation, reward, is_terminal, _ = self._environment.step(action)
        total_reward += reward
        global_step += 1
        local_step += 1
        visualizer.visualize(self._environment, self._agent, start_state,
                             set_source_state)
        start_state = False
        if self._environment.game_over or global_step >= num_global_steps:
          break
        elif is_terminal:
          self._agent.end_episode(reward)
          action = self._agent.begin_episode(observation)
          start_state = True
        else:
          action = self._agent.step(reward, observation,
                                    set_source_state=set_source_state)
      self._end_episode(reward)
    visualizer.generate_video()
    sorted_dir = os.path.join(self._visualize_dir, 'sorted')
    tf.gfile.MakeDirs(sorted_dir)
    visualizer.print_sorted_frames(sorted_dir)
