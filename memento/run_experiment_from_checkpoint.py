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

"""Module defining classes and helper methods for general agents."""
import os

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment

import gin
import numpy as np
import tensorflow as tf


@gin.configurable
def create_runner_checkpoint(base_dir,
                             create_agent=run_experiment.create_agent,
                             schedule='load_from_best'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    create_agent: function, function used to create the agent.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None

  # used to train original agent, and keep track of best states
  if schedule == 'save_best':
    return SaveBestRunner(base_dir, create_agent)
  # used to train Memento agent
  elif schedule == 'load_from_best':
    return LoadFromRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class SaveBestRunner(run_experiment.Runner):
  """The SaveBestRunner which adds saving functionality.

  In addition to training and evaluating the agent, this runner
  also keeps track of the state in each trajectory that has the highest
  return so far. This data is saved and stored in a numpy array file.
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the SaveBestRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    tf.logging.info('Creating SaveBestRunner ...')
    super(SaveBestRunner, self).__init__(base_dir, create_agent_fn,
                                         create_environment_fn)

    self._max_returns = []
    self._best_states = []

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False
    max_return = 0.
    best_state = self._environment.environment.clone_state()
    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Keep track of the state with highest return till date
      if total_reward > max_return and not is_terminal:
        max_return = total_reward
        best_state = self._environment.environment.clone_state()

      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward)

    self._max_returns.append(max_return)
    self._best_states.append(best_state)

    return step_number, total_reward

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    super(SaveBestRunner, self)._checkpoint_experiment(iteration)

    # Checkpoint best states seen so far to best_states.npz
    best_state_data = dict(
        score=np.stack(self._max_returns),
        state=np.stack(self._best_states),
    )
    with tf.gfile.Open(os.path.join(self._base_dir, 'best_states.npz'),
                       'wb') as f:
      np.savez(f, **best_state_data)

    # Start fresh for new iteration
    self._max_returns = []
    self._best_states = []


@gin.configurable
class LoadFromRunner(run_experiment.Runner):
  """The Memento Runner.

  This runner initializes a Memento agent, which starts at the best state
  seen by the original agent, and makes further progress.
  """

  def __init__(
      self,
      base_dir,
      create_agent_fn,
      create_environment_fn=atari_lib.create_atari_environment,
      original_base_dir=None,
      original_agent_iteration_number=199,
      load_from_many=False,
      original_agent_weights=True,
  ):
    """Initialize the Runner object in charge of running the Memento agent.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      original_base_dir: str, the base directory used for the original agent.
      original_agent_iteration_number: Final iteration number for original
        agent.
      load_from_many: bool, if True, the initial state is stochasticly sampled
        from the set of Memento states saved in best_states.npz.
      original_agent_weights: bool, if True, value function parameters are
        initialized with original agent parameters.
    """
    tf.logging.info('Creating LoadFromRunner ...')
    tf.logging.info('\t Iteration: %d', original_agent_iteration_number)
    tf.logging.info('\t Load from Many: %s', load_from_many)
    tf.logging.info('\t Initialize Value Function: %s', original_agent_weights)

    assert original_base_dir is not None, ('Must pass in original agent '
                                           'directory')
    self.original_base_dir = original_base_dir

    self.original_agent_weights = original_agent_weights
    self.original_agent_iteration_number = original_agent_iteration_number

    super(LoadFromRunner, self).__init__(base_dir, create_agent_fn,
                                         create_environment_fn)

    state_file_name = os.path.join(self.original_base_dir, 'best_states.npz')
    tf.logging.info('Loading best states from {}'.format(state_file_name))
    self.load_from_many = load_from_many

    with tf.gfile.Open(state_file_name, 'rb') as f:
      data = np.load(f)
      self.starting_returns = data['score']
      self.starting_states = data['state']
      if self.load_from_many:
        tf.logging.info('Starting from Average Score of {}'.format(
            np.mean(self.starting_returns)))
      else:
        tf.logging.info('Starting from Average Score of {}'.format(
            self.starting_returns[0]))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will initialize the value function parameters with the
    parameters from the original agent, if the `original_agent_weights`
    flag is True.

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number from which to start the
        experiment (typically set at iteration 199).
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    super(
        LoadFromRunner,
        self)._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    if self._start_iteration == 0 and self.original_agent_weights:
      original_checkpoint_dir = os.path.join(self.original_base_dir,
                                             'checkpoints')
      # pylint: disable=protected-access
      self._agent._saver.restore(
          self._sess,
          os.path.join(
              original_checkpoint_dir,
              'tf_ckpt-{}'.format(self.original_agent_iteration_number)))
      # pylint: enable=protected-access
      tf.logging.info('Loaded agent from {}'.format(
          os.path.join(
              original_checkpoint_dir,
              'tf_ckpt-{}'.format(self.original_agent_iteration_number))))

  def _initialize_episode(self, index=0):
    """Initialization for a new episode.

    Args:
      index: int, chooses the state to which the environment is reset from the
        available states.  Default to 0 for the first state.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    _ = self._environment.reset()
    self._environment.environment.restore_state(self.starting_states[index])
    # pylint: disable=protected-access
    initial_observation = self._environment._pool_and_resize()
    # pylint: enable=protected-access
    return self._agent.begin_episode(initial_observation)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    if self.load_from_many:
      index = np.random.choice(len(self.starting_returns))
    else:
      index = 0

    total_reward = float(self.starting_returns[index])
    action = self._initialize_episode(index)
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
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward)
    return step_number, total_reward
