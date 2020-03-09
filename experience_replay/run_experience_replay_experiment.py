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

"""Run a experience_replay experiment."""

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import run_experiment

import gin
import tensorflow.compat.v1 as tf

from experience_replay.agents import dqn_agent
from experience_replay.agents import rainbow_agent


@gin.configurable
def create_agent(sess,
                 environment,
                 agent_name=None,
                 summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent for
      in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.ElephantDQNAgent(
        sess=sess,
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.ElephantRainbowAgent(
        sess,
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
class ElephantRunner(run_experiment.Runner):
  """Extends the base Runner for every-n-step checkpoint writing."""

  def __init__(self,
               base_dir,
               create_agent_fn,
               checkpoint_every_n=1,
               **kwargs):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      checkpoint_every_n: int, the frequency for writing checkpoints.
      **kwargs:  key-word arguments to base-class Runner.
    """
    self._checkpoint_every_n = checkpoint_every_n
    run_experiment.Runner.__init__(self, base_dir, create_agent_fn, **kwargs)
    # pylint: disable=protected-access
    self._training_steps = int(self._training_steps *
                               self._agent._gin_param_multiplier)
    # pylint: enable=protected-access

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(
        self._checkpoint_dir,
        checkpoint_file_prefix,
        checkpoint_frequency=self._checkpoint_every_n)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    if iteration % self._checkpoint_every_n == 0:
      experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                          iteration)
      if experiment_data:
        experiment_data['current_iteration'] = iteration
        experiment_data['logs'] = self._logger.data
        self._checkpointer.save_checkpoint(iteration, experiment_data)
