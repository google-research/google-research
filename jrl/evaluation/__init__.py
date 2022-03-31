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

"""Slightly customized Acme Evaluator.
"""

from absl import logging

import abc
import signal
import sys
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import dm_env

import acme
from acme import core
from acme import environment_loop
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib


def _average_nested_dicts(list_of_dicts):
  """Average an iterable of nested dicts with same keys.

  Args:
    list_of_dicts: an iterable of nested dicts.
  Returns:
    A dict with same keys as the input dicts and average values among list.
  """
  if not list_of_dicts:
    return {}
  result = {}
  for key, value in list_of_dicts[0].items():
    if not isinstance(value, dict):
      result[key] = np.mean([x[key] for x in list_of_dicts])
    else:
      result[key] = _average_nested_dicts([x[key] for x in list_of_dicts])
  return result


class Evaluator(core.Worker):
  """A base evaluator."""

  @abc.abstractmethod
  def run_once(self):
    """Runs one chunk of evaluation."""

  def run(self):
    while True:
      self.run_once()


class EvaluatorStandard(Evaluator):
  """An evaluator logging the average return over num_episodes."""

  def __init__(
      self,
      eval_actor,
      environment,
      num_episodes,
      counter,
      logger,
      eval_sync = None,
      progress_counter_name = 'actor_steps',
      min_steps_between_evals = None,
      self_cleanup = False,
      observers = (),
  ):
    super().__init__()
    assert num_episodes >= 1
    self._eval_actor = eval_actor
    self._num_episodes = num_episodes
    self._counter = counter
    self._logger = logger
    # Create the run loop and return it.
    self._env = environment
    self._environment_loop = environment_loop.EnvironmentLoop(
        environment, eval_actor, should_update=False, observers=observers)
    self._eval_sync = eval_sync or (lambda _: None)
    self._progress_counter_name = progress_counter_name
    self._last_steps = None
    self._eval_every_steps = min_steps_between_evals

    self._pending_tear_down = False
    if self_cleanup:
      # Do not rely on the instance owner to cleanup this evaluator.
      # Register a signal handler to perform some resource cleanup.
      try:
        signal.signal(signal.SIGTERM, self._signal_handler)  # pytype: disable=wrong-arg-types
      except ValueError:
        logging.warning(
            'Caught ValueError when registering signal handler. '
            'This probably means we are not running in the main thread. ')

  def run_once(self):
    if self._pending_tear_down:
      self.tear_down()
      sys.exit()

    if self._eval_every_steps is not None:
      counts = self._counter.get_counts()
      current_steps = counts.get(self._progress_counter_name, 0)
      if (self._last_steps is not None and
          current_steps < self._last_steps + self._eval_every_steps):
        time.sleep(0.1)
        return
      self._last_steps = current_steps

    counts = self._counter.increment(steps=1)
    current_steps = counts.get(self._progress_counter_name, 0)
    self._eval_actor.update(wait=True)
    self._eval_sync(current_steps)
    results = []
    for _ in range(self._num_episodes):
      if self._pending_tear_down:
        self.tear_down()
        sys.exit()
      results.append(self._environment_loop.run_episode())

    results = _average_nested_dicts(results)
    self._logger.write({**results, **counts})

  def _signal_handler(self, signum, frame):
    del signum, frame
    logging.info('Caught SIGTERM: cleaning up the evaluator.')
    # We defer the exit so that evaluation is performed on full episodes only
    # and not on partial episodes.
    self._pending_tear_down = True

  def tear_down(self):
    if self._env is not None:
      self._env.close()
      self._env = None


class EvaluatorStandardWithFinalRewardLogging(EvaluatorStandard):
  def run_once(self):
    if self._pending_tear_down:
      self.tear_down()
      sys.exit()

    if self._eval_every_steps is not None:
      counts = self._counter.get_counts()
      current_steps = counts.get(self._progress_counter_name, 0)
      if (self._last_steps is not None and
          current_steps < self._last_steps + self._eval_every_steps):
        time.sleep(0.1)
        return
      self._last_steps = current_steps

    counts = self._counter.increment(steps=1)
    current_steps = counts.get(self._progress_counter_name, 0)
    self._eval_actor.update(wait=True)
    self._eval_sync(current_steps)
    results = []
    for _ in range(self._num_episodes):
      if self._pending_tear_down:
        self.tear_down()
        sys.exit()
      cur_result = self._environment_loop.run_episode()
      if hasattr(self._environment_loop._environment, '_compute_reward'):
        cur_result['final_reward'] = self._environment_loop._environment._compute_reward()
      results.append(cur_result)

    episode_returns = []
    for result in results:
      episode_returns.append(result['episode_return'])
    episode_return_std = np.std(episode_returns)

    results = _average_nested_dicts(results)
    results['episode_return_std'] = episode_return_std
    self._logger.write({**results, **counts})
