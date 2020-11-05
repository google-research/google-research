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

"""Hooks for model pruning.

Model pruning hooks are used in estimators (instances of tf.estimator.Estimator)
to explicitly update the graph.
"""
import tensorflow.compat.v1 as tf


class ModelPruningListener(tf.estimator.CheckpointSaverListener):
  """Listener class for ModelPruningHook.

  Used for pruning python update functions that are run periodically.
  """

  def __init__(self, pruning_obj):
    """Initializer.

    Args:
      pruning_obj: Pruning object whose update function needs to be run.
    """
    self.pruning_obj = pruning_obj

  def before_save(self, session, global_step_value):
    """Before save processing."""
    # Disable all the protected-access violations in this function as
    # need to unfinalize the graph to call run_update_step.
    # pylint: disable=protected-access
    session.graph._unsafe_unfinalize()
    self.pruning_obj.run_update_step(session, global_step_value)


class ModelPruningHook(tf.estimator.SessionRunHook):
  """Prune the model every N steps."""

  _STEPS_PER_RUN = 1

  def __init__(self, every_steps=None, listeners=None):
    """Initialize a `ModelPruningHook`.

    Args:
      every_steps: `int`, prune every N steps.
      listeners: List of `ModelPruningListener` subclass instances.
    """
    tf.logging.info("Creating ModelPruningHook.")
    self._every_steps = every_steps
    self._listeners = listeners
    self._timer = tf.estimator.SecondOrStepTimer(every_steps=every_steps)

  def _call_prune_listener(self, session, step):
    """Calls model pruning listeners, return should_step_training."""
    tf.logging.info("Calling model pruning listeners at step %d...",
                    step)
    for listener in self._listeners:
      listener.before_save(session, step)

    should_stop_training = False
    for listener in self._listeners:
      if listener.after_save(session, step):
        tf.logging.info(
            "A model pruning listener requested that training be stopped. "
            "listener: {}".format(listener))
        should_stop_training = True
    return should_stop_training

  def begin(self):
    self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use ModelPruningHook.")
    for l in self._listeners:
      l.begin()

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)
    self._call_prune_listener(session, global_step)
    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.estimator.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results
    if not self._timer.should_trigger_for_step(stale_global_step +
                                               self._STEPS_PER_RUN):
      return

    # Get the real value after train op.
    global_step = run_context.session.run(self._global_step_tensor)
    if not self._timer.should_trigger_for_step(global_step):
      return

    self._timer.update_last_triggered_step(global_step)
    if self._call_prune_listener(run_context.session, global_step):
      run_context.request_stop()

  def end(self, session):
    last_step = session.run(self._global_step_tensor)
    if last_step != self._timer.last_triggered_step():
      self._call_prune_listener(session, last_step)
    for l in self._listeners:
      l.end(session, last_step)
