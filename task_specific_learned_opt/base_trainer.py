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

"""Encapsulation around different training strategies.

These loosely construct everything needed to perform training.

See truncated_training.py for how this is used.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import gin
import sonnet as snt
import tensorflow.compat.v1 as tf

nest = tf.contrib.framework.nest

TruncatedTrainerEndpoints = collections.namedtuple(
    "TruncatedTrainerEndpoints", [
        "should_do_truncation_op", "worker_compute_op", "maybe_train_op",
        "init_learner_state_op", "run_on_reset_fn", "finished_truncation_op"
    ])


# The driver that runs all the computation
@gin.configurable
class TruncatedTrainerLoopRunner(object):
  """Manage outer-training involving truncations."""

  def __init__(
      self,
      truncated_trainer_endpoints,
      sess,
      steps_per_reset=100,
  ):
    self.truncated_trainer_endpoints = truncated_trainer_endpoints
    self.sess = sess
    self.steps_per_reset = steps_per_reset

  def worker_iterations(self, file_writer=None):
    """Return an iterator that performs iterations run on each worker.

    This iterator possibly initializers the Learner
    checks if we should run a truncation
    runs a single unroll for a worker
    attempts to train the parameters of the outer-optimizers.

    Args:
      file_writer: tensorflow filewriter
    Yields:
      None
    """

    steps_done = 0
    g = self.truncated_trainer_endpoints

    tf.logging.info("resetting learner state")
    self.sess.run(g.init_learner_state_op)

    while True:
      if self.sess.run(g.should_do_truncation_op):
        self.sess.run(g.worker_compute_op)
        self.sess.run(g.maybe_train_op)
        steps_done += 1
        yield
      else:
        tf.logging.info("finished_truncation_op")
        self.sess.run(g.finished_truncation_op)

        tf.logging.info("resetting learner state")
        self.sess.run(g.init_learner_state_op)

        g.run_on_reset_fn(self.sess, file_writer)

        if self.steps_per_reset is not None and steps_done > self.steps_per_reset:
          return
        else:
          yield


class TruncatedTrainer(snt.AbstractModule):
  """Abstract class that encapsulates a training algorithm.

  In addition to the methods, this also has a custom_getter instance that is
  passed into the theta_mod to do training strategies such as evolutionary
  strategies or parameter noise.
  """

  def __init__(
      self,
      learner=None,
      name="Trainer",
      local_device="",
      remote_device="",
      index_remote_device="",
      make_run_on_reset_fn=None,
  ):
    super(TruncatedTrainer, self).__init__(name=name)
    # TODO(lmetz) make this much more flexible / dynamic?
    assert learner is not None

    self.learner = learner

    self.index_remote_device = index_remote_device
    self.local_device = local_device
    self.remote_device = remote_device
    self.make_run_on_reset_fn = make_run_on_reset_fn

    self()

  def _build(self):
    pass

  def get_saved_remote_variables(self):
    return []

  def get_not_saved_remote_variables(self):
    return []

  def maybe_train_op(self):
    return tf.no_op()

  def worker_compute_op(self):
    return tf.no_op()

  def should_do_truncation_op(self):
    return tf.no_op()

  def ps_was_reset_op(self):
    return tf.no_op()

  def finished_truncation_op(self):
    return tf.no_op()

  def init_learner_state_op(self):
    return self.learner.assign_state(self.learner.initial_state())

  def get_local_variables(self):
    return []

  def build_endpoints(self):
    if self.make_run_on_reset_fn is None:
      run_on_reset_fn = lambda sess, filewriter: None
    else:
      run_on_reset_fn = self.make_run_on_reset_fn(self)

    return TruncatedTrainerEndpoints(
        should_do_truncation_op=self.should_do_truncation_op(),
        worker_compute_op=self.worker_compute_op(),
        maybe_train_op=self.maybe_train_op(),
        init_learner_state_op=self.init_learner_state_op(),
        # TODO(lmetz) remove this element?
        run_on_reset_fn=run_on_reset_fn,
        finished_truncation_op=self.finished_truncation_op(),
    )
