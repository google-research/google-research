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

"""Session creators for tf.train.MonitoredSession."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf


# Need to initialize local variables or TF will hang on initialization
# The meta opt / global variables will be initialized for us.
# There is no SessionCreator that supports this naively, so I am hacking
# in a reset op into the check for ready
# I don't know what will happen in the case that i can't just span this
# init function (as is the case with random initializations).
# This should be fixed with a new session creator that
# calls an init op,
# then waits for remote variables to be initialized.
class WorkerSessionCreator(tf.train.WorkerSessionCreator):
  """A session creator for tf.train.MonitoredSession for our clusters.

  See tf.train.WorkerSessionCreator for more information.
  Do NOT pass a scaffold in,

  Args:
    remote_vars: list of tf.Variable list of variables located on remote (PS)
      devices.
    local_vars: list of tf.Variable list of variables located on the local
      device. There is one copy of these variables per worker and are stored on
      the worker.
  """

  def __init__(self, remote_vars, local_vars, *args, **kwargs):
    if local_vars:
      self.has_init = tf.is_variable_initialized(local_vars[0])
    else:
      self.has_init = tf.constant(True)

    ready_for_local_init_op = tf.report_uninitialized_variables(
        var_list=remote_vars)

    self.init_op = tf.group(
        tf.initialize_variables(local_vars),
        *tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))

    if "scaffold" in kwargs:
      # TODO(lmetz) I think this could technically be supported?
      raise ValueError("Do not set scaffold on the session creator.")

    scaffold = tf.train.Scaffold(
        ready_for_local_init_op=ready_for_local_init_op,
        ready_op=ready_for_local_init_op,
        local_init_op=ready_for_local_init_op,
        init_op=ready_for_local_init_op,
        init_fn=self._maybe_initialize_local_vars_and_state_fn,
        summary_op=tf.summary.merge([tf.summary.scalar("dummy", 0)]))

    kwargs["scaffold"] = scaffold
    super(WorkerSessionCreator, self).__init__(*args, **kwargs)

  # Sadly, this must be done in python control flow and passed in as a function
  # Tensorflow provides no way to conditionally reset variables (like with cond)
  def _maybe_initialize_local_vars_and_state_fn(self, scaffold, sess):
    if not sess.run(self.has_init):
      sess.run(self.init_op)

  def create_session(self):
    self._scaffold.finalize()
    session = self._get_session_manager().wait_for_session(
        self._master,
        config=self._config,
        max_wait_secs=30 * 60  # Wait up to 30 mins for the session to be ready.
    )
    self._maybe_initialize_local_vars_and_state_fn(self._scaffold, session)
    return session


class ChiefSessionCreator(tf.train.ChiefSessionCreator):
  """Session creator for chief workers.

  Identical to tf.train.ChiefSessionCreator except summaries are turned off
  by default.
  """

  def __init__(self, *args, **kwargs):
    no_summary_op = tf.summary.merge([tf.summary.scalar("dummy", 0.)])
    scaffold = tf.train.Scaffold(summary_op=no_summary_op)

    if "scaffold" in kwargs:
      # TODO(lmetz) this could technically be supported.
      raise ValueError("Do not set scaffold on the session creator")

    kwargs["scaffold"] = scaffold
    super(ChiefSessionCreator, self).__init__(*args, **kwargs)
