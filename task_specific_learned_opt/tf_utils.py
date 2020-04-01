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

"""Data Structures in TF to manage state."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time
import numpy as np
import py_utils
import tensorflow.compat.v1 as tf

nest = tf.contrib.framework.nest


def assign_variables(targets, values):
  """Creates an Op that assigns a list of values to a target variables.

  Args:
    targets: list or structure of tf.Variable
    values: list or structure of tf.Tensor

  Returns:
    tf.Operation that performs the assignment
  """
  return tf.group(
      *itertools.starmap(
          tf.assign, py_utils.eqzip(
              nest.flatten(targets), nest.flatten(values))),
      name="assign_variables")


def make_variables_matching(
    state,
    trainable=False,
    name="make_variables_matching",
    use_get_variable=True,
):
  """Create variables of the same shape as the tensors in shape.

  These variables can be used to store state from truncated step to truncated
  step.
  This is useful for any sequential state being used for truncated backprop.

  Args:
    state: namedtuple/list/tuple of tf.Tensor the tensors to convert to
      variables
    trainable: bool argument set on the new variable
    name: str
    use_get_variable: bool to use tf.get_variable or tf.Variable

  Returns:
    the same structure as state, but with tf.Variable.
  """

  def _get_var_like_tensor(t, i):
    # NOTE(lmetz) don't use the names here, as those can change.
    full_name = "%s_%d" % (name, i)
    if use_get_variable:
      return tf.get_variable(
          name=full_name, shape=t.shape, dtype=t.dtype, trainable=trainable)
    else:
      return tf.Variable(
          name=full_name,
          initial_value=tf.zeros(shape=t.shape, dtype=t.dtype),
          trainable=trainable)

  flat = [_get_var_like_tensor(t, i) for i, t in enumerate(nest.flatten(state))]

  return nest.pack_sequence_as(flat_sequence=flat, structure=state)


class _PyStepsPerSecond(object):
  """Measure iterations/steps per second."""

  def __init__(self, history_length=10):
    """Python state used to keep track of steps per second.

    Args:
      history_length: int length of history to do running mean over
    """

    self.last_time = None
    self.last_step = None
    self.history = collections.deque(maxlen=history_length)

  def get_steps_per_second(self, global_step):
    if self.last_step is not None:
      self.history.append(
          float(global_step - self.last_step) / (time.time() - self.last_time))
    self.last_step = global_step
    self.last_time = time.time()
    if self.history:
      return np.asarray(np.mean(self.history), dtype=np.float32)
    return np.zeros([], dtype=np.float32)


_PyStepStates = {}  # pylint: disable=invalid-name


def steps_per_second(step, name="shared_state"):
  """Creates summaries for counting steps per second.

  Note: this must run on a python instance (not a PS) as it uses pyfunc.

  Args:
    step: tensor to monitor. Normally global step.
    name: str Value used to share history when in eager mode. has no effect in
      graph mode.

  Returns:
    steps per second tf.Tensor
  """
  with tf.control_dependencies(None):
    # in eager mode, this gets called over and over again.
    # need to cache the python state.
    if tf.executing_eagerly():
      if name not in _PyStepStates:
        _PyStepStates[name] = _PyStepsPerSecond()
      ss = _PyStepStates[name]
    else:
      ss = _PyStepsPerSecond()

    steps_per_second_tensor = tf.py_func(ss.get_steps_per_second, [step],
                                         [tf.float32])
    if isinstance(steps_per_second_tensor, list):
      steps_per_second_tensor = steps_per_second_tensor[0]

    steps_per_second_tensor.set_shape([])
    return steps_per_second_tensor


def force_copy(structure):
  """Force copy tensors into memory.

  See: https://github.com/tensorflow/tensorflow/issues/11186
  Args:
    structure: tf.Tensor or nest-able datastructure

  Returns:
    structure: Same structure but coppied into host memory.
  """
  with tf.name_scope("force_copy"):

    def copy(x):
      """Force copy a tensorflow tensor."""
      # Do this regardless of if it is a var or a tensor.
      # if it is a tensor, it could just be a tf.identity of a var and have
      # a similar problem.
      if x.dtype.is_floating:
        return x * 1.0
      elif x.dtype.is_integer:
        return x * 1
      elif x.dtype.is_bool:
        return tf.equal(x, True)
      else:
        raise ValueError("dtype [%s] not implemented" % str(x.dtype))

    all_ops = nest.map_structure(copy, structure)
    with tf.control_dependencies(nest.flatten(all_ops)):
      return nest.map_structure(tf.identity, all_ops)
