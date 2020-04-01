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

"""Utilities for outer-training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import learning_process
import py_utils
import tensorflow.compat.v1 as tf
import tf_utils

nest = tf.contrib.framework.nest


@gin.configurable
def compute_meta_loss(learner,
                      unroll_n_steps,
                      init_state=None,
                      extra_loss_eval=5):
  """Helper function to compute the training objective.

  This function unrolls `unroll_n_steps` and accumulates the loss.
  Additionally, to lower variance, at each new state, an extra extra_loss_eval
  losses are computed and added to the loss.

  TODO(lmetz) a more rigorous anylisis of variance of gradients to pick these
  parameters.

  Args:
    learner: Learner instance
    unroll_n_steps: number of steps to unroll
    init_state: initial LearnerState
    extra_loss_eval: int
  Returns:
    meta_loss, final LearnerState
  """
  if init_state is None:
    init_state = learner.current_state()
    init_state = tf_utils.force_copy(init_state)

  current_state = (tf.constant(0., dtype=tf.float32), init_state)
  loss_and_next_state_fn = lambda (l, state): learner.loss_and_next_state(state)

  def accumulate_fn((l, s), a):
    if extra_loss_eval > 0:
      cond = lambda i, a: tf.less(i, extra_loss_eval)
      body = lambda i, a: (i + 1, a + learner.meta_loss(s))
      _, extra_losses = tf.while_loop(cond, body, loop_vars=[0, 0.])
      return a + extra_losses
    else:
      return a + l

  (_, final_state), training_loss = learning_process.fold_learning_process(
      unroll_n_steps,
      loss_and_next_state_fn,
      accumulate_fn=accumulate_fn,
      start_state=current_state,
      accumulator_start_state=tf.constant(0., dtype=tf.float32),
  )

  meta_loss = (training_loss) / (
      tf.to_float(unroll_n_steps) * tf.to_float(extra_loss_eval))

  return tf.identity(meta_loss), nest.map_structure(tf.identity, final_state)


def make_push_op(learner, ds, failed_push, should_push, to_push,
                 final_state, pre_step_index):
  """Helper that make the op that pushes gradients, and assigns next state."""
  # This is what pushes gradient tensors to a shared location.
  push = lambda: ds.push_tensors(to_push, pre_step_index)

  def fail_push():
    pop = tf.Print(failed_push, [failed_push], "Failed to push")
    return tf.group(failed_push.assign_add(1), pop, name="fail_push")

  push_gradients_op = tf.cond(should_push, push, fail_push)

  pre_assign = tf.group(push_gradients_op, name="push_gradient_op")
  with tf.control_dependencies([pre_assign]):
    worker_compute_op = learner.assign_state(final_state)

  return worker_compute_op


def clip_grads_vars(grads_vars, clip_by_value):
  grads_vars = [(tf.clip_by_value(g, -clip_by_value, clip_by_value), v)
                for g, v in grads_vars]
  return grads_vars


def assert_grads_vars_not_nan(grads_vars):
  ops = [
      tf.check_numerics(g, "grads_vars check_numerics: [%s]" % v.name)
      for g, v in grads_vars
  ]
  with tf.control_dependencies(ops):
    return [(tf.identity(g), v) for g, v in grads_vars]


def assert_post_update_not_nan(grads_vars):
  ops = [
      tf.check_numerics(v, "post_update check_numerics: [%s]" % v.name)
      for _, v in grads_vars
  ]
  return tf.group(ops)


def check_grads_finite(grads):
  if not len(grads):  # pylint: disable=g-explicit-length-test
    return tf.constant(True)
  else:
    finites = [tf.reduce_sum(1 - tf.to_float(tf.is_finite(g))) for g in grads]
    return tf.equal(tf.add_n(finites), 0.)


def merge_into(s1, s2, keys):
  """Merge 2 namedtuples. Copy keys from s2 into s1."""
  dd = {}
  for k in s1._fields:
    dd[k] = getattr(s1, k)

  for k in keys:
    dd[k] = getattr(s2, k)
  return s1.__class__(**dd)


def merge_field_into(s1, key, value):
  """Create a new namedtuple that is a copy of s1 with the key,value replaced."""
  dd = {}
  for k in s1._fields:
    dd[k] = getattr(s1, k)
  dd[key] = value
  return s1.__class__(**dd)


def average_dicts(d1, d2, ratio):
  """Average 2 dictionaries d1 and d2.

  ratio=0 means all d1, and ratio=1 is all d2

  Args:
    d1: dict
    d2: dict
    ratio: float
  Returns:
    OrderedDict
  """
  dd = []
  assert set(d1.keys()) == set(d2.keys())
  for key in d1:
    dd.append((key, (1. - ratio) * d1[key] + (ratio) * d2[key]))
  return collections.OrderedDict(dd)


# TODO(lmetz) this is horribly complex... Should be shifted
# to the inner loop calculations done outside -- pass in fn to eval meta_loss.
# and other aggregators.
@gin.configurable
class DeterministicMetaLossEvaluator(object):
  """Compute the meta loss, but using the same batches of data for each call.

  in the same session run.

  This is used for things like ES where shared random numbers reduce gradient
  variance.
  """

  def __init__(self,
               meta_loss_state_fn,
               inner_loss_state_fn,
               unroll_n_steps,
               meta_loss_evals=5):
    # Compute 1 nested structure of tensor array for the inner loop data
    # and a second for the extra evaluations.

    # Fill a tensor array with all the batches requested, then use these instead
    # of resampling.

    self.meta_loss_evals = meta_loss_evals
    self.unroll_n_steps = unroll_n_steps

    def fill_batches(size, state_fn):
      """Fill a tensor array with batches of data."""
      dummy_batch = state_fn()
      tas = nest.map_structure(
          lambda b: tf.TensorArray(b.dtype, size=size, clear_after_read=False),
          dummy_batch)

      cond = lambda i, ta: tf.less(i, size)

      def body(i, tas):
        batch = state_fn()
        out_tas = []
        for ta, b in py_utils.eqzip(nest.flatten(tas), nest.flatten(batch)):
          out_tas.append(ta.write(i, b))
        return (i + 1, nest.pack_sequence_as(dummy_batch, out_tas))

      _, batches = tf.while_loop(cond, body,
                                 [tf.constant(0, dtype=tf.int32), tas])
      return batches

    self.meta_batches = fill_batches(meta_loss_evals * unroll_n_steps,
                                     meta_loss_state_fn)

    self.inner_batches_eval = fill_batches(meta_loss_evals * unroll_n_steps,
                                           inner_loss_state_fn)

    self.inner_batches = fill_batches(unroll_n_steps, inner_loss_state_fn)

  def get_batch(self, idx, batches):
    return nest.map_structure(lambda ta: ta.read(idx), batches)

  def get_meta_batch_state(self):

    def _fn(x):
      return x.stack()

    return nest.map_structure(_fn, self.meta_batches)

  def get_inner_batch_state(self):

    def _fn(x):
      return x.stack()

    return nest.map_structure(_fn, self.inner_batches)

  def _nest_bimap(self, fn, data1, data2):
    data = py_utils.eqzip(nest.flatten(data1), nest.flatten(data2))
    out = [fn(*a) for a in data]
    return nest.pack_sequence_as(data1, out)

  def get_phi_trajectory(self, learner, inner_batches=None, init_state=None):
    """Compute a inner-parameter trajectory."""
    if inner_batches is None:
      inner_batches = self.inner_batches
    else:
      # convert the batches object to a tensorarray.
      def to_ta(t):
        return tf.TensorArray(
            dtype=t.dtype, size=self.unroll_n_steps).unstack(t)

      inner_batches = nest.map_structure(to_ta, inner_batches)

    if init_state is None:
      init_state = learner.current_state()
      init_state = tf_utils.force_copy(init_state)

    def body(learner_state, ta, i):
      batch = self.get_batch(i, batches=inner_batches)
      _, next_state = learner.loss_and_next_state(
          learner_state, loss_state=batch)

      # shift because this is the next state.
      next_ta = self._nest_bimap(lambda t, v: t.write(i + 1, v), ta, next_state)
      return next_state, next_ta, i + 1

    def cond(learner_state, ta, i):  # pylint: disable=unused-argument
      return tf.less(i, self.unroll_n_steps)

    def make_ta(x):
      ta = tf.TensorArray(dtype=x.dtype, size=self.unroll_n_steps + 1)
      return ta.write(0, x)

    _, ta, _ = tf.while_loop(
        cond, body,
        [init_state,
         nest.map_structure(make_ta, init_state),
         tf.constant(0)])
    return nest.map_structure(lambda x: x.stack(), ta)

  def get_avg_loss(self, learner, inner_batches, init_state):
    """Compute average loss for unroll."""
    if inner_batches is None:
      inner_batches = self.inner_batches
    else:
      # convert the batches object to a tensorarray.
      def to_ta(t):
        return tf.TensorArray(
            dtype=t.dtype, size=self.unroll_n_steps).unstack(t)

      inner_batches = nest.map_structure(to_ta, inner_batches)

    if init_state is None:
      init_state = learner.current_state()
      init_state = tf_utils.force_copy(init_state)

    def body(a, i):
      batch = self.get_batch(i * self.batches_per_step, batches=inner_batches)
      l = learner.meta_loss(init_state, loss_state=batch)
      return a + l, i + 1

    def cond(_, i):
      return tf.less(i, self.unroll_n_steps)

    a, _ = tf.while_loop(cond, body, [tf.constant(0.0), tf.constant(0)])
    return a / self.unroll_n_steps

  def __call__(self,
               learner,
               meta_batches=None,
               inner_batches=None,
               init_state=None,
               unroll_n_steps=None):
    if unroll_n_steps is None:
      unroll_n_steps = self.unroll_n_steps
    else:
      print("Using passed in unroll steps")

    if inner_batches is None:
      inner_batches = self.inner_batches
    else:
      # convert the batches object to a tensorarray.
      def to_ta(t):
        return tf.TensorArray(
            dtype=t.dtype, size=self.unroll_n_steps).unstack(t)

      inner_batches = nest.map_structure(to_ta, inner_batches)

    if meta_batches is None:
      meta_batches = self.meta_batches
    else:
      # convert the batches object to a tensorarray.
      def ml_to_ta(t):
        return tf.TensorArray(
            dtype=t.dtype,
            size=self.meta_loss_evals * self.unroll_n_steps).unstack(t)

      meta_batches = nest.map_structure(ml_to_ta, meta_batches)

    if init_state is None:
      init_state = learner.current_state()
      init_state = tf_utils.force_copy(init_state)

    current_state = (tf.constant(0, dtype=tf.int32),
                     tf.constant(0., dtype=tf.float32), init_state)

    def loss_and_next_state_fn((idx, l, state)):
      batch = self.get_batch(idx, batches=inner_batches)
      l, s = learner.loss_and_next_state(state, loss_state=batch)
      return (idx + 1, l, s)

    def accumulate_fn((idx, _, s), (a_meta, a_inner)):
      """Accumulate loss for fold learning process."""
      cond = lambda i, a: tf.less(i, self.meta_loss_evals)

      def body_meta(i, a):
        # minus 1 as this takes the following step.
        batch = self.get_batch(
            (idx - 1) * (self.meta_loss_evals) + i, batches=meta_batches)
        return (i + 1, a + learner.meta_loss(s, loss_state=batch))

      _, extra_losses = tf.while_loop(cond, body_meta, loop_vars=[0, 0.])

      def body_inner(i, a):
        # minus 1 as this takes the following step.
        batch = self.get_batch(
            (idx - 1) * (self.meta_loss_evals) + i, batches=meta_batches)
        return (i + 1, a + learner.inner_loss(s, loss_state=batch))

      _, inner_losses = tf.while_loop(cond, body_inner, loop_vars=[0, 0.])

      return a_meta + extra_losses, a_inner + inner_losses

    (_, _,
     final_state), (meta_loss_sum,
                    _) = learning_process.fold_learning_process(
                        unroll_n_steps,  # Note this is not self version.
                        loss_and_next_state_fn,
                        accumulate_fn=accumulate_fn,
                        start_state=current_state,
                        accumulator_start_state=(tf.constant(
                            0.,
                            dtype=tf.float32), tf.constant(
                                0., dtype=tf.float32)))

    # TODO(lmetz) this should be shifted to compute loss for all but shifted 1
    meta_loss = (meta_loss_sum) / tf.to_float(unroll_n_steps) / tf.to_float(
        self.meta_loss_evals)

    return meta_loss, final_state
