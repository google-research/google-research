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

"""Subclass for Learners and ThetaMods.

A Learner is a class that manages the process of inner-loop training.
It has some state, `LearnerState`, that represents the inner problem state.
All variables on a learner are local, with separate copies per worker.
The ThetaMod is a class that manages all tf variables for the learned optimizer
architecture and is shared across parameter servers.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import custom_getters
import py_utils
import sonnet as snt
import tensorflow.compat.v1 as tf
import tf_utils

nest = tf.contrib.framework.nest

BaseLearnerState = collections.namedtuple(
    "BaseLearnerState", ["phi_var_dict", "training_step", "initial_loss"])


def merged_namedtuple(cls, source, **kwargs):
  """Fill a namedtuple with elements from a different namedtuple and extra kwargs.

  Args:
    cls: namedtuple class Type that will be created.
    source: namedtuple Instance whose members are used for the new instance of
      cls.
    **kwargs: Extra arguments used to fill cls.

  Returns:
    A namedtuple of type cls
  """
  f = {}
  for field in source._fields:
    f[field] = getattr(source, field)
  for field in kwargs:
    f[field] = kwargs[field]
  return cls(**f)


class BaseThetaMod(snt.AbstractModule):
  """Base class for class containing all outer-variables."""

  def __init__(self, name="BaseThetaMod", device="", **kwargs):
    custom_getter = custom_getters.DynamicCustomGetter()
    super(BaseThetaMod, self).__init__(
        name=name, custom_getter=custom_getter, **kwargs)
    self.custom_getter = custom_getter
    self.device = device
    self()

  def _build(self):
    pass


class BaseLearner(snt.AbstractModule):
  """Base class."""

  def __init__(self, loss_module, theta_mod, name="BaseLearner"):
    super(BaseLearner, self).__init__(name=name)

    with self._enter_variable_scope():
      self.loss_module = loss_module

      with tf.control_dependencies(None):
        self.training_step = tf.get_variable(
            "training_step",
            initializer=tf.constant(0, dtype=tf.int32),
            trainable=False)

        self.initial_loss = tf.get_variable(
            "initial_loss",
            initializer=tf.constant(0, dtype=tf.float32),
            trainable=False)

      self.theta_mod = theta_mod

    self()

  def _build(self):
    pass

  @snt.reuse_variables
  def assign_state(self, state):
    # This also assigns the loss module's state.
    current = self.current_state()
    nest.assert_same_structure(current, state)
    current_flat = nest.flatten(current)
    state_flat = nest.flatten(state)
    assign_ops = [
        v.assign(s) for v, s in py_utils.eqzip(current_flat, state_flat)
    ]
    assign_ops.append(self.training_step.assign(state.training_step))
    return tf.group(assign_ops, name="assign_state")

  @snt.reuse_variables
  def meta_loss(self, state, loss_state=None):
    loss = self.loss_module.call_outer(state.phi_var_dict, state=loss_state)
    loss = tf.minimum(loss, 10.)
    return loss

  @snt.reuse_variables
  def inner_loss(self, state, loss_state=None):
    loss = self.loss_module.call_inner(state.phi_var_dict, state=loss_state)
    loss = tf.minimum(loss, 10.)
    return loss

  @snt.reuse_variables
  def meta_loss_state(self):
    return self.loss_module.get_outer_batch()

  @snt.reuse_variables
  def inner_loss_state(self):
    return self.loss_module.get_inner_batch()

  @snt.reuse_variables
  def _phi_variables(self):
    value_dict = self.loss_module.initial_state()
    return tf_utils.make_variables_matching(value_dict, trainable=True)

  @snt.reuse_variables
  def loss_and_next_state(self, current_state):
    raise NotImplementedError("Implement this")

  @snt.reuse_variables
  def initial_state(self):
    value_dict = self.loss_module.initial_state()
    initial_loss = self.loss_module.call_inner(value_dict)

    return BaseLearnerState(
        phi_var_dict=value_dict,
        training_step=tf.constant(0, dtype=tf.int32),
        initial_loss=initial_loss)

  @snt.reuse_variables
  def current_state(self):
    var_dict = self._phi_variables()
    return BaseLearnerState(
        phi_var_dict=var_dict,
        training_step=self.training_step,
        initial_loss=self.initial_loss,
    )
