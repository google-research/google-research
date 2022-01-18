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

"""Agent module for learning policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import utils


class Agent(object):
  """Class for learning policy and interacting with environment."""

  def __init__(
      self,
      observation_spec=None,
      action_spec=None,
      time_step_spec=None,
      modules=None,
      optimizers=(('adam', 0.001),),
      batch_size=64,
      weight_decays=(0.0,),
      update_freq=1,
      update_rate=0.005,
      discount=0.99,
      train_data=None,
      ):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._time_step_spec = time_step_spec
    self._modules = modules
    self._optimizers = optimizers
    self._batch_size = batch_size
    self._weight_decays = weight_decays
    self._train_data = train_data
    self._update_freq = update_freq
    self._update_rate = update_rate
    self._discount = discount
    self._build_agent()

  def _build_agent(self):
    """Builds agent components."""
    self._build_fns()
    self._build_optimizers()
    self._global_step = tf.Variable(0)
    self._train_info = collections.OrderedDict()
    self._checkpointer = self._build_checkpointer()
    self._test_policies = collections.OrderedDict()
    self._build_test_policies()
    self._online_policy = self._build_online_policy()
    train_batch = self._get_train_batch()
    self._init_vars(train_batch)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)

  def _get_vars(self):
    return []

  def _build_optimizers(self):
    opt = self._optimizers[0]
    opt_fn = utils.get_optimizer(opt[0])
    self._optimizer = opt_fn(lr=opt[1])

  def _build_loss(self, batch):
    raise NotImplementedError

  def _build_checkpointer(self):
    return tf.train.Checkpoint(
        agent=self._agent_module,
        global_step=self._global_step,
        )

  def _build_test_policies(self):
    raise NotImplementedError

  def _build_online_policy(self):
    return None

  @property
  def test_policies(self):
    return self._test_policies

  @property
  def online_policy(self):
    return self._online_policy

  def _get_train_batch(self):
    """Samples and constructs batch of transitions."""
    batch_indices = np.random.choice(self._train_data.size, self._batch_size)
    batch_ = self._train_data.get_batch(batch_indices)
    transition_batch = batch_
    batch = dict(
        s1=transition_batch.s1,
        s2=transition_batch.s2,
        r=transition_batch.reward,
        dsc=transition_batch.discount,
        a1=transition_batch.a1,
        a2=transition_batch.a2,
        )
    return batch

  def _optimize_step(self, batch):
    with tf.GradientTape() as tape:
      loss, info = self._build_loss(batch)
    trainable_vars = self._get_vars()
    grads = tape.gradient(loss, trainable_vars)
    grads_and_vars = tuple(zip(grads, trainable_vars))
    self._optimizer.apply_gradients(grads_and_vars)
    return info

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for key, val in info.items():
      self._train_info[key] = val.numpy()
    self._global_step.assign_add(1)

  def _init_vars(self, batch):
    pass

  def _get_source_target_vars(self):
    return [], []

  def _update_target_fns(self, source_vars, target_vars):
    utils.soft_variables_update(
        source_vars,
        target_vars,
        tau=self._update_rate)

  def print_train_info(self):
    info = self._train_info
    step = self._global_step.numpy()
    summary_str = utils.get_summary_str(step, info)
    logging.info(summary_str)

  def write_train_summary(self, summary_writer):
    info = self._train_info
    step = self._global_step.numpy()
    utils.write_summary(summary_writer, step, info)

  def save(self, ckpt_name):
    self._checkpointer.write(ckpt_name)

  def restore(self, ckpt_name):
    self._checkpointer.restore(ckpt_name)

  @property
  def global_step(self):
    return self._global_step.numpy()


class AgentModule(tf.Module):
  """Tensorflow module for agent."""

  def __init__(
      self,
      modules=None,
      ):
    super(AgentModule, self).__init__()
    self._modules = modules
    self._build_modules()

  def _build_modules(self):
    pass


class Config(object):
  """Class for handling agent parameters."""

  def __init__(self, agent_flags):
    self._agent_flags = agent_flags
    self._agent_args = self._get_agent_args()

  def _get_agent_args(self):
    """Gets agent parameters associated with config."""
    agent_flags = self._agent_flags
    agent_args = utils.Flags(
        action_spec=agent_flags.action_spec,
        optimizers=agent_flags.optimizers,
        batch_size=agent_flags.batch_size,
        weight_decays=agent_flags.weight_decays,
        update_rate=agent_flags.update_rate,
        update_freq=agent_flags.update_freq,
        discount=agent_flags.discount,
        train_data=agent_flags.train_data,
        )
    agent_args.modules = self._get_modules()
    return agent_args

  def _get_modules(self):
    raise NotImplementedError

  @property
  def agent_args(self):
    return self._agent_args
