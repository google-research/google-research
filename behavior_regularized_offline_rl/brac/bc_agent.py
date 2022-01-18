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

"""Behavior cloning via maximum likelihood."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import agent
from behavior_regularized_offline_rl.brac import networks
from behavior_regularized_offline_rl.brac import policies
from behavior_regularized_offline_rl.brac import utils


ALPHA_MAX = 500.0
CLIP_EPS = 1e-3


@gin.configurable
class Agent(agent.Agent):
  """Behavior cloning agent."""

  def __init__(
      self,
      train_alpha_entropy=True,
      alpha_entropy=1.0,
      target_entropy=None,
      **kwargs):
    self._train_alpha_entropy = train_alpha_entropy
    self._alpha_entropy = alpha_entropy
    self._target_entropy = target_entropy
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    self._p_fn = self._agent_module.p_fn
    self._get_log_density = self._agent_module.p_net.get_log_density
    if self._target_entropy is None:
      self._target_entropy = - self._action_spec.shape[0]
    self._get_alpha_entropy = self._agent_module.get_alpha_entropy
    self._agent_module.assign_alpha_entropy(self._alpha_entropy)

  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_p_weight_norm(self):
    weights = self._agent_module.p_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _build_p_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
    log_pi_a_b = self._get_log_density(s, a_b)
    _, _, log_pi_a_p = self._p_fn(s)
    p_loss = tf.reduce_mean(
        self._get_alpha_entropy() * log_pi_a_p
        - log_pi_a_b)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[0] * p_w_norm
    loss = p_loss + norm_loss
    # Construct information about current training.
    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm
    return loss, info

  def _build_ae_loss(self, batch):
    s = batch['s1']
    _, _, log_pi_a = self._p_fn(s)
    alpha = self._get_alpha_entropy()
    ae_loss = tf.reduce_mean(alpha * (- log_pi_a - self._target_entropy))
    # Construct information about current training.
    info = collections.OrderedDict()
    info['ae_loss'] = ae_loss
    info['alpha_entropy'] = alpha
    return ae_loss, info

  def _build_optimizers(self):
    opts = self._optimizers
    if not opts:
      raise ValueError('No optimizers provided.')
    if len(opts) == 1:
      opts = tuple([opts[0]] * 2)
    self._p_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    self._ae_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])

  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    p_info = self._optimize_p(batch)
    if self._train_alpha_entropy:
      ae_info = self._optimize_ae(batch)
    info.update(p_info)
    if self._train_alpha_entropy:
      info.update(ae_info)
    return info

  def _optimize_p(self, batch):
    vars_ = self._p_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_p_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._p_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_ae(self, batch):
    vars_ = self._ae_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_ae_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._ae_optimizer.apply_gradients(grads_and_vars)
    return info

  def _build_test_policies(self):
    policy = policies.DeterministicSoftPolicy(
        a_network=self._agent_module.p_net)
    self._test_policies['main'] = policy

  def _build_online_policy(self):
    return policies.RandomSoftPolicy(
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    self._build_p_loss(batch)
    self._p_vars = self._get_p_vars()
    self._ae_vars = self._agent_module.ae_variables

  def _build_checkpointer(self):
    state_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        agent=self._agent_module,
        global_step=self._global_step,
        )
    behavior_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net)
    return dict(state=state_ckpt, behavior=behavior_ckpt)

  def save(self, ckpt_name):
    self._checkpointer['state'].write(ckpt_name)
    self._checkpointer['behavior'].write(ckpt_name + '_behavior')

  def restore(self, ckpt_name):
    self._checkpointer['state'].restore(ckpt_name)


class AgentModule(agent.AgentModule):
  """Tensorflow module for agent."""

  def _build_modules(self):
    self._p_net = self._modules.p_net_factory()
    self._alpha_entropy_var = tf.Variable(1.0)

  def get_alpha_entropy(self):
    return utils.relu_v2(self._alpha_entropy_var)

  def assign_alpha_entropy(self, alpha):
    self._alpha_entropy_var.assign(alpha)

  @property
  def ae_variables(self):
    return [self._alpha_entropy_var]

  @property
  def p_net(self):
    return self._p_net

  def p_fn(self, s):
    return self._p_net(s)

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables


def get_modules(model_params, action_spec):
  def p_net_factory():
    return networks.ActorNetwork(
        action_spec,
        fc_layer_params=model_params[0])
  modules = utils.Flags(p_net_factory=p_net_factory)
  return modules


class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
