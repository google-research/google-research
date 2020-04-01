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

"""BCQ agent.

Based on 'Off-Policy Deep Reinforcement Learning without Exploration' by Scott
Fujimoto, David Meger, Doina Precup.
"""
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


@gin.configurable
class Agent(agent.Agent):
  """BCQ agent class."""

  def __init__(
      self,
      ensemble_q_lambda=0.75,
      n_action_samples=10,
      use_target_policy=True,
      **kwargs):
    self._ensemble_q_lambda = ensemble_q_lambda
    self._n_action_samples = n_action_samples
    self._use_target_policy = use_target_policy
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_net
    self._p_fn_target = self._agent_module.p_net_target
    self._b_fn = self._agent_module.b_net

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_b_vars(self):
    return self._agent_module.b_variables

  def _get_q_weight_norm(self):
    weights = self._agent_module.q_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_p_weight_norm(self):
    weights = self._agent_module.p_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_b_weight_norm(self):
    weights = self._agent_module.b_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def ensemble_q(self, qs):
    lambda_ = self._ensemble_q_lambda
    return (lambda_ * tf.reduce_min(qs, axis=-1)
            + (1 - lambda_) * tf.reduce_max(qs, axis=-1))

  def _ensemble_q2_target(self, q2_targets):
    return self.ensemble_q(q2_targets)

  def _ensemble_q1(self, q1s):
    return self.ensemble_q(q1s)

  def _build_q_loss(self, batch):
    s1 = batch['s1']
    s2 = batch['s2']
    a1 = batch['a1']
    r = batch['r']
    dsc = batch['dsc']
    s2_dup = tf.tile(s2, [self._n_action_samples, 1])
    a2_samples = self._b_fn.sample(s2_dup)
    if self._use_target_policy:
      a2s = self._p_fn_target(s2_dup, a2_samples)
    else:
      a2s = self._p_fn(s2_dup, a2_samples)
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = q_fn_target(s2_dup, a2s)
      q1_pred = q_fn(s1, a1)
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = tf.stack(q2_targets, axis=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    q2_target = tf.reshape(q2_target, [self._n_action_samples, -1])
    v2_target = tf.reduce_max(q2_target, axis=0)
    q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)
    q_losses = []
    for q1_pred in q1_preds:
      q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_loss = tf.add_n(q_losses)
    q_w_norm = self._get_q_weight_norm()
    norm_loss = self._weight_decays[0] * q_w_norm
    loss = q_loss + norm_loss

    info = collections.OrderedDict()
    info['q_loss'] = q_loss
    info['q_norm'] = q_w_norm
    info['r_mean'] = tf.reduce_mean(r)
    info['dsc_mean'] = tf.reduce_mean(dsc)
    info['q2_target_mean'] = tf.reduce_mean(v2_target)
    info['q1_target_mean'] = tf.reduce_mean(q1_target)

    return loss, info

  def _build_p_loss(self, batch):
    s = batch['s1']
    sampled_actions = self._b_fn.sample(s)
    a = self._p_fn(s, sampled_actions)
    q1 = self._q_fns[0][0](s, a)
    p_loss = tf.reduce_mean(-q1)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info

  def _build_b_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    a_recon, mean, std = self._b_fn.forward(s, a_b)
    recon_loss = tf.reduce_mean(tf.square(a_recon - a_b))
    kl_losses = -0.5 * (1.0 + tf.log(tf.square(std)) - tf.square(mean) -
                        tf.square(std))
    kl_loss = tf.reduce_mean(kl_losses)
    b_loss = recon_loss + kl_loss * 0.5  # Based on the pytorch implementation.
    b_w_norm = self._get_b_weight_norm()
    norm_loss = self._weight_decays[2] * b_w_norm
    loss = b_loss + norm_loss

    info = collections.OrderedDict()
    info['recon_loss'] = recon_loss
    info['kl_loss'] = kl_loss
    info['b_loss'] = b_loss
    info['b_norm'] = b_w_norm

    return loss, info

  def _get_source_target_vars(self):
    return (self._agent_module.q_source_variables
            + self._agent_module.p_variables,
            self._agent_module.q_target_variables
            + self._agent_module.p_target_variables)

  def _build_optimizers(self):
    opts = self._optimizers
    if len(opts) == 1:
      opts = tuple([opts[0]] * 3)
    elif len(opts) < 3:
      raise ValueError('Bad optimizers %s.' % opts)
    self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._b_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
    if len(self._weight_decays) == 1:
      self._weight_decays = tuple([self._weight_decays[0]] * 3)

  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    if tf.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
    q_info = self._optimize_q(batch)
    p_info = self._optimize_p(batch)
    b_info = self._optimize_b(batch)
    info.update(p_info)
    info.update(q_info)
    info.update(b_info)
    return info

  def _optimize_q(self, batch):
    vars_ = self._q_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_q_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._q_optimizer.apply_gradients(grads_and_vars)
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

  def _optimize_b(self, batch):
    vars_ = self._b_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_b_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._p_optimizer.apply_gradients(grads_and_vars)
    return info

  def _build_test_policies(self):
    policy = policies.BCQPolicy(
        a_network=self._agent_module.p_net,
        q_network=self._agent_module.q_nets[0][0],
        b_network=self._agent_module.b_net,
        n=self._n_action_samples,
        )
    self._test_policies['main'] = policy

  def _build_online_policy(self):
    policy = policies.BCQPolicy(
        a_network=self._agent_module.p_net,
        q_network=self._agent_module.q_nets[0][0],
        b_network=self._agent_module.b_net,
        n=self._n_action_samples,
        )
    return policy

  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self._build_b_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._b_vars = self._get_b_vars()

  def _build_checkpointer(self):
    state_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        agent=self._agent_module,
        global_step=self._global_step,
        )
    behavior_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.b_net)
    return dict(state=state_ckpt, behavior=behavior_ckpt)

  def save(self, ckpt_name):
    self._checkpointer['state'].write(ckpt_name)

  def restore(self, ckpt_name):
    self._checkpointer['state'].restore(ckpt_name)


class AgentModule(agent.AgentModule):
  """Tensorflow module for BCQ agent."""

  def _build_modules(self):
    self._q_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),
           self._modules.q_net_factory(),]  # source and target
          )
    self._p_net = self._modules.p_net_factory()
    self._p_net_target = self._modules.p_net_factory()
    self._b_net = self._modules.b_net_factory()

  @property
  def q_nets(self):
    return self._q_nets

  @property
  def q_source_weights(self):
    q_weights = []
    for q_net, _ in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_target_weights(self):
    q_weights = []
    for _, q_net in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_source_variables(self):
    vars_ = []
    for q_net, _ in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def p_net(self):
    return self._p_net

  @property
  def p_net_target(self):
    return self._p_net_target

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables

  @property
  def p_target_variables(self):
    return self._p_net.trainable_variables

  @property
  def b_net(self):
    return self._b_net

  @property
  def b_weights(self):
    return self._b_net.weights

  @property
  def b_variables(self):
    return self._b_net.trainable_variables


def get_modules(model_params, action_spec):
  """Gets Tensorflow modules for Q-function, policy, and behavior."""
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 3)
  elif len(model_params) < 3:
    raise ValueError('Bad model parameters %s.' % model_params)
  model_params, n_q_fns, max_perturbation = model_params
  def q_net_factory():
    return networks.CriticNetwork(
        fc_layer_params=model_params[0])
  def p_net_factory():
    return networks.BCQActorNetwork(
        action_spec,
        fc_layer_params=model_params[1],
        max_perturbation=max_perturbation,
        )
  def b_net_factory():
    return networks.BCQVAENetwork(
        action_spec,
        fc_layer_params=model_params[2])
  modules = utils.Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      b_net_factory=b_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules


class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
