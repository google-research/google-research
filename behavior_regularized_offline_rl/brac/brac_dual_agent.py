# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Behavior Regularized Actor Critic without estimated behavior policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from behavior_regularized_offline_rl.brac import agent
from behavior_regularized_offline_rl.brac import divergences
from behavior_regularized_offline_rl.brac import networks
from behavior_regularized_offline_rl.brac import policies
from behavior_regularized_offline_rl.brac import utils


ALPHA_MAX = 500.0


@gin.configurable
class Agent(agent.Agent):
  """BRAC dual agent class."""

  def __init__(
      self,
      alpha=1.0,
      alpha_max=ALPHA_MAX,
      train_alpha=False,
      value_penalty=True,
      target_divergence=0.0,
      alpha_entropy=0.0,
      train_alpha_entropy=False,
      target_entropy=None,
      divergence_name='kl',
      warm_start=20000,
      c_iter=3,
      ensemble_q_lambda=1.0,
      **kwargs):
    self._alpha = alpha
    self._alpha_max = alpha_max
    self._train_alpha = train_alpha
    self._value_penalty = value_penalty
    self._target_divergence = target_divergence
    self._divergence_name = divergence_name
    self._train_alpha_entropy = train_alpha_entropy
    self._alpha_entropy = alpha_entropy
    self._target_entropy = target_entropy
    self._warm_start = warm_start
    self._c_iter = c_iter
    self._ensemble_q_lambda = ensemble_q_lambda
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_fn
    self._c_fn = self._agent_module.c_net
    self._divergence = divergences.get_divergence(
        name=self._divergence_name)
    self._agent_module.assign_alpha(self._alpha)
    if self._target_entropy is None:
      self._target_entropy = - self._action_spec.shape[0]
    self._get_alpha_entropy = self._agent_module.get_alpha_entropy
    self._agent_module.assign_alpha_entropy(self._alpha_entropy)

  def _get_alpha(self):
    return self._agent_module.get_alpha(
        alpha_max=self._alpha_max)

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_c_vars(self):
    return self._agent_module.c_variables

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

  def _get_c_weight_norm(self):
    weights = self._agent_module.c_weights
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
    a2_b = batch['a2']
    r = batch['r']
    dsc = batch['dsc']
    _, a2_p, log_pi_a2_p = self._p_fn(s2)
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = q_fn_target(s2, a2_p)
      q1_pred = q_fn(s1, a1)
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = tf.stack(q2_targets, axis=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    div_estimate = self._divergence.dual_estimate(
        s2, a2_p, a2_b, self._c_fn)
    v2_target = q2_target - self._get_alpha_entropy() * log_pi_a2_p
    if self._value_penalty:
      v2_target = v2_target - self._get_alpha() * div_estimate
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
    info['q2_target_mean'] = tf.reduce_mean(q2_target)
    info['q1_target_mean'] = tf.reduce_mean(q1_target)

    return loss, info

  def _build_p_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, log_pi_a_p = self._p_fn(s)
    q1s = []
    for q_fn, _ in self._q_fns:
      q1_ = q_fn(s, a_p)
      q1s.append(q1_)
    q1s = tf.stack(q1s, axis=-1)
    q1 = self._ensemble_q1(q1s)
    div_estimate = self._divergence.dual_estimate(
        s, a_p, a_b, self._c_fn)
    q_start = tf.cast(
        tf.greater(self._global_step, self._warm_start),
        tf.float32)
    p_loss = tf.reduce_mean(
        self._get_alpha_entropy() * log_pi_a_p
        + self._get_alpha() * div_estimate
        - q1 * q_start)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays[1] * p_w_norm
    loss = p_loss + norm_loss

    info = collections.OrderedDict()
    info['p_loss'] = p_loss
    info['p_norm'] = p_w_norm

    return loss, info

  def _build_c_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, _ = self._p_fn(s)
    c_loss = self._divergence.dual_critic_loss(
        s, a_p, a_b, self._c_fn)
    c_w_norm = self._get_c_weight_norm()
    norm_loss = self._weight_decays[2] * c_w_norm
    loss = c_loss + norm_loss

    info = collections.OrderedDict()
    info['c_loss'] = c_loss
    info['c_norm'] = c_w_norm

    return loss, info

  def _build_a_loss(self, batch):
    s = batch['s1']
    a_b = batch['a1']
    _, a_p, _ = self._p_fn(s)
    alpha = self._get_alpha()
    div_estimate = self._divergence.dual_estimate(
        s, a_p, a_b, self._c_fn)
    a_loss = - tf.reduce_mean(alpha * (div_estimate - self._target_divergence))

    info = collections.OrderedDict()
    info['a_loss'] = a_loss
    info['alpha'] = alpha
    info['div_mean'] = tf.reduce_mean(div_estimate)
    info['div_std'] = tf.math.reduce_std(div_estimate)

    return a_loss, info

  def _build_ae_loss(self, batch):
    s = batch['s1']
    _, _, log_pi_a = self._p_fn(s)
    alpha = self._get_alpha_entropy()
    ae_loss = tf.reduce_mean(alpha * (- log_pi_a - self._target_entropy))

    info = collections.OrderedDict()
    info['ae_loss'] = ae_loss
    info['alpha_entropy'] = alpha

    return ae_loss, info

  def _get_source_target_vars(self):
    return (self._agent_module.q_source_variables,
            self._agent_module.q_target_variables)

  def _build_optimizers(self):
    opts = self._optimizers
    if len(opts) == 1:
      opts = tuple([opts[0]] * 4)
    elif len(opts) < 4:
      raise ValueError('Bad optimizers %s.' % opts)
    self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._c_optimizer = utils.get_optimizer(opts[2][0])(lr=opts[2][1])
    self._a_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
    self._ae_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
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
    c_info = self._optimize_c(batch)
    if self._train_alpha:
      a_info = self._optimize_a(batch)
    if self._train_alpha_entropy:
      ae_info = self._optimize_ae(batch)
    info.update(p_info)
    info.update(q_info)
    info.update(c_info)
    if self._train_alpha:
      info.update(a_info)
    if self._train_alpha_entropy:
      info.update(ae_info)
    return info

  @tf.function
  def _extra_c_step(self, batch):
    return self._optimize_c(batch)

  def train_step(self):
    train_batch = self._get_train_batch()
    info = self._optimize_step(train_batch)
    for _ in range(self._c_iter - 1):
      train_batch = self._get_train_batch()
      self._extra_c_step(train_batch)
    for key, val in info.items():
      self._train_info[key] = val.numpy()
    self._global_step.assign_add(1)

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

  def _optimize_c(self, batch):
    vars_ = self._c_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_c_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._c_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_a(self, batch):
    vars_ = self._a_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_a_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._a_optimizer.apply_gradients(grads_and_vars)
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
    policy = policies.MaxQSoftPolicy(
        a_network=self._agent_module.p_net,
        q_network=self._agent_module.q_nets[0][0],
        )
    self._test_policies['max_q'] = policy

  def _build_online_policy(self):
    return policies.RandomSoftPolicy(
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self._build_c_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._c_vars = self._get_c_vars()
    self._a_vars = self._agent_module.a_variables
    self._ae_vars = self._agent_module.ae_variables

  def _build_checkpointer(self):
    return tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        agent=self._agent_module,
        global_step=self._global_step,
        )


class AgentModule(agent.AgentModule):
  """Tensorflow module for BRAC dual agent."""

  def _build_modules(self):
    self._q_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),  # Learned Q-value.
           self._modules.q_net_factory(),]  # Target Q-value.
          )
    self._p_net = self._modules.p_net_factory()
    self._c_net = self._modules.c_net_factory()
    self._alpha_var = tf.Variable(1.0)
    self._alpha_entropy_var = tf.Variable(1.0)

  def get_alpha(self, alpha_max=ALPHA_MAX):
    return utils.clip_v2(
        self._alpha_var, 0.0, alpha_max)

  def get_alpha_entropy(self):
    return utils.relu_v2(self._alpha_entropy_var)

  def assign_alpha(self, alpha):
    self._alpha_var.assign(alpha)

  def assign_alpha_entropy(self, alpha):
    self._alpha_entropy_var.assign(alpha)

  @property
  def a_variables(self):
    return [self._alpha_var]

  @property
  def ae_variables(self):
    return [self._alpha_entropy_var]

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

  def p_fn(self, s):
    return self._p_net(s)

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables

  @property
  def c_net(self):
    return self._c_net

  @property
  def c_weights(self):
    return self._c_net.weights

  @property
  def c_variables(self):
    return self._c_net.trainable_variables


def get_modules(model_params, action_spec):
  """Gets Tensorflow modules for Q-function, policy, and discriminator."""
  model_params, n_q_fns = model_params
  if len(model_params) == 1:
    model_params = tuple([model_params[0]] * 3)
  elif len(model_params) < 3:
    raise ValueError('Bad model parameters %s.' % model_params)
  def q_net_factory():
    return networks.CriticNetwork(
        fc_layer_params=model_params[0])
  def p_net_factory():
    return networks.ActorNetwork(
        action_spec,
        fc_layer_params=model_params[1])
  def c_net_factory():
    return networks.CriticNetwork(
        fc_layer_params=model_params[2])
  modules = utils.Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      c_net_factory=c_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules


class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.action_spec)
