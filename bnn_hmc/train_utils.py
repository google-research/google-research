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

"""Utility functions for DNN training."""

from typing import Callable

import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as onp

from bnn_hmc import data
from bnn_hmc import hmc
from bnn_hmc import nn_loss


LRSchedule = Callable[[jnp.ndarray], jnp.ndarray]
Opt = optix.GradientTransformation


def make_cosine_lr_schedule(init_lr,
                            total_steps):
  """Cosine LR schedule."""
  def schedule(step):
    t = step / total_steps
    return 0.5 * init_lr * (1 + jnp.cos(t * onp.pi))
  return schedule


def make_optimizer(lr_schedule, momentum_decay):
  return optix.chain(optix.trace(decay=momentum_decay, nesterov=False),
                     optix.scale_by_schedule(lr_schedule),
                     optix.scale(-1))


def make_hmc_update_eval_fns(net,
                             train_set,
                             test_set,
                             likelihood_fn,
                             prior_fn):
  """Make update and eval functions for HMC training."""
  n_devices = len(jax.local_devices())

  def log_prob_and_grad_fn(params):
    params_p = jax.pmap(lambda _: params)(jnp.arange(n_devices))
    log_prob, _, grad = nn_loss.pmap_get_loss_acc_grad(net, params_p,
                                                       likelihood_fn, prior_fn,
                                                       train_set)
    return -log_prob[0], jax.tree_map(lambda g: -g[0], grad)

  def log_prob_and_acc(params, dataset):
    params_p = jax.pmap(lambda _: params)(jnp.arange(n_devices))
    log_prob, acc = nn_loss.pmap_get_loss_and_acc(net, params_p, likelihood_fn,
                                                  prior_fn, dataset)
    return -log_prob[0], acc[0]

  hmc_update = hmc.make_adaptive_hmc_update(log_prob_and_grad_fn)

  def update(params, log_prob, state_grad, key, step_size, trajectory_len):
    params, log_prob, state_grad, step_size, accept_prob = hmc_update(
        params, log_prob, state_grad, key, step_size, trajectory_len)
    key, = jax.random.split(key, 1)
    return params, log_prob, state_grad, step_size, key, accept_prob

  def evaluate(params):
    test_log_prob, test_acc = log_prob_and_acc(params, test_set)
    train_log_prob, train_acc = log_prob_and_acc(params, train_set)
    return test_log_prob, test_acc, train_log_prob, train_acc

  return update, evaluate, log_prob_and_grad_fn


def make_ckpt_dict(params, key, step_size, trajectory_len):
  ckpt_dict = {
      "params": params,
      "key": key,
      "step_size": step_size,
      "traj_len": trajectory_len
  }
  return ckpt_dict


def parse_ckpt_dict(ckpt_dict):
  field_names = ["params", "key", "step_size", "traj_len"]
  return [ckpt_dict[name] for name in field_names]
