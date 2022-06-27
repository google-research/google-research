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

"""Schedule functions for hyperparameters."""
import jax
import jax.numpy as jnp

import hparam_utils


def get_fixed_value(theta, param_name, t, T, **kwargs):
  return theta


def get_linear_sched_val(theta, param_name, t, T, **kwargs):
  """Here theta is assumed to be 2D"""
  init_value = theta[0]
  final_value = theta[1]
  return init_value * (T - t) / T + final_value * t / T


def get_inv_decay_sched_val(theta, param_name, t, T, **kwargs):
  init_value = theta[0]
  decay = theta[1]
  return init_value / ((1 + t / 5000.0)**decay)


def get_fixed_val_single(opt_params, theta, param_name, t, T, **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  optim_tree = jax.tree_map(lambda x: theta, opt_params[param_name])
  return optim_tree


def get_linear_sched_val_single(opt_params, theta, param_name, t, T, **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  optim_tree = jax.tree_map(
      lambda x: get_linear_sched_val(theta, param_name, t, T),
      opt_params[param_name])
  return optim_tree


def get_inv_time_decay_sched_val_single(opt_params, theta, param_name, t, T,
                                        **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  optim_tree = jax.tree_map(
      lambda x: get_inv_decay_sched_val(theta, param_name, t, T),
      opt_params[param_name])
  return optim_tree


def get_fixed_val_pl(opt_params, theta, param_name, t, T, **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  return kwargs['unflatten_func_dict'][param_name](theta)


def get_linear_sched_val_pl(opt_params, theta, param_name, t, T, **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  hparam_tree = kwargs['unflatten_func_dict'][param_name](theta)
  optim_tree = jax.tree_map(
      lambda theta_val: get_linear_sched_val(theta_val, param_name, t, T),
      hparam_tree)
  return optim_tree


def get_inverse_time_decay_sched_value_pl(opt_params, theta, param_name, t, T,
                                          **kwargs):
  theta = hparam_utils.cons_funcs[param_name](theta)
  hparam_tree = kwargs['unflatten_func_dict'][param_name](theta)
  optim_tree = jax.tree_map(
      lambda theta_val: get_inv_decay_sched_val(theta_val, param_name, t, T),
      hparam_tree)
  return optim_tree


schedule_funcs = {
    'fixed': get_fixed_val_single,
    'linear': get_linear_sched_val_single,
    'inverse-time-decay': get_inv_time_decay_sched_val_single,
    'fixed-pl': get_fixed_val_pl,
    'linear-pl': get_linear_sched_val_pl,
    'inverse-time-decay-pl': get_inverse_time_decay_sched_value_pl,
}
