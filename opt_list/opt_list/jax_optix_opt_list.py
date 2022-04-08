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

"""Learned optimizer search spaces in Jax using Optix style optimizers!

At the time of writing, optix does not support AdamW style weight decay.
This is a modification of the optix API to remain as close as possible while
adding the ability to use parameter values to inform the updates.
"""
import collections
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_multimap
from jax.tree_util import tree_unflatten

from opt_list import common
from opt_list import jax_common

InitUpdateWithParams = collections.namedtuple("InitUpdateWithParams",
                                              ("init", "update_with_params"))


def optimizer_for_idx(idx, training_steps):
  """Get a nadamw optimizer for the given configuration and training_steps.

  Unlike regular Optix functions, the update function returned here additionally
  takes a parameter argument.

  Args:
    idx: int The index into the learned optimizer list.
    training_steps: int total number of training steps that the model will be
      trained.

  Returns:
    An (init_fn, update_with_params_fn) tuple.
  """
  config = common.get_optimizer_config(idx)
  config["training_steps"] = training_steps
  config["use_bias_correction"] = True  # always true for now.
  hyper_params = jax_common.NAdamWHyperParams(**config)

  def init(params):
    zero_initial = tree_multimap(jnp.zeros_like, params)
    return zero_initial, zero_initial, 0

  def update_fn(grads, params, state):
    """Compute the update.

    Args:
      grads: pytree of ndarray
        Gradient values.
      params: pytree of ndarray
        Parameter values.
      state:
        A tuple of (gradient accumulators, squared gradient accumulators, idx)
    Returns:
      step: pytree of ndarray
        The step to be added to the parameter values.
      next_state:
        A tuple of (gradient accumulators, squared gradient accumulators, idx)
    """

    grad_acc, grad_sq_acc, idx = state

    def update_one(g, p, g_acc, g_sq_acc):
      s = jax_common.NAdamWParamState(g_acc, g_sq_acc)
      new_x, new_s = jax_common.nadamw_update(idx, hyper_params, p, s, g)
      return new_x, new_s

    # the following flattens, applies a map, extracts values out via zip,
    # then unflattens.
    flat_gs, tree_def = tree_flatten(grads)
    flat_ps, _ = tree_flatten(params)
    flat_s0, _ = tree_flatten(grad_acc)
    flat_s1, _ = tree_flatten(grad_sq_acc)

    next_param_states = tree_multimap(update_one, flat_gs, flat_ps, flat_s0,
                                      flat_s1)

    flat_step, flat_next_ss = zip(*next_param_states)
    flat_next_grad_acc, flat_next_grad_sq_acc = zip(*flat_next_ss)

    step = tree_unflatten(tree_def, flat_step)
    next_grad_acc = tree_unflatten(tree_def, flat_next_grad_acc)
    next_grad_sq_acc = tree_unflatten(tree_def, flat_next_grad_sq_acc)

    return step, (next_grad_acc, next_grad_sq_acc, idx + 1)

  return InitUpdateWithParams(init, update_fn)
