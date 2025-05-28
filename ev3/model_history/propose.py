# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Methods for EV3's propose step."""

from typing import Callable, Tuple

import jax
import numpy as np
import optax

from ev3 import base
from ev3.model_history import eval_util
from ev3.model_history import struct


def generate_an_update(
    get_batch_fn,
    model,
    state,
    loss_ind,
    tx_ind,
):
  """Proposes an update to model parameters based on a loss and an SGD method.

  This method applies state.trajectory_length SGD updates using the loss indexed
  by 'loss_ind' and the SGD method indexed by 'tx_ind'.

  Args:
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    model: An object containing both the model graph and the model parameters.
    state: The state of the propose step of EV3, containing losses, SGD methods
      and a data iterator.
    loss_ind: The index of the loss whose gradients are to be used to generate
      an update.
    tx_ind: The index of the SGD method that is to be used to modify the
      gradients. Note on tx: In the JAX ecosystem, it is customary to refer to
      the gradient transformation as tx.

  Returns:
    A tuple (update, opt_state), where 'update' is a vector in the model
    parameter space connecting model.params to the end of the SGD trajectory and
    opt_state is the updated SGD state.
  """
  new_params = jax.tree.map(lambda a: a + 0.0, model.params)
  grad_fn = state.grad_fn_list[loss_ind]
  loss_state = state.loss_states[loss_ind]
  tx = state.tx_list[tx_ind]
  opt_state = state.opt_states[loss_ind][tx_ind]

  # A multiplicative factor to increase trajectory length based on history.
  insignificant_update_count = 0
  if model.history:
    significant_update = [
        'updated_graph' in h or h['significantly_better']
        for h in model.history[::-1]
    ]
    insignificant_update_count = (
        1 - np.maximum.accumulate(significant_update)
    ).sum()

  # If there is no significant improvement in the last few updates for the
  # same nn_model, increase the trajectory length.
  mul_factor = state.traj_mul_factor**insignificant_update_count
  trajectory_length = int(state.trajectory_length * mul_factor)

  # Roll out an SGD trajectory.
  for _ in range(trajectory_length):
    batch = get_batch_fn(state)
    grad = grad_fn(new_params, model.graph, loss_state, batch)
    params_update, opt_state = tx.update(grad, opt_state, model.params)
    new_params = jax.tree.map(lambda p, u: p + u, new_params, params_update)
  return new_params, opt_state


def generate_an_update_bn(
    get_batch_fn,
    model,
    state,
    loss_ind,
    tx_ind,
):
  """Proposes an update to model parameters based on a loss and an SGD method.

  This method applies state.trajectory_length SGD updates using the loss indexed
  by 'loss_ind' and the SGD method indexed by 'tx_ind'. This is a modification
  of generate_an_update, where the model can have batch normalization.

  Args:
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    model: An object containing both the model graph and the model parameters.
    state: The state of the propose step of EV3, containing losses, SGD methods
      and a data iterator.
    loss_ind: The index of the loss whose gradients are to be used to generate
      an update.
    tx_ind: The index of the SGD method that is to be used to modify the
      gradients. Note on tx: In the JAX ecosystem, it is customary to refer to
      the gradient transformation as tx.

  Returns:
    A tuple (update, opt_state), where 'update' is a vector in the model
    parameter space connecting model.params to the end of the SGD trajectory and
    opt_state is the updated SGD state.
  """
  new_params = jax.tree.map(lambda a: a + 0.0, model.params)
  grad_fn = state.grad_fn_list[loss_ind]
  loss_state = state.loss_states[loss_ind]
  tx = state.tx_list[tx_ind]
  opt_state = state.opt_states[loss_ind][tx_ind]

  # A multiplicative factor to increase trajectory length based on history.
  insignificant_update_count = 0
  if model.history:
    significant_update = [
        'updated_graph' in h or h['significantly_better']
        for h in model.history[::-1]
    ]
    insignificant_update_count = (
        1 - np.maximum.accumulate(significant_update)
    ).sum()

  # If there is no significant improvement in the last few updates for the
  # same nn_model, increase the trajectory length.
  mul_factor = state.traj_mul_factor**insignificant_update_count
  trajectory_length = int(state.trajectory_length * mul_factor)

  # Roll out an SGD trajectory.
  for _ in range(trajectory_length):
    batch = get_batch_fn(state)
    params_grad, batch_stats = grad_fn(
        new_params, model.graph, loss_state, batch
    )
    params_update, opt_state = tx.update(params_grad, opt_state, new_params)
    new_params = jax.tree.map(lambda p, u: p + u, new_params, params_update)
    new_params = {
        'params': new_params['params'],
        'batch_stats': batch_stats['batch_stats'],
    }

  return new_params, opt_state


def propose_init(
    state, model
):
  """Initializes the propose state.

  This method takes the gradients of the loss functions and initializes the SGD
  methods.

  Args:
    state: An object containing information relevant to the propose step of EV3,
      including loss functions and SGD methods.
    model: An object containing both the model graph and the initialized model
      parameters.

  Returns:
    The initialized propose state object.
  """
  grad_fn_list = tuple(
      [
          jax.jit(jax.grad(loss, has_aux=state.has_aux))
          for loss in state.loss_fn_list
      ]
  )
  tx_list = tuple([optax.with_extra_args_support(tx) for tx in state.tx_list])
  opt_states = tuple(
      [
          tuple([tx.init(model.params) for tx in tx_list])
          for _ in state.loss_fn_list
      ]
  )
  return state.replace(
      grad_fn_list=grad_fn_list, opt_states=opt_states, tx_list=tx_list
  )


def propose_update(
    state,
    model,
    get_batch_fn = eval_util.get_batch,
    generate_an_update_fn = generate_an_update,
):
  """Generates a list of proposed updates to the model parameters.

  Args:
    state: An object containing information relevant to the propose step of EV3,
      including loss functions, SGD methods and a data iterator.
    model: An object containing both the model graph and the model parameters.
    get_batch_fn: A function that generates batches of data using the data
      iterator in state.
    generate_an_update_fn: A function that produces an update to model
      parameters.

  Returns:
    A tuple (updates, state), where 'updates' contains the proposed updates
    and 'state' is the updated propose state.
  """
  if model.just_expanded:
    opt_states = []
    for _ in state.loss_fn_list:
      per_loss_opt_states = []
      for tx in state.tx_list:
        opt_state = tx.init(model.params)
        # Use a smaller lr for expanded models.
        # if 'learning_rate' in opt_state.hyperparams:
        #   opt_state.hyperparams['learning_rate'] *= 0.1
        per_loss_opt_states.append(opt_state)
      opt_states.append(tuple(per_loss_opt_states))
    state = state.replace(opt_states=tuple(opt_states))

  params_update_list = []
  opt_states_list = [list(t) for t in state.opt_states]
  for loss_ind in range(len(state.loss_fn_list)):
    for tx_ind in range(len(state.tx_list)):
      update, opt_state = generate_an_update_fn(
          get_batch_fn, model, state, loss_ind, tx_ind
      )
      params_update_list.append(update)
      opt_states_list[loss_ind][tx_ind] = opt_state
  model_updates = struct.ModelUpdates(params_list=tuple(params_update_list))  # pytype: disable=wrong-keyword-args  # dataclass_transform
  opt_states = tuple([tuple(t) for t in opt_states_list])
  return model_updates, state.replace(opt_states=opt_states)
