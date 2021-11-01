# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Stochastic Polyak solver."""

import dataclasses
import functools
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
import jax
import jax.numpy as jnp


from jaxopt import base
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot
from jaxopt.tree_util import tree_zeros_like


class StochasticPolyakState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  velocity: Optional[Any]
  aux: Any


@dataclasses.dataclass
class DiagonalStochasticPolyak:
  """Stochastic Polyak solver.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    learning_rate: a learning rate to use
    delta: a dampening parameter
    momentum: momentum parameter, 0 corresponding to no momentum.
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
  """
  fun: Callable
  learning_rate: float
  delta: float
  momentum: float
  has_aux: bool = False

  def init(self,
           init_params):
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if self.momentum == 0:
      velocity = None
    else:
      velocity = tree_zeros_like(init_params)

    state = StochasticPolyakState(
        iter_num=0, value=jnp.inf, velocity=velocity, aux=None)
    return base.OptStep(params=init_params, state=state)

  def update(self,
             params,
             state,
             data,
             *args,
             **kwargs):
    """Performs one iteration of the optax solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      data: dict.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """

    del args, kwargs  # unused
    (value, aux), update = self._spsdiag_update(params, data)
    if self.momentum == 0:
      new_params = tree_add_scalar_mul(params, self.learning_rate, update)
      new_velocity = None
    else:
      new_velocity = tree_sub(
          tree_scalar_mul(self.momentum, state.velocity),
          tree_scalar_mul(self.learning_rate, update))
      new_params = tree_add(params, new_velocity)

    new_params = tree_add_scalar_mul(
        params, self.learning_rate, update)
    aux['loss'] = jnp.mean(aux['loss'])
    aux['accuracy'] = jnp.mean(aux['accuracy'])

    if state.iter_num % 10 == 0:
      print('Number of iterations', state.iter_num,
            '. Objective function value: ', value)

    new_state = StochasticPolyakState(
        iter_num=state.iter_num+1, value=value, velocity=new_velocity, aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def __post_init__(self):

    # Pre-compile useful functions.
    def fun(params, args):
      losses, aux = self.fun(params, args)  # assumes has_aux = True
      return jnp.mean(losses), aux

    def least_square_regularizor_1d(a, b, delta):
      # Computes the solution to min || a^Tx -b||^2 + delta ||x||^2
      scale = -b/(tree_vdot(a, a) + delta)
      return tree_scalar_mul(scale, a)

    def single_update(params, data):
      data_expanded = {
          'image': jnp.expand_dims(data['image'], axis=0),
          'label': jnp.expand_dims(data['label'], axis=0)
      }
      (value, aux), grad = jax.value_and_grad(
          fun, has_aux=self.has_aux)(params, data_expanded)

      new_update = least_square_regularizor_1d(grad, value, self.delta)
      return (value, aux), new_update

    all_updates = jax.vmap(
        single_update, in_axes=(None, {'image': 0, 'label': 0}))

    agg_exdim = functools.partial(jnp.sum, axis=0)
    self._spsdiag_update = jax.jit(
        lambda params, data: jax.tree_map(agg_exdim, all_updates(params, data)))
