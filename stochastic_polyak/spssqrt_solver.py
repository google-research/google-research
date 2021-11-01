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

"""The Square Root Slack Stochastic Polyak solver."""

import dataclasses
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.numpy as jnp

from jaxopt import base
# from jaxopt import tree_util
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like


class SPSsqrtState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  slack: float
  velocity: Optional[Any]
  aux: Any


@dataclasses.dataclass
class SPSsqrt:
  r"""Square Root Slack Stochastic Polyak solver SPSsqrt.

  Based on Lemma 3.3. Each update is given by the projection
    w',  s' = argmin_{w\in\R^d} (1-lmbda)||w - w^t||^2
                      + (1-lmbda) (s-s^t)^2+ lmbda s^2
                    subject to value + <grad, w - w^t> <= (s +s^t)/\sqrt(s^t)
       To which the solution is

          step = (value - (1-lmbda/2) sqrt(s))_+) / (4s||grad||^2 + 1 - lmbda)
          w = w -  4 step s *grad,
          s = (1-lmbda)*sqrt(s)*(sqrt(s) + step)

       When lmbda -> 1 this method STOPS. Thus lmbda <1. Also the s variable
       (slack) must be initialized away from 0.
       Otherwise the method also never runs.
  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    lmbda: a regularization parameter that also acts like a learning rate
    momentum: momentum parameter, 0 corresponding to no momentum.
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
  """
  fun: Callable
  lmbda: float
  momentum: float = 0.0
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
    state = SPSsqrtState(
        iter_num=0, value=jnp.inf, slack=0.0, velocity=velocity, aux=None)
    return base.OptStep(params=init_params, state=state)

  def update(self,
             params,
             state,
             epoch,
             *args,
             **kwargs):
    """Perform one update of the algorithm.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      epoch: number of epoch.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    del epoch   # unused
    if self.lmbda == 1:
      raise ValueError(
          'lmbda =1 was passed to SPSsqrt solver. This solver does not work with lmbda =1 because then the parameters are never updated! '
      )

    if self.has_aux:
      (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)
    else:
      value, grad = self._value_and_grad_fun(params, *args, **kwargs)
      aux = None

    # If slack hits zero, reset to be the current value.
    # This stops the method from halting.
    if state.slack == 0.0:
      state = state._replace(slack=value)
    ## The mathematical expression of the this update is:
    # step = (value - (1-lmbda/2) sqrt(s))_+) / (4s||grad||^2 + 1 - lmbda)
    # w = w -  4 step s *grad,
    # s = (1-lmbda)*sqrt(s)*(sqrt(s) + step)
    step_size = jax.nn.relu(
        value - (1 - self.lmbda / 2) * jnp.sqrt(state.slack)) / (
            4 * state.slack * tree_l2_norm(grad, squared=True) + 1 - self.lmbda)
    newslack = (1 - self.lmbda) * jnp.sqrt(state.slack) * (
        jnp.sqrt(state.slack) + step_size)
    step_size = 4*state.slack*step_size

    if self.momentum == 0:
      new_params = tree_add_scalar_mul(params, -step_size, grad)
      new_velocity = None
    else:
      # new_v = momentum * v - step_size * grad
      # new_params = params + new_v
      new_velocity = tree_sub(tree_scalar_mul(self.momentum, state.velocity),
                              tree_scalar_mul(step_size, grad))
      new_params = tree_add(params, new_velocity)

    new_state = SPSsqrtState(
        iter_num=state.iter_num + 1,
        value=value,
        slack=newslack,
        velocity=new_velocity,
        aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def __post_init__(self):
    # Pre-compile useful functions.
    self._value_and_grad_fun = jax.value_and_grad(self.fun,
                                                  has_aux=self.has_aux)
