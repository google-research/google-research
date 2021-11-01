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

"""The L2 Slack Stochastic Polyak solver."""

import dataclasses
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.numpy as jnp

from jaxopt import base
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like


class SPSL1State(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  value: float
  slack: float
  velocity: Optional[Any]
  aux: Any


@dataclasses.dataclass
class SPSL1:
  r"""The L1 slack SPS method (SPSL1).

  Based on Lemma 3.2. Each update solves the projection problem
         w',  s' = argmin_{w\in\R^d} ||w - w^t||^2
             + delta^-1 (s-s^t)^2+ lmbda |s|
                   subject to value + <grad, w - w^t> <= s,  s >= 0.
      To which the solution is
        step1 = (value-s+\delta\lmbda)_+/ (\delta+ ||grad||^2)
        spsstep = value /||grad||^2
        w  = w - min( step1, spsstep) grad.
        s  =   ( s-lmbda delta + delta *step1)_+
      Consequently as delta -> 0 with s_0 =0 this method becomes the SPS method
  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    lmbda: a regularization parameter that also acts like a learning rate
    delta: damping factor
    momentum: momentum parameter, 0 corresponding to no momentum.
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
  """
  fun: Callable
  lmbda: float
  delta: float
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

    state = SPSL1State(
        iter_num=0, value=jnp.inf, slack=0, velocity=velocity, aux=None)
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
      epoch: int.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    if self.has_aux:
      (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)
    else:
      value, grad = self._value_and_grad_fun(params, *args, **kwargs)
      aux = None

    gradnorm = tree_l2_norm(grad, squared=True)
    step1 = jax.nn.relu(value - state.slack + self.delta * self.lmbda) / (
        self.delta + gradnorm)
    spsstep = value / gradnorm
    step_size = jnp.minimum(step1, spsstep)
    newslack = jax.nn.relu(state.slack - self.lmbda * self.delta +
                           self.delta * step1)
    # new_params = tree_add_scalar_mul(params, -step_size, grad)

    if self.momentum == 0:
      new_params = tree_add_scalar_mul(params, -step_size, grad)
      new_velocity = None
    else:
      # new_v = momentum * v - step_size * grad
      # new_params = params + new_v
      new_velocity = tree_sub(tree_scalar_mul(self.momentum, state.velocity),
                              tree_scalar_mul(step_size, grad))
      new_params = tree_add(params, new_velocity)
    new_state = SPSL1State(
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
