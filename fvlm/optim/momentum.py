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

"""Momentum optimizer."""

from flax import struct
import jax.numpy as jnp
import numpy as np
from optim.base import OptimizerDef


@struct.dataclass
class _MomentumHyperParams:
  learning_rate: np.ndarray
  beta: np.ndarray
  weight_decay: np.ndarray
  nesterov: bool


@struct.dataclass
class _MomentumParamState:
  momentum: np.ndarray


class Momentum(OptimizerDef):
  """Momentum optimizer."""

  def __init__(self, learning_rate=None, beta=0.9, weight_decay=0,
               nesterov=False):
    """Constructor for the Momentum optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta: the coefficient used for the moving average of the
        gradient (default: 0.9).
      weight_decay: weight decay coefficient to apply (default: 0).
      nesterov: whether to use Nesterov momentum (default: False).
    """

    hyper_params = _MomentumHyperParams(
        learning_rate, beta, weight_decay, nesterov)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _MomentumParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    if hyper_params.weight_decay != 0:
      grad += hyper_params.weight_decay * param
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + grad
    if hyper_params.nesterov:
      d_p = grad + hyper_params.beta * new_momentum
    else:
      d_p = new_momentum
    new_param = param - hyper_params.learning_rate * d_p
    new_state = _MomentumParamState(new_momentum)
    return new_param, new_state
