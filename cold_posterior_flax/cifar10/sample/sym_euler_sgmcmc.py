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

"""SGMCMC Sampler based on [1].

[1] Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L.,
  Mandt,
  S., et al. (2020, February 6). How Good is the Bayes Posterior in Deep Neural
  Networks Really? http://arxiv.org/abs/2002.02405.
"""

from flax import struct
from flax.deprecated import nn
from flax.optim import OptimizerDef
import jax
import jax.numpy as jnp
import numpy as onp


@struct.dataclass
class _SymEulerSGMCMCHyperParamsDL:
  """Deep Learning style parametrization [1] for SGMCMC.

  - [1] Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L.,
  Mandt,
  S., et al. (2020, February 6). How Good is the Bayes Posterior in Deep Neural
  Networks Really? http://arxiv.org/abs/2002.02405.
  """
  learning_rate: onp.ndarray
  beta: onp.ndarray
  weight_decay: onp.ndarray
  train_size: onp.ndarray
  temperature: onp.ndarray
  # For SGMCMC you want to schedule not the lr but the step_size using a factor.
  step_size_factor: onp.ndarray

  @property
  def step_size(self):
    step_size = jnp.sqrt(
        self.learning_rate / self.train_size) * self.step_size_factor
    return step_size

  @property
  def friction(self):
    friction = (1 - self.beta) * jnp.sqrt(self.train_size / self.learning_rate)
    return friction

  @step_size.setter
  def step_size(self, value):
    raise NotImplementedError(
        "Using DL parametrization, modulate step_size with step_size_factor.")

  @friction.setter
  def friction(self, value):
    raise NotImplementedError(
        "Using DL parametrization, setting friction not supported.")


@struct.dataclass
class _SymEulerSGMCMCHyperParamsSDE:
  """SDE style parametrization for SGMCMC."""
  step_size: onp.ndarray
  friction: onp.ndarray
  weight_decay: onp.ndarray
  train_size: onp.ndarray
  temperature: onp.ndarray

  @property
  def learning_rate(self):
    raise NotImplementedError(
        "Using SDE parametrization, use step_size instead.")

  @property
  def beta(self):
    raise NotImplementedError("Using SDE parametrization, use beta instead.")

  @learning_rate.setter
  def learning_rate(self, value):
    raise NotImplementedError(
        "Using SDE parametrization, use step_size instead.")

  @beta.setter
  def beta(self, value):
    raise NotImplementedError(
        "Using SDE parametrization, use friction instead.")


@struct.dataclass
class _SymEulerSGMCMCParamState:
  momentum: onp.ndarray
  preconditioner: onp.ndarray  # diagonal of the preconditioner matrix.


class SymEulerSGMCMC(OptimizerDef):
  """Symplectic Euler SGMCMC sampler."""
  needs_rng = True

  def __init__(self,
               train_size,
               init_rng,
               learning_rate=None,
               beta=0.9,
               weight_decay=0,
               temperature=1.0,
               step_size=None,
               friction=None,
               step_size_factor=1.0):
    """Constructor for the Symplectic Euler SGMCMC sampler.

    Args:
      train_size: number of training datapoints.
      init_rng: Initialization PRNG.
      learning_rate: the step size used to update the parameters.
      beta: the coefficient used for the moving average of the
        gradient (default: 0.9).
      weight_decay: weight decay coefficient to apply (default: 0).
      temperature:  temperature for the sampler.
      step_size: SDE discretization step size.
      friction: SDE friction.
      step_size_factor: scaling factor of the step_size.  Either set (step_size
        & friction) or (learning_rate, beta & step_size_factor).
    """

    if learning_rate is not None and beta is not None and step_size_factor is not None:
      hyper_params = _SymEulerSGMCMCHyperParamsDL(learning_rate, beta,
                                                  weight_decay, train_size,
                                                  temperature, step_size_factor)
    elif step_size is not None and friction is not None:
      hyper_params = _SymEulerSGMCMCHyperParamsSDE(step_size, friction,
                                                   weight_decay, train_size,
                                                   temperature)
    else:
      raise ValueError(
          "Either specify step_size and friction for SDE parametrization, "
          "or learning_rate and beta for DL parametrization.")
    self.init_rng = init_rng

    super().__init__(hyper_params)

  def init_param_state(self, param):
    # TODO(basv): do we want to init momentum randomly?
    return _SymEulerSGMCMCParamState(
        jax.random.normal(nn.make_rng(), param.shape, param.dtype),
        jnp.ones_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, "no learning rate provided."
    if hyper_params.weight_decay != 0:
      raise NotImplementedError("Weight decay not supported")

    noise = jax.random.normal(
        key=nn.make_rng(), shape=param.shape, dtype=param.dtype)

    momentum = state.momentum
    h = hyper_params.step_size
    gamma = hyper_params.friction
    t = hyper_params.temperature
    n = hyper_params.train_size

    new_momentum = (
        (1 - h * gamma) * momentum - h * n * grad +
        jnp.sqrt(2 * gamma * h * t) * jnp.sqrt(state.preconditioner) * noise)

    new_param = param + h * (1. / state.preconditioner) * new_momentum
    new_state = _SymEulerSGMCMCParamState(new_momentum, state.preconditioner)
    return new_param, new_state
