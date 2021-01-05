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

# Lint as: python3
"""Learned optimizer search spaces in Jax using Flax style optimizers!"""
from flax import struct
from flax.optim import OptimizerDef

import jax.numpy as jnp
import numpy as onp

from opt_list import common
from opt_list import jax_common


@struct.dataclass
class _NAdamWHyperParams:
  """HyperParameters for the NAdamW optimizer."""
  learning_rate: onp.ndarray
  beta1: onp.ndarray
  beta2: onp.ndarray
  epsilon: onp.ndarray
  adamw_weight_decay: onp.ndarray
  l2_weight_decay: onp.ndarray
  use_nesterov: onp.ndarray
  constant_fraction: onp.ndarray
  warmup_fraction: onp.ndarray
  min_learning_rate_mult: onp.ndarray
  training_steps: onp.ndarray
  use_bias_correction: onp.ndarray


@struct.dataclass
class _NAdamWParamState:
  grad_ema: onp.ndarray
  grad_sq_ema: onp.ndarray


class NAdamWCosineDecay(OptimizerDef):
  """NAdam optimizer."""

  def __init__(
      self,
      learning_rate=None,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      adamw_weight_decay=0.0,
      l2_weight_decay=0.0,
      use_nesterov=False,
      use_bias_correction=True,
      constant_fraction=1.0,
      warmup_fraction=0.0,
      min_learning_rate_mult=1.0,
      training_steps=10000,
  ):
    """Construct a new  NAdam / Adam / AdamW / NAdamW optimizer.

    Args:
      learning_rate: A Tensor or a floating point value. The base learning rate.
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      adamw_weight_decay: A floating point value. Weight decay similar to that
        in AdamW.
      l2_weight_decay: A floating point value. Weight decay similar to that of
        adding L2 loss.
      use_nesterov: A boolean for whether or not to use the NAdam algorithm.
      use_bias_correction: A boolean for whether or not to use bias correction.
      constant_fraction: the fraction of training steps number of steps to take
        before starting the decay. This includes the time spent warming up the
      warmup_fraction: the fraction of training steps to use for a learning rate
        warmup.
      min_learning_rate_mult: a multiplicative factor to control how low the
        learning rate should be decayed to. learning rate.
      training_steps: number of training steps the schedule should be run for.
    """
    hyper_params = _NAdamWHyperParams(learning_rate, beta1, beta2, epsilon,
                                      adamw_weight_decay, l2_weight_decay,
                                      use_nesterov, constant_fraction,
                                      warmup_fraction, min_learning_rate_mult,
                                      training_steps, use_bias_correction)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _NAdamWParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    update, state = jax_common.nadamw_update(step, hyper_params, param, state,
                                             grad)
    return param + update, state


def optimizer_for_idx(idx, training_steps):
  """Get a OptimizerDef for the given configuration and training_steps."""
  config = common.get_optimizer_config(idx)
  config['training_steps'] = training_steps
  return NAdamWCosineDecay(**config)
