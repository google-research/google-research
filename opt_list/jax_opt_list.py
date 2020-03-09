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

# Lint as: python3
"""Learned optimizer search lists in jax!"""

from typing import Callable
from . import common
from flax import struct
from flax.optim import OptimizerDef
from jax import lax
import jax.numpy as jnp
import numpy as onp


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


class NadamWCosineDecay(OptimizerDef):
  """Adam optimizer."""

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
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2

    lr = get_cosine_learning_rate_fn(hyper_params.training_steps,
                                     hyper_params.learning_rate,
                                     hyper_params.min_learning_rate_mult,
                                     hyper_params.constant_fraction,
                                     hyper_params.warmup_fraction)(
                                         step)

    grad = grad - param * hyper_params.l2_weight_decay

    grad_sq = lax.square(grad)

    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad

    grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

    t = step + 1.

    # correction
    if hyper_params.use_bias_correction:
      lr_t = lr * jnp.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
    else:
      lr_t = lr

    if hyper_params.use_nesterov:
      numerator = (beta1 * grad_ema + (1.0 - beta1) * grad)
      denom = jnp.sqrt(grad_sq_ema) + hyper_params.epsilon
      step = lr_t * numerator / denom
    else:
      denom = jnp.sqrt(grad_sq_ema) + hyper_params.epsilon
      step = lr_t * grad_ema / denom

    step = step + lr_t * hyper_params.adamw_weight_decay * param

    new_param = param - step

    new_state = _NAdamWParamState(grad_ema, grad_sq_ema)
    return new_param, new_state


def get_cosine_learning_rate_fn(
    training_steps, learning_rate, min_learning_rate_mult,
    constant_fraction, warmup_fraction):
  """Get a function that does cosine learning rate decay with warmup.

  The learning rate starts at zero, is "warmed up" linearly over
  `warmup_fraction * training_steps` iterations to achieve a final value of
  `learning_rate`. A constant learning rate of `learning_rate` is held up until
  `training_steps*constant_fraction` at which point a cosine decay is started
  to a final learning rate of `min_learning_rate_mult * learning_rate`.

  The cosine decay sets the learning rate using a monotomically decreasing
  section of the cosine function from 0 to pi/2. It has been proven to be useful
  in large large language modeling (gpt, megatron-lm) and image classification.
  See https://arxiv.org/abs/1608.03983 for more information on the cosine decay.


  Args:
    training_steps: number of training steps the schedule should be run for.
    learning_rate: base learning rate. This is the learning rate used just after
      warmup and where the decay starts from.
    min_learning_rate_mult: a multiplicative factor to control how low the
      learning rate should be decayed to.
    constant_fraction: the fraction of training steps number of steps to take
      before starting the decay. This includes the time spent warming up the
      learning rate.
    warmup_fraction: the fraction of training steps to use for a learning rate
      warmup.

  Returns:
    A function that takes as input a training iteration and returns the learning
    rate from the specified schedule.
  """

  def ff(x):
    """Convert input to float32."""
    return jnp.asarray(x, dtype=onp.float32)

  def fn(global_step):
    """Returns a learning rate given the current training iteration."""

    float_training_steps = ff(training_steps)
    global_step = ff(global_step)

    # ensure we don't train longer than training steps
    global_step = jnp.minimum(global_step, float_training_steps)

    constant_steps = float_training_steps * constant_fraction
    x = jnp.maximum(ff(global_step), ff(constant_steps))

    min_learning_rate = min_learning_rate_mult * learning_rate

    if warmup_fraction:
      min_warmup_fraction = jnp.maximum(warmup_fraction, constant_fraction)
      warmup_steps = float_training_steps * min_warmup_fraction
      is_warmup = ff(jnp.greater(ff(warmup_steps), ff(global_step)))
      warmup_lr = (global_step / warmup_steps) * learning_rate
    else:
      warmup_lr = learning_rate
      is_warmup = 0.0

    step = x - constant_steps

    constant_and_decay = (learning_rate - min_learning_rate) * (
        jnp.cos(step * onp.pi / (float_training_steps - constant_steps)) / 2.0 +
        0.5) + min_learning_rate

    new_learning_rate = constant_and_decay * (1.0 - is_warmup) + is_warmup * (
        warmup_lr)
    return new_learning_rate

  return fn


def optimizer_for_idx(idx, training_steps):
  """Get a OptimizerDef for the given configuration and training_steps."""
  config = common.get_optimizer_config(idx)
  config['training_steps'] = training_steps
  return NadamWCosineDecay(**config)
