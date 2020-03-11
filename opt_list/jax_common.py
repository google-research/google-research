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
"""Functions shared across different jax optimization libraries."""

import collections
from typing import Callable
from jax import lax
import jax.numpy as jnp
import numpy as onp

NAdamWHyperParams = collections.namedtuple("NAdamWHyperParams", [
    "learning_rate", "beta1", "beta2", "epsilon", "adamw_weight_decay",
    "l2_weight_decay", "use_nesterov", "constant_fraction", "warmup_fraction",
    "min_learning_rate_mult", "training_steps", "use_bias_correction"
])

NAdamWParamState = collections.namedtuple("NAdamWParamState",
                                          ["grad_ema", "grad_sq_ema"])


def nadamw_update(step, hyper_params, param, state, grad):
  """Apply a parameter update using the nadamw optimizer.

  Args:
    step: int
      Current training iteration.
    hyper_params: NAdamWHyperParams
      A object containing all of the hyper parameters to perform a step.
    param: ndarray
      Current parameter value.
    state: NAdamWParamState
      State consiting of EMA of the gradient and gradient squared.
    grad: ndarray
      Gradient to use when computing the update.
  Returns:
    new_param: ndarray
      The next parameter value
    new_state: NAdamWParamState
      The updated state (gradient and gradient squared) value.
  """
  assert hyper_params.learning_rate is not None, "no learning rate provided."
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

  new_state = NAdamWParamState(grad_ema, grad_sq_ema)
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
