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

"""Optimizer definitions."""

import re
from typing import Optional, Sequence, Union, Mapping

from flax.core import frozen_dict
from flax.linen import initializers
import gin
import jax.example_libraries.optimizers as jax_opt
import jax.numpy as jnp
from jax.tree_util import tree_map
from optim import momentum

gin.external_configurable(momentum.Momentum)
gin.external_configurable(initializers.lecun_uniform, 'lecun_uniform')
gin.external_configurable(initializers.lecun_normal, 'lecun_normal')
gin.external_configurable(initializers.he_uniform, 'he_uniform')
gin.external_configurable(initializers.he_normal, 'he_normal')
gin.external_configurable(initializers.xavier_uniform, 'xavier_uniform')
gin.external_configurable(initializers.xavier_normal, 'xavier_normal')

Array = jnp.ndarray
FrozenDict = frozen_dict.FrozenDict


@gin.configurable
def create_learning_rate_scheduler(
    global_step,
    factors = 'constant * linear_warmup * cosine_decay',
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_factor = 0.5,
    steps_per_decay = 20000,
    steps_per_cycle = 100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * linear_decay: Linear decay, uses steps_per_cycle_parameter. The learning
    rate goes to 0 after steps_per_cycle
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    global_step: An int or scalar array indicating current training step.
    factors: string, factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]
  global_step = jnp.array(global_step, jnp.float32)
  ret = 1.0
  for name in factors:
    if name == 'constant':
      ret *= base_learning_rate
    elif name == 'linear_warmup':
      ret *= jnp.minimum(1.0, global_step / warmup_steps)
    elif name == 'rsqrt_decay':
      ret /= jnp.sqrt(jnp.maximum(global_step, warmup_steps))
    elif name == 'rsqrt_normalized_decay':
      ret *= jnp.sqrt(warmup_steps)
      ret /= jnp.sqrt(jnp.maximum(global_step, warmup_steps))
    elif name == 'decay_every':
      ret *= (decay_factor**(global_step // steps_per_decay))
    elif name == 'linear_decay':
      progress = jnp.maximum(0.0, (global_step - warmup_steps) /
                             float(steps_per_cycle - warmup_steps))
      ret *= jnp.maximum(0., 1.0 - progress)
    elif name == 'cosine_decay':
      progress = jnp.maximum(0.0, (global_step - warmup_steps) /
                             float(steps_per_cycle - warmup_steps))
      ret *= jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * progress)))
    else:
      raise ValueError('Unknown factor %s.' % name)
  return jnp.asarray(ret, dtype=jnp.float32)


@gin.configurable
def step_learning_rate_with_linear_warmup(
    global_step,
    init_learning_rate = 0.08,
    warmup_learning_rate = 0.0067,
    warmup_steps = 500,
    decay_factor = None,
    learning_rate_levels = None,
    learning_rate_steps = None,
    total_steps = 25000,
    learning_rate_step_ratios = (0.7, 0.9)
):
  """Creates the step learning rate tensor with linear warmup.

  Args:
    global_step: An int or scalar array indicating current training step.
    init_learning_rate: Initial learning rate after warm up.
    warmup_learning_rate: Warmup learning rate to start with.
    warmup_steps: Number of warmup training iterations.
    decay_factor: The factor to decay learning rate at. When
      learning_rate_levels is not specified, this value is used to scale the
      learning rate at each learning rate step.
    learning_rate_levels: A sequence of learning rates to use over the training
      process.
    learning_rate_steps: A sequence of steps to apply different learning rates
      over the training process. If not specified, decay_factor must be given.
    total_steps: Total number of steps.
    learning_rate_step_ratios: Ratio of total steps to scale down the learning
      rates.

  Returns:
    learning_rate: A scalar array indicating the current learning rate.
  """
  if learning_rate_steps is None:
    learning_rate_steps = [
        total_steps * lr_ratio for lr_ratio in learning_rate_step_ratios
    ]

  if learning_rate_levels is None:
    learning_rate_levels = [
        init_learning_rate * decay_factor**x
        for x in range(1, 1 + len(learning_rate_steps))
    ]
  elif len(learning_rate_steps) != len(learning_rate_levels):
    raise ValueError('Learning rate steps and levels must be equal in length!')
  else:
    if decay_factor is not None:
      raise ValueError(
          'Decay factor must be None when learning rate levels are given!')

  global_step = jnp.array(global_step, jnp.float32)
  linear_warmup = (
      warmup_learning_rate + global_step / warmup_steps *
      (init_learning_rate - warmup_learning_rate))
  learning_rate = jnp.where(global_step < warmup_steps, linear_warmup,
                            init_learning_rate)

  for next_learning_rate, start_step in zip(learning_rate_levels,
                                            learning_rate_steps):
    learning_rate = jnp.where(global_step >= start_step, next_learning_rate,
                              learning_rate)
  return learning_rate


@gin.register
def gradient_clipping(grads, max_norm=0.0):
  """Global norm gradient clipping."""
  if max_norm > 0.0:
    return jax_opt.clip_grads(grads, max_norm)
  else:
    return grads


@gin.configurable(denylist=['grads'])
def submodule_gradient_scaling(
    grads, filter_regex_scales):
  """Scale the sub-module gradients by path to filter.

  Args:
    grads: A gradient PyTree.
    filter_regex_scales: A dictionary where the keys are regex filters to select
      top-level sub-modules and values are factors to scale the gradients of
      submodules by.

  Returns:
    scaled_grads: A new PyTree with scaled gradients.
  """
  grad_dict = frozen_dict.unfreeze(grads)
  for key, sub_grad_dict in grad_dict.items():
    for regex, scale in filter_regex_scales.items():
      if re.match(regex, key):
        grad_dict[key] = tree_map(lambda x: x * scale, sub_grad_dict)

  return frozen_dict.freeze(grad_dict)

