# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Temperature, learning rate and step size scheduler selection functions."""
import ast

from flax.training import lr_schedule
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections


def get_make_lr_fn(config):
  """Construct the learning rate schedule based on config.

  Args:
    config: ConfigDict

  Returns:
    lr scheduling function.
  """
  if config.lr_schedule == 'constant':

    def make_lr_fn(base_lr, steps_per_epoch):
      return lr_schedule.create_constant_learning_rate_schedule(
          base_lr, steps_per_epoch, warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'stepped':
    if not config.lr_sched_steps:
      lr_sched_steps = [[60, 0.2], [120, 0.04], [160, 0.008]]
    else:
      lr_sched_steps = ast.literal_eval(config.lr_sched_steps)

    def make_lr_fn(base_lr, steps_per_epoch):
      return lr_schedule.create_stepped_learning_rate_schedule(
          base_lr,
          steps_per_epoch,
          lr_sched_steps,
          warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'cosine':

    def make_lr_fn(base_lr, steps_per_epoch):
      _, temp_end = ast.literal_eval(config.temp_ramp)
      return create_cold_posterior_scheduler(
          base_lr,
          steps_per_epoch,
          config.cycle_length,
          wait_epochs=temp_end,
          warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'halfcos':

    def make_lr_fn(base_lr, steps_per_epoch):
      return lr_schedule.create_cosine_learning_rate_schedule(
          base_lr,
          steps_per_epoch,
          config.num_epochs,
          warmup_length=config.warmup_epochs)
  else:
    raise ValueError('Unknown LR schedule type {}'.format(config.lr_schedule))
  return make_lr_fn


def create_cold_posterior_scheduler(base_learning_rate, steps_per_epoch,
                                    halfcos_epochs, wait_epochs, warmup_length):
  """Cosine scheduler as in Wenzel et al.

  Wenzel et al. How Good is the Bayes Posterior in Deep Neural Networks Really?

  Args:
    base_learning_rate: the base learning rate
    steps_per_epoch: the number of iterations per epoch
    halfcos_epochs: the number of epochs to complete half a cosine wave;
      normally the number of epochs used for training
    wait_epochs: will hold of on cycling for number of epochs.
    warmup_length: # steps for linear learning rate warmup at start.

  Returns:
    Function `f(step) -> lr` that computes the learning rate for a given step.
  """
  halfwavelength_steps = halfcos_epochs * steps_per_epoch
  wait_steps = wait_epochs * steps_per_epoch

  min_value = 0.

  def learning_rate_fn(step):
    d_step = step - wait_steps
    pfraction = jnp.mod(d_step, halfwavelength_steps) / halfwavelength_steps
    scale_factor = min_value + (1.0 - min_value) * 0.5 * (
        jnp.cos(jnp.pi * pfraction) + 1.0)

    scale_factor = lax.cond(d_step < 0., d_step, lambda d_step: 1.0, d_step,
                            lambda d_step: scale_factor)
    lr = base_learning_rate * scale_factor
    if warmup_length > 0.0:
      lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)
    return lr

  return learning_rate_fn


def get_make_temp_fn(config):
  """Construct the SGMCMC sampling temperature schedule constructor.

  Args:
    config: ConfigDict

  Returns:
    Temperature scheduling function.
  """
  if config.temp_schedule == 'constant':

    def make_temp_fn(base_temp, steps_per_epoch):
      return lr_schedule.create_constant_learning_rate_schedule(
          base_temp, steps_per_epoch)
  elif config.temp_schedule == 'ramp_up':

    def make_temp_fn(base_temp, steps_per_epoch):
      temp_start, temp_end = ast.literal_eval(config.temp_ramp)

      def temp_fn(step):
        epoch = step // steps_per_epoch
        temp_scale = jax.lax.clamp(0.0, (epoch - temp_start) /
                                   (temp_end - temp_start), 1.0)
        return base_temp * temp_scale

      return temp_fn
  else:
    raise ValueError('Unknown temp schedule type {}'.format(
        config.temp_schedule))
  return make_temp_fn


def get_make_step_size_fn(config):
  """Construct the SGMCMC step size schedule constructor.

  Step size is analogous to learning rate in the SGMCMC setting.

  Args:
    config: ConfigDict

  Returns:
    Step size scheduling function.
  """
  if config.lr_schedule == 'constant':

    def make_step_size_fn(steps_per_epoch):
      return lr_schedule.create_constant_learning_rate_schedule(
          1.0, steps_per_epoch, warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'cosine':

    def make_step_size_fn(steps_per_epoch):
      _, temp_end = ast.literal_eval(config.temp_ramp)
      return create_cold_posterior_scheduler(
          1.0,
          steps_per_epoch,
          config.cycle_length,
          wait_epochs=temp_end,
          warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'stepped':
    if not config.lr_sched_steps:
      lr_sched_steps = [[60, 0.2], [120, 0.04], [160, 0.008]]
    else:
      lr_sched_steps = ast.literal_eval(config.lr_sched_steps)

    def make_step_size_fn(steps_per_epoch):
      return lr_schedule.create_stepped_learning_rate_schedule(
          1.0,
          steps_per_epoch,
          lr_sched_steps,
          warmup_length=config.warmup_epochs)
  elif config.lr_schedule == 'halfcos':

    def make_step_size_fn(steps_per_epoch):
      return lr_schedule.create_cosine_learning_rate_schedule(
          1.0,
          steps_per_epoch,
          config.num_epochs,
          warmup_length=config.warmup_epochs)
  else:
    raise ValueError('Unknown lr schedule type {}'.format(config.lr_schedule))
  return make_step_size_fn
