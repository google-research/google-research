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

"""Defines different learning_rate schedules."""

import jax.numpy as jnp


def polynomial_lr_scheduler(step, decay_steps, end_factor, power):
  """Same behavior as tf.train.polynomial_decay.

  This is the original formula for this learning rate scheduler:
    ```
    end_learning_rate = hparams['initial_learning_rate'] * hparams['end_factor']
    step = min(hparams['decay_steps'], step)
    decayed_learning_rate = (hparams['initial_learning_rate'] -
                             end_learning_rate) * (
                                 1 - step / hparams['decay_steps'])**(
                                     hparams['power']) + end_learning_rate
    ```
  We rewrite this as a multiplicative factor for the initial learning rate.
  Args:
    step: int; Current step.
    decay_steps: int; Parameter of the decay function.
    end_factor: float; Final lr is: initial lr x end_factor.
    power: int; Parameter of the decay function.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """

  step = min(decay_steps, step)
  decayed_learning_rate = (1 - end_factor) * (1 - step / decay_steps)**(
      power) + end_factor
  return decayed_learning_rate


def piecewise_constant_scheduler(step, decay_events, decay_factors):
  """Gives a scaling factor based on Piecewise Constant scheduling.

  Args:
    step: int; Current step.
    decay_events: list(int); List of steps in which a decay is applied.
    decay_factors: list(int); List containing the absolute ratio of the decay
      applied on the decay events. Note that each element of decay_factors is
      absolute (not relative). For example, to decay the learning rate to 0.5 of
      its initial value after 100 steps, followed by 0.1 of its *initial value*
      after 200 steps, with a plateau of 0.1 of its initial value thereafter,
      use decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  boundaries = jnp.array([0] + decay_events)
  factors = [1.0] + decay_factors
  index = jnp.sum(boundaries[1:] < step)
  ratio = jnp.take(factors, index)
  return ratio


def piecewise_linear_scheduler(step, decay_events, decay_factors):
  """Gives a scaling factor based on Piecewise Linear scheduling.

  Args:
    step: int; Current step.
    decay_events: list(int); List of steps in which a decay is applied.
    decay_factors: list(int); List containing the absolute ratio of the decay
      applied on the decay events.  Note that each element of decay_factors is
      absolute (not relative). For example, to decay the learning rate to 0.5 of
      its initial value after 100 steps, followed by 0.1 of its *initial value*
      after 200 steps, with a plateau of 0.1 of its initial value thereafter,
      use decay_events = [100, 200] and decay_factors = [0.5, 0.1].

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  boundaries = jnp.array([0] + decay_events)
  factors = [1.0] + decay_factors
  index = jnp.sum(boundaries[1:] < step)
  if index + 1 == len(factors):
    return jnp.take(factors, index)
  else:
    m = (jnp.take(factors, index + 1) - jnp.take(factors, index)) / (
        jnp.take(boundaries, index + 1) - jnp.take(boundaries, index))
    interpolated_factor = (
        m * (step - jnp.take(boundaries, index)) + jnp.take(factors, index))
    return interpolated_factor


def linear_warmup_scheduler(step, warmup_steps):
  """Gives a scaling factor based on scheduling with a Linear Warmup.

  Args:
    step: int; Current step.
    warmup_steps: int; How many steps to warm up for in the warmup schedule.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  return jnp.minimum(1.0, step / warmup_steps)


def rsqrt_decay_scheduler(step):
  """Gives a scaling factor based on scheduling with a rsqrt decay.

  Args:
    step: int; Current step.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  return 1. / jnp.sqrt(step)


def decay_every_scheduler(step, steps_per_decay, decay_factor):
  """Gives a scaling factor based on scheduling with a decay every n-steps.

  Args:
    step: int; Current step.
    steps_per_decay: int; How often to decay.
    decay_factor: float; The amount to decay.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  return decay_factor**(step // steps_per_decay)


def cosine_decay_scheduler(step, steps_per_cycle, t_mul=1, m_mul=1., alpha=0.):
  """Gives a scaling factor based on scheduling with a cosine decay.

  Args:
    step: int; Current step.
    steps_per_cycle: int; Number of steps to reset the decay cycle.
    t_mul: int; Used to derive the number of iterations in the i-th period.
    m_mul: float; Used to derive the initial learning rate of the i-th period.
    alpha: float; The minimum value as a fraction of the initial value.

  Returns:
    Scaling factor applied to the learning rate on the given step.
  """
  progress = step / float(steps_per_cycle)
  if t_mul == 1.0:
    i_restart = jnp.floor(progress)
    progress -= i_restart
  else:
    i_restart = jnp.floor(
        jnp.log(1.0 - progress * (1.0 - t_mul)) / jnp.log(t_mul))
    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
    progress = (progress - sum_r) / t_mul**i_restart
  m_fac = m_mul**i_restart
  cosine_decay = jnp.maximum(
      0.0, 0.5 * m_fac * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
  return (1 - alpha) * cosine_decay + alpha


def compound_lr_scheduler(hparams):
  """Creates a learning rate scheduler by comnining multiple factors.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay.

  For instance, `hparams['factors'] = 'constant*linear_warmup'` combines
  constant
  learning rate schadule with linear warmup. This requires to have related
  hparams
  that are: hparams['warmup_steps'] and hparams['initial_learning_rate'].

  Args:
    hparams: Relevant hparams based on the chosen factors.

  Returns:
    lr_fn: A function mapping global_step to lr.
  """

  ratio_factors = [n.strip() for n in hparams['factors'].split('*')]

  def lr_fn(step):
    """Step to learning rate function."""
    ratio = 1.0
    for name in ratio_factors:
      if name == 'constant':
        ratio *= hparams['initial_learning_rate']
      elif name == 'polynomial':
        decay_steps = hparams['decay_steps']
        end_factor = hparams['end_factor']
        power = hparams['power']
        ratio *= polynomial_lr_scheduler(step, decay_steps, end_factor, power)
      elif name == 'piecewise_constant':
        decay_events = hparams['decay_events']
        decay_factors = hparams['decay_factors']
        ratio *= piecewise_constant_scheduler(step, decay_events, decay_factors)

      elif name == 'piecewise_linear':
        decay_events = hparams['decay_events']
        decay_factors = hparams['decay_factors']
        ratio *= piecewise_linear_scheduler(step, decay_events, decay_factors)

      elif name == 'linear_warmup':
        warmup_steps = hparams['warmup_steps']
        ratio *= linear_warmup_scheduler(step, warmup_steps)

      elif name == 'rsqrt_decay':
        adjusted_step = jnp.maximum(step, hparams.get('warmup_steps', 0.))
        ratio *= rsqrt_decay_scheduler(adjusted_step)

      elif name == 'rsqrt_normalized_decay':
        warmup_steps = hparams.get('warmup_steps', 0.)
        adjusted_step = jnp.maximum(step, warmup_steps)
        ratio *= jnp.sqrt(warmup_steps) * rsqrt_decay_scheduler(adjusted_step)

      elif name == 'decay_every':
        steps_per_decay = hparams['steps_per_decay']
        decay_factor = hparams['decay_factor']
        ratio *= decay_every_scheduler(step, steps_per_decay, decay_factor)

      elif name == 'cosine_decay':
        steps_per_cycle = hparams['steps_per_cycle']
        t_mul = hparams.get('t_mul', 1.)
        m_mul = hparams.get('m_mul', 1.)
        alpha = hparams.get('alpha', 0.0)
        adjusted_step = jnp.maximum(
            0.0, (step - (hparams.get('warmup_steps', 0.) +
                          hparams.get('start_decay_step', 0.))))

        ratio *= cosine_decay_scheduler(
            adjusted_step,
            steps_per_cycle,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha)
      elif name == 'linear_decay':
        warmup_steps = hparams.get('warmup_steps', 0.)
        if hparams.get('total_steps') == warmup_steps:
          progress = 0
        else:
          progress = jnp.maximum(
              0.0, (step - warmup_steps) /
              float(hparams.get('total_steps') - warmup_steps))
        ratio -= hparams.get('end_learning_rate', 0.)
        ratio *= jnp.maximum(1.0 - progress, 0.0)
        ratio += hparams.get('end_learning_rate', 0.)
      elif name == 'minimum':
        ratio = jnp.minimum(ratio, hparams.get('min_learning_rate', 0.))
      elif name == 'epsilon_plus':
        ratio += hparams.get('epsilon_plus_value', 1e-5)
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ratio, dtype=jnp.float32)

  return lr_fn


lr_fn_dict = {
    'compound': compound_lr_scheduler,
}


def get_learning_rate_fn(hparams):
  """Looks up for the learning rate scheduler and return lr_fn.

  Args:
    hparams: Hyper parameters.

  Returns:
    A function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.

  """
  return lr_fn_dict[hparams.lr_hparams['learning_rate_schedule']](
      hparams.lr_hparams)
