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
"""Learned optimizer search spaces in TensorFlow!"""

from typing import Text, Callable, Optional
from absl import logging
import numpy as np
from opt_list import common
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class NAdamWOptimizer(tf1.train.AdamOptimizer):
  """Optimizer that implements Nadam / Adam / AdamW / NadamW type optimizers.

  This implements the default TF Optimizer API.
  """

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               adamw_weight_decay=0.0,
               l2_weight_decay=0.0,
               use_bias_correction=True,
               use_nesterov=False,
               use_locking=False,
               name="NAdamW"):
    """Construct a new  Nadam / Adam / AdamW / NadamW optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
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
      use_bias_correction: A boolean for whether or not to use bias correction.
      use_nesterov: A boolean for whether or not to use the NAdam algorithm.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".  @compatibility(eager) When eager execution is
        enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
        callable that takes no arguments and returns the actual value to use.
        This can be useful for changing these values across different
        invocations of optimizer functions. @end_compatibility
    """
    super(NAdamWOptimizer, self).__init__(learning_rate, beta1, beta2, epsilon,
                                          use_locking, name)
    self._use_bias_correction = use_bias_correction
    self._use_nesterov = use_nesterov
    self._l2_weight_decay = l2_weight_decay
    self._adamw_weight_decay = adamw_weight_decay

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")

    lr = tf.cast(self._lr_t, grad.dtype.base_dtype)
    beta1 = tf.cast(self._beta1_t, grad.dtype.base_dtype)
    beta2 = tf.cast(self._beta2_t, grad.dtype.base_dtype)
    epsilon = tf.cast(self._epsilon_t, grad.dtype.base_dtype)

    grad = grad - var * self._l2_weight_decay

    # m_t = beta_1 * m_{t-1} + (1-beta_1) * g_t
    m_t = m.assign(beta1 * m + (1.0 - beta1) * grad)

    # v_t = beta_2 * v_{t-1} + (1-beta_2) * g_t ** 2
    v_t = v.assign(beta2 * v + (1.0 - beta2) * grad * grad)

    if self._use_bias_correction:
      beta1_power, beta2_power = self._get_beta_accumulators()
      beta1_power = tf.cast(beta1_power, grad.dtype.base_dtype)
      beta2_power = tf.cast(beta2_power, grad.dtype.base_dtype)
      lr_t = lr * tf.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
    else:
      lr_t = lr

    if self._use_nesterov:
      # delta theta = lr_t * (
      #    (beta_1 * m_t + (1-beta1) * g_t) / (sqrt(v_t) + epsilon))
      step = lr_t * ((beta1 * m_t + (1.0 - beta1) * grad) /
                     (tf.sqrt(v_t) + epsilon))
    else:
      # delta theta = lr_t * m_t / (sqrt(v_t) + epsilon)
      step = lr_t * m_t / (tf.sqrt(v_t) + epsilon)

    # AdamW style weight decay term.
    step = step + lr_t * self._adamw_weight_decay * var

    theta_t = tf.assign_sub(var, step)

    return tf.group(*[theta_t, m_t, v_t])

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    return self._apply_dense(tf.convert_to_tensor(grad), var)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)


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

  def fn(global_step):
    """Returns a learning rate given the current training iteration."""

    float_training_steps = tf.to_float(training_steps)
    global_step = tf.to_float(global_step)

    # ensure we don't train longer than training steps
    global_step = tf.minimum(global_step, float_training_steps)

    constant_steps = float_training_steps * constant_fraction
    x = tf.maximum(tf.to_float(global_step), tf.to_float(constant_steps))

    min_learning_rate = min_learning_rate_mult * learning_rate

    if warmup_fraction:
      min_warmup_fraction = tf.minimum(warmup_fraction, constant_fraction)
      warmup_steps = float_training_steps * min_warmup_fraction
      is_warmup = tf.to_float(
          tf.greater(tf.to_float(warmup_steps), tf.to_float(global_step)))
      warmup_lr = (global_step / warmup_steps) * learning_rate
    else:
      warmup_lr = learning_rate
      is_warmup = 0.0

    step = x - constant_steps

    constant_and_decay = (learning_rate - min_learning_rate) * (
        tf.math.cos(step * np.pi /
                    (float_training_steps - constant_steps)) / 2.0 +
        0.5) + min_learning_rate

    new_learning_rate = constant_and_decay * (1.0 - is_warmup) + is_warmup * (
        warmup_lr)
    return new_learning_rate

  return fn


class NAdamWKeras(adam.Adam, NAdamWOptimizer):
  """Keras Optimizer that implements Nadam / Adam / AdamW / NadamW."""

  def __init__(self,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               adamw_weight_decay=0.0,
               l2_weight_decay=0.0,
               use_bias_correction=True,
               use_nesterov=False):
    """Construct a new NAdam / Adam / AdamW / NadamW optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
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
      use_bias_correction: A boolean for whether or not to use bias correction.
      use_nesterov: A boolean for whether or not to use the NAdam algorithm.
    """

    adam.Adam.__init__(
        self,
        learning_rate=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon)

    NAdamWOptimizer.__init__(
        self,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon)

    self._set_hyper("l2_weight_decay", l2_weight_decay)
    self._set_hyper("adamw_weight_decay", adamw_weight_decay)
    self._set_hyper("use_nesterov", use_nesterov)

  def get_config(self):
    config = {
        "learning_rate":
            self._serialize_hyperparameter("learning_rate"),
        "beta_1":
            self._serialize_hyperparameter("beta_1"),
        "beta_2":
            self._serialize_hyperparameter("beta_2"),
        "epsilon":
            self.epsilon,
        "l2_weight_decay":
            self._serialize_hyperparameter("l2_weight_decay"),
        "adamw_weight_decay":
            self._serialize_hyperparameter("adam2_weight_decay"),
        "use_bias_correction":
            self.use_bias_correction,
        "use_nesterov":
            self.use_nesterov,
    }
    base_config = super(NAdamWKeras, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CustomCosineDecay(learning_rate_schedule.LearningRateSchedule):
  """A custom cosine learning rate decay that is used by the optimizer list."""

  def __init__(self,
               training_steps,
               learning_rate,
               min_learning_rate_mult,
               constant_fraction,
               warmup_fraction,
               name = None):
    """Cosine learning rate decay with warmup.

    The learning rate starts at zero, is "warmed up" linearly over
    `warmup_fraction * training_steps` iterations to achieve a final value of
    `learning_rate`. A constant learning rate of `learning_rate` is held up
    until
    `training_steps*constant_fraction` at which point a cosine decay is started
    to a final learning rate of `min_learning_rate_mult * learning_rate`.

    The cosine decay sets the learning rate using a monotomically decreasing
    section of the cosine function from 0 to pi/2. It has been proven to be
    useful
    in large large language modeling (gpt, megatron-lm) and image
    classification.
    See https://arxiv.org/abs/1608.03983 for more information on the cosine
    decay.


    Args:
      training_steps: number of training steps the schedule should be run for.
      learning_rate: base learning rate. This is the learning rate used just
        after warmup and where the decay starts from.
      min_learning_rate_mult: a multiplicative factor to control how low the
        learning rate should be decayed to.
      constant_fraction: the fraction of training steps number of steps to take
        before starting the decay. This includes the time spent warming up the
        learning rate.
      warmup_fraction: the fraction of training steps to use for a learning rate
        warmup.
      name: Name of the LearningRateSchedule object.
    """
    super(CustomCosineDecay, self).__init__()
    self.config = {
        "training_steps": training_steps,
        "learning_rate": learning_rate,
        "min_learning_rate_mult": min_learning_rate_mult,
        "constant_fraction": constant_fraction,
    }

    self.lr_fn = get_cosine_learning_rate_fn(
        training_steps=training_steps,
        learning_rate=learning_rate,
        min_learning_rate_mult=min_learning_rate_mult,
        constant_fraction=constant_fraction,
        warmup_fraction=warmup_fraction)

    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "CustomCosineDecay"):
      return self.lr_fn(step)

  def get_config(self):
    return self.config


def optimizer_for_idx(
    idx,
    training_steps,
    iteration = None):
  """Get a nadamw optimizer for the given configuration and training_steps."""
  # TODO(lmetz) the global step is obtained here. Ideally, we should be using
  # the value used by the underlying tensorflow optimizer but at this moment
  # we don't have access to it.
  if not iteration:
    logging.warning("Iteration not passed in! Using the default global_step for"
                    "keeping track of training progress")
    iteration = tf.train.get_or_create_global_step()

  cfg = common.get_optimizer_config(idx)

  fn = get_cosine_learning_rate_fn(
      training_steps=training_steps,
      learning_rate=cfg["learning_rate"],
      min_learning_rate_mult=cfg["min_learning_rate_mult"],
      constant_fraction=cfg["constant_fraction"],
      warmup_fraction=cfg["warmup_fraction"])

  return NAdamWOptimizer(
      learning_rate=fn(iteration),
      beta1=cfg["beta1"],
      beta2=cfg["beta2"],
      epsilon=cfg["epsilon"],
      l2_weight_decay=cfg["l2_weight_decay"],
      adamw_weight_decay=cfg["adamw_weight_decay"],
  )


def keras_optimizer_for_idx(idx,
                            training_steps):
  """Get a nadamw optimizer for the given configuration and training_steps."""
  cfg = common.get_optimizer_config(idx)

  decay = CustomCosineDecay(
      training_steps=training_steps,
      learning_rate=cfg["learning_rate"],
      min_learning_rate_mult=cfg["min_learning_rate_mult"],
      constant_fraction=cfg["constant_fraction"],
      warmup_fraction=cfg["warmup_fraction"])

  return NAdamWKeras(
      learning_rate=decay,
      beta1=cfg["beta1"],
      beta2=cfg["beta2"],
      epsilon=cfg["epsilon"],
      l2_weight_decay=cfg["l2_weight_decay"],
      adamw_weight_decay=cfg["adamw_weight_decay"],
  )
