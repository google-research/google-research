# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Learned optimizer search lists in pytorch!"""
from typing import Callable
from . import common
import numpy as np
import torch

from torch.optim.optimizer import Optimizer


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
    return torch.tensor(x, dtype=torch.float32)

  def fn(global_step):
    """Returns a learning rate given the current training iteration."""

    float_training_steps = ff(training_steps)
    global_step = ff(global_step)

    # ensure we don't train longer than training steps
    global_step = torch.min(global_step, float_training_steps)

    constant_steps = float_training_steps * constant_fraction
    x = torch.max(ff(global_step), ff(constant_steps))

    min_learning_rate = min_learning_rate_mult * learning_rate

    if warmup_fraction:
      min_warmup_fraction = max(warmup_fraction, constant_fraction)
      warmup_steps = float_training_steps * min_warmup_fraction
      is_warmup = ff(ff(warmup_steps) > ff(global_step))
      warmup_lr = (global_step / warmup_steps) * learning_rate
    else:
      warmup_lr = learning_rate
      is_warmup = 0.0

    step = x - constant_steps

    constant_and_decay = (learning_rate - min_learning_rate) * (
        torch.cos(step * np.pi /
                  (float_training_steps - constant_steps)) / 2.0 +
        0.5) + min_learning_rate

    new_learning_rate = constant_and_decay * (1.0 - is_warmup) + is_warmup * (
        warmup_lr)
    return new_learning_rate

  return fn


class NadamWCosineDecay(Optimizer):
  """Optimizer that implements Nadam / Adam / AdamW / NadamW type optimizers.

  This implements the default TF Optimizer API.
  """

  def __init__(
      self,
      params,
      learning_rate=1e-3,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      adamw_weight_decay=0.0,
      l2_weight_decay=0.0,
      use_bias_correction=True,
      use_nesterov=False,
      constant_fraction=1.0,
      warmup_fraction=0.0,
      min_learning_rate_mult=1.0,
      training_steps=10000,
  ):
    """Construct a new  Nadam / Adam / AdamW / NadamW optimizer.

    Args:
      params: Model parameters.
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
      use_bias_correction: A boolean for whether or not to use bias correction.
      use_nesterov: A boolean for whether or not to use the NAdam algorithm.
      constant_fraction: the fraction of training steps number of steps to take
        before starting the decay. This includes the time spent warming up the
      warmup_fraction: the fraction of training steps to use for a learning rate
        warmup.
      min_learning_rate_mult: a multiplicative factor to control how low the
        learning rate should be decayed to. learning rate.
      training_steps: number of training steps the schedule should be run for.
    """
    defaults = dict(
        lr=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        adamw_weight_decay=adamw_weight_decay,
        l2_weight_decay=l2_weight_decay,
        use_nesterov=use_nesterov,
        constant_fraction=constant_fraction,
        warmup_fraction=warmup_fraction,
        min_learning_rate_mult=min_learning_rate_mult,
        training_steps=training_steps,
        use_bias_correction=use_bias_correction)

    super(NadamWCosineDecay, self).__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Arguments:
      closure (callable, optional): A closure that reevaluates the model and
        returns the loss.

    Returns:
      loss: tensor

    Raises:
      RuntimeError: if sparse gradients are used.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad

        if grad.is_sparse:
          raise RuntimeError("No SparseGrads supported at this time.")

        state = self.state[p]

        # State initialization
        if len(state) == 0:  # pylint: disable=g-explicit-length-test
          state["step"] = 0
          # Exponential moving average of gradient values
          state["exp_avg"] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state["exp_avg_sq"] = torch.zeros_like(
              p, memory_format=torch.preserve_format)

        lr = get_cosine_learning_rate_fn(group["training_steps"], group["lr"],
                                         group["min_learning_rate_mult"],
                                         group["constant_fraction"],
                                         group["warmup_fraction"])(
                                             state["step"])

        grad = grad - p * group["l2_weight_decay"]

        beta1, beta2 = group["beta1"], group["beta2"]

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        state["step"] += 1
        t = state["step"]

        # correction
        if group["use_bias_correction"]:
          lr_t = lr * np.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
        else:
          lr_t = lr

        if group["use_nesterov"]:
          numerator = (beta1 * exp_avg + (1.0 - beta1) * grad)
          denom = torch.sqrt(exp_avg_sq) + group["epsilon"]
          step = lr_t * numerator / denom
        else:
          denom = torch.sqrt(exp_avg_sq) + group["epsilon"]
          step = lr_t * exp_avg / denom

        step = step + (lr_t * group["adamw_weight_decay"] * p)

        p.add_(-step)

    return loss


def optimizer_for_idx(params, idx, training_steps):
  """Get a Optimizer for the given configuration and training_steps."""
  config = common.get_optimizer_config(idx)
  config["training_steps"] = training_steps
  return NadamWCosineDecay(params, **config)
