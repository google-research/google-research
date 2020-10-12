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

"""Helper utilities for training."""
import time
from typing import Any

from absl import logging
from flax import struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


class TrainStateHistory:
  """Container for training history/metrics.

  The eventual design goal is to have this container store a long of all
  training metrics, as well as handle logging them (including to tensorboard).
  The learning rate can be a function of training history, for example in
  approaches that decay the learning rate whenever validation performance stops
  improving. Therefore the learning rate calculation also belongs here, though
  the current API still needs work.
  """

  def __init__(self, learning_rate_fn, print_every=200):
    self.learning_rate_fn = learning_rate_fn
    self.print_every = print_every
    self.last_printed = None

  def write(self, step, metrics):
    """TODO(kitaev): doc."""
    if step % self.print_every != 0:
      return
    # Only retrieve metrics from the device if they are actually used.
    metrics = jax.tree_map(lambda x: x[0].item(), metrics)
    for i, k in enumerate(sorted(metrics)):
      if i == 0:
        line = f"Step {step:<7d} {k} = {metrics[k]}"
      else:
        line = f"             {k} = {metrics[k]}"
      print(line, flush=True)
      logging.info(line)

    now = time.time()
    if self.last_printed:
      last_step, last_time = self.last_printed
      seconds_per_step = (now - last_time) /  (step - last_step)
      line = f"             seconds_per_step = {seconds_per_step}"
      print(line, flush=True)
      logging.info(line)
    self.last_printed = (step, now)

  def initial_state(self):
    return TrainState(
        rng=common_utils.shard_prng_key(jax.random.PRNGKey(0)),
        step=None,
        metrics=None,
        history=self)


@struct.dataclass
class TrainState():
  """Container for misc training state that's not handeled by the optimizer.

  This includes:
  - The base RNG key for each step, replicated across devices.
  - Any metrics output by the training step (that are then logged to the history
    object)
  """
  rng: Any
  step: Any
  metrics: Any
  history: TrainStateHistory = struct.field(pytree_node=False)

  def take_step(self, optimizer, grad, metrics, rng):
    step = optimizer.state.step
    new_optimizer = optimizer.apply_gradient(
        grad, learning_rate=self.history.learning_rate_fn(step))
    # TODO(marcvanzee): Remove this when b/162398046 is fixed.
    new_train_state = self.replace(rng=rng, step=step, metrics=metrics)  # pytype: disable=attribute-error
    return new_optimizer, new_train_state

  def write_history(self):
    step = self.step[0]
    self.history.write(step, self.metrics)
    return self.replace(step=None, metrics=None)  # pytype: disable=attribute-error


def create_train_step(loss_and_metrics_fn, clip_grad_norm=None):
  """Constructs a function that runs a single training update."""
  def train_step(optimizer, batch, train_state):
    rng, new_rng = jax.random.split(train_state.rng)
    grad_fn = jax.value_and_grad(
        lambda model: loss_and_metrics_fn(model, batch, rng), has_aux=True)
    (unused_loss, metrics), grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, "batch")
    if clip_grad_norm is not None:
      grad_norm = sum([jnp.sum(x ** 2) for x in jax.tree_leaves(grad)])
      metrics["grad_norm"] = grad_norm
      grad_scale = jnp.where(
          grad_norm < clip_grad_norm, 1.0, clip_grad_norm / grad_norm)
      grad = jax.tree_map(lambda x: x * grad_scale, grad)
    new_optimizer, new_train_state = train_state.take_step(
        optimizer, grad, metrics, new_rng)
    return new_optimizer, new_train_state
  p_train_step = jax.pmap(train_step, axis_name="batch")

  def distributed_train_step(optimizer, batch, train_state):
    new_optimizer, new_train_state = p_train_step(
        optimizer, common_utils.shard(batch), train_state)
    new_train_state = new_train_state.write_history()
    return new_optimizer, new_train_state

  return distributed_train_step


def create_eval_fn(stat_fn, sample_feature_name="idx"):
  """Constructs a function that runs evaluation given a batched data stream."""
  p_stat_fn = jax.pmap(stat_fn, axis_name="batch")
  n_devices = jax.local_device_count()

  def eval_step_fn(optimizer, batch, all_stats):
    batch_size = batch[sample_feature_name].shape[0]
    remainder = batch_size % n_devices
    if remainder:
      pad_amount = n_devices - remainder
      def pad(x):
        assert x.shape[0] == batch_size
        return np.concatenate([x] + [x[:1]] * pad_amount, axis=0)
      batch = jax.tree_map(pad, batch)
    batch = common_utils.shard(batch)
    stats = p_stat_fn(optimizer.target, batch)
    stats = jax.tree_map(np.array, stats)
    stats = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), stats)
    if remainder:
      stats = jax.tree_map(lambda x: x[:-pad_amount], stats)
    all_stats.append(stats)

  def eval_fn(optimizer, data_stream):
    all_stats = []
    for batch in data_stream:
      eval_step_fn(optimizer, batch, all_stats)
    res = {}
    for k in all_stats[0]:
      res[k] = np.concatenate([stats[k] for stats in all_stats], axis=0)
    return res

  return eval_fn
