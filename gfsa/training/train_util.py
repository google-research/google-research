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
"""Training helper functions that are shared across tasks."""

import contextlib
import functools
import operator
import signal
import typing
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

from absl import logging
import dataclasses
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax

from gfsa import jax_util
from gfsa.datasets import data_loading


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class ExampleWithMetadata:
  """Stores an example or batch of examples.

  Attributes:
    epoch: Integer representing the epoch that this example comes from.
    example_id: Integer ID uniquely identifying this example in the dataset.
    example: The example itself.
    mask: Array that is True for actual examples, False for padding.
    static_metadata: Metadata about this example or batch that should result in
      a new `jit` XLA computation (i.e. padded shapes).
  """
  epoch: Any
  example_id: Any
  example: Any
  mask: jax_util.NDArray = np.array(True)
  static_metadata: Any = None


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class RatioMetric:
  """A ratio, where numerator and denominator should be summed separately.

  Attributes:
    numerator: Numerator of the metric.
    denominator: Denominator of the metric.
  """
  numerator: jax_util.NDArray
  denominator: jax_util.NDArray


MetricValue = Union[float, jax_util.NDArray, RatioMetric]

# A loss function is a callable (model, example, static_metadata)
#                                 -> (loss, metrics)
# pyformat: disable
LossFunWithMetrics = Callable[
    [Any, Any, Any],
    Tuple[jax_util.NDArray, Dict[str, MetricValue]]]
# pyformat: enable

# A validation function is a callable (replicated_model) -> (objective, metrics)
# where model is a tree of ShardedDeviceArrays, and objective is the value we
# want to make decrease.
ValidationFunction = Callable[[Any], Tuple[float, Dict[str, MetricValue]]]


def device_broadcast(x, num_devices):
  """Broadcast a value to all devices."""
  return jax.pmap(lambda _: x)(jnp.arange(num_devices))


def _parallel_train_step(
    optimizer,
    batched_examples,
    static_batch_metadata,
    loss_fn,
    max_global_norm = None,
    **optimizer_hyper_params,
):
  """Train the model for one step in parallel across devices.

  Args:
    optimizer: Optimizer that tracks the model and parameter state. Should be
      replicated to each device, i.e. should contain ShardedDeviceArrays with a
      leading axis (num_devices, ...) but with the same content on each device.
    batched_examples: A structure of NDArrays representing a batch of examples.
      Should have two leading batch dimensions: (num_devices,
        batch_size_per_device, ...)
    static_batch_metadata: Metadata about this batch, which will be shared
      across all batched examples. Each value of this results in a separate
      XLA-compiled module.
    loss_fn: Task-specific non-batched loss function to apply. Should take the
      current model (optimizer.target) and an example from batched_examples, and
      return a tuple of the current loss (as a scalar) and a dictionary from
      string names to metric values (also scalars, or RatioMetrics).
    max_global_norm: Maximum global norm to clip gradients to. Should be a
      scalar, which will be broadcast automatically.
    **optimizer_hyper_params: Hyperparameters to pass to the optimizer's
      `apply_gradient` function, which will be broadcast across devices
      automatically.

  Returns:
    Tuple (updated_optimizer, grads_ok, metrics). Metrics will be as returned by
    loss_fn, with an extra elements "loss". All metrics will be averaged
    across all elements of the batch. Both optimizer and metrics will contain
    ShardedDeviceArrays that are identical across devices. grads_ok will be
    a replicated bool ndarray that is True if the gradients were finite.
  """

  def batched_loss_fn(model):
    """Apply loss function across a batch of examples."""
    loss, metrics = jax.vmap(loss_fn, (None, 0, None))(model, batched_examples,
                                                       static_batch_metadata)
    return jnp.mean(loss), metrics

  # Compute gradients of loss, along with metrics.
  (loss, metrics), grads = jax.value_and_grad(
      batched_loss_fn, has_aux=True)(
          optimizer.target)
  metrics["loss"] = loss

  # Exchange average gradients and metrics across devices.
  agg_grads = jax.lax.pmean(grads, "devices")
  agg_metrics = {}
  for k, v in metrics.items():
    if isinstance(v, RatioMetric):
      num = jax.lax.psum(jnp.sum(v.numerator), "devices")
      denom = jax.lax.psum(jnp.sum(v.denominator), "devices")
      new_value = num / denom
    else:
      # Use nanmean to aggregate bare floats.
      new_value = jnp.nanmean(jax.lax.all_gather(v, "devices"))
    agg_metrics[k] = new_value

  # Compute global norm and possibly clip.
  global_norm = optax.global_norm(agg_grads)
  agg_metrics["gradient_global_norm"] = global_norm
  if max_global_norm is not None:
    should_clip = global_norm > max_global_norm
    agg_grads = jax.tree_map(
        lambda g: jnp.where(should_clip, g * max_global_norm / global_norm, g),
        agg_grads)
    agg_metrics["gradient_was_clipped"] = should_clip.astype("float32")

  # Check for non-finite gradients.
  grads_ok = jnp.all(
      jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_leaves(agg_grads)]))

  # Apply updates.
  updated_optimizer = optimizer.apply_gradient(agg_grads,
                                               **optimizer_hyper_params)

  return updated_optimizer, grads_ok, agg_metrics, agg_grads


def _build_parallel_train_step():
  """Builds an accelerated version of the train step function."""
  # We need to wrap and unwrap so that the final function can be called with
  # keyword arguments, but we still maintain the proper axes.

  @functools.partial(
      jax.pmap,
      axis_name="devices",
      in_axes=(0, 0, None, None, None, None),
      static_broadcasted_argnums=(2, 3))
  def wrapped(optimizer, batched_examples, static_batch_metadata, loss_fn,
              max_global_norm, optimizer_hyper_params):
    return _parallel_train_step(optimizer, batched_examples,
                                static_batch_metadata, loss_fn, max_global_norm,
                                **optimizer_hyper_params)

  @functools.wraps(_parallel_train_step)
  def wrapper(optimizer, batched_examples, static_batch_metadata, loss_fn,
              max_global_norm, **optimizer_hyper_params):
    return wrapped(optimizer, batched_examples, static_batch_metadata, loss_fn,
                   max_global_norm, optimizer_hyper_params)

  return wrapper


# The primary version of the training step, with the associated jit cache.
parallel_train_step = _build_parallel_train_step()


def warmup_train_step(
    optimizer,
    batched_example,
    static_batch_metadata,
    loss_fn,
    optimizer_is_replicated = False,
    profile = False,
    runner=None,
):
  """Run a fake train step to warm up JIT cache.

  Args:
    optimizer: Optimizer that tracks the model and parameter state.
    batched_example: A structure of NDArrays representing a batch of examples.
    static_batch_metadata: Metadata about the batch, which will be shared across
      all batched examples.
    loss_fn: Task-specific non-batched loss function to apply. Should take the
      current model (optimizer.target) and an example from batched_examples, and
      return a tuple of the current loss (as a scalar) and a dictionary from
      string names to metric values (also scalars).
    optimizer_is_replicated: Whether optimizer is already replicated.
    profile: Whether to enable profiling during warmup.
    runner: If profile=True, the runner to use when profiling.
  """
  num_devices = jax.local_device_count()
  if optimizer_is_replicated:
    replicated_optimizer = optimizer
  else:
    replicated_optimizer = device_broadcast(optimizer, num_devices)

  (replicated_optimizer,
   batched_example) = jax.tree_map(jax.device_put,
                                   (replicated_optimizer, batched_example))

  try:
    max_global_norm = gin.query_parameter(
        "train_util.training_loop.max_global_norm")
  except ValueError:
    max_global_norm = None

  def go():
    # Note that value for learning_rate is arbitrary, but we pass it here to
    # warm up the jit cache (since we are passing a learning rate at training
    # time).
    res = parallel_train_step(
        replicated_optimizer,
        batched_example,
        static_batch_metadata,
        loss_fn,
        max_global_norm=max_global_norm,
        learning_rate=0.0)
    jax.tree_map(lambda x: x.block_until_ready(), res)

  if profile:
    stats = runner.try_run_and_profile(go, catch_resource_exhausted=False)
    logging.info("Warmed up train step with stats: %s", stats)
  else:
    go()
    logging.info("Warmed up train step")


def build_averaging_validator(
    loss_fn,
    valid_iterator_factory,
    objective_metric_name = None,
    include_total_counts = False,
    prefetch = True,
):
  """Validate by computing averages over a validation set.

  Args:
    loss_fn: Loss function for the task.
    valid_iterator_factory: Constructs iterators of batched examples from the
      validation set, with two batch axes. To iterate over a fixed part of the
      validation set, consider using build_one_pass_iterator_factory. To
      randomly sample from a validation set, you can use something like
      `lambda: itertools.islice(validation_iterator, num_batches)`.
    objective_metric_name: Name of the metric that is the objective value.
    include_total_counts: Whether to report numerator and denominator separately
      for RatioMetric objects, along with the "validation_total_example_count"
      metric.
    prefetch: Whether to prefetch validation examples.

  Returns:
    Validation function that runs loss_fn and aggregates the results, reporting
    the loss as the objective, and using sum to accumulate metrics.
  """
  if objective_metric_name is None:
    objective_metric_name = "loss"

  @functools.partial(
      jax.pmap, axis_name="devices", static_broadcasted_argnums=3)
  def parallel_metrics_batch(model, batched_examples, batch_mask,
                             static_metadata):
    loss, metrics = jax.vmap(loss_fn, (None, 0, None))(model, batched_examples,
                                                       static_metadata)
    metrics["loss"] = loss
    metrics = jax.tree_map(
        lambda x: jnp.where(batch_mask, x, jnp.zeros_like(x)), metrics)
    metrics = jax.tree_map(lambda x: jax.lax.psum(jnp.sum(x), "devices"),
                           metrics)
    return metrics

  def validation_function(model):
    with contextlib.ExitStack() as exit_stack:
      valid_iterator = valid_iterator_factory()
      if prefetch:
        valid_iterator = exit_stack.enter_context(
            data_loading.ThreadedPrefetcher(valid_iterator, 4))
      accumulated = None
      example_count = 0
      for batch in valid_iterator:
        results = parallel_metrics_batch(model, batch.example, batch.mask,
                                         batch.static_metadata)
        metrics = jax.tree_map(float, flax.jax_utils.unreplicate(results))
        metrics["epoch"] = np.sum(batch.epoch)
        if accumulated is None:
          accumulated = metrics
        else:
          accumulated = jax.tree_multimap(operator.add, accumulated, metrics)
        example_count += jnp.count_nonzero(batch.mask)

      assert example_count > 0, "Validation iterator must be nonempty"
      accumulated = typing.cast(Dict[str, Any], accumulated)

      final_metrics = {}
      for k, v in accumulated.items():
        if isinstance(v, RatioMetric):
          final_metrics[k] = v.numerator / v.denominator
          if include_total_counts:
            final_metrics[k + "_numerator"] = v.numerator
            final_metrics[k + "_denominator"] = v.denominator
        else:
          final_metrics[k] = v / example_count

      objective = final_metrics[objective_metric_name]
      if include_total_counts:
        final_metrics["validation_total_example_count"] = example_count
      return (objective, final_metrics)

  return validation_function


@contextlib.contextmanager
def catch_interrupts_once(callback,
                          catch_signals = (signal.SIGINT,
                                                          signal.SIGABRT)):
  # pylint: disable=g-doc-return-or-yield
  """Context manager to catch interrupt signals.

  Only catches the first signal sent, so that repeated interrupts will kill the
  job as normal.

  Args:
    callback: Function to run when the signal is caught the first time.
    catch_signals: Signals to catch.

  Returns:
    A context manager that will catch interrupts inside the block.
  """

  # pylint: enable=g-doc-return-or-yield
  known_signals = {
      signal.SIGINT: "SIGINT",
      signal.SIGABRT: "SIGABRT",
  }

  def _handler(signal_number, frame):
    del frame  # Unused.
    logging.warning("Caught interrupt signal %s",
                    known_signals.get(signal_number, signal_number))
    callback(signal_number)
    _restore_handlers()

  original_handlers = {}
  for signal_number in catch_signals:
    original_handlers[signal_number] = signal.signal(signal_number, _handler)

  already_restored = False

  def _restore_handlers():
    nonlocal already_restored
    if already_restored:
      return
    else:
      already_restored = True
      for signal_number in catch_signals:
        current_handler = signal.signal(signal_number,
                                        original_handlers[signal_number])
        if current_handler is not _handler:
          logging.error(
              "Unexpected active signal handler %s for %s; "
              "expected the signal hander from "
              "`catch_interrupts_once`! Restored to %s anyways.",
              current_handler, known_signals.get(signal_number, signal_number),
              original_handlers[signal_number])

  try:
    yield
  finally:
    _restore_handlers()
