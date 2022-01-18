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

# Lint as: python3
"""Simple training runner."""

import contextlib
import itertools
import json
import os
import pickle
import re
import time
import typing
from typing import Any, Callable, Dict, IO, Iterable, Optional, Tuple, Type

from absl import logging
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from gfsa import jax_util
from gfsa.training import learning_rate_schedules
from gfsa.training import train_util


def try_run_and_profile(fun, **kwargs):
  raise NotImplementedError("try_run_and_profile not implemented.")


@gin.configurable
def build_sampling_iterator(
    tfrecord_path,
    example_type,
    num_parallel_reads = 16,
    shuffle_buffer = 2048,
    truncate_at = None,
):
  """Build a sampling dataset iterator for individual examples.

  Args:
    tfrecord_path: Path to the TFRecord files to use. Can include a * glob
      pattern to load multiple files.
    example_type: Dataclass to use to deserialize the results.
    num_parallel_reads: How many files to read from at the same time.
    shuffle_buffer: How many examples to store in the shuffle buffer (after
      interleaving chunks).
    truncate_at: How many examples to produce.

  Yields:
    train_util.ExampleWithMetadata objects, where epoch starts at 0
    and increments every time we make a full pass through the dataset. No
    batching is performed.
  """
  if truncate_at is not None and num_parallel_reads is not None:
    # Can't guarantee iteration order when truncating
    logging.warning("Disabling num_parallel_reads due to truncated dataset.")
    num_parallel_reads = None

  dataset = tf.data.TFRecordDataset(
      tf.io.gfile.glob(tfrecord_path), num_parallel_reads=num_parallel_reads)
  if truncate_at:
    dataset = dataset.take(truncate_at)
  dataset = dataset.shuffle(shuffle_buffer)

  prototype_object = (0, jax_util.synthesize_dataclass(example_type))

  for epoch in itertools.count():
    for item in dataset.as_numpy_iterator():
      ex_id, ex = flax.serialization.from_bytes(
          target=prototype_object, encoded_bytes=item)
      yield train_util.ExampleWithMetadata(epoch, ex_id, ex)


def build_one_pass_iterator_factory(
    tfrecord_path,
    example_type,
    truncate_at = None,
    skip_first = 0,
):
  """Build a deterministic one-epoch iterator for unbatched examples.

  Args:
    tfrecord_path: Path to the TFRecord files to use. Can include a * glob
      pattern to load multiple files.
    example_type: Dataclass to use to deserialize the results.
    truncate_at: Number of examples to truncate the table at. Determines the
      effective size of the dataset.
    skip_first: Number of examples to skip at the beginnning of the dataset.

  Returns:
    Callable with no args that, when called, returns a new dataset iterator.
    This iterator produces train_util.ExampleWithMetadata objects, where epoch
    gives the
    number of times the factory function has been called. Each returned iterator
    will make exactly one pass through the dataset and then stop.
  """
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_path))
  dataset = dataset.skip(skip_first)
  if truncate_at:
    dataset = dataset.take(truncate_at)

  prototype_object = (0, jax_util.synthesize_dataclass(example_type))
  epoch = 0

  def factory():
    nonlocal epoch
    epoch += 1
    for item in dataset.as_numpy_iterator():
      ex_id, ex = flax.serialization.from_bytes(
          target=prototype_object, encoded_bytes=item)
      yield train_util.ExampleWithMetadata(epoch, ex_id, ex)

  return factory


def load_from_checkpoint(
    optimizer_structure,
    checkpoint_path = None,
):
  """Load an optimizer from a checkpoint in the artifacts directory.

  Args:
    optimizer_structure: An optimizer object with the desired tree structure.
      Parameters will be ignored, and will be replaced with values from the
      checkpoint.
    checkpoint_path: File or directory to load the checkpoint from.

  Returns:
    Loaded optimizer, along with extra relevant information about the restored
    state.
  """
  if tf.io.gfile.isdir(checkpoint_path):
    best_step = -1
    best_filename = None
    for filename in tf.io.gfile.listdir(checkpoint_path):
      m = re.match(r"best_at_(\d+)\.msgpack$", filename)
      if m:
        step = int(m.group(1))
        if step > best_step:
          best_step = step
          best_filename = filename
    assert best_filename is not None
    filename = os.path.join(checkpoint_path, best_filename)
    metadata = {"filename": filename}
  else:
    filename = checkpoint_path
    metadata = {}

  with tf.io.gfile.GFile(filename, "rb") as fp:
    loaded = flax.serialization.from_bytes(
        target=optimizer_structure, encoded_bytes=fp.read())

  return loaded, metadata


@gin.configurable
def training_loop(
    optimizer,
    train_iterator,
    loss_fn,
    validation_fn,
    artifacts_dir = None,
    log_dir = None,
    max_iterations = 10_000_000,
    max_seconds = None,
    learning_rate_schedule = (
        gin.REQUIRED),
    max_global_norm = None,
    steps_per_summary = 200,
    steps_per_validate = 200,
    steps_per_save = 1000,
    extra_artifacts = None):
  """Run the main training loop.

  Args:
    optimizer: Initial optimizer object. Also responsible for tracking the model
      state (as optimizer.target).
    train_iterator: Iterator of batched examples from the training set. Should
      have two batch axes (num_devices, batch_size_per_device, ...).
    loss_fn: Loss function to apply to the model, which computes the loss as
      well as a dictionary of scalar metrics. Metrics will be averaged across
      the batch and serialized. If a metric is nan, it will be skipped when
      computing the average.
    validation_fn: Function to evaluate the model on the validation script.
    artifacts_dir: Directory to save artifacts, in particuar the model
      parameters and gin config.
    log_dir: Directory to save TensorBoard summaries.
    max_iterations: Maximum number of steps to take.
    max_seconds: Maximum numer of seconds.
    learning_rate_schedule: Learning rate schedule to use.
    max_global_norm: Maximum gradient global norm to clip to.
    steps_per_summary: Number of steps between each time training metrics are
      computed.
    steps_per_validate: Number of steps between each time validation metrics are
      computed and written to TensorBoard
    steps_per_save: Number of steps between each time the parameters are saved.
    extra_artifacts: Other artifacts to save. Should be a dictionary from
      filenames to objects. If filename ends with ".json" the value will be JSON
      serialized, if it ends in ".pickle" the value will be pickled, otherwise
      it is assumed to be a str/bytes object and will be written directly.

  Returns:
    The final optimizer.

  Raises:
    RuntimeError: If gradients are infinite or NaN at any step.
  """
  logging.info("Setting up training loop.")
  train_iterator = iter(train_iterator)

  if log_dir is not None:
    tf.io.gfile.makedirs(log_dir)
    train_metrics_file = tf.io.gfile.GFile(
        os.path.join(log_dir, "train_metrics.json"), "w")
    valid_metrics_file = tf.io.gfile.GFile(
        os.path.join(log_dir, "valid_metrics.json"), "w")
    train_metrics_file = typing.cast(IO[str], train_metrics_file)
    valid_metrics_file = typing.cast(IO[str], valid_metrics_file)

  # Peek at the first example in our dataset.
  logging.info("Peeking at dataset format...")
  first_batch = next(train_iterator)
  train_iterator = itertools.chain((first_batch,), train_iterator)
  num_devices = first_batch.epoch.shape[0]

  # Vectorize our optimizer accordingly.
  replicated_optimizer = train_util.device_broadcast(optimizer, num_devices)
  del optimizer

  # Prepare for graceful job shutdown
  shutdown_after_this_iteration = False

  def graceful_shutdown_handler(unused_signal_number):
    del unused_signal_number
    nonlocal shutdown_after_this_iteration
    shutdown_after_this_iteration = True

  start_time = last_summary_time = time.time()
  last_summary_step = None
  best_objective_value = np.inf
  best_optimizer = None
  best_at_step = None
  last_best_at_step = None

  if artifacts_dir is None:
    logging.warning("Running training loop without a parameter save directory!")
    should_write_gin_config = False

    def save_checkpoint(step):
      del step
      pass
  else:
    checkpoints_path = os.path.join(artifacts_dir, "checkpoints")
    tf.io.gfile.makedirs(checkpoints_path)
    should_write_gin_config = True

    if extra_artifacts:
      # Save any requested extra artifact files.
      for filename, value in extra_artifacts.items():
        _, ext = os.path.splitext(filename)
        cur_path = os.path.join(artifacts_dir, filename)
        if ext == ".json":
          with tf.io.gfile.GFile(cur_path, "w") as fp:
            json.dump(value, fp, indent=2)
        elif ext == ".pickle":
          with tf.io.gfile.GFile(cur_path, "wb") as fp:
            pickle.dump(value, fp, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(value, (str, bytes)):
          mode = "w" if isinstance(value, str) else "wb"
          with tf.io.gfile.GFile(cur_path, mode) as fp:
            fp.write(value)
        else:
          raise ValueError(
              f"Couldn't save extra artifact {filename}: unrecognized "
              f"extension {ext} and not already serialized")

    def save_checkpoint(step):
      logging.info("Saving a checkpoint at %d", step)
      with tf.io.gfile.GFile(
          os.path.join(checkpoints_path, f"current_at_{step}.msgpack"),
          "wb") as fp:
        fp.write(
            flax.serialization.to_bytes(
                jax.tree_map(np.asarray,
                             flax.jax_utils.unreplicate(replicated_optimizer))))

      nonlocal last_best_at_step
      if best_optimizer and best_at_step != last_best_at_step:
        last_best_at_step = best_at_step
        with tf.io.gfile.GFile(
            os.path.join(checkpoints_path, f"best_at_{best_at_step}.msgpack"),
            "wb") as fp:
          fp.write(flax.serialization.to_bytes(best_optimizer))

  logging.info("Starting main training loop.")
  with contextlib.ExitStack() as exit_stack:
    exit_stack.enter_context(
        train_util.catch_interrupts_once(graceful_shutdown_handler))
    if log_dir is not None:
      exit_stack.enter_context(contextlib.closing(train_metrics_file))  # pytype: disable=wrong-arg-types
      exit_stack.enter_context(contextlib.closing(valid_metrics_file))  # pytype: disable=wrong-arg-types

    for step in range(1, max_iterations + 1):

      # Do a training step.
      cur_learning_rate = learning_rate_schedule.learning_rate_for_step(step)
      batch = next(train_iterator)

      try:
        updated_optimizer, grads_ok, metrics, agg_grads = train_util.parallel_train_step(
            replicated_optimizer,
            batch.example,
            batch.static_metadata,
            loss_fn,
            max_global_norm=max_global_norm,
            learning_rate=cur_learning_rate)
      except BaseException:
        logging.error(
            "Error while processing batch\n%s\nwith static metadata\n%s",
            batch.example_id, batch.static_metadata)
        raise

      grads_ok, metrics = flax.jax_utils.unreplicate((grads_ok, metrics))
      metrics = {metric: float(value) for metric, value in metrics.items()}
      metrics["step"] = step
      metrics["epoch"] = int(np.max(batch.epoch))
      metrics["learning_rate"] = cur_learning_rate

      if not grads_ok:
        # Save current state for debugging purposes.
        save_checkpoint(step)
        if artifacts_dir is not None:
          postmortem_path = os.path.join(artifacts_dir, "postmortem.pickle")
          with tf.io.gfile.GFile(postmortem_path, "wb") as fp:
            postmortem = {
                "step":
                    step,
                "optimizer":
                    flax.serialization.to_state_dict(replicated_optimizer),
                "batch":
                    batch,
                "metrics":
                    metrics,
                "agg_grads":
                    flax.serialization.to_state_dict(agg_grads),
            }
            pickle.dump(postmortem, fp, protocol=pickle.HIGHEST_PROTOCOL)

        bad_grads = jax.tree_map(
            lambda x: float(jnp.count_nonzero(~jnp.isfinite(x)) / x.size),
            flax.serialization.to_state_dict(agg_grads))
        bad_grads_str = json.dumps(bad_grads, indent=2)
        raise RuntimeError(f"Non-finite gradients at step {step}!\n\n"
                           f"Bad fraction:\n{bad_grads_str}")

      elapsed_sec = time.time() - start_time
      metrics["elapsed_hours"] = elapsed_sec / 3600
      if max_seconds is not None and elapsed_sec > max_seconds:
        logging.info("Hit max train timeout at step %d", step)
        shutdown_after_this_iteration = True

      # Do a validation step.
      if validation_fn and step % steps_per_validate == 0:
        logging.info("Running validation at step %d", step)
        objective, valid_metrics = validation_fn(replicated_optimizer.target)
        valid_metrics["step"] = step
        valid_metrics["elapsed_hours"] = elapsed_sec / 3600
        logging.info(
            "valid %d: %s", step,
            " ".join(f"{metric}: {value}"
                     for metric, value in valid_metrics.items()))

        # Update learning rate
        learning_rate_schedule.update_with_validation(objective)

        # Write summaries:
        if log_dir is not None:
          json.dump(
              {metric: float(value) for metric, value in valid_metrics.items()},
              valid_metrics_file)
          valid_metrics_file.write("\n")
          valid_metrics_file.flush()

        # Check for improvements
        if objective < best_objective_value:
          # Copy the current optimizer state into the checkpoint. Don't save
          # it to disk yet to reduce disk writes.
          best_optimizer = jax.tree_map(
              np.asarray, flax.jax_utils.unreplicate(replicated_optimizer))
          best_objective_value = objective
          best_at_step = step
          logging.info("New best validation objective %f at step %d", objective,
                       step)

      # Add summaries.
      if step % steps_per_summary == 0:
        if last_summary_step is not None:
          seconds_per_step = (time.time() - last_summary_time) / (
              step - last_summary_step)
          metrics["seconds_per_step"] = seconds_per_step
        last_summary_time = time.time()
        last_summary_step = step

        # To file
        if log_dir is not None:
          json.dump({metric: float(value) for metric, value in metrics.items()},
                    train_metrics_file)
          train_metrics_file.write("\n")
          train_metrics_file.flush()

        # To log:
        sorted_metric_names = sorted(
            metrics.keys(), key=lambda name: (name != "loss", name))
        logging.info(
            "%d: %s", step, " ".join(
                f"{name}: {metrics[name]}" for name in sorted_metric_names))

      # Save gin config if we haven't done so yet.
      # (We do this here so that anything used during the step shows up in the
      # operative config).
      if should_write_gin_config:
        assert artifacts_dir is not None
        should_write_gin_config = False
        op_config_path = os.path.join(artifacts_dir, "operative_config.gin")

        with tf.io.gfile.GFile(op_config_path, "w") as fp:
          fp.write(gin.operative_config_str())

      # Apply the update.
      replicated_optimizer = updated_optimizer

      should_save = (
          shutdown_after_this_iteration or step % steps_per_save == 0)
      # Save optimizer state.
      if should_save:
        save_checkpoint(step)
        if shutdown_after_this_iteration:
          logging.warning("Shutting down by request after step %d.", step)
          break

    logging.info("Training loop finished after %d steps.", step)
    save_checkpoint(step)
    return replicated_optimizer
