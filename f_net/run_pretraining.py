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

"""Run masked LM/next-sentence prediction pre-training."""

import functools
import os
from typing import Any, Dict, Mapping, Tuple

from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np

from f_net import input_pipeline
from f_net import models
from f_net import train_utils
import sentencepiece as spm

# Type Stubs
PRNGKey = Any


def _init_params(model, key,
                 config):
  """Initializes model state.

  Args:
    model: Model to initialize.
    key: Random number generator key.
    config: Model specifications; used to configure model input shapes.

  Returns:
    Initial model parameters.
  """
  init_batch = {
      "input_ids":
          jnp.ones((1, config.max_seq_length), jnp.int32),
      "input_mask":
          jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids":
          jnp.ones((1, config.max_seq_length), jnp.int32),
      "masked_lm_positions":
          jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_labels":
          jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_weights":
          jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "next_sentence_labels":
          jnp.ones((1, 1), jnp.int32)
  }

  key, dropout_key = random.split(key)

  jit_init = jax.jit(model.init)
  initial_variables = jit_init({
      "params": key,
      "dropout": dropout_key
  }, **init_batch)
  return initial_variables["params"]


def _create_adam_optimizer(learning_rate,
                           params):
  """Creates Adam optimizer.

  Args:
    learning_rate: Initial learning rate.
    params: Model state (parameters).

  Returns:
    Adam optimizer.
  """
  optimizer_def = optim.Adam(
      learning_rate=learning_rate,
      beta1=0.9,
      beta2=0.999,
      eps=1e-6,
      weight_decay=0.01)
  optimizer = optimizer_def.create(params)
  return optimizer


def _compute_loss_and_metrics(
    params, batch, rng,
    model,
    pad_id):
  """Computes cross-entropy loss and metrics for MLM and NSP tasks.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    rng: Random number generator key.
    model: The model itself. Flax separates model state and architecture.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.

  Returns:
    Model loss and raw metrics (predictions and example labels).
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "input_mask": (batch["input_ids"] != pad_id).astype(np.int32),
      "type_ids": batch["type_ids"],
      "masked_lm_positions": batch["masked_lm_positions"],
      "masked_lm_labels": batch["masked_lm_ids"],
      "masked_lm_weights": batch["masked_lm_weights"],
      "next_sentence_labels": batch["next_sentence_labels"]
  }

  metrics = model.apply({"params": params}, rngs={"dropout": rng}, **inputs)
  return metrics["loss"], metrics


def _compute_eval_stats(params, batch,
                        model,
                        pad_id):
  """Computes pre-training task predictions and stats.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    model: The model itself. Flax separates model state and architecture.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.

  Returns:
    Model predictions and metrics.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "input_mask": (batch["input_ids"] != pad_id).astype(np.int32),
      "type_ids": batch["type_ids"],
      "masked_lm_positions": batch["masked_lm_positions"],
      "masked_lm_labels": batch["masked_lm_ids"],
      "masked_lm_weights": batch["masked_lm_weights"],
      "next_sentence_labels": batch["next_sentence_labels"],
      "deterministic": True
  }

  return model.apply({"params": params}, **inputs)


def _compute_loss_and_accuracy_metrics(
    stats):
  """Computes loss and accuracy metrics.

  Args:
    stats: Raw model predictions and example labels; see models.PreTrainingModel
      for keys.

  Returns:
    Model loss and accuracy metrics.
  """
  metrics = {
      "masked_lm_loss":
          jnp.sum(stats["masked_lm_loss"]) /
          jnp.sum(stats["masked_lm_normalization"]),
      "next_sentence_loss":
          jnp.sum(stats["next_sentence_loss"]) /
          jnp.sum(stats["num_next_sentence_labels"]),
      "masked_lm_accuracy":
          jnp.sum(stats["masked_lm_correct"]) /
          jnp.sum(stats["masked_lm_total"]),
      "next_sentence_accuracy":
          jnp.sum(stats["next_sentence_correct"]) /
          jnp.sum(stats["num_next_sentence_labels"]),
  }
  metrics["loss"] = metrics["masked_lm_loss"] + metrics["next_sentence_loss"]
  return metrics


def train_and_evaluate(config, workdir,
                       vocab_filepath):
  """Runs a training and evaluation loop.

  Args:
    config: Model and training configuration.
    workdir: Working directory for checkpoints and Tensorboard summaries. If
      this contains a checkpoint, training will be resumed from the latest
      checkpoint.
    vocab_filepath: Absolute path to SentencePiece vocab model.

  Raises:
    ValueError: If training or eval batch sizes won't fit number of processes
      and devices, or config is underspecified.
  """
  n_processes = jax.process_count()  # Number of processes
  n_devices = jax.local_device_count()  # Number of local devices per process

  if config.train_batch_size % (n_processes * n_devices) > 0:
    raise ValueError(
        "Training batch size must be divisible by the total number of devices, "
        "but training batch size = %d, while total number of devices = %d "
        "(%d processes, each with %d devices)" %
        (config.train_batch_size, n_processes * n_devices, n_processes,
         n_devices))

  if config.eval_batch_size % (n_processes * n_devices) > 0:
    raise ValueError(
        "Eval batch size must be divisible by the total number of devices, "
        "but eval batch size = %d, while total number of devices = %d "
        "(%d processes, each with %d devices)" %
        (config.eval_batch_size, n_processes * n_devices, n_processes,
         n_devices))

  per_process_train_batch_size = config.train_batch_size // n_processes
  per_process_eval_batch_size = config.eval_batch_size // n_processes

  if jax.process_index() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(workdir, "train"))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(workdir, "eval"))
  else:
    train_summary_writer = None
    eval_summary_writer = None

  rng = random.PRNGKey(config.seed)
  rng, init_rng = random.split(rng)

  tokenizer = spm.SentencePieceProcessor()
  tokenizer.Load(vocab_filepath)
  tokenizer.SetEncodeExtraOptions("")
  # Note: [CLS] and [SEP] will be added by the data pipeline, not the tokenizer.

  with config.unlocked():
    config.vocab_size = tokenizer.GetPieceSize()
  frozen_config = ml_collections.FrozenConfigDict(config)
  model = models.PreTrainingModel(config=frozen_config, random_seed=config.seed)

  params = _init_params(model, init_rng, frozen_config)

  optimizer = _create_adam_optimizer(config.learning_rate, params)
  # We access model state only from optimizer via optimizer.target.
  del params

  # In case current job restarts, ensure that we continue from where we left
  # off.
  optimizer = checkpoints.restore_checkpoint(workdir, optimizer)
  start_step = int(optimizer.state.step)

  # Otherwise, try to restore optimizer and model state from config checkpoint.
  if start_step == 0 and "init_checkpoint_dir" in config and config.init_checkpoint_dir:
    optimizer = checkpoints.restore_checkpoint(config.init_checkpoint_dir,
                                               optimizer)

  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * linear_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=config.num_warmup_steps,
      decay_steps=config.num_train_steps - config.num_warmup_steps,
  )

  c4_masked_lm_inputs = functools.partial(
      input_pipeline.c4_masked_lm_inputs,
      tokenizer=tokenizer,
      max_seq_length=config.max_seq_length,
      max_predictions_per_seq=config.max_predictions_per_seq,
      masking_rate=config.masking_rate,
      mask_token_proportion=config.mask_token_proportion,
      random_token_proportion=config.random_token_proportion)
  train_ds = c4_masked_lm_inputs(batch_size=per_process_train_batch_size)
  train_iter = iter(train_ds)
  eval_ds = c4_masked_lm_inputs(batch_size=per_process_eval_batch_size)

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  rngs = random.split(rng, n_devices)

  loss_and_metrics_fn = functools.partial(
      _compute_loss_and_metrics, model=model, pad_id=tokenizer.pad_id())
  p_train_step = jax.pmap(
      functools.partial(
          train_utils.train_step,
          loss_and_metrics_fn=loss_and_metrics_fn,
          learning_rate_fn=learning_rate_fn,
          clipped_grad_norm=config.clipped_grad_norm),
      axis_name="batch")

  metric_fn = functools.partial(
      _compute_eval_stats, model=model, pad_id=tokenizer.pad_id())
  p_eval_step = jax.pmap(
      functools.partial(train_utils.eval_step, metric_fn=metric_fn),
      axis_name="batch")

  train_metrics = []
  logging.info("Starting training loop.")
  logging.info("====================")

  for step in range(start_step, config.num_train_steps):
    with jax.profiler.StepTraceContext("train", step_num=step):
      train_batch = next(train_iter)
      train_batch = common_utils.shard(train_batch)

      optimizer, train_step_metrics, rngs = p_train_step(
          optimizer, train_batch, rng=rngs)
      train_metrics.append(train_step_metrics)

    if (step > 0 and config.save_checkpoints_steps and
        step % config.save_checkpoints_steps == 0 and jax.process_index() == 0):
      # Save un-replicated optimizer + model state.
      checkpoints.save_checkpoint(
          workdir, jax_utils.unreplicate(optimizer), step, keep=2)

    # Periodic metric handling.
    if step % config.eval_frequency != 0 and step > 0:
      continue

    logging.info("Gathering training metrics at step: %d", step)
    train_metrics = common_utils.get_metrics(train_metrics)
    train_summary = _compute_loss_and_accuracy_metrics(train_metrics)
    # Add training specific metrics.
    train_summary["unclipped_grad_l2_norm"] = jnp.sqrt(
        jnp.sum(train_metrics["unclipped_grad_l2_sum"]))
    train_summary["clipped_grad_l2_norm"] = jnp.sqrt(
        jnp.sum(train_metrics["clipped_grad_l2_sum"]))
    train_summary["learning_rate"] = learning_rate_fn(step)

    if jax.process_index() == 0:
      assert train_summary_writer
      for key, val in train_summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
    # Reset metric accumulation for next training evaluation cycle.
    train_metrics = []

    logging.info("Gathering evaluation metrics at step: %d", step)

    all_stats = []
    for _, eval_batch in zip(range(config.max_num_eval_steps), eval_ds):
      eval_batch = common_utils.shard(eval_batch)
      all_stats.append(p_eval_step(optimizer.target, eval_batch))
    flat_stats = {}
    for k in all_stats[0]:
      flat_stats[k] = np.concatenate([stats[k] for stats in all_stats], axis=0)
    eval_summary = _compute_loss_and_accuracy_metrics(flat_stats)

    if jax.process_index() == 0:
      assert eval_summary_writer
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, step)
      eval_summary_writer.flush()
