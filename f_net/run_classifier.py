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

"""Run sequence-level classification (and regression) fine-tuning."""

import functools
import math
import os
from typing import Any, Callable, Dict, Mapping, Tuple

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
from scipy import stats as scipy_stats
import tensorflow_datasets as tfds

from f_net import input_pipeline
from f_net import models
from f_net import train_utils
import sentencepiece as spm

# Type Stubs
PRNGKey = Any


def _restore_pretrained_model(
    optimizer, params,
    config):
  """Restores model state from pre-trained model with fresh output layer.

  We use a fresh output layer because the classification tasks differ from the
  MLM and NSP pre-training tasks.

  Args:
    optimizer: Empty optimizer object to rebuild via deserialized state-dict.
    params: Initialized model state (parameters).
    config: Model configuration.

  Returns:
    Restore model optimizer.
  """
  # target=None returns the state data instead of updating the optimizer.
  ckpt_contents = checkpoints.restore_checkpoint(
      config.init_checkpoint_dir, target=None)

  # "classification" is the name of the output layer.
  output_init_params = params["classification"]
  ckpt_contents["target"]["classification"] = output_init_params
  cls_state = ckpt_contents["state"]["param_states"]["classification"]
  for param in cls_state.keys():
    for grad_key in cls_state[param].keys():
      cls_state[param][grad_key] = jnp.zeros_like(output_init_params[param])

  return optimizer.restore_state(ckpt_contents)


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
      "input_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "input_mask": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "labels": jnp.ones((1, 1), jnp.int32)
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
    params,
    batch,
    rng,
    model,
    pad_id,
):
  """Computes cross-entropy loss and metrics for classification tasks.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    rng: Random number generator key.
    model: The model itself. Flax separates model state and architecture.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.

  Returns:
    Model loss and metrics.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "input_mask": (batch["input_ids"] != pad_id).astype(np.int32),
      "type_ids": batch["type_ids"],
      "labels": batch["label"]
  }

  metrics = model.apply({"params": params}, rngs={"dropout": rng}, **inputs)
  return metrics["loss"], metrics


def _compute_classification_stats(params, batch,
                                  model,
                                  pad_id):
  """Computes classification predictions.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    model: The model itself. Flax separates model state and architecture.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.

  Returns:
    Model predictions along with example labels.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "input_mask": (batch["input_ids"] != pad_id).astype(np.int32),
      "type_ids": batch["type_ids"],
      "deterministic": True
  }

  y = model.apply({"params": params}, **inputs)
  return {
      "idx": batch["idx"],
      "label": batch["label"],
      "prediction": y.argmax(-1)
  }


def _compute_regression_stats(params, batch,
                              model,
                              pad_id):
  """Computes regression predictions.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    model: The model itself. Flax separates model state and architecture.
    pad_id: Token ID representing padding. A mask is used to distinguish padding
      from actual inputs.

  Returns:
    Model predictions along with example labels.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "input_mask": (batch["input_ids"] != pad_id).astype(np.int32),
      "type_ids": batch["type_ids"],
      "deterministic": True
  }

  y = model.apply({"params": params}, **inputs)
  return {
      "idx": batch["idx"],
      "label": batch["label"],
      "prediction": y[Ellipsis, 0],
  }


def _create_eval_metrics_fn(
    dataset_name, is_regression_task
):
  """Creates a function that computes task-relevant metrics.

  Args:
    dataset_name: TFDS name of dataset.
    is_regression_task: If true, includes Spearman's rank correlation
      coefficient computation in metric function; otherwise, defaults to
      accuracy computation.

  Returns:
    Relevant metric function.
  """

  def get_accuracy(guess, gold):
    return (guess == gold).mean()

  def get_mcc(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    tn = ((guess == 0) & (gold == 0)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (mcc_denom + 1e-6)
    return mcc

  def get_f1(guess, gold):
    tp = ((guess == 1) & (gold == 1)).sum()
    fp = ((guess == 1) & (gold == 0)).sum()
    fn = ((guess == 0) & (gold == 1)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    return f1

  def get_f1_accuracy_mean(guess, gold):
    return (get_f1(guess, gold) + get_accuracy(guess, gold)) / 2.0

  def get_spearmanr(x, y):
    return scipy_stats.spearmanr(x, y).correlation

  eval_metrics = {}
  if is_regression_task:
    eval_metrics["spearmanr"] = get_spearmanr
  else:
    eval_metrics["accuracy"] = get_accuracy

  if dataset_name == "glue/cola":
    eval_metrics["mcc"] = get_mcc
  elif dataset_name in ("glue/mrpc", "glue/qqp"):
    eval_metrics["f1_accuracy_mean"] = get_f1_accuracy_mean

  def metrics_fn(stats):
    res = {}
    for name, fn in eval_metrics.items():
      res[name] = fn(stats["prediction"], stats["label"])
    return res

  return metrics_fn


def _evaluate(p_eval_step, model,
              eval_batch,
              n_devices):
  """Computes evaluation metrics.

  Args:
    p_eval_step: Parallelized evaluation step computation.
    model: Model architecture.
    eval_batch: Batch of evaluation examples.
    n_devices: Number of local devices.

  Returns:
    Raw model predictions and metrics.
  """
  batch_size = eval_batch["idx"].shape[0]
  remainder = batch_size % n_devices
  if remainder:
    pad_amount = n_devices - remainder

    def pad(x):
      assert x.shape[0] == batch_size
      return np.concatenate([x] + [x[:1]] * pad_amount, axis=0)

    eval_batch = jax.tree_map(pad, eval_batch)

  eval_batch = common_utils.shard(eval_batch)
  metrics = p_eval_step(model, eval_batch)

  metrics = jax.tree_map(np.array, metrics)
  metrics = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), metrics)
  if remainder:
    metrics = jax.tree_map(lambda x: x[:-pad_amount], metrics)

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

  ds_info = tfds.builder(config.dataset_name).info
  num_train_examples = ds_info.splits[tfds.Split.TRAIN].num_examples

  num_train_steps = int(num_train_examples * config.num_train_epochs //
                        config.train_batch_size)
  num_warmup_steps = int(config.warmup_proportion * num_train_steps)
  # Round up evaluation frequency to power of 10.
  eval_frequency = int(
      math.ceil(config.eval_proportion * num_train_steps / 10)) * 10

  is_regression_task = config.dataset_name == "glue/stsb"

  num_classes = (1 if is_regression_task else
                 ds_info.features["label"].num_classes)

  tokenizer = spm.SentencePieceProcessor()
  tokenizer.Load(vocab_filepath)
  with config.unlocked():
    config.vocab_size = tokenizer.GetPieceSize()

  frozen_config = ml_collections.FrozenConfigDict(config)
  model = models.SequenceClassificationModel(
      config=frozen_config, n_classes=num_classes)

  params = _init_params(model, init_rng, config)

  optimizer = _create_adam_optimizer(config.learning_rate, params)

  # In case current job restarts, ensure that we continue from where we left
  # off.
  optimizer = checkpoints.restore_checkpoint(workdir, optimizer)
  start_step = int(optimizer.state.step)

  # Otherwise, try to restore optimizer and model state from config checkpoint.
  if (start_step == 0 and "init_checkpoint_dir" in config and
      config.init_checkpoint_dir):
    optimizer = _restore_pretrained_model(optimizer, params, config)

  # We access model state only from optimizer via optimizer.target.
  del params

  optimizer = jax_utils.replicate(optimizer)

  if is_regression_task:
    compute_stats = functools.partial(
        _compute_regression_stats, model=model, pad_id=tokenizer.pad_id())
  else:
    compute_stats = functools.partial(
        _compute_classification_stats, model=model, pad_id=tokenizer.pad_id())

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * linear_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=num_warmup_steps,
      decay_steps=num_train_steps - num_warmup_steps,
  )

  glue_inputs = functools.partial(
      input_pipeline.glue_inputs,
      dataset_name=config.dataset_name,
      max_seq_length=config.max_seq_length,
      tokenizer=tokenizer)
  train_ds = glue_inputs(
      split=tfds.Split.TRAIN,
      batch_size=per_process_train_batch_size,
      training=True)
  train_iter = iter(train_ds)

  if config.dataset_name == "glue/mnli":
    # MNLI contains two validation and test datasets.
    split_suffixes = ["_matched", "_mismatched"]
  else:
    split_suffixes = [""]

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  rngs = random.split(rng, n_devices)

  loss_and_metrics_fn = functools.partial(
      _compute_loss_and_metrics, model=model, pad_id=tokenizer.pad_id())
  p_train_step = jax.pmap(
      functools.partial(
          train_utils.train_step,
          loss_and_metrics_fn=loss_and_metrics_fn,
          learning_rate_fn=learning_rate_fn),
      axis_name="batch")
  p_eval_step = jax.pmap(
      functools.partial(train_utils.eval_step, metric_fn=compute_stats),
      axis_name="batch")
  eval_metrics_fn = _create_eval_metrics_fn(config.dataset_name,
                                            is_regression_task)

  train_metrics = []

  logging.info("Starting training loop.")
  logging.info("====================")

  for step in range(start_step, num_train_steps):
    with jax.profiler.StepTraceContext("train", step_num=step):
      train_batch = next(train_iter)
      train_batch = common_utils.shard(train_batch)

      optimizer, train_step_metrics, rngs = p_train_step(
          optimizer, train_batch, rng=rngs)
      train_metrics.append(train_step_metrics)

    if ((step > 0 and config.save_checkpoints_steps and
         step % config.save_checkpoints_steps == 0) or
        step == num_train_steps - 1) and jax.process_index() == 0:
      # Save un-replicated optimizer and model state.
      checkpoints.save_checkpoint(
          workdir, jax_utils.unreplicate(optimizer), step, keep=2)

    # Periodic metric handling.
    if step % eval_frequency != 0 and step < num_train_steps - 1:
      continue

    logging.info("Gathering training metrics at step: %d", step)

    train_metrics = common_utils.get_metrics(train_metrics)
    train_summary = {
        "loss":
            jnp.sum(train_metrics["loss"]) /
            jnp.sum(train_metrics["num_labels"]),
        "learning_rate":
            learning_rate_fn(step)
    }
    if not is_regression_task:
      train_summary["accuracy"] = jnp.sum(
          train_metrics["correct_predictions"]) / jnp.sum(
              train_metrics["num_labels"])

    if jax.process_index() == 0:
      assert train_summary_writer
      for key, val in train_summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
    # Reset metric accumulation for next evaluation cycle.
    train_metrics = []

    logging.info("Gathering validation metrics at step: %d", step)

    for split_suffix in split_suffixes:
      eval_ds = glue_inputs(
          split=tfds.Split.VALIDATION + split_suffix,
          batch_size=per_process_eval_batch_size,
          training=False)

      all_stats = []
      for _, eval_batch in zip(range(config.max_num_eval_steps), eval_ds):
        all_stats.append(
            _evaluate(p_eval_step, optimizer.target, eval_batch, n_devices))
      flat_stats = {}
      for k in all_stats[0]:  # All batches of output stats are the same size
        flat_stats[k] = np.concatenate([stat[k] for stat in all_stats], axis=0)
      eval_summary = eval_metrics_fn(flat_stats)

      if jax.process_index() == 0:
        assert eval_summary_writer
        for key, val in eval_summary.items():
          eval_summary_writer.scalar(f"{key}{split_suffix}", val, step)
        eval_summary_writer.flush()
