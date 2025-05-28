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

"""Run sequence-level classification (and regression) fine-tuning."""

import functools
import math
import os
from typing import Callable, Dict, Mapping, Optional, Tuple, Union

from absl import logging
from flax import jax_utils
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scipy import stats as scipy_stats
from sklearn import metrics as skl_metrics
import tensorflow_datasets as tfds

from sparse_mixers import checkpoints
from sparse_mixers import core_utils
from sparse_mixers import input_pipeline
from sparse_mixers import models
from sparse_mixers import train_utils
import sentencepiece as spm

# Type Stubs
Batch = train_utils.Batch
ClassificationStats = models.ClassificationStats
Loss = train_utils.Loss
Params = train_utils.Params
PRNGKey = train_utils.PRNGKey
FlaxTrainState = train_utils.FlaxTrainState


def _replicate_and_shard_target(target,
                                sharded_match_fn,
                                not_sharded_match_fn):
  """Replicates and shards parameters and state accordingly.

  Args:
    target: Train state or parameters to replicate and shard.
    sharded_match_fn: Filter function for identifying sharded (mixture of
      expert) parameters.
    not_sharded_match_fn: Filter function for identifying replicated parameters.

  Returns:
    Replicated and (potentially) sharded target.
  """
  if sharded_match_fn:
    target = core_utils.tree_replicate_by_name(target, not_sharded_match_fn)
    target = core_utils.tree_shard_by_name(target, sharded_match_fn)
  else:
    target = jax_utils.replicate(target)
  return target


def _clear_pretrained_output_layer(state_cpu,
                                   ckpt_state):
  """Clear ("classification") output layer weights.

  We use a fresh output layer because the classification tasks differ from the
  MLM and NSP pre-training tasks.

  Args:
    state_cpu: CPU-initialized train state, containing shape initialized
      parameters.
    ckpt_state: Initialized model state (parameters) from restored checkpoint.

  Returns:
    Inputs parameters, but with output layer cleared.
  """
  ckpt_state["params"]["classification"] = state_cpu.params["classification"]
  ckpt_state["opt_state"] = core_utils.tree_map_with_names(
      jnp.zeros_like,
      ckpt_state["opt_state"],
      filter_fn=core_utils.match_fn(r".*classification.*"))
  return ckpt_state


def _restore_state_from_checkpoint(
    workdir, state_cpu,
    sharded_match_fn,
    not_sharded_match_fn,
    config):
  """Attempts to restore train state from latest checkpoint or config.

  Args:
    workdir: Working directory for model training. We first attempt to resume
      training from this directory.
    state_cpu: CPU-initialized train state, containing shape initialized
      parameters.
    sharded_match_fn: Filter function for identifying sharded (mixture of
      expert) parameters.
    not_sharded_match_fn: Filter function for identifying replicated parameters.
    config: Model and training configuration.

  Returns:
    - Restored and replicated train state.
    - Start step based on restored model.
  """
  # If current job restarts, attempt to continue from most recent checkpoint.
  state = checkpoints.restore_checkpoint(workdir, state_cpu, sharded_match_fn)

  if state:
    start_step = int(state.step)
    state = _replicate_and_shard_target(state, sharded_match_fn,
                                        not_sharded_match_fn)
  else:
    start_step = 0

    if "init_checkpoint_dir" in config and config.init_checkpoint_dir:
      # Otherwise, try to restore model state from config checkpoint.
      ckpt_state = checkpoints.restore_checkpoint(
          config.init_checkpoint_dir,
          target=None,
          sharded_match_fn=sharded_match_fn)
      ckpt_state = _clear_pretrained_output_layer(state_cpu, ckpt_state)
      ckpt_state = _replicate_and_shard_target(ckpt_state, sharded_match_fn,
                                               not_sharded_match_fn)
      state = jax_utils.replicate(state_cpu)
      state = state.restore_state(ckpt_state)
    else:
      # Failing the above attempts, we replicate all parameters (including any
      # experts) equally across all devices.
      state = jax_utils.replicate(state_cpu)

  return state, start_step


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
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "labels": jnp.ones((1, 1), jnp.int32)
  }

  key, dropout_key, jitter_key = random.split(key, num=3)

  # Ensure parameters created in host RAM. Send them to devices as needed.
  jit_init = jax.jit(model.init, backend="cpu")
  initial_variables = jit_init(
      {
          "params": key,
          "dropout": dropout_key,
          "jitter": jitter_key
      }, **init_batch)

  return initial_variables["params"]


def _compute_loss_and_metrics(
    params, batch, rng,
    model, is_experts_model,
    auxiliary_loss_factor,
    router_z_loss_factor):
  """Computes cross-entropy loss and metrics for classification tasks.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    rng: Random number generator key.
    model: The model itself. Flax separates model state and architecture.
    is_experts_model: If true, treat this model as a mixture of experts model
      and attempt to retrieve expert diversity metrics.
    auxiliary_loss_factor: Factor by which to scale auxiliary load balancing
      loss for mixture of experts models.
    router_z_loss_factor: Factor by which to scale router z-loss for mixture of
      experts models.

  Returns:
    - Model loss.
    - Raw metrics.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "type_ids": batch["type_ids"],
      "labels": batch["label"]
  }

  dropout_key, jitter_key = random.split(rng)
  output, state = model.apply({"params": params},
                              rngs={
                                  "dropout": dropout_key,
                                  "jitter": jitter_key
                              },
                              mutable=["intermediates"],
                              **inputs)

  # To correctly normalize loss, we must first sum the model output across all
  # devices.
  output = jax.lax.psum(output, axis_name="batch")
  total_loss = output.batch_loss / output.num_labels

  if is_experts_model:
    # Experts are sharded so we can gather their metrics independently on each
    # device.
    expert_metrics = train_utils.summarize_expert_metrics(
        state, auxiliary_loss_factor, router_z_loss_factor)
    total_loss += expert_metrics.auxiliary_loss + expert_metrics.router_z_loss
    output = output.replace(expert_metrics=expert_metrics)

  return total_loss, output


def _compute_stats(
    params, batch, model,
    scoring_fn):
  """Runs inference and computes model predictions.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    model: The model itself. Flax separates model state and architecture.
    scoring_fn: Task dependent function mapping raw model output logits to
      prediction.

  Returns:
    Model predictions along with example labels.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "type_ids": batch["type_ids"],
      "deterministic": True
  }

  y = model.apply({"params": params}, **inputs)
  result = {
      "idx": batch["idx"],
      "label": batch["label"],
      "prediction": scoring_fn(y),
      "input_ids": batch["input_ids"],  # Required for SQuAD F1 metric
  }

  return result


def _create_eval_metrics_fn(
    dataset_name
):
  """Creates a function that computes task-relevant metrics.

  Args:
    dataset_name: TFDS name of dataset.

  Returns:
    Relevant metric function.
  """

  def get_accuracy(gold, guess):
    """Computes accuracy."""
    return (gold == guess).mean()

  def get_spearmanr(x, y):
    """Computes Spearman correlation coefficient."""
    return scipy_stats.spearmanr(x, y).correlation

  eval_metrics = {}
  if dataset_name == "glue/stsb":
    eval_metrics["spearmanr"] = get_spearmanr
  elif (dataset_name == "super_glue/multirc" or
        dataset_name == "super_glue/record"):
    # MultiRC and ReCoRD answers are grouped by premise/query (see
    # maybe_group_stats()), so accuracy over the question group is equivalent to
    # the exact match for all question answers.
    eval_metrics["exact_match"] = get_accuracy
  else:
    eval_metrics["accuracy"] = get_accuracy

  if dataset_name in ("glue/mrpc", "glue/qqp", "super_glue/record"):
    eval_metrics["f1"] = lambda gold, guess, ids: skl_metrics.f1_score(  # pylint:disable=g-long-lambda
        gold, guess)
  elif dataset_name == "super_glue/cb":
    eval_metrics["f1"] = lambda gold, guess, ids: skl_metrics.f1_score(  # pylint:disable=g-long-lambda
        gold, guess, average="macro")
  elif dataset_name == "super_glue/multirc":
    # F1 on all answer-options.
    eval_metrics["f1a"] = lambda gold, guess, ids: skl_metrics.f1_score(  # pylint:disable=g-long-lambda
        np.concatenate(gold),
        np.concatenate(guess),
        average="micro")

  def maybe_group_stats(
      stats):
    """Task-dependent pre-processing of raw model stats.

    The original COPA, MultiRC, ReCoRD tasks contain multiple answer examples,
    which our data pipeline has split into single answer examples. Here, we
    regroup the examples by idx.

    - For COPA and ReCoRD, we then use the most likely candidate (per idx) as
      the model's True prediction.
    - For MultRC, we simply group results and run metrics over the groups.
    - All other tasks use the raw (ungrouped) stats.

    Args:
      stats: Raw model predictions and input batch ids.

    Returns:
      - For COPA/ReCoRD: Most likely model candidate predictions and labels.
      - For MultiRC: Grouped predictions and labels.
      - For all other tasks: Unprocessed predictions and labels.
    """
    if (dataset_name == "super_glue/copa" or
        dataset_name == "super_glue/multirc" or
        dataset_name == "super_glue/record"):
      grouped = {  # pylint:disable=g-complex-comprehension
          idx: {
              "prediction": [],
              "label": [],
          } for idx in stats["idx"]
      }
      for idx, prediction, label in zip(stats["idx"], stats["prediction"],
                                        stats["label"]):
        grouped[idx]["prediction"].append(prediction)
        grouped[idx]["label"].append(label)

      if (dataset_name == "super_glue/record" or
          dataset_name == "super_glue/copa"):
        predictions = []
        labels = []
        for idx in grouped:
          i = np.asarray(grouped[idx]["prediction"]).argmax()
          labels.append(grouped[idx]["label"][i])
          # The most likely prediction is always our True prediction.
          predictions.append(True)
        return np.array(labels), np.array(predictions)
      else:
        idxs = grouped.keys()
        predictions = np.array([grouped[idx]["prediction"] for idx in idxs])
        labels = np.array([grouped[idx]["label"] for idx in idxs])
        return labels, predictions

    else:
      return stats["label"], stats["prediction"]

  def metrics_fn(stats):
    labels, predictions = maybe_group_stats(stats)
    res = {}
    for name, fn in eval_metrics.items():
      res[name] = fn(labels, predictions)
    return res

  return metrics_fn  # pytype: disable=bad-return-type  # jax-ndarray


def _evaluate(p_eval_step,
              params, eval_batch):
  """Computes evaluation metrics.

  Args:
    p_eval_step: Parallelized evaluation step computation.
    params: Model state.
    eval_batch: Batch of evaluation examples.

  Returns:
    Raw model predictions and metrics.
  """
  n_devices_per_host = jax.local_device_count()
  batch_size = eval_batch["idx"].shape[0]
  remainder = batch_size % n_devices_per_host
  if remainder:
    pad_amount = n_devices_per_host - remainder

    def pad(x):
      assert x.shape[0] == batch_size
      return np.concatenate([x] + [x[:1]] * pad_amount, axis=0)

    eval_batch = jax.tree.map(pad, eval_batch)

  eval_batch = common_utils.shard(eval_batch)
  metrics = p_eval_step(params, eval_batch)

  metrics = jax.tree.map(np.array, metrics)
  metrics = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), metrics)
  if remainder:
    metrics = jax.tree.map(lambda x: x[:-pad_amount], metrics)

  return metrics


def train_and_evaluate(config, workdir,
                       vocab_filepath):
  """Runs a training and evaluation loop.

  Args:
    config: Model and training configuration.
    workdir: Working directory for checkpoints and TensorBoard summaries. If
      this contains a checkpoint, training will be resumed from the latest
      checkpoint.
    vocab_filepath: Absolute path to SentencePiece vocab model.

  Raises:
    ValueError: If training or eval batch sizes won't fit number of hosts and
      devices, or config is underspecified.
  """
  # Update config before config validation.
  with config.unlocked():
    # Numeric floating point type to use for model computations.
    config.dtype = jnp.float32

  train_utils.validate_config(config)

  per_host_train_batch_size = config.train_batch_size // jax.process_count()
  per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

  if jax.process_index() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(workdir, "train"))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(workdir, "eval"))
  else:
    train_summary_writer = None
    eval_summary_writer = None

  tokenizer = spm.SentencePieceProcessor()
  tokenizer.Load(vocab_filepath)

  ds_info = tfds.builder(config.dataset_name).info
  num_train_examples = ds_info.splits[tfds.Split.TRAIN].num_examples

  num_train_steps = int(num_train_examples * config.num_train_epochs //
                        config.train_batch_size)
  num_warmup_steps = int(config.warmup_proportion * num_train_steps)
  # Round up evaluation frequency to power of 10.
  eval_frequency = int(
      math.ceil(config.eval_proportion * num_train_steps / 10)) * 10

  # STSB is a regression task. COPA and ReCoRD are treated as scalar/regression
  # tasks during training.
  is_regression_task = (
      config.dataset_name == "glue/stsb" or
      config.dataset_name == "super_glue/copa" or
      config.dataset_name == "super_glue/record")
  if is_regression_task:
    num_classes = 1
  else:
    num_classes = ds_info.features["label"].num_classes

  with config.unlocked():
    config.vocab_size = tokenizer.GetPieceSize()
    config.pad_id = tokenizer.pad_id()

  config = ml_collections.FrozenConfigDict(config)
  model = models.SequenceClassificationModel(config, num_classes)
  rng = random.PRNGKey(config.seed)
  rng, init_rng = random.split(rng)
  params = _init_params(model, init_rng, config)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * linear_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=num_warmup_steps,
      decay_steps=num_train_steps - num_warmup_steps,
  )

  tx = optax.adamw(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.01)
  if config.clipped_grad_norm:
    tx = optax.chain(optax.clip_by_global_norm(config.clipped_grad_norm), tx)

  # jit state creation to ensure arrays are created on same device as input
  # (i.e. CPU).
  state_cpu = jax.jit(
      functools.partial(
          FlaxTrainState.create, apply_fn=model.apply, params=params, tx=tx))()

  # We access model params only via state.params
  del params

  if config.num_experts > 1:
    sharded_match_fn = core_utils.match_fn(r".*expert.*")
    not_sharded_match_fn = lambda name: not sharded_match_fn(name)
  else:
    sharded_match_fn = None
    not_sharded_match_fn = lambda name: True

  state, start_step = _restore_state_from_checkpoint(workdir, state_cpu,
                                                     sharded_match_fn,
                                                     not_sharded_match_fn,
                                                     config)

  if is_regression_task:
    scoring_fn = lambda y: y[Ellipsis, 0]
  else:
    scoring_fn = lambda y: y.argmax(-1)
  compute_stats = functools.partial(
      _compute_stats, model=model, scoring_fn=scoring_fn)

  classification_inputs = functools.partial(
      input_pipeline.classification_inputs,
      dataset_name=config.dataset_name,
      max_seq_length=config.max_seq_length,
      tokenizer=tokenizer)
  train_ds = classification_inputs(
      split=tfds.Split.TRAIN,
      batch_size=per_host_train_batch_size,
      training=True)
  train_iter = iter(train_ds)

  if config.dataset_name == "glue/mnli":
    # MNLI contains two validation and test datasets.
    split_suffixes = ["_matched", "_mismatched"]
  else:
    split_suffixes = [""]

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  rngs = random.split(rng, jax.local_device_count())

  loss_and_metrics_fn = functools.partial(
      _compute_loss_and_metrics,
      model=model,
      is_experts_model=config.num_experts > 1,
      auxiliary_loss_factor=config.auxiliary_loss_factor,
      router_z_loss_factor=config.router_z_loss_factor)
  train_step = functools.partial(
      train_utils.pmap_train_step,
      loss_and_metrics_fn=loss_and_metrics_fn,
      axis_name="batch",
      sharded_match_fn=sharded_match_fn,
      gradient_accum_steps=config.gradient_accum_steps)
  p_train_step = jax.pmap(train_step, axis_name="batch")
  p_eval_step = jax.pmap(compute_stats, axis_name="batch")
  eval_metrics_fn = _create_eval_metrics_fn(config.dataset_name)

  train_stats = []
  logging.info("Starting training loop.")
  logging.info("====================")

  for step in range(start_step, num_train_steps):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      train_batch = next(train_iter)
      train_batch = common_utils.shard(train_batch)

      state, train_step_stats, rngs = p_train_step(state, train_batch, rng=rngs)

      train_stats.append(train_step_stats)

    if ((step > 0 and config.save_checkpoints_steps and
         step % config.save_checkpoints_steps == 0) or
        step == num_train_steps - 1):
      # We allow all hosts to potentially save checkpoints because some model
      # parameters are sharded across devices. Parameters replicated across
      # devices (i.e. not sharded) will only be checkpointed by host 0.
      unreplicated_train_state = jax.tree.map(
          np.array,
          core_utils.tree_unreplicate_by_name(state, not_sharded_match_fn))
      checkpoints.save_checkpoint(
          workdir,
          unreplicated_train_state,
          sharded_match_fn,
          step,
          keep=config.checkpoints_to_keep)
      del unreplicated_train_state  # Only used for checkpointing.

    # Periodic metric handling.
    if step % eval_frequency != 0 and step < num_train_steps - 1:
      continue

    logging.info("Gathering training metrics at step: %d", step)
    train_metrics = train_utils.collect_metrics(train_stats)
    train_summary = train_utils.compute_classification_metrics(
        train_metrics, is_regression_task)
    train_summary["learning_rate"] = learning_rate_fn(step)

    if jax.process_index() == 0:
      assert train_summary_writer
      for key, val in train_summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
    # Reset metric accumulation for next training evaluation cycle.
    train_stats = []

    logging.info("Gathering validation metrics at step: %d", step)

    for split_suffix in split_suffixes:
      eval_ds = classification_inputs(
          split=tfds.Split.VALIDATION + split_suffix,
          batch_size=per_host_eval_batch_size,
          training=False)

      eval_stats = []
      for _, eval_batch in zip(range(config.max_num_eval_steps), eval_ds):
        eval_stats.append(_evaluate(p_eval_step, state.params, eval_batch))
      eval_metrics = {}
      for k in eval_stats[0]:  # All batches of output stats are the same size
        eval_metrics[k] = np.concatenate([stat[k] for stat in eval_stats],
                                         axis=0)
      eval_summary = eval_metrics_fn(eval_metrics)

      if jax.process_index() == 0:
        assert eval_summary_writer
        for key, val in eval_summary.items():
          eval_summary_writer.scalar(f"{key}{split_suffix}", val, step)
        eval_summary_writer.flush()
