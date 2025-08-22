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

"""Run masked LM/next-sentence prediction pre-training."""

import functools
import os
import time
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

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

from sparse_mixers import checkpoints
from sparse_mixers import core_utils
from sparse_mixers import input_pipeline
from sparse_mixers import models
from sparse_mixers import train_utils
from sparse_mixers.models import PretrainingStats
import sentencepiece as spm

# Type Stubs
Batch = train_utils.Batch
FlaxTrainState = train_utils.FlaxTrainState
Loss = train_utils.Loss
Params = train_utils.Params
PRNGKey = train_utils.PRNGKey


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
      "type_ids":
          jnp.ones((1, config.max_seq_length), jnp.int32),
      "masked_lm_positions":
          jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_labels":
          jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_weights":
          jnp.ones((1, config.max_predictions_per_seq), config.dtype),
      "next_sentence_labels":
          jnp.ones((1, 1), jnp.int32)
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
      ckpt_state = _replicate_and_shard_target(ckpt_state, sharded_match_fn,
                                               not_sharded_match_fn)
      state = jax_utils.replicate(state_cpu)
      state = state.restore_state(ckpt_state)
    else:
      # Failing the above attempts, we replicate all parameters (including any
      # experts) equally across all devices.
      state = jax_utils.replicate(state_cpu)

  return state, start_step


def _init_train_and_eval_ds(
    tokenizer,
    config
):
  """Initialize training and evaluation datasets.

  Args:
    tokenizer: Tokenizer for converting text to integers representations.
    config: Model and training specifications.

  Returns:
    - Training dataset.
    - Evaluation dataset.
  """
  n_hosts = jax.process_count()
  c4_masked_lm_inputs = functools.partial(
      input_pipeline.c4_masked_lm_inputs,
      tokenizer=tokenizer,
      max_seq_length=config.max_seq_length,
      max_predictions_per_seq=config.max_predictions_per_seq,
      masking_rate=config.masking_rate,
      mask_token_proportion=config.mask_token_proportion,
      random_token_proportion=config.random_token_proportion)
  train_ds = c4_masked_lm_inputs(batch_size=config.train_batch_size // n_hosts)
  eval_ds = c4_masked_lm_inputs(batch_size=config.eval_batch_size // n_hosts)
  return train_ds, eval_ds


def _compute_loss_and_metrics(
    params, batch, rng, model,
    is_experts_model, auxiliary_loss_factor,
    router_z_loss_factor):
  """Computes cross-entropy loss and metrics for MLM and NSP tasks.

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
    - Raw metrics (predictions and example labels).
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "type_ids": batch["type_ids"],
      "masked_lm_positions": batch["masked_lm_positions"],
      "masked_lm_labels": batch["masked_lm_ids"],
      "masked_lm_weights": batch["masked_lm_weights"],
      "next_sentence_labels": batch["next_sentence_labels"]
  }
  dropout_key, jitter_key = random.split(rng)
  output, state = model.apply({"params": params},
                              rngs={
                                  "dropout": dropout_key,
                                  "jitter": jitter_key
                              },
                              mutable=["intermediates"],
                              **inputs)

  # To correctly normalize the MLM and NSP losses, we must first sum the model
  # output across all devices.
  output = jax.lax.psum(output, axis_name="batch")
  total_loss = (
      output.masked_lm_loss / output.masked_lm_normalization +
      output.next_sentence_loss / output.num_next_sentence_labels)

  if is_experts_model:
    # Experts are sharded so we can gather their metrics independently on each
    # device.
    expert_metrics = train_utils.summarize_expert_metrics(
        state, auxiliary_loss_factor, router_z_loss_factor)
    total_loss += expert_metrics.auxiliary_loss + expert_metrics.router_z_loss
    output = output.replace(expert_metrics=expert_metrics)

  return total_loss, output


def _compute_eval_stats(params, batch,
                        model):
  """Computes pre-training task predictions and stats.

  Args:
    params: Model state (parameters).
    batch: Current batch of examples.
    model: The model itself. Flax separates model state and architecture.

  Returns:
    Model predictions and metrics.
  """
  inputs = {
      "input_ids": batch["input_ids"],
      "type_ids": batch["type_ids"],
      "masked_lm_positions": batch["masked_lm_positions"],
      "masked_lm_labels": batch["masked_lm_ids"],
      "masked_lm_weights": batch["masked_lm_weights"],
      "next_sentence_labels": batch["next_sentence_labels"],
      "deterministic": True
  }

  return model.apply({"params": params}, **inputs)


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
  tokenizer.SetEncodeExtraOptions("")
  # Note: [CLS] and [SEP] will be added by the data pipeline, not the tokenizer.

  with config.unlocked():
    config.vocab_size = tokenizer.GetPieceSize()
    config.pad_id = tokenizer.pad_id()

  config = ml_collections.FrozenConfigDict(config)

  model = models.PreTrainingModel(config=config)
  rng = random.PRNGKey(config.seed)
  rng, init_rng = random.split(rng)
  params = _init_params(model, init_rng, config)

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * linear_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=config.num_warmup_steps,
      decay_steps=config.num_train_steps - config.num_warmup_steps,
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
  train_ds, eval_ds = _init_train_and_eval_ds(tokenizer, config)
  train_iter = iter(train_ds)

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

  eval_step = functools.partial(_compute_eval_stats, model=model)
  p_eval_step = jax.pmap(eval_step, axis_name="batch")

  seconds = 0.
  train_stats = []
  logging.info("Starting training loop.")
  logging.info("====================")

  for step in range(start_step, config.num_train_steps):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      train_batch = next(train_iter)
      train_batch = common_utils.shard(train_batch)

      tick = time.time()
      state, train_step_stats, rngs = p_train_step(state, train_batch, rng=rngs)
      if config.measure_step_speed:
        jax.tree.map(lambda opt: opt.block_until_ready(), state)
        tock = time.time()
        seconds += tock - tick

      train_stats.append(train_step_stats)

    if (step > 0 and config.save_checkpoints_steps and
        step % config.save_checkpoints_steps == 0):
      # We allow all hosts to potentially save checkpoints because some model
      # parameters are sharded across devices. Parameters replicated across
      # devices (i.e. not sharded) will only be checkpointed by host 0.
      unreplicated_state = jax.tree.map(
          np.array,
          core_utils.tree_unreplicate_by_name(state, not_sharded_match_fn))
      checkpoints.save_checkpoint(
          workdir,
          unreplicated_state,
          sharded_match_fn,
          step,
          keep=config.checkpoints_to_keep)
      del unreplicated_state  # Only used for checkpointing.

    # Periodic metric handling.
    if step % config.eval_frequency != 0 and step > 0:
      continue

    logging.info("Gathering training metrics at step: %d", step)
    train_metrics = train_utils.collect_metrics(train_stats)
    train_summary = train_utils.compute_pretraining_metrics(train_metrics)
    train_summary["learning_rate"] = learning_rate_fn(step)
    if config.measure_step_speed:
      train_summary["steps_per_sec"] = (step - start_step + 1) / seconds

    if jax.process_index() == 0:
      assert train_summary_writer
      for key, val in train_summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
    # Reset metric accumulation for next training evaluation cycle.
    train_stats = []

    logging.info("Gathering evaluation metrics at step: %d", step)

    eval_stats = []
    for _, eval_batch in zip(range(config.max_num_eval_steps), eval_ds):
      eval_batch = common_utils.shard(eval_batch)
      eval_stats.append(p_eval_step(state.params, eval_batch))
    eval_metrics = train_utils.collect_metrics(eval_stats)
    eval_summary = train_utils.compute_pretraining_metrics(
        eval_metrics, record_grad_norm=False)

    if jax.process_index() == 0:
      assert eval_summary_writer
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, step)
      eval_summary_writer.flush()
