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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

import itertools
import os

from absl import logging
from flax import nn
from flax import optim
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from tensorflow.io import gfile

from flax_models.bert import import_weights
from flax_models.bert import input_pipeline
from flax_models.bert import models
from flax_models.bert import train_utils
import sentencepiece as spm


def create_model(config):
  """Create a model, possibly starting from a pre-trained checkpoint."""
  model_kwargs = dict(config=config,)
  model_def = models.BertForPreTraining.partial(**model_kwargs)
  if "init_checkpoint" in config and config.init_checkpoint:
    initial_params = import_weights.load_params_from_tf(
        init_checkpoint=config.init_checkpoint,
        d_model=config.d_model,
        num_heads=config.num_heads,
        keep_masked_lm_head=True)
  else:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(
          jax.random.PRNGKey(0),
          [((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_predictions_per_seq), jnp.int32)],
          deterministic=True)
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(config, model):
  optimizer_def = optim.Adam(
      learning_rate=config.learning_rate,
      beta1=0.9,
      beta2=0.999,
      eps=1e-6,
      weight_decay=0.0)
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_pretraining_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch["input_ids"],
        (batch["input_ids"] > 0).astype(np.int32),
        batch["type_ids"],
        batch["masked_lm_positions"],
        batch["masked_lm_ids"],
        batch["masked_lm_weights"],
        batch["next_sentence_labels"])
  return metrics["loss"], metrics


def compute_pretraining_stats(model, batch):
  """Used for computing eval metrics during pre-training."""
  with nn.stochastic(jax.random.PRNGKey(0)):
    masked_lm_logits, next_sentence_logits = model(
        batch["input_ids"],
        (batch["input_ids"] > 0).astype(np.int32),
        batch["type_ids"],
        batch["masked_lm_positions"],
        deterministic=True)
    stats = model.compute_metrics(
        masked_lm_logits, next_sentence_logits,
        batch["masked_lm_ids"],
        batch["masked_lm_weights"],
        batch["next_sentence_labels"])

  masked_lm_correct = jnp.sum(
      (masked_lm_logits.argmax(-1) == batch["masked_lm_ids"].reshape((-1,))
       ) * batch["masked_lm_weights"].reshape((-1,)))
  next_sentence_labels = batch["next_sentence_labels"].reshape((-1,))
  next_sentence_correct = jnp.sum(
      next_sentence_logits.argmax(-1) == next_sentence_labels)
  stats = {
      "masked_lm_correct": masked_lm_correct,
      "masked_lm_total": jnp.sum(batch["masked_lm_weights"]),
      "next_sentence_correct": next_sentence_correct,
      "next_sentence_total": jnp.sum(jnp.ones_like(next_sentence_labels)),
      **stats
  }
  return stats


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  model = create_model(config)
  optimizer = create_optimizer(config, model)
  del model  # don't keep a copy of the initial model

  output_dir = os.path.join(workdir, "pretraining")
  gfile.makedirs(output_dir)

  # Restore from a local checkpoint, if one exists.
  optimizer = checkpoints.restore_checkpoint(output_dir, optimizer)
  start_step = int(optimizer.state.step)

  optimizer = optimizer.replicate()

  tokenizer = spm.SentencePieceProcessor()
  if "vocab_file" in config:
    tokenizer.Load(config.vocab_file)
  tokenizer.SetEncodeExtraOptions("")
  # Note: [CLS] and [SEP] will be added by the data pipeline, not the tokenizer

  learning_rate_fn = train_utils.create_learning_rate_scheduler(
      factors="constant * linear_warmup * cosine_decay",
      base_learning_rate=config.learning_rate,
      warmup_steps=config.num_warmup_steps,
      steps_per_cycle=config.num_train_steps - config.num_warmup_steps,
  )

  train_history = train_utils.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_iter = input_pipeline.c4_masked_lm_inputs(
        config.train_batch_size, tokenizer, config.max_seq_length,
        config.max_predictions_per_seq)
    train_step_fn = train_utils.create_train_step(
        compute_pretraining_loss_and_metrics, clip_grad_norm=1.0)

    for step in range(start_step, config.num_train_steps):
      with jax.profiler.StepTraceContext("train", step_num=step):
        batch = next(train_iter)
        optimizer, train_state = train_step_fn(optimizer, batch, train_state)
        if step % config.save_checkpoints_steps == 0 and jax.host_id() == 0:
          checkpoints.save_checkpoint(output_dir,
                                      optimizer.unreplicate(),
                                      step)

  if config.do_eval:
    eval_iter = input_pipeline.c4_masked_lm_inputs(
        config.eval_batch_size, tokenizer, config.max_seq_length,
        config.max_predictions_per_seq)
    eval_iter = itertools.islice(eval_iter, config.max_eval_steps)
    eval_fn = train_utils.create_eval_fn(
        compute_pretraining_stats, sample_feature_name="input_ids")
    eval_stats = eval_fn(optimizer, eval_iter)

    eval_metrics = {
        "loss": jnp.mean(eval_stats["loss"]),
        "masked_lm_loss": jnp.mean(eval_stats["masked_lm_loss"]),
        "next_sentence_loss": jnp.mean(eval_stats["next_sentence_loss"]),
        "masked_lm_accuracy": jnp.sum(
            eval_stats["masked_lm_correct"]
            ) / jnp.sum(eval_stats["masked_lm_total"]),
        "next_sentence_accuracy": jnp.sum(
            eval_stats["next_sentence_correct"]
            ) / jnp.sum(eval_stats["next_sentence_total"]),
    }

    eval_results = []
    for name, val in sorted(eval_metrics.items()):
      line = f"{name} = {val:.06f}"
      print(line, flush=True)
      logging.info(line)
      eval_results.append(line)

    eval_results_path = os.path.join(output_dir, "eval_results.txt")
    with gfile.GFile(eval_results_path, "w") as f:
      for line in eval_results:
        f.write(line + "\n")
