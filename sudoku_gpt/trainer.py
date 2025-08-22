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

"""Transformer LM trainer."""

import functools

from clu import periodic_actions
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import numpy as jnp
import numpy as np
import optax

from sudoku_gpt import model


def lr_scheduler(
    n_tokens, learning_rate, warmup_tokens, final_tokens, config
    ):
  """Learning rate scheduler, adapted from Mikhail Grankin."""

  # Decay the learning rate based on our progress.
  progress = (n_tokens - warmup_tokens) / max(
      1,
      final_tokens - warmup_tokens,
  )
  lr_mult = jnp.where(
      n_tokens < warmup_tokens,
      # Linear warmup.
      n_tokens / jnp.fmax(1, warmup_tokens),
      # Cosine learning rate decay.
      jnp.fmax(config.end_lr_factor, 0.5 * (1.0 + jnp.cos(np.pi * progress))),
  )
  return learning_rate * lr_mult


def train_step(state, batch, config, hyperparams, learning_rate_fn,
               dropout_rng=None):
  """One step of the training loop.

  Args:
    state: train state.
    batch: input batch
    config: experiment config
    hyperparams: hyperparameter dictionary
    learning_rate_fn: learning rate function
    dropout_rng: rng to be used for dropout

  Returns:
    A new train state, train metrics and computed model predictions.
  """

  inputs = batch[:, :-1]
  label = batch[:, 1:]

  dropout_rng = jax.random.fold_in(dropout_rng, state.step)
  dropout_rng_dict = {"dropout": dropout_rng}

  def loss_fn(params):
    corrupted_inputs = inputs
    pred_logits = model.TransformerLMHeadModel(config).apply(
        {"params": params}, corrupted_inputs, rngs=dropout_rng_dict)

    label_one_hot = jax.nn.one_hot(label, num_classes=config.vocab_size)

    assert label_one_hot.shape == pred_logits.shape, ("one hot label shape",
                                                      label_one_hot.shape,
                                                      label.shape,
                                                      pred_logits.shape)
    if "sudoku" in hyperparams.dataset:
      pred_logits_sol = pred_logits[:, :, :]
      label_one_hot_sol = label_one_hot[:, :, :]

      ce_loss = optax.softmax_cross_entropy(
          logits=pred_logits_sol, labels=label_one_hot_sol
      )
      mask = np.repeat(
          np.arange(len(ce_loss[0])).reshape(1, -1), len(ce_loss), axis=0
      )
      avg_ce_loss = (ce_loss * mask).sum() / mask.sum()
      assert avg_ce_loss.ndim == 2, avg_ce_loss.shape
      return jnp.mean(avg_ce_loss), pred_logits
    elif hyperparams.dataset == "othello":
      ce_loss = optax.softmax_cross_entropy(
          logits=pred_logits, labels=label_one_hot
          )

      assert ce_loss.ndim == 2, ce_loss.shape
      return jnp.mean(ce_loss), pred_logits

  step = state.step
  lr = learning_rate_fn(step)
  (loss, pred_logits), grads = jax.value_and_grad(loss_fn,
                                                  has_aux=True)(state.params)
  grads = jax.lax.pmean(grads, "batch")
  new_state = state.apply_gradients(grads=grads)
  metrics = {
      "step": step, "loss": loss * inputs.shape[0], "learning_rate": lr,
      "pred_logits": pred_logits, "weights": inputs.shape[0]
  }
  return new_state, metrics, pred_logits


def get_metrics_report_progress(config, workdir, writer):
  hooks = []

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.max_steps, writer=writer)

  if jax.process_index() == 0:
    hooks += [report_progress,
              periodic_actions.Profile(logdir=workdir, num_profile_steps=5)]
  train_metrics = []
  return hooks, report_progress, train_metrics


def get_state(config, net, initial_variables):
  """Get the train state given an experiment config, a model and initial variables."""
  lr_scheduler_fn = functools.partial(
      lr_scheduler,
      learning_rate=config.learning_rate,
      warmup_tokens=config.warmup_tokens,
      final_tokens=config.max_steps,
      config=config,
  )
  optim_fn = None
  if config.optimizer == "adamw":
    optim_fn = optax.adamw(
        lr_scheduler_fn, weight_decay=config.weight_decay, b1=0.9, b2=0.95
    )
  elif config.optimizer == "lion":
    optim_fn = optax.lion(lr_scheduler_fn, weight_decay=config.weight_decay)

  optimizer = optax.chain(optax.clip_by_global_norm(1), optim_fn)

  state = train_state.TrainState.create(
      apply_fn=net.apply, params=initial_variables["params"],
      tx=optimizer
      )

  return state, lr_scheduler_fn


def train_one_step(
    p_train_step,
    config,
    state,
    step,
    dropout_rngs,
    train_data_iter,
):
  """Single step of the training loop."""
  with jax.profiler.StepTraceAnnotation("train", step_num=step):

    batch = next(train_data_iter)
    inputs = None
    start_index = None
    if "sudoku" in config.dataset:
      inputs = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch[0]))
      if "dependent" in config.start_index:
        start_index = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, batch[2])
        )
      else:
        start_index = np.ones(len(batch[2])) * config.start_index
        start_index = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, start_index)
        )

    elif config.dataset == "othello":
      inputs = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch))

    state, metrics, _ = p_train_step(
        state, inputs, start_index, dropout_rng=dropout_rngs
    )

  return state, metrics
