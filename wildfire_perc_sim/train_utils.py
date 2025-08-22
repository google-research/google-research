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

"""Shared utilities for Training."""

import dataclasses as dc
import os
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, TypeVar
from typing import Type

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax import linen as nn
from flax import struct
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
from jax import random
import optax
import tensorflow as tf

from wildfire_perc_sim import config
from wildfire_perc_sim import datasets
from wildfire_perc_sim import utils

TES = TypeVar('TES', bound='TrainEvalSetup')


@struct.dataclass
class TrainState:
  """Data structure for checkpoint the model."""
  step: int
  params: frozen_dict.FrozenDict
  opt_state: optax.OptState


class EvalStepCallable(Protocol):

  def __call__(self, m, ts,
               b, k,
               *args):
    Ellipsis


@struct.dataclass
class EvalMetrics(metrics.Collection):

  hstate_loss: metrics.Average.from_output('hstate_loss')
  hstate_seq_loss: metrics.Average.from_output('hstate_seq_loss')
  observation_loss: metrics.Average.from_output('observation_loss')


@struct.dataclass
class TrainMetrics(metrics.Collection):
  """Stores Metrics during Training."""

  loss: metrics.Average.from_output('loss')
  loss_std: metrics.Std.from_output('loss')
  hstate_loss: metrics.Average.from_output('hstate_loss')
  hstate_loss_std: metrics.Std.from_output('hstate_loss')
  hstate_seq_loss: metrics.Average.from_output('hstate_seq_loss')
  hstate_seq_loss_std: metrics.Std.from_output('hstate_seq_loss')
  observation_loss: metrics.Average.from_output('observation_loss')
  observation_loss_std: metrics.Std.from_output('observation_loss')
  kld_loss: metrics.Average.from_output('kld_loss')
  kld_loss_std: metrics.Std.from_output('kld_loss')


def l2_reconstruction_loss(x, y):
  """L2 Distance between `x` and `y`."""
  return jnp.mean((x - y)**2)


def kl_divergence(mean, logvar):
  """KL Divergence with N(0, 1)."""
  return 0.5 * jnp.mean(jnp.exp(logvar) + jnp.square(mean) - 1 - logvar)


def binary_cross_entropy_loss(x, y):
  """"Binary Cross Entropy Loss."""
  return -jnp.mean(y * jnp.log(jnp.clip(x, min=utils.EPS)) +
                   (1 - y) * jnp.log(jnp.clip(1 - x, min=utils.EPS)))


def logit_binary_cross_entropy_loss(x,
                                    y):
  """Logit Binary Cross Entropy Loss."""
  return optax.sigmoid_binary_cross_entropy(x, y).mean()


def evaluate(model, state, eval_ds,
             step_fn, rng,
             *args):
  """Evaluate the model on the given dataset."""
  eval_metrics = None
  for batch in eval_ds:
    rng, step_rng = random.split(rng)
    step_rng = random.split(step_rng, jax.local_device_count())
    batch = datasets.preprocess_fn(batch)
    update = flax.jax_utils.unreplicate(
        step_fn(model, state, batch, step_rng, *args))
    eval_metrics = (
        update if eval_metrics is None else eval_metrics.merge(update))
  assert eval_metrics is not None  # Needed for type-checking to succeed.
  return eval_metrics


def with_warmup(warmup_steps, end_value,
                remainder_schedule,
                warmup_init_value = 0):
  if not warmup_steps:
    return remainder_schedule
  init_value = warmup_init_value if warmup_init_value >= 0 else end_value
  warmup_fn = optax.linear_schedule(
      init_value=init_value, end_value=end_value, transition_steps=warmup_steps)
  return optax.join_schedules(
      schedules=[warmup_fn, remainder_schedule], boundaries=[warmup_steps])


def cosine_schedule(init_value = .1,
                    num_training_steps = 250_000,
                    warmup_epochs = 5.0,
                    end_value = 0.0,
                    steps_per_epoch = 2_000,
                    warmup_init_value = 0.):
  """Warm up, then apply cosine decay.

  Args:
    init_value: The initial learning rate.
    num_training_steps: The number of steps to train for.
    warmup_epochs: Number of epochs for which to ramp up learning rate.
    end_value: the learning rate value at the end of the training
    steps_per_epoch: How many steps per epoch (used for learning rate decay)
    warmup_init_value: what learning rate warm up from.
  Returns:
    A schedule based on the parameters.
  """
  warmup_steps = int(warmup_epochs * steps_per_epoch)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=init_value,
      alpha=end_value / init_value,
      decay_steps=max(1, num_training_steps - warmup_steps))
  return with_warmup(warmup_steps, init_value, cosine_fn,
                     warmup_init_value=warmup_init_value)


def get_schedule(cfg, num_train_steps
                 ):
  return cosine_schedule(init_value=cfg.init_value,
                         num_training_steps=num_train_steps,
                         warmup_epochs=cfg.warmup_epochs,
                         warmup_init_value=cfg.warmup_init_value,
                         end_value=cfg.end_value,
                         steps_per_epoch=cfg.steps_per_epoch)


def get_weight_decay(cfg
                     ):
  if cfg.weight_decay is None:
    return optax.identity()
  mask_fn = lambda p: jax.tree_util.tree_map(lambda x: x.ndim > 1, p)
  return optax.add_decayed_weights(cfg.weight_decay, mask_fn)


def get_optimizer(cfg):
  lr = get_schedule(cfg.opt.schedule, num_train_steps=cfg.train.num_train_steps)
  main_opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
  return optax.chain(get_weight_decay(cfg.opt), main_opt)


@dc.dataclass
class TrainEvalSetup:
  """Prepares Training and Evaluation.

  Use `TrainEvalSetup.create` to create an instance of this dataclass.

  Attributes:
    global_rng: PRNGKey
    model: Neural Network being trained
    variables: Parameters and States of the `model`
    datasets: Tuple of `train`, `eval` and `test` datasets. Can be a sequence of
      `None`s, if dataloading is skipped when `create` is called
    optimizer: optax Optimizer State
    tstate: Training State (can be serialized)
    loggers: Loggers
    checkpoint: Checkpoint Utility
    step: Initial step
  """
  global_rng: jax.Array
  model: nn.Module
  variables: frozen_dict.FrozenDict
  datasets: Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset],
                  Optional[tf.data.Dataset]]
  optimizer: optax.OptState
  tstate: TrainState
  loggers: Mapping[str, Any]
  checkpoint: checkpoint.Checkpoint
  step: int

  @classmethod
  def create(
      cls,
      cfg,
      get_model,
      testing = False,  # For testing the pipeline on CI
  ):
    """Prepare training and evaluation."""
    workdir = cfg.train.output_dir
    tf.io.gfile.makedirs(workdir)
    rng = random.PRNGKey(cfg.global_init_rng)

    # Input pipeline.
    rng, data_rng = random.split(rng)
    if not testing:
      # Make sure each host uses a different RNG for the training data.
      data_rng = random.fold_in(data_rng, jax.process_index())
      _, train_ds, eval_ds, test_ds = datasets.create_datasets(
          cfg.data, data_rng)
    else:
      train_ds, eval_ds, test_ds = None, None, None

    # Initialize model
    rng, model_rng = random.split(rng)
    model, variables = get_model(cfg.model, model_rng)
    parameter_overview.log_parameter_overview(variables)  # pytype: disable=wrong-arg-types

    tx = get_optimizer(cfg)
    opt_state = tx.init(variables['params'])
    state = TrainState(step=1, opt_state=opt_state, params=variables['params'])

    checkpoint_dir = os.path.join(workdir, 'checkpoints')
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=5)
    if not testing:
      ckpt_ = ckpt.get_latest_checkpoint_to_restore_from()
      if ckpt_ is not None:
        state = ckpt.restore(state, ckpt_)
      elif jax.process_index() == 0:
        ckpt.save(state)

    initial_step = int(state.step)
    # Replicate our parameters.
    state = flax.jax_utils.replicate(state)

    if not testing:
      # Only write metrics on host 0, write to logs on all other hosts.
      writer = metric_writers.create_default_writer(
          workdir, just_logging=jax.process_index() > 0)
      writer.write_hparams(dc.asdict(cfg))

      logging.info('Starting training loop at step %d.', initial_step)
      report_progress = periodic_actions.ReportProgress(
          num_train_steps=cfg.train.num_train_steps, writer=writer)

      loggers = {'writer': writer, 'report_progress': report_progress}
    else:
      loggers = {'writer': None, 'report_progress': None}

    return cls(
        global_rng=rng,
        model=model,
        variables=variables,
        datasets=(train_ds, eval_ds, test_ds),
        optimizer=tx,
        tstate=state,
        loggers=loggers,
        checkpoint=ckpt,
        step=initial_step)
