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

"""Training Deterministic Predictor Propagator Models."""
import functools as ft
import os
import sys
import time

from typing import Sequence, Tuple

from absl import app
from absl import logging

from clu import metric_writers
from clu import metrics

import flax
from flax import linen as nn
from flax import struct
from flax.core import frozen_dict

import jax
from jax import numpy as jnp
from jax import random

from ml_collections import config_flags

import numpy as np
import optax
import tensorflow as tf

from wildfire_perc_sim import config
from wildfire_perc_sim import datasets
from wildfire_perc_sim import models
from wildfire_perc_sim import train_utils
from wildfire_perc_sim import utils


@struct.dataclass
class TrainStepConfig:
  forward_observation_length: int
  backward_observation_length: int
  weight_observation: float
  weight_hidden_state: float


@ft.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 1, 4))
def train_step(
    model,
    tx,
    state,
    batch,
    tstep_config,
):
  """Perform a single training step.

  Args:
    model: Module to compute predictions.
    tx: optax GradientTransformation.
    state: Current training state. Updated training state will be returned.
    batch: Training inputs for this step.
    tstep_config: TrainStepConfig.

  Returns:
    Tuple of the updated state and dictionary with metrics.
  """

  def loss_fn(params):
    predicted_hidden_states, predicted_observations = model.apply(
        {'params': params}, batch['observation_sequence']
        [:tstep_config.backward_observation_length],
        tstep_config.forward_observation_length)

    hidden_state_reconstruction_loss = train_utils.l2_reconstruction_loss(
        predicted_hidden_states[0], batch['hidden_state'])

    observation_reconstruction_loss = sum([
        train_utils.logit_binary_cross_entropy_loss(o_pred, o_gt) for o_gt,
        o_pred in zip(batch['observation_sequence'], predicted_observations)
    ])

    loss = (
        tstep_config.weight_observation * observation_reconstruction_loss +
        tstep_config.weight_hidden_state * hidden_state_reconstruction_loss)

    return (loss, (hidden_state_reconstruction_loss,
                   observation_reconstruction_loss))

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (hstate_loss, obs_loss)), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, opt_state = tx.update(grad, state.opt_state, state.params)
  params = optax.apply_updates(state.params, updates)
  new_state = state.replace(
      opt_state=opt_state,
      params=params,
      step=state.step + 1,
  )
  metrics_update = train_utils.TrainMetrics.gather_from_model_output(
      loss=loss,
      hstate_loss=hstate_loss,
      observation_loss=obs_loss,
      hstate_seq_loss=jnp.ones((1,)) * -1,
      kld_loss=jnp.ones((1,)) * -1)
  return new_state, metrics_update


@ft.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 4))
def eval_step(model, state,
              batch, _,
              backward_observation_length):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs using all devices on the host. Afterwards
  metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Module to compute predictions.
    state: Replicated model state.
    batch: Inputs that should be evaluated.
    _: PRNGKeyArray.
    backward_observation_length: Number of observations used for backward model.

  Returns:
    The evaluation metrics.
  """
  logging.info('eval_step(batch=%s)', batch)
  predicted_hidden_states, predicted_observations = model.apply(
      {'params': state.params},
      batch['observation_sequence'][:backward_observation_length],
      len(batch['observation_sequence']), True)

  hidden_state_reconstruction_loss = train_utils.l2_reconstruction_loss(
      predicted_hidden_states[0], batch['hidden_state'])

  observation_reconstruction_loss = sum([
      train_utils.logit_binary_cross_entropy_loss(o_pred, o_gt) for o_gt, o_pred
      in zip(batch['observation_sequence'], predicted_observations)
  ])

  hidden_state_reconstruction_loss_seq = sum([
      train_utils.l2_reconstruction_loss(h_pred, h_gt)
      for h_gt, h_pred in zip(batch['hstate_sequence'], predicted_hidden_states)
  ])

  return train_utils.EvalMetrics.gather_from_model_output(
      hstate_loss=hidden_state_reconstruction_loss,
      hstate_seq_loss=hidden_state_reconstruction_loss_seq,
      observation_loss=observation_reconstruction_loss)


def get_model(
    cfg,
    rng):
  """Construct the model and variables from cfg."""
  model = models.DeterministicPredictorPropagator(
      cfg.observation_channels, cfg.latent_dim, cfg.field_shape,
      cfg.hidden_state_channels, cfg.stage_sizes,
      cfg.decoder_num_starting_filters)
  rng, model_rng = random.split(rng)
  observations = [
      jnp.ones((1, *cfg.field_shape, cfg.observation_channels))
      for _ in range(5)
  ]
  variables = model.init(model_rng, observations)

  return model, variables


def train_and_evaluate(cfg):
  """Run training and evaluation."""
  train_eval_setup = train_utils.TrainEvalSetup.create(cfg, get_model, False)

  train_ds, eval_ds, test_ds = train_eval_setup.datasets  # pylint: disable=unused-variable

  train_iter = iter(train_ds)
  train_metrics = None

  writer = train_eval_setup.loggers['writer']
  report_progress = train_eval_setup.loggers['report_progress']

  tstate = train_eval_setup.tstate
  rng = train_eval_setup.global_rng
  model = train_eval_setup.model
  tx = train_eval_setup.optimizer

  tstep_config = TrainStepConfig(
      forward_observation_length=cfg.train.forward_observation_length,
      backward_observation_length=cfg.train.backward_observation_length,
      weight_observation=cfg.loss.weight_observation,
      weight_hidden_state=cfg.loss.weight_hidden_state,
  )

  with metric_writers.ensure_flushes(writer):
    for step in range(train_eval_setup.step, cfg.train.num_train_steps + 1):
      is_last_step = step == cfg.train.num_train_steps

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        batch = datasets.preprocess_fn(next(train_iter))
        tstate, metrics_update = train_step(
            model, tx, tstate, batch, tstep_config)
        metric_update = flax.jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
      report_progress(step, time.time())

      if step % cfg.train.log_every == 0 or is_last_step or step == 1:
        writer.write_scalars(
            step, utils.prepend_dict_keys(train_metrics.compute(), 'train/'))
        train_metrics = None

      if step % cfg.train.eval_every == 0 or is_last_step:
        rng, eval_rng = random.split(rng)
        with report_progress.timed('eval'):
          eval_metrics = train_utils.evaluate(
              model, tstate, eval_ds, eval_step, eval_rng,
              cfg.train.backward_observation_length)
        eval_metrics_cpu = jax.tree.map(np.array, eval_metrics.compute())
        eval_metrics_cpu = utils.prepend_dict_keys(eval_metrics_cpu, 'eval/')
        writer.write_scalars(step, eval_metrics_cpu)

      if step % cfg.train.save_every == 0 or is_last_step:
        with report_progress.timed('checkpoint'):
          if jax.process_index() == 0:
            train_eval_setup.checkpoint.save(flax.jax_utils.unreplicate(tstate))

  rng, test_rng = random.split(rng)
  with report_progress.timed('test'):
    test_metrics = train_utils.evaluate(model, tstate, test_ds, eval_step,
                                        test_rng,
                                        cfg.train.backward_observation_length)
  test_metrics_cpu = jax.tree.map(np.array, test_metrics.compute())
  test_metrics_cpu = utils.prepend_dict_keys(test_metrics_cpu, 'test/')
  writer.write_scalars(step, test_metrics_cpu)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(f'Too many command-line arguments. {sys.argv[1:]}')
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  tf.config.experimental.set_visible_devices([], 'GPU')
  cfg = _CONFIG.value
  train_and_evaluate(cfg)


if __name__ == '__main__':
  _CONFIG = config_flags.DEFINE_config_dataclass(
      'cfg',
      config.ExperimentConfig(),
      'Configuration flags',
      parse_fn=config.ExperimentConfig.parse_config)
  app.run(main)
