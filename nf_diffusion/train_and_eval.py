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

"""Methods for training/evaluating neural-field diffusion using JAX."""

import functools
import json
import os

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import periodic_actions
import flax.jax_utils as flax_utils
import jax
import ml_collections
import numpy as np
import tensorflow as tf

from nf_diffusion import utils


def train_and_evaluate(
    config, workdir):
  """Runs a training and evaluation loop with multiple accelerators.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Load the trainer module
  trainer = utils.load_trainer_module(config.trainer.name)

  tf.io.gfile.makedirs(workdir)
  # Save the configuration, only on ONE process
  host_id = jax.process_index()
  if host_id == 0:
    config_name = "config.json"
  else:
    config_name = "config-%d.json" % host_id
  with tf.io.gfile.GFile(os.path.join(workdir, config_name), "w") as out_file:
    json.dump(config.to_json(), out_file)

  workdir_evalvis = os.path.join(workdir, "eval_vis",
                                 "%d" % jax.process_index())
  tf.io.gfile.makedirs(workdir_evalvis)
  workdir_trainvis = os.path.join(workdir, "train_vis",
                                  "%d" % jax.process_index())
  tf.io.gfile.makedirs(workdir_trainvis)

  # Deterministic training.
  rng = jax.random.PRNGKey(config.seed)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())
  data_info, train_ds, eval_ds = utils.create_datasets(config, data_rng)
  num_train_steps = data_info.num_train_steps

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Get learning rate schedule.
  learning_rate_fn = trainer.get_learning_rate_scheduler(config, data_info)

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, opt, state = trainer.create_train_state(config, model_rng, data_info)

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(
      checkpoint_dir, {"train_iter": train_iter},
      max_to_keep=config.get("max_checkpoints_to_keep", 2))
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step) + 1
  logging.info("Initial step: %d", initial_step)

  # Get trainer step.
  logging.info("Set up train_step function.")
  train_step = functools.partial(
      trainer.train_step,
      config=config,
      model=model,
      opt=opt,
      learning_rate_fn=learning_rate_fn)
  if config.get("multi"):
    # Distribute training.
    state = flax_utils.replicate(state)
    train_step = jax.pmap(train_step, axis_name="batch")

  logging.info("Set up metric writers.")
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if initial_step == 1:
    writer.write_hparams(dict(config))

  logging.info("Initialize loop")
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0 and config.get("do_profiling", True):
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    ]
  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.
      is_last_step = step == num_train_steps
      is_first_step = step == initial_step

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree_util.tree_map(np.asarray, next(train_iter))
        rng, step_rng = jax.random.split(rng)
        if config.get("multi"):
          # TODO(guandao) should we give different rng for differeng
          # hosts or threads?
          # rngs = jax.random.split(rng, 1 + jax.local_device_count())
          step_rng = flax_utils.replicate(step_rng)
        state, metric_update, train_info = train_step(
            state=state, batch=batch, rng=step_rng)
        if metric_update is not None:
          if config.get("multi"):
            metric_update = flax_utils.unreplicate(metric_update)
          train_metrics = (
              metric_update if train_metrics is None
              else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if (train_metrics is not None) and (step % config.log_loss_every_steps
                                          == 0 or is_last_step):
        logging.info("Writing training metricsat step %d", step)
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      train_visualize_steps = config.get("train_visualize_steps", -1)
      if train_visualize_steps > 0 and (step % train_visualize_steps == 0
                                        or is_last_step):
        logging.info("Visualizing training info at step %d", step)
        with report_progress.timed("visualize"):
          if hasattr(trainer, "train_visualize"):
            trainer.train_visualize(config, writer, step, model, state,
                                    train_info, workdir_trainvis)

      if (config.eval_every_steps > 0 and
          (step % config.eval_every_steps == 0 or is_last_step)) or (config.get(
              "eval_at_first_step", False) and is_first_step):

        logging.info("Evaluating at step %d", step)
        with report_progress.timed("eval"):
          if config.get("multi"):
            state = utils.merge_batch_stats(state)
          rng, rng_e = jax.random.split(rng)
          eval_metrics, eval_info = trainer.evaluate(
              config, model, state, eval_ds, rng_e, config.num_eval_steps)
        if eval_metrics is not None:
          eval_metrics_cpu = jax.tree_util.tree_map(np.array,
                                                    eval_metrics.compute())
          eval_metrics_cpu = {
              "eval/{}".format(k): v for k, v in eval_metrics_cpu.items()
          }
          writer.write_scalars(step, eval_metrics_cpu)

        with report_progress.timed("visualize"):
          if hasattr(trainer, "eval_visualize"):
            trainer.eval_visualize(
                config, writer, step, model, state, eval_info, workdir_evalvis)

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          logging.info("Saving checkpoint at step %d", step)
          if config.get("multi"):
            state = utils.merge_batch_stats(state)
            state = flax_utils.unreplicate(state)
          ckpt.save(state)
          if config.get("multi"):
            state = flax_utils.replicate(state)

      # Check at the end of epoch whether there are anything to be done to
      # update the training stage etc.
      if hasattr(trainer, "end_of_iter"):
        if config.get("multi"):
          # Distribute training.
          state = flax_utils.unreplicate(state)
        spl, rng = jax.random.split(rng)
        (model, opt, state, learning_rate_fn,
         recompile_trainstep) = trainer.end_of_iter(
             step=step,
             model=model,
             opt=opt,
             state=state,
             rng=spl,
             config=config,
             learning_rate_fn=learning_rate_fn,
             data_info=data_info)
        if config.get("multi"):
          state = flax_utils.replicate(state)
        if recompile_trainstep:
          # Get trainer step.
          logging.info("Set up train_step function.")
          train_step = functools.partial(
              trainer.train_step,
              config=config,
              model=model,
              opt=opt,
              learning_rate_fn=learning_rate_fn)
          if config.get("multi"):
            train_step = jax.pmap(train_step, axis_name="batch")
  logging.info("Finishing training.")
