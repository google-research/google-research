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

"""File that launches the main training loop for a given configuration."""

import os
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import gin
import jax
import jax.numpy as jnp
import tensorflow.compat.v1 as tf

from d3pm.text import configs
from d3pm.text import datasets

# this registers the diffusion task
from d3pm.text import diffusion  # pylint: disable=unused-import
from d3pm.text import models
from d3pm.text import tasks
from d3pm.text import trainers
from d3pm.text import utils

tf.enable_eager_execution(True)

flags.DEFINE_string('ckpt_dir', None,
                    'Directory for storing experiment info and log files.')
flags.DEFINE_string('name', 'Example experiment', 'Name of current experiment.')
flags.DEFINE_string('description', None, 'Experiment description string.')

flags.DEFINE_string(
    'hyper_fn', '',
    'Name of a function inside `hyper_module` that returns a hyper object.')

FLAGS = flags.FLAGS


def _write_gin_configs(output_file, operative=True):
  """Writes current gin configs to `output_file`."""
  if operative:
    config_str = gin.operative_config_str()
  else:
    config_str = gin.config_str()

  logging.info('=' * 80)
  logging.info('Gin configs\n%s', config_str)
  logging.info('=' * 80)
  with tf.io.gfile.GFile(output_file, 'w') as f:
    f.write(config_str)


@gin.configurable(denylist=['model_dir'])
def run_experiment(
    model_dir,
    model_cls=models.CategoricalDiffusionModel,
    task_name='diffusion',
    dataset_name='lm1b',
    batch_size_per_device = 128,
    max_train_steps = 1000000,
    validate_every = 2500,
    train_summary_frequency = 1000,
    num_eval_steps = 50,
    num_predict_steps = 1,
    restore_checkpoint = True,
    checkpoint_frequency = 5000,
    checkpoint_keep = False,
    fail_on_nan = False,
):
  """Train a model using a given trainer, dataset, and configuration.

  Args:
    model_dir: directory to use for storing and loading training artifacts.
    model_cls: the model class to use.
    task_name: the name of the task to use.
    dataset_name: the name of a dataset registered with d3pm.text.datasets.
    batch_size_per_device: an integer which specifies how many datapoints to use
      per device. All computation is replicated across jax.device_count()
      devices, so the effective batch size will be `batch_size_per_device *
      jax.device_count()`.
    max_train_steps: number of train steps to run for. Each batch is considered
      one train step.
    validate_every: run an evaluation step every n iterations using
      `evaluate_batch`.
    train_summary_frequency: aggregates training statistics and reports the
      average every N iterations.
    num_eval_steps: number of evaluation batches to use during each eval step.
    num_predict_steps: number of batches to use during each pred step.
    restore_checkpoint: if True, will attempt to load a checkpoint from the
      model_dir and use those weights. Will also automatically restore the saved
      training step.
    checkpoint_frequency: save checkpoints every N steps. If None, will not save
      checkpoints.
    checkpoint_keep: if True, will save all checkpoints instead of overwriting
      with the most recent checkpoint.
    fail_on_nan: whether to fail the entire job on encountering an NaN in
      metrics.

  Returns:
    the trainer itself.
  """
  if jax.process_index() == 0:
    if not tf.io.gfile.exists(model_dir):
      logging.info('Creating model directory %s.', model_dir)
      tf.gfile.MakeDirs(model_dir)

  if jax.process_index() == 0:
    _write_gin_configs(
        os.path.join(model_dir, 'all_config.gin'), operative=False)

  ## Load the task object
  task = tasks.load(name=task_name)

  ## Initialize the trainer and dataset objects
  batch_size = batch_size_per_device * jax.local_device_count()
  ds = datasets.load(
      name=dataset_name,
      batch_size=batch_size,
      preprocessors=task.preprocessors)

  assert 'train' in ds, f'dataset {dataset_name} must contain a train set.'
  train_ds = ds['train']

  if 'valid' not in ds:
    logging.info('No validation dataset found. Falling back to the test set.')
    assert 'test' in ds, f'dataset {dataset_name} must contain a test or validation set.'

    valid_ds = ds['test']
  else:
    valid_ds = ds['valid']

  trainer = trainers.Trainer(
      dataset_info=train_ds.info,
      task=task,
      model_cls=model_cls,
  )

  train_iter = iter(train_ds)

  if restore_checkpoint:
    trainer.load_checkpoint(model_dir)

  start_step = trainer.get_state(unreplicate=True).step

  train_metrics = []
  tick = time.time()
  eval_time = 0

  if jax.process_index() == 0:
    n_params = trainer.num_params(unreplicate=True)
    logging.info('num_params: %d', n_params)

  for step, batch in zip(range(start_step, max_train_steps), train_iter):

    if step % validate_every == 0:
      eval_tick = time.time()

      eval_summary = utils.evaluate(
          trainer, valid_ds, num_eval_steps=num_eval_steps)
      logging.info('eval in step: %d, loss: %.4f', step, eval_summary['loss'])

      prediction_summary = utils.predict(
          trainer,
          valid_ds,
          metric_fns=task.metric_fns,
          dataset_info=train_ds.info,
          num_predict_steps=num_predict_steps)

      if jax.process_index() == 0:

        # for each metric function, apply it to the outputs
        for metric in prediction_summary:
          logging.info('eval/%s @ step %d: %s', metric.name, step,
                       str(metric.value))

      #  awful hack to ignore evaluation time in train metrics
      eval_tock = time.time()
      eval_time += eval_tock - eval_tick

      if jax.process_index() == 0:
        logging.info('eval/eval_time: %f (@ step %d)', eval_tock - eval_tick,
                     step)

    metrics = trainer.fit_batch(batch)
    train_metrics.append(metrics)

    if jax.process_index() == 0 and (
        (checkpoint_frequency and step % checkpoint_frequency == 0 and
         step > 0) or step == max_train_steps - 1):
      trainer.save_checkpoint(
          model_dir, keep=100_000_000 if checkpoint_keep else 1)

    if (step + 1) % train_summary_frequency == 0:
      summary = utils.combine_metrics(train_metrics)
      logging.info('train in step: %d, loss: %.4f', step, summary['loss'])

      if jax.process_index() == 0:
        tock = time.time()
        train_time = (tock - tick) - eval_time

        steps_per_sec = train_summary_frequency / train_time
        tick = tock
        elapsed_time = step / steps_per_sec
        logging.info('Elapsed time (minus evaluation): %s', elapsed_time)

        for key, val in summary.items():
          if jnp.isnan(val):
            if fail_on_nan:
              raise ValueError(
                  f'NaN in {key} at step {step}. Summary is: {summary}. Train metrics are: {train_metrics}.'
              )
            else:
              logging.warning('Encountered NaN in %s at step %d.', key, step)

        tock = time.time()
        logging.info('summary_writer_time %f (at step %d)', tock - tick, step)
        tick = tock

      # reset metric accumulation for next evaluation cycle.
      train_metrics = []
      eval_time = 0

  return trainer


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  hyper_fn = FLAGS.hyper_fn

  configs.gin_load(hyper_fn)
  run_experiment(model_dir=FLAGS.ckpt_dir)


if __name__ == '__main__':
  app.run(main)
