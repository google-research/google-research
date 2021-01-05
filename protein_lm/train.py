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

# Lint as: python3
"""Train a transformer on dataset of sequences."""

import contextlib
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import jax
import jax.nn
import jax.numpy as jnp
import tensorflow.compat.v1 as tf

from protein_lm import data
from protein_lm import evaluation
from protein_lm import logging as logging_lib
from protein_lm import models

tf_summary = logging_lib.tf_summary

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'work_dir', default=None, help=('Directory to store model data.'))

flags.DEFINE_multi_string('gin_files', [], 'List of paths to the config files.')

flags.DEFINE_multi_string('gin_bindings', [],
                          'Newline separated list of Gin parameter bindings.')


def _write_gin_configs(output_file):
  """Writes current gin configs to `output_file`."""
  config_str = gin.operative_config_str()
  logging.info('=' * 80)
  logging.info('Gin configs\n%s', config_str)
  logging.info('=' * 80)
  with tf.gfile.GFile(output_file, 'w') as f:
    f.write(config_str)


@gin.configurable('experiment')
def run_experiment(
    model_dir,
    data_dir=None,
    xid=None,
    batch_size_per_device=128,
    eval_frequency=500,
    checkpoint_frequency=10000,
    save_checkpoints=True,
    restore_checkpoint=True,
    num_eval_steps=None,
    epochs=None,
    max_train_steps=1000000,  # 1 million
    max_train_length=512,
    train_summary_frequency=100,
    max_eval_length=None,
    model_cls=models.FlaxLM):
  """Run experiment.

  Args:
    model_dir: Directory to save checkpoints and metrics to.
    data_dir: Directory to load data.
    xid: Optional experiment id.
    batch_size_per_device: Batch size per device.
    eval_frequency: Steps per eval.
    checkpoint_frequency: How often to checkpoint. If None, only checkpoint once
      at end of run.
    save_checkpoints: If True, checkpoints model according to
      checkpoint_frequency
    restore_checkpoint: If True, will restore checkpoint from directory. Useful
      for robustness to preemption.
    num_eval_steps: Number of eval steps to take on eval dataset.
    epochs: Number of train epochs.
    max_train_steps: Stop training after N steps.
    max_train_length: Crop training sequences to this length.
    train_summary_frequency: Frequency to write train metrics.
    max_eval_length: Maximum eval length. Defaults to max_train_length.
    model_cls: Model class to use.

  Returns:
    FlaxLM resulting from running training.
  """
  if xid is not None:
    model_dir = os.path.join(model_dir, '%s_l%s' % (str(xid), max_train_length))
  tf.enable_v2_behavior()
  if jax.host_id() == 0:
    summary_writer = tf_summary.create_file_writer(
        os.path.join(model_dir, 'metrics'), max_queue=1, flush_millis=1000)
    train_summary_writer = logging_lib.ScalarSummary(
        step=None,
        scope='train/',
        enable_tf=True,
        verbose=0)
    eval_summary_writer = logging_lib.ScalarSummary(
        step=None,
        scope='eval/',
        enable_tf=True,
        verbose=0)

  batch_size = batch_size_per_device * jax.local_device_count()
  max_eval_length = max_eval_length or max_train_length
  train_files, test_files = data.get_train_valid_files(directory=data_dir)
  train_ds, eval_ds = data.load_dataset(
      train_files=train_files,
      test_files=test_files,
      batch_size=batch_size,
      max_train_length=max_train_length,
      max_eval_length=max_eval_length,
      shuffle_buffer=16384)

  with contextlib.ExitStack() as stack:  # pylint: disable=using-constant-test
    if jax.host_id() == 0:
      # Only need metric writer context manager on host 0.
      stack.enter_context(summary_writer.as_default())
    model = model_cls(domain=data.protein_domain, batch_size=batch_size)

    if restore_checkpoint:
      try:
        model.load_checkpoint(model_dir)
      except ValueError:
        # No checkpoint to load -> raises ValueError.
        pass
    start_step = model.train_step

    train_ds = train_ds.repeat(epochs)
    train_iter = iter(train_ds)
    train_metrics = []
    tick = time.time()

    if jax.host_id() == 0:
      _write_gin_configs(os.path.join(model_dir, 'config.gin'))

    num_evals = 0
    for step, batch in zip(range(start_step, max_train_steps), train_iter):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      metrics = model.fit_batch(batch)
      train_metrics.append(metrics)

      if jax.host_id() == 0 and ((save_checkpoints and checkpoint_frequency and
                                  step % checkpoint_frequency == 0 and step > 0)
                                 or step == max_train_steps - 1):
        model.save_checkpoint(model_dir)

      if (step + 1) % train_summary_frequency == 0:
        summary = evaluation.combine_metrics(train_metrics)
        logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
        if jax.host_id() == 0:
          tock = time.time()
          steps_per_sec = eval_frequency / (tock - tick)
          tick = tock
          train_summary_writer('steps per second', steps_per_sec, step)
          for key, val in summary.items():
            if jnp.isnan(val):
              raise ValueError(f'NaN in {key} at step {step}.')
            train_summary_writer(key, val, step)

        # reset metric accumulation for next evaluation cycle.
        train_metrics = []

      if eval_frequency and (step + 1) % eval_frequency == 0:
        eval_summary = evaluation.evaluate(
            model=model, eval_ds=eval_ds, num_eval_steps=num_eval_steps)

        logging.info('eval in step: %d, loss: %.4f', step, eval_summary['loss'])
        if jax.host_id() == 0:
          for key, val in eval_summary.items():
            eval_summary_writer(key, val, step)
          tf_summary.flush()
          summary_writer.flush()

          if num_evals == 0:
            # Write out config on first eval.
            _write_gin_configs(os.path.join(model_dir, 'config_after_eval.gin'))
          num_evals += 1

  if jax.host_id() == 0:
    tf_summary.flush()
    summary_writer.close()
    _write_gin_configs(os.path.join(model_dir, 'config_end.gin'))
  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Main called')

  gin_bindings = FLAGS.gin_bindings
  gin_files = FLAGS.gin_files

  # Parse gin configs.
  logging.info('Gin files: %s', str(gin_files))
  logging.info('Gin bindings: %s', str(gin_bindings))
  gin.parse_config_files_and_bindings(gin_files, gin_bindings)
  run_experiment(model_dir=FLAGS.work_dir)


if __name__ == '__main__':
  app.run(main)
