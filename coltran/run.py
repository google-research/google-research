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

"""ColTran: Training and Continuous Evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
import tensorflow as tf

from coltran import datasets
from coltran.models import colorizer
from coltran.models import upsampler
from coltran.utils import train_utils


# pylint: disable=g-direct-tensorflow-import

# pylint: disable=missing-docstring
# pylint: disable=not-callable
# pylint: disable=g-long-lambda

flags.DEFINE_enum('mode', 'train', [
    'train', 'eval_train', 'eval_valid', 'eval_test'], 'Operation mode.')

flags.DEFINE_string('logdir', '/tmp/svt', 'Main directory for logs.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU', 'TPU'],
                  'Hardware type.')
flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_string('summaries_log_dir', 'summaries', 'Summaries parent.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('devices_per_worker', 1, 'Number of devices per worker.')
flags.DEFINE_integer('num_workers', 1, 'Number workers.')
config_flags.DEFINE_config_file(
    'config',
    default='test_configs/colorizer.py',
    help_string='Training configuration file.')

FLAGS = flags.FLAGS


def is_tpu():
  return FLAGS.accelerator_type == 'TPU'


def loss_on_batch(inputs, model, config, training=False):
  """Loss on a batch of inputs."""
  logits, aux_output = model.get_logits(
      inputs_dict=inputs, train_config=config, training=training)
  loss, aux_loss_dict = model.loss(
      targets=inputs, logits=logits, train_config=config, training=training,
      aux_output=aux_output)
  loss_factor = config.get('loss_factor', 1.0)

  loss_dict = collections.OrderedDict()
  loss_dict['loss'] = loss
  total_loss = loss_factor * loss

  for aux_key, aux_loss in aux_loss_dict.items():
    aux_loss_factor = config.get(f'{aux_key}_loss_factor', 1.0)
    loss_dict[aux_key] = aux_loss
    total_loss += aux_loss_factor * aux_loss
  loss_dict['total_loss'] = total_loss

  extra_info = collections.OrderedDict([
      ('scalar', loss_dict),
  ])
  return total_loss, extra_info


def train_step(config,
               model,
               optimizer,
               metrics,
               ema=None,
               strategy=None):
  """Training StepFn."""

  def step_fn(inputs):
    """Per-Replica StepFn."""
    with tf.GradientTape() as tape:
      loss, extra = loss_on_batch(inputs, model, config, training=True)
      scaled_loss = loss
      if strategy:
        scaled_loss /= float(strategy.num_replicas_in_sync)

    grads = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    for metric_key, metric in metrics.items():
      metric.update_state(extra['scalar'][metric_key])

    if ema is not None:
      ema.apply(model.trainable_variables)
    return loss

  return train_utils.step_with_strategy(step_fn, strategy)


def build(config, batch_size, is_train=False):
  optimizer = train_utils.build_optimizer(config)
  ema_vars = []

  downsample = config.get('downsample', False)
  downsample_res = config.get('downsample_res', 64)
  h, w = config.resolution

  if config.model.name == 'coltran_core':
    if downsample:
      h, w = downsample_res, downsample_res
    zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
    model = colorizer.ColTranCore(config.model)
    model(zero, training=is_train)

  c = 1 if is_train else 3
  if config.model.name == 'color_upsampler':
    if downsample:
      h, w = downsample_res, downsample_res
    zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
    zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
    model = upsampler.ColorUpsampler(config.model)
    model(zero, inputs_slice=zero_slice, training=is_train)
  elif config.model.name == 'spatial_upsampler':
    zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
    zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
    model = upsampler.SpatialUpsampler(config.model)
    model(zero, inputs_slice=zero_slice, training=is_train)

  ema_vars = model.trainable_variables
  ema = train_utils.build_ema(config, ema_vars)
  return model, optimizer, ema


###############################################################################
## Train.
###############################################################################
def train(logdir):
  config = FLAGS.config
  steps_per_write = FLAGS.steps_per_summaries
  train_utils.write_config(config, logdir)

  strategy, batch_size = train_utils.setup_strategy(
      config, FLAGS.master,
      FLAGS.devices_per_worker, FLAGS.mode, FLAGS.accelerator_type)

  def input_fn(_=None):
    dataset = datasets.get_dataset(
        name=config.dataset,
        config=config,
        batch_size=config.batch_size,
        subset='train')
    return dataset

  # DATASET CREATION.
  logging.info('Building dataset.')
  train_dataset = train_utils.dataset_with_strategy(input_fn, strategy)
  data_iterator = iter(train_dataset)

  # MODEL BUILDING
  logging.info('Building model.')
  model, optimizer, ema = train_utils.with_strategy(
      lambda: build(config, batch_size, True), strategy)
  model.summary(120, print_fn=logging.info)

  # METRIC CREATION.
  metrics = {}
  metric_keys = ['loss', 'total_loss']
  metric_keys += model.metric_keys
  for metric_key in metric_keys:
    func = functools.partial(tf.keras.metrics.Mean, metric_key)
    curr_metric = train_utils.with_strategy(func, strategy)
    metrics[metric_key] = curr_metric

  # CHECKPOINTING LOGIC.
  latest_checkpoint = tf.train.latest_checkpoint(logdir)
  ckpt_func = functools.partial(
      train_utils.create_checkpoint, models=model, optimizer=optimizer,
      ema=ema)
  checkpoint = train_utils.with_strategy(ckpt_func, strategy)
  if latest_checkpoint:
    train_utils.with_strategy(
        lambda: checkpoint.restore(latest_checkpoint), strategy)
    logging.info('Loaded checkpoint %s', latest_checkpoint)
  checkpoint = tf.train.CheckpointManager(
      checkpoint, directory=logdir, checkpoint_name='model', max_to_keep=10)
  if optimizer.iterations.numpy() == 0:
    checkpoint_name = checkpoint.save()
    logging.info('Saved checkpoint to %s', checkpoint_name)

  train_summary_dir = os.path.join(logdir, 'train_summaries')
  writer = tf.summary.create_file_writer(train_summary_dir)
  start_time = time.time()

  logging.info('Start Training.')

  # This hack of wrapping up multiple train steps with a tf.function call
  # speeds up training significantly.
  # See: https://www.tensorflow.org/guide/tpu#improving_performance_by_multiple_steps_within_tffunction # pylint: disable=line-too-long
  @tf.function
  def train_multiple_steps(iterator, steps_per_epoch):

    train_step_f = train_step(config, model, optimizer, metrics, ema,
                              strategy)

    for _ in range(steps_per_epoch):
      train_step_f(iterator)

  while optimizer.iterations.numpy() < config.get('max_train_steps', 1000000):
    num_train_steps = optimizer.iterations

    for metric_key in metric_keys:
      metrics[metric_key].reset_states()

    start_run = time.time()

    train_multiple_steps(data_iterator, tf.convert_to_tensor(steps_per_write))

    steps_per_sec = steps_per_write / (time.time() - start_run)
    with writer.as_default():
      for metric_key, metric in metrics.items():
        metric_np = metric.result().numpy()
        tf.summary.scalar(metric_key, metric_np, step=num_train_steps)

        if metric_key == 'total_loss':
          logging.info('Loss: %.3f bits/dim, Speed: %.3f steps/second',
                       metric_np, steps_per_sec)

    if time.time() - start_time > config.save_checkpoint_secs:
      checkpoint_name = checkpoint.save()
      logging.info('Saved checkpoint to %s', checkpoint_name)
      start_time = time.time()


###############################################################################
## Evaluating.
###############################################################################


def evaluate(logdir, subset):
  """Executes the evaluation loop."""
  config = FLAGS.config
  strategy, batch_size = train_utils.setup_strategy(
      config, FLAGS.master,
      FLAGS.devices_per_worker, FLAGS.mode, FLAGS.accelerator_type)

  def input_fn(_=None):
    return datasets.get_dataset(
        name=config.dataset,
        config=config,
        batch_size=config.eval_batch_size,
        subset=subset)

  model, optimizer, ema = train_utils.with_strategy(
      lambda: build(config, batch_size, False), strategy)

  metric_keys = ['loss', 'total_loss']
  # metric_keys += model.metric_keys
  metrics = {}
  for metric_key in metric_keys:
    func = functools.partial(tf.keras.metrics.Mean, metric_key)
    curr_metric = train_utils.with_strategy(func, strategy)
    metrics[metric_key] = curr_metric

  checkpoints = train_utils.with_strategy(
      lambda: train_utils.create_checkpoint(model, optimizer, ema),
      strategy)
  dataset = train_utils.dataset_with_strategy(input_fn, strategy)

  def step_fn(batch):
    _, extra = loss_on_batch(batch, model, config, training=False)

    for metric_key in metric_keys:
      curr_metric = metrics[metric_key]
      curr_scalar = extra['scalar'][metric_key]
      curr_metric.update_state(curr_scalar)

  num_examples = config.eval_num_examples
  eval_step = train_utils.step_with_strategy(step_fn, strategy)
  ckpt_path = None
  wait_max = config.get(
      'eval_checkpoint_wait_secs', config.save_checkpoint_secs * 100)
  is_ema = True if ema else False

  eval_summary_dir = os.path.join(
      logdir, 'eval_{}_summaries_pyk_{}'.format(subset, is_ema))
  writer = tf.summary.create_file_writer(eval_summary_dir)

  while True:
    ckpt_path = train_utils.wait_for_checkpoint(logdir, ckpt_path, wait_max)
    logging.info(ckpt_path)
    if ckpt_path is None:
      logging.info('Timed out waiting for checkpoint.')
      break

    train_utils.with_strategy(
        lambda: train_utils.restore(model, checkpoints, logdir, ema),
        strategy)
    data_iterator = iter(dataset)
    num_steps = num_examples // batch_size

    for metric_key, metric in metrics.items():
      metric.reset_states()

    logging.info('Starting evaluation.')
    done = False
    for i in range(0, num_steps, FLAGS.steps_per_summaries):
      start_run = time.time()
      for k in range(min(num_steps - i, FLAGS.steps_per_summaries)):
        try:
          if k % 10 == 0:
            logging.info('Step: %d', (i + k + 1))
          eval_step(data_iterator)
        except tf.errors.OutOfRangeError:
          done = True
          break
      if done:
        break
      bits_per_dim = metrics['loss'].result()
      logging.info('Bits/Dim: %.3f, Speed: %.3f seconds/step, Step: %d/%d',
                   bits_per_dim,
                   (time.time() - start_run) / FLAGS.steps_per_summaries,
                   i + k + 1, num_steps)

    # logging.info('Final Bits/Dim: %.3f', bits_per_dim)
    with writer.as_default():
      for metric_key, metric in metrics.items():
        curr_scalar = metric.result().numpy()
        tf.summary.scalar(metric_key, curr_scalar, step=optimizer.iterations)


def main(_):
  logging.info('Logging to %s.', FLAGS.logdir)
  if FLAGS.mode == 'train':
    logging.info('[main] I am the trainer.')
    try:
      train(FLAGS.logdir)
    # During TPU Preemeption, the coordinator hangs with the error below.
    # the exception forces the coordinator to fail, and it will be restarted.
    except (tf.errors.UnavailableError, tf.errors.CancelledError):
      os._exit(os.EX_TEMPFAIL)  # pylint: disable=protected-access
  elif FLAGS.mode.startswith('train'):
    logging.info('[main] I am the trainer.')
    train(os.path.join(FLAGS.logdir, FLAGS.mode))
  elif FLAGS.mode == 'eval_train':
    logging.info('[main] I am the training set evaluator.')
    evaluate(FLAGS.logdir, subset='train')
  elif FLAGS.mode == 'eval_valid':
    logging.info('[main] I am the validation set evaluator.')
    evaluate(FLAGS.logdir, subset='valid')
  elif FLAGS.mode == 'eval_test':
    logging.info('[main] I am the test set evaluator.')
    evaluate(FLAGS.logdir, subset='test')
  else:
    raise ValueError(
        'Unknown mode {}. '
        'Must be one of [train, eval_train, eval_valid, eval_test]'.format(
            FLAGS.mode))


if __name__ == '__main__':
  app.run(main)
