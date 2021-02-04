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

"""Utils for training."""

import os
import time
from absl import logging
import numpy as np
import tensorflow as tf
import yaml


def step_with_strategy(step_fn, strategy):

  def _step(iterator):
    if strategy is None:
      step_fn(next(iterator))
    else:
      strategy.experimental_run(step_fn, iterator)

  return _step


def write_config(config, logdir):
  """Write config dict to a directory."""
  tf.io.gfile.makedirs(logdir)
  with tf.io.gfile.GFile(os.path.join(logdir, 'config.yaml'), 'w') as f:
    yaml.dump(config.to_dict(), f)


def wait_for_checkpoint(observe_dirs, prev_path=None, max_wait=-1):
  """Returns new checkpoint paths, or None if timing out."""
  is_single = isinstance(observe_dirs, str)
  if is_single:
    observe_dirs = [observe_dirs]
    if prev_path:
      prev_path = [prev_path]

  start_time = time.time()
  prev_path = prev_path or [None for _ in observe_dirs]
  while True:
    new_path = [tf.train.latest_checkpoint(d) for d in observe_dirs]
    if all(a != b for a, b in zip(new_path, prev_path)):
      if is_single:
        latest_ckpt = new_path[0]
      else:
        latest_ckpt = new_path
      if latest_ckpt is not None:
        return latest_ckpt
    if max_wait > 0 and (time.time() - start_time) > max_wait:
      return None
    tf.logging.info('Sleeping 60s, waiting for checkpoint.')
    time.sleep(60)


def build_optimizer(config):
  """Builds optimizer."""
  optim_config = dict(config.optimizer)
  optim_type = optim_config.pop('type', 'rmsprop')
  if optim_type == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(**optim_config)
  elif optim_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(**optim_config)
  elif optim_type == 'sgd':
    optimizer = tf.keras.optimizers.SGD(**optim_config)
  else:
    raise ValueError('Unknown optimizer %s.' % optim_type)
  return optimizer


def build_ema(config, ema_vars):
  """Builds exponential moving average."""
  ema = None
  polyak_decay = config.get('polyak_decay', 0.0)
  if polyak_decay:
    ema = tf.train.ExponentialMovingAverage(polyak_decay)
    ema.apply(ema_vars)
    logging.info('Built with exponential moving average.')
  return ema


def setup_strategy(config, master, devices_per_worker, mode, accelerator_type):
  """Set up strategy."""
  if accelerator_type == 'TPU':
    cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=master)
    tf.config.experimental_connect_to_cluster(cluster)
    tf.tpu.experimental.initialize_tpu_system(cluster)
    strategy = tf.distribute.experimental.TPUStrategy(cluster)
    logging.info('Training on TPUs')
  else:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
    strategy = None

  num_cores = devices_per_worker
  tpu_batch_size = config.get('eval_batch_size', 0)
  if mode.startswith('train') or not tpu_batch_size:
    tpu_batch_size = config.batch_size
  tpu_batch_size *= num_cores
  logging.info('Running on %d number of cores with total batch_size of %d.',
               num_cores, tpu_batch_size)

  return strategy, tpu_batch_size


def dataset_with_strategy(dataset_fn, strategy):
  if strategy:
    return strategy.experimental_distribute_datasets_from_function(dataset_fn)
  else:
    return dataset_fn(None)


def with_strategy(fn, strategy):
  logging.info(strategy)
  if strategy:
    with strategy.scope():
      return fn()
  else:
    return fn()


def create_checkpoint(models, optimizer=None, ema=None, scope=None):
  """Creates tf.train.Checkpoint instance."""
  single_model = not isinstance(models, (tuple, list))
  checkpoints = []
  for m in [models] if single_model else models:
    ema_vars = get_ema_vars(ema, m)
    if filter is None:
      to_save = {v.name: v for v in m.variables if scope in v.name}
    else:
      to_save = {v.name: v for v in m.variables}
    to_save.update(ema_vars)
    if optimizer is not None and scope is None:
      to_save['optimizer'] = optimizer
    checkpoints.append(
        tf.train.Checkpoint(**to_save))
  return checkpoints[0] if single_model else checkpoints


def get_curr_step(ckpt_path):
  """Parse curr training step from checkpoint path."""
  var_names = tf.train.list_variables(ckpt_path)
  for var_name, _ in var_names:
    if 'iter' in var_name:
      step = tf.train.load_variable(ckpt_path, var_name)
      return step


def get_ema_vars(ema, model):
  """Get ema variables."""
  if ema:
    try:
      return {
          ema.average(v).name: ema.average(v) for v in model.trainable_variables
      }
    except:  # pylint: disable=bare-except
      ema.apply(model.trainable_variables)
      return {
          ema.average(v).name: ema.average(v) for v in model.trainable_variables
      }
    else:
      return {}
  else:
    return {}


def restore(model, ckpt, ckpt_dir, ema=None):
  if not isinstance(model, (tuple, list)):
    model, ckpt, ckpt_dir = [model], [ckpt], [ckpt_dir]
  for model_, ckpt_, ckpt_dir_ in zip(model, ckpt, ckpt_dir):
    logging.info('Restoring from %s.', ckpt_dir_)
    ckpt_.restore(tf.train.latest_checkpoint(ckpt_dir_)).expect_partial()
    if ema:
      for v in model_.trainable_variables:
        v.assign(ema.average(v))


def save_nparray_to_disk(filename, nparray):
  fdir, _ = os.path.split(filename)
  if not tf.io.gfile.exists(fdir):
    tf.io.gfile.makedirs(fdir)
  with tf.io.gfile.GFile(filename, 'w') as f:
    np.save(f, nparray)
