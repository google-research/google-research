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

"""Runs training given a model and losses."""
import os

from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
import six
import tensorflow as tf
from tf3d.utils import callback_utils

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('import_module', None, 'List of modules to import.')

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('train_dir', '/tmp/masternet/',
                    'Directory where to write event logs.')

flags.DEFINE_string('config_file', None, 'The path to the config file.')

flags.DEFINE_multi_string('params', None,
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_enum('distribution_strategy', None,
                  ['multi_worker_mirrored', 'mirrored'],
                  'The Distribution Strategy to use.')

flags.DEFINE_bool('run_functions_eagerly', False,
                  'Run function eagerly for easy debugging.')

flags.DEFINE_integer(
    'num_steps_per_epoch', 1000,
    'Number of batches to train before saving the model weights again. The next'
    'epoch will continue loading the data stream from where the current epoch'
    'left behind.')

flags.DEFINE_integer(
    'log_freq', 100,
    'Number of batches to train before log_freq the model losses again.')

flags.DEFINE_integer('num_epochs', 100, 'Number of epochs.')

flags.DEFINE_integer(
    'num_workers', 1,
    'Number of workers (including chief) for calculating total batch size')

flags.DEFINE_integer(
    'num_gpus', 1, 'Number of gpus per worker for calculating total batch size')

flags.DEFINE_integer('batch_size', 1, 'Per worker batch size.')

flags.DEFINE_integer('gpu_memory_limit', 14700,
                     'Memory size to request per GPU.')


@gin.configurable
def train(strategy,
          write_path,
          learning_rate_fn=None,
          model_class=None,
          input_fn=None,
          optimizer_fn=tf.keras.optimizers.SGD):
  """A function that build the model and train.

  Args:
    strategy: A tf.distribute.Strategy object.
    write_path: A string of path to write training logs and checkpoints.
    learning_rate_fn: A learning rate function.
    model_class: The class of the model to train.
    input_fn: A input function that returns a tf.data.Dataset.
    optimizer_fn: A function that returns the optimizer.
  """
  if learning_rate_fn is None:
    raise ValueError('learning_rate_fn is not set.')

  with strategy.scope():
    logging.info('Model creation starting')
    model = model_class(
        train_dir=os.path.join(write_path, 'train'),
        summary_log_freq=FLAGS.log_freq)

    logging.info('Model compile starting')
    model.compile(optimizer=optimizer_fn(learning_rate=learning_rate_fn()))

    backup_checkpoint_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=os.path.join(write_path, 'backup_model'))
    checkpoint_callback = callback_utils.CustomModelCheckpoint(
        ckpt_dir=os.path.join(write_path, 'model'),
        save_epoch_freq=1,
        max_to_keep=3)

    logging.info('Input creation starting')
    total_batch_size = FLAGS.batch_size * FLAGS.num_workers * FLAGS.num_gpus
    inputs = input_fn(is_training=True, batch_size=total_batch_size)
    logging.info(
        'Model fit starting for %d epochs, %d step per epoch, total batch size:%d',
        flags.FLAGS.num_epochs, flags.FLAGS.num_steps_per_epoch,
        total_batch_size)

  model.fit(
      x=inputs,
      callbacks=[backup_checkpoint_callback, checkpoint_callback],
      steps_per_epoch=FLAGS.num_steps_per_epoch,
      epochs=FLAGS.num_epochs,
      verbose=1 if FLAGS.run_functions_eagerly else 2)
  model.close_writer()


def main(argv):
  del argv
  try:
    logging.info('TF_CONFIG is %s', os.environ.get('TF_CONFIG', 'Empty...'))
    if flags.FLAGS.distribution_strategy == 'multi_worker_mirrored':
      # MultiWorkerMirroredStrategy for multi-worker distributed training.
      # Using AUTO because NCCL sometimes results in seg fault(SIGSEGV).
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
          communication=tf.distribute.experimental.CollectiveCommunication.AUTO)
      task_type = strategy.cluster_resolver.task_type
      task_id = strategy.cluster_resolver.task_id
      write_path = write_filepath(FLAGS.train_dir, task_type, task_id)
    elif flags.FLAGS.distribution_strategy == 'mirrored':
      # single worker with one or multiple GPUs
      strategy = tf.distribute.MirroredStrategy()
      write_path = FLAGS.train_dir
    else:
      raise ValueError(
          'Only `multi_worker_mirrored` and `mirrored` are supported'
          ' strategy at this time. Strategy passed '
          'in is %s' % flags.FLAGS.distribution_strategy)

    logging.info('writing path is %s', write_path)

    # Import modules BEFORE running Gin.
    if FLAGS.import_module:
      for module_name in FLAGS.import_module:
        __import__(module_name)

    # First, try to parse from a config file.
    if FLAGS.config_file:
      bindings = None
      if bindings is None:
        with tf.io.gfile.GFile(FLAGS.config_file) as f:
          bindings = f.readlines()
      bindings = [six.ensure_str(b) for b in bindings if b.strip()]
      gin.parse_config('\n'.join(bindings))

    if FLAGS.params:
      gin.parse_config(FLAGS.params)

    if FLAGS.run_functions_eagerly:
      tf.config.experimental_run_functions_eagerly(True)
    logging.info('Training starting. '
                 'Is run_functions_eagerly: %s', FLAGS.run_functions_eagerly)

    if not tf.io.gfile.exists(FLAGS.train_dir):
      tf.io.gfile.makedirs(FLAGS.train_dir)

    train(strategy=strategy, write_path=write_path)

  except (tf.errors.UnavailableError, tf.errors.FailedPreconditionError) as e:
    logging.warning('Catching error: %s', e)
    # Any non zero exit code will do, avoid exceeding task failure limit.
    exit(42)
  finally:
    if (flags.FLAGS.distribution_strategy == 'multi_worker_mirrored' and
        (not _is_chief(task_type, task_id)) and tf.io.gfile.exists(write_path)):
      tf.io.gfile.rmtree(write_path)


@gin.configurable
def step_decay(initial_learning_rate, boundary_list, ratio_list):
  rates = []
  for ratio in ratio_list:
    rates.append(initial_learning_rate * ratio)
  return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=boundary_list, values=rates)


def _is_chief(task_type, task_id):
  # If `task_type` is None, this may be operating as single worker, which works
  # effectively as chief.
  del task_id
  return task_type is None or task_type == 'chief'


def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir


def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
