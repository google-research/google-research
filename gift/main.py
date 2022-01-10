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

"""Main script for the GIFT project."""
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
from flax.metrics import tensorboard
import jax
from jax.config import config
from ml_collections import config_flags
import tensorflow as tf
from tensorflow.io import gfile

from gift.models import all_models
from gift.pipelines import all_pipelines
from gift.tasks import all_tasks

# Enable flax xprof trace labelling.
os.environ['FLAX_PROFILE'] = 'true'

FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_dir', None, 'Experiment directory.')
config_flags.DEFINE_config_file(
    'config', None, 'Path to the experiment configuration.', lock_config=True)


def run(hparams, experiment_dir, summary_writer=None):
  """Prepares model, and dataset for training.

  This creates summary directories, summary writers, model definition, and
  builds datasets to be sent to the main training script.

  Args:
    hparams:  ConfigDict; Hyper parameters.
    experiment_dir: string; Root directory for the experiment.
    summary_writer: Summary writer object.

  Returns:
    output of the trainer.train(), which are traing metric summaries.
  """
  # Set up the train_dir and log_dir.
  gfile.makedirs(experiment_dir)

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.host_count())
  logging.info('host_id : %d', jax.host_id())

  rng = jax.random.PRNGKey(hparams.rng_seed)
  logging.info('rng: %s', rng)

  batch_size = hparams.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = hparams.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  # Set batch sizes in hparams and log them too
  with hparams.unlocked():
    hparams.local_batch_size = batch_size // jax.host_count()
    hparams.eval_local_batch_size = eval_batch_size // jax.host_count()
    hparams.device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', hparams.local_batch_size)
  logging.info('device_batch_size : %d', hparams.device_batch_size)

  # Get model class
  model_cls = all_models.get_model_class(hparams.model_name)

  # Create task
  task_cls = all_tasks.get_task_class(hparams.task_name)
  task = task_cls(task_params=hparams, num_shards=jax.local_device_count())

  # Create trainer
  trainer_cls = all_pipelines.get_trainer_class(hparams.train_mode)
  trainer = trainer_cls(
      model_cls=model_cls,
      task=task,
      hparams=hparams,
      experiment_dir=experiment_dir,
      tb_summary_writer=summary_writer,
      rng=rng)

  # Run training
  return trainer.train()


def main(_):
  master = jax.host_id() == 0
  # make sure TF does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # load configs from a config json string
  hparams = FLAGS.config
  logging.info('=========== Hyperparameters ============')
  logging.info(hparams)

  if hparams.get('debug'):
    logging.warning('DEBUG MODE IS ENABLED!')

  # set tensorflow random seed
  tf.random.set_seed(jax.host_id() + hparams.rng_seed)
  experiment_dir = FLAGS.experiment_dir
  logging.info('Experiment directory: %s', experiment_dir)
  summary_writer = None

  if master and hparams.write_summary:
    tensorboard_dir = os.path.join(experiment_dir, 'tb_summaries')
    gfile.makedirs(tensorboard_dir)
    summary_writer = tensorboard.SummaryWriter(tensorboard_dir)

  run(hparams, experiment_dir, summary_writer)

  pool.close()
  pool.join()


if __name__ == '__main__':
  config.config_with_absl()
  app.run(main)
