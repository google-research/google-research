# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Evaluate embeddings on downstream tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf

from tcc.algorithms import get_algo
from tcc.config import CONFIG
from tcc.datasets import create_dataset
from tcc.datasets import create_one_epoch_dataset
from tcc.tasks import get_tasks

from tcc.utils import get_embeddings_dataset
from tcc.utils import get_lr_opt_global_step
from tcc.utils import restore_ckpt
from tcc.utils import setup_eval_dir


layers = tf.keras.layers


flags.DEFINE_boolean('continuous_eval', True, 'Evaluate continously.')
flags.DEFINE_string('logdir', '/tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_boolean('defun', True, 'Defun everything!')
flags.DEFINE_integer(
    'max_embs', 0, 'Max number of videos to embed. 0 or less '
    'means embed all videos in dataset.')
FLAGS = flags.FLAGS

evaluated_last_ckpt = False


def evaluate_once(algo, iterator_tasks, embedding_tasks, iterators,
                  summary_writer):
  """Evaluate learnt embeddings on downstream tasks."""

  # Sets up model for training.
  _, optimizer, global_step = get_lr_opt_global_step()
  restore_ckpt(logdir=CONFIG.LOGDIR, optimizer=optimizer, **algo.model)

  if global_step.numpy() == CONFIG.TRAIN.MAX_ITERS:
    global evaluated_last_ckpt
    evaluated_last_ckpt = True

  metrics = {}
  if iterator_tasks:
    with summary_writer.as_default():
      with tf.summary.record_if(True):
        for task_name, task in iterator_tasks.items():
          metrics[task_name] = task.evaluate(algo, global_step,
                                             iterators=iterators)

  max_embs = None if FLAGS.max_embs <= 0 else FLAGS.max_embs
  if embedding_tasks:
    frames_per_batch = CONFIG.EVAL.FRAMES_PER_BATCH
    for dataset_name in CONFIG.DATASETS:
      dataset = {'name': dataset_name}
      train_iterator = create_one_epoch_dataset(
          dataset_name,
          'train',
          mode='eval',
          path_to_tfrecords=CONFIG.PATH_TO_TFRECORDS)
      dataset['train_dataset'] = get_embeddings_dataset(
          algo.model, train_iterator, frames_per_batch=frames_per_batch,
          max_embs=max_embs)

      val_iterator = create_one_epoch_dataset(
          dataset_name,
          'val',
          mode='eval',
          path_to_tfrecords=CONFIG.PATH_TO_TFRECORDS)
      dataset['val_dataset'] = get_embeddings_dataset(
          algo.model, val_iterator, frames_per_batch=frames_per_batch,
          max_embs=max_embs)

      with summary_writer.as_default():
        with tf.summary.record_if(True):
          for task_name, task in embedding_tasks.items():
            if task_name not in metrics:
              metrics[task_name] = {}
            metrics[task_name][dataset_name] = task.evaluate(
                algo, global_step, embeddings_dataset=dataset)

  # Add all metrics in a separate tag so that analysis is easier.
  with summary_writer.as_default():
    with tf.summary.record_if(True):
      for task_name in embedding_tasks.keys():
        for dataset in CONFIG.DATASETS:
          tf.summary.scalar('metrics/%s_%s' % (dataset, task_name),
                            metrics[task_name][dataset],
                            step=global_step)
        avg_metric = sum(metrics[task_name].values())
        avg_metric /= len(CONFIG.DATASETS)
        tf.summary.scalar('metrics/all_%s' % task_name,
                          avg_metric, step=global_step)


def timeout_fn():
  global evaluated_last_ckpt
  return evaluated_last_ckpt


def evaluate():
  """Evaluate embeddings."""
  CONFIG.LOGDIR = FLAGS.logdir
  logdir = CONFIG.LOGDIR
  setup_eval_dir(logdir)

  algo = get_algo(CONFIG.TRAINING_ALGO)

  if FLAGS.defun:
    algo.call = tf.function(algo.call)
    algo.compute_loss = tf.function(algo.compute_loss)

  iterator_tasks, embedding_tasks = get_tasks(CONFIG.EVAL.TASKS)

  # Setup summary writer.
  summary_writer = tf.summary.create_file_writer(
      os.path.join(logdir, 'eval_logs'), flush_millis=10000)

  iterators = {}
  if iterator_tasks:
    # Setup Dataset Iterators from train and val datasets.
    iterators['train_iterator'] = create_dataset('train', mode='eval')
    iterators['val_iterator'] = create_dataset('val', mode='eval')

  if FLAGS.continuous_eval:
    for _ in tf.train.checkpoints_iterator(logdir, timeout=1,
                                           timeout_fn=timeout_fn):
      evaluate_once(algo, iterator_tasks, embedding_tasks, iterators,
                    summary_writer)
  else:
    evaluate_once(algo, iterator_tasks, embedding_tasks, iterators,
                  summary_writer)


def main(_):
  tf.enable_v2_behavior()
  tf.keras.backend.set_learning_phase(0)
  evaluate()


if __name__ == '__main__':
  app.run(main)
