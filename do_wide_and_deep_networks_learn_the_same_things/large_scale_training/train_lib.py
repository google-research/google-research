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

"""Library for model training."""
import collections
import contextlib
import os
import pickle
import re
import sys
import time

from absl import flags
from absl import logging

import orbit
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import single_task_evaluator
from do_wide_and_deep_networks_learn_the_same_things.large_scale_training import single_task_trainer

import scipy.special

import tensorflow.compat.v2 as tf


tf.enable_v2_behavior()

FLAGS = flags.FLAGS
# Define training setup
flags.DEFINE_string('tpu', None, 'Address of TPU to use for training.')
flags.DEFINE_enum('mode', 'train_and_evaluate',
                  ['train', 'evaluate', 'train_and_evaluate'],
                  'Execution mode.')
flags.DEFINE_integer('checkpoints_to_keep', 1,
                     'Number of checkpoints to keep, in addition to the best.')

EVAL_ACCURACY_KEY = 'accuracy/eval'


def get_summaries_from_dir(path):
  """Load summaries from event files in a directory."""
  events_files = tf.io.gfile.glob(os.path.join(path, 'events.out.tfevents.*'))
  values = collections.defaultdict(lambda: [])
  global_step = collections.defaultdict(lambda: [])
  for events_file in events_files:
    try:
      for e in tf.compat.v1.train.summary_iterator(events_file):
        for v in e.summary.value:
          if not v.tensor:
            continue
          if not global_step[v.tag] or e.step >= global_step[v.tag][-1]:
            global_step[v.tag].append(e.step)
            values[v.tag].append(tf.make_ndarray(v.tensor))
    except tf.errors.DataLossError:
      return {}
  return values, global_step


def save_predictions(model, dataset, output_dir):
  """Get predictions and save to a file."""
  preds = model.predict(dataset, verbose=1)
  preds = scipy.special.softmax(preds, 1)  # Apply softmax
  with tf.io.gfile.GFile(os.path.join(output_dir, 'test_preds.pkl'), 'wb') as f:
    pickle.dump(preds, f)


def train(model_optimizer_fn, train_steps, eval_steps, steps_between_evals,
          train_dataset, test_dataset, experiment_dir):
  """Perform training.

  Arguments:
    model_optimizer_fn: Function that returns a tuple containing the model and
      its optimizer.
    train_steps: Total number of steps to train for.
    eval_steps: Number of steps to evaluate for.
    steps_between_evals: Number of steps to train for between evaluations.
    train_dataset: Dataset to use for training.
    test_dataset: Size of test dataset.
    experiment_dir: Directory in which to save results.
  """

  test_dataset_orig = test_dataset
  if FLAGS.tpu is not None:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

  summary_and_checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
  best_checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best')
  final_model_dir = os.path.join(experiment_dir, 'final_model')
  best_model_dir = os.path.join(experiment_dir, 'best_model')

  # Load previous best accuracy and previous eval step out of summaries.
  summaries, summary_steps = get_summaries_from_dir(summary_and_checkpoint_dir)
  previous_best_accuracy = 0.0
  previous_eval_steps = 0
  if summaries[EVAL_ACCURACY_KEY]:
    previous_best_accuracy = max(summaries[EVAL_ACCURACY_KEY])
    previous_eval_steps = max(summary_steps[EVAL_ACCURACY_KEY])

  try:
    with strategy.scope() if FLAGS.tpu else contextlib.suppress():
      model, optimizer = model_optimizer_fn()

      def _loss_fn(labels, logits):
        """Compute total loss."""
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
        # Add weight decay losses to final loss.
        return loss + tf.reduce_sum(model.losses)

      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
      manager = tf.train.CheckpointManager(
          checkpoint=checkpoint,
          directory=summary_and_checkpoint_dir,
          max_to_keep=(None
                       if FLAGS.mode == 'train' else FLAGS.checkpoints_to_keep),
          step_counter=optimizer.iterations,
          checkpoint_interval=steps_between_evals)

      trainer = single_task_trainer.SingleTaskTrainer(
          train_dataset=train_dataset,
          label_key='label',
          model=model,
          loss_fn=_loss_fn,
          optimizer=optimizer,
          metrics=[
              tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy/train'),
              tf.keras.metrics.SparseCategoricalCrossentropy(name='loss/train'),
          ])
      evaluator = single_task_evaluator.SingleTaskEvaluator(
          eval_dataset=test_dataset,
          label_key='label',
          model=model,
          metrics=[
              tf.keras.metrics.SparseCategoricalAccuracy(
                  name=(EVAL_ACCURACY_KEY)),
              tf.keras.metrics.SparseCategoricalCrossentropy(name='loss/eval'),
          ])
      controller = orbit.Controller(
          trainer=trainer,
          evaluator=evaluator,
          steps_per_loop=steps_between_evals,
          global_step=optimizer.iterations,
          checkpoint_manager=manager)

      controller.restore_checkpoint()
      while optimizer.iterations < train_steps:
        current_steps = optimizer.iterations.numpy()
        if ('evaluate' in FLAGS.mode and
            (current_steps // steps_between_evals >
             previous_eval_steps // steps_between_evals)):
          logging.info('Skipping training because eval is out-of-date.')
        else:
          current_train_steps = min(
              (current_steps // steps_between_evals + 1) * steps_between_evals,
              train_steps)
          if 'train' in FLAGS.mode:
            controller.train(current_train_steps)
          elif 'evaluate' in FLAGS.mode:
            next_checkpoint_path = os.path.join(
                summary_and_checkpoint_dir,
                'ckpt-{}'.format(current_train_steps))
            while not tf.io.gfile.exists(next_checkpoint_path + '.index'):
              logging.info('Checkpoint %s not yet ready.', next_checkpoint_path)
              time.sleep(15)
            checkpoint.restore(next_checkpoint_path)

        if 'evaluate' in FLAGS.mode:
          controller.evaluate(eval_steps)
          current_accuracy = evaluator.eval_end()[EVAL_ACCURACY_KEY]
          current_train_loss = trainer.train_loop_end()['loss/train']
          previous_eval_steps = optimizer.iterations.numpy()


          if current_accuracy > previous_best_accuracy:
            logging.info(
                'New accuracy %.4f beats best previous accuracy %.4f; saving '
                'new best checkpoint.', current_accuracy,
                previous_best_accuracy)
            previous_best_accuracy = current_accuracy
            checkpoint.write(best_checkpoint_path)

        if FLAGS.mode == 'evaluate':
          # Delete checkpoints if we have hit max. We do this in the eval job
          # to make sure that we aren't deleting checkpoints we haven't yet
          # evaluated.
          checkpoint_paths = tf.io.gfile.glob(
              os.path.join(summary_and_checkpoint_dir, 'ckpt-*.index'))
          checkpoint_paths_nums = [
              (int(re.search(r'/ckpt-([0-9]+).index$', x).group(1)), x)
              for x in checkpoint_paths
          ]
          checkpoint_paths_nums.sort()
          for num, path in checkpoint_paths_nums[:-FLAGS.checkpoints_to_keep]:
            if num <= optimizer.iterations.numpy():
              # Don't delete unevaluated checkpoints.
              logging.info('Removing old checkpoint %s.', path)
              tf.io.gfile.remove(path)

      if 'evaluate' in FLAGS.mode:
        # Save final model.
        tf.io.gfile.mkdir(final_model_dir)
        save_predictions(model, test_dataset_orig, final_model_dir)
        tf.keras.models.save_model(model, final_model_dir)

        # At end of training, load best checkpoint and write Keras saved model.
        # We do not do this during training because it is very slow.
        checkpoint.restore(best_checkpoint_path)
        tf.io.gfile.mkdir(best_model_dir)
        save_predictions(model, test_dataset_orig, best_model_dir)
        tf.keras.models.save_model(model, best_model_dir)
  except tf.errors.UnavailableError:
    logging.info('Lost contact with TPU; restaarting.')
    sys.exit(42)
