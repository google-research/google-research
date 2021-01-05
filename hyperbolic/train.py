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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a collaborative filtering model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import json
import logging as native_logging
import os
import sys
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from hyperbolic.config import CONFIG
from hyperbolic.datasets.datasets import DatasetClass
from hyperbolic.learning.trainer import CFTrainer
import hyperbolic.models as models
import hyperbolic.utils.train as train_utils

tf.enable_v2_behavior()

flag_fns = {
    'string': flags.DEFINE_string,
    'integer': flags.DEFINE_integer,
    'boolean': flags.DEFINE_boolean,
    'float': flags.DEFINE_float,
}
for dtype, flag_fn in flag_fns.items():
  for arg, (description, default) in CONFIG[dtype].items():
    flag_fn(arg, default=default, help=description)
FLAGS = flags.FLAGS


def main(_):
  # get logger
  save_path = FLAGS.save_dir
  if FLAGS.save_logs:
    if not tf.gfile.Exists(os.path.join(save_path, 'train.log')):
      tf.gfile.MakeDirs(save_path)
      write_mode = 'w'
    else:
      write_mode = 'a'
    stream = tf.gfile.Open(os.path.join(save_path, 'train.log'), write_mode)
    log_handler = native_logging.StreamHandler(stream)
    print('Saving logs in {}'.format(save_path))
  else:
    log_handler = native_logging.StreamHandler(sys.stdout)
  formatter = native_logging.Formatter(
      '%(asctime)s %(levelname)-8s %(message)s')
  log_handler.setFormatter(formatter)
  log_handler.setLevel(logging.INFO)
  logger = logging.get_absl_logger()
  logger.addHandler(log_handler)

  # set up tf.summary
  train_log_dir = save_path + '/train'
  valid_log_dir = save_path + '/valid'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

  # load data
  dataset_path = os.path.join(FLAGS.data_dir, FLAGS.dataset)
  dataset = DatasetClass(dataset_path, FLAGS.debug)
  sizes = dataset.get_shape()
  train_examples_reversed = dataset.get_examples('train')
  valid_examples = dataset.get_examples('valid')
  test_examples = dataset.get_examples('test')
  filters = dataset.get_filters()
  logging.info('\t Dataset shape: %s', (str(sizes)))

  # save config
  config_path = os.path.join(save_path, 'config.json')
  if FLAGS.save_logs and not tf.gfile.Exists(config_path):
    with tf.gfile.Open(config_path, 'w') as fjson:
      json.dump(train_utils.get_config_dict(CONFIG), fjson)

  # create and build model
  tf.keras.backend.set_floatx(FLAGS.dtype)
  model = getattr(models, FLAGS.model)(sizes, FLAGS)
  model.build(input_shape=(1, 2))
  trainable_params = train_utils.count_params(model)
  trainer = CFTrainer(sizes, FLAGS)
  logging.info('\t Total number of trainable parameters %s', (trainable_params))

  # restore or create checkpoint
  if FLAGS.save_model:
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=trainer.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=1)
    if manager.latest_checkpoint:
      ckpt.restore(manager.latest_checkpoint)
      logging.info('\t Restored from %s', (manager.latest_checkpoint))
    else:
      logging.info('\t Initializing from scratch.')
  else:
    logging.info('\t Initializing from scratch.')

  # train model
  logging.info('\t Start training')
  early_stopping_counter = 0
  best_mrr = None
  best_epoch = None
  best_weights = None
  if FLAGS.save_model:
    epoch = ckpt.step
  else:
    epoch = 0

  if int(epoch) < FLAGS.max_epochs:
    while int(epoch) < FLAGS.max_epochs:
      if FLAGS.save_model:
        epoch.assign_add(1)
      else:
        epoch += 1

      # Train step
      start = time.perf_counter()
      train_batch = train_examples_reversed.batch(FLAGS.batch_size)
      train_loss = trainer.train_step(model, train_batch).numpy()
      end = time.perf_counter()
      execution_time = (end - start)
      logging.info('\t Epoch %i | train loss: %.4f | total time: %.4f',
                   int(epoch), train_loss, execution_time)
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch)

      if FLAGS.save_model and int(epoch) % FLAGS.checkpoint == 0:
        save_path = manager.save()
        logging.info('\t Saved checkpoint for epoch %i: %s', int(epoch),
                     save_path)

      if int(epoch) % FLAGS.valid == 0:
        # compute valid loss
        valid_batch = valid_examples.batch(FLAGS.batch_size)
        valid_loss = trainer.valid_step(model, valid_batch).numpy()
        logging.info('\t Epoch %i | average valid loss: %.4f', int(epoch),
                     valid_loss)
        with valid_summary_writer.as_default():
          tf.summary.scalar('loss', valid_loss, step=epoch)

        # compute validation metrics
        valid = train_utils.metric_dict_full_and_random(
            *model.random_eval(valid_examples, filters))
        logging.info(train_utils.format_metrics(valid, split='valid'))
        with valid_summary_writer.as_default():
          tf.summary.scalar('mrs', valid['MR'], step=epoch)
          tf.summary.scalar('mrrs', valid['MRR'], step=epoch)
          tf.summary.scalar('hits@[1]', valid['hits@[1,3,10]'][1], step=epoch)
          tf.summary.scalar('hits@[3]', valid['hits@[1,3,10]'][3], step=epoch)
          tf.summary.scalar('hits@[10]', valid['hits@[1,3,10]'][10], step=epoch)
          tf.summary.scalar('mrs_r', valid['MR_r'], step=epoch)
          tf.summary.scalar('mrrs_r', valid['MRR_r'], step=epoch)
          tf.summary.scalar(
              'hits@[1]_r', valid['hits@[1,3,10]_r'][1], step=epoch)
          tf.summary.scalar(
              'hits@[3]_r', valid['hits@[1,3,10]_r'][3], step=epoch)
          tf.summary.scalar(
              'hits@[10]_r', valid['hits@[1,3,10]_r'][10], step=epoch)

        # early stopping
        valid_mrr = valid['MRR']
        if not best_mrr or valid_mrr > best_mrr:
          best_mrr = valid_mrr
          early_stopping_counter = 0
          best_epoch = int(epoch)
          best_weights = copy.copy(model.get_weights())
        else:
          early_stopping_counter += 1
          if early_stopping_counter == FLAGS.patience:
            logging.info('\t Early stopping')
            break

    logging.info('\t Optimization finished')
    logging.info('\t Evaluating best model from epoch %s', best_epoch)
    model.set_weights(best_weights)
    if FLAGS.save_model:
      model.save_weights(os.path.join(save_path, 'best_model.ckpt'))

    # validation metrics
    valid = train_utils.metric_dict_full_and_random(
        *model.random_eval(valid_examples, filters))
    logging.info(train_utils.format_metrics(valid, split='valid'))

    # test metrics
    test = train_utils.metric_dict_full_and_random(
        *model.random_eval(test_examples, filters))
    logging.info(train_utils.format_metrics(test, split='test'))
  else:
    logging.info('\t Training completed')


if __name__ == '__main__':
  app.run(main)
