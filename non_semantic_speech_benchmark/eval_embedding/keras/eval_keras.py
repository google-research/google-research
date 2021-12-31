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
"""Eval a Keras model on embeddings."""

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from non_semantic_speech_benchmark.eval_embedding import metrics
from non_semantic_speech_benchmark.eval_embedding.keras import get_data
from non_semantic_speech_benchmark.eval_embedding.keras import models

FLAGS = flags.FLAGS

flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_string('embedding_name', None, 'Embedding name.')
flags.DEFINE_alias('en', 'embedding_name')
flags.DEFINE_integer('embedding_dimension', None, 'Embedding dimension.')
flags.DEFINE_alias('ed', 'embedding_dimension')
flags.DEFINE_string('label_name', None, 'Name of label to use.')
flags.DEFINE_list('label_list', None, 'List of possible label values.')
flags.DEFINE_list('bucket_boundaries', ['99999'],
                  'bucket_boundaries for data. Default is all one bucket.')

flags.DEFINE_integer('batch_size', None, 'The number of images in each batch.')
flags.DEFINE_integer('tbs', None, 'not used')

flags.DEFINE_integer('num_clusters', None, 'num_clusters')
flags.DEFINE_alias('nc', 'num_clusters')
flags.DEFINE_float('alpha_init', None, 'Initial autopool alpha.')
flags.DEFINE_alias('ai', 'alpha_init')
flags.DEFINE_boolean('use_batch_normalization', None,
                     'Whether to use batch normalization.')
flags.DEFINE_alias('ubn', 'use_batch_normalization')
flags.DEFINE_float('lr', None, 'not used')

flags.DEFINE_string('logdir', None,
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', None,
                    'Directory where the results are saved to.')
flags.DEFINE_integer('take_fixed_data', None,
                     'If not `None`, take a fixed number of data elements.')
flags.DEFINE_integer('timeout', 7200, 'Wait-for-checkpoint timeout.')
flags.DEFINE_boolean('calculate_equal_error_rate', False,
                     'Whether to calculate the Equal Error Rate. Only '
                     'applicable for binary classification problems.')
flags.DEFINE_boolean('preaverage', False, 'Whether to preaverage.')


def eval_and_report():
  """Eval on voxceleb."""
  logging.info('embedding_name: %s', FLAGS.embedding_name)
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.batch_size)

  writer = tf.summary.create_file_writer(FLAGS.eval_dir)
  num_classes = len(FLAGS.label_list)
  model = models.get_keras_model(
      num_classes, FLAGS.use_batch_normalization,
      num_clusters=FLAGS.num_clusters, alpha_init=FLAGS.alpha_init)
  checkpoint = tf.train.Checkpoint(model=model)

  for ckpt in tf.train.checkpoints_iterator(
      FLAGS.logdir, timeout=FLAGS.timeout):
    assert 'ckpt-' in ckpt, ckpt
    step = ckpt.split('ckpt-')[-1]
    logging.info('Starting to evaluate step: %s.', step)

    checkpoint.restore(ckpt)

    logging.info('Loaded weights for eval step: %s.', step)

    reader = tf.data.TFRecordDataset
    ds = get_data.get_data(
        file_pattern=FLAGS.file_pattern,
        reader=reader,
        embedding_name=FLAGS.embedding_name,
        embedding_dim=FLAGS.embedding_dimension,
        label_name=FLAGS.label_name,
        label_list=FLAGS.label_list,
        bucket_boundaries=FLAGS.bucket_boundaries,
        bucket_batch_sizes=[FLAGS.batch_size] * (len(FLAGS.bucket_boundaries) + 1),  # pylint:disable=line-too-long
        loop_forever=False,
        shuffle=False,
        preaverage=FLAGS.preaverage)
    logging.info('Got dataset for eval step: %s.', step)
    if FLAGS.take_fixed_data:
      ds = ds.take(FLAGS.take_fixed_data)

    acc_m = tf.keras.metrics.Accuracy()
    xent_m = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)

    logging.info('Starting the ds loop...')
    count, ex_count = 0, 0
    all_logits, all_real, balanced_acc_scores = [], [], []
    s = time.time()
    for emb, y_onehot in ds:
      emb.shape.assert_has_rank(3)
      assert emb.shape[2] == FLAGS.embedding_dimension
      y_onehot.shape.assert_has_rank(2)
      assert y_onehot.shape[1] == len(FLAGS.label_list)

      logits = model(emb, training=False)
      all_logits.extend(logits.numpy()[:, 1])
      all_real.extend(y_onehot.numpy()[:, 1])
      balanced_acc_scores.append(metrics.balanced_accuracy(
          np.argmax(y_onehot.numpy(), 1), np.argmax(logits.numpy(), 1)))
      acc_m.update_state(y_true=tf.argmax(y_onehot, 1),
                         y_pred=tf.argmax(logits, 1))
      xent_m.update_state(y_true=y_onehot, y_pred=logits)
      ex_count += logits.shape[0]
      count += 1
      logging.info('Saw %i examples after %i iterations as %.2f secs...',
                   ex_count, count,
                   time.time() - s)
    if FLAGS.calculate_equal_error_rate:
      eer_score = metrics.calculate_eer(all_real, all_logits)
    auc_score = metrics.calculate_auc(all_real, all_logits)
    dprime_score = metrics.dprime_from_auc(auc_score)
    with writer.as_default():
      tf.summary.scalar('accuracy', acc_m.result().numpy(), step=int(step))
      tf.summary.scalar('balanced_accuracy', np.mean(balanced_acc_scores),
                        step=int(step))
      tf.summary.scalar('xent_loss', xent_m.result().numpy(), step=int(step))
      tf.summary.scalar('auc', auc_score, step=int(step))
      tf.summary.scalar('dprime', dprime_score, step=int(step))
      if FLAGS.calculate_equal_error_rate:
        tf.summary.scalar('eer', eer_score, step=int(step))
    logging.info('Done with eval step: %s in %.2f secs.', step, time.time() - s)


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.embedding_name
  assert FLAGS.embedding_dimension
  assert FLAGS.label_name
  assert FLAGS.label_list
  assert FLAGS.bucket_boundaries
  assert FLAGS.batch_size
  assert FLAGS.logdir

  eval_and_report()


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
