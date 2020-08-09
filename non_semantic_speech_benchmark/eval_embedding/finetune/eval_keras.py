# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import tensorflow.compat.v2 as tf

from non_semantic_speech_benchmark.eval_embedding import eer_metric
from non_semantic_speech_benchmark.eval_embedding.finetune import get_data
from non_semantic_speech_benchmark.eval_embedding.finetune import models

FLAGS = flags.FLAGS

flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_string('sk', None, 'Samples name.')
flags.DEFINE_alias('samples_key', 'sk')
flags.DEFINE_integer('ml', 16000, 'Minimum length.')
flags.DEFINE_alias('min_length', 'ml')
flags.DEFINE_string('label_key', None, 'Name of label to use.')
flags.DEFINE_list('label_list', None, 'List of possible label values.')

flags.DEFINE_integer('batch_size', None, 'The number of images in each batch.')
flags.DEFINE_integer('tbs', None, 'not used')

flags.DEFINE_integer('nc', None, 'num_clusters')
flags.DEFINE_boolean('ubn', None, 'Whether to normalize')
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


def eval_and_report():
  """Eval on voxceleb."""
  tf.logging.info('samples_key: %s', FLAGS.samples_key)
  logging.info('Logdir: %s', FLAGS.logdir)
  logging.info('Batch size: %s', FLAGS.batch_size)

  writer = tf.summary.create_file_writer(FLAGS.eval_dir)
  num_classes = len(FLAGS.label_list)
  model = models.get_keras_model(num_classes, FLAGS.ubn, num_clusters=FLAGS.nc)
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
        samples_key=FLAGS.samples_key,
        min_length=FLAGS.min_length,
        label_key=FLAGS.label_key,
        label_list=FLAGS.label_list,
        batch_size=FLAGS.batch_size,
        loop_forever=False,
        shuffle=False)
    logging.info('Got dataset for eval step: %s.', step)
    if FLAGS.take_fixed_data:
      ds = ds.take(FLAGS.take_fixed_data)

    acc_m = tf.keras.metrics.Accuracy()
    xent_m = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)

    logging.info('Starting the ds loop...')
    count, ex_count = 0, 0
    s = time.time()
    all_logits, all_real = [], []
    for wav_samples, y_onehot in ds:
      wav_samples.shape.assert_is_compatible_with(
          [None, FLAGS.min_length])
      y_onehot.shape.assert_is_compatible_with(
          [None, len(FLAGS.label_list)])

      logits = model(wav_samples, training=False)
      all_logits.extend(logits.numpy()[:, 1])
      all_real.extend(y_onehot.numpy()[:, 1])
      acc_m.update_state(y_true=tf.argmax(y_onehot, 1),
                         y_pred=tf.argmax(logits, 1))
      xent_m.update_state(y_true=y_onehot, y_pred=logits)
      ex_count += logits.shape[0]
      count += 1
      logging.info('Saw %i examples after %i iterations as %.2f secs...',
                   ex_count, count,
                   time.time() - s)
    if FLAGS.calculate_equal_error_rate:
      eer_score = eer_metric.calculate_eer(all_logits, all_real)
    with writer.as_default():
      tf.summary.scalar('accuracy', acc_m.result().numpy(), step=int(step))
      tf.summary.scalar('xent_loss', xent_m.result().numpy(), step=int(step))
      if FLAGS.calculate_equal_error_rate:
        tf.summary.scalar('eer', eer_score, step=int(step))
    logging.info('Done with eval step: %s in %.2f secs.', step, time.time() - s)


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.samples_key
  assert FLAGS.label_key
  assert FLAGS.label_list
  assert FLAGS.logdir

  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  eval_and_report()


if __name__ == '__main__':
  app.run(main)
