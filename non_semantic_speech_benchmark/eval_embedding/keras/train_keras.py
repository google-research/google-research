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
"""Trains on embeddings using Keras."""

import os
from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from non_semantic_speech_benchmark.eval_embedding.keras import get_data
from non_semantic_speech_benchmark.eval_embedding.keras import models

FLAGS = flags.FLAGS

flags.DEFINE_string('file_pattern', None, 'Dataset location.')
flags.DEFINE_string('en', None, 'Embedding name.')
flags.DEFINE_string('ed', None, 'Embedding dimension.')
flags.DEFINE_string('label_name', None, 'Name of label to use.')
flags.DEFINE_list('label_list', None, 'List of possible label values.')

flags.DEFINE_integer('tbs', 1, 'Hyperparameter: batch size.')
flags.DEFINE_integer('shuffle_buffer_size', None, 'shuffle_buffer_size')

flags.DEFINE_integer('nc', None, 'num_clusters')
flags.DEFINE_boolean('ubn', None, 'Whether to use batch normalization.')
flags.DEFINE_float('lr', 0.001, 'Hyperparameter: learning rate.')

flags.DEFINE_string('logdir', None,
                    'Path to directory where to store summaries.')

flags.DEFINE_integer('training_steps', 1000,
                     'The number of steps to run training for.')
flags.DEFINE_integer('measurement_store_interval', 10,
                     'The number of steps between storing objective value in '
                     'measurements.')


def train_and_report(debug=False):
  """Trains the classifier."""
  tf.logging.info('embedding_name: %s', FLAGS.en)
  tf.logging.info('Logdir: %s', FLAGS.logdir)
  tf.logging.info('Batch size: %s', FLAGS.tbs)

  with tf.Graph().as_default():
    ds = get_data.get_data(
        file_pattern=FLAGS.file_pattern,
        reader=tf.data.TFRecordDataset,
        embedding_name=FLAGS.en,
        embedding_dim=FLAGS.ed,
        preaveraged=False,
        label_name=FLAGS.label_name,
        label_list=FLAGS.label_list,
        batch_size=FLAGS.tbs,
        loop_forever=True,
        shuffle=True,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size)
    emb, y_onehot = ds.make_one_shot_iterator().get_next()
    emb.shape.assert_has_rank(3)
    assert emb.shape[2] == FLAGS.ed
    y_onehot.shape.assert_has_rank(2)
    assert y_onehot.shape[1] == len(FLAGS.label_list)

    # Loss and train.
    loss, train_op = make_graph(emb, y_onehot, FLAGS.ubn, FLAGS.nc)

    # Run training.
    train_ops_to_run = {
        'train_loss': loss,
        'train_op': train_op,
        'tensorboard_summary': tf.summary.merge_all()
    }
    if debug: return
    train_loop(train_ops_to_run)
    tf.logging.info('Finished training.')


def train_loop(train_ops_to_run):
  """Simple train loop."""
  summary_writer = None
  assert FLAGS.logdir
  summary_writer = tf.summary.FileWriter(FLAGS.logdir)
  saver = tf.train.Saver(max_to_keep=None)
  save_fn = os.path.join(FLAGS.logdir, 'ckpt')

  # Gstep must be in TF so that reloads work properly.
  assert 'gstep' not in train_ops_to_run
  train_ops_to_run['gstep'] = tf.train.get_or_create_global_step()
  gstep = 0

  with tf.train.SingularMonitoredSession(
      hooks=[tf.train.StepCounterHook(summary_writer=summary_writer)],
      ) as sess:
    while gstep < FLAGS.training_steps:
      train_values = sess.run(train_ops_to_run)
      # Report objective value periodically.
      # Step must be in TF so it deals with job restarts correctly.
      gstep = train_values['gstep']
      if gstep % 10 == 0:
        tf.logging.info('{{\'step\': {}, \'train_loss\': {}}}'.format(
            gstep, train_values['train_loss']))
      if gstep % FLAGS.measurement_store_interval == 0:
        tensorboard_summary = train_values['tensorboard_summary']
        summary_writer.add_summary(tensorboard_summary, gstep)
        saver.save(sess.raw_session(), save_fn, global_step=gstep)


def make_graph(emb, y_onehot, ubn=None, nc=None):
  """Make a graph on data."""
  num_classes = y_onehot.shape[1]
  model = models.get_keras_model(num_classes, ubn, num_clusters=nc)
  logits = model(emb, training=True)
  logits.shape.assert_is_compatible_with(y_onehot.shape)
  for u_op in model.updates:
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u_op)

  # Loss.
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
      y_true=y_onehot,
      y_pred=logits)
  tf.summary.scalar('xent_loss', loss)

  # Gradient.
  opt = tf.train.AdamOptimizer(
      learning_rate=FLAGS.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
  update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
  with tf.control_dependencies(update_ops):
    total_loss = tf.identity(loss)
  var_list = tf.trainable_variables()
  assert var_list
  train_op = opt.minimize(total_loss, tf.train.get_or_create_global_step(),
                          var_list)

  return total_loss, train_op


def main(unused_argv):
  assert FLAGS.file_pattern
  assert FLAGS.shuffle_buffer_size
  assert FLAGS.en
  assert FLAGS.ed
  assert FLAGS.label_name
  assert FLAGS.label_list
  assert FLAGS.logdir
  train_and_report()


if __name__ == '__main__':
  app.run(main)
