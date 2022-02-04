# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Run Classifier to score all data.

Data scorer wtih classifier.

This file is intended for a dataset that is split into 14 chunks.
"""

import os
import pathlib
import pickle
from typing import Any, Dict, Sequence, Union, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import transformers



FLAGS = flags.FLAGS

STRUCT = 'struct'
ONE_HOT = 'onehot'

flags.DEFINE_integer(
    'train_data_size', default=500_000,
    help='Size of training data.')
flags.DEFINE_integer(
    'epochs', default=5,
    help='Epochs of training.')
flags.DEFINE_integer(
    'batch_size', default=64,
    help='Batch size.')
flags.DEFINE_integer(
    'eval_freq', default=1_000,
    help='Num steps between evals.')
flags.DEFINE_string(
    'loss', default=STRUCT,
    help='Type of loss to use.')
flags.DEFINE_string(
    'cns_save_dir', default='routing/domain_clf/',
    help='Path to save dir.')
flags.DEFINE_string(
    'bert_path', default='bert_base',
    help='Path to bert base.')
flags.DEFINE_string(
    'routing_path', default='routing/',
    help='Path to save logs and losses.')


class CNSCheckpointCallback(tf.keras.callbacks.Callback):
  """Write checkpoints to CNS."""

  def __init__(self,
               tmp_save_dir = '/tmp/model_ckpt',
               cns_save_dir = 'routing/domain_clf/'):
    super().__init__()
    self.best_loss = 10000
    self.tmp_save_dir = tmp_save_dir
    self.cns_save_dir = cns_save_dir

  def _save_model(self, new_loss):
    pathlib.Path(self.tmp_save_dir).mkdir(parents=True, exist_ok=True)
    self.model.save_pretrained(self.tmp_save_dir)
    for filename in tf.io.gfile.glob(self.tmp_save_dir + '/*'):
      print(filename)
      base_filename = os.path.basename(filename)
      if not tf.io.gfile.exists(self.cns_save_dir):
        tf.io.gfile.mkdir(self.cns_save_dir)
      tf.io.gfile.copy(
          filename, self.cns_save_dir + '/' + base_filename, overwrite=True)
      with tf.io.gfile.GFile(
          self.cns_save_dir + '/chkpt{}.txt'.format(new_loss), 'w') as f:
        f.write(str(new_loss))

  def on_test_end(self, logs = None):
    new_loss = logs['loss']
    if new_loss < self.best_loss:
      self.best_loss = new_loss
      self._save_model(new_loss)


def custom_loss_function(y_true, y_pred):
  """Structed loss function."""
  lowest_loss = tf.reduce_min(y_true, axis=1)
  lowest_index = tf.argmin(y_true, axis=1)
  indexes = tf.expand_dims(lowest_index, 1)
  rows = tf.expand_dims(tf.range(tf.shape(indexes)[0], dtype=tf.int64), 1)
  ind = tf.concat([rows, indexes], axis=1)
  s_xystar = tf.gather_nd(y_pred, ind)

  # cost of each point
  cost = tf.subtract(y_true, tf.expand_dims(lowest_loss, 1))
  addition = tf.add(cost, y_pred)
  sub = tf.subtract(addition, tf.expand_dims(s_xystar, 1))
  # max_y
  max_y = tf.reduce_max(sub, axis=1)
  return tf.maximum(0.0, max_y)


class CustomAccuracy(tf.keras.metrics.Metric):
  """Compute accuracy with explicit score inputs."""

  def __init__(self, name = 'custom_acc', **kwargs):
    super(CustomAccuracy, self).__init__(name=name, **kwargs)
    self.custom_acc = tf.keras.metrics.Accuracy(name='custom_acc', dtype=None)

  def update_state(
      self,
      y_true,
      y_pred,
      sample_weight = None
  ):
    lowest_index = tf.argmin(y_true, axis=1)
    highest_logit = tf.argmax(y_pred, axis=1)
    self.custom_acc.update_state(lowest_index, highest_logit)

  def result(self):
    return self.custom_acc.result()

  def reset_states(self):
    self.custom_acc.reset_states()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with tf.io.gfile.GFile(
      FLAGS.cns_save_dir + '/analysis_full/data.pkl', 'rb') as f:
    wmt_labels = pickle.load(f)

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    path = FLAGS.bert_path  # pretrained model (ie. bert-base)
    cache_dir = '/tmp/'
    config = transformers.BertConfig.from_pretrained(
        os.path.join(path, 'config.json'), num_labels=100, cache_dir=cache_dir)
    tokenizer = transformers.BertTokenizer.from_pretrained(
        path, cache_dir=cache_dir)
    model = transformers.TFBertForSequenceClassification.from_pretrained(
        os.path.join(path, 'tf_model.h5'), config=config, cache_dir=cache_dir)

  with tf.device('/CPU:0'):
    all_train_ds = []
    all_eval_ds = []
    for i in range(100):
      data_dir = (FLAGS.routing_path + '/cluster_data/k100/id_{}/'
                  .format(i))
      wmt_train = 'test_large.tsv'
      train_files = [data_dir + '/' + wmt_train]

      train_data = tf.data.experimental.CsvDataset(
          train_files,
          record_defaults=[tf.string, tf.string],
          field_delim='\t',
          use_quote_delim=False)

      def to_features_dict(eng, _):
        return eng

      train_data = train_data.map(to_features_dict)

      all_scores = [np.array(wmt_labels[i][j]) for j in range(100)]
      all_scores = np.stack(all_scores)
      if FLAGS.loss == STRUCT:
        split_scores = tf.split(all_scores, all_scores.shape[1], axis=1)
        split_scores = [tf.reshape(scores, -1) for scores in split_scores]
        label = tf.data.Dataset.from_tensor_slices(split_scores)
      elif FLAGS.loss == ONE_HOT:
        labels = np.argmin(all_scores, axis=0)
        label = tf.data.Dataset.from_tensor_slices(labels)
      train_data_w_label = tf.data.Dataset.zip((train_data, label))
      eval_data_w_label = train_data_w_label.take(100)
      train_data_w_label = train_data_w_label.skip(100)

      all_train_ds.append(train_data_w_label)
      all_eval_ds.append(eval_data_w_label)

    sample_dataset = tf.data.experimental.sample_from_datasets(all_train_ds)
    eval_sample_dataset = tf.data.experimental.sample_from_datasets(all_eval_ds)

  with tf.device('/CPU:0'):
    text = []
    labels = []
    for ex in sample_dataset:
      text.append(str(ex[0].numpy()))
      labels.append(ex[1].numpy())
      if len(labels) > FLAGS.train_data_size:
        break

    encoding = tokenizer(
        text,
        return_tensors='tf',
        padding=True,
        truncation=True,
        max_length=128)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(encoding), labels))
    train_dataset = train_dataset.batch(
        FLAGS.batch_size).shuffle(100_000).repeat(20)

    eval_text = []
    eval_labels = []
    for ex in eval_sample_dataset:
      eval_text.append(str(ex[0].numpy()))
      eval_labels.append(ex[1].numpy())
      if len(eval_labels) > 5000:
        break

    eval_encoding = tokenizer(
        eval_text,
        return_tensors='tf',
        padding=True,
        truncation=True,
        max_length=128)
    eval_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(eval_encoding), eval_labels))
    eval_dataset = eval_dataset.batch(FLAGS.batch_size)

  num_train_steps = int(FLAGS.train_data_size / FLAGS.batch_size) * FLAGS.epochs
  decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=4e-5,
      decay_steps=num_train_steps,
      end_learning_rate=0)

  with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=decay_schedule)
    if FLAGS.loss == ONE_HOT:
      model.compile(
          optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    elif FLAGS.loss == STRUCT:
      model.compile(
          optimizer=optimizer, loss=custom_loss_function,
          metrics=[CustomAccuracy()])

  steps_per_epoch = int(FLAGS.train_data_size / FLAGS.batch_size)
  num_meta_epochs = int(steps_per_epoch / FLAGS.eval_freq * FLAGS.epochs)
  logging.info('Training %d epochs of %d steps each', num_meta_epochs,
               steps_per_epoch)
  model.fit(
      train_dataset,
      epochs=num_meta_epochs,
      steps_per_epoch=FLAGS.eval_freq,
      validation_freq=1,
      validation_data=eval_dataset,
      callbacks=CNSCheckpointCallback(cns_save_dir=FLAGS.cns_save_dir),
      verbose=2)  # 1 line per epoch logging


if __name__ == '__main__':
  app.run(main)
