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

"""LSTM model for classifying 20 news group."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import pickle
import random
import sys
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.keras import callbacks
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import optimizers
from tensorflow.compat.v1.keras import regularizers
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl

# parameters
FLAGS = flags.FLAGS

flags.DEFINE_integer('random_seed', 1234, 'The random seed')
flags.DEFINE_integer('fix_len', 250, 'sequence length')
flags.DEFINE_integer('batch_size', 100, 'The number of images in each batch.')
flags.DEFINE_integer('num_epochs', 1000, 'The number of trainig steps')
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate')
flags.DEFINE_integer('vocab_size', 30000, 'vocab size')
flags.DEFINE_integer('n_class', 20, 'number of classes in total')
flags.DEFINE_integer('emb_size', 300, 'The word embedding dimensions')
flags.DEFINE_integer('hidden_lstm_size', 30,
                     'The number of hidden units in generator LSTM.')
flags.DEFINE_float('clip_norm', 10,
                   'The threshold of the norm for cliping gradient')
flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate')
flags.DEFINE_float('dropout_rate_lstm', 0.0, 'dropout rate in LSTM')
flags.DEFINE_float(
    'reg_weight', 0,
    'The regularization weight for parameters in lstm and dense layers.')
flags.DEFINE_string(
    'data_pkl_file',
    '20news_encode_maxlen250_vs30000_in0-2-4-6-8-10-12-14-16-18_trfrac0.9.pkl',
    'directory of encoded numpy data')
flags.DEFINE_string('tr_out_dir', './',
                    'Directory where to write log and models.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')
flags.DEFINE_bool('variational', False, 'variational last layer')

FLAGS = flags.FLAGS


def create_out_dir(params):
  """Setup the output directory."""

  if not params['variational']:
    if params['dropout_rate'] == 0 and params['dropout_rate_lstm'] == 0:
      sub_dir = 'vanilla'
    elif params['dropout_rate'] > 0 and params['dropout_rate_lstm'] > 0:
      sub_dir = 'dropout'
    elif params['dropout_rate'] > 0 and params['dropout_rate_lstm'] == 0:
      sub_dir = 'll-dropout'
  else:
    sub_dir = 'll-svi'

  out_dir = os.path.join(params['tr_out_dir'], sub_dir)
  new_job_id = 1
  if tf.gfile.Exists(out_dir):
    # if out_dir exist, then count how many subdirectories are there
    # each subdirectory contains results from an indepdendent training
    sub_dirs = tf.gfile.ListDirectory(out_dir)
    if sub_dirs:
      sub_dirs_int = [int(sub_dir) for sub_dir in sub_dirs]
      max_job_id = max(sub_dirs_int)
      new_job_id = max_job_id + 1

  tr_log_dir = os.path.join(params['tr_out_dir'], sub_dir, str(new_job_id),
                            'log')
  params['tr_log_dir_tr'] = os.path.join(tr_log_dir, 'tr')
  params['tr_model_dir'] = tr_log_dir.replace('log', 'model')

  tf.io.gfile.makedirs(params['tr_out_dir'])
  tf.io.gfile.makedirs(params['tr_log_dir_tr'])
  tf.io.gfile.makedirs(params['tr_model_dir'])
  logging.info('output in %s', params['tr_out_dir'])


def pickle_load(f):
  # make pickle load compatible for both PY2 and PY3
  if sys.version_info.major > 2:
    data = pickle.load(f, encoding='latin1')  # pylint: disable=unexpected-keyword-arg
  else:
    data = pickle.load(f)
  return data


def load_np_dataset(data_pkl_file):
  """Load dataset which contains numpy arrays for tr, val, and test."""

  logging.info('data_pkl_file=%s', data_pkl_file)
  with tf.gfile.Open(data_pkl_file, 'rb') as f:
    # xx_sample_examples is of the size n_sample x max_example_len.
    # for sentences > max_len, truncate from the beginning.
    # for sentences < max_len, pad with 0s.
    # dataset was generated under PY2,
    # need to add encoding=latin1 for PY3 compatibility
    in_sample_examples = pickle_load(f)
    in_sample_labels = pickle_load(f)
    oos_examples = pickle_load(f)  # for oos_examples
    oos_labels = pickle_load(f)  # for oos_labels

    dev_in_sample_examples = pickle_load(f)
    dev_in_sample_labels = pickle_load(f)
    dev_oos_examples = pickle_load(f)  # for dev_oos_examples
    dev_oos_labels = pickle_load(f)  # for dev_oos_labels

    test_in_sample_examples = pickle_load(f)
    test_in_sample_labels = pickle_load(f)
    test_oos_examples = pickle_load(f)
    test_oos_labels = pickle.load(f)

    vocab = pickle_load(f)

  logging.info('dev_in_sample_labels=%s, len(dev_in_sample_labels)=%s',
               dev_in_sample_labels, len(dev_in_sample_labels))

  n_class_in = np.max(in_sample_labels) + 1

  data = collections.namedtuple('data', [
      'in_sample_examples',
      'in_sample_labels',
      'oos_examples',
      'oos_labels',
      'dev_in_sample_examples',
      'dev_in_sample_labels',
      'dev_oos_examples',
      'dev_oos_labels',
      'test_in_sample_examples',
      'test_in_sample_labels',
      'test_oos_examples',
      'test_oos_labels',
      'vocab',
      'n_class_in',
  ])

  return data(in_sample_examples, in_sample_labels, oos_examples, oos_labels,
              dev_in_sample_examples, dev_in_sample_labels, dev_oos_examples,
              dev_oos_labels, test_in_sample_examples, test_in_sample_labels,
              test_oos_examples, test_oos_labels, vocab, n_class_in)


def rnn_model(params, training_dr_lstm=True, training_dr_ll=True):
  """RNN model for text."""
  input_shape = (params['fix_len'])
  seq_input = layers.Input(shape=input_shape)
  # vocab+1 because of padding
  seq_emb = layers.Embedding(
      params['vocab_size'] + 1,
      params['emb_size'],
      input_length=params['fix_len'])(
          seq_input)
  lstm_out = layers.LSTM(
      params['hidden_lstm_size'], dropout=params['dropout_rate_lstm'])(
          seq_emb, training=training_dr_lstm)
  out = layers.Dropout(
      rate=params['dropout_rate'], seed=params['random_seed'])(
          lstm_out, training=training_dr_ll)
  if params['variational']:
    # scale kl loss by number of training examples.
    # larger training dataset depends less on prior
    def scaled_kl_fn(p, q, _):
      return tfp.distributions.kl_divergence(q, p) / params['n_train']

    logits = tfpl.DenseReparameterization(
        params['n_class_in'],
        activation=None,
        kernel_divergence_fn=scaled_kl_fn,
        bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
        name='last_layer')(
            out)
  else:
    logits = layers.Dense(
        params['n_class_in'],
        activation=None,
        kernel_regularizer=regularizers.l2(params['reg_weight']),
        bias_regularizer=regularizers.l2(params['reg_weight']),
        name='last_layer')(
            out)
  probs = layers.Softmax(axis=1)(logits)
  return models.Model(seq_input, probs, name='rnn')


def main(_):

  random.seed(FLAGS.random_seed)

  params = {
      'num_epochs': FLAGS.num_epochs,
      'fix_len': FLAGS.fix_len,
      'batch_size': FLAGS.batch_size,
      'n_class': FLAGS.n_class,
      'emb_size': FLAGS.emb_size,
      'vocab_size': FLAGS.vocab_size,
      'hidden_lstm_size': FLAGS.hidden_lstm_size,
      'dropout_rate': FLAGS.dropout_rate,
      'dropout_rate_lstm': FLAGS.dropout_rate_lstm,
      'learning_rate': FLAGS.learning_rate,
      'reg_weight': FLAGS.reg_weight,
      'tr_out_dir': FLAGS.tr_out_dir,
      'data_pkl_file': FLAGS.data_pkl_file,
      'master': FLAGS.master,
      'clip_norm': FLAGS.clip_norm,
      'random_seed': FLAGS.random_seed,
      'variational': FLAGS.variational,
  }

  # setup output directory
  create_out_dir(params)

  # load dataset
  data = load_np_dataset(params['data_pkl_file'])

  # has to convert the following np.int to int for PY3 compatibility
  params['n_class_in'] = int(data.n_class_in)
  params['n_train'] = data.in_sample_labels.shape[0]

  # print and write parameter settings
  logging.info(params)
  with tf.io.gfile.GFile(
      os.path.join(params['tr_model_dir'], 'params.json'), mode='w') as f:
    f.write(json.dumps(params, sort_keys=True))

  # build model
  logging.info('Building Keras RNN model')
  model = rnn_model(params)
  logging.info('Compiling model.')
  metrics = ['sparse_categorical_crossentropy', 'acc']
  model.compile(
      optimizer=optimizers.Adam(
          learning_rate=params['learning_rate'], clipnorm=params['clip_norm']),
      loss=['sparse_categorical_crossentropy'],
      metrics=metrics)

  # setup ckpt, earlystop, tensorboard
  earlystop = callbacks.EarlyStopping(
      monitor='val_loss', min_delta=0.0001, patience=5, verbose=1)
  tensorboard = callbacks.TensorBoard(log_dir=params['tr_log_dir_tr'])
  training_callbacks = [earlystop, tensorboard]

  # fit model
  model.fit(
      data.in_sample_examples,
      data.in_sample_labels,
      epochs=params['num_epochs'],
      batch_size=params['batch_size'],
      callbacks=training_callbacks,
      shuffle=True,
      verbose=1,
      validation_data=(data.dev_in_sample_examples, data.dev_in_sample_labels))

  # save model
  model.save_weights(params['tr_model_dir'] + '/model.ckpt')


if __name__ == '__main__':
  tf.app.run()
