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

r"""Build an autoregressive generative model for DNA sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random
from absl import flags
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow.compat.v1 as tf  # tf
from genomics_ood import utils
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import training as contrib_training

# parameters
FLAGS = flags.FLAGS

flags.DEFINE_integer('random_seed', 1234, 'The random seed')
flags.DEFINE_integer('batch_size', 100, 'The number of images in each batch.')
flags.DEFINE_integer('num_steps', 1000000, 'The number of trainig steps')
flags.DEFINE_integer('val_freq', 1000, 'How often to eval validation (# steps)')
flags.DEFINE_float('learning_rate', 0.0005, 'The learning rate')
flags.DEFINE_boolean(
    'emb_variable', False,
    'If the word embedding is variables. If not, use one-hot encoding.')
flags.DEFINE_integer('emb_size', 4, 'The word embedding dimensions')
flags.DEFINE_integer('hidden_lstm_size', 2000,
                     'The number of hidden units in LSTM.')
flags.DEFINE_boolean('norm_lstm', False,
                     'If turn on the layer normalization for LSTM.')
flags.DEFINE_float('dropout_rate', 0.1, 'The learning rate')
flags.DEFINE_string(
    'reg_type', 'l2',
    'l2 or l1 regularization for parameters in lstm and dense layers.')
flags.DEFINE_float(
    'reg_weight', 0.0,
    'The regularization weight for parameters in lstm and dense layers.')
flags.DEFINE_integer('seq_len', 250, 'sequence length')
flags.DEFINE_float('mutation_rate', 0.0, 'Mutation rate for data augmentation.')
flags.DEFINE_integer(
    'filter_label', -1,
    ('If only sequences from the class=filter_label are used for training.'
     'if -1, no filter.'))
flags.DEFINE_string('in_tr_data_dir', '/tmp/data/before_2011_in_tr',
                    'data directory of in-distribution training')
flags.DEFINE_string('in_val_data_dir', '/tmp/data/between_2011-2016_in_val',
                    'data directory of in-distribution validation')
flags.DEFINE_string('ood_val_data_dir', '/tmp/data/between_2011-2016_ood_val',
                    'data directory of OOD validation')
flags.DEFINE_string('out_dir', '/tmp/out_generative',
                    'Directory where to write log and models.')
flags.DEFINE_boolean('save_meta', False, 'Save meta graph file for each ckpt.')
flags.DEFINE_string('master', '', 'TensorFlow master to use.')

FLAGS = flags.FLAGS


def create_out_dir(params):
  """Setup the output directory."""
  params.in_tr_data_dir = utils.clean_last_slash_if_any(params.in_tr_data_dir)
  params.in_val_data_dir = utils.clean_last_slash_if_any(params.in_val_data_dir)
  params.ood_val_data_dir = utils.clean_last_slash_if_any(
      params.ood_val_data_dir)

  sub_dir = ('tr%s_ood%s_rnn_l%d_bs%d_lr%.4f'
             '_hr%d_nr%s_reg%s_regw%.6f_fi%d_mt%.2f') % (
                 os.path.basename(params.in_tr_data_dir),
                 os.path.basename(params.ood_val_data_dir), params.seq_len,
                 params.batch_size, params.learning_rate,
                 params.hidden_lstm_size, params.norm_lstm, params.reg_type,
                 params.reg_weight, params.filter_label, params.mutation_rate)
  log_dir = os.path.join(params.out_dir, sub_dir, 'log')
  params.add_hparam('log_dir_in_tr', os.path.join(log_dir, 'in_tr'))
  params.add_hparam('log_dir_in_val', os.path.join(log_dir, 'in_val'))
  params.add_hparam('model_dir', log_dir.replace('log', 'model'))

  if not tf.gfile.Exists(params.out_dir):
    tf.gfile.MakeDirs(params.out_dir)
  if not tf.gfile.Exists(params.log_dir_in_tr):
    tf.gfile.MakeDirs(params.log_dir_in_tr)
  if not tf.gfile.Exists(params.log_dir_in_val):
    tf.gfile.MakeDirs(params.log_dir_in_val)
  if not tf.gfile.Exists(params.model_dir):
    tf.gfile.MakeDirs(params.model_dir)

  tf.logging.info('model_dir=%s', params.model_dir)


def filter_for_label(features, target_label):
  """A filter for dataset to get seqs with a specific label."""
  ### TODO(jjren) not working
  return tf.equal(features['y'],
                  tf.convert_to_tensor(target_label, dtype=tf.int32))


def load_datasets(params, mode_eval=False):
  """load class labels, in_tr_data, in_val_data, ood_val_data."""
  if mode_eval:  # For evaluation, no need to prepare training data
    in_tr_dataset = None
  else:
    in_tr_file_list = [
        os.path.join(params.in_tr_data_dir, x)
        for x in tf.gfile.ListDirectory(params.in_tr_data_dir)
        if params.in_tr_file_pattern in x
    ]

    # load in-distribution training sequence
    in_tr_data_file_list = [x for x in in_tr_file_list if '.tfrecord' in x]
    tf.logging.info('in_tr_data_file_list=%s', in_tr_data_file_list)

    def parse_single_tfexample_addmutations_short(unused_key, v):
      return utils.parse_single_tfexample_addmutations(unused_key, v,
                                                       params.mutation_rate,
                                                       params.seq_len)

    # for training a background model, we mutate input sequences
    if params.mutation_rate == 0:
      in_tr_dataset = tf.data.TFRecordDataset(in_tr_data_file_list).map(
          lambda v: utils.parse_single_tfexample(v, v))
    else:
      in_tr_dataset = tf.data.TFRecordDataset(in_tr_data_file_list).map(
          lambda v: parse_single_tfexample_addmutations_short(v, v))

    if params.filter_label != -1:

      def filter_fn(v):
        return filter_for_label(v, params.filter_label)

      in_tr_dataset = in_tr_dataset.filter(filter_fn)

  # in-distribution validation
  in_val_data_file_list = [
      os.path.join(params.in_val_data_dir, x)
      for x in tf.gfile.ListDirectory(params.in_val_data_dir)
      if params.in_val_file_pattern in x and '.tfrecord' in x
  ]
  tf.logging.info('in_val_data_file_list=%s', in_val_data_file_list)
  in_val_dataset = tf.data.TFRecordDataset(
      in_val_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))

  # ood validation
  ood_val_data_file_list = [
      os.path.join(params.ood_val_data_dir, x)
      for x in tf.gfile.ListDirectory(params.ood_val_data_dir)
      if params.ood_val_file_pattern in x and '.tfrecord' in x
  ]
  tf.logging.info('ood_val_data_file_list=%s', ood_val_data_file_list)
  ood_val_dataset = tf.data.TFRecordDataset(
      ood_val_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))

  return in_tr_dataset, in_val_dataset, ood_val_dataset


class SeqModel(object):
  """DNA sequence modeling."""

  def __init__(self, params):
    """Create the model."""
    self._params = params

    self._make_dataset()
    self._make_placeholders()
    if self._params.emb_variable:
      self._make_variables()
    else:
      self._one_hot_encode_x()
    self._make_rnn_model()
    self._make_losses()
    self._make_summary_stats()
    self._make_train_op()

  def _make_dataset(self):
    """make data generators."""
    self.handle = tf.placeholder(tf.string, shape=[])
    self.iterator = tf.data.Iterator.from_string_handle(self.handle, {
        'x': tf.int32,
        'y': tf.int32
    }, {
        'x': [None, self._params.seq_len],
        'y': [None]
    })
    features = self.iterator.get_next()
    self.x, self.y0 = features['x'], features['y']

  def _make_placeholders(self):
    """Make placeholders for dropout rate."""
    self.dropout_rate = tf.placeholder_with_default(
        self._params.dropout_rate, shape=(), name='dropout_rnn')

  def _make_variables(self):
    """make variables."""
    # emb_size must equal to vocab_size,
    # otherwise exceed vocab will be encoded as zeros
    tf.logging.info('using variable dict for embedding')
    self.emb_dict = tf.Variable(
        tf.one_hot(
            list(range(self._params.vocab_size)), depth=self._params.emb_size))
    self.x_emb = tf.nn.embedding_lookup(
        self.emb_dict, tf.cast(self.x, dtype=tf.int64), name='embx')

  def _one_hot_encode_x(self):
    """Make embedding layer."""
    # input for encoder
    tf.logging.info('use one hot encoding')
    self.x_emb = tf.one_hot(
        tf.cast(self.x, dtype=tf.int64), depth=self._params.vocab_size)
    tf.logging.info('shape of x_emb=%s', self.x_emb.shape)

  def _make_rnn_model(self):
    """Make rnn model."""
    self.y = tf.cast(self.x[:, 1:], dtype=tf.int64)
    self.y_emb = tf.one_hot(self.y, depth=self._params.emb_size)
    tf.logging.info('y.shape=%s', self.y.shape)

    lstm_fw_cell_g = contrib_rnn.LayerNormBasicLSTMCell(
        self._params.hidden_lstm_size,
        layer_norm=self._params.norm_lstm,
        dropout_keep_prob=1 - self.dropout_rate)
    lstm_hidden, _ = tf.nn.dynamic_rnn(
        lstm_fw_cell_g, self.x_emb, dtype=tf.float32)
    # stagger two directional vectors so that the backward RNN does not reveal
    # medium.com/@plusepsilon/the-bidirectional-language-model-1f3961d1fb27
    self.logits = tf.layers.dense(
        lstm_hidden[:, :-1, :],
        units=self._params.vocab_size,
        activation=None,
        name='logits')
    tf.logging.info('shape of logits=%s', self.logits.shape)

    # cross entropy
    self.loss_i_t = tf.nn.softmax_cross_entropy_with_logits(
        labels=self.y_emb, logits=self.logits)
    self.loss_i = tf.reduce_mean(self.loss_i_t, axis=1)

  def _make_losses(self):
    """make loss functions."""
    self.loss = tf.reduce_mean(self.loss_i)
    # l2 norm
    self.variables = tf.trainable_variables()
    if self._params.reg_type == 'l2':
      self.loss_reg = tf.add_n(
          [tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name])
    else:
      self.loss_reg = tf.add_n([
          tf.reduce_sum(tf.abs(v))
          for v in self.variables
          if 'bias' not in v.name
      ])
    # total loss
    self.loss_total = self.loss + self._params.reg_weight * self.loss_reg

  def _make_summary_stats(self):
    """make summary stats."""
    probs = tf.nn.softmax(self.logits)
    pred_words = tf.argmax(probs, axis=2)
    self.acc_i_t = tf.equal(pred_words, tf.cast(self.y, dtype=tf.int64))
    self.acc_i = tf.reduce_mean(tf.cast(self.acc_i_t, dtype=tf.float32), axis=1)
    self.acc = tf.reduce_mean(self.acc_i)

    self.summary = tf.summary.merge([
        tf.summary.scalar('loss', self.loss),
        tf.summary.scalar('acc', self.acc),
        tf.summary.scalar('loss_total', self.loss_total),
        tf.summary.scalar('loss_reg', self.loss_reg)
    ])

  def _make_train_op(self):
    """make train op."""
    # training operations
    optimizer = tf.train.AdamOptimizer(self._params.learning_rate)
    grads = optimizer.compute_gradients(
        self.loss_total, var_list=self.variables)
    self.minimize = optimizer.apply_gradients(grads)

  def reset(self):
    """prepare sess."""
    # setup session and
    self.sess = tf.Session(self._params.master)
    self.sess.run(tf.global_variables_initializer())
    self.tr_writer = tf.summary.FileWriter(self._params.log_dir_in_tr,
                                           self.sess.graph)
    self.val_writer = tf.summary.FileWriter(self._params.log_dir_in_val,
                                            self.sess.graph)
    self.saver = tf.train.Saver(max_to_keep=500)

  def train(self, in_tr_dataset, in_val_dataset, ood_val_dataset, prev_steps):
    """training steps."""
    in_tr_dataset = in_tr_dataset.repeat().shuffle(1000).batch(
        self._params.batch_size)
    in_val_dataset = in_val_dataset.repeat().shuffle(1000).batch(
        self._params.batch_size)
    ood_val_dataset = ood_val_dataset.repeat().shuffle(1000).batch(
        self._params.batch_size)

    in_tr_iterator = in_tr_dataset.make_one_shot_iterator()
    in_val_iterator = in_val_dataset.make_one_shot_iterator()
    ood_val_iterator = ood_val_dataset.make_one_shot_iterator()

    self.in_tr_handle = self.sess.run(in_tr_iterator.string_handle())
    self.in_val_handle = self.sess.run(in_val_iterator.string_handle())
    self.ood_val_handle = self.sess.run(ood_val_iterator.string_handle())

    num_steps = self._params.num_steps
    for i in range(prev_steps, num_steps, 1):
      _, in_tr_loss, _, in_tr_acc, in_tr_summary = self.sess.run(
          [self.minimize, self.loss, self.loss_i, self.acc, self.summary],
          feed_dict={
              self.handle: self.in_tr_handle,
              self.dropout_rate: self._params.dropout_rate
          })
      if i % self._params.val_freq == 0:
        in_val_loss, in_val_loss_i, in_val_acc, in_val_summary = self.sess.run(
            [self.loss, self.loss_i, self.acc, self.summary],
            feed_dict={
                self.handle: self.in_val_handle,
                self.dropout_rate: 0
            })

        ood_val_loss, ood_val_loss_i, ood_val_acc, _ = self.sess.run(
            [self.loss, self.loss_i, self.acc, self.summary],
            feed_dict={
                self.handle: self.ood_val_handle,
                self.dropout_rate: 0
            })

        # auc using raw likelihood, larger for OOD
        neg = in_val_loss_i
        pos = ood_val_loss_i
        auc = roc_auc_score([0] * neg.shape[0] + [1] * pos.shape[0],
                            np.concatenate((neg, pos), axis=0))

        tf.logging.info(
            ('i=%d \t in_tr_loss=%.4f, in_val_loss=%.4f, ood_val_loss=%.4f\n'
             'in_tr_acc=%.4f, in_val_acc=%.4f, ood_val_acc=%.4f\n'
             'auc=%.4f'), i, in_tr_loss, in_val_loss, ood_val_loss, in_tr_acc,
            in_val_acc, ood_val_acc, auc)

        _ = self.saver.save(
            self.sess,
            os.path.join(self._params.model_dir, 'model_{}.ckpt'.format(i)),
            write_meta_graph=self._params.save_meta)  # if meta file is too big

        self.tr_writer.add_summary(in_tr_summary, i)
        self.tr_writer.flush()
        self.val_writer.add_summary(in_val_summary, i)
        self.val_writer.flush()

        auc_summary = tf.Summary()
        auc_summary.value.add(
            tag='AUROC_using_raw_likelihood', simple_value=auc)
        self.val_writer.add_summary(auc_summary, i)
        self.val_writer.flush()

  def finish(self):
    tf.logging.info('training is done')
    self.tr_writer.close()
    self.val_writer.close()
    self.saver.close()

  def restore_from_ckpt(self, ckpt_path):
    """restore model from a ckpt."""
    # meta_file = ckpt_path + '.meta'
    # saver = tf.train.import_meta_graph(meta_file)
    self.saver.restore(self.sess, ckpt_path)

  def pred_from_ckpt(self, test_dataset, num_samples):
    """make prediction from a ckpt."""
    test_dataset = test_dataset.batch(self._params.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()

    self.test_handle = self.sess.run(test_iterator.string_handle())

    loss_test = []
    loss_total_test = []
    acc_test = []
    y_test = []
    x_test = []
    for _ in range(num_samples // self._params.batch_size):
      out = self.sess.run(
          [self.loss_i, self.loss_total, self.acc_i, self.y0, self.y],
          feed_dict={
              self.handle: self.test_handle,
              self.dropout_rate: 0
          })
      loss_test.append(out[0])
      loss_total_test.append(out[1])
      acc_test.append(out[2])
      y_test.append(out[3])
      x_test.append(out[4])
    return loss_test, loss_total_test, acc_test, y_test, x_test


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  random.seed(FLAGS.random_seed)

  params = contrib_training.HParams(
      num_steps=FLAGS.num_steps,
      val_freq=FLAGS.val_freq,
      seq_len=FLAGS.seq_len,
      batch_size=FLAGS.batch_size,
      emb_variable=FLAGS.emb_variable,
      emb_size=FLAGS.emb_size,
      vocab_size=4,
      hidden_lstm_size=FLAGS.hidden_lstm_size,
      norm_lstm=FLAGS.norm_lstm,
      dropout_rate=FLAGS.dropout_rate,
      learning_rate=FLAGS.learning_rate,
      reg_type=FLAGS.reg_type,
      reg_weight=FLAGS.reg_weight,
      out_dir=FLAGS.out_dir,
      in_tr_data_dir=FLAGS.in_tr_data_dir,
      in_val_data_dir=FLAGS.in_val_data_dir,
      ood_val_data_dir=FLAGS.ood_val_data_dir,
      master=FLAGS.master,
      save_meta=FLAGS.save_meta,
      filter_label=FLAGS.filter_label,
      mutation_rate=FLAGS.mutation_rate,
  )

  # setup output directory
  create_out_dir(params)

  # load datasets
  params.add_hparam('in_tr_file_pattern', 'in_tr')
  params.add_hparam('in_val_file_pattern', 'in_val')
  params.add_hparam('ood_val_file_pattern', 'ood_val')
  (in_tr_dataset, in_val_dataset, ood_val_dataset) = load_datasets(params)

  # print parameter settings
  tf.logging.info(params)
  with tf.gfile.GFile(
      os.path.join(params.model_dir, 'params.json'), mode='w') as f:
    f.write(json.dumps(params.to_json(), sort_keys=True))

  # construct model
  model = SeqModel(params)
  model.reset()

  ## if previous model ckpt exists, restore the model from there
  tf.logging.info('model dir=%s', os.path.join(params.out_dir, '*.ckpt.index'))
  prev_steps, ckpt_file = utils.get_latest_ckpt(params.model_dir)
  if ckpt_file:
    tf.logging.info('previous ckpt exist, prev_steps=%s', prev_steps)
    model.restore_from_ckpt(ckpt_file)

  # training
  model.train(in_tr_dataset, in_val_dataset, ood_val_dataset, prev_steps)


if __name__ == '__main__':
  tf.app.run()
