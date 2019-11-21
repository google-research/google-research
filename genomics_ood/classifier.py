# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

r"""Build a classifier for DNA sequences using ConvNets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random
from absl import flags
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf  # tf
import yaml
from genomics_ood import utils

# parameters
FLAGS = flags.FLAGS

flags.DEFINE_boolean('embedding', True, 'if use word embedding')
flags.DEFINE_integer('batch_size', 100, 'The number of images in each batch.')
flags.DEFINE_integer('num_steps', 1000000, 'The number of trainig steps')
flags.DEFINE_integer('val_freq', 1000, 'How often to eval validation (# steps)')
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate')
flags.DEFINE_integer('emb_size', 4, 'The word embedding dimensions')
flags.DEFINE_integer('hidden_lstm_size', 2000,
                     'The number of hidden units in generator BiLSTM.')
flags.DEFINE_integer('hidden_dense_size', 200,
                     'The number of hidden units in encoder BiLSTM.')
flags.DEFINE_integer('num_motifs', 1000, 'The number of motifs used in conv1d.')
flags.DEFINE_integer('len_motifs', 20, 'The len of motifs used in conv1d.')
flags.DEFINE_float('temperature', 1.0, 'Temperature scaleing parameter.')
flags.DEFINE_integer('random_seed', 1234, 'The random seed')
flags.DEFINE_boolean('reweight_sample', False,
                     'If normalize loss function by class sample size')
flags.DEFINE_boolean('save_meta', False, 'Save meta graph file for each ckpt.')
flags.DEFINE_float('dropout_rate', 0.1, 'The learning rate')
flags.DEFINE_float('l2_reg', 0.0,
                   'The regularization parameter for lstm and dense layers.')
flags.DEFINE_float('mutation_rate', 0.0, 'Mutation rate for data augmentation.')
flags.DEFINE_float('epsilon', 0,
                   'epsilon for odin method, the size of permutation.')

# sequence directory, length, kmer length
flags.DEFINE_string('in_tr_data_dir', '/tmp/data/before_2011_in_tr',
                    'data directory of in-distribution training')
flags.DEFINE_string('in_val_data_dir', '/tmp/data/between_2011-2016_in_val',
                    'data directory of in-distribution validation')
flags.DEFINE_string('ood_val_data_dir', '/tmp/data/between_2011-2016_ood_val',
                    'data directory of OOD validation')
flags.DEFINE_string('label_dict_file', ('/tmp/data/label_dict.json'),
                    'json file saving the encoded label')
flags.DEFINE_string('out_dir', '/tmp/out_classifier',
                    'Directory where to write log and models.')
flags.DEFINE_integer('seq_len', 250, 'sequence length')
flags.DEFINE_string('master', '', 'TensorFlow master to use.')

FLAGS = flags.FLAGS


def create_out_dir(params):
  """Setup the output directory."""
  params.in_tr_data_dir = utils.clean_last_slash_if_any(params.in_tr_data_dir)
  params.in_val_data_dir = utils.clean_last_slash_if_any(params.in_val_data_dir)
  params.ood_val_data_dir = utils.clean_last_slash_if_any(
      params.ood_val_data_dir)

  sub_dir = ('%s_emb%s_l%d_bs%d_lr%.4f_nm%d_lm%d_hd%d'
             '_t%.2f_l2r%.6f_rw%s_mr%.2f') % (
                 os.path.basename(params.in_tr_data_dir), params.embedding,
                 params.seq_len, params.batch_size, params.learning_rate,
                 params.num_motifs, params.len_motifs, params.hidden_dense_size,
                 params.temperature, params.l2_reg, params.reweight_sample,
                 params.mutation_rate)
  log_dir = os.path.join(params.out_dir, sub_dir, 'log')
  params.add_hparam('log_dir_in_tr', os.path.join(log_dir, 'tr'))
  params.add_hparam('log_dir_in_val', os.path.join(log_dir, 'val'))
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


def load_datasets_and_labels(params):
  """load class labels, in_tr_data, in_val_data, ood_val_data (called test)."""
  in_tr_file_list = [
      os.path.join(params.in_tr_data_dir, x)
      for x in tf.gfile.ListDirectory(params.in_tr_data_dir)
      if params.in_tr_file_pattern in x
  ]
  tf.logging.info('in_tr_file_list=%s', in_tr_file_list)

  in_tr_label_file = [
      x for x in in_tr_file_list if 'nsample' in x and '.json' in x
  ][0]
  tf.logging.info('nsample_dict_file=%s', in_tr_label_file)
  with tf.gfile.GFile(os.path.join(in_tr_label_file), 'rb') as f_label_code:
    # label_sample_size
    # keys: class names (strings), values: sample size of classes (ints)
    label_sample_size = yaml.safe_load(f_label_code)
    tf.logging.info('# of label_dict=%s', len(label_sample_size))

  # load in-distribution training sequence, add mutations to the input seqs
  in_tr_data_file_list = [x for x in in_tr_file_list if '.tfrecord' in x]
  tf.logging.info('in_tr_data_file_list=%s', in_tr_data_file_list)

  def parse_single_tfexample_addmutations_short(unused_key, v):
    return utils.parse_single_tfexample_addmutations(unused_key, v,
                                                     params.mutation_rate,
                                                     params.seq_len)

  # for training, we optionally mutate input sequences to overcome over-fitting.
  if params.mutation_rate == 0:
    in_tr_dataset = tf.data.TFRecordDataset(
        in_tr_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))
  else:
    in_tr_dataset = tf.data.TFRecordDataset(in_tr_data_file_list).map(
        lambda v: parse_single_tfexample_addmutations_short(v, v))

  # in-distribution validation
  in_val_data_file_list = [
      os.path.join(params.in_val_data_dir, x)
      for x in tf.gfile.ListDirectory(params.in_val_data_dir)
      if params.in_val_file_pattern in x and '.tfrecord' in x
  ]
  tf.logging.info('in_val_data_file_list=%s', in_val_data_file_list)
  in_val_dataset = tf.data.TFRecordDataset(
      in_val_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))

  # OOD validation
  ood_val_data_file_list = [
      os.path.join(params.ood_val_data_dir, x)
      for x in tf.gfile.ListDirectory(params.ood_val_data_dir)
      if params.ood_val_file_pattern in x and '.tfrecord' in x
  ]
  tf.logging.info('ood_val_data_file_list=%s', ood_val_data_file_list)
  ood_val_dataset = tf.data.TFRecordDataset(
      ood_val_data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))

  return label_sample_size, in_tr_dataset, in_val_dataset, ood_val_dataset


class SeqPredModel(object):
  """Model for classifying genomic sequences."""

  def __init__(self, params):
    """create the model."""
    self._params = params

    self._make_dataset()
    self._one_hot_encode_y()

    if self._params.embedding:
      self._embedding_encode_x()
    else:
      self._one_hot_encode_x()

    self._make_cnn_model()

    self._make_losses()
    self._make_summary_stats()
    self._permute_z_for_y_tilde()  # this is only for ODIN

  def _make_dataset(self):
    """Make data generators."""
    self.handle = tf.placeholder(tf.string, shape=[])
    self.iterator = tf.data.Iterator.from_string_handle(self.handle, {
        'x': tf.int32,
        'y': tf.int32
    }, {
        'x': [None, None],
        'y': [None]
    })
    features = self.iterator.get_next()
    self.x, self.y = features['x'], features['y']

  def _one_hot_encode_y(self):
    self.y_emb = tf.one_hot(self.y, depth=self._params.n_class)

  def _one_hot_encode_x(self):
    """Make embedding layer."""
    self.x_emb = tf.one_hot(self.x, depth=self._params.vocab_size)
    tf.logging.info('shape of x_emb=%s', self.x_emb.shape)

  def _embedding_encode_x(self):
    """Make variables."""
    # emb_size must equal to vocab_size,
    # otherwise exceed vocab will be encoded as zeros
    self.emb_dict = tf.Variable(
        tf.one_hot(
            list(range(self._params.vocab_size)), depth=self._params.emb_size))
    # if x = -1 (not in the emb_dict), embedding = all zeros
    # [TODO], how to add -1=[0,0,0,0] into emb_dict
    # https://github.com/tensorflow/tensorflow/issues/13642 not solved
    self.x_emb = tf.nn.embedding_lookup(self.emb_dict, self.x, name='embx')

  def _make_cnn_model(self):
    """Make cnn model."""
    out = tf.layers.conv1d(
        inputs=self.x_emb,
        filters=self._params.num_motifs,
        kernel_size=self._params.len_motifs,
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    out = tf.reduce_max(out, axis=[1])
    out = tf.nn.dropout(out, keep_prob=1 - self._params.dropout_rate)
    out = tf.layers.dense(
        out,
        units=self._params.hidden_dense_size,
        activation=tf.nn.relu,
        name='dense',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    out = tf.nn.dropout(out, keep_prob=1 - self._params.dropout_rate)
    self.out = out

  def _make_losses(self):
    """Make losses."""
    self.logits_dense_fn = tf.keras.layers.Dense(
        self._params.n_class,
        activation=None,
        name='logits',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    self.logits = self.logits_dense_fn(self.out)
    tf.logging.info('shape of logits=%s', self.logits.shape)
    tf.logging.info('self.logits.shape=%s', self.logits.shape)
    self.probs = tf.nn.softmax(self.logits /
                               tf.to_float(self._params.temperature))
    self.y_pred = tf.argmax(self.probs, axis=1)
    tf.logging.info('self.probs.shape=%s', self.probs.shape)

    if self._params.reweight_sample:
      # get weights for each data point in the batch based on their labels
      sample_weights = tf.gather_nd(self._params.label_weights,
                                    tf.expand_dims(self.y, 1))
      # normalize the weights such that the sum is batch_size.
      self.sample_weights = sample_weights / tf.reduce_sum(
          sample_weights) * self._params.batch_size
      self.loss_i = tf.losses.softmax_cross_entropy(
          onehot_labels=self.y_emb,
          logits=self.logits / tf.to_float(self._params.temperature),
          weights=self.sample_weights)
    else:
      self.sample_weights = 1.0
      self.loss_i = tf.losses.softmax_cross_entropy(
          onehot_labels=self.y_emb,
          logits=self.logits / tf.to_float(self._params.temperature))

    self.loss = tf.reduce_mean(self.loss_i)

    self.acc_i = tf.equal(self.y_pred, tf.cast(self.y, dtype=tf.int64))
    self.acc = tf.reduce_mean(
        tf.multiply(self.sample_weights, tf.cast(self.acc_i, dtype=tf.float32)))

    # l2 norm
    self.variables = tf.trainable_variables()
    self.loss_l2 = tf.add_n(
        [tf.nn.l2_loss(v) for v in self.variables if 'bias' not in v.name])
    # total loss
    self.loss_total = self.loss + self._params.l2_reg * self.loss_l2

    # Optimization using constant learning rate
    opt = tf.train.AdamOptimizer(self._params.learning_rate)
    self.lr = tf.convert_to_tensor(self._params.learning_rate)
    self.minimize = opt.minimize(
        self.loss_total, global_step=tf.train.get_or_create_global_step())

  def _make_summary_stats(self):
    """Make summary stats."""
    self.summary = tf.summary.merge([
        tf.summary.scalar('loss', self.loss),
        tf.summary.scalar('acc', self.acc),
        tf.summary.scalar('loss_total', self.loss_total),
        tf.summary.scalar('loss_l2', self.loss_l2)
    ])

  def reset(self):
    """Setup session and writers for statistics."""
    self.saver = tf.train.Saver(max_to_keep=500)
    self.sess = tf.Session(self._params.master)
    self.sess.run(tf.global_variables_initializer())

    self.tr_writer = tf.summary.FileWriter(self._params.log_dir_in_tr,
                                           self.sess.graph)
    self.val_writer = tf.summary.FileWriter(self._params.log_dir_in_val,
                                            self.sess.graph)

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
      _, in_tr_loss, in_tr_acc, in_tr_summary = self.sess.run(
          [self.minimize, self.loss, self.acc, self.summary],
          feed_dict={self.handle: self.in_tr_handle})

      if i % self._params.val_freq == 0:
        # self.sess.run(val_iterator.initializer)
        (in_val_loss, in_val_acc, in_val_summary, in_val_probs,
         in_val_y) = self.sess.run(
             [self.loss, self.acc, self.summary, self.probs, self.y],
             feed_dict={self.handle: self.in_val_handle})
        ood_val_probs = self.sess.run(
            self.probs, feed_dict={self.handle: self.ood_val_handle})

        # auc using max(p(y|x)), smaller for OOD
        neg = np.max(in_val_probs, axis=1)
        pos = np.max(ood_val_probs, axis=1)
        auc = roc_auc_score([1] * neg.shape[0] + [0] * pos.shape[0],
                            np.concatenate((neg, pos), axis=0))

        tf.logging.info(('i=%d, in_tr_loss=%.4f, in_val_loss=%.4f, ',
                         'in_tr_acc=%.4f, in_val_acc=%.4f \n auc=%.4f'), i,
                        in_tr_loss, in_val_loss, in_tr_acc, in_val_acc, auc)
        tf.logging.info('probs_val[0]=%s', in_val_probs[0, int(in_val_y[0])])

        if self._params.save_meta:
          _ = self.saver.save(
              self.sess,
              os.path.join(self._params.model_dir, 'model_{}.ckpt'.format(i)))
        else:
          _ = self.saver.save(
              self.sess,
              os.path.join(self._params.model_dir, 'model_{}.ckpt'.format(i)),
              write_meta_graph=False)  # if meta file is too big
        self.tr_writer.add_summary(in_tr_summary, i)
        self.tr_writer.flush()
        self.val_writer.add_summary(in_val_summary, i)
        self.val_writer.flush()

        auc_summary = tf.Summary()
        auc_summary.value.add(tag='AUROC_using_max(p(y|x))', simple_value=auc)
        self.val_writer.add_summary(auc_summary, i)
        self.val_writer.flush()

  def restore_from_ckpt(self, ckpt_path):
    """Restore from a ckpt."""
    # meta_file = ckpt_path + '.meta'
    # saver = tf.train.import_meta_graph(meta_file)
    self.saver.restore(self.sess, ckpt_path)

  def pred_from_ckpt(self, test_dataset, num_samples):
    """Make prediction from ckpt."""
    test_dataset = test_dataset.batch(self._params.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()

    self.test_handle = self.sess.run(test_iterator.string_handle())
    # self.sess.run(test_iterator.initializer)
    loss_test = []
    acc_test = []
    probs_list = []
    y_test = []

    for _ in range(num_samples // self._params.batch_size):
      out = self.sess.run([self.loss, self.acc, self.probs, self.y],
                          feed_dict={self.handle: self.test_handle})

      loss_test.append(out[0])
      acc_test.append(out[1])
      probs_list.append(out[2])
      y_test.append(out[3])

    probs_test = np.stack(probs_list).reshape(-1, self._params.n_class)
    return loss_test, acc_test, probs_test, y_test

  def pred_from_ckpt_ood(self, test_dataset, num_samples):
    """Make prediction from ckpt."""
    test_dataset = test_dataset.batch(self._params.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()

    self.test_handle = self.sess.run(test_iterator.string_handle())
    # self.sess.run(test_iterator.initializer)
    probs_list = []
    y_test = []

    for _ in range(num_samples // self._params.batch_size):
      out = self.sess.run([self.probs, self.y],
                          feed_dict={self.handle: self.test_handle})

      probs_list.append(out[0])
      y_test.append(out[1])

    probs_test = np.stack(probs_list).reshape(-1, self._params.n_class)
    return probs_test, y_test

  def pred_from_ckpt_ood_odin(self, test_dataset, num_samples):
    """Make prediction from ckpt."""
    test_dataset = test_dataset.batch(self._params.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()

    self.test_handle = self.sess.run(test_iterator.string_handle())
    # self.sess.run(test_iterator.initializer)
    probs_list = []
    probs_tilde_list = []
    y_test = []

    for _ in range(num_samples // self._params.batch_size):
      out = self.sess.run([self.probs, self.probs_tilde, self.y],
                          feed_dict={self.handle: self.test_handle})

      probs_list.append(out[0])
      probs_tilde_list.append(out[1])
      y_test.append(out[2])

    probs_test = np.stack(probs_list).reshape(-1, self._params.n_class)
    probs_tilde_test = np.stack(probs_tilde_list).reshape(
        -1, self._params.n_class)
    return probs_test, probs_tilde_test, y_test

  def _permute_z_for_y_tilde(self):
    """Permute z using ODIN: z_tilde = z - e * tf.sign(-1 * grad(predy/z)).

    This is a method modified based on Liang, Shiyu, Yixuan Li, & R. Srikant.
    Enhancing the reliability of out-of-distribution image detection
    in neural networks (2017). See Eq. (2) in the paper.
    The original method was proposed for adding permutation to input x,
    where x is a continuous variable. Since our input x is discrete variable,
    we add permuations to the input of the last layer of the neural networks.
    The function works for any intermediate variable, but we chosoe the input
    to the last layer, self.out.

    Thus, we take gradient from max(self.logits) to self.out, and then
    self.out_tilde = self.out - self._params.epsilon * tf.sign(-1 * self.grads)
    self.out is of the size [batch_size, hidden_dense_size]
    self.logits is of the size [batch_size, n_class].
    We first create ids_set for paris (m, id), where m=1,2,...,batch_size, and
    id = argmax(self.logits[m]) .
    Then we take gradient from self.logits[m] wrt the self.out[m].
    The resulting grads is of the size [hidden_dense_size].
    """

    def create_ids(y):
      """Create triplets (id_ymax, m, b), m in num_samps, b in batch_size."""
      ids_max = np.argmax(y, axis=1)
      # ids_set: m in batch_size, ids_max[m]
      ids_set = np.array([
          (m, ids_max[m]) for m in range(self._params.batch_size)
      ])
      return ids_set

    def grads_from_y_to_z(x):
      """Take gradient from each yp[id_ymax, m, b] to z[m, b]."""
      # self.probs [batch_size, n_class]
      # self.out [batch_size, hidden_dense_size]
      grad = tf.gradients(self.probs[x[0], x[1]], self.out)[0][x[0], :]
      return grad

    ids_set = tf.py_func(create_ids, [self.logits], tf.int64)
    grads_flat = tf.map_fn(grads_from_y_to_z, ids_set, dtype=tf.float32)
    self.grads = tf.reshape(
        grads_flat, [self._params.batch_size, self._params.hidden_dense_size])
    tf.logging.info('grads_flat.shape=%s', grads_flat.shape)
    tf.logging.info('grads.shape=%s', self.grads.shape)
    # odin permutation
    self.out_tilde = self.out - self._params.epsilon * tf.sign(-1 * self.grads)

    self.logits_tilde = self.logits_dense_fn(self.out_tilde)
    tf.logging.info('logits_tilde.shape=%s', self.logits_tilde.shape)
    tf.logging.info('logits.shape=%s', self.logits.shape)
    self.probs_tilde = tf.nn.softmax(self.logits_tilde /
                                     tf.to_float(self._params.temperature))
    tf.logging.info('probs_tilde.shape=%s', self.probs_tilde.shape)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  random.seed(FLAGS.random_seed)

  params = tf.contrib.training.HParams(
      embedding=FLAGS.embedding,
      num_steps=FLAGS.num_steps,
      val_freq=FLAGS.val_freq,
      seq_len=FLAGS.seq_len,
      batch_size=FLAGS.batch_size,
      emb_size=FLAGS.emb_size,
      vocab_size=4,
      hidden_lstm_size=FLAGS.hidden_lstm_size,
      hidden_dense_size=FLAGS.hidden_dense_size,
      dropout_rate=FLAGS.dropout_rate,
      learning_rate=FLAGS.learning_rate,
      num_motifs=FLAGS.num_motifs,
      len_motifs=FLAGS.len_motifs,
      temperature=FLAGS.temperature,
      reweight_sample=FLAGS.reweight_sample,
      l2_reg=FLAGS.l2_reg,
      out_dir=FLAGS.out_dir,
      in_tr_data_dir=FLAGS.in_tr_data_dir,
      in_val_data_dir=FLAGS.in_val_data_dir,
      ood_val_data_dir=FLAGS.ood_val_data_dir,
      master=FLAGS.master,
      save_meta=FLAGS.save_meta,
      label_dict_file=FLAGS.label_dict_file,
      mutation_rate=FLAGS.mutation_rate,
      epsilon=FLAGS.epsilon,
  )

  # create output directories
  create_out_dir(params)

  # load datasets and labels for training
  params.add_hparam('in_tr_file_pattern', 'in_tr')
  params.add_hparam('in_val_file_pattern', 'in_val')
  params.add_hparam('ood_val_file_pattern', 'ood_val')
  label_sample_size, in_tr_dataset, in_val_dataset, ood_val_dataset = load_datasets_and_labels(
      params)
  params.add_hparam('n_class', len(label_sample_size))
  tf.logging.info('label_sample_size=%s', label_sample_size)

  # compute weights for labels
  # load the dictionary for class labels.
  # Key: class name (string), values: encoded class label (int)
  with tf.gfile.GFile(os.path.join(params.label_dict_file),
                      'rb') as f_label_code:
    # label_dict_after_2016_new_species0 = json.load(f)
    params.add_hparam('label_dict', yaml.safe_load(f_label_code))
    tf.logging.info('# of label_dict=%s', len(params.label_dict))

  label_weights = utils.compute_label_weights_using_sample_size(
      params.label_dict, label_sample_size)
  params.add_hparam('label_weights', label_weights)

  # print parameter settings
  tf.logging.info(params)
  with tf.gfile.GFile(
      os.path.join(params.model_dir, 'params.json'), mode='w') as f:
    f.write(json.dumps(params.to_json(), sort_keys=True))

  # construct model
  tf.logging.info('create model')
  model = SeqPredModel(params)
  model.reset()

  ## if previous model ckpt exists, restore the model from there
  tf.logging.info('model dir=%s', os.path.join(params.model_dir,
                                               '*.ckpt.index'))
  prev_steps, ckpt_file = utils.get_latest_ckpt(params.model_dir)
  if ckpt_file:
    tf.logging.info('previous ckpt exist, prev_steps=%s', prev_steps)
    model.restore_from_ckpt(ckpt_file)

  # training
  tf.logging.info('strart training')
  model.train(in_tr_dataset, in_val_dataset, ood_val_dataset, prev_steps)


if __name__ == '__main__':
  tf.app.run()
