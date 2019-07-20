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

"""Eval for classifier.py on in-dist., skewed, and completely OOD."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import re

from absl import flags
from absl import logging
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import models
import tensorflow_datasets as tfds
import yaml

from uq_benchmark_2019 import calibration_lib
from uq_benchmark_2019.news import classifier
from uq_benchmark_2019.news import data_utils_from_hendrycks as data_utilsh

# parameters
FLAGS = flags.FLAGS

flags.DEFINE_enum('method', 'vanilla',
                  ['vanilla', 'll-svi', 'll-dropout', 'dropout'],
                  'Which method?')
flags.DEFINE_bool('is_tempscale', False, 'If True, do temperature scaling')
flags.DEFINE_string('model_dir', '20news_ckpt_and_predictions',
                    'Directory to all model ckpts')
flags.DEFINE_bool('is_ensemble', False,
                  'If True, ensemble predictions by taking mean')
flags.DEFINE_integer(
    'n_pred_sample', 5,
    'Number of times for sampling predictions using models involving svi and dropout'
)


def compute_auc(neg, pos, pos_label=1):
  ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
  neg = np.array(neg)[np.logical_not(np.isnan(neg))]
  pos = np.array(pos)[np.logical_not(np.isnan(pos))]
  scores = np.concatenate((neg, pos), axis=0)
  auc = metrics.roc_auc_score(ys, scores)
  if pos_label == 1:
    return auc
  else:
    return 1 - auc


def load_ood_dataset(n_ood, fix_len, vocab, vocab_size):
  """Load LM1B dataset for OOD test."""
  ## import ood dataset
  data = tfds.load('lm1b')
  _, test_lm1b_tfds = data['train'], data['test']
  i = 0
  test_lm1b_list = []
  for example in tfds.as_numpy(test_lm1b_tfds):
    x = example['text'].decode('utf-8')  # for PY3
    x = re.sub(r'\W+', ' ', x).strip()  # remove "," "."
    test_lm1b_list.append(x)
    i += 1
    if i % n_ood == 0:
      break

  test_lm1b_x = data_utilsh.text_to_rank(
      test_lm1b_list, vocab, desired_vocab_size=vocab_size)
  # pad text to achieve the same length
  test_lm1b_x_pad = data_utilsh.pad_sequences(test_lm1b_x, maxlen=fix_len)
  test_lm1b_y = -1 * np.ones(len(test_lm1b_x))

  return test_lm1b_x_pad, test_lm1b_y


def main(_):

  if FLAGS.is_tempscale:
    tf.enable_v2_behavior()

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
      'n_class_in': None,
      'n_train': None,
  }

  # load in-dist. and skewed in-dist. datasets
  data = classifier.load_np_dataset(params['data_pkl_file'])

  # load OOD dataset
  n_ood = 5600
  test_lm1b_x_pad, _ = load_ood_dataset(n_ood, params['fix_len'], data.vocab,
                                        params['vocab_size'])

  # list of ckpt dir
  model_dir = os.path.join(FLAGS.model_dir, FLAGS.method)
  ckpt_dirs = tf.io.gfile.listdir(model_dir)

  # how many replicates for ensemble
  if FLAGS.is_ensemble:
    assert len(ckpt_dirs) > 1
    n_ensemble = len(ckpt_dirs)
    if n_ensemble == 0:
      logging.fatal('no model ckpt')
  else:
    n_ensemble = 1

  pred = {}  # dict for final prediction score
  # dict for saving pred from different models
  pred_accum = {'in': [], 'skew': [], 'ood': []}

  for i in range(n_ensemble):

    ckpt_dir = os.path.join(model_dir, ckpt_dirs[i], 'model')
    if not tf.io.gfile.isdir(ckpt_dir):
      continue
    print('ckpt_dir={}'.format(ckpt_dir))

    # load params
    with tf.gfile.GFile(os.path.join(ckpt_dir, 'params.json'), mode='rb') as f:
      params_json = yaml.safe_load(f)
      params.update(params_json)
      params['master'] = ''
    print('params after load={}'.format(params))

    tf.reset_default_graph()
    # create model
    model = classifier.rnn_model(
        params,
        training_dr_lstm=params['dropout_rate_lstm'] != 0.0,
        training_dr_ll=params['dropout_rate'] != 0.0)

    # load model
    model.load_weights(ckpt_dir + '/model.ckpt')

    # predict
    if FLAGS.method in ['ll-svi', 'dropout', 'll-dropout']:
      # need to run multiple times and get mean prediction
      assert FLAGS.n_pred_sample > 1
    else:
      FLAGS.n_pred_sample = 1

    pred_k = {'in': [], 'skew': [], 'ood': []}
    for _ in range(FLAGS.n_pred_sample):
      pred_tr_in = model.predict(data.in_sample_examples)
      acc_tr_in = np.mean(
          data.in_sample_labels == np.argmax(pred_tr_in, axis=1))

      pred_test_in = model.predict(data.test_in_sample_examples)
      acc_test_in = np.mean(
          data.test_in_sample_labels == np.argmax(pred_test_in, axis=1))
      print('in-dist. acc_tr={}, acc_test={}'.format(acc_tr_in, acc_test_in))

      pred_test_skew = model.predict(data.test_oos_examples)
      pred_test_ood = model.predict(test_lm1b_x_pad)

      if FLAGS.is_tempscale:
        # temperature scaling
        # logits for temp scaling
        last_layer_model = models.Model(
            inputs=model.input, outputs=model.get_layer('last_layer').output)
        logits = last_layer_model.predict(data.dev_in_sample_examples)
        opt_temp = calibration_lib.find_scaling_temperature(
            data.dev_in_sample_labels, logits, temp_range=(1e-5, 1e5))
        pred_test_in = calibration_lib.apply_temperature_scaling(
            opt_temp, pred_test_in)
        pred_test_skew = calibration_lib.apply_temperature_scaling(
            opt_temp, pred_test_skew)
        pred_test_ood = calibration_lib.apply_temperature_scaling(
            opt_temp, pred_test_ood)

      # save in a list
      pred_k['in'].append(pred_test_in)
      pred_k['skew'].append(pred_test_skew)
      pred_k['ood'].append(pred_test_ood)

    pred_k_in_mean = np.mean(np.stack(pred_k['in']), axis=0)
    pred_k_skew_mean = np.mean(np.stack(pred_k['skew']), axis=0)
    pred_k_ood_mean = np.mean(np.stack(pred_k['ood']), axis=0)

    pred_accum['in'].append(pred_k_in_mean)
    pred_accum['skew'].append(pred_k_skew_mean)
    pred_accum['ood'].append(pred_k_ood_mean)

  # if ensemble, then take the mean
  pred['in'] = np.mean(np.stack(pred_accum['in']), axis=0)
  pred['skew'] = np.mean(np.stack(pred_accum['skew']), axis=0)
  pred['ood'] = np.mean(np.stack(pred_accum['ood']), axis=0)

  # prediction accuracy for in-dist.
  pred['in_true_labels'] = data.test_in_sample_labels
  acc = np.mean(data.test_in_sample_labels == np.argmax(pred['in'], axis=1))
  print('== (optionally ensemble) acc={} =='.format(acc))

  print('== eval in and skew using max(Py|x) ==')
  neg = list(np.max(pred['in'], axis=1))
  pos = list(np.max(pred['skew'], axis=1))
  print('auc={}'.format(compute_auc(neg, pos, pos_label=0)))

  print('== eval in and ood using max(Py|x) ==')
  neg = list(np.max(pred['in'], axis=1))
  pos = list(np.max(pred['ood'], axis=1))
  print('auc={}'.format(compute_auc(neg, pos, pos_label=0)))

  # save the predictions
  pred_file_name = 'pred_nensemb{}_npred{}_tempscale{}.pkl'.format(
      len(pred_accum['in']), FLAGS.n_pred_sample, FLAGS.is_tempscale)
  with tf.gfile.Open(os.path.join(model_dir, pred_file_name), 'wb') as f:
    pickle.dump(pred, f, protocol=2)


if __name__ == '__main__':
  tf.app.run()
