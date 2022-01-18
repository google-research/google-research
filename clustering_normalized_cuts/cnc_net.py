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

"""Contains run function for CNC."""
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import OneHotEncoder
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
from tensorflow.compat.v1.keras.layers import Input

from clustering_normalized_cuts import networks
from clustering_normalized_cuts.util import print_accuracy


def run_net(data, params):
  """run the network with the parameters."""
  #
  # UNPACK DATA
  #

  x_train, y_train, x_val, y_val, x_test, y_test = data['cnc']['train_and_test']
  x_train_unlabeled, _, x_train_labeled, y_train_labeled = data['cnc'][
      'train_unlabeled_and_labeled']
  x_val_unlabeled, _, _, _ = data['cnc']['val_unlabeled_and_labeled']

  if 'siamese' in params['affinity']:
    pairs_train, dist_train, pairs_val, dist_val = data['siamese'][
        'train_and_test']

  x = np.concatenate((x_train, x_val, x_test), axis=0)
  y = np.concatenate((y_train, y_val, y_test), axis=0)

  if x_train_labeled:
    y_train_labeled_onehot = OneHotEncoder().fit_transform(
        y_train_labeled.reshape(-1, 1)).toarray()
  else:
    y_train_labeled_onehot = np.empty((0, len(np.unique(y))))

  #
  # SET UP INPUTS
  #

  # create true y placeholder (not used in unsupervised training)
  y_true = tf.placeholder(
      tf.float32, shape=(None, params['n_clusters']), name='y_true')

  batch_sizes = {
      'Unlabeled': params['batch_size'],
      'Labeled': params['batch_size']
  }

  input_shape = x.shape[1:]

  # inputs to CNC
  inputs = {
      'Unlabeled': Input(shape=input_shape, name='UnlabeledInput'),
      'Labeled': Input(shape=input_shape, name='LabeledInput'),
  }

  #
  # DEFINE AND TRAIN SIAMESE NET
  # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.

  # DEFINE AND TRAIN Siamese NET
  if params['affinity'] == 'siamese':
    siamese_net = networks.SiameseNet(inputs, params['siam_arch'],
                                      params.get('siam_reg'),
                                      params['main_path'], y_true)

    siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                      params['siam_lr'], params['siam_drop'],
                      params['siam_patience'], params['siam_ne'],
                      params['siam_batch_size'], params['dset'])

  else:
    siamese_net = None

  #
  # DEFINE AND TRAIN CNC NET
  #
  cnc_net = networks.CncNet(inputs, params['cnc_arch'], params.get('cnc_reg'),
                            y_true, y_train_labeled_onehot,
                            params['n_clusters'], params['affinity'],
                            params['scale_nbr'], params['n_nbrs'], batch_sizes,
                            params['result_path'], params['dset'], siamese_net,
                            x_train, params['cnc_lr'], params['cnc_tau'],
                            params['bal_reg'])

  cnc_net.train(x_train_unlabeled, x_train_labeled, x_val_unlabeled,
                params['cnc_drop'], params['cnc_patience'], params['min_tem'],
                params['cnc_epochs'])

  #
  # EVALUATE
  #

  x_cncnet = cnc_net.predict(x)
  prediction = np.argmax(x_cncnet, 1)
  accuray_all = print_accuracy(prediction, y, params['n_clusters'])
  nmi_score_all = nmi(prediction, y)
  print('NMI: {0}'.format(np.round(nmi_score_all, 3)))

  if params['generalization_metrics']:
    x_cncnet_train = cnc_net.predict(x_train_unlabeled)
    x_cncnet_test = cnc_net.predict(x_test)

    prediction_train = np.argmax(x_cncnet_train, 1)
    accuray_train = print_accuracy(prediction_train, y_train,
                                   params['n_clusters'])
    nmi_score_train = nmi(prediction_train, y_train)
    print('TRAIN NMI: {0}'.format(np.round(nmi_score_train, 3)))

    prediction_test = np.argmax(x_cncnet_test, 1)
    accuray_test = print_accuracy(prediction_test, y_test, params['n_clusters'])
    nmi_score_test = nmi(prediction_test, y_test)
    print('TEST NMI: {0}'.format(np.round(nmi_score_test, 3)))
    with gfile.Open(params['result_path'] + 'results', 'w') as f:
      f.write(accuray_all + ' ' + accuray_train + ' ' + accuray_test + '\n')
      f.write(
          str(np.round(nmi_score_all, 3)) + ' ' +
          str(np.round(nmi_score_train, 3)) + ' ' +
          str(np.round(nmi_score_test, 3)) + '\n')

  else:
    with gfile.Open(params['result_path'] + 'results', 'w') as f:
      f.write(accuray_all + ' ' + str(np.round(nmi_score_all, 3)) + '\n')
