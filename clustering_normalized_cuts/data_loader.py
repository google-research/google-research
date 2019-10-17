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

"""Contains all data generating code for datasets used in the script."""

from __future__ import division

import os
import tempfile
import time

import numpy as np
from tensorflow import gfile
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import model_from_json

from clustering_normalized_cuts import pairs

_IGNORE_SSL_ERROR = True  # Don't verify SSL certs.


def get_data(params):
  """preprocesses all data.

  Args:
    params: all parameters.

  Returns:
    A nested dictionary nested dict with the following keys:

  the permutations (if any) used to shuffle the training and validation sets
  'p_train'                           - p_train
  'p_val'                             - p_val

  the data used for CNC
  'cnc'
      'train_and_test'                - (x_train, y_train, x_val, y_val,
      x_test, y_test)
      'train_unlabeled_and_labeled'   - (x_train_unlabeled, y_train_unlabeled,
      x_train_labeled, y_train_labeled)
      'val_unlabeled_and_labeled'     - (x_val_unlabeled, y_val_unlabeled,
      x_val_labeled, y_val_labeled)

  the data used for siamese net, if the architecture uses the siamese net
  'siamese'
      'train_and_test'                - (pairs_train, dist_train, pairs_val,
      dist_val)
      'train_unlabeled_and_labeled'   - (pairs_train_unlabeled,
      dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
      'val_unlabeled_and_labeled'     - (pairs_val_unlabeled,
      dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)
  """
  ret = {}

  # get data
  x_train, x_test, y_train, y_test = load_data(params)

  ret['cnc'] = {}
  if params.get('use_all_data'):
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    x_test = np.zeros((0,) + x_train.shape[1:])
    y_test = np.zeros((0,))

  # split x training, validation, and test subsets
  if 'val_set_fraction' not in params:
    train_val_split = (.9, .1)
  elif params['val_set_fraction'] > 0 and params['val_set_fraction'] <= 1:
    train_val_split = (1 - params['val_set_fraction'],
                       params['val_set_fraction'])
  else:
    raise ValueError('val_set_fraction is invalid! must be in range (0, 1]')

  # shuffle training and test data separately into themselves and concatenate
  p = np.concatenate([
      np.random.permutation(len(x_train)),
      len(x_train) + np.random.permutation(len(x_test))
  ],
                     axis=0)

  x_train, y_train, p_train, x_val, y_val, p_val = split_data(
      x_train, y_train, train_val_split, permute=p[:len(x_train)])

  # split training and validation subset into its supervised and unsupervised
  if params.get('train_labeled_fraction'):
    train_split = (1 - params['train_labeled_fraction'],
                   params['train_labeled_fraction'])
  else:
    train_split = (1, 0)
  x_train_unlabeled, y_train_unlabeled, _, x_train_labeled, y_train_labeled, _ = split_data(
      x_train, y_train, train_split)

  if params.get('val_labeled_fraction'):
    val_split = (1 - params['val_labeled_fraction'],
                 params['val_labeled_fraction'])
  else:
    val_split = (1, 0)
  x_val_unlabeled, y_val_unlabeled, _, x_val_labeled, y_val_labeled, _ = split_data(
      x_val, y_val, val_split)

  # embed data in code space, if necessary
  all_data = [
      x_train, x_val, x_test, x_train_unlabeled, x_train_labeled,
      x_val_unlabeled, x_val_labeled
  ]
  if params.get('use_code_space'):
    for i, d in enumerate(all_data):
      all_data[i] = embed_data(d, dset=params['dset'], path=params['main_path'])
  else:
    # otherwise just flatten it
    for i, d in enumerate(all_data):
      all_data[i] = all_data[i].reshape((-1, np.prod(all_data[i].shape[1:])))
  (x_train, x_val, x_test, x_train_unlabeled, x_train_labeled, x_val_unlabeled,
   x_val_labeled) = all_data

  # collect everything into a dictionary
  ret['cnc']['train_and_test'] = (x_train, y_train, x_val, y_val, x_test,
                                  y_test)
  ret['cnc']['train_unlabeled_and_labeled'] = (x_train_unlabeled,
                                               y_train_unlabeled,
                                               x_train_labeled, y_train_labeled)
  ret['cnc']['val_unlabeled_and_labeled'] = (x_val_unlabeled, y_val_unlabeled,
                                             x_val_labeled, y_val_labeled)

  ret['p_train'] = p_train
  ret['p_val'] = p_val

  # get siamese data if necessary
  if 'siamese' in params['affinity']:
    ret['siamese'] = {}
    pairs_train_unlabeled, dist_train_unlabeled = pairs.create_pairs_from_unlabeled_data(
        x1=x_train_unlabeled,
        p=None,
        k=params.get('siam_k'),
        tot_pairs=params.get('siamese_tot_pairs'),
        pre_shuffled=True,
    )

    pairs_val_unlabeled, dist_val_unlabeled = pairs.create_pairs_from_unlabeled_data(
        x1=x_val_unlabeled,
        p=None,
        k=params.get('siam_k'),
        tot_pairs=params.get('siamese_tot_pairs'),
        pre_shuffled=True,
    )

    # get pairs for labeled data
    class_indices = [
        np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])
    ]

    pairs_train_labeled, dist_train_labeled = pairs.create_pairs_from_labeled_data(
        x_train_labeled, class_indices)
    class_indices = [
        np.where(y_val_labeled == i)[0] for i in range(params['n_clusters'])
    ]

    pairs_val_labeled, dist_val_labeled = pairs.create_pairs_from_labeled_data(
        x_val_labeled, class_indices)

    ret['siamese']['train_unlabeled_and_labeled'] = (pairs_train_unlabeled,
                                                     dist_train_unlabeled,
                                                     pairs_train_labeled,
                                                     dist_train_labeled)
    ret['siamese']['val_unlabeled_and_labeled'] = (pairs_val_unlabeled,
                                                   dist_val_unlabeled,
                                                   pairs_val_labeled,
                                                   dist_val_labeled)

    # combine labeled and unlabeled pairs for training the siamese
    print('pairs_train_unlabeled shape', pairs_train_unlabeled.shape)
    print('pairs_train_labeled shape', pairs_train_labeled.shape)
    pairs_train = np.concatenate((pairs_train_unlabeled, pairs_train_labeled),
                                 axis=0)
    dist_train = np.concatenate((dist_train_unlabeled, dist_train_labeled),
                                axis=0)
    pairs_val = np.concatenate((pairs_val_unlabeled, pairs_val_labeled), axis=0)
    dist_val = np.concatenate((dist_val_unlabeled, dist_val_labeled), axis=0)

    ret['siamese']['train_and_test'] = (pairs_train, dist_train, pairs_val,
                                        dist_val)

  return ret


def load_data(params):
  """reads the data specified in params."""
  if params['dset'] == 'mnist':
    x_train, x_test, y_train, y_test = get_mnist()
  else:
    raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

  return x_train, x_test, y_train, y_test


def embed_data(x, dset, path):
  """embeds x into the code space using the autoencoder."""

  if x:
    return np.zeros(shape=(0, 10))
  # load model and weights
  json_path = os.path.join(path, 'ae_{}.json'.format(dset))
  print('load model from json file:', json_path)
  with gfile.Open(json_path) as f:
    pt_ae = model_from_json(f.read())
  weights_path = os.path.join(path, 'ae_{}_weights.h5'.format(dset))
  print('load code spase from:', weights_path)
  local_filename = weights_path.split('/')[-1]
  tmp_filename = os.path.join(tempfile.gettempdir(),
                              str(int(time.time())) + '_' + local_filename)
  gfile.Copy(weights_path, tmp_filename)
  pt_ae.load_weights(tmp_filename)
  gfile.Remove(tmp_filename)

  print('***********************', x.shape)
  x = x.reshape(-1, np.prod(x.shape[1:]))
  print('***********************', x.shape)

  get_embeddings = K.function([pt_ae.input], [pt_ae.layers[3].output])

  get_reconstruction = K.function([pt_ae.layers[4].input], [pt_ae.output])
  x_embedded = predict_with_k_fn(get_embeddings, x)[0]
  x_recon = predict_with_k_fn(get_reconstruction, x_embedded)[0]
  reconstruction_mse = np.mean(np.square(x - x_recon))
  print(
      'using pretrained embeddings; sanity check, total reconstruction error:',
      np.mean(reconstruction_mse))

  del pt_ae

  return x_embedded


def predict_with_k_fn(k_fn, x, bs=1000):
  """evaluates x by k_fn(x), where k_fn is a Keras function, by batches of size 1000."""
  if not isinstance(x, list):
    x = [x]
  num_outs = len(k_fn.outputs)
  shape_y = k_fn.outputs[0].get_shape().as_list()
  shape_y[0] = len(x[0])
  y = [np.empty(shape_y) for _ in k_fn.outputs]

  for i in range(int(x[0].shape[0] / bs + 1)):
    x_batch = []
    for x_ in x:
      x_batch.append(x_[i * bs:(i + 1) * bs])
    temp = k_fn(x_batch)
    for j in range(num_outs):
      y[j][i * bs:(i + 1) * bs] = temp[j]

  return y


def split_data(x, y, split, permute=None):
  """Splits arrays x and y.

  Args:
    x: matrix of shape n x d1
    y: matrix of shape n x d2
    split: a list of floats of length 2 (e.g. [a1, a2]) where a, b > 0, a, b <
      1, and a + b == 1
    permute: a list or array of length n that can be used to shuffle x and y
      identically before splitting it

  Returns:
    Splitted arrays of x and y
  """
  n = len(x)
  if permute is not None:
    if not isinstance(permute, np.ndarray):
      raise ValueError(
          'Provided permute array should be an np.ndarray, not {}!'.format(
              type(permute)))
    if len(permute.shape) != 1:
      raise ValueError(
          'Provided permute array should be of dimension 1, not {}'.format(
              len(permute.shape)))
    if len(permute) != n:
      raise ValueError(
          'Provided permute should be the same length as x! (len(permute) = {}, n = {}'
          .format(len(permute), n))
  else:
    permute = np.arange(n)

  if np.sum(split) != 1:
    raise ValueError('Split elements must sum to 1!')

  ret_x_y_p = []
  prev_idx = 0
  for s in split:
    idx = prev_idx + np.round(s * n).astype(np.int)
    p_ = permute[prev_idx:idx]
    x_ = x[p_]
    y_ = y[p_]
    prev_idx = idx
    ret_x_y_p.append(x_)
    ret_x_y_p.append(y_)
    ret_x_y_p.append(p_)

  return ret_x_y_p[0], ret_x_y_p[1], ret_x_y_p[2], ret_x_y_p[3], ret_x_y_p[
      4], ret_x_y_p[5]


def get_mnist():
  """Returns the train and test splits of the MNIST digits dataset.

  x_train and x_test are shaped into the tensorflow image data
  shape and normalized to fit in the range [0, 1]
  """
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # reshape and standardize x arrays
  x_train = np.expand_dims(x_train, -1) / 255.
  x_test = np.expand_dims(x_test, -1) / 255.
  return x_train, x_test, y_train, y_test
