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

"""Script to train and compute features for Mnist."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from uncertainties.sources.mnist.mnist_input import load_data
from uncertainties.sources.mnist.network_mnist import NetworkMnist

# TO BE CHANGED, baseroute folder

DATA_PATH_MNIST = 'baseroute/data/mnist'
DATA_PATH_NOTMNIST = 'baseroute/data/notmnist'
PATH_MODEL = 'baseroute/models/mnist/model'

FLAGS = flags.FLAGS

flags.DEFINE_boolean('train_model', False, 'train the network or not')


def main(_):
  """Main function."""

  dataset = load_data()
  (x_train, _), (x_test, _) = dataset

  n = x_train.shape[0]
  batch_size = 128
  num_epochs = 3.0
  num_iters = int(num_epochs * n / batch_size)

  # define network and hyper-parameters
  hparams = {}
  network_params = {}
  network_params['dim_input'] = 784
  network_params['num_units'] = [512, 20]
  network_params['num_classes'] = 10
  hparams['num_iters'] = num_iters
  hparams['batch_size'] = batch_size

  # Load notmnist pictures test dataset, shape (10000, 784)
  data_path = os.path.join(DATA_PATH_NOTMNIST, 'pictures_test_notmnist.npy')
  with tf.gfile.Open(data_path, 'r') as f:
    x_notmnist = np.load(f)

  path_model = os.path.join(PATH_MODEL, 'full_network')
  model = NetworkMnist(
      dataset,
      network_params,
      hparams,
      path_model,
      train_model=FLAGS.train_model)
  if FLAGS.train_model:
    _, _ = model.train()

  # Features for x_test
  features = model.features(x_test)
  str_file = 'features_ll_test.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features)

  # Features for x_notmnist
  features_notmnist = model.features(x_notmnist)
  str_file = 'features_ll_test_notmnist.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features_notmnist)

  # Features for x_train
  features_train = model.features(x_train)
  str_file = 'features_ll_train.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features_train)


if __name__ == '__main__':
  app.run(main)
