# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Script to compute and store the learned representation for Cifar10/100.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
import numpy as np
import tensorflow.compat.v1 as tf

from uncertainties.sources.cifar.cifar_input_python import load_data
from uncertainties.sources.cifar.network_cifar import NetworkCifar

# TO BE CHANGED, baseroute folder

PATH_MODEL = 'baseroute/models/cifar100/model'
DATA_PATH = 'baseroute/data/cifar100_data/cifar-100-python/'


def main(argv):
  """Main function."""

  del argv

  dataset = load_data(distorted=True, data_path=DATA_PATH, dataset='cifar100')
  (x_train_distorted, _), (x_test_distorted, _) = dataset
  dataset = load_data(distorted=False, data_path=DATA_PATH, dataset='cifar100')
  (x_train, _), (x_test, _) = dataset
  model = NetworkCifar()
  features = model.features(x_train)
  str_file = 'features_ll_train.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features)
  features = model.features(x_train_distorted)
  str_file = 'features_ll_train_distorted.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features)
  features = model.features(x_test)
  str_file = 'features_ll_test.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features)
  features = model.features(x_test_distorted)
  str_file = 'features_ll_test_distorted.npy'
  data_path = os.path.join(PATH_MODEL, str_file)
  with tf.gfile.Open(data_path, 'wb') as f:
    np.save(f, features)

if __name__ == '__main__':
  app.run(main)
