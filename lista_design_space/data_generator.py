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

"""Data generator to synthesize data for LISTA."""
import os

from absl import app
import numpy as np
import tensorflow as tf

_DATA_PATH = './data'
_NUM_TEST_IMAGES = 10240


def main(_):
  if not tf.io.gfile.exists(_DATA_PATH):
    tf.io.gfile.mkdir(_DATA_PATH)

  np.random.seed(0)

  # Pretrained checkpoints only work with the dictionary used for training.
  if not tf.io.gfile.exists(os.path.join(_DATA_PATH, 'A_250_500.npy')):
    dictionary = np.random.normal(
        scale=1.0 / np.sqrt(250), size=(250, 500)).astype(np.float32)
    colnorms = np.sqrt(np.sum(np.square(dictionary), axis=0, keepdims=True))
    dictionary = dictionary / colnorms
    np.save(
        tf.io.gfile.GFile(os.path.join(_DATA_PATH, 'A_250_500.npy'), 'w'),
        dictionary)
  else:
    dictionary = np.load(
        os.path.join(_DATA_PATH, 'A_250_500.npy'), allow_pickle=True)

  supp = np.random.uniform(size=[_NUM_TEST_IMAGES, 500])
  supp = np.array(supp <= 0.1, np.float32)
  mag = np.random.normal(size=[_NUM_TEST_IMAGES, 500])
  x = mag * supp
  y = np.matmul(x, dictionary.transpose())
  data = np.concatenate((y, x), 1)
  np.save(
      tf.io.gfile.GFile(os.path.join(_DATA_PATH, 'sc_test_data.npy'), 'w'),
      data.astype(np.float32))


if __name__ == '__main__':
  app.run(main)
