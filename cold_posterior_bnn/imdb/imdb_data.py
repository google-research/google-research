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

# Lint as: python3
"""Load the IMDB data set using Keras dataset.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

tf.compat.v1.enable_v2_behavior()

MAX_FEATURES = 20000
MAX_LEN = 100


def load_data(num_words=MAX_FEATURES, maxlen=MAX_LEN, training_take=0):
  """Load IMDB sentiment classification data set.

  Args:
    num_words: int, >=1, maximum number of words.
    maxlen: int, >=1, sequence length for padding.
    training_take: int, if 0 the full training data set is used.  If >0, only
      the first training_take samples are used as training set.
      Must be <= 20000.

  Returns:
    x_train, y_train: training data set.
    x_test, y_test: testing data set.

  Raises:
    ValueError: invalid arguments.
  """
  if training_take > 20000:
    raise ValueError('training_take must be 0 or in the range [1,20000].')

  imdb_dest = '/tmp/imdb.npz'

  (x_train, y_train), (x_test, y_test) = imdb.load_data(path=imdb_dest,
                                                        num_words=num_words)
  # Split of training set (20k training, 5k validation)
  num_train = 20000
  x_train, x_val = x_train[:num_train], x_train[num_train:]
  y_train, y_val = y_train[:num_train], y_train[num_train:]

  if training_take > 0:
    x_train = x_train[:training_take]
    y_train = y_train[:training_take]

  def convert_dataset(x, y, maxlen):
    x = sequence.pad_sequences(x, maxlen=maxlen)
    y = np.array(y)
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    return x, y

  x_train, y_train = convert_dataset(x_train, y_train, maxlen=maxlen)
  x_val, y_val = convert_dataset(x_val, y_val, maxlen=maxlen)
  x_test, y_test = convert_dataset(x_test, y_test, maxlen=maxlen)

  return (x_train, y_train), (x_val, y_val), (x_test, y_test)

