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

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training Loop for Linear log reg. oracle."""
import pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import CategoryEncoding
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

rng = np.random.default_rng(67489)

data_dir = (pathlib.Path(__file__).parent /
            '../Dataset_Preprocessing/Dataset/').resolve()

list_of_cols = [
    'label', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11',
    'N12', 'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C17',
    'C18', 'C19', 'C20', 'C22', 'C23', 'C25'
]

offsets = [
    0, 22, 53, 81, 98, 138, 170, 197, 219, 245, 252, 266, 287, 309, 1768, 2350,
    2654, 2676, 3308, 3310, 8992, 12185, 12194, 17845, 20016, 20018, 20034,
    20048
]

NUM_TOTAL_FEATURES = 20151 + len(offsets)


def my_model():
  internal_model = Sequential()
  internal_model.add(
      CategoryEncoding(
          num_tokens=NUM_TOTAL_FEATURES, output_mode='multi_hot', sparse=True))
  internal_model.add(Dense(1, input_dim=NUM_TOTAL_FEATURES))

  return internal_model


epochs = 20

for split in range(5):

  random_seed = rng.integers(low=1000000, size=1)[0]
  numpy_seed = rng.integers(low=1000000, size=1)[0]
  tf_seed = rng.integers(low=1000000, size=1)[0]

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  tf.random.set_seed(tf_seed)

  train_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/train_Split_' + str(split) +
      '-processed-allints_selected_cols_C15_C14_bucket_index_C7_offsets.csv')

  test_file_to_read = (
      str(data_dir) + '/Split_' + str(split) + '/test_Split_' + str(split) +
      '-processed-allints_selected_cols_C15_C14_bucket_index_C7_offsets.csv')

  result_file = (
      str(data_dir) + '/Split_' + str(split) + '/result_lin_' + str(split))

  df_train = pd.read_csv(train_file_to_read, usecols=list_of_cols)

  df_test = pd.read_csv(test_file_to_read, usecols=list_of_cols)

  X_train = df_train.drop(['label'], axis=1)
  y_train = df_train['label']

  X_test = df_test.drop(['label'], axis=1)
  y_test = df_test['label']

  len_test_set = len(df_test)

  csv_logger = tf.keras.callbacks.CSVLogger(result_file)

  model = my_model()
  model.compile(
      optimizer='adam',
      metrics=[tf.keras.metrics.AUC(name='auc', from_logits=True)],
      loss=tf.keras.losses.BinaryCrossentropy(
          from_logits=True, name='binary_crossentropy'))

  model.fit(
      X_train,
      y_train,
      epochs=epochs,
      batch_size=1024,
      validation_data=(X_test, y_test),
      validation_batch_size=len_test_set,
      callbacks=[csv_logger])
