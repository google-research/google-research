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
"""Keras models for the IMDB task.
"""
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Sequential


NUM_WORDS = 20000
SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 128
CNNLSTM_CELL_SIZE = 70


# no dropout
def cnn_lstm_nd(pfac,
                max_features=NUM_WORDS,
                maxlen=SEQUENCE_LENGTH,
                lstm_cell_size=CNNLSTM_CELL_SIZE,
                embedding_size=EMBEDDING_SIZE):
  """CNN-LSTM model, modified from Keras example."""
  # From github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py
  filters = 64
  kernel_size = 5
  pool_size = 4

  model = Sequential()
  model.add(pfac(Embedding(max_features, embedding_size, input_length=maxlen,
                           name='embedding')))
  model.add(pfac(Conv1D(filters,
                        kernel_size,
                        padding='valid',
                        activation='relu',
                        strides=1,
                        name='conv')))
  model.add(MaxPooling1D(pool_size=pool_size))
  model.add(pfac(LSTM(lstm_cell_size, name='lstm')))
  model.add(pfac(Dense(2, name='dense')))

  return model
