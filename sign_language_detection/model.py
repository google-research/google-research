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

"""Sign language sequence tagging keras model."""

import tensorflow as tf

from sign_language_detection.args import FLAGS


def input_size():
  """Calculate the size of the input pose by desired components."""
  points = 0
  if 'pose_keypoints_2d' in FLAGS.input_components:
    points += 25
  if 'face_keypoints_2d' in FLAGS.input_components:
    points += 70
  if 'hand_left_keypoints_2d' in FLAGS.input_components:
    points += 21
  if 'hand_right_keypoints_2d' in FLAGS.input_components:
    points += 21
  return points


def get_model():
  """Create keras sequential model following the hyperparameters."""

  model = tf.keras.Sequential(name='tgt')

  # model.add(SequenceMasking())  # Mask padded sequences
  model.add(tf.keras.layers.Dropout(
      FLAGS.input_dropout))  # Random feature dropout

  # Add LSTM
  for _ in range(FLAGS.encoder_layers):
    rnn = tf.keras.layers.LSTM(FLAGS.hidden_size, return_sequences=True)
    if FLAGS.encoder_bidirectional:
      rnn = tf.keras.layers.Bidirectional(rnn)
    model.add(rnn)

  # Project and normalize to labels space
  model.add(tf.keras.layers.Dense(2, activation='softmax'))

  return model


def build_model():
  """Apply input shape, loss, optimizer, and metric to the model."""
  model = get_model()
  model.build(input_shape=(None, None, input_size()))
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
      metrics=['accuracy'],
  )
  model.summary()

  return model
