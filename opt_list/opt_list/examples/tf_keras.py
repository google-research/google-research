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

"""Example using opt_list with TF + Keras."""

from absl import app

from opt_list import tf_opt_list

import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def main(_):
  # Construct a keras model
  model = Sequential()

  model.add(Dense(units=128, activation='relu', input_dim=2))
  model.add(Dense(units=128, activation='relu', input_dim=2))
  model.add(Dense(units=2, activation='linear'))

  # Define the total number of training steps
  training_iters = 200

  # Create the optimizer corresponding to the 0th hyperparameter configuration
  # with the specified amount of training steps.
  opt = tf_opt_list.keras_optimizer_for_idx(0, training_iters)

  model.compile(loss='mse', optimizer=opt, metrics=[])

  for _ in range(training_iters):
    # Construct a batch of random
    inp = tf.random.normal([512, 2]) / 4.
    target = tf.math.tanh(1 / (1e-6 + inp))

    # Train and evaluate the model
    model.train_on_batch(inp, target)
    model.evaluate(inp, target, steps=1)


if __name__ == '__main__':
  app.run(main)
