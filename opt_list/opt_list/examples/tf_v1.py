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

# Lint as: python3
"""Example using opt_list with TF1.0."""

from absl import app

from opt_list import tf_opt_list

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def main(_):
  # Construct tensors representing a batch of data. For this example we use
  # random data.
  inp = tf.random.normal([512, 2]) / 4.
  target = tf.math.tanh(1 / (1e-6 + inp))

  # Define the neural network computation
  net = tf.layers.dense(inp, 1024, activation="relu")
  net = tf.layers.dense(net, 1024, activation="relu")
  net = tf.layers.dense(net, 2, activation="linear")

  # Create some loss.
  loss = tf.reduce_mean(tf.square(net - target))

  # Define the total number of training steps
  training_iters = 200

  # The TF V1 optimizers make use of the global step. Create it.
  global_step = tf.train.get_or_create_global_step()

  # Create the optimizer corresponding to the 0th hyperparameter configuration
  # with the specified amount of training steps.
  # global_step is used here to track training progress (e.g. for schedules).
  opt = tf_opt_list.optimizer_for_idx(0, training_iters, iteration=global_step)

  # construct the op that updates the model's parameters.
  train_op = opt.minimize(loss)

  # Train the network!
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(training_iters):
      _, l = sess.run([train_op, loss])
      if i % 10 == 0:
        print(i, l)


if __name__ == "__main__":
  app.run(main)
