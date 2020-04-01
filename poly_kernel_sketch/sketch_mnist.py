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

# Compares various matrix compression methods on MNIST.
#
# Usage: python sketch_mnist.py
#
# Output goes into ${HOME}/nn_compression/mnist/experiments/020119
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.compat.v1.keras.models import Model

from absl import app

from common import *

# Load mnist dataset, global, used by functions below.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Run sketching experiment
def sketching_experiment(model, sketch_type, k2, k4, sketch_layer_2,
                         sketch_layer_4, x2_weights, x2_bias, x4_weights,
                         x4_bias):
  layers = model.layers
  x2_layer = layers[2]
  x4_layer = layers[4]
  if sketch_layer_2:
    x2_hat = sketch_matrix(x2_weights, sketch_type, k2)
    x2_layer.set_weights([x2_hat, x2_bias])
  if sketch_layer_4:
    x4_hat = sketch_matrix(x4_weights, sketch_type, k4)
    x4_layer.set_weights([x4_hat, x4_bias])
  [loss, acc] = model.evaluate(x_test, y_test)
  return acc


# Run SVD experiment
def svd_experiment(model, k2, k4, sketch_layer_2, sketch_layer_4, x2_weights,
                   x2_bias, x4_weights, x4_bias):
  layers = model.layers
  x2_layer = layers[2]
  x4_layer = layers[4]
  if sketch_layer_2:
    x2_hat = truncated_svd(x2_weights, k2)
    x2_layer.set_weights([x2_hat, x2_bias])
  if sketch_layer_4:
    x4_hat = truncated_svd(x4_weights, k4)
    x4_layer.set_weights([x4_hat, x4_bias])
  [loss, acc] = model.evaluate(x_test, y_test)
  return acc


# Build the model where the number of hidden units is hidden_size
def build_model(hidden_size):
  inputs = Input(shape=(28, 28))
  x1 = Flatten()(inputs)
  x2 = Dense(hidden_size, activation=tf.nn.relu)(x1)
  x3 = Dropout(0.2)(x2)
  x4 = Dense(10, activation=tf.nn.softmax)(x3)
  model = Model(inputs=inputs, outputs=x4)

  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  # Train and fit model
  model.fit(x_train, y_train, epochs=5)
  [loss, acc] = model.evaluate(x_test, y_test)
  return [model, acc]


def run_experiment(compression, hidden_size, k_values, sketch_layer_2,
                   sketch_layer_4):
  [model, acc] = build_model(hidden_size)
  layers = model.layers
  x2_layer = layers[2]
  x4_layer = layers[4]
  [x2_weights, x2_bias] = x2_layer.get_weights()
  [x4_weights, x4_bias] = x4_layer.get_weights()
  if compression == 'sketching':
    return run_sketching_experiment(k_values, model, sketch_layer_2,
                                    sketch_layer_4, x2_weights, x2_bias,
                                    x4_weights, x4_bias)
  elif compression == 'svd':
    return run_svd_experiment(k_values, model, sketch_layer_2, sketch_layer_4,
                              x2_weights, x2_bias, x4_weights, x4_bias)
  else:
    print('wrong compression type')


def run_sketching_experiment(k_values, model, sketch_layer_2, sketch_layer_4,
                             x2_weights, x2_bias, x4_weights, x4_bias):
  arora_acc_list = []
  our_acc_list = []
  for k in k_values:
    print(k)
    arora_acc = sketching_experiment(model, 'arora', k, k, sketch_layer_2,
                                     sketch_layer_4, x2_weights, x2_bias,
                                     x4_weights, x4_bias)
    arora_acc_list.append(arora_acc)
    our_acc = sketching_experiment(model, 'our_sketch', k, k, sketch_layer_2,
                                   sketch_layer_4, x2_weights, x2_bias,
                                   x4_weights, x4_bias)
    our_acc_list.append(our_acc)
  return [arora_acc_list, our_acc_list]


def run_svd_experiment(k_values, model, sketch_layer_2, sketch_layer_4,
                       x2_weights, x2_bias, x4_weights, x4_bias):
  acc_list = []
  for k in k_values:
    print(k)
    acc = svd_experiment(model, k, k, sketch_layer_2, sketch_layer_4,
                         x2_weights, x2_bias, x4_weights, x4_bias)
    acc_list.append(acc)
  return acc_list


def plot_and_save_sketching_experiment(folder, fn, k_values, arora_acc_list,
                                       our_acc_list, sketch_layer_2,
                                       sketch_layer_4, hidden_size, retrain):
  plt.clf()
  title = 'Matrix Sketching: MNIST Accuracy vs Sketch Size\n'
  if sketch_layer_2:
    title += 'Sketched 1st layer (dim=784x' + str(hidden_size) + ')\n'
  else:
    title += 'Kept 1st layer fixed (dim=784x' + str(hidden_size) + ')\n'
  if sketch_layer_4:
    title += 'Sketched 2nd layer (dim=' + str(hidden_size) + 'x10)\n'
  else:
    title += 'Kept 2nd layer fixed (dim=' + str(hidden_size) + 'x10)\n'
  title += 'Dark Blue=Arora, Cyan=Our Sketch\n'
  plt.title(title)
  plt.plot(k_values, arora_acc_list, 'b', k_values, our_acc_list, 'c')
  plt.tight_layout()
  save_fig(folder, fn)
  save_array(np.array(k_values), folder, fn + '_k', '%i')
  save_array(np.array(arora_acc_list), folder, fn + '_arora', '%.18e')
  save_array(np.array(our_acc_list), folder, fn + '_ours', '%.18e')


def plot_and_save_svd_experiment(folder, fn, k_values, acc_list, sketch_layer_2,
                                 sketch_layer_4, hidden_size, retrain):
  plt.clf()
  title = 'Truncated SVD: MNIST Accuracy vs Rank k\n'
  if sketch_layer_2:
    title += 'Truncated 1st layer (dim=784x' + str(hidden_size) + ')\n'
  else:
    title += 'Kept 1st layer fixed (dim=784x' + str(hidden_size) + ')\n'
  if sketch_layer_4:
    title += 'Truncated 2nd layer (dim=' + str(hidden_size) + 'x10)\n'
  else:
    title += 'Kept 2nd layer fixed (dim=' + str(hidden_size) + 'x10)\n'
  plt.title(title)
  plt.plot(k_values, acc_list)
  plt.tight_layout()
  save_fig(folder, fn)
  save_array(np.array(k_values), folder, fn + '_k', '%i')
  save_array(np.array(acc_list), folder, fn + '_acc', '%.18e')



def plot_and_save_svd_plus_sketch_experiment(
    folder, fn, param_list, arora_acc_list, our_acc_list, svd_acc_list,
    sketch_layer_2, sketch_layer_4, hidden_size, retrain):
  plt.clf()
  title = 'MNIST Accuracy vs Number of Sketching Parameters\n'
  if sketch_layer_2:
    title += 'Sketched 1st layer (dim=784x' + str(hidden_size) + ')\n'
  else:
    title += 'Kept 1st layer fixed (dim=784x' + str(hidden_size) + ')\n'
  if sketch_layer_4:
    title += 'Sketched 2nd layer (dim=' + str(hidden_size) + 'x10)\n'
  else:
    title += 'Kept 2nd layer fixed (dim=' + str(hidden_size) + 'x10)\n'
  title += 'Dark Blue=Arora, Cyan=Our Sketch, Magenta=SVD Truncation\n'
  plt.title(title)
  plt.plot(param_list, arora_acc_list, 'b', param_list, our_acc_list, 'c',
           param_list, svd_acc_list, 'm')
  plt.tight_layout()
  save_fig(folder, fn)
  save_array(np.array(param_list), folder, fn + '_k', '%i')
  save_array(np.array(arora_acc_list), folder, fn + '_arora', '%.18e')
  save_array(np.array(our_acc_list), folder, fn + '_ours', '%.18e')
  save_array(np.array(svd_acc_list), folder, fn + '_svd', '%.18e')


def save_sketching_experiment(folder, filename, k_values, arora_list, our_list):
  save_fig(folder, fn)
  save_array(np.array(k_values), folder, fn + '_k', '%i')
  save_array(np.array(arora_list), folder, fn + '_arora', '%.18e')
  save_array(np.array(our_list), folder, fn + '_ours', '%.18e')


def save_svd_experiment(folder, filename, k_values, acc_list):
  save_fig(folder, fn)
  save_array(np.array(k_values), folder, fn + '_k', '%i')
  save_array(np.array(acc_list), folder, fn + '_acc', '%.18e')


def main(argv):
  FOLDER = make_experiment_dir('nn_compression/mnist/experiments/020119')
  BASELINE_SIZE = 512

  hidden_size = BASELINE_SIZE
  retrain = False
  sketch_layer_2 = False
  sketch_layer_4 = True
  k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  NCOLS = 512
  NROWS = 10
  num_params = count_parameters_list(k_values, NROWS, NCOLS)

  svd_acc = run_experiment('svd', hidden_size, k_values,
                                        sketch_layer_2, sketch_layer_4)
  plot_and_save_svd_experiment(FOLDER, 'svd_expt', k_values, svd_acc, sketch_layer_2,
                                 sketch_layer_4, hidden_size, retrain)

  [arora_acc, our_acc] = run_experiment('sketching', hidden_size, num_params,
                                        sketch_layer_2, sketch_layer_4)
  plot_and_save_sketching_experiment(FOLDER, 'sketching_expt', num_params, arora_acc,
                                     our_acc, sketch_layer_2, sketch_layer_4,
                                     hidden_size, retrain)

  plot_and_save_svd_plus_sketch_experiment(FOLDER, 'combo_expt', num_params,
                                           arora_acc, our_acc, svd_acc,
                                           sketch_layer_2, sketch_layer_4,
                                           hidden_size, retrain)

if __name__ == '__main__':
  app.run(main)
