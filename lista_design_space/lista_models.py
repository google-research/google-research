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

"""Model definition for LISTA."""
import numpy as np
import tensorflow as tf


def shrink(data, theta):
  theta = tf.nn.relu(theta)
  return tf.sign(data) * tf.nn.relu(tf.abs(data) - theta)


def shrink_free(data, theta):
  return tf.sign(data) * tf.nn.relu(tf.abs(data) - theta)


class ListaCell(tf.keras.layers.Layer):
  """Lista cell."""

  def __init__(self, layer_id, learnable_matrix_x, learnable_matrix_b, theta,
               input_indices, dict_shape, neuron='soft', name=None):
    super(ListaCell, self).__init__(name=name)
    self.layer_id = layer_id
    self.learnable_matrix_b = learnable_matrix_b
    self.theta = tf.Variable(theta, trainable=True, name=name + '/theta')
    self.input_indices = input_indices
    if layer_id != 0:
      self.learnable_matrix_x = learnable_matrix_x
      self.coefficient = tf.Variable(
          tf.ones((len(input_indices) + 1,)),
          trainable=True,
          name=name + '/coefficient')
    self.dict_shape = dict_shape
    self.neuron = neuron

  def call(self, inputs):
    side_connection = tf.matmul(inputs[:, :self.dict_shape[0]],
                                self.learnable_matrix_b)
    if self.layer_id == 0:
      output = side_connection
    else:
      new_inputs = [inputs[:, -self.dict_shape[1]:]]
      for idx in self.input_indices:
        new_inputs.append(inputs[:, self.dict_shape[0] +
                                 idx * self.dict_shape[1]:self.dict_shape[0] +
                                 (idx + 1) * self.dict_shape[1]])
      new_inputs = tf.stack(new_inputs, axis=-1) * self.coefficient
      new_inputs = tf.reduce_mean(new_inputs, axis=-1)
      output = tf.matmul(new_inputs, self.learnable_matrix_x) + side_connection

    if self.neuron == 'soft':
      output = shrink_free(output, self.theta)

    return tf.concat([inputs, output], 1)


class Lista(tf.keras.Sequential):
  """Lista model."""

  def __init__(self, dictionary, lam, arch_str, num_layers=16, name='Lista'):
    super(Lista, self).__init__(name=name)
    self.dictionary = dictionary.astype(np.float32)

    self.scale = 1.001 * np.linalg.norm(dictionary, ord=2) ** 2
    self.theta = (lam / self.scale).astype(np.float32)

    matrix_b = np.transpose(dictionary) / self.scale
    matrix_x = np.eye(dictionary.shape[1]) - np.matmul(matrix_b, dictionary)

    self.learnable_matrix_b = tf.Variable(
        np.transpose(matrix_b.astype(np.float32)), trainable=True, name='W_b')
    self.learnable_matrix_x = tf.Variable(
        np.transpose(matrix_x.astype(np.float32)), trainable=True, name='W_x')

    if not arch_str:
      arch = [0] * num_layers
    else:
      arch = [int(s) for s in arch_str.split('_')]
    for i in range(num_layers):
      self.create_cell(arch[i], i)

  def create_cell(self, arch, layer_id):
    indices = []
    i = 2
    while arch != 0:
      if arch % 2 != 0:
        indices.append(layer_id - i)
      arch //= 2
      i += 1
    cell = ListaCell(
        layer_id=layer_id,
        learnable_matrix_x=self.learnable_matrix_x,
        learnable_matrix_b=self.learnable_matrix_b,
        theta=self.theta,
        input_indices=indices,
        dict_shape=self.dictionary.shape,
        name='Lista_layer_{}'.format(layer_id + 1))
    self.add(cell)
