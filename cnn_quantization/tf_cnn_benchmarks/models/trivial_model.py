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

"""Trivial model configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks.models import model


class TrivialModel(model.CNNModel):
  """Trivial model configuration."""

  def __init__(self, params=None):
    super(TrivialModel, self).__init__(
        'trivial', 224 + 3, 32, 0.005, params=params)

  def add_inference(self, cnn):
    cnn.reshape([-1, 227 * 227 * 3])
    cnn.affine(1)
    cnn.affine(4096)


class TrivialCifar10Model(model.CNNModel):
  """Trivial cifar10 model configuration."""

  def __init__(self, params=None):
    super(TrivialCifar10Model, self).__init__(
        'trivial', 32, 32, 0.005, params=params)

  def add_inference(self, cnn):
    cnn.reshape([-1, 32 * 32 * 3])
    cnn.affine(1)
    cnn.affine(4096)


class TrivialSSD300Model(model.CNNModel):
  """Trivial SSD300 model configuration."""

  def __init__(self, params=None):
    super(TrivialSSD300Model, self).__init__(
        'trivial', 300, 32, 0.005, params=params)

  def add_inference(self, cnn):
    cnn.reshape([-1, 300 * 300 * 3])
    cnn.affine(1)
    cnn.affine(4096)

  def get_input_shapes(self, subset):
    return [[32, 300, 300, 3], [32, 8732, 4], [32, 8732, 1], [32]]

  def loss_function(self, inputs, build_network_result):
    images, _, _, labels = inputs
    labels = tf.cast(labels, tf.int32)
    return super(TrivialSSD300Model, self).loss_function(
        (images, labels), build_network_result)
