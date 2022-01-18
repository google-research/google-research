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

"""Vgg model configuration.

Includes multiple models: vgg11, vgg16, vgg19, corresponding to
  model A, D, and E in Table 1 of [1].

References:
[1]  Simonyan, Karen, Andrew Zisserman
     Very Deep Convolutional Networks for Large-Scale Image Recognition
     arXiv:1409.1556 (2014)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from cnn_quantization.tf_cnn_benchmarks.models import model


def _construct_vgg(cnn, num_conv_layers):
  """Build vgg architecture from blocks."""
  assert len(num_conv_layers) == 5
  for _ in xrange(num_conv_layers[0]):
    cnn.conv(64, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[1]):
    cnn.conv(128, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[2]):
    cnn.conv(256, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[3]):
    cnn.conv(512, 3, 3)
  cnn.mpool(2, 2)
  for _ in xrange(num_conv_layers[4]):
    cnn.conv(512, 3, 3)
  cnn.mpool(2, 2)
  cnn.reshape([-1, 512 * 7 * 7])
  cnn.affine(4096)
  cnn.dropout()
  cnn.affine(4096)
  cnn.dropout()


class Vgg11Model(model.CNNModel):

  def __init__(self, params=None):
    super(Vgg11Model, self).__init__('vgg11', 224, 64, 0.005, params=params)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [1, 1, 2, 2, 2])


class Vgg16Model(model.CNNModel):

  def __init__(self, params=None):
    super(Vgg16Model, self).__init__('vgg16', 224, 64, 0.005, params=params)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [2, 2, 3, 3, 3])


class Vgg19Model(model.CNNModel):

  def __init__(self, params=None):
    super(Vgg19Model, self).__init__('vgg19', 224, 64, 0.005, params=params)

  def add_inference(self, cnn):
    _construct_vgg(cnn, [2, 2, 4, 4, 4])
