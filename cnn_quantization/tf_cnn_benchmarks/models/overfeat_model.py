# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Overfeat model configuration.

References:
  OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks
  Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus,
  Yann LeCun, 2014
  http://arxiv.org/abs/1312.6229
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn_quantization.tf_cnn_benchmarks.models import model


class OverfeatModel(model.CNNModel):
  """OverfeatModel."""

  def __init__(self, params=None):
    super(OverfeatModel, self).__init__(
        'overfeat', 231, 32, 0.005, params=params)

  def add_inference(self, cnn):
    # Note: VALID requires padding the images by 3 in width and height
    cnn.conv(96, 11, 11, 4, 4, mode='VALID')
    cnn.mpool(2, 2)
    cnn.conv(256, 5, 5, 1, 1, mode='VALID')
    cnn.mpool(2, 2)
    cnn.conv(512, 3, 3)
    cnn.conv(1024, 3, 3)
    cnn.conv(1024, 3, 3)
    cnn.mpool(2, 2)
    cnn.reshape([-1, 1024 * 6 * 6])
    cnn.affine(3072)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()
