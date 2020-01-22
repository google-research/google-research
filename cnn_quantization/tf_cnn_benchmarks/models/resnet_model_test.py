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

"""Tests for resnet_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow.compat.v1 as tf

from cnn_quantization.tf_cnn_benchmarks.models import resnet_model


class ResNetModelTest(tf.test.TestCase):

  def testGetScaledBaseLearningRateOneGpuLrFromParams(self):
    """Verifies setting params.resnet_base_lr pipes through."""
    lr = self._get_scaled_base_learning_rate(1,
                                             'parameter_server',
                                             256,
                                             base_lr=.050)
    self.assertEquals(lr, .050)

  def testGetScaledBaseLearningRateOneGpu(self):
    lr = self._get_scaled_base_learning_rate(1, 'parameter_server', 128, 0.128)
    self.assertEquals(lr, .064)

  def testGetScaledBaseLearningRateEightGpuReplicated(self):
    lr = self._get_scaled_base_learning_rate(8, 'replicated', 256 * 8, 0.128)
    self.assertEquals(lr, .128)

  def testGetScaledBaseLearningRateTwoGpuParameter(self):
    lr = self._get_scaled_base_learning_rate(2, 'parameter_server', 256 * 2,
                                             0.128)
    self.assertEquals(lr, .256)

  def testGetScaledBaseLearningRateTwoGpuUneven(self):
    lr = self._get_scaled_base_learning_rate(2, 'replicated', 13, 0.128)
    self.assertEquals(lr, 0.0032500000000000003)

  def _get_scaled_base_learning_rate(self,
                                     num_gpus,
                                     variable_update,
                                     batch_size,
                                     base_lr=None):
    """Simplifies testing different learning rate calculations.

    Args:
      num_gpus: Number of GPUs to be used.
      variable_update: Type of variable update used.
      batch_size: Total batch size.
      base_lr: Base learning rate before scaling.

    Returns:
      Base learning rate that would be used to create lr schedule.
    """
    params = mock.Mock()
    params.num_gpus = num_gpus
    params.variable_update = variable_update
    if base_lr:
      params.resnet_base_lr = base_lr
    resnet50_model = resnet_model.ResnetModel('resnet50', 50, params=params)
    return resnet50_model.get_scaled_base_learning_rate(batch_size)


if __name__ == '__main__':
  tf.test.main()
