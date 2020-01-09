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

"""Configuration file for mnist_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow.compat.v1 as tf


def l2_loss(var, l2_loss_wt):
  """Computes l2 regularization loss."""
  return tf.multiply(float(l2_loss_wt), tf.nn.l2_loss(var))


def get_l2_loss(l2_loss_wt):
  """Returns a l2 regularization function scaled by weight."""
  return functools.partial(l2_loss, l2_loss_wt=l2_loss_wt)


class ConfigDict(object):
  """MNIST configration."""

  def __init__(self):
    # Optimization parameters.
    self.init_lr = 0.1
    self.l2_loss_wt = 5e-4
    self.xent_loss_wt = 0.0
    self.margin_loss_wt = 1.0
    self.gamma = 10000
    self.alpha = 4
    self.top_k = 1
    self.dist_norm = np.inf
    self.num_classes = 10

    # List of tuples specify (kernel_size, number of filters) for each layer.
    self.filter_sizes_conv_layers = [(5, 32), (5, 64)]
    # Dictionary of pooling type ("max"/"average", size and stride).
    self.pool_params = {"type": "max", "size": 2, "stride": 2}
    self.num_units_fc_layers = [512]
    self.dropout_rate = 0
    self.batch_norm = True
    self.activation = tf.nn.relu
    self.l2_loss_wt = 5e-4
    self.regularizer = get_l2_loss(self.l2_loss_wt)

  def get_l2_loss_wt(self):
    return self._l2_loss_wt

  def set_l2_loss_wt(self, value):
    self._l2_loss_wt = value
    self.regularizer = get_l2_loss(self._l2_loss_wt)
    return self._l2_loss_wt

  l2_loss_wt = property(get_l2_loss_wt, set_l2_loss_wt)



