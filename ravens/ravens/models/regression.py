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

"""Regression module."""

import numpy as np

from ravens.models import ConvMLP
from ravens.models import DeepConvMLP
from ravens.models import mdn_utils

import tensorflow as tf


class Regression:
  """Regression module."""

  def __init__(self, input_shape, preprocess, use_mdn):
    self.preprocess = preprocess

    resnet = False

    if resnet:
      self.model = DeepConvMLP(input_shape, d_action=6, use_mdn=use_mdn)
    else:
      self.model = ConvMLP(d_action=6, use_mdn=use_mdn)

    self.optim = tf.keras.optimizers.Adam(lr=2e-4)
    self.metric = tf.keras.metrics.Mean(name='regression_loss')
    self.val_metric = tf.keras.metrics.Mean(name='regression_loss_validate')

    if not use_mdn:
      self.loss_criterion = tf.keras.losses.MeanSquaredError()
    else:
      self.loss_criterion = mdn_utils.mdn_loss

  def set_batch_size(self, batch_size):
    self.model.set_batch_size(batch_size)

  def forward(self, in_img):
    """Forward pass.

    Args:
      in_img: [B, H, W, C]

    Returns:
      output tensor.
    """
    input_data = self.preprocess(in_img)
    in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    output = self.model(in_tensor)
    return output

  def train_pick(self, batch_obs, batch_act, train_step, validate=False):
    """Train pick."""
    self.metric.reset_states()
    self.val_metric.reset_states()

    input_data = self.preprocess(batch_obs)
    in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    loss = train_step(self.model, self.optim, in_tensor, batch_act,
                      self.loss_criterion)

    if not validate:
      self.metric(loss)
    else:
      self.val_metric(loss)
    return np.float32(loss)

  def save(self, fname):
    pass

  def load(self, fname):
    pass
