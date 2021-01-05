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

"""Tests for google_research.google_research.cold_posterior_bnn.core.model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from cold_posterior_bnn.core import model as bnnmodel
from cold_posterior_bnn.core import prior


class ModelTest(tf.test.TestCase):

  def test_clone(self):
    reg = prior.SpikeAndSlabRegularizer(weight=1.0)
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu',
                              kernel_regularizer=reg,
                              bias_regularizer=reg),
        tf.keras.layers.Dense(10)])

    ndata = 256
    data = tf.random.normal((ndata, 10))
    pred1 = model1(data)

    model2 = bnnmodel.clone_model_and_weights(model1, (None, 10))
    model1 = None
    pred2 = model2(data)

    self.assertAllClose(pred1, pred2, msg='Model cloning failed.')

  def test_clone_subclassed(self):
    class TestModel(tf.keras.Model):

      def __init__(self):
        super(TestModel, self).__init__()
        self.hidden = tf.keras.layers.Dense(10, activation='relu')
        self.out1 = tf.keras.layers.Dense(10, name='a')
        self.out2 = tf.keras.layers.Dense(10, name='nolabel')

      def call(self, inputs):
        x = self.hidden(inputs)
        return [self.out1(x), self.out2(x)]

      def get_config(self):
        return dict()

      @staticmethod
      def from_config(config):
        return TestModel()

    model1 = TestModel()
    input_shape = (None, 10)
    model1.build(input_shape)

    data = tf.random.normal((20, 10))
    model2 = bnnmodel.clone_model_and_weights(model1, input_shape)

    pred1 = model1(data)
    pred2 = model2(data)
    self.assertAllClose(pred1, pred2, msg='model2 output differs from model1')


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
