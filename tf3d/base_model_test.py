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

"""Tests for tf3d.base_model."""
import os

import tensorflow as tf
from tf3d import base_model


class BaseModelTest(tf.test.TestCase):

  def test_base_model(self):
    class TestModel(base_model.BaseModel):

      def call(self, inputs, training=True):
        """A dummy call function that simply returns the inputs as outputs and calculates loss."""
        if training:
          self.calculate_losses(
              inputs=inputs['labels'], outputs=inputs['inputs'])
        return inputs

    log_dir = '/tmp/tf3d/base_model_test'
    if tf.io.gfile.exists(log_dir):
      tf.io.gfile.rmtree(log_dir)

    inputs = tf.ones([10, 10])
    labels = tf.ones([10, 10]) * 5.
    inputs_dict = {'inputs': inputs, 'labels': labels}

    loss_names_to_functions = {
        'loss_1':
            lambda inputs, outputs: tf.reduce_mean(tf.abs(inputs - outputs)),
        'loss_2':
            lambda inputs, outputs: tf.reduce_mean(tf.abs(inputs + outputs)),
    }
    loss_names_to_weights = {
        'loss_1': 1.,
        'loss_2': 2.,
    }
    model = TestModel(
        loss_names_to_functions=loss_names_to_functions,
        loss_names_to_weights=loss_names_to_weights,
        summary_log_freq=1,
        train_dir=log_dir)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.01, 1, 0.1)))
    # indirectly calls model.train_step function
    model.train_on_batch(x=inputs_dict)
    # run forward pass which calls 'calculate_losses' method.
    model(inputs_dict)
    total_loss = model.loss_names_to_losses['total_loss']
    model.close_writer()

    self.assertAllEqual(total_loss, 16.)
    self.assertNotEmpty(
        (tf.io.gfile.glob(os.path.join(log_dir, 'events*'))))


if __name__ == '__main__':
  tf.test.main()
