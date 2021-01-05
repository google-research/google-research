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

"""Tests for model construction and training."""

import tensorflow as tf

import slot_attention.model as model_utils
import slot_attention.utils as utils


class ModelTests(tf.test.TestCase):
  """Test model construction and training."""

  def test_object_discovery_model(self):
    """Test object discovery model."""

    learning_rate = 0.001
    resolution = (128, 128)
    batch_size = 2
    num_slots = 3
    num_iterations = 2

    optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)
    model = model_utils.build_model(
        resolution, batch_size, num_slots, num_iterations,
        model_type="object_discovery")

    input_shape = (batch_size, resolution[0], resolution[1], 3)
    random_input = tf.random.uniform(input_shape)

    with tf.GradientTape() as tape:
      preds = model(random_input, training=True)
      recon_combined, _, _, _ = preds
      loss_value = utils.l2_loss(random_input, recon_combined)

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    assert True  # If we make it to this line, we're all good!

  def test_set_prediction_model(self):
    """Test set prediction model."""

    learning_rate = 0.001
    resolution = (128, 128)
    batch_size = 2
    num_slots = 3
    num_iterations = 2

    optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)
    model = model_utils.build_model(resolution, batch_size, num_slots,
                                    num_iterations, model_type="set_prediction")

    input_shape = (batch_size, resolution[0], resolution[1], 3)
    random_input = tf.random.uniform(input_shape)
    output_shape = (batch_size, num_slots, 19)
    random_output = tf.random.uniform(output_shape)

    with tf.GradientTape() as tape:
      preds = model(random_input, training=True)
      loss_value = utils.hungarian_huber_loss(preds, random_output)

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    assert True  # If we make it to this line, we're all good!


if __name__ == "__main__":
  tf.test.main()
