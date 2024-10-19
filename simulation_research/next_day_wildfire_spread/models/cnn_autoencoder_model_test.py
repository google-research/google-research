# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for cnn_autoencoder_model."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2 import keras

from simulation_research.next_day_wildfire_spread.models import cnn_autoencoder_model

# TODO(b/237700589): Remove this once the global flag is on.
tf.compat.v2.keras.backend.experimental.enable_tf_random_generator()


class CNNAutoencoderModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_encoder_default(self):
    """Tests encoder with default inputs."""
    input_tensor = tf.ones([1, 4, 4, 2])
    output_tensor = cnn_autoencoder_model.encoder(
        input_tensor, layers_list=(64, 2), pool_list=(2, 2))
    self.assertAllEqual(output_tensor.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-0.02436768, -0.27847868], [-0.0774256, -0.5111736]],
          [[0.50436425, -0.1713084], [0.2803106, -0.5315974]]]],
        dtype=tf.float32)
    self.assertAllClose(output_tensor, expected)

  def test_encoder_pool_list_values(self):
    """Tests encoder with default inputs."""
    input_tensor = tf.ones([1, 4, 4, 2])
    output_tensor = cnn_autoencoder_model.encoder(
        input_tensor, layers_list=(64, 2, 4), pool_list=(2, 1, 2))
    self.assertAllEqual(output_tensor.shape, [1, 2, 2, 4])
    expected = tf.constant(
        [[[[0.04343234, 0.3203296, 0.2921386, -0.39414376],
           [0.21853307, 0.6474583, 0.18453243, 0.08784209]],
          [[0.13576293, 0.43033206, 0.05750746, -0.06081235],
           [0.46124727, 0.45607585, -0.2642231, 0.03700671]]]],
        dtype=tf.float32)
    self.assertAllClose(output_tensor, expected)

  def test_encoder_batch_norm_all(self):
    """Tests encoder with batch_norm 'all'."""
    input_tensor = tf.ones([1, 4, 4, 2])
    output_tensor = cnn_autoencoder_model.encoder(
        input_tensor, layers_list=(64, 2), pool_list=(2, 2), batch_norm='all')
    self.assertAllEqual(output_tensor.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-0.02436293, -0.2781912], [-0.07729936, -0.5105717]],
          [[0.5036894, -0.17111939], [0.27998263, -0.5309998]]]],
        dtype=tf.float32)
    self.assertAllClose(output_tensor, expected)

  def test_encoder_batch_norm_some(self):
    """Tests encoder with batch_norm 'some'."""
    input_tensor = tf.ones([1, 4, 4, 2])
    output_tensor = cnn_autoencoder_model.encoder(
        input_tensor, layers_list=(64, 2), pool_list=(2, 2), batch_norm='some')
    self.assertAllEqual(output_tensor.shape, [1, 2, 2, 2])
    expected = tf.constant(
        [[[[-0.02438722, -0.27846932], [-0.07737678, -0.5110824]],
          [[0.50419325, -0.17129041], [0.2802627, -0.53153074]]]],
        dtype=tf.float32)
    self.assertAllClose(output_tensor, expected)

  _SAMPLE_SIZES = (64, 128)
  _BATCH_SIZES = (4, 8)
  _NUM_INPUT_CHANNELS = (2, 3)
  _NUM_OUTPUT_CHANNELS = (1, 2)
  _LAYERS_LIST = ((32, 64, 128, 256, 256), (64, 128, 256, 512, 512))
  _POOL_LIST = ((2, 2, 2, 2, 2), (2, 1, 2, 2, 2))

  @parameterized.parameters(*zip(_SAMPLE_SIZES, _BATCH_SIZES,
                                 _NUM_INPUT_CHANNELS, _NUM_OUTPUT_CHANNELS,
                                 _LAYERS_LIST, _POOL_LIST))
  def test_create_model(
      self,
      sample_size,
      batch_size,
      num_input_channels,
      num_output_channels,
      layers_list,
      pool_list,
  ):
    """Checks that the input passes through the model without crashing."""
    input_img = keras.Input(
        shape=(sample_size, sample_size, num_input_channels))
    model = cnn_autoencoder_model.create_model(
        input_tensor=input_img,
        num_out_channels=num_output_channels,
        encoder_layers=layers_list,
        decoder_layers=tuple(reversed(layers_list)),
        encoder_pools=pool_list,
        decoder_pools=tuple(reversed(pool_list)))
    keras_model = keras.Model(input_img, model)
    input_tensor = tf.ones(
        [batch_size, sample_size, sample_size, num_input_channels])
    output_tensor = keras_model(input_tensor)
    self.assertAllEqual(
        [batch_size, sample_size, sample_size, num_output_channels],
        output_tensor.shape)


if __name__ == '__main__':
  tf.test.main()
