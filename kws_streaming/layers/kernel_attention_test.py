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

"""Tests for kws_streaming.layers.kernel_attention."""

from absl.testing import parameterized
import numpy as np
from kws_streaming.layers import kernel_attention
from kws_streaming.layers import modes
from kws_streaming.layers import test_utils
from kws_streaming.layers.compat import tf
from kws_streaming.models import utils


def toy_model(batch_size, time_size, feature1_size, feature2_size):
  """Toy model with streaming aware KernelAttention.

  Input arguments define input tensor shape:
    [batch_size, time_size, feature1_size, feature2_size] for query and value.

  Args:
    batch_size: Batch size.
    time_size: Sequence length.
    feature1_size: Size of the first feature dim.
    feature2_size: Size of the second feature dim.
  Returns:
    Keras model
  """
  input1 = tf.keras.layers.Input(
      shape=(time_size, feature1_size, feature2_size),
      batch_size=batch_size)
  input2 = tf.keras.layers.Input(
      shape=(time_size, feature1_size, feature2_size),
      batch_size=batch_size)
  inputs = [input1, input2]

  feature_transform = 'relu'
  num_random_features = 0
  random_features_seed = 0
  num_projection_freq_bins = feature1_size
  window_length = time_size  # Making it equal to input sequence length.
  window_decay = 0.97
  num_heads = 3
  head_dim = 2
  dropout = 0.0

  output = kernel_attention.KernelAttention(
      feature_transform=feature_transform,
      num_random_features=num_random_features,
      seed=random_features_seed,
      use_causal_windowed=True,
      causal_chunk_length=num_projection_freq_bins,
      causal_window_length=window_length,
      causal_window_decay=window_decay,
      num_heads=num_heads,
      key_dim=head_dim,
      dropout=dropout,
      use_bias=True,
      name='kernel_attention')(inputs)

  return tf.keras.Model(inputs, output)


class KernelAttentionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.set_seed(123)
    self.batch_size = 1
    self.time_size = 16
    self.feature1_size = 4
    self.feature2_size = 2
    self.query = np.random.rand(self.batch_size, self.time_size,
                                self.feature1_size,
                                self.feature2_size).astype(np.float32)

    self.value = np.random.rand(self.batch_size, self.time_size,
                                self.feature1_size,
                                self.feature2_size).astype(np.float32)

  @parameterized.parameters(1, 2)
  def test_streaming(self, stream_chunk_length):
    # Non streaming model.
    non_streaming_model = toy_model(self.batch_size, self.time_size,
                                    self.feature1_size, self.feature2_size)
    non_streaming_output = non_streaming_model.predict([self.query, self.value])

    params = test_utils.Params([1], 1)
    params.preprocess = 'custom'
    # This model has two inputs:
    params.data_shape = (stream_chunk_length, self.feature1_size,
                         self.feature2_size)
    params.cond_shape = (stream_chunk_length, self.feature1_size,
                         self.feature2_size)

    # Prepare streaming aware model.
    streaming_model = utils.to_streaming_inference(
        non_streaming_model, params,
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming_model.summary()

    # Run streaming model.
    num_chunks = self.time_size // stream_chunk_length
    streaming_output = []
    for i in range(num_chunks):
      query_step = self.query[:, i * stream_chunk_length:(i + 1) *
                              stream_chunk_length, :]
      value_step = self.value[:, i * stream_chunk_length:(i + 1) *
                              stream_chunk_length, :]

      streaming_step_output = streaming_model.predict([query_step, value_step])
      streaming_output.append(streaming_step_output)

    streaming_output = tf.concat(streaming_output, axis=1)
    self.assertAllClose(streaming_output, non_streaming_output)

    # Convert TF non-streaming model to TFLite internal-state streaming model.
    tflite_streaming_model = utils.model_to_tflite(
        None, non_streaming_model, params,
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    self.assertTrue(tflite_streaming_model)


if __name__ == '__main__':
  tf.test.main()
