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

"""Util functions used for testing."""

import random
import numpy as np
from kws_streaming.layers import data_frame
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf
from kws_streaming.train import model_flags


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


class Params(object):
  """Parameters for data and other settings.

  Attributes:
    cnn_strides: list of strides
    clip_duration_ms: duration of audio clipl in ms
    sample_rate: sample rate of the data
    preprocess: method of preprocessing
    data_shape: shape of the data in streaming inference mode
    batch_size: batch size
    desired_samples: number of samples in one sequence
  """

  def __init__(self, cnn_strides, clip_duration_ms=16):
    self.sample_rate = 16000
    self.clip_duration_ms = clip_duration_ms

    # it is a special case to customize input data shape
    self.preprocess = 'custom'

    # defines the step of feeding input data
    self.data_shape = (np.prod(cnn_strides),)

    self.batch_size = 1
    self.desired_samples = int(
        self.sample_rate * self.clip_duration_ms / model_flags.MS_PER_SECOND)

    # align data length with the step
    self.desired_samples = (
        self.desired_samples // self.data_shape[0]) * self.data_shape[0]


def get_test_batch_features_and_labels_numpy(input_shape=None,
                                             output_shape=None):
  """Returns an example of inputs and labels based on the input shape.


  (Hint: For SVDF layers, the shapes would normally be  of format [2, 7, _])

  Args:
    input_shape: The dimentionality of the input. (ex: [2,7,5])
    output_shape: The dimentionality of the output. (ex: [2,7,2])
  """
  if input_shape is None:
    input_shape = [2, 7, 5]

  if output_shape is None:
    output_shape = [2, 7, 2]

  input_values = np.arange(
      np.prod(input_shape), dtype=np.float32) / np.prod(input_shape)
  output_values = np.arange(
      np.prod(output_shape), dtype=np.float32) / np.prod(output_shape)
  return input_values.reshape(input_shape), output_values.reshape(output_shape)


def _get_test_svdf_cell_weights():
  """Returns weights for an SvdfCell with following params.

    (units=4, memory_size=3, rank=1, output_projection_dim=2, use_bias=True).
  """
  return [
      np.array([[-0.31614766, 0.37929568, 0.27584907, -0.36453721],
                [-0.35801932, 0.22514193, 0.27241215, -0.06950231],
                [0.01112892, 0.12732419, 0.38735834, -0.10957076],
                [-0.09451947, 0.15611194, 0.39319292, -0.03019224],
                [0.39612538, 0.16101542, 0.21615031, 0.30737072]],
               dtype=np.float32),
      np.array([[-0.31614769, 0.37929571, 0.27584907, -0.36453718],
                [-0.35801938, 0.22514194, 0.27241215, -0.06950228],
                [0.01112869, 0.12732419, 0.38735834, -0.10957073]],
               dtype=np.float32),
      np.array([-0.00316226, 0.00316225, -0.00316227, 0.00316227],
               dtype=np.float32),
      np.array([[-0.31614763, 0.37929574], [0.2821736, -0.35821268],
                [-0.35801929, 0.22514199], [0.27873668, -0.06317782]],
               dtype=np.float32),
      np.array([0.00316228, 0.00316227], dtype=np.float32)
  ]


class TestBase(tf.test.TestCase):
  """Base class for dense, depthwise conv, svdf layers testing."""

  def setUp(self):
    super(TestBase, self).setUp()
    self.memory_size = 3
    self.batch_size = 2
    self.input_data, self.input_labels = (
        get_test_batch_features_and_labels_numpy())
    self.weights = _get_test_svdf_cell_weights()


class FrameTestBase(tf.test.TestCase):
  """Base class for data frame testing."""

  def setUp(self):
    super(FrameTestBase, self).setUp()

    self.frame_size = 7
    self.frame_step = 5
    self.inference_batch_size = 1

    # generate input signal
    set_seed(1)
    self.data_size = 33
    self.signal = np.random.rand(self.inference_batch_size, self.data_size)

    # non streaming frame extraction based on tf.signal.frame
    data_frame_tf = data_frame.DataFrame(
        mode=modes.Modes.TRAINING,
        inference_batch_size=self.inference_batch_size,
        frame_size=self.frame_size,
        frame_step=self.frame_step)
    # it receives all data with size: data_size
    input1 = tf.keras.layers.Input(
        shape=(self.data_size,),
        batch_size=self.inference_batch_size,
        dtype=tf.float32)
    output1 = data_frame_tf(inputs=input1)
    self.model_tf = tf.keras.models.Model(input1, output1)

    # generate frames for the whole signal (no streaming here)
    self.output_frames_tf = self.model_tf.predict(self.signal)
