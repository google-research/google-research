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

"""Util functions used for testing."""

import random
from typing import List
import dataclasses
import numpy as np
from kws_streaming.layers import data_frame
from kws_streaming.layers import modes
from kws_streaming.layers.compat import tf


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


@dataclasses.dataclass
class Params(object):
  """Parameters for data and other settings."""

  cnn_strides: List[int]  # all strides in the model
  clip_duration_ms: float = 16.0  # duration of audio clipl in ms
  preprocess: str = 'custom'  # special case to customize input data shape
  sample_rate: int = 16000  # sample rate of the data
  data_stride: int = 1  # strides for data
  batch_size: int = 1  # batch size

  def __post_init__(self):
    # defines the step of feeding input data
    self.data_shape = (int(np.prod(self.cnn_strides)),)

    self.desired_samples = int(
        self.sample_rate * self.clip_duration_ms / 1000)

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


def generate_img(img_size_y=12,
                 img_size_x=12,
                 obj_a=2,
                 back_bias=1.0,
                 noise_scale=0.1):
  """Generates image with square in the center.

  Args:
    img_size_y: vertical image size
    img_size_x: horizontal image size
    obj_a: amplitude of square in the center. if None then produces
      image without square and with noise only
    back_bias: background level
    noise_scale: noise parameter
  Returns:
    2D image
  """
  img = np.zeros((img_size_y, img_size_x)) + back_bias
  obj_size_y = img_size_y // 2
  obj_size_x = img_size_x // 2
  obj_y = obj_size_y
  obj_x = obj_size_x
  if obj_a is not None:
    for dy in range(obj_size_y):
      for dx in range(obj_size_x):
        y = obj_y + dy - obj_size_y//2
        x = obj_x + dx - obj_size_x//2
        img[y][x] = img[y][x] + obj_a
  return img + np.random.normal(
      size=(img_size_y, img_size_x), scale=noise_scale)


def generate_data(img_size_y=12, img_size_x=12, n_samples=16):
  """Generates 2d images with labels for two category.

  Args:
    img_size_y: vertical image size
    img_size_x: horizontal image size
    n_samples: number of samples
  Returns:
    array of 2D images with labels
  """
  data = []
  labels = []
  for _ in range(n_samples):
    rnd = np.random.uniform()
    label = 0
    if rnd > 0.5:
      img = generate_img(img_size_y, img_size_x, obj_a=2)
      label = 1
    else:
      img = generate_img(img_size_y, img_size_x, obj_a=None)
    labels.append(label)
    data.append(img)

  data = np.asarray(data)
  labels = np.asarray(labels)
  return data, labels
