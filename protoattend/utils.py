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

"""Util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import input_data
import numpy as np
from options import FLAGS
import tensorflow as tf


def class_explainability(labels, weights):
  """Helper function to determine confidence level."""
  expl_class = np.zeros((weights.shape[0], FLAGS.num_classes))
  for bi in range(weights.shape[0]):
    for di in range(labels.shape[0]):
      expl_class[bi, labels[di]] += weights[bi, di]
  return expl_class.astype(np.float32)


def put_text(images, texts):
  """This function allows adding a text to an image for visualization.

  (Tensorboard normally does not allow adding labels or titles to the
  images with tf.summary.image.) We use this function to visualize
  the coefficient for each prototype along with the corresponding image.

  Args:
    images: Image to visualize.
    texts: Text string to be added.

  Returns:
    Images with visualized text.

  """

  # Extend image extension_amount times in one direction to locate the text of
  # floats (prototype coefficients).
  num_images = images.shape[0]
  num_image_channels = images.shape[3]

  extension_amount = 3
  text_location = (8, 20)
  text_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
  color_point = (255, 0, 0)

  result = np.zeros((num_images, input_data.IMG_SIZE,
                     input_data.IMG_SIZE * extension_amount, 3))
  for i in range(num_images):
    text = np.array2string(texts[i], precision=2)
    if isinstance(text, bytes):
      text = text.decode()
    for j in range(3):
      result[i, :, :input_data.IMG_SIZE, j] = images[i, :, :, 0]
    result[i, :, input_data.IMG_SIZE:, :] = cv2.putText(
        np.zeros(
            (input_data.IMG_SIZE, input_data.IMG_SIZE * (extension_amount - 1),
             num_image_channels)), text, text_location, text_font, 1.0,
        color_point, 2, cv2.LINE_AA)
  return result.astype(np.float32)


def tf_put_text(imgs, texts):
  """Convert helper function to Tensorflow."""
  return tf.py_func(put_text, [imgs, texts], Tout=imgs.dtype)
