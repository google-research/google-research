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

"""Input reading functions."""

from options import FLAGS
import tensorflow as tf

IMG_SIZE = FLAGS.img_size
NUM_CHANNELS = 1
CROP_AUG = 4
NUM_CLASSES = FLAGS.num_classes


def parse_function_train(image, label):
  """Sampling function for training dataset."""
  image = tf.cast(image, tf.float32)
  image_orig = tf.reshape(image, [IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
  image = tf.image.resize_image_with_crop_or_pad(image_orig,
                                                 IMG_SIZE + CROP_AUG,
                                                 IMG_SIZE + CROP_AUG)
  image = tf.random_crop(image, [IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
  image = tf.image.random_flip_left_right(image)
  image = tf.image.per_image_standardization(image)
  label = tf.cast(label, tf.int32)
  return image, image_orig, label


def parse_function_eval(image, label):
  """Sampling function for eval dataset."""
  image = tf.cast(image, tf.float32)
  image_orig = tf.reshape(image, [IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
  image = tf.image.per_image_standardization(image_orig)
  label = tf.cast(label, tf.int32)
  return image, image_orig, label


def parse_function_test(image):
  """Sampling function for test dataset."""
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
  image = tf.image.per_image_standardization(image)
  image = tf.expand_dims(image, 0)
  return image
