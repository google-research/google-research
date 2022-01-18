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

"""Driver class for running inference."""
import tensorflow.compat.v1 as tf


class Driver(object):
  """A driver for running inference.

  Attributes:
    batch_size: int. Eval batch size.
    image_size: int. Input image size, determined by model name.
  """

  def __init__(self, batch_size=1, image_size=224):
    """Initialize internal variables."""
    self.batch_size = batch_size
    self.image_size = image_size

  def restore_model(self, sess, ckpt_dir):
    """Restore variables from checkpoint dir."""
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    tf.train.get_or_create_global_step()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

  def build_model(self, features):
    """Build model with input features."""
    del features
    raise ValueError('Must be implemented by subclasses.')

  def preprocess_fn(self, image_bytes, image_size):
    """Preprocesses the given image for evaluation."""
    del image_bytes, image_size
    raise ValueError('Must be implemented by subclasses.')

  def build_dataset(self, filenames, labels):
    """Build input dataset."""
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      image = self.preprocess_fn(image_string, self.image_size)
      return image, label

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(self.batch_size, drop_remainder=False)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels
