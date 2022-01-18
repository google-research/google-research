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

# Lint as: python3
"""Tests for video_structure.vision."""

import os
from absl import flags
from absl.testing import absltest
import tensorflow.compat.v1 as tf
from video_structure import datasets
from video_structure import hyperparameters
from video_structure import vision

FLAGS = flags.FLAGS

TESTDATA_DIR = 'video_structure/testdata'


class VisionTest(tf.test.TestCase):

  def setUp(self):

    # Hyperparameter config for test models:
    self.cfg = hyperparameters.get_config()
    self.cfg.train_dir = os.path.join(FLAGS.test_srcdir, TESTDATA_DIR)
    self.cfg.batch_size = 4
    self.cfg.observed_steps = 2
    self.cfg.predicted_steps = 2
    self.cfg.heatmap_width = 16
    self.cfg.layers_per_scale = 1
    self.cfg.num_keypoints = 3

    # Shapes of test dataset:
    self.time_steps = self.cfg.observed_steps + self.cfg.predicted_steps
    self.data_shapes = {
        'filename': (None, self.time_steps),
        'frame_ind': (None, self.time_steps),
        'image': (None, self.time_steps, 64, 64, 3),
        'true_object_pos': (None, self.time_steps, 0, 2)}

    super().setUp()

  def testAutoencoderTrainingLossGoesDown(self):
    """Tests a minimal Keras training loop for the non-dynamic model parts."""
    dataset, data_shapes = datasets.get_sequence_dataset(
        data_dir=self.cfg.train_dir,
        file_glob='acrobot*',
        batch_size=self.cfg.batch_size,
        num_timesteps=self.cfg.observed_steps + self.cfg.predicted_steps,
        random_offset=True)
    autoencoder = Autoencoder(self.cfg, data_shapes)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    autoencoder.compile(optimizer)
    history = autoencoder.fit(dataset, steps_per_epoch=1, epochs=3)
    self.assertLess(history.history['loss'][-1], history.history['loss'][0])

  def testImagesToKeypointsNetShapes(self):
    model = vision.build_images_to_keypoints_net(
        self.cfg, self.data_shapes['image'][1:])
    images = tf.zeros((self.cfg.batch_size,) + self.data_shapes['image'][1:])
    keypoints, heatmaps = model(images)
    self.assertEqual(
        keypoints.shape.as_list(),
        [self.cfg.batch_size, self.time_steps, self.cfg.num_keypoints, 3])
    self.assertEqual(
        heatmaps.shape.as_list(),
        [self.cfg.batch_size, self.time_steps, self.cfg.heatmap_width,
         self.cfg.heatmap_width, 3])

  def testKeypointsToImagesNetShapes(self):
    model = vision.build_keypoints_to_images_net(
        self.cfg, self.data_shapes['image'][1:])
    keypoints = tf.zeros(
        (self.cfg.batch_size, self.time_steps, self.cfg.num_keypoints, 3))
    first_frame = tf.zeros(
        (self.cfg.batch_size,) + self.data_shapes['image'][2:])
    reconstructed_images = model([keypoints, first_frame, keypoints[:, 0, Ellipsis]])
    self.assertEqual(
        reconstructed_images.shape.as_list(),
        [self.cfg.batch_size] + list(self.data_shapes['image'][1:]))

  def testImageEncoderShapes(self):
    model = vision.build_image_encoder(
        self.data_shapes['image'][2:], **self.cfg.conv_layer_kwargs)
    images = tf.zeros((self.cfg.batch_size,) + self.data_shapes['image'][2:])
    encoded = model(images)
    self.assertEqual(
        encoded.shape.as_list()[:-1],
        [self.cfg.batch_size, self.cfg.heatmap_width, self.cfg.heatmap_width])

  def testImageDecoderShapes(self):
    features_shape = [
        self.cfg.batch_size, self.cfg.heatmap_width, self.cfg.heatmap_width, 64]
    image_width = self.data_shapes['image'][-2]
    model = vision.build_image_decoder(
        features_shape[1:], output_width=image_width,
        **self.cfg.conv_layer_kwargs)
    output = model(tf.zeros(features_shape))
    self.assertEqual(
        output.shape.as_list()[:-1],
        [self.cfg.batch_size, image_width, image_width])


class Autoencoder(tf.keras.Model):
  """Simple image autoencoder without dynamics.

  This architecture is meant for testing the image-processing submodels.

  Model architecture:

    image_sequence --> keypoints --> reconstructed_image_sequence

  The model takes a standard [batch_size, timesteps, H, W, C] image sequence as
  input. It "observes" all frames, detects keypoints, and reconstructs the
  images.
  """

  def __init__(self, cfg, data_shapes):
    """Constructs the autoencoder.

    Args:
      cfg: ConfigDict with model hyperparameters.
      data_shapes: Dict of shapes of model input tensors, as returned by
        datasets.get_sequence_dataset.
    """
    input_sequence = tf.keras.Input(
        shape=data_shapes['image'][1:],
        name='image')

    image_shape = data_shapes['image'][1:]
    keypoints, _ = vision.build_images_to_keypoints_net(
        cfg, image_shape)(input_sequence)
    reconstructed_sequence = vision.build_keypoints_to_images_net(
        cfg, image_shape)([
            keypoints,
            input_sequence[:, 0, Ellipsis],
            keypoints[:, 0, Ellipsis]])

    super(Autoencoder, self).__init__(
        inputs=input_sequence, outputs=reconstructed_sequence,
        name='autoencoder')

    self.add_loss(tf.nn.l2_loss(input_sequence - reconstructed_sequence))


if __name__ == '__main__':
  absltest.main()
