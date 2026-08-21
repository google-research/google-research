# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

r"""Minimal example for training a video_structure model.

See README.md for installation instructions. To run on GPU device 0:

CUDA_VISIBLE_DEVICES=0 python -m video_structure.train
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from video_structure import datasets
from video_structure import dynamics
from video_structure import hyperparameters
from video_structure import losses
from video_structure import vision

FLAGS = flags.FLAGS


def build_model(cfg, data_shapes):
  """Builds the complete model with image encoder plus dynamics model.

  This architecture is meant for testing/illustration only.

  Model architecture:

    image_sequence --> keypoints --> reconstructed_image_sequence
                          |
                          V
                    dynamics_model --> predicted_keypoints

  The model takes a [batch_size, timesteps, H, W, C] image sequence as input. It
  "observes" all frames, detects keypoints, and reconstructs the images. The
  dynamics model learns to predict future keypoints based on the detected
  keypoints.

  Args:
    cfg: ConfigDict with model hyperparameters.
    data_shapes: Dict of shapes of model input tensors, as returned by
      datasets.get_sequence_dataset.
  Returns:
    tf.keras.Model object.
  """
  input_shape_no_batch = data_shapes['image'][1:]  # Keras uses shape w/o batch.
  input_images = tf.keras.Input(shape=input_shape_no_batch, name='image')

  # Vision model:
  observed_keypoints, _ = vision.build_images_to_keypoints_net(
      cfg, input_shape_no_batch)(input_images)
  keypoints_to_images_net = vision.build_keypoints_to_images_net(
      cfg, input_shape_no_batch)
  reconstructed_images = keypoints_to_images_net([
      observed_keypoints,
      input_images[:, 0, Ellipsis],
      observed_keypoints[:, 0, Ellipsis]])

  # Dynamics model:
  observed_keypoints_stop = tf.keras.layers.Lambda(tf.stop_gradient)(
      observed_keypoints)
  dynamics_model = dynamics.build_vrnn(cfg)
  predicted_keypoints, kl_divergence = dynamics_model(observed_keypoints_stop)

  model = tf.keras.Model(
      inputs=[input_images],
      outputs=[reconstructed_images, observed_keypoints, predicted_keypoints],
      name='autoencoder')

  # Losses:
  image_loss = tf.nn.l2_loss(input_images - reconstructed_images)
  # Normalize by batch size and sequence length:
  image_loss /= tf.to_float(
      tf.shape(input_images)[0] * tf.shape(input_images)[1])
  model.add_loss(image_loss)

  separation_loss = losses.temporal_separation_loss(
      cfg, observed_keypoints[:, :cfg.observed_steps, Ellipsis])
  model.add_loss(cfg.separation_loss_scale * separation_loss)

  vrnn_coord_pred_loss = tf.nn.l2_loss(
      observed_keypoints_stop - predicted_keypoints)

  # Normalize by batch size and sequence length:
  vrnn_coord_pred_loss /= tf.to_float(
      tf.shape(input_images)[0] * tf.shape(input_images)[1])
  model.add_loss(vrnn_coord_pred_loss)

  kl_loss = tf.reduce_mean(kl_divergence)  # Mean over batch and timesteps.
  model.add_loss(cfg.kl_loss_scale * kl_loss)

  return model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  cfg = hyperparameters.get_config()

  train_dataset, data_shapes = datasets.get_sequence_dataset(
      data_dir=os.path.join(cfg.data_dir, cfg.train_dir),
      batch_size=cfg.batch_size,
      num_timesteps=cfg.observed_steps + cfg.predicted_steps)

  test_dataset, _ = datasets.get_sequence_dataset(
      data_dir=os.path.join(cfg.data_dir, cfg.test_dir),
      batch_size=cfg.batch_size,
      num_timesteps=cfg.observed_steps + cfg.predicted_steps)

  model = build_model(cfg, data_shapes)
  optimizer = tf.keras.optimizers.Adam(
      lr=cfg.learning_rate, clipnorm=cfg.clipnorm)
  model.compile(optimizer)

  model.fit(
      x=train_dataset,
      steps_per_epoch=cfg.steps_per_epoch,
      epochs=cfg.num_epochs,
      validation_data=test_dataset,
      validation_steps=1)


if __name__ == '__main__':
  app.run(main)
