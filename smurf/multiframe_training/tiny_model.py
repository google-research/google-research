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

"""Implements the tiny model used for the SMURF multi-frame self-supervision."""

from typing import Dict, Tuple

import tensorflow as tf


class TinyModel(tf.keras.Model):
  """Tiny model that should resemble the motion model."""

  def __init__(self):
    super().__init__()
    self._layers = (
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(2, 3, padding='same'),
    )

  def call(self, x):
    x *= -1.0  # Flip flow direction (constant motion model).
    for layer in self._layers:
      x = layer(x)
    return x


def _positions_center_origin(height, width):
  """Returns image coordinates where the origin is at the image center."""
  h = tf.range(0.0, height, 1) / (height - 1) * 2
  w = tf.range(0.0, width, 1) / (width - 1) * 2
  return tf.stack(tf.meshgrid(h - 1, w - 1, indexing='ij'), -1)


def train_and_run_tiny_model(
    flow_forward,
    flow_backward,
    mask_forward,
    mask_backward,
    iterations = 2000):
  """Fuses temporal forward/backward flow via a learned proflow like model.

  The non-masked locations of the (temporal) forward (t -> t+1) and (temporal)
  backward flow (t -> t-1) are used as ground truth values to train a tiny model
  that realizes a mapping from backward to forward flow (t -> t-1)->(t -> t+1).
  This allows to fill in masked locations in the forward flow using backward
  flow estimates (this can be done for all locations were only mask_forward is 0
  and mask_backward is 1.

  Args:
    flow_forward: Temporal forward flow (t -> t+1).
    flow_backward: Temporal backward flow (t -> t-1).
    mask_forward: Mask associated with the temporal forward flow. The values are
      in
      {0,1}, where 0: is occluded/invalid and 1: is non-occluded/valid.
    mask_backward: Mask associated with the temporal backward flow. The values
      are in
      {0,1}, where 0: is occluded/invalid and 1: is non-occluded/valid.
    iterations: Number of iterations used to train the model.

  Returns:
    Fused flow field (a temporal forward flow) and new mask.
  """
  tf.debugging.assert_shapes([
      (flow_forward, ['batch_size', 'height', 'width', 2]),
      (flow_backward, ['batch_size', 'height', 'width', 2]),
      (mask_forward, ['batch_size', 'height', 'width', 1]),
      (mask_backward, ['batch_size', 'height', 'width', 1]),
  ])
  height = flow_backward.shape[-3]
  width = flow_backward.shape[-2]

  # Create network input, i.e. stack image locations to the backward flow.
  coords = _positions_center_origin(height, width)
  net_input = tf.concat([flow_backward, coords[tf.newaxis]], -1)

  # Create a dataset from the flow pair.
  dataset = tf.data.Dataset.from_tensor_slices((net_input, {
      'flow_forward': flow_forward,
      'mask_forward': mask_forward,
      'mask_backward': mask_backward,
  })).repeat().batch(1)
  dataset_iterator = iter(dataset)

  # Define the loss function.
  def loss_fn(labels, predictions):
    proxy_ground_truth = labels['flow_forward']
    mask = labels['mask_forward'] * labels['mask_backward']
    error = tf.norm((predictions - proxy_ground_truth),
                    ord='euclidean',
                    axis=-1,
                    keepdims=True)
    return tf.reduce_sum(error * mask)

  # Create an optimizer.
  initial_learning_rate = 0.01
  decay_steps = iterations // 20
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=decay_steps,
      decay_rate=0.8,
      staircase=False)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

  # Define one training iteration.
  @tf.function
  def train_step(input_data, labels):
    with tf.GradientTape() as tape:
      predictions = model(input_data)
      loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Create and train the model.
  model = TinyModel()
  for _ in range(iterations):
    images, labels = dataset_iterator.next()
    train_step(images, labels)

  # Use model to convert backward flow into forward flow.
  predicted_flow_forward = model(net_input)

  # Fuse flow fields.
  mask_backward_no_forward = (1 - mask_forward) * mask_backward
  fused_flow = (
      flow_forward * mask_forward +
      predicted_flow_forward * mask_backward_no_forward)
  fused_mask = tf.clip_by_value(mask_forward + mask_backward, 0, 1)
  return (fused_flow, fused_mask)
