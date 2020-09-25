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

"""Slot Attention model for object discovery and set prediction."""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


class SlotAttention(layers.Layer):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.

    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon

    self.norm_inputs = layers.LayerNormalization()
    self.norm_slots = layers.LayerNormalization()
    self.norm_mlp = layers.LayerNormalization()

    # Parameters for Gaussian init (shared by all slots).
    self.slots_mu = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],
        dtype=tf.float32,
        name="slots_mu")
    self.slots_log_sigma = self.add_weight(
        initializer="glorot_uniform",
        shape=[1, 1, self.slot_size],
        dtype=tf.float32,
        name="slots_log_sigma")

    # Linear maps for the attention module.
    self.project_q = layers.Dense(self.slot_size, use_bias=False, name="q")
    self.project_k = layers.Dense(self.slot_size, use_bias=False, name="k")
    self.project_v = layers.Dense(self.slot_size, use_bias=False, name="v")

    # Slot update functions.
    self.gru = layers.GRUCell(self.slot_size)
    self.mlp = tf.keras.Sequential([
        layers.Dense(self.mlp_hidden_size, activation="relu"),
        layers.Dense(self.slot_size)
    ], name="mlp")

  def call(self, inputs):
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
        [tf.shape(inputs)[0], self.num_slots, self.slot_size])

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
      slots_prev = slots
      slots = self.norm_slots(slots)

      # Attention.
      q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
      q *= self.slot_size ** -0.5  # Normalization.
      attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1)
      attn = tf.nn.softmax(attn_logits, axis=-1)
      # `attn` has shape: [batch_size, num_inputs, num_slots].

      # Weigted mean.
      attn += self.epsilon
      attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
      updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # Slot update.
      slots, _ = self.gru(updates, [slots_prev])
      slots += self.mlp(self.norm_mlp(slots))

    return slots


def spatial_broadcast(slots, resolution):
  """Broadcast slot features to a 2D grid and collapse slot dimension."""
  # `slots` has shape: [batch_size, num_slots, slot_size].
  slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])
  # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
  return grid


def spatial_flatten(x):
  return tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
  channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
  return channels, masks


class SlotAttentionAutoEncoder(layers.Layer):
  """Slot Attention-based auto-encoder for object discovery."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
    ], name="encoder_cnn")

    self.decoder_initial_size = (8, 8)
    self.decoder_cnn = tf.keras.Sequential([
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            4, 3, strides=(1, 1), padding="SAME", activation=None)
    ], name="decoder_cnn")

    self.encoder_pos = SoftPositionEmbed(64, self.resolution)
    self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64)
    ], name="feedforward")

    self.slot_attention = SlotAttention(
        num_iterations=self.num_iterations,
        num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].

    # Slot Attention module.
    slots = self.slot_attention(x)
    # `slots` has shape: [batch_size, num_slots, slot_size].

    # Spatial broadcast decoder.
    x = spatial_broadcast(slots, self.decoder_initial_size)
    # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    x = self.decoder_pos(x)
    x = self.decoder_cnn(x)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

    # Undo combination of slot and batch dimension; split alpha masks.
    recons, masks = unstack_and_split(x, batch_size=image.shape[0])
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = tf.nn.softmax(masks, axis=1)
    recon_combined = tf.reduce_sum(recons * masks, axis=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

    return recon_combined, recons, masks, slots


class SlotAttentionClassifier(layers.Layer):
  """Slot Attention-based classifier for property prediction."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based classifier.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    self.encoder_cnn = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, strides=(2, 2),
                      padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, strides=(2, 2),
                      padding="SAME", activation="relu"),
        layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
    ], name="encoder_cnn")

    self.encoder_pos = SoftPositionEmbed(64, (32, 32))

    self.layer_norm = layers.LayerNormalization()
    self.mlp = tf.keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64)
    ], name="feedforward")

    self.slot_attention = SlotAttention(
        num_iterations=self.num_iterations,
        num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)

    self.mlp_classifier = tf.keras.Sequential(
        [layers.Dense(64, activation="relu"),
         layers.Dense(19, activation="sigmoid")],  # Number of targets in CLEVR.
        name="mlp_classifier")

  def call(self, image):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn(image)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].

    # Slot Attention module.
    slots = self.slot_attention(x)
    # `slots` has shape: [batch_size, num_slots, slot_size].

    # Apply classifier per slot. The predictions have shape
    # [batch_size, num_slots, set_dimension].

    predictions = self.mlp_classifier(slots)

    return predictions


def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()
    self.dense = layers.Dense(hidden_size, use_bias=True)
    self.grid = build_grid(resolution)

  def call(self, inputs):
    return inputs + self.dense(self.grid)


def build_model(resolution, batch_size, num_slots, num_iterations,
                num_channels=3, model_type="object_discovery"):
  """Build keras model."""
  if model_type == "object_discovery":
    model_def = SlotAttentionAutoEncoder
  elif model_type == "set_prediction":
    model_def = SlotAttentionClassifier
  else:
    raise ValueError("Invalid name for model type.")

  image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
  outputs = model_def(resolution, num_slots, num_iterations)(image)
  model = tf.keras.Model(inputs=image, outputs=outputs)
  return model




