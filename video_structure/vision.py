# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Vision-related components of the structured video representation model.

These components perform the pixels <--> keypoints transformation.
"""

import functools
import numpy as np
import tensorflow.compat.v1 as tf
from video_structure import ops

layers = tf.keras.layers


# Wrap commonly used Tensorflow ops in Keras layers:
def stack_time(inputs):
  return layers.Lambda(lambda x: tf.stack(x, axis=1, name='stack_time'))(inputs)


def unstack_time(inputs):
  return layers.Lambda(lambda x: tf.unstack(x, axis=1, name='unstack_time'))(
      inputs)


def add_coord_channels(inputs):
  return layers.Lambda(ops.add_coord_channels)(inputs)


def maps_to_keypoints(inputs):
  return layers.Lambda(ops.maps_to_keypoints)(inputs)


def build_images_to_keypoints_net(cfg, image_shape):
  """Builds a model that encodes an image sequence into a keypoint sequence.

  The model applies the same convolutional feature extractor to all images in
  the sequence. The feature maps are then reduced to num_keypoints heatmaps, and
  the heatmaps to (x, y, scale)-keypoints.

  Args:
    cfg: ConfigDict with model hyperparamters.
    image_shape: Image shape tuple: (num_timesteps, H, W, C).

  Returns:
    A tf.keras.Model object.
  """

  image_sequence = tf.keras.Input(shape=image_shape, name='encoder_input')

  # Adjust channel number to account for add_coord_channels:
  encoder_input_shape = image_shape[1:-1] + (image_shape[-1] + 2,)
  # Build feature extractor:
  image_encoder = build_image_encoder(
      input_shape=encoder_input_shape,
      initial_num_filters=cfg.num_encoder_filters,
      output_map_width=cfg.heatmap_width,
      layers_per_scale=cfg.layers_per_scale,
      **cfg.conv_layer_kwargs)

  # Build final layer that maps to the desired number of heatmaps:
  features_to_keypoint_heatmaps = layers.Conv2D(
      filters=cfg.num_keypoints,
      kernel_size=1,
      padding='same',
      activation=tf.nn.softplus,  # Heatmaps must be non-negative.
      activity_regularizer=functools.partial(
          _get_heatmap_penalty, factor=cfg.heatmap_regularization))

  # Separate timesteps into list:
  image_list = unstack_time(image_sequence)
  heatmaps_list = []
  keypoints_list = []

  # Image to keypoints:
  for image in image_list:
    image = add_coord_channels(image)
    encoded = image_encoder(image)
    heatmaps = features_to_keypoint_heatmaps(encoded)
    keypoints = maps_to_keypoints(heatmaps)
    heatmaps_list.append(heatmaps)
    keypoints_list.append(keypoints)

  # Combine timesteps:
  heatmaps = stack_time(heatmaps_list)
  keypoints = stack_time(keypoints_list)

  return tf.keras.Model(inputs=image_sequence, outputs=[keypoints, heatmaps])


def build_keypoints_to_images_net(cfg, image_shape):
  """Builds a model to reconstructs an image sequence from a keypoint sequence.

  Model architecture:

    (keypoint_sequence, image[0], keypoints[0]) --> reconstructed_image_sequence

  For all frames image[t] we also we also concatenate the Gaussian maps for
  the keypoints obtained from the initial frame image[0]. This helps the
  decoder "inpaint" the image regions that are occluded by objects in the first
  frame.

  Args:
    cfg: ConfigDict with model hyperparameters.
    image_shape: Image shape tuple: (num_timesteps, H, W, C).

  Returns:
    A tf.keras.Model object.
  """
  num_timesteps = cfg.observed_steps + cfg.predicted_steps
  keypoints_shape = [num_timesteps, cfg.num_keypoints, 3]
  keypoints_sequence = tf.keras.Input(shape=keypoints_shape, name='keypoints')
  first_frame = tf.keras.Input(shape=image_shape[1:], name='first_frame')
  first_frame_keypoints = tf.keras.Input(
      shape=keypoints_shape[1:], name='first_frame_keypoints')

  # Build encoder net to extract appearance features from the first frame:
  appearance_feature_extractor = build_image_encoder(
      input_shape=image_shape[1:],
      initial_num_filters=cfg.num_encoder_filters,
      layers_per_scale=cfg.layers_per_scale,
      **cfg.conv_layer_kwargs)

  # Build image decoder that goes from Gaussian maps to reconstructed images:
  num_encoder_output_channels = (
      cfg.num_encoder_filters * image_shape[1] // cfg.heatmap_width)
  input_shape = [
      cfg.heatmap_width, cfg.heatmap_width, num_encoder_output_channels]
  image_decoder = build_image_decoder(
      input_shape=input_shape,
      output_width=image_shape[1],
      layers_per_scale=cfg.layers_per_scale,
      **cfg.conv_layer_kwargs)

  # Build layers to adjust channel numbers for decoder input and output image:
  kwargs = dict(cfg.conv_layer_kwargs)
  kwargs['kernel_size'] = 1
  adjust_channels_of_decoder_input = layers.Conv2D(
      num_encoder_output_channels, **kwargs)

  kwargs = dict(cfg.conv_layer_kwargs)
  kwargs['kernel_size'] = 1
  kwargs['activation'] = None
  adjust_channels_of_output_image = layers.Conv2D(
      image_shape[-1], **kwargs)

  # Build keypoints_to_maps layer:
  keypoints_to_maps = layers.Lambda(
      functools.partial(
          ops.keypoints_to_maps,
          sigma=cfg.keypoint_width,
          heatmap_width=cfg.heatmap_width))

  # Get features and maps for first frame:
  # Note that we cannot use the Gaussian maps above because the
  # first_frame_keypoints may be different than the keypoints (i.e. obs vs
  # pred).
  first_frame_features = appearance_feature_extractor(first_frame)
  first_frame_gaussian_maps = keypoints_to_maps(first_frame_keypoints)

  # Separate timesteps:
  keypoints_list = unstack_time(keypoints_sequence)
  image_list = []

  # Loop over timesteps:
  for keypoints in keypoints_list:
    # Convert keypoints to pixel maps:
    gaussian_maps = keypoints_to_maps(keypoints)

    # Reconstruct image:
    combined_representation = layers.Concatenate(axis=-1)(
        [gaussian_maps, first_frame_features, first_frame_gaussian_maps])
    combined_representation = add_coord_channels(combined_representation)
    combined_representation = adjust_channels_of_decoder_input(
        combined_representation)
    decoded_representation = image_decoder(combined_representation)
    image_list.append(adjust_channels_of_output_image(decoded_representation))

  # Combine timesteps:
  image_sequences = stack_time(image_list)

  # Add in the first frame of the sequence such that the model only needs to
  # predict the change from the first frame:
  image_sequences = layers.Add()([image_sequences, first_frame[:, None, Ellipsis]])

  return tf.keras.Model(
      inputs=[keypoints_sequence, first_frame, first_frame_keypoints],
      outputs=image_sequences)


def build_image_encoder(
    input_shape, initial_num_filters=32, output_map_width=16,
    layers_per_scale=1, **conv_layer_kwargs):
  """Extracts feature maps from images.

  The encoder iteratively halves the resolution and doubles the number of
  filters until the size of the feature maps is output_map_width by
  output_map_width.

  Args:
    input_shape: Shape of the input image (without batch dimension).
    initial_num_filters: Number of filters to apply at the input resolution.
    output_map_width: Width of the output feature maps.
    layers_per_scale: How many additional size-preserving conv layers to apply
      at each map scale.
    **conv_layer_kwargs: Passed to layers.Conv2D.

  Raises:
    ValueError: If the width of the input image is not compatible with
      output_map_width, i.e. if input_width/output_map_width is not a perfect
      square.
  """

  inputs = tf.keras.Input(shape=input_shape, name='encoder_input')

  if np.log2(input_shape[0] / output_map_width) % 1:
    raise ValueError(
        'The ratio of input width and output_map_width must be a perfect '
        'square, but got {} and {} with ratio {}'.format(
            input_shape[0], output_map_width, inputs[0]/output_map_width))

  # Expand image to initial_num_filters maps:
  x = layers.Conv2D(initial_num_filters, **conv_layer_kwargs)(inputs)
  for _ in range(layers_per_scale):
    x = layers.Conv2D(initial_num_filters, **conv_layer_kwargs)(x)

  # Apply downsampling blocks until feature map width is output_map_width:
  width = int(inputs.get_shape()[1])
  num_filters = initial_num_filters
  while width > output_map_width:
    num_filters *= 2
    width //= 2

    # Reduce resolution:
    x = layers.Conv2D(num_filters, strides=2, **conv_layer_kwargs)(x)

    # Apply additional layers:
    for _ in range(layers_per_scale):
      x = layers.Conv2D(num_filters, strides=1, **conv_layer_kwargs)(x)

  return tf.keras.Model(inputs=inputs, outputs=x, name='image_encoder')


def build_image_decoder(
    input_shape, output_width, layers_per_scale=1, **conv_layer_kwargs):
  """Decodes images from feature maps.

  The encoder iteratively doubles the resolution and halves the number of
  filters until the size of the feature maps is output_width.

  Args:
    input_shape: Shape of the input image (without batch dimension).
    output_width: Width of the output image.
    layers_per_scale: How many additional size-preserving conv layers to apply
      at each map scale.
    **conv_layer_kwargs: Passed to layers.Conv2D.

  Raises:
    ValueError: If the width of the input feature maps is not compatible with
      output_width, i.e. if output_width/input_map_width is not a perfect
      square.
  """

  feature_maps = tf.keras.Input(shape=input_shape, name='feature_maps')
  num_levels = np.log2(output_width / input_shape[0])

  if num_levels % 1:
    raise ValueError(
        'The ratio of output_width and input width must be a perfect '
        'square, but got {} and {} with ratio {}'.format(
            output_width, input_shape[0], output_width/input_shape[0]))

  # Expand until we have filters_out channels:
  x = feature_maps
  num_filters = input_shape[-1]
  def upsample(x):
    new_size = [x.get_shape()[1] * 2, x.get_shape()[2] * 2]
    return tf.image.resize_bilinear(x, new_size, align_corners=True)
  for _ in range(int(num_levels)):
    num_filters //= 2
    x = layers.Lambda(upsample)(x)

    # Apply additional layers:
    for _ in range(layers_per_scale):
      x = layers.Conv2D(num_filters, **conv_layer_kwargs)(x)

  return tf.keras.Model(inputs=feature_maps, outputs=x, name='image_decoder')


def _get_heatmap_penalty(weight_matrix, factor):
  """L1-loss on mean heatmap activations, to encourage sparsity."""
  weight_shape = weight_matrix.shape.as_list()
  assert len(weight_shape) == 4, weight_shape

  heatmap_mean = tf.reduce_mean(weight_matrix, axis=(1, 2))
  penalty = tf.reduce_mean(tf.abs(heatmap_mean))
  return penalty * factor
