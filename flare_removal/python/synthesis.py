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

"""Generates synthetic scenes containing lens flare."""
import math

import tensorflow as tf

from flare_removal.python import utils


def add_flare(scene,
              flare,
              noise,
              flare_max_gain = 10.0,
              apply_affine = True,
              training_res = 512):
  """Adds flare to natural images.

  Here the natural images are in sRGB. They are first linearized before flare
  patterns are added. The result is then converted back to sRGB.

  Args:
    scene: Natural image batch in sRGB.
    flare: Lens flare image batch in sRGB.
    noise: Strength of the additive Gaussian noise. For each image, the Gaussian
      variance is drawn from a scaled Chi-squared distribution, where the scale
      is defined by `noise`.
    flare_max_gain: Maximum gain applied to the flare images in the linear
      domain. RGB gains are applied randomly and independently, not exceeding
      this maximum.
    apply_affine: Whether to apply affine transformation.
    training_res: Resolution of training images. Images must be square, and this
      value specifies the side length.

  Returns:
    - Flare-free scene in sRGB.
    - Flare-only image in sRGB.
    - Scene with flare in sRGB.
    - Gamma value used during synthesis.
  """
  batch_size, flare_input_height, flare_input_width, _ = flare.shape

  # Since the gamma encoding is unknown, we use a random value so that the model
  # will hopefully generalize to a reasonable range of gammas.
  gamma = tf.random.uniform([], 1.8, 2.2)
  flare_linear = tf.image.adjust_gamma(flare, gamma)

  # Remove DC background in flare.
  flare_linear = utils.remove_background(flare_linear)

  if apply_affine:
    rotation = tf.random.uniform([batch_size], minval=-math.pi, maxval=math.pi)
    shift = tf.random.normal([batch_size, 2], mean=0.0, stddev=10.0)
    shear = tf.random.uniform([batch_size, 2],
                              minval=-math.pi / 9,
                              maxval=math.pi / 9)
    scale = tf.random.uniform([batch_size, 2], minval=0.9, maxval=1.2)

    flare_linear = utils.apply_affine_transform(
        flare_linear,
        rotation=rotation,
        shift_x=shift[:, 0],
        shift_y=shift[:, 1],
        shear_x=shear[:, 0],
        shear_y=shear[:, 1],
        scale_x=scale[:, 0],
        scale_y=scale[:, 1])

  flare_linear = tf.clip_by_value(flare_linear, 0.0, 1.0)
  flare_linear = tf.image.crop_to_bounding_box(
      flare_linear,
      offset_height=(flare_input_height - training_res) // 2,
      offset_width=(flare_input_width - training_res) // 2,
      target_height=training_res,
      target_width=training_res)
  flare_linear = tf.image.random_flip_left_right(
      tf.image.random_flip_up_down(flare_linear))

  # First normalize the white balance. Then apply random white balance.
  flare_linear = utils.normalize_white_balance(flare_linear)
  rgb_gains = tf.random.uniform([3], 0, flare_max_gain, dtype=tf.float32)
  flare_linear *= rgb_gains

  # Further augmentation on flare patterns: random blur and DC offset.
  blur_size = tf.random.uniform([], 0.1, 3)
  flare_linear = utils.apply_blur(flare_linear, blur_size)
  offset = tf.random.uniform([], -0.02, 0.02)
  flare_linear = tf.clip_by_value(flare_linear + offset, 0.0, 1.0)

  flare_srgb = tf.image.adjust_gamma(flare_linear, 1.0 / gamma)

  # Scene augmentation: random crop and flips.
  scene_linear = tf.image.adjust_gamma(scene, gamma)
  scene_linear = tf.image.random_crop(scene_linear, flare_linear.shape)
  scene_linear = tf.image.random_flip_left_right(
      tf.image.random_flip_up_down(scene_linear))

  # Additive Gaussian noise. The Gaussian's variance is drawn from a Chi-squared
  # distribution. This is equivalent to drawing the Gaussian's standard
  # deviation from a truncated normal distribution, as shown below.
  sigma = tf.abs(tf.random.normal([], 0, noise))
  noise = tf.random.normal(scene_linear.shape, 0, sigma)
  scene_linear += noise

  # Random digital gain.
  gain = tf.random.uniform([], 0, 1.2)  # varying the intensity scale
  scene_linear = tf.clip_by_value(gain * scene_linear, 0.0, 1.0)

  scene_srgb = tf.image.adjust_gamma(scene_linear, 1.0 / gamma)

  # Combine the flare-free scene with a flare pattern to produce a synthetic
  # training example.
  combined_linear = scene_linear + flare_linear
  combined_srgb = tf.image.adjust_gamma(combined_linear, 1.0 / gamma)
  combined_srgb = tf.clip_by_value(combined_srgb, 0.0, 1.0)

  return (utils.quantize_8(scene_srgb), utils.quantize_8(flare_srgb),
          utils.quantize_8(combined_srgb), gamma)


def run_step(scene,
             flare,
             model,
             loss_fn,
             noise = 0.0,
             flare_max_gain = 10.0,
             flare_loss_weight = 0.0,
             training_res = 512):
  """Executes a forward step."""
  scene, flare, combined, gamma = add_flare(
      scene,
      flare,
      flare_max_gain=flare_max_gain,
      noise=noise,
      training_res=training_res)

  pred_scene = model(combined)
  pred_flare = utils.remove_flare(combined, pred_scene, gamma)

  flare_mask = utils.get_highlight_mask(flare)
  # Fill the saturation region with the ground truth, so that no L1/L2 loss
  # and better for perceptual loss since it matches the surrounding scenes.
  masked_scene = pred_scene * (1 - flare_mask) + scene * flare_mask
  loss_value = loss_fn(scene, masked_scene)
  if flare_loss_weight > 0:
    masked_flare = pred_flare * (1 - flare_mask) + flare * flare_mask
    loss_value += flare_loss_weight * loss_fn(flare, masked_flare)

  image_summary = tf.concat([combined, pred_scene, scene, pred_flare, flare],
                            axis=2)

  return loss_value, image_summary
