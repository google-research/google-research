# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Data preprocessing and augmentation."""
# pylint: skip-file

import functools
from typing import Any, Mapping, Text

import jax
import jax.numpy as jnp

# typing
JaxBatch = Mapping[Text, jnp.ndarray]
ConfigDict = Mapping[Text, Any]

strength = 1.0
augment_config = dict(
    view1=dict(
        random_flip=True,  # Random left/right flip
        color_transform=dict(
            apply_prob=1.0,
            # Range of jittering
            brightness=0.4 * strength,
            contrast=0.4 * strength,
            saturation=0.2 * strength,
            hue=0.1 * strength,
            # Probability of applying color jittering
            color_jitter_prob=0.8,
            # Probability of converting to grayscale
            to_grayscale_prob=0.2,
            # Shuffle the order of color transforms
            shuffle=True),
        gaussian_blur=dict(
            apply_prob=1.0,
            # apply_prob=0.0,
            # Kernel size ~ image_size / blur_divider
            blur_divider=10.,
            # Kernel distribution
            sigma_min=0.1,
            sigma_max=2.0),
        solarize=dict(apply_prob=0.0, threshold=0.5),
    ),
    view2=dict(
        random_flip=True,
        color_transform=dict(apply_prob=1.0,
                             brightness=0.4 * strength,
                             contrast=0.4 * strength,
                             saturation=0.2 * strength,
                             hue=0.1 * strength,
                             color_jitter_prob=0.8,
                             to_grayscale_prob=0.2,
                             shuffle=True),
        gaussian_blur=dict(
            # apply_prob=0.0,
            apply_prob=0.1,
            blur_divider=10.,
            sigma_min=0.1,
            sigma_max=2.0),
        solarize=dict(apply_prob=0.2, threshold=0.5),
    ))


def postprocess(inputs, rng, add_gaussian_blur=True):
  """Apply the image augmentations to crops in inputs (view1 and view2)."""

  def _postprocess_image(
      images,
      rng,
      presets,
  ):
    """Applies augmentations in post-processing.

    Args:
      images: an NHWC tensor (with C=3), with float values in [0, 1].
      rng: a single PRNGKey.
      presets: a dict of presets for the augmentations.

    Returns:
      A batch of augmented images with shape NHWC, with keys view1, view2
      and labels.
    """
    flip_rng, color_rng, blur_rng, solarize_rng = jax.random.split(rng, 4)
    out = images
    if presets['random_flip']:
      out = random_flip(out, flip_rng)
    if presets['color_transform']['apply_prob'] > 0:
      out = color_transform(out, color_rng, **presets['color_transform'])
    if presets['gaussian_blur']['apply_prob'] > 0 and add_gaussian_blur:
      out = gaussian_blur(out, blur_rng, **presets['gaussian_blur'])
    if presets['solarize']['apply_prob'] > 0:
      out = solarize(out, solarize_rng, **presets['solarize'])
    out = jnp.clip(out, 0., 1.)
    return jax.lax.stop_gradient(out)

  rng1, rng2 = jax.random.split(rng, num=2)
  view1 = _postprocess_image(inputs['view1'], rng1, augment_config['view1'])
  view2 = _postprocess_image(inputs['view2'], rng2, augment_config['view2'])
  result = dict(view1=view1, view2=view2)
  for k in inputs:
    if k not in result:
      result[k] = inputs[k]
  return result


def postprocess1(inputs, rng, add_gaussian_blur=True):
  """Apply the image augmentations to crops in inputs (view1 and view2)."""

  def _postprocess_image(
      images,
      rng,
      presets,
  ):
    """Applies augmentations in post-processing.

    Args:
      images: an NHWC tensor (with C=3), with float values in [0, 1].
      rng: a single PRNGKey.
      presets: a dict of presets for the augmentations.

    Returns:
      A batch of augmented images with shape NHWC, with keys view1, view2
      and labels.
    """
    flip_rng, color_rng, blur_rng, solarize_rng = jax.random.split(rng, 4)
    out = images
    if presets['random_flip']:
      out = random_flip(out, flip_rng)
    if presets['color_transform']['apply_prob'] > 0:
      out = color_transform(out, color_rng, **presets['color_transform'])
    if presets['gaussian_blur']['apply_prob'] > 0 and add_gaussian_blur:
      out = gaussian_blur(out, blur_rng, **presets['gaussian_blur'])
    if presets['solarize']['apply_prob'] > 0:
      out = solarize(out, solarize_rng, **presets['solarize'])
    out = jnp.clip(out, 0., 1.)
    return jax.lax.stop_gradient(out)

  image = _postprocess_image(inputs['image'], rng, augment_config['view1'])
  return dict(image=image, label=inputs['label'])


def _maybe_apply(apply_fn, inputs, rng, apply_prob):
  should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
  return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)


def _depthwise_conv2d(inputs, kernel, strides, padding):
  """Computes a depthwise conv2d in Jax.

  Args:
    inputs: an NHWC tensor with N=1.
    kernel: a [H", W", 1, C] tensor.
    strides: a 2d tensor.
    padding: "SAME" or "VALID".

  Returns:
    The depthwise convolution of inputs with kernel, as [H, W, C].
  """
  return jax.lax.conv_general_dilated(inputs,
                                      kernel,
                                      strides,
                                      padding,
                                      feature_group_count=inputs.shape[-1],
                                      dimension_numbers=('NHWC', 'HWIO',
                                                         'NHWC'))


def _gaussian_blur_single_image(image, kernel_size, padding, sigma):
  """Applies gaussian blur to a single image, given as NHWC with N=1."""
  radius = int(kernel_size / 2)
  kernel_size_ = 2 * radius + 1
  x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
  blur_filter = jnp.exp(-x**2 / (2. * sigma**2))
  blur_filter = blur_filter / jnp.sum(blur_filter)
  blur_v = jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1])
  blur_h = jnp.reshape(blur_filter, [1, kernel_size_, 1, 1])
  num_channels = image.shape[-1]
  blur_h = jnp.tile(blur_h, [1, 1, 1, num_channels])
  blur_v = jnp.tile(blur_v, [1, 1, 1, num_channels])
  expand_batch_dim = len(image.shape) == 3
  if expand_batch_dim:
    image = image[jnp.newaxis, Ellipsis]
  blurred = _depthwise_conv2d(image, blur_h, strides=[1, 1], padding=padding)
  blurred = _depthwise_conv2d(blurred, blur_v, strides=[1, 1], padding=padding)
  blurred = jnp.squeeze(blurred, axis=0)
  return blurred


def _random_gaussian_blur(image, rng, kernel_size, padding, sigma_min,
                          sigma_max, apply_prob):
  """Applies a random gaussian blur."""
  apply_rng, transform_rng = jax.random.split(rng)

  def _apply(image):
    sigma_rng, = jax.random.split(transform_rng, 1)
    sigma = jax.random.uniform(sigma_rng,
                               shape=(),
                               minval=sigma_min,
                               maxval=sigma_max,
                               dtype=jnp.float32)
    return _gaussian_blur_single_image(image, kernel_size, padding, sigma)

  return _maybe_apply(_apply, image, apply_rng, apply_prob)


def rgb_to_hsv(r, g, b):
  """Converts R, G, B  values to H, S, V values.

  Reference TF implementation:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
  Only input values between 0 and 1 are guaranteed to work properly, but this
  function complies with the TF implementation outside of this range.

  Args:
    r: A tensor representing the red color component as floats.
    g: A tensor representing the green color component as floats.
    b: A tensor representing the blue color component as floats.

  Returns:
    H, S, V values, each as tensors of shape [...] (same as the input without
    the last dimension).
  """
  vv = jnp.maximum(jnp.maximum(r, g), b)
  range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
  sat = jnp.where(vv > 0, range_ / vv, 0.)
  norm = jnp.where(range_ != 0, 1. / (6. * range_), 1e9)

  hr = norm * (g - b)
  hg = norm * (b - r) + 2. / 6.
  hb = norm * (r - g) + 4. / 6.

  hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
  hue = hue * (range_ > 0)
  hue = hue + (hue < 0)

  return hue, sat, vv


def hsv_to_rgb(h, s, v):
  """Converts H, S, V values to an R, G, B tuple.

  Reference TF implementation:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/adjust_saturation_op.cc
  Only input values between 0 and 1 are guaranteed to work properly, but this
  function complies with the TF implementation outside of this range.

  Args:
    h: A float tensor of arbitrary shape for the hue (0-1 values).
    s: A float tensor of the same shape for the saturation (0-1 values).
    v: A float tensor of the same shape for the value channel (0-1 values).

  Returns:
    An (r, g, b) tuple, each with the same dimension as the inputs.
  """
  c = s * v
  m = v - c
  dh = (h % 1.) * 6.
  fmodu = dh % 2.
  x = c * (1 - jnp.abs(fmodu - 1))
  hcat = jnp.floor(dh).astype(jnp.int32)
  rr = jnp.where(
      (hcat == 0) | (hcat == 5), c, jnp.where(
          (hcat == 1) | (hcat == 4), x, 0)) + m
  gg = jnp.where(
      (hcat == 1) | (hcat == 2), c, jnp.where(
          (hcat == 0) | (hcat == 3), x, 0)) + m
  bb = jnp.where(
      (hcat == 3) | (hcat == 4), c, jnp.where(
          (hcat == 2) | (hcat == 5), x, 0)) + m
  return rr, gg, bb


def adjust_brightness(rgb_tuple, delta):
  return jax.tree.map(lambda x: x + delta, rgb_tuple)


def adjust_contrast(image, factor):

  def _adjust_contrast_channel(channel):
    mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
    return factor * (channel - mean) + mean

  return jax.tree.map(_adjust_contrast_channel, image)


def adjust_saturation(h, s, v, factor):
  return h, jnp.clip(s * factor, 0., 1.), v


def adjust_hue(h, s, v, delta):
  # Note: this method exactly matches TF"s adjust_hue (combined with the hsv/rgb
  # conversions) when running on GPU. When running on CPU, the results will be
  # different if all RGB values for a pixel are outside of the [0, 1] range.
  return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, rng, max_delta):
  delta = jax.random.uniform(rng,
                             shape=(),
                             minval=-max_delta,
                             maxval=max_delta)
  return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, rng, max_delta):
  factor = jax.random.uniform(rng,
                              shape=(),
                              minval=1 - max_delta,
                              maxval=1 + max_delta)
  return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, rng, max_delta):
  h, s, v = rgb_to_hsv(*rgb_tuple)
  factor = jax.random.uniform(rng,
                              shape=(),
                              minval=1 - max_delta,
                              maxval=1 + max_delta)
  return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, rng, max_delta):
  h, s, v = rgb_to_hsv(*rgb_tuple)
  delta = jax.random.uniform(rng,
                             shape=(),
                             minval=-max_delta,
                             maxval=max_delta)
  return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
  rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
  grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[Ellipsis,
                                                               jnp.newaxis]
  return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.


def _color_transform_single_image(image, rng, brightness, contrast, saturation,
                                  hue, to_grayscale_prob, color_jitter_prob,
                                  apply_prob, shuffle):
  """Applies color jittering to a single image."""
  apply_rng, transform_rng = jax.random.split(rng)
  perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
      transform_rng, 7)

  # Whether the transform should be applied at all.
  should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
  # Whether to apply grayscale transform.
  should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
  # Whether to apply color jittering.
  should_apply_color = jax.random.uniform(cj_rng,
                                          shape=()) <= color_jitter_prob

  # Decorator to conditionally apply fn based on an index.
  def _make_cond(fn, idx):

    def identity_fn(x, unused_rng, unused_param):
      return x

    def cond_fn(args, i):

      def clip(args):
        return jax.tree.map(lambda arg: jnp.clip(arg, 0., 1.), args)

      out = jax.lax.cond(should_apply & should_apply_color & (i == idx), args,
                         lambda a: clip(fn(*a)), args,
                         lambda a: identity_fn(*a))
      return jax.lax.stop_gradient(out)

    return cond_fn

  random_brightness_cond = _make_cond(_random_brightness, idx=0)
  random_contrast_cond = _make_cond(_random_contrast, idx=1)
  random_saturation_cond = _make_cond(_random_saturation, idx=2)
  random_hue_cond = _make_cond(_random_hue, idx=3)

  def _color_jitter(x):
    rgb_tuple = tuple(jax.tree.map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
    if shuffle:
      order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
    else:
      order = range(4)
    for idx in order:
      if brightness > 0:
        rgb_tuple = random_brightness_cond((rgb_tuple, b_rng, brightness), idx)
      if contrast > 0:
        rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
      if saturation > 0:
        rgb_tuple = random_saturation_cond((rgb_tuple, s_rng, saturation), idx)
      if hue > 0:
        rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
    return jnp.stack(rgb_tuple, axis=-1)

  out_apply = _color_jitter(image)
  out_apply = jax.lax.cond(should_apply & should_apply_gs, out_apply,
                           _to_grayscale, out_apply, lambda x: x)
  return jnp.clip(out_apply, 0., 1.)


def _random_flip_single_image(image, rng):
  _, flip_rng = jax.random.split(rng)
  should_flip_lr = jax.random.uniform(flip_rng, shape=()) <= 0.5
  image = jax.lax.cond(should_flip_lr, image, jnp.fliplr, image, lambda x: x)
  return image


def random_flip(images, rng):
  rngs = jax.random.split(rng, images.shape[0])
  return jax.vmap(_random_flip_single_image)(images, rngs)


def color_transform(images,
                    rng,
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2,
                    color_jitter_prob=0.8,
                    to_grayscale_prob=0.2,
                    apply_prob=1.0,
                    shuffle=True):
  """Applies color jittering and/or grayscaling to a batch of images.

  Args:
    images: an NHWC tensor, with C=3.
    rng: a single PRNGKey.
    brightness: the range of jitter on brightness.
    contrast: the range of jitter on contrast.
    saturation: the range of jitter on saturation.
    hue: the range of jitter on hue.
    color_jitter_prob: the probability of applying color jittering.
    to_grayscale_prob: the probability of converting the image to grayscale.
    apply_prob: the probability of applying the transform to a batch element.
    shuffle: whether to apply the transforms in a random order.

  Returns:
    A NHWC tensor of the transformed images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  jitter_fn = functools.partial(_color_transform_single_image,
                                brightness=brightness,
                                contrast=contrast,
                                saturation=saturation,
                                hue=hue,
                                color_jitter_prob=color_jitter_prob,
                                to_grayscale_prob=to_grayscale_prob,
                                apply_prob=apply_prob,
                                shuffle=shuffle)
  return jax.vmap(jitter_fn)(images, rngs)


def gaussian_blur(images,
                  rng,
                  blur_divider=10.,
                  sigma_min=0.1,
                  sigma_max=2.0,
                  apply_prob=1.0):
  """Applies gaussian blur to a batch of images.

  Args:
    images: an NHWC tensor, with C=3.
    rng: a single PRNGKey.
    blur_divider: the blurring kernel will have size H / blur_divider.
    sigma_min: the minimum value for sigma in the blurring kernel.
    sigma_max: the maximum value for sigma in the blurring kernel.
    apply_prob: the probability of applying the transform to a batch element.

  Returns:
    A NHWC tensor of the blurred images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  kernel_size = images.shape[1] / blur_divider
  blur_fn = functools.partial(_random_gaussian_blur,
                              kernel_size=kernel_size,
                              padding='SAME',
                              sigma_min=sigma_min,
                              sigma_max=sigma_max,
                              apply_prob=apply_prob)
  return jax.vmap(blur_fn)(images, rngs)


def _solarize_single_image(image, rng, threshold, apply_prob):

  def _apply(image):
    return jnp.where(image < threshold, image, 1. - image)

  return _maybe_apply(_apply, image, rng, apply_prob)


def solarize(images, rng, threshold=0.5, apply_prob=1.0):
  """Applies solarization.

  Args:
    images: an NHWC tensor (with C=3).
    rng: a single PRNGKey.
    threshold: the solarization threshold.
    apply_prob: the probability of applying the transform to a batch element.

  Returns:
    A NHWC tensor of the transformed images.
  """
  rngs = jax.random.split(rng, images.shape[0])
  solarize_fn = functools.partial(_solarize_single_image,
                                  threshold=threshold,
                                  apply_prob=apply_prob)
  return jax.vmap(solarize_fn)(images, rngs)
