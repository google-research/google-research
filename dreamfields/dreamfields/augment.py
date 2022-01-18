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

"""Data augmentation helpers with random crops and backgrounds."""

from . import optvis

import dm_pix
import jax
from jax import random
import jax.numpy as np
import numpy as onp


def crop(image, ix, iy, n_crop):
  """Crop image to a n_crop sized square with the top left at (ix, iy)."""
  return np.roll(np.roll(image, -ix, axis=-2), -iy, axis=-3)[:n_crop, :n_crop]


def resize(image, n_clip):
  """Resize image bilinearly to size n_clip."""
  return jax.image.resize(image, (n_clip, n_clip, 3), method='bilinear')


def checkerboard(key, nsq, size, dtype=np.float32):
  """Create a checkerboard background image with random colors.

  NOTE: only supports a single value for nsq (number squares).

  Args:
    key: JAX PRNGkey.
    nsq (int): number of squares per side of the checkerboard.
    size (int): size of one side of the checkerboard in pixels.
    dtype: desired return data type.

  Returns:
    canvas (np.array): checkerboard background image.
  """
  assert size % nsq == 0
  sq = size // nsq
  color1, color2 = random.uniform(key, (2, 3), dtype=dtype)
  canvas = np.full((nsq, sq, nsq, sq, 3), color1, dtype=dtype)
  canvas = jax.ops.index_update(canvas, jax.ops.index[::2, :, 1::2, :, :],
                                color2)
  canvas = jax.ops.index_update(canvas, jax.ops.index[1::2, :, ::2, :, :],
                                color2)
  return canvas.reshape(sq * nsq, sq * nsq, 3)


def augment_rendering(config, z, acc, rng):
  """Augment rendering z and acc map.

  Args:
    config (ml_collections.ConfigDict): configuration for the run.
    z (np.ndarray): rendered image.
    acc (np.ndarray): opacity.
    rng: JAX PRNGkey

  Returns:
    crop (np.ndarray): augmented views of z.
    crop_acc (np.ndarray): augmented views of acc.
  """
  n_crop = config.crop_width
  n_clip = config.clip_width
  key, rng = random.split(rng)

  def random_bg(key):
    """Generate and augment background."""
    noise_key, checker_key, fft_key, bg_sel_key, blur_key, bg_saturation_key = (
        random.split(key, 6))
    # Uniform noise background.
    noise_bg = random.uniform(noise_key, (n_clip, n_clip, 3))

    # Checkerboard background.
    checker_bg = checkerboard(checker_key, config.checker_bg_nsq, n_clip)

    # Random smoothed gaussian bg used in
    # https://distill.pub/2018/differentiable-parameterizations/#section-rgba
    fft_bg = optvis.image_sample(
        fft_key, [1, n_clip, n_clip, 3], sd=0.2, decay_power=1.5)[0]

    # Select background.
    probs = [config.noise_bg_prob, config.checker_bg_prob, config.fft_bg_prob]
    assert onp.isclose(sum(probs), 1)
    bgs = np.stack([noise_bg, checker_bg, fft_bg])
    bg = random.choice(bg_sel_key, bgs, p=np.array(probs))

    # Blur background.
    if config.get('bg_blur_std_range', None):
      min_blur, max_blur = config.bg_blur_std_range
      blur_std = random.uniform(blur_key) * (max_blur - min_blur) + min_blur
      bg = dm_pix.gaussian_blur(bg, blur_std, kernel_size=15)

    # (de)saturate background. values < 1 indicate desaturation (grayscale)
    if config.get('bg_random_saturation_range', None):
      lower, upper = config.bg_random_saturation_range
      bg = dm_pix.random_saturation(bg_saturation_key, bg, lower, upper)

    return bg

  def aug_fg(key):
    """augment, crop, and resize foreground."""
    saturation_key, crop_key = random.split(key, 2)
    fg = z

    if config.get('fg_random_saturation_range', None):
      lower, upper = config.fg_random_saturation_range
      fg = dm_pix.random_saturation(saturation_key, fg, lower, upper)

    # Crop and resize.
    ix, iy = random.randint(crop_key, (2,), 0, fg.shape[-3] - n_crop)
    fg = crop(fg, ix, iy, n_crop)
    fg = resize(fg, n_clip)

    acc_crop = crop(acc, ix, iy, n_crop)
    acc_crop = resize(acc_crop, n_clip)
    if config.get('min_aug_acc', 0.):
      acc_crop = np.clip(acc_crop, config.min_aug_acc)

    return fg, acc_crop

  def random_aug(key):
    fg_key, bg_key = random.split(key, 2)

    fg, acc_crop = aug_fg(fg_key)
    if not config.augment_backgrounds:
      return fg

    bg = random_bg(bg_key)

    # Composite background.
    return fg * acc_crop + bg * (1 - acc_crop)

  return jax.vmap(random_aug)(random.split(key, config.n_local_aug))
