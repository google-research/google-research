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

"""Data augmentation helpers with random crops and backgrounds."""

# pylint: disable=g-bad-import-order
import dm_pix
import jax
from jax import random
import jax.numpy as np
import numpy as onp
import torch

import optvis
# pylint: enable=g-bad-import-order


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
  canvas = canvas.at[::2, :, 1::2, :, :].set(color2)
  canvas = canvas.at[1::2, :, ::2, :, :].set(color2)
  return canvas.reshape(sq * nsq, sq * nsq, 3)


def random_bg(key,
              n_clip,
              checker_bg_nsq,
              white_bg_prob=0.25,
              noise_bg_prob=0.25,
              checker_bg_prob=0.25,
              fft_bg_prob=0.25,
              bg_blur_std_range=(0, 10),
              bg_random_saturation_range=None):
  """Generate and augment background."""
  noise_key, checker_key, fft_key, bg_sel_key, blur_key, bg_saturation_key = (
      random.split(key, 6))
  # White background.
  white_bg = np.ones((n_clip, n_clip, 3), dtype=np.float32)

  # Uniform noise background.
  noise_bg = random.uniform(noise_key, (n_clip, n_clip, 3))

  # Checkerboard background.
  checker_bg = checkerboard(checker_key, checker_bg_nsq, n_clip)

  # Random smoothed gaussian bg used in
  # https://distill.pub/2018/differentiable-parameterizations/#section-rgba.
  fft_bg = optvis.image_sample(
      fft_key, [1, n_clip, n_clip, 3], sd=0.2, decay_power=1.5)[0]

  # Select background.
  probs = [white_bg_prob, noise_bg_prob, checker_bg_prob, fft_bg_prob]
  assert onp.isclose(sum(probs), 1)
  bgs = np.stack([white_bg, noise_bg, checker_bg, fft_bg])
  bg = random.choice(bg_sel_key, bgs, p=np.array(probs))

  # Blur background.
  if bg_blur_std_range is not None:
    min_blur, max_blur = bg_blur_std_range
    blur_std = random.uniform(blur_key) * (max_blur - min_blur) + min_blur
    bg = dm_pix.gaussian_blur(bg, blur_std, kernel_size=15)

  # (de)saturate background. values < 1 indicate desaturation (grayscale).
  if bg_random_saturation_range is not None:
    lower, upper = bg_random_saturation_range
    bg = dm_pix.random_saturation(bg_saturation_key, bg, lower, upper)

  return bg


def sample_backgrounds(num, res, checkerboard_nsq, min_blur_std, max_blur_std,
                       device):
  """Generate random background images."""
  bgs = []
  key = random.PRNGKey(onp.random.randint(0, 100000000))
  keys = random.split(key, num)
  for key in keys:
    bg = random_bg(
        key,
        res,
        checkerboard_nsq,
        bg_blur_std_range=(min_blur_std, max_blur_std))
    bgs.append(bg)

  # Stack and convert to torch.
  bgs = onp.array(np.stack(bgs))  # [num, res, res, 3].
  bgs = torch.from_numpy(bgs).movedim(3, 1).to(device)  # [num, 3, res, res].
  return bgs
