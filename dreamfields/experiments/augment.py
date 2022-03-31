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

"""Data augmentation helpers with random crops and backgrounds, with PyTorch."""

# pylint: disable=g-bad-import-order
import functools

import jax
import numpy as np
import torch
import torchvision

import optvis
# pylint: enable=g-bad-import-order


cpu = torch.device("cpu")


def checkerboard(batch_size, nsq, size, device=cpu):
  """Create a checkerboard background image."""
  assert size % nsq == 0
  sq = size // nsq
  color1, color2 = torch.rand((2, batch_size, 1, 1, 1, 1, 3), device=device)
  canvas = color1.expand((batch_size, nsq, sq, nsq, sq, 3)).clone()
  canvas[:, ::2, :, 1::2, :] = color2
  canvas[:, 1::2, :, ::2, :] = color2
  canvas = canvas.reshape(batch_size, size, size, 3)
  canvas = canvas.movedim(-1, 1)
  return canvas


def fft_images(batch_size, size, device=cpu):
  """Random FFT backgrounds from dreamfields/dreamfields/optvis.py."""
  key = jax.random.PRNGKey(np.random.randint(1000000))
  keys = jax.random.split(key, batch_size)
  fn = functools.partial(
      optvis.image_sample, shape=[1, size, size, 3], sd=0.2, decay_power=1.5)
  bg = jax.vmap(fn)(keys)[:, 0]  # NHWC.
  bg = torch.from_numpy(np.asarray(bg)).to(device)
  bg = bg.movedim(-1, 1)  # NHWC to NCHW.
  return bg


def random_blur(image, min_blur, max_blur, kernel_size=15):
  blur_std = np.random.uniform() * (max_blur - min_blur) + min_blur
  image = torchvision.transforms.functional.gaussian_blur(
      image, kernel_size=kernel_size, sigma=blur_std)
  return image


def sample_backgrounds(num,
                       res,
                       *,
                       checkerboard_nsq,
                       min_blur_std,
                       max_blur_std,
                       device=cpu):
  """Sample random backgrounds."""
  per_type_num = int(np.ceil(num / 5))

  bgs = []
  # White background.
  bgs.append(torch.ones((per_type_num, 3, res, res)))
  # Black background.
  bgs.append(torch.zeros((per_type_num, 3, res, res)))
  # Randomly colored checkerboard.
  bgs.append(checkerboard(per_type_num, checkerboard_nsq, res))
  # Random noise.
  bgs.append(torch.rand((per_type_num, 3, res, res)))
  # FFT background.
  bgs.append(fft_images(per_type_num, res))
  bgs = torch.cat(bgs, dim=0)

  # Sample and blur backgrounds.
  idx = np.random.choice(len(bgs), size=num, replace=False)
  bgs = torch.stack(
      [random_blur(bgs[i], min_blur_std, max_blur_std) for i in idx])

  bgs = bgs.to(device)
  return bgs
