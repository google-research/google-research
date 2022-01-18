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

"""Noise generators."""
import numpy as np
from scipy import ndimage
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_kernel(size=3, bounds=3):
  """Create Gaussian kernel."""
  kernel_basis = np.linspace(-bounds, bounds, size+1)

  # Create gaussian kernel
  kernel_1d = np.diff(scipy.stats.norm.cdf(kernel_basis))
  kernel = np.outer(kernel_1d, kernel_1d)

  # Normalize kernel
  kernel = kernel / kernel.sum()

  # Reshape to dim for pytorch conv2d and repeat
  kernel = torch.tensor(kernel).float()
  kernel = kernel.reshape(1, 1, *kernel.size())
  kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
  return kernel


def add_gaussian_blur(x, k_size=3):
  """Add Gaussian blur to image.

  Adapted from
  https://github.com/kechan/FastaiPlayground/blob/master/Quick%20Tour%20of%20Data%20Augmentation.ipynb
  Args:
    x: source image.
    k_size: kernel size.

  Returns:
    x: Gaussian blurred image.
  """
  kernel = make_kernel(k_size)
  padding = (k_size - 1) // 2

  x = x.unsqueeze(dim=0)
  padded_x = F.pad(x, [padding] * x.dim(), mode='reflect')
  x = F.conv2d(padded_x, kernel, groups=3)
  return x.squeeze()


def add_patch(tensor,
              noise_location,
              patch_type=False,
              min_size=16,
              max_size=32):
  """Add focus/occluding patch."""
  _, h, w = tensor.shape
  if noise_location == 'random':
    w_size = np.random.randint(min_size, max_size+1)
    h_size = w_size
    x1 = np.random.randint(0, w - w_size + 1)
    y1 = np.random.randint(0, h - h_size + 1)
  elif noise_location == 'center':
    w_size = min_size
    h_size = min_size
    # Center
    x1 = (w - w_size) // 2
    y1 = (h - h_size) // 2

  x2 = x1 + w_size
  y2 = y1 + h_size

  if patch_type == 'focus':
    blured_tensor = add_gaussian_blur(tensor.clone())
    blured_tensor[:, y1:y2, x1:x2] = tensor[:, y1:y2, x1:x2]
    tensor = blured_tensor.clone()
  elif patch_type == 'occlusion':
    tensor[:, y1:y2, x1:x2] = 0
  else:
    assert False, f'{patch_type} not implemented!'
  return tensor


def pad_image(img, padding=32 * 2):
  """Pad image."""
  c, h, w = img.shape

  x1 = padding
  x2 = padding + w
  y1 = padding
  y2 = padding + h

  # Base
  x_padded = torch.zeros((c, h + padding * 2, w + padding * 2))
  # Left
  x_padded[:, y1:y2, :padding] = img[:, :, 0:1].repeat(1, 1, padding)
  # Right
  x_padded[:, y1:y2, x2:] = img[:, :, w - 1:w].repeat(1, 1, padding)
  # Top
  x_padded[:, :padding, x1:x2] = img[:, 0:1, :].repeat(1, padding, 1)
  # Bottom
  x_padded[:, y2:, x1:x2] = img[:, h - 1:h, :].repeat(1, padding, 1)
  # Top Left corner
  x_padded[:, :padding, :padding] = img[:, 0:1, 0:1].repeat(1, padding, padding)
  # Bottom left corner
  x_padded[:, y2:, :padding] = img[:, h - 1:h, 0:1].repeat(1, padding, padding)
  # Top right corner
  x_padded[:, :padding, x2:] = img[:, 0:1, w - 1:w].repeat(1, padding, padding)
  # Bottom right corner
  x_padded[:, y2:, x2:] = img[:, h - 1:h, w - 1:w].repeat(1, padding, padding)
  # Fill in source image
  x_padded[:, y1:y2, x1:x2] = img

  return x_padded, (x1, y1)


def crop_image(img, top_left, offset=(0, 0), dim=32):
  """Crop image."""
  _, h, w = img.shape
  x_offset, y_offset = offset
  x1, y1 = top_left

  x1 += x_offset
  x1 = min(max(x1, 0), w - dim)
  x2 = x1 + dim

  y1 += y_offset
  y1 = min(max(y1, 0), h - dim)
  y2 = y1 + dim
  return img[:, y1:y2, x1:x2]


def shift_image(img, shift_at_t, dim=32):
  """Shift image."""
  # Pad image
  padding = dim * 2
  padded_img, (x1, y1) = pad_image(img, padding=padding)

  # Crop with offset
  cropped_img = crop_image(padded_img,
                           top_left=(x1, y1),
                           offset=shift_at_t,
                           dim=dim)
  return cropped_img


def rotate_image(img, max_rot_angle, dim=32):
  """Rotate image."""
  # Pad image
  padding = int(dim * 1.5)
  padded_img, (x1, y1) = pad_image(img, padding=padding)

  # Rotate image
  rotation_deg = np.random.uniform(-max_rot_angle, max_rot_angle)
  x_np = padded_img.permute(1, 2, 0).numpy()
  x_np = ndimage.rotate(x_np, rotation_deg, reshape=False)
  rotated_img = torch.tensor(x_np).permute(2, 0, 1)

  # Crop image
  cropped_img = crop_image(rotated_img,
                           top_left=(x1, y1),
                           offset=(0, 0),
                           dim=dim)
  return cropped_img


def translate_image(img, shift_at_t, dim=32):
  """Translate image."""
  # Pad image
  padding = dim * 2
  padded_img, (x1, y1) = pad_image(img, padding=padding)

  # Crop with offset
  cropped_img = crop_image(padded_img,
                           top_left=(x1, y1),
                           offset=shift_at_t,
                           dim=dim)
  return cropped_img


def change_resolution(img):
  """Change resolution of image."""
  scale_factor = np.random.choice(list(range(0, 6, 2)))
  if scale_factor == 0:
    return img
  downsample = nn.AvgPool2d(scale_factor)
  upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
  new_res_img = upsample(downsample(img.unsqueeze(dim=1))).squeeze()
  return new_res_img


class RandomWalkGenerator:
  """Random walk handler."""

  def __init__(self, n_timesteps, n_total_samples):
    """Initializes Randon walk."""
    self.n_timesteps = n_timesteps if n_timesteps > 0 else 5
    self.n_total_samples = n_total_samples
    self._setup_random_walk()

  def _generate(self, max_vals=(8, 8), move_prob=(1, 1)):
    """Generate Randon walk."""
    init_loc = (0, 0)
    max_x, max_y = max_vals
    move_x_prob, move_y_prob = move_prob
    locations = [init_loc]
    for _ in range(self.n_timesteps - 1):
      prev_x, prev_y = locations[-1]
      new_x, new_y = prev_x, prev_y
      if np.random.uniform() < move_x_prob:
        new_x = prev_x + np.random.choice([-1, 1])
      if np.random.uniform() < move_y_prob:
        new_y = prev_y + np.random.choice([-1, 1])
      new_x = max(min(new_x, max_x), -max_x)
      new_y = max(min(new_y, max_y), -max_y)
      loc_i = (new_x, new_y)
      locations.append(loc_i)
    return locations

  def _setup_random_walk(self):
    self._sample_shift_schedules = [
        self._generate() for _ in range(self.n_total_samples)
    ]
    np.random.shuffle(self._sample_shift_schedules)

  def __call__(self, img, sample_i=None, t=None):
    if sample_i is None:
      sample_i = np.random.randint(len(self._sample_shift_schedules))
      n_ts = self._sample_shift_schedules[sample_i]
      t = np.random.randint(len(n_ts))

    shift_at_t = self._sample_shift_schedules[sample_i][t]
    noised_img = translate_image(img, shift_at_t)
    return noised_img


class PerlinNoise(object):
  """Perlin noise handler."""

  def __init__(self,
               half=False,
               half_dim='height',
               frequency=5,
               proportion=0.4,
               b_w=True):
    """Initializes PerlinNoise generator."""

    self.half = half
    self.half_dim = half_dim
    self.frequency = frequency
    self.proportion = proportion
    self.b_w = b_w

  def _perlin(self, x, y, seed=0):
    """Perlin noise."""
    def lerp(a, b, x):
      return a + x * (b - a)

    def fade(t):
      return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(h, x, y):
      vectors = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])
      g = vectors[h % 4].float()
      return g[:, :, 0] * x + g[:, :, 1] * y

    # permutation table
    np.random.seed(seed)

    p = torch.randperm(256)
    p = torch.stack([p, p]).flatten()

    # coordinates of the top-left
    xi = x.long()
    yi = y.long()

    # internal coordinates
    xf = x - xi.float()
    yf = y - yi.float()

    # fade factors
    u = fade(xf)
    v = fade(yf)

    x00 = p[p[xi]   + yi]
    x01 = p[p[xi]   + yi+1]
    x11 = p[p[xi+1] + yi+1]
    x10 = p[p[xi+1] + yi]

    n00 = gradient(x00, xf, yf)
    n01 = gradient(x01, xf, yf-1)
    n11 = gradient(x11, xf-1, yf-1)
    n10 = gradient(x10, xf-1, yf)

    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)

    return lerp(x1, x2, v)

  def _create_mask(self, dim, seed=None):
    """Create mask."""
    t_lin = torch.linspace(0, self.frequency, dim)
    y, x = torch.meshgrid([t_lin, t_lin])

    if seed is None:
      seed = np.random.randint(1, 1000000)

    mask = self._perlin(x, y, seed)

    if self.b_w:
      sorted_vals = np.sort(np.ndarray.flatten(mask.data.numpy()))
      idx = int(np.round(len(sorted_vals) * (1 - self.proportion)))
      threshold = sorted_vals[idx]
      mask = (mask < threshold)*1.0

    return mask

  def __call__(self, img):
    img_shape = img.shape

    mask = torch.zeros_like(img)
    dim = mask.shape[1]
    perlin_mask = self._create_mask(dim)
    for i in range(mask.shape[0]):
      mask[i] = perlin_mask

    if self.half:
      half = img_shape[1]//2
      if self.half_dim == 'height':
        mask[:, :half, :] = 1
      else:
        mask[:, :, :half] = 1

    noisy_image = img * mask

    return noisy_image


class FocusBlur:
  """Average Blurring noise handler."""

  def __init__(self):
    """Initializes averge blurring."""
    self._factor_step = 2
    self._max_factor = 6
    self.res_range = range(0, self._max_factor, self._factor_step)

  def __call__(self, img):
    scale_factor = np.random.choice(list(self.res_range))
    if scale_factor == 0:
      return img

    downsample_op = nn.AvgPool2d(scale_factor)
    upsample_op = nn.UpsamplingNearest2d(scale_factor=scale_factor)
    new_res_img = upsample_op(downsample_op(img.unsqueeze(dim=1))).squeeze()
    return new_res_img


class NoiseHandler:
  """Noise handler."""

  def __init__(self,
               noise_type,
               n_total_samples=1000,
               n_total_timesteps=0,
               n_timesteps_per_item=0,
               n_transition_steps=0):
    """Initializes noise handler."""
    self.noise_type = noise_type
    self.n_total_samples = n_total_samples
    self.n_total_timesteps = n_total_timesteps
    self.n_timesteps_per_item = n_timesteps_per_item
    self.n_transition_steps = n_transition_steps

    self._min_size = 16
    self._max_size = 16
    self._max_rot_angle = 60

    self._random_walker = None
    if noise_type == 'translation':
      self._random_walker = RandomWalkGenerator(n_total_timesteps,
                                                n_total_samples)

  def __call__(self, x_src, sample_i=None, t=None):
    x = x_src.clone()
    if self.noise_type in ['occlusion', 'focus']:
      x_noised = add_patch(x,
                           noise_location='random',
                           patch_type=self.noise_type,
                           min_size=self._min_size,
                           max_size=self._max_size)
    elif self.noise_type == 'resolution':
      x_noised = FocusBlur()(x)
    elif self.noise_type == 'Perlin':
      x_noised = PerlinNoise()(x)
    elif self.noise_type == 'translation':
      x_noised = self._random_walker(x, sample_i, t)
    elif self.noise_type == 'rotation':
      x_noised = rotate_image(x, max_rot_angle=self._max_rot_angle)

    return x_noised
