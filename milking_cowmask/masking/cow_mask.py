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

"""Cow mask generation."""
import math
import jax
import jax.numpy as jnp

_ROOT_2 = math.sqrt(2.0)
_ROOT_2_PI = math.sqrt(2.0 * math.pi)


def gaussian_kernels(sigmas, max_sigma):
  """Make Gaussian kernels for Gaussian blur.

  Args:
      sigmas: kernel sigmas as a [N] jax.numpy array
      max_sigma: sigma upper limit as a float (this is used to determine
        the size of kernel required to fit all kernels)

  Returns:
      a (N, kernel_width) jax.numpy array
  """
  sigmas = sigmas[:, None]
  size = round(max_sigma * 3) * 2 + 1
  x = jnp.arange(-size, size + 1)[None, :].astype(jnp.float32)
  y = jnp.exp(-0.5 * x ** 2 / sigmas ** 2)
  return y / (sigmas * _ROOT_2_PI)


def cow_masks(n_masks, mask_size, log_sigma_range, max_sigma,
              prop_range, rng_key):
  """Generate Cow Mask.

  Args:
      n_masks: number of masks to generate as an int
      mask_size: image size as a `(height, width)` tuple
      log_sigma_range: the range of the sigma (smoothing kernel)
          parameter in log-space`(log(sigma_min), log(sigma_max))`
      max_sigma: smoothing sigma upper limit
      prop_range: range from which to draw the proportion `p` that
        controls the proportion of pixel in a mask that are 1 vs 0
      rng_key: a `jax.random.PRNGKey`

  Returns:
      Cow Masks as a [v, height, width, 1] jax.numpy array
  """
  rng_k1, rng_k2 = jax.random.split(rng_key)
  rng_k2, rng_k3 = jax.random.split(rng_k2)

  # Draw the per-mask proportion p
  p = jax.random.uniform(
      rng_k1, (n_masks,), minval=prop_range[0], maxval=prop_range[1],
      dtype=jnp.float32)
  # Compute threshold factors
  threshold_factors = jax.scipy.special.erfinv(2 * p - 1) * _ROOT_2

  sigmas = jnp.exp(jax.random.uniform(
      rng_k2, (n_masks,), minval=log_sigma_range[0],
      maxval=log_sigma_range[1]))

  # Create initial noise with the batch and channel axes swapped so we can use
  # tf.nn.depthwise_conv2d to convolve it with the Gaussian kernels
  noise = jax.random.normal(rng_k3, (1,) + mask_size + (n_masks,))

  # Generate a kernel for each sigma
  kernels = gaussian_kernels(sigmas, max_sigma)
  # kernels: [batch, width] -> [width, batch]
  kernels = kernels.transpose((1, 0))
  # kernels in y and x
  krn_y = kernels[:, None, None, :]
  krn_x = kernels[None, :, None, :]

  # Apply kernels in y and x separately
  smooth_noise = jax.lax.conv_general_dilated(
      noise, krn_y, (1, 1), 'SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=n_masks)
  smooth_noise = jax.lax.conv_general_dilated(
      smooth_noise, krn_x, (1, 1), 'SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=n_masks)

  # [1, height, width, batch] -> [batch, height, width, 1]
  smooth_noise = smooth_noise.transpose((3, 1, 2, 0))

  # Compute mean and std-dev
  noise_mu = smooth_noise.mean(axis=(1, 2, 3), keepdims=True)
  noise_sigma = smooth_noise.std(axis=(1, 2, 3), keepdims=True)
  # Compute thresholds
  thresholds = threshold_factors[:, None, None, None] * noise_sigma + noise_mu
  # Apply threshold
  masks = (smooth_noise <= thresholds).astype(jnp.float32)
  return masks
