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

"""Direct JAX port of some helpers in lucid.optvis.

Ported from https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/
"""

import flax.linen as nn
import jax
from jax import random
import jax.numpy as np
import numpy as onp


# Constants from lucid/optvis/param/color.py
color_correlation_svd_sqrt = onp.asarray(
    [[0.26, 0.09, 0.02],
     [0.27, 0.00, -0.05],
     [0.27, -0.09, 0.03]]).astype("float32")
max_norm_svd_sqrt = onp.max(onp.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_mean = [0.48, 0.46, 0.41]


def _linear_correlate_color(t):
  """Multiply input by sqrt of empirical (ImageNet) color correlation matrix.

  If you interpret t's innermost dimension as describing colors in a
  decorrelated version of the color space (which is a very natural way to
  describe colors -- see discussion in Feature Visualization article) the way
  to map back to normal colors is multiply the square root of your color
  correlations.

  Args:
    t: input whitened color array, with trailing dimension 3.

  Returns:
    t_correlated: RGB color array.
  """
  assert t.shape[-1] == 3
  t_flat = np.reshape(t, [-1, 3])
  color_correlation_normalized = (
      color_correlation_svd_sqrt / max_norm_svd_sqrt)
  t_flat = np.matmul(t_flat, color_correlation_normalized.T)
  t_correlated = np.reshape(t_flat, t.shape)
  return t_correlated


def constrain_l_inf(x):
  # NOTE(jainajay): does not use custom grad unlike Lucid
  return x / np.maximum(1.0, np.abs(x))


def to_valid_rgb(t, decorrelated=False, sigmoid=True):
  """Transform inner dimension of t to valid rgb colors.

  In practice this consists of two parts:
  (1) If requested, transform the colors from a decorrelated color space to RGB.
  (2) Constrain the color channels to be in [0,1], either using a sigmoid
      function or clipping.

  Args:
    t: Input tensor, trailing dimension will be interpreted as colors and
      transformed/constrained.
    decorrelated: If True, the input tensor's colors are interpreted as coming
      from a whitened space.
    sigmoid: If True, the colors are constrained elementwise using sigmoid. If
      False, colors are constrained by clipping infinity norm.

  Returns:
    t with the innermost dimension transformed.
  """
  if decorrelated:
    t = _linear_correlate_color(t)
  if decorrelated and not sigmoid:
    t += color_mean

  if sigmoid:
    return nn.sigmoid(t)

  return constrain_l_inf(2 * t - 1) / 2 + 0.5


def rfft2d_freqs(h, w):
  """Computes 2D spectrum frequencies."""
  fy = np.fft.fftfreq(h)[:, None]
  # when we have an odd input dimension we need to keep one additional
  # frequency and later cut off 1 pixel
  fx = np.fft.fftfreq(w)[:w // 2 + 1 + w % 2]
  return np.sqrt(fx * fx + fy * fy)


def rand_fft_image(key, shape, sd=None, decay_power=1):
  """Generate a random background."""
  b, h, w, ch = shape
  sd = 0.01 if sd is None else sd

  imgs = []
  for _ in range(b):
    freqs = rfft2d_freqs(h, w)
    fh, fw = freqs.shape
    spectrum_var = sd * random.normal(key, [2, ch, fh, fw], dtype=np.float32)
    spectrum = jax.lax.complex(spectrum_var[0], spectrum_var[1])
    spectrum_scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w))**decay_power
    # Scale the spectrum by the square-root of the number of pixels
    # to get a unitary transformation. This allows to use similar
    # learning rates to pixel-wise optimisation.
    spectrum_scale *= np.sqrt(w * h)
    scaled_spectrum = spectrum * spectrum_scale
    # img = tf.signal.irfft2d(scaled_spectrum)
    img = np.fft.irfft2(scaled_spectrum)
    # in case of odd input dimension we cut off the additional pixel
    # we get from irfft2d length computation
    img = img[:ch, :h, :w]
    img = np.transpose(img, [1, 2, 0])
    imgs.append(img)
  return np.stack(imgs) / 4.0


def image_sample(key, shape, decorrelated=True, sd=None, decay_power=1):
  raw_spatial = rand_fft_image(key, shape, sd=sd, decay_power=decay_power)
  return to_valid_rgb(raw_spatial, decorrelated=decorrelated)
