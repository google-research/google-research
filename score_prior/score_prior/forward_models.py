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

"""A library of forward models with associated log-likelihood functions."""

import abc
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float  # pylint:disable=g-multiple-import
import numpy as np
import scipy.linalg


class IndependentGaussianLikelihood(abc.ABC):
  """Likelihood module for inverse problems w/ independent Gaussian noise.

  An inverse problem with independent Gaussian noise is one whose
  forward model is given by:
    y = f(x) + noise, noise ~ N(0, diag(sigmas))

  Abstract methods should take mini-batched inputs.
  """

  @property
  @abc.abstractmethod
  def sigmas(self):
    """Std. dev. of the noise for each measured channel."""

  def invert_measurement(self,
                         y):
    """Invert measurement to a naive estimate of `x`. Optional method."""

  @abc.abstractmethod
  def apply_forward_operator(self,
                             x):
    """Apply noiseless forward operator to `x`: A*x.

    Must return measurements `y` as a flattened vector.

    Args:
      x: Source image(s).
    """

  def get_measurement(self,
                      rng,
                      x):
    """Apply forward operator to `x` and add noise."""
    # Draw Gaussian noise.
    y_dim = len(self.sigmas)
    noise = jax.random.normal(rng, (len(x), y_dim)) * self.sigmas

    y = self.apply_forward_operator(x) + noise
    return y

  def unnormalized_log_likelihood(self,
                                  x,
                                  y):
    """Unnormalized log p(y|x)."""
    residual = y - self.apply_forward_operator(x)
    log_llh = -0.5 * jnp.sum(jnp.square(residual / self.sigmas), axis=-1)
    return log_llh

  def likelihood_score(self,
                       x,
                       y):
    """Gradient of log p(y|x) with respect to `x`."""
    # `jax.grad` only takes scalar-valued functions, so we make this wrapper
    # around `self.unnormalized_log_likelihood`.
    def grad_fn(x_sample, y_sample):
      return self.unnormalized_log_likelihood(
          x_sample[None, Ellipsis], y_sample[None, Ellipsis])[0]
    return jax.vmap(jax.grad(grad_fn))(x, y)


def get_isotropic_dft_comps(n_freqs_per_orientation,
                            image_size):
  """Get the rows of the 2D DFT matrix that correspond to observed frequencies.

  Args:
    n_freqs_per_orientation: The number of lowest spatial frequencies
      that are observable either the horizontal or vertical direction.
    image_size: Height = width of the image.

  Returns:
    A 1D array containing the rows of the DFT matrix that correspond to the
      measured DFT components, assuming we can only measure up to
      `n_freqs_per_orientation` in each direction.
  """
  horizontal_dft_comps = np.arange(n_freqs_per_orientation)
  vertical_dft_comps = np.arange(n_freqs_per_orientation)
  dft_comps_2d = np.array(
      np.meshgrid(vertical_dft_comps, horizontal_dft_comps)).T.reshape(-1, 2)
  return np.ravel_multi_index(dft_comps_2d.T, (image_size, image_size))


def get_dft_matrix(image_size, dft_comps):
  """Returns the DFT operator matrix.

  Args:
    image_size: Height = width the image.
    dft_comps: A 1D array containing the indices of the rows of the full DFT
      matrix to keep.

  Returns:
    A 2D array representing the DFT measurement matrix, where only the rows
      corresponding to `dft_comps` are kept. The first half of the rows
      corresponds to the real part of the measurements, while the second
      half of the rows corresponds to the imaginary part.
  """
  dft_matrix_1d = scipy.linalg.dft(image_size)
  dft_matrix = np.kron(dft_matrix_1d, dft_matrix_1d)
  dft_matrix = dft_matrix[dft_comps]
  # Split matrix into real and imaginary submatrices.
  dft_matrix_expanded = jnp.concatenate(
      (dft_matrix.real, dft_matrix.imag), axis=0)
  return dft_matrix_expanded


class Deblurring(IndependentGaussianLikelihood):
  """Deblurring, where we observe low-frequency DFT measurements."""

  def __init__(self,
               n_freqs_per_direction,
               sigmas,
               image_shape):
    """Initialize `CompressedSensing` module.

    Args:
      n_freqs_per_direction: The number of lowest DFT components measured in
        each direction (horizontal and vertical).
      sigmas: A 1D array of the noise std. dev. for each measurement dimension.
      image_shape: The shape of one image: (image_size, image_size, n_channels).
    """
    super().__init__()
    self.n_dft = n_freqs_per_direction
    dft_comps = get_isotropic_dft_comps(n_freqs_per_direction, image_shape[0])
    self.dft_matrix = get_dft_matrix(image_shape[0], dft_comps)

    # Assume the noise level is the same for real and imaginary parts and for
    # each color channel. For a complex Gaussian random variable with std. dev.
    # `sigma`, the real and imaginary parts are independently Gaussian with
    # std. dev. `sigma / sqrt(2)`.
    self.real_and_imag_sigmas = np.tile(
        sigmas / np.sqrt(2), (2 * image_shape[-1]))
    self.image_shape = image_shape

  def apply_forward_operator(self, x):
    """Take subset of DFT of mini-batch `x`."""
    dft = jnp.fft.fft2(x, axes=(1, 2))
    dft = dft[:, :self.n_dft, :self.n_dft]
    measurement = dft.reshape(len(x), -1)
    measurement = jnp.concatenate((measurement.real, measurement.imag), axis=1)
    return measurement

  @property
  def sigmas(self):
    return self.real_and_imag_sigmas

  def invert_measurement(self, y):
    """Zero-fill higher DFT components and perform inverse FFT."""
    y_dim = len(self.sigmas)
    dft = y[:, :y_dim // 2] + 1j * y[:, y_dim // 2:]
    dft = dft.reshape(y.shape[0], self.n_dft, self.n_dft, -1)
    dft_zero_filled = jnp.pad(dft,
                              ((0, 0), (0, self.image_shape[0] - self.n_dft),
                               (0, self.image_shape[1] - self.n_dft), (0, 0)))
    x_recon = jnp.fft.ifft2(dft_zero_filled, axes=(1, 2))
    return x_recon.real


class Denoising(IndependentGaussianLikelihood):
  """Denoising images with iid Gaussian noise."""

  def __init__(self, scale, image_shape):
    super().__init__()
    self.scale = scale
    self.image_shape = image_shape

  @property
  def sigmas(self):
    dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
    return np.ones(dim) * self.scale

  def apply_forward_operator(self, x):
    """Identity."""
    return x.reshape(len(x), -1)

  def invert_measurement(self, y):
    """Identity."""
    return y.reshape(len(y), *self.image_shape)


class EHT(IndependentGaussianLikelihood):
  """EHT measurements with closure phase. Assumes grayscale, square images."""

  def __init__(self,
               forward_matrix,
               sigmas,
               image_size):
    """Initialize `EHT` module.

    Args:
      forward_matrix: The measurement matrix (complex-valued) for EHT
        observations.
      sigmas: The noise std. dev. (real-valued) for each measurement.
      image_size: The image height = width.
    """
    self.forward_matrix = forward_matrix
    self.forward_matrix_expanded = jnp.concatenate(
        (forward_matrix.real, forward_matrix.imag), axis=0)
    self.noise_sigmas = sigmas
    # Note: `inverse_matrix` should only be used for visualizing a naive
    # reconstruction. Since `forward_matrix` might be ill-conditioned, taking
    # the pseudo-inverse of it might not be a good idea.
    self.inverse_matrix = jnp.linalg.pinv(self.forward_matrix_expanded)
    self.image_size = image_size
    # Assume the noise level is the same for real and imaginary parts.
    self.real_and_imag_sigmas = jnp.concatenate((sigmas, sigmas))

  @property
  def sigmas(self):
    return self.real_and_imag_sigmas

  def apply_forward_operator(self, x):
    return jnp.einsum(
        'ij,bj->bi', self.forward_matrix_expanded, x.reshape(len(x), -1))

  def invert_measurement(self, y):
    x = jnp.einsum('ij,bj->bi', self.inverse_matrix, y)
    return x.reshape(len(y), self.image_size, self.image_size, 1).real
