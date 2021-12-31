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

"""Helper functions for frequency perturbations of images.

Based on https://arxiv.org/abs/1906.08988.
"""

import numpy as np


def _get_symmetric_pos(dims, pos):
  """Compute the symmetric position of the point in 2D FFT.

  Args:
    dims: a tuple of 2 positive integers, dimensions of the 2D array.
    pos: a tuple of 2 integers, coordinate of the query point.
  Returns:
    a numpy array of shape [2], the coordinate of the symmetric point of the
      query point.
  """
  x = np.array(dims)
  p = np.array(pos)
  return np.where(np.mod(x, 2) == 0, np.mod(x - p, x), x - 1 - p)


def get_fourier_basis_image(i, j, x=32, y=32):
  """Compute the (i,j) spatial frequency basis vector over an x by y grid."""
  freq = np.zeros([x, y], dtype=np.complex64)
  sym = _get_symmetric_pos((x, y), (i, j))
  sym_i = sym[0]
  sym_j = sym[1]
  if (sym_i, sym_j) == (i, j):
    freq[i, j] = 1.0
  else:
    freq[i, j] = 0.5 + 0.5j
    freq[sym_i, sym_j] = 0.5 - 0.5j
  basis = np.fft.ifft2(np.fft.ifftshift(freq))
  basis = np.sqrt(x * y) * np.real(basis)
  # Repeat in the three color channels for a grayscale image
  basis = np.concatenate((basis[Ellipsis, None], basis[Ellipsis, None], basis[Ellipsis, None]),
                         axis=2)
  return basis


def get_spatial_freqij(freq_norm, unscaledi=1, unscaledj=1, dim=32):
  """Find spatial frequencies i, j with desired spatial frequency norm."""
  original_norm = np.sqrt(unscaledi**2 + unscaledj**2)
  scaledi = int(unscaledi * freq_norm / original_norm)
  scaledj = int(unscaledj * freq_norm / original_norm)
  scaledi = min(dim-1, scaledi + dim//2)
  scaledj = min(dim-1, scaledj + dim//2)
  return scaledi, scaledj


def get_fourier_composite_image(kind='1/f', dim=32):
  """Sum all the Fourier basis images with weights proportional to 1/sqrt(f)."""
  image = np.zeros((dim, dim, 3))
  for i in range(dim):
    for j in range(dim):
      basis_img = get_fourier_basis_image(i, j)
      # Get the spatial frequency norm, then apply the weighting and sum
      square_norm = max(1, (i - dim//2)**2 + (j - dim//2)**2)
      if kind == '1/f':
        weight = 1.0 / square_norm
      elif kind == 'f':
        weight = square_norm
      else:
        weight = 1
      image = image + weight * basis_img
  return image





