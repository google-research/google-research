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

# pylint: skip-file
import jax
import jax.numpy as np
from jax import random
import jax.ops

# physics


def laplacian(array, step=1.0):
  """Finite difference approx of the Laplacian operator in 1D or 2D."""
  if array.ndim == 1:
    kernel = np.array([1, -2, 1])
  elif array.ndim == 2:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
  else:
    raise NotImplementedError
  lhs = array[np.newaxis, np.newaxis, Ellipsis]
  rhs = kernel[np.newaxis, np.newaxis, Ellipsis] / step**2
  result = jax.lax.conv(
      lhs, rhs, window_strides=(1,) * array.ndim, padding='SAME')
  squeezed = np.squeeze(result, axis=(0, 1))
  return squeezed


# to make rectangles
def make_mask(size, aspect_ratio):
  h = 1.0 / (size + 1)
  n_y = (np.floor(aspect_ratio / h) - 1).astype(int)
  ind = np.arange(size * size).reshape(size, size)
  mask = np.where(ind >= size * n_y, 0, 1)
  return mask


def make_mask_dual(size, aspect_ratio):
  return 1- make_mask(size, aspect_ratio)


# to make L-shapes
def make_mask_L(size, *kwargs):
  n_y = (np.floor((size + 1) / 2) - 1).astype(int)
  ind1 = np.arange(size * size).reshape(size, size)
  ind2 = ind1.T
  mask = np.where(((ind1 >= size * n_y) & (ind2 >= size * n_y)), 0, 1)
  return mask


def make_mask_L_dual(size, *kwargs):
  return 1- make_mask_L(size)


def helmholtz(array,
              k,
              step=1.0,
              aspect_ratio=1.0,
              mask_f=make_mask,
              mask_f_dual=make_mask_dual):
  """Finite difference approx of the helmholtz operator in 2D."""
  if array.ndim == 2:
    kernel = np.array([[0, 1, 0], [1, -4 + np.sign(k) * k**2 * step**2, 1],
                       [0, 1, 0]])
  else:
    raise NotImplementedError
  mask = mask_f(array.shape[0], aspect_ratio)
  array_masked = np.multiply(array, mask)
  mask_dual = mask_f_dual(array.shape[0], aspect_ratio)
  arr2 = np.multiply(array, mask_dual)
  lhs = array_masked[np.newaxis, np.newaxis, Ellipsis]
  rhs = kernel[np.newaxis, np.newaxis, Ellipsis] / step**2
  result = jax.lax.conv(
      lhs, rhs, window_strides=(1,) * array.ndim, padding='SAME')
  squeezed = np.squeeze(result, axis=(0, 1))
  squeezed = np.multiply(squeezed, mask)
  return squeezed + arr2


def num_row(size, aspect_ratio):
  h = 1.0 / (size + 1)
  n_y = (np.floor(aspect_ratio / h) - 1).astype(int)
  return n_y
