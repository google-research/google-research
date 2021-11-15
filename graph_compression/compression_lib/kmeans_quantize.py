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

# Lint as: python3
"""Functions for k-means quantization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import flags
from absl import logging
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq

FLAGS = flags.FLAGS


def prune(matrix, fraction=0.5):
  """Prune a matrix based on magnitude of the absolute value of the entries.

  Args:
    matrix: input matrix, a 2d numpy array
    fraction: nonnegative float number between 0 and 1.0

  Returns:
    a pruned matrix with bottom `fraction` entries set to zero.
  """
  matrix_shape = matrix.shape
  matrix = np.reshape(matrix, np.product(matrix_shape))
  sorted_indexes = np.argsort(np.absolute(matrix))
  bottom = sorted_indexes[:int(math.ceil(sorted_indexes.size * fraction))]
  for i in bottom:
    matrix[i] = 0
  return np.reshape(matrix, matrix_shape)


def kmeans_quantize_block(matrix,
                          levels=8,
                          pruning_factor=0.0,
                          block_size=16,
                          is_padded=True,
                          seed=42):
  """Perform kmeans quantization.

  View 1 x block_size blocks of the matrix as vectors in a vector space and
  perform kmeans quantization on the matrix.

  Optionally perform magnitude base pruning.

  Args:
    matrix: input matrix, a Numpy 2d array;
    levels: number of kmeans centroids, int, default is 8;
    pruning_factor: fraction of entries to be pruned, no pruning is performed if
      pruning_factor equals 0.0, float, between 0 and 1, default is 0.0;
    block_size: block size to be used for kmeans quantization, int, must divide
      matrix.shape[1], default is 16;
    is_padded: whether to pad the number of kmeans centroids, if True, zero
      vectors will be added to codebook so that the number of centroids will be
      exactly `levels`, boolean, default is True;
    seed: random seed, int. If None, no seed is set. Default value is 42.

  Returns:
    codebook: a matrix whose rows are the kmeans centroid vectors, if
              `is_padded` is True, the codebook.shape[0] = levels, otherwise,
              codebook.shape[0] = number of kmeans centroid the kmeans algorithm
              returns, a Numpy 2d array;
    encoding: an array whose entries are the indices of centroids in `codebook`
              that are the closest to the rows of `matrix_in_block`, a Numpy
              1d array.
  """
  if pruning_factor < 0:
    logging.warning('kmeans_quantize_block: pruning_factor is less than 0, '
                    'round up to 0.')
    pruning_factor = 0.0
  if pruning_factor > 1:
    logging.warning('kmeans_quantize_block: pruning_factor is greater than 1, '
                    'round down to 1.')
    pruning_factor = 1.0
  if matrix.shape[1] % block_size != 0:
    raise ValueError('kmeans_quantize_block: block_size doesn\'t divide '
                     'matrix.shape[1].')
  logging.info('In kmeans_quantize_block, levels is %d, pruning_factor is %f.',
               levels, pruning_factor)

  if seed:
    np.random.seed(seed)

  matrix_in_block = np.reshape(matrix, (-1, block_size))
  if pruning_factor > 0:
    matrix_in_block = prune(matrix_in_block, pruning_factor)

  codebook, _ = kmeans(matrix_in_block, levels, thresh=1e-05)
  encoding, dist = vq(matrix_in_block, codebook)

  # Zero blocks should stay as zero. Here we look up the codebook item that is
  # closest to the zero block and set the item to be zero.
  zero_block = np.zeros((1, block_size))
  zero_encoding, _ = vq(zero_block, codebook)
  zero_index = zero_encoding[0]
  codebook[zero_index, :] = zero_block[:]

  logging.info(
      'kmeans_quantize_block: %d levels, mean square distance is %.4f.', levels,
      np.sqrt(np.mean(dist**2)))
  logging.info('matrix_in_block = %s, codebook = %s, encoding = %s.',
               matrix_in_block, repr(codebook), encoding)

  if is_padded:
    padded_codebook = np.zeros((levels, block_size))
    padded_codebook[:codebook.shape[0], :] = codebook
    codebook = padded_codebook

  reshaped_encoding = np.reshape(encoding, (matrix.shape[0], -1))
  return [codebook, reshaped_encoding]
