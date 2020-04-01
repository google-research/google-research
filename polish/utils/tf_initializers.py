# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Wrappers for different weight initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class Orthogonal(object):
  """Orthogonal initializer.

  Adopted from OpenAI implementation:
  https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py

  This implementation is simpler than the TF implementation. In order to prevent
  bringing any uncertainity into the program, we decided to use this
  initializer. Later, we can test our implementation with
  tf.keras.initializers.Orthogonal and deprecate this implementation.

  Attributes:
    _scale: Used to scale the initializer value.
  """

  def __init__(self, scale=1.0):
    self._scale = scale

  def __call__(self, shape, dtype=None, partition_info=None):
    """Function definition for an initializer."""
    del dtype  # unused, but needed for compatibility
    del partition_info  # unused, but needed for compatibility
    shape = tuple(shape)
    if len(shape) == 2:
      flat_shape = shape
    # Assumes NHWC format.
    # Flattens all the dimensions, but the last dimension (channel dimension).
    elif len(shape) == 4:
      flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
      raise NotImplementedError
    normal_vec = np.random.normal(0.0, 1.0, flat_shape)
    left_unitary_mat, _, right_unitary_mat = np.linalg.svd(
        normal_vec, full_matrices=False)
    # Pick the one with the correct shape
    if left_unitary_mat.shape == flat_shape:
      out_mat = left_unitary_mat
    else:
      out_mat = right_unitary_mat
    out_mat = out_mat.reshape(shape)
    return (self._scale * out_mat[:shape[0], :shape[1]]).astype(np.float32)
