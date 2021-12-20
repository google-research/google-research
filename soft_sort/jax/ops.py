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
"""Jax implementation of soft sort operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from soft_sort.jax import soft_quantilizer


DIRECTIONS = ('ASCENDING', 'DESCENDING')


def _preprocess(x, axis):
  """Reshapes the input data to make it rank 2 as required by SoftQuantilizer.

  The SoftQuantilizer expects an input tensor of rank 2, where the first
  dimension is the batch dimension and the soft sorting is applied on the second
  one.

  Args:
   x: np.ndarray<float> of any dimension.
   axis: (int) the axis to be turned into the second dimension.

  Returns:
   a np.ndarray<float>[batch, n] where n is the dimensions over the axis and
   batch the product of all other dimensions.
  """
  dims = list(range(len(x.shape)))
  dims[-1], dims[axis] = dims[axis], dims[-1]
  z = np.transpose(x, dims) if dims[axis] != dims[-1] else x
  return np.reshape(z, (-1, x.shape[axis]))


def _postprocess(x, shape, axis):
  """Applies the inverse transformation of _preprocess.

  Args:
   x: np.ndarray<float>[batch, n]
   shape: TensorShape of the desired output.
   axis: (int) the axis along which the original tensor was processed.

  Returns:
   A np.ndarray<float> with the shape given in argument.
  """
  s = list(shape)
  s[axis], s[-1] = s[-1], s[axis]
  z = np.reshape(x, s)

  # Transpose to get back to the original shape
  dims = list(range(len(shape)))
  dims[-1], dims[axis] = dims[axis], dims[-1]
  return np.transpose(z, dims) if dims[axis] != dims[-1] else z


def softsort(x, direction='ASCENDING', axis=-1, **kwargs):
  """Applies the softsort operator on input tensor x.

  This operator acts as differentiable alternative to tf.sort.

  Args:
   x: the input np.ndarray. It can be either of shape [batch, n] or [n].
   direction: the direction 'ASCENDING' or 'DESCENDING'
   axis: the axis on which to operate the sort.
   **kwargs: see SoftQuantilizer for possible parameters.

  Returns:
   A np.ndarray of the same shape as the input.
  """
  if direction not in DIRECTIONS:
    raise ValueError('`direction` should be one of {}'.format(DIRECTIONS))

  x = np.array(x)
  z = _preprocess(x, axis)
  descending = (direction == 'DESCENDING')
  sorter = soft_quantilizer.SoftQuantilizer(z, descending=descending, **kwargs)

  # In case we are applying some quantization while sorting, the number of
  # outputs should be the number of targets.
  shape = list(x.shape)
  shape[axis] = sorter.target_weights.shape[1]
  return _postprocess(sorter.softsort, shape, axis)


def softranks(x, direction='ASCENDING', axis=-1, zero_based=True, **kwargs):
  """A differentiable argsort-like operator that returns directly the ranks.

  Note that it behaves as the 'inverse' of the argsort operator since it returns
  soft ranks, i.e. real numbers that play the role of indices and quantify the
  relative standing (among all n entries) of each entry of x.

  Args:
   x: np.ndarray<float> of any shape.
   direction: (str) either 'ASCENDING' or 'DESCENDING', as in tf.sort.
   axis: (int) the axis along which to sort, as in tf.sort.
   zero_based: (bool) to return values in [0, n-1] or in [1, n].
   **kwargs: see SoftQuantilizer for possible parameters.

  Returns:
   A np.ndarray<float> of the same shape as the input containing the soft ranks.
  """
  if direction not in DIRECTIONS:
    raise ValueError('`direction` should be one of {}'.format(DIRECTIONS))

  x = np.array(x)
  descending = (direction == 'DESCENDING')
  z = _preprocess(x, axis)
  sorter = soft_quantilizer.SoftQuantilizer(z, descending=descending, **kwargs)
  ranks = sorter.softcdf * z.shape[1]
  if zero_based:
    ranks -= 1

  return _postprocess(ranks, x.shape, axis)


def softquantile(x, quantile, quantile_width=0.05, axis=-1, **kwargs):
  """Computes soft quantiles via optimal transport.

  This operator takes advantage of the fact that an exhaustive softsort is not
  required to recover a single quantile. Instead, one can transport all
  input values in x onto only 3 weighted values. Target weights are adjusted so
  that those values in x that are transported to the middle value in the target
  vector y correspond to those concentrating around the quantile of interest.

  This idea generalizes to more quantiles, interleaving small weights on the
  quantile indices and bigger weights in between, corresponding to the gap from
  one desired quantile to the next one.

  Args:
   x: np.ndarray<float> of any shape.
   quantile: (float) the quantile to be returned.
   quantile_width: (float) mass given to the bucket supposed to attract points
    whose value concentrate around the desired quantile value. Bigger width
    means that we allow the soft quantile to be a mixture of
    more points further away from the quantile. If None, the width is set at 1/n
    where n is the number of values considered (the size along the 'axis').
   axis: (int) the axis along which to compute the quantile.
   **kwargs: see SoftQuantilizer for possible extra parameters.

  Returns:
    A np.ndarray<float> similar to the input tensor, but without the axis
    dimension that is squeezed into a single value: its soft quantile.
  """
  target_weights = [quantile - 0.5 * quantile_width,
                    quantile_width,
                    1.0 - quantile - 0.5 * quantile_width]
  x = np.array(x)
  z = _preprocess(x, axis=axis)
  sorter = soft_quantilizer.SoftQuantilizer(
      z, target_weights=target_weights, **kwargs)
  shape = list(x.shape)
  shape.pop(axis)
  return np.reshape(sorter.softsort[:, 1], shape)
