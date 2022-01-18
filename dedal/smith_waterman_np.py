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

"""Smith-Waterman functions for protein alignment in NumPy."""

from typing import Optional

import numpy as np
from scipy import special


def _alignment_matrices(len_1, len_2, i = 0, j = 0,
                        curr = "M", prev = "S",
                        alignment = None):
  """Helper function for alignment_matrices."""
  if alignment is None:
    alignment = np.zeros((len_1, len_2, 9))

  # M=match, X=gap_x, Y=gap_y
  lookup = {
      ("S", "M"): 0,
      ("M", "M"): 1,
      ("X", "M"): 2,
      ("Y", "M"): 3,
      ("M", "X"): 4,
      ("X", "X"): 5,
      ("M", "Y"): 6,
      ("X", "Y"): 7,
      ("Y", "Y"): 8,
  }

  alignment[i, j, lookup[(prev, curr)]] = 1
  if curr == "M":
    yield alignment

  if i < len_1 - 1:
    # We can go down.
    yield from _alignment_matrices(len_1, len_2, i=i+1, j=j, curr="Y",
                                   prev=curr, alignment=alignment.copy())

  if i < len_1 - 1 and j < len_2 - 1:
    # We can go in diagonal.
    yield from _alignment_matrices(len_1, len_2, i=i+1, j=j+1, curr="M",
                                   prev=curr, alignment=alignment.copy())
  if j < len_2 - 1 and curr != "Y":
    # We can go right.
    yield from _alignment_matrices(len_1, len_2, i=i, j=j+1,
                                   curr="X", prev=curr,
                                   alignment=alignment.copy())


def alignment_matrices(len_1, len_2):
  """Generator of all alignment matrices of shape (len_1, len_2, 9).

  Args:
    len_1: length of first sequence.
    len_2: length of second sequence.

  Yields:
    All alignment matrices of shape (len_1, len_2, 9)
  """
  # Iterates over all possible starting states.
  for i in range(len_1):
    for j in range(len_2):
      yield from _alignment_matrices(len_1, len_2, i=i, j=j)


def _make_op(temperature=1.0):
  """Make softmax + softargmax operator."""
  def op(*args):
    if len(args) == 1:  # op(arr)
      arr = np.array(args[0])
    else:  # lse_op(ele1, ele2, ...)
      arr = np.array(args)
    if temperature > 0:
      return (temperature * special.logsumexp(arr / temperature),
              special.softmax(arr / temperature))
    else:
      ret = np.zeros_like(arr)
      ret[np.argmax(arr)] = 1
      return np.max(arr), ret
  return op


def _soft_sw_affine(sim_mat,
                    gap_open,
                    gap_extend,
                    temperature = 1.0,
                    ret_grads = False):
  """Computes soft Smith-Waterman with affine gaps.

  Args:
    sim_mat: a np.ndarray<float>[len1, len2] the substitution
     values for pairs of sequences.
    gap_open: float penalty in the substitution values for opening a gap.
    gap_extend: float of penalty in the substitution values for extending a gap.
    temperature: float controlling the regularization strength.
    ret_grads: whether to return gradients or not.

  Returns:
    value if ret_grads is False
    value, g_sim_mat, g_gap_open, g_gap_extend if ret_grads is True

    value = float of softmax values
    g_sim_mat = np.ndarray<float>[len_1, len_2]
    g_gap_open = float
    g_gap_extend = float
  """
  len_1, len_2 = sim_mat.shape

  match = np.zeros((len_1 + 1, len_2 + 1))
  match_p = np.zeros((len_1 + 2, len_2 + 2, 4))

  gap_x = np.zeros((len_1 + 1, len_2 + 1))
  gap_x_p = np.zeros((len_1 + 2, len_2 + 2, 2))

  gap_y = np.zeros((len_1 + 1, len_2 + 1))
  gap_y_p = np.zeros((len_1 + 2, len_2 + 2, 3))

  float_max = np.finfo(np.float32).max

  if temperature > 0:
    for mat in (match, gap_x, gap_y):
      mat[0, :] = mat[:, 0] = -float_max

  op = _make_op(temperature=temperature)

  for i in range(1, len_1 + 1):
    for j in range(1, len_2 + 1):
      match[i, j], match_p[i, j] = op(0, match[i-1, j-1],
                                      gap_x[i-1, j-1], gap_y[i-1, j-1])
      match[i, j] += sim_mat[i-1, j-1]
      gap_x[i, j], gap_x_p[i, j] = op(match[i, j-1] - gap_open,
                                      gap_x[i, j-1] - gap_extend)
      gap_y[i, j], gap_y_p[i, j] = op(match[i-1, j] - gap_open,
                                      gap_x[i-1, j] - gap_open,
                                      gap_y[i-1, j] - gap_extend)

  value, probas = op(match.ravel())
  probas = probas.reshape(match.shape)

  if not ret_grads:
    return value

  match_e = np.zeros((len_1 + 2, len_2 + 2))
  gap_x_e = np.zeros((len_1 + 2, len_2 + 2))
  gap_y_e = np.zeros((len_1 + 2, len_2 + 2))

  for j in reversed(range(1, len_2 + 1)):
    for i in reversed(range(1, len_1 + 1)):
      gap_y_e[i, j] = (match_e[i+1, j+1] * match_p[i+1, j+1, 3] +
                       gap_y_e[i+1, j] * gap_y_p[i+1, j, 2])

      gap_x_e[i, j] = (match_e[i+1, j+1] * match_p[i+1, j+1, 2] +
                       gap_x_e[i, j+1] * gap_x_p[i, j+1, 1] +
                       gap_y_e[i+1, j] * gap_y_p[i+1, j, 1])

      match_e[i, j] = (match_e[i+1, j+1] * match_p[i+1, j+1, 1] +
                       gap_x_e[i, j+1] * gap_x_p[i, j+1, 0] +
                       gap_y_e[i+1, j] * gap_y_p[i+1, j, 0] +
                       probas[i, j])

  g_sim_mat = np.zeros_like(sim_mat)
  g_gap_open = np.zeros_like(sim_mat)
  g_gap_extend = np.zeros_like(sim_mat)
  for i in range(1, len_1 + 1):
    for j in range(1, len_2 + 1):
      g_sim_mat[i-1, j-1] = match_e[i, j]
      g_gap_open[i-1, j-1] = (gap_x_e[i, j+1] * (-gap_x_p[i, j+1, 0]) +
                              gap_y_e[i+1, j] * (-gap_y_p[i+1, j, 0] -
                                                 gap_y_p[i+1, j, 1]))
      g_gap_extend[i-1, j-1] = (gap_x_e[i, j+1] * (-gap_x_p[i, j+1, 1]) +
                                gap_y_e[i+1, j] * (-gap_y_p[i+1, j, 2]))

  return value, g_sim_mat, np.sum(g_gap_open), np.sum(g_gap_extend)


def soft_sw_affine(sim_mat,
                   gap_open,
                   gap_extend,
                   temperature = 1.0,
                   ret_grads = False):
  """Computes soft Smith-Waterman with affine gaps.

  Args:
    sim_mat: a np.ndarray<float>[batch, len1, len2] the substitution
     values for pairs of sequences.
    gap_open: a np.ndarray<float>[batch] of penalty in the substitution values
     for opening a gap.
    gap_extend: a np.ndarray<float>[batch] of penalty in the substitution values
     for extending a gap.
    temperature: float controlling the regularization strength.
    ret_grads: whether to return gradients or not.

  Returns:
    values if ret_grads is False
    values, g_sim_mat, g_gap_open, g_gap_extend if ret_grads is True

    values = np.ndarray<float>[batch] of softmax values
    g_sim_mat = np.ndarray<float>[batch, len_1, len_2]
    g_gap_open = np.ndarray<float>[batch]
    g_gap_extend = np.ndarray<float>[batch]
  """
  # TODO(mblondel): avoid naive for loop.
  arr = [_soft_sw_affine(sim_mat[i], gap_open[i], gap_extend[i],
                         temperature=temperature, ret_grads=ret_grads)
         for i in range(sim_mat.shape[0])]
  if ret_grads:
    return tuple(np.array(elements) for elements in zip(*arr))
  else:
    return np.array(arr)
