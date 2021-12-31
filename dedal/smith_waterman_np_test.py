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

"""Tests for the NumPy ops."""

import itertools
from typing import Callable
from absl.testing import parameterized

import numpy as np
from scipy import special

import tensorflow as tf

from dedal import alignment
from dedal import smith_waterman_np as npy_ops


# For test purposes.
def _sw_general(sim_mat,
                gap_func):
  """Computes soft Smith-Waterman with general gap function.

  Args:
    sim_mat: a np.ndarray<float>[len1, len2] containing
     the substitution values for pairs of sequences.
    gap_func: a function of the form gap_func(k), where k is an integer.

  Returns:
    Smith-Waterman value (float).
  """
  len_1, len_2 = sim_mat.shape

  match = np.zeros((len_1 + 1, len_2 + 1))

  for i in range(1, len_1 + 1):
    for j in range(1, len_2 + 1):
      tmp = match[i - 1, j - 1] + sim_mat[i-1, j-1]
      delete = max([match[i - k, j] - gap_func(k) for k in range(1, i + 1)])
      insert = max([match[i, j - k] - gap_func(k) for k in range(1, j + 1)])
      match[i, j] = max(tmp, delete, insert, 0)

  return max(match[1:, 1:].ravel())


def scores_brute_force(weights):
  len_1, len_2, _ = weights.shape
  ret = []
  for alignment_mat in npy_ops.alignment_matrices(len_1, len_2):
    ret.append(np.vdot(alignment_mat, weights))
  return np.array(ret)


def max_brute_force(weights):
  scores = scores_brute_force(weights)
  return np.max(scores)


def lse_brute_force(weights, temperature=1.0):
  scores = scores_brute_force(weights)
  return temperature * special.logsumexp(scores / temperature)


class SmithWatermanNumpyTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    single_sub = 6*tf.eye(5) - 5*tf.ones((5, 5))
    second_sub = tf.tensor_scatter_nd_update(single_sub,
                                             indices=[[0, 0], [1, 1]],
                                             updates=[-5, -5])
    third_sub = tf.tensor_scatter_nd_update(single_sub,
                                            indices=[[0, 0], [2, 2]],
                                            updates=[-5, -5])
    fourth_sub = tf.tensor_scatter_nd_update(single_sub,
                                             indices=[[0, 0], [4, 4]],
                                             updates=[-5, -5])
    self._toy_sub = tf.stack([single_sub, second_sub, third_sub, fourth_sub])
    self._toy_gap_open = 0.03 * tf.ones((self._toy_sub.shape[0],))
    self._toy_gap_extend = 0.02 * tf.ones((self._toy_sub.shape[0],))
    self._weights = alignment.weights_from_sim_mat(self._toy_sub,
                                                   self._toy_gap_open,
                                                   self._toy_gap_extend)

  def test_max_against_brute_force(self):
    maxes = [max_brute_force(self._weights[i])
             for i in range(self._weights.shape[0])]
    maxes2 = npy_ops.soft_sw_affine(self._toy_sub.numpy(),
                                    self._toy_gap_open.numpy(),
                                    self._toy_gap_extend.numpy(),
                                    temperature=0)
    self.assertAllClose(maxes, maxes2)

  def test_max_against_sw_general(self):
    def make_gap_func(gap_open, gap_extend):
      def gap_func(k):
        return gap_open + gap_extend * (k-1)
      return gap_func

    toy_sub = self._toy_sub.numpy()
    toy_gap_open = self._toy_gap_open.numpy()
    toy_gap_extend = self._toy_gap_extend.numpy()

    maxes = [_sw_general(toy_sub[i],
                         make_gap_func(toy_gap_open[i], toy_gap_extend[i]))
             for i in range(self._weights.shape[0])]
    maxes2 = npy_ops.soft_sw_affine(toy_sub, toy_gap_open, toy_gap_extend,
                                    temperature=0)
    self.assertAllClose(maxes, maxes2)

  def test_lse_against_brute_force(self):
    for temp in (1e-3, 1.0, 1e3):
      softmaxes = np.array([lse_brute_force(self._weights[i],
                                            temperature=temp)
                            for i in range(self._weights.shape[0])])
      softmaxes2 = npy_ops.soft_sw_affine(self._toy_sub.numpy(),
                                          self._toy_gap_open.numpy(),
                                          self._toy_gap_extend.numpy(),
                                          temperature=temp)
      self.assertAllClose(softmaxes, softmaxes2)

  @parameterized.parameters(itertools.product([0, 0.1, 1, 10], range(4)))
  def test_gradients_against_finite_difference(self, temp=1.0, j=0):
    def num_grad(f, x, eps=1e-4):
      shape = x.shape
      x_flat = x.ravel()
      grad_flat = np.zeros_like(x_flat)
      for j in range(len(x_flat)):
        e = np.zeros_like(x_flat)
        e[j] = 1
        call1 = f((x_flat + eps * e).reshape(shape))
        call2 = f((x_flat - eps * e).reshape(shape))
        grad_flat[j] = (call1 - call2) / (2 * eps)
      return grad_flat.reshape(shape)

    toy_sub = self._toy_sub.numpy()
    toy_gap_open = self._toy_gap_open.numpy()
    toy_gap_extend = self._toy_gap_extend.numpy()

    _, g_sim_mat, g_gap_open, g_gap_extend = (
        npy_ops._soft_sw_affine(sim_mat=toy_sub[j], gap_open=toy_gap_open[j],
                                gap_extend=toy_gap_extend[j], temperature=temp,
                                ret_grads=True))
    # Gradient w.r.t. sim_mat.
    def f_sim_mat(x):
      return npy_ops._soft_sw_affine(sim_mat=x, gap_open=toy_gap_open[j],
                                     gap_extend=toy_gap_extend[j],
                                     temperature=temp, ret_grads=False)

    g_sim_mat_num = num_grad(f_sim_mat, toy_sub[j].astype(np.float64))
    self.assertAllClose(g_sim_mat, g_sim_mat_num)

    # Gradient w.r.t. gap_open.
    def f_gap_open(x):
      return npy_ops._soft_sw_affine(sim_mat=toy_sub[j], gap_open=x,
                                     gap_extend=toy_gap_extend[j],
                                     temperature=temp, ret_grads=False)

    g_gap_open_num = num_grad(f_gap_open, toy_gap_open[j].astype(np.float64))
    self.assertAllClose(g_gap_open, g_gap_open_num)

    # Gradient w.r.t. gap_extend.
    def f_gap_ext(x):
      return npy_ops._soft_sw_affine(sim_mat=toy_sub[j],
                                     gap_open=toy_gap_open[j], gap_extend=x,
                                     temperature=temp, ret_grads=False)

    g_gap_extend_num = num_grad(f_gap_ext, toy_gap_extend[j].astype(np.float64))
    self.assertAllClose(g_gap_extend, g_gap_extend_num)

  def test_gradient_shape(self):
    toy_sub = self._toy_sub.numpy()
    toy_gap_open = self._toy_gap_open.numpy()
    toy_gap_extend = self._toy_gap_extend.numpy()

    values, g_sim_mat, g_gap_open, g_gap_extend = npy_ops.soft_sw_affine(
        sim_mat=toy_sub,
        gap_open=toy_gap_open,
        gap_extend=toy_gap_extend,
        ret_grads=True)
    self.assertEqual(values.shape[0], toy_sub.shape[0])
    self.assertAllEqual(g_sim_mat.shape, toy_sub.shape)
    self.assertAllEqual(g_gap_open.shape, toy_gap_open.shape)
    self.assertAllEqual(g_gap_extend.shape, toy_gap_extend.shape)


if __name__ == '__main__':
  tf.test.main()
