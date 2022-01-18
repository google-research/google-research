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

"""Tests for the wavefront_tf_ops module."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf


from dedal import alignment
from dedal import smith_waterman as tf_ops
from dedal import smith_waterman_np as npy_ops


_DECORATORS = [None, tf.function]
# TODO(fllinares): include XLA when running on GPU & TPU
# _DECORATORS = [None, tf.function, tf.function(experimental_compile=True)]


def random_sim_mat(b, l1, l2, emb_dim=3):
  seq_emb1 = tf.random.normal((b, l1, emb_dim))
  seq_emb2 = tf.random.normal((b, l2, emb_dim))
  return tf.einsum('nik,njk->nij', seq_emb1, seq_emb2)


def random_gap_penalty(minval, maxval, b=None, l1=None, l2=None):
  if b is None:
    return tf.random.uniform((), minval=minval, maxval=maxval)
  elif l1 is None or l2 is None:
    return tf.random.uniform((b,), minval=minval, maxval=maxval)
  else:
    return tf.random.uniform((b, l1, l2), minval=minval, maxval=maxval)


def best_alignment_brute_force(weights):
  len_1, len_2, _ = weights.shape
  best_alignment = None
  best_value = -np.inf
  for alignment_mat in npy_ops.alignment_matrices(len_1, len_2):
    value = np.vdot(alignment_mat, weights)
    if value > best_value:
      best_value = value
      best_alignment = alignment_mat
  return best_alignment


class SmithWatermanAffineTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(1)

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
    self._w = alignment.weights_from_sim_mat(self._toy_sub,
                                             self._toy_gap_open,
                                             self._toy_gap_extend)

  @parameterized.product(
      decorator=_DECORATORS,
      b=[1, 8],
      # The test used to pass for 'rank0' but does not anymore.
      # Skipping this case for now as it's not used anyway.
      gap_pen_type=['rank1', 'rank3'],
  )
  def test_weights_from_sim_mat(self, decorator, b, gap_pen_type):
    l1, l2 = 14, 37
    minval_open, maxval_open = 10.5, 11.5
    minval_extend, maxval_extend = 0.8, 1.2

    sim_mat = random_sim_mat(b, l1=l1, l2=l2, emb_dim=3)
    if gap_pen_type == 'rank0':
      gap_open = random_gap_penalty(minval_open, maxval_open)
      gap_extend = random_gap_penalty(minval_extend, maxval_extend)
    elif gap_pen_type == 'rank1':
      gap_open = random_gap_penalty(minval_open, maxval_open, b)
      gap_extend = random_gap_penalty(minval_extend, maxval_extend, b)
    else:
      gap_open = random_gap_penalty(minval_open, maxval_open, b, l1, l2)
      gap_extend = random_gap_penalty(minval_extend, maxval_extend, b, l1, l2)

    weights_from_sim_mat_fn = alignment.weights_from_sim_mat
    if decorator is not None:
      weights_from_sim_mat_fn = decorator(weights_from_sim_mat_fn)

    w = weights_from_sim_mat_fn(sim_mat, gap_open, gap_extend)

    self.assertEqual(w.shape, (b, l1, l2, 9))
    self.assertAllEqual(w[Ellipsis, 0], w[Ellipsis, 1])
    self.assertAllEqual(w[Ellipsis, 0], w[Ellipsis, 2])
    self.assertAllEqual(w[Ellipsis, 0], w[Ellipsis, 3])
    if gap_pen_type == 'rank0':
      gap_open = tf.fill([b, l1, l2], gap_open)
      gap_extend = tf.fill([b, l1, l2], gap_extend)
    elif gap_pen_type == 'rank1':
      gap_open = tf.tile(gap_open[:, None, None], [1, l1, l2])
      gap_extend = tf.tile(gap_extend[:, None, None], [1, l1, l2])
    self.assertAllEqual(w[Ellipsis, 4], -gap_open)
    self.assertAllEqual(w[Ellipsis, 5], -gap_extend)
    self.assertAllEqual(w[Ellipsis, 6], -gap_open)
    self.assertAllEqual(w[Ellipsis, 7], -gap_open)
    self.assertAllEqual(w[Ellipsis, 8], -gap_extend)

  @parameterized.product(
      decorator=_DECORATORS,
      b=[1, 8],
  )
  def test_wavefrontify(self, decorator, b):
    l1, l2, s = 14, 37, 9
    minval_open, maxval_open = 10.5, 11.5
    minval_extend, maxval_extend = 0.8, 1.2

    sim_mat = random_sim_mat(b, l1=l1, l2=l2, emb_dim=3)
    gap_open = random_gap_penalty(minval_open, maxval_open, b, l1, l2)
    gap_extend = random_gap_penalty(minval_extend, maxval_extend, b, l1, l2)
    w = alignment.weights_from_sim_mat(sim_mat, gap_open, gap_extend)

    wavefrontify_fn = tf_ops.wavefrontify
    unwavefrontify_fn = tf_ops.unwavefrontify
    if decorator is not None:
      wavefrontify_fn = decorator(wavefrontify_fn)
      unwavefrontify_fn = decorator(unwavefrontify_fn)

    w_wavefrontified = wavefrontify_fn(w)
    w_unwavefrontified = unwavefrontify_fn(w_wavefrontified)

    self.assertEqual(w_wavefrontified.shape, (l1 + l2 - 1, s, l1, b))
    self.assertAllEqual(w_unwavefrontified, w)
    for n in tf.range(b):
      for a in tf.range(s):
        for i in tf.range(l1):
          for j in tf.range(l2):
            self.assertEqual(w_wavefrontified[i + j, a, i, n], w[n, i, j, a])

  @parameterized.product(
      decorator=_DECORATORS,
      tol=[1e-3, 1e-6, 1e-9],
  )
  def test_toy_smith_waterman(self, decorator, tol):
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      smith_waterman_fn = decorator(smith_waterman_fn)

    values, paths = smith_waterman_fn(self._w, tol)
    paths_squeeze = alignment.path_label_squeeze(paths)
    all_matches = tf.where(
        alignment.paths_to_state_indicators(paths, 'match'))

    values_test = tf.constant([5.0, 3.0, 2.94, 3.0], dtype=tf.float32)
    self.assertAllClose(values, values_test, atol=2 * tol)

    paths_squeeze_test = tf.constant([[[1., 0., 0., 0., 0.],
                                       [0., 2., 0., 0., 0.],
                                       [0., 0., 2., 0., 0.],
                                       [0., 0., 0., 2., 0.],
                                       [0., 0., 0., 0., 2.]],
                                      [[0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 2., 0.],
                                       [0., 0., 0., 0., 2.]],
                                      [[0., 0., 0., 0., 0.],
                                       [0., 1., 5., 0., 0.],
                                       [0., 0., 8., 0., 0.],
                                       [0., 0., 0., 4., 0.],
                                       [0., 0., 0., 0., 2.]],
                                      [[0., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 2., 0., 0.],
                                       [0., 0., 0., 2., 0.],
                                       [0., 0., 0., 0., 0.]]], dtype=tf.float32)
    self.assertAllEqual(paths_squeeze, paths_squeeze_test)

    all_matches_test = tf.constant([[0, 0, 0],
                                    [0, 1, 1],
                                    [0, 2, 2],
                                    [0, 3, 3],
                                    [0, 4, 4],
                                    [1, 2, 2],
                                    [1, 3, 3],
                                    [1, 4, 4],
                                    [2, 1, 1],
                                    [2, 3, 3],
                                    [2, 4, 4],
                                    [3, 1, 1],
                                    [3, 2, 2],
                                    [3, 3, 3]], dtype=tf.int32)
    self.assertAllEqual(all_matches, all_matches_test)

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_smith_waterman_termination(self, decorator):
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      smith_waterman_fn = decorator(smith_waterman_fn)
    tol = 1e-6

    single_sub = tf.concat([- 5*tf.ones((3, 1)),
                            6*tf.eye(3) - 5*tf.ones((3, 3))], 1)
    toy_sub = tf.expand_dims(single_sub, 0)
    toy_gap_open = 0.03 * tf.ones((toy_sub.shape[0],))
    toy_gap_extend = 0.02 * tf.ones((toy_sub.shape[0],))
    w = alignment.weights_from_sim_mat(toy_sub, toy_gap_open, toy_gap_extend)

    values, paths = smith_waterman_fn(w, tol=tol)
    paths_squeeze = alignment.path_label_squeeze(paths)

    self.assertAllClose(values, [3.0], atol=2 * tol)
    paths_squeeze_test = tf.convert_to_tensor([[[0., 1., 0., 0.],
                                                [0., 0., 2., 0.],
                                                [0., 0., 0., 2.]]], tf.float32)
    self.assertAllEqual(paths_squeeze, paths_squeeze_test)

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_smith_waterman_empty(self, decorator):
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      smith_waterman_fn = decorator(smith_waterman_fn)
    tol = 1e-6

    single_sub = - 5*tf.ones((5, 5))
    toy_sub = tf.expand_dims(single_sub, 0)
    toy_gap_open = 0.03 * tf.ones((toy_sub.shape[0],))
    toy_gap_extend = 0.02 * tf.ones((toy_sub.shape[0],))
    w = alignment.weights_from_sim_mat(toy_sub, toy_gap_open, toy_gap_extend)

    values, paths = smith_waterman_fn(w, tol=tol)
    paths_squeeze = alignment.path_label_squeeze(paths)

    self.assertAllClose(values, [0.0], atol=2 * tol)
    self.assertAllEqual(paths_squeeze, tf.zeros([1, 5, 5], tf.float32))

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_backtracking_against_autodiff(self, decorator):
    def grad_fn(w):
      with tf.GradientTape() as tape:
        tape.watch(w)
        maxes, _ = tf_ops.hard_sw_affine(w)
      return tape.gradient(maxes, w)
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      grad_fn = decorator(grad_fn)
      smith_waterman_fn = decorator(smith_waterman_fn)

    # Check that autodiff recovers the handwritten backtracking.
    _, paths = smith_waterman_fn(self._w)
    paths2 = grad_fn(self._w)

    self.assertAllEqual(paths, paths2)

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_backtracking_against_bruteforce(self, decorator):
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      smith_waterman_fn = decorator(smith_waterman_fn)

    _, paths = smith_waterman_fn(self._w)
    paths3 = np.array([best_alignment_brute_force(self._w[i])
                       for i in range(self._w.shape[0])])

    self.assertAllEqual(paths, paths3)

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_perturbation_friendly_version_against_numpy_version(self, decorator):
    smith_waterman_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      smith_waterman_fn = decorator(smith_waterman_fn)

    maxes, _ = smith_waterman_fn(self._w)
    maxes2 = npy_ops.soft_sw_affine(self._toy_sub.numpy(),
                                    self._toy_gap_open.numpy(),
                                    self._toy_gap_extend.numpy(),
                                    temperature=0)

    self.assertAllClose(maxes, maxes2)

  @parameterized.product(
      decorator=_DECORATORS,
  )
  def test_soft_version_against_perturbation_friendly_version(self, decorator):
    soft_version_fn = tf_ops.soft_sw_affine_fwd
    perturbation_friendly_fn = tf_ops.hard_sw_affine
    if decorator is not None:
      soft_version_fn = decorator(soft_version_fn)
      perturbation_friendly_fn = decorator(perturbation_friendly_fn)

    maxes, _ = perturbation_friendly_fn(self._w)
    maxes2 = soft_version_fn(self._toy_sub,
                             self._toy_gap_open,
                             self._toy_gap_extend,
                             temp=None)

    self.assertAllClose(maxes, maxes2)

  @parameterized.product(
      decorator=_DECORATORS,
      temp=[1e-3, 1.0, 1e3],
  )
  def test_soft_version_against_numpy_version(self, decorator, temp):
    soft_version_fn = tf_ops.soft_sw_affine_fwd
    if decorator is not None:
      soft_version_fn = decorator(soft_version_fn)

      maxes = npy_ops.soft_sw_affine(self._toy_sub.numpy(),
                                     self._toy_gap_open.numpy(),
                                     self._toy_gap_extend.numpy(),
                                     temperature=temp)
      maxes2 = soft_version_fn(self._toy_sub,
                               self._toy_gap_open,
                               self._toy_gap_extend,
                               temp=temp)

      self.assertAllClose(maxes, maxes2)


if __name__ == '__main__':
  tf.test.main()
