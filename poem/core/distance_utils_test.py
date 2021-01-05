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

"""Tests for distance utility functions."""

import tensorflow.compat.v1 as tf

from poem.core import distance_utils
tf.disable_v2_behavior()


class DistanceUtilsTest(tf.test.TestCase):

  def test_compute_l2_distances(self):
    # Shape = [2, 1, 2, 2]
    lhs_points = [[[[0.0, 1.0], [2.0, 3.0]]], [[[10.0, 11.0], [12.0, 13.0]]]]
    rhs_points = [[[[0.0, 1.1], [2.3, 3.4]]], [[[10.4, 11.0], [12.4, 13.3]]]]
    # Shape = [2, 1, 2]
    distances = distance_utils.compute_l2_distances(lhs_points, rhs_points)
    self.assertAllClose(distances, [[[0.1, 0.5]], [[0.4, 0.5]]])

  def test_compute_l2_distances_keepdims(self):
    # Shape = [2, 1, 2, 2]
    lhs = [[[[0.0, 1.0], [2.0, 3.0]]], [[[10.0, 11.0], [12.0, 13.0]]]]
    rhs = [[[[0.0, 1.1], [2.3, 3.4]]], [[[10.4, 11.0], [12.4, 13.3]]]]
    # Shape = [2, 1, 2]
    distances = distance_utils.compute_l2_distances(lhs, rhs, keepdims=True)
    self.assertAllClose(distances, [[[[0.1], [0.5]]], [[[0.4], [0.5]]]])

  def test_compute_squared_l2_distances(self):
    # Shape = [2, 1, 2, 2]
    lhs_points = [[[[0.0, 1.0], [2.0, 3.0]]], [[[10.0, 11.0], [12.0, 13.0]]]]
    rhs_points = [[[[0.0, 1.1], [2.3, 3.4]]], [[[10.4, 11.0], [12.4, 13.3]]]]
    # Shape = [2, 1, 2]
    distances = distance_utils.compute_l2_distances(
        lhs_points, rhs_points, squared=True)
    self.assertAllClose(distances, [[[0.01, 0.25]], [[0.16, 0.25]]])

  def test_compute_sigmoid_matching_probabilities(self):
    inner_distances = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    matching_probabilities = distance_utils.compute_sigmoid_matching_probabilities(
        inner_distances,
        a_initializer=tf.initializers.constant(-4.60517018599),
        b_initializer=tf.initializers.ones())

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      matching_probabilities_result = sess.run(matching_probabilities)

    self.assertAllClose(matching_probabilities_result,
                        [[0.70617913, 0.704397395, 0.702607548],
                         [0.700809625, 0.69900366, 0.697189692]])

  def test_compute_sigmoid_matching_distances(self):
    inner_distances = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    matching_distances = distance_utils.compute_sigmoid_matching_distances(
        inner_distances,
        a_initializer=tf.initializers.constant(-4.60517018599),
        b_initializer=tf.initializers.ones())

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      matching_distances_result = sess.run(matching_distances)

    self.assertAllClose(matching_distances_result,
                        [[0.347886348, 0.350412601, 0.352956796],
                         [0.355519006, 0.3580993, 0.360697751]])

  def test_compute_all_pair_squared_l2_distances(self):
    # Shape = [2, 2, 2].
    lhs = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]])
    # Shape = [2, 3, 2].
    rhs = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                       [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
    # Shape = [2, 2, 3].
    distances = distance_utils.compute_all_pair_l2_distances(
        lhs, rhs, squared=True)
    self.assertAllClose(distances, [[[0.0, 8.0, 32.0], [8.0, 0.0, 8.0]],
                                    [[8.0, 0.0, 8.0], [32.0, 8.0, 0.0]]])

  def test_compute_all_pair_l2_distances(self):
    # Shape = [2, 2, 2].
    lhs = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]])
    # Shape = [2, 3, 2].
    rhs = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                       [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
    # Shape = [2, 2, 3].
    distances = distance_utils.compute_all_pair_l2_distances(lhs, rhs)
    self.assertAllClose(
        distances,
        [[[0.0, 2.828427125, 5.656854249], [2.828427125, 0.0, 2.828427125]],
         [[2.828427125, 0.0, 2.828427125], [5.656854249, 2.828427125, 0.0]]])

  def test_compute_gaussian_likelihoods(self):
    # Shape = [2, 1, 1].
    means = tf.constant([[[1.0]], [[2.0]]])
    # Shape = [2, 1, 1].
    stddevs = tf.constant([[[1.0]], [[2.0]]])
    # Shape = [2, 3, 1].
    samples = tf.constant([[[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]]])
    # Shape = [2, 3, 1].
    likelihoods = distance_utils.compute_gaussian_likelihoods(
        means, stddevs, samples)
    self.assertAllClose(
        likelihoods, [[[1.0], [0.3173], [0.0455]], [[1.0], [0.3173], [0.0455]]],
        atol=1e-4)

  def test_compute_distance_matrix(self):
    # Shape = [2, 1]
    start_points = tf.constant([[1], [2]])
    # Shape = [3, 1]
    end_points = tf.constant([[3], [4], [5]])
    distance_matrix = distance_utils.compute_distance_matrix(
        start_points, end_points, distance_fn=tf.math.subtract)
    self.assertAllEqual(distance_matrix,
                        [[[-2], [-3], [-4]], [[-1], [-2], [-3]]])

  def test_compute_distance_matrix_with_both_masks(self):
    # Shape = [2, 3, 1].
    start_points = tf.constant([
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
    ])
    # Shape = [3, 3, 1].
    end_points = tf.constant([
        [[11.0], [12.0], [13.0]],
        [[14.0], [15.0], [16.0]],
        [[17.0], [18.0], [19.0]],
    ])
    # Shape = [2, 3].
    start_point_masks = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    # Shape = [3, 3].
    end_point_masks = tf.constant([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0]])

    def masked_add(lhs, rhs, masks):
      masks = tf.expand_dims(masks, axis=-1)
      return tf.math.reduce_sum((lhs + rhs) * masks, axis=[-2, -1])

    # Shape = [2, 3].
    distance_matrix = distance_utils.compute_distance_matrix(
        start_points,
        end_points,
        distance_fn=masked_add,
        start_point_masks=start_point_masks,
        end_point_masks=end_point_masks)

    with self.session() as sess:
      distance_matrix_result = sess.run(distance_matrix)

    self.assertAllClose(distance_matrix_result,
                        [[28.0, 15.0, 60.0], [15.0, 18.0, 44.0]])

  def test_compute_distance_matrix_with_start_masks(self):
    # Shape = [2, 3, 1].
    start_points = tf.constant([
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
    ])
    # Shape = [3, 3, 1].
    end_points = tf.constant([
        [[11.0], [12.0], [13.0]],
        [[14.0], [15.0], [16.0]],
        [[17.0], [18.0], [19.0]],
    ])
    # Shape = [2, 3].
    start_point_masks = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    def masked_add(lhs, rhs, masks):
      masks = tf.expand_dims(masks, axis=-1)
      return tf.math.reduce_sum((lhs + rhs) * masks, axis=[-2, -1])

    # Shape = [2, 3].
    distance_matrix = distance_utils.compute_distance_matrix(
        start_points,
        end_points,
        distance_fn=masked_add,
        start_point_masks=start_point_masks)

    with self.session() as sess:
      distance_matrix_result = sess.run(distance_matrix)

    self.assertAllClose(distance_matrix_result,
                        [[42.0, 51.0, 60.0], [32.0, 38.0, 44.0]])

  def test_compute_distance_matrix_with_end_masks(self):
    # Shape = [2, 3, 1].
    start_points = tf.constant([
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
    ])
    # Shape = [3, 3, 1].
    end_points = tf.constant([
        [[11.0], [12.0], [13.0]],
        [[14.0], [15.0], [16.0]],
        [[17.0], [18.0], [19.0]],
    ])
    # Shape = [3, 3].
    end_point_masks = tf.constant([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0]])

    def masked_add(lhs, rhs, masks):
      masks = tf.expand_dims(masks, axis=-1)
      return tf.math.reduce_sum((lhs + rhs) * masks, axis=[-2, -1])

    # Shape = [2, 3].
    distance_matrix = distance_utils.compute_distance_matrix(
        start_points,
        end_points,
        distance_fn=masked_add,
        end_point_masks=end_point_masks)

    with self.session() as sess:
      distance_matrix_result = sess.run(distance_matrix)

    self.assertAllClose(distance_matrix_result,
                        [[28.0, 15.0, 60.0], [34.0, 18.0, 69.0]])

  def test_compute_gaussian_kl_divergence_unit_univariate(self):
    lhs_means = [0.0]
    lhs_stddevs = [1.0]
    kl_divergence = distance_utils.compute_gaussian_kl_divergence(
        lhs_means, lhs_stddevs, rhs_means=0.0, rhs_stddevs=1.0)

    with self.session() as sess:
      kl_divergence_result = sess.run(kl_divergence)

    self.assertAlmostEqual(kl_divergence_result, 0.0)

  def test_compute_gaussian_kl_divergence_unit_multivariate_to_univariate(self):
    lhs_means = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    lhs_stddevs = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    kl_divergence = distance_utils.compute_gaussian_kl_divergence(
        lhs_means, lhs_stddevs, rhs_means=0.0, rhs_stddevs=1.0)

    with self.session() as sess:
      kl_divergence_result = sess.run(kl_divergence)

    self.assertAllClose(kl_divergence_result, [0.0, 0.0])

  def test_compute_gaussian_kl_divergence_multivariate_to_multivariate(self):
    lhs_means = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lhs_stddevs = tf.constant([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    rhs_means = tf.constant([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    rhs_stddevs = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kl_divergence = distance_utils.compute_gaussian_kl_divergence(
        lhs_means, lhs_stddevs, rhs_means=rhs_means, rhs_stddevs=rhs_stddevs)

    with self.session() as sess:
      kl_divergence_result = sess.run(kl_divergence)

    self.assertAllClose(kl_divergence_result, [31.198712171, 2.429343385])


if __name__ == '__main__':
  tf.test.main()
