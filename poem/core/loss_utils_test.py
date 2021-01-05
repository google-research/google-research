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

"""Tests loss utility functions."""

import functools

import tensorflow.compat.v1 as tf

from poem.core import common
from poem.core import loss_utils
tf.disable_v2_behavior()


class LossUtilsTest(tf.test.TestCase):

  def _assert_dict_equal_or_almost_equal(self,
                                         result,
                                         expected_result,
                                         float_equal_places=4):
    self.assertCountEqual(result.keys(), expected_result.keys())
    for key, ev in expected_result.items():
      if isinstance(ev, int):
        self.assertEqual(result[key], ev, msg='Key = `%s`.' % key)
      elif isinstance(ev, float):
        self.assertAlmostEqual(
            result[key], ev, places=float_equal_places, msg='Key = `%s`.' % key)
      elif isinstance(ev, (list, tuple)):
        if ev:
          if isinstance(ev[0], int):
            self.assertAllEqual(result[key], ev, msg='Key = `%s`.' % key)
          elif isinstance(ev[0], float):
            self.assertAllClose(result[key], ev, msg='Key = `%s`.' % key)
          else:
            raise ValueError(
                'Unsupported expected value type for key `%s`: list/tuple of %s.'
                % (key, type(ev[0])))
        else:
          self.assertEqual(result[key], ev, msg='Key = `%s`.' % key)
      else:
        raise ValueError('Unsupported expected value type for key `%s`: %s.' %
                         (key, type(ev)))

  def test_create_sample_distance_fn_case_1(self):
    # Shape = [2, 3, 2].
    lhs = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                       [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    # Shape = [2, 4, 2].
    rhs = tf.constant([[[16.0, 15.0], [14.0, 13.0], [12.0, 11.0], [10.0, 9.0]],
                       [[8.0, 7.0], [6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
    self.assertAllClose(
        loss_utils.create_sample_distance_fn(
            pair_type=common.DISTANCE_PAIR_TYPE_ALL_PAIRS,
            distance_kernel=common.DISTANCE_KERNEL_SQUARED_L2,
            pairwise_reduction=functools.partial(
                tf.math.reduce_min, axis=[-2, -1]),
            componentwise_reduction=tf.identity)(lhs, rhs), [34.0, 2.0])

  def test_create_sample_distance_fn_case_2(self):
    # Shape = [1, 2, 2].
    lhs = tf.constant([[[1.0, 2.0], [3.0, 4.0]]])
    # Shape = [1, 3, 2].
    rhs = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    distances = loss_utils.create_sample_distance_fn(
        pair_type=common.DISTANCE_PAIR_TYPE_ALL_PAIRS,
        distance_kernel=common.DISTANCE_KERNEL_SQUARED_L2,
        pairwise_reduction=common.DISTANCE_REDUCTION_MEAN,
        componentwise_reduction=tf.identity)(lhs, rhs)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      distances_result = sess.run(distances)

    self.assertAllClose(distances_result, [56.0 / 6.0])

  def test_compute_negative_indicator_matrix(self):
    anchors = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    matches = tf.constant([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    indicators = loss_utils.compute_negative_indicator_matrix(
        anchors,
        matches,
        distance_fn=tf.math.squared_difference,
        min_negative_distance=5.0)

    self.assertAllEqual(indicators,
                        [[[True, True], [True, False], [False, False]],
                         [[True, False], [False, False], [False, True]]])

  def test_compute_hard_negative_distances(self):
    anchor_match_distance_matrix = tf.constant([
        [1.0, 3.0, 2.0, 0.0],
        [6.0, 5.0, 4.0, 0.0],
        [7.0, 8.0, 9.0, 10.0],
    ])
    negative_indicator_matrix = tf.constant([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ])
    hard_negative_distances, hard_negative_mining_distances = (
        loss_utils.compute_hard_negative_distances(anchor_match_distance_matrix,
                                                   negative_indicator_matrix))

    self.assertAllEqual(hard_negative_distances, [1.0, 5.0, tf.float32.max])
    self.assertAllEqual(hard_negative_mining_distances,
                        [1.0, 5.0, tf.float32.max])

  def test_compute_semi_hard_negative_distances(self):
    anchor_match_distance_matrix = tf.constant([
        [1.0, 3.0, 2.0, 0.0],
        [6.0, 5.0, 4.0, 0.0],
        [7.0, 8.0, 9.0, 10.0],
    ])
    negative_indicator_matrix = tf.constant([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
    ])
    anchor_positive_mining_distances = tf.constant([0.0, 6.5, 10.0])
    hard_negative_distances, hard_negative_mining_distances = (
        loss_utils.compute_hard_negative_distances(
            anchor_match_distance_matrix,
            negative_indicator_matrix,
            use_semi_hard=True,
            anchor_positive_mining_distances=anchor_positive_mining_distances))
    self.assertAllEqual(hard_negative_distances,
                        [1.0, tf.float32.max, tf.float32.max])
    self.assertAllEqual(hard_negative_mining_distances,
                        [1.0, tf.float32.max, tf.float32.max])

  def test_compute_semi_hard_negative_distances_with_mining_distances(self):
    anchor_match_distance_matrix = tf.constant([
        [4.0, 3.0, 2.0, 1.0],
        [8.0, 5.0, 4.0, 0.0],
        [8.0, 9.0, 7.0, 10.0],
    ])
    negative_indicator_matrix = tf.constant([
        [True, True, True, True],
        [True, True, True, False],
        [False, True, True, True],
    ])
    anchor_positive_mining_distances = tf.constant([0.0, 6.5, 2.0])
    anchor_match_mining_distance_matrix = tf.constant([
        [1.0, 3.0, 2.0, 0.0],
        [6.0, 5.0, 4.0, 0.0],
        [7.0, 8.0, 9.0, 10.0],
    ])
    hard_negative_distances, hard_negative_mining_distances = (
        loss_utils.compute_hard_negative_distances(
            anchor_match_distance_matrix,
            negative_indicator_matrix,
            use_semi_hard=True,
            anchor_positive_mining_distances=anchor_positive_mining_distances,
            anchor_match_mining_distance_matrix=(
                anchor_match_mining_distance_matrix)))
    self.assertAllEqual(hard_negative_distances, [4.0, tf.float32.max, 9.0])
    self.assertAllEqual(hard_negative_mining_distances,
                        [1.0, tf.float32.max, 8.0])

  def test_compute_semi_hard_negative_triplet_loss(self):
    anchor_positive_distances = tf.constant([0.0, 8.0, 2.0])
    anchor_match_distance_matrix = tf.constant([
        [2.0, 8.0, 32.0, 72.0],
        [2.0, 0.0, 8.0, 32.0],
        [18.0, 8.0, 0.0, 8.0],
    ])
    anchor_match_negative_indicator_matrix = tf.constant([
        [True, True, True, True],
        [True, True, False, True],
        [True, False, True, False],
    ])
    (loss, num_active_triplets, anchor_negative_distances, mining_loss,
     num_active_mining_triplets, anchor_negative_mining_distances) = (
         loss_utils.compute_hard_negative_triplet_loss(
             anchor_positive_distances,
             anchor_match_distance_matrix,
             anchor_match_negative_indicator_matrix,
             margin=20.0,
             use_semi_hard=True))

    with self.session() as sess:
      (loss_result, num_active_triplets_result,
       anchor_negative_distances_result, mining_loss_result,
       num_active_mining_triplets_result,
       anchor_negative_mining_distances_result) = sess.run([
           loss, num_active_triplets, anchor_negative_distances, mining_loss,
           num_active_mining_triplets, anchor_negative_mining_distances
       ])

    self.assertAlmostEqual(loss_result, 22.0 / 3.0, places=4)
    self.assertEqual(num_active_triplets_result, 2)
    self.assertAllClose(anchor_negative_distances_result, [2.0, 32.0, 18.0])
    self.assertAlmostEqual(mining_loss_result, 22.0 / 3.0, places=4)
    self.assertEqual(num_active_mining_triplets_result, 2)
    self.assertAllClose(anchor_negative_mining_distances_result,
                        [2.0, 32.0, 18.0])

  def test_compute_semi_hard_negative_triplet_loss_with_mining_distances(self):
    anchor_positive_distances = tf.constant([1.0, 2.0, 3.0])
    anchor_match_distance_matrix = tf.constant([
        [4.0, 3.0, 2.0, 1.0],
        [8.0, 7.0, 6.0, 5.0],
        [12.0, 11.0, 10.0, 9.0],
    ])
    anchor_match_negative_indicator_matrix = tf.constant([
        [True, True, True, True],
        [True, True, False, True],
        [True, False, True, False],
    ])
    anchor_positive_mining_distances = tf.constant([0.0, 8.0, 2.0])
    anchor_match_mining_distance_matrix = tf.constant([
        [2.0, 8.0, 32.0, 72.0],
        [2.0, 0.0, 8.0, 32.0],
        [18.0, 8.0, 0.0, 8.0],
    ])
    (loss, num_active_triplets, anchor_negative_distances, mining_loss,
     num_active_mining_triplets, anchor_negative_mining_distances) = (
         loss_utils.compute_hard_negative_triplet_loss(
             anchor_positive_distances,
             anchor_match_distance_matrix,
             anchor_match_negative_indicator_matrix,
             margin=5.0,
             use_semi_hard=True,
             anchor_positive_mining_distances=anchor_positive_mining_distances,
             anchor_match_mining_distance_matrix=(
                 anchor_match_mining_distance_matrix)))

    with self.session() as sess:
      (loss_result, num_active_triplets_result,
       anchor_negative_distances_result, mining_loss_result,
       num_active_mining_triplets_result,
       anchor_negative_mining_distances_result) = sess.run([
           loss, num_active_triplets, anchor_negative_distances, mining_loss,
           num_active_mining_triplets, anchor_negative_mining_distances
       ])

    self.assertAlmostEqual(loss_result, 4.0 / 3.0, places=4)
    self.assertEqual(num_active_triplets_result, 2)
    self.assertAllClose(anchor_negative_distances_result, [4.0, 5.0, 12.0])
    self.assertAlmostEqual(mining_loss_result, 1.0, places=4)
    self.assertEqual(num_active_mining_triplets_result, 1)
    self.assertAllClose(anchor_negative_mining_distances_result,
                        [2.0, 32.0, 18.0])

  def test_compute_keypoint_triplet_losses(self):
    # Shape = [3, 1, 1, 2].
    anchor_embeddings = tf.constant([
        [[[1.0, 2.0]]],
        [[[3.0, 4.0]]],
        [[[5.0, 6.0]]],
    ])
    # Shape = [3, 1, 1, 2].
    positive_embeddings = tf.constant([
        [[[1.0, 2.0]]],
        [[[5.0, 6.0]]],
        [[[6.0, 7.0]]],
    ])
    # Shape = [4, 1, 1, 2].
    match_embeddings = tf.constant([
        [[[2.0, 3.0]]],
        [[[3.0, 4.0]]],
        [[[5.0, 6.0]]],
        [[[7.0, 8.0]]],
    ])
    # Shape = [3, 1].
    anchor_keypoints = tf.constant([[1], [2], [3]])
    # Shape = [4, 1].
    match_keypoints = tf.constant([[1], [2], [3], [4]])

    def mock_keypoint_distance_fn(unused_lhs, unused_rhs):
      # Shape = [3, 4].
      return tf.constant([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0, 0.0]])

    loss, summaries = loss_utils.compute_keypoint_triplet_losses(
        anchor_embeddings,
        positive_embeddings,
        match_embeddings,
        anchor_keypoints,
        match_keypoints,
        margin=20.0,
        min_negative_keypoint_distance=0.5,
        use_semi_hard=True,
        exclude_inactive_triplet_loss=True,
        keypoint_distance_fn=mock_keypoint_distance_fn)

    with self.session() as sess:
      loss_result, summaries_result = sess.run([loss, summaries])

    self.assertAlmostEqual(loss_result, 11.0)

    expected_summaries_result = {
        'triplet_loss/Margin': 20.0,
        'triplet_loss/Anchor/Positive/Distance/Mean': 10.0 / 3,
        'triplet_loss/Anchor/Positive/Distance/Median': 2.0,
        'triplet_loss/Anchor/HardNegative/Distance/Mean': 2.0 / 3,
        'triplet_loss/Anchor/HardNegative/Distance/Median': 0.0,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Mean': 52.0 / 3,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Median': 18.0,
        'triplet_loss/HardNegative/Loss/All': 68.0 / 3,
        'triplet_loss/HardNegative/Loss/Active': 68.0 / 3,
        'triplet_loss/HardNegative/ActiveTripletNum': 3,
        'triplet_loss/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_loss/SemiHardNegative/Loss/All': 22.0 / 3,
        'triplet_loss/SemiHardNegative/Loss/Active': 22.0 / 2,
        'triplet_loss/SemiHardNegative/ActiveTripletNum': 2,
        'triplet_loss/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Mean': 10.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Median': 2.0,
        'triplet_mining/Anchor/HardNegative/Distance/Mean': 2.0 / 3,
        'triplet_mining/Anchor/HardNegative/Distance/Median': 0.0,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Mean': 52.0 / 3,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Median': 18.0,
        'triplet_mining/HardNegative/Loss/All': 68.0 / 3,
        'triplet_mining/HardNegative/Loss/Active': 68.0 / 3,
        'triplet_mining/HardNegative/ActiveTripletNum': 3,
        'triplet_mining/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_mining/SemiHardNegative/Loss/All': 22.0 / 3,
        'triplet_mining/SemiHardNegative/Loss/Active': 22.0 / 2,
        'triplet_mining/SemiHardNegative/ActiveTripletNum': 2,
        'triplet_mining/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
    }
    self._assert_dict_equal_or_almost_equal(summaries_result,
                                            expected_summaries_result)

  def test_compute_sample_keypoint_triplet_losses(self):
    # Shape = [3, 3, 2, 2].
    anchor_embeddings = tf.constant([
        [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]],
         [[1.0, 2.0], [1.0, 2.0]]],
        [[[3.0, 4.0], [3.0, 4.0]], [[3.0, 4.0], [3.0, 4.0]],
         [[3.0, 4.0], [3.0, 4.0]]],
        [[[5.0, 6.0], [5.0, 6.0]], [[5.0, 6.0], [5.0, 6.0]],
         [[5.0, 6.0], [5.0, 6.0]]],
    ])
    # Shape = [3, 3, 2, 2].
    positive_embeddings = tf.constant([
        [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]],
         [[1.0, 2.0], [1.0, 2.0]]],
        [[[5.0, 6.0], [5.0, 6.0]], [[5.0, 6.0], [5.0, 6.0]],
         [[5.0, 6.0], [5.0, 6.0]]],
        [[[6.0, 7.0], [6.0, 7.0]], [[6.0, 7.0], [6.0, 7.0]],
         [[6.0, 7.0], [6.0, 7.0]]],
    ])
    # Shape = [4, 3, 2, 2].
    match_embeddings = tf.constant([
        [[[2.0, 3.0], [2.0, 3.0]], [[2.0, 3.0], [2.0, 3.0]],
         [[2.0, 3.0], [2.0, 3.0]]],
        [[[3.0, 4.0], [3.0, 4.0]], [[3.0, 4.0], [3.0, 4.0]],
         [[3.0, 4.0], [3.0, 4.0]]],
        [[[5.0, 6.0], [5.0, 6.0]], [[5.0, 6.0], [5.0, 6.0]],
         [[5.0, 6.0], [5.0, 6.0]]],
        [[[7.0, 8.0], [7.0, 8.0]], [[7.0, 8.0], [7.0, 8.0]],
         [[7.0, 8.0], [7.0, 8.0]]],
    ])
    # Shape = [3, 1].
    anchor_keypoints = tf.constant([[1], [2], [3]])
    # Shape = [4, 1].
    match_keypoints = tf.constant([[1], [2], [3], [4]])

    def mock_keypoint_distance_fn(unused_lhs, unused_rhs):
      return tf.constant([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0, 0.0]])

    loss, summaries = loss_utils.compute_keypoint_triplet_losses(
        anchor_embeddings,
        positive_embeddings,
        match_embeddings,
        anchor_keypoints,
        match_keypoints,
        margin=240.0,
        min_negative_keypoint_distance=0.5,
        use_semi_hard=True,
        exclude_inactive_triplet_loss=True,
        embedding_sample_distance_fn=loss_utils.create_sample_distance_fn(
            pairwise_reduction=functools.partial(
                tf.math.reduce_sum, axis=[-2, -1]),
            componentwise_reduction=functools.partial(
                tf.math.reduce_sum, axis=[-1])),
        keypoint_distance_fn=mock_keypoint_distance_fn)

    with self.session() as sess:
      loss_result, summaries_result = sess.run([loss, summaries])

    self.assertAlmostEqual(loss_result, 132.0)

    expected_summaries_result = {
        'triplet_loss/Margin': 240.0,
        'triplet_loss/Anchor/Positive/Distance/Mean': 120.0 / 3,
        'triplet_loss/Anchor/Positive/Distance/Median': 24.0,
        'triplet_loss/Anchor/HardNegative/Distance/Mean': 24.0 / 3,
        'triplet_loss/Anchor/HardNegative/Distance/Median': 0.0,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Mean': 624.0 / 3,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Median': 216.0,
        'triplet_loss/HardNegative/Loss/All': 816.0 / 3,
        'triplet_loss/HardNegative/Loss/Active': 816.0 / 3,
        'triplet_loss/HardNegative/ActiveTripletNum': 3,
        'triplet_loss/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_loss/SemiHardNegative/Loss/All': 264.0 / 3,
        'triplet_loss/SemiHardNegative/Loss/Active': 264.0 / 2,
        'triplet_loss/SemiHardNegative/ActiveTripletNum': 2,
        'triplet_loss/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Mean': 120.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Median': 24.0,
        'triplet_mining/Anchor/HardNegative/Distance/Mean': 24.0 / 3,
        'triplet_mining/Anchor/HardNegative/Distance/Median': 0.0,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Mean': 624.0 / 3,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Median': 216.0,
        'triplet_mining/HardNegative/Loss/All': 816.0 / 3,
        'triplet_mining/HardNegative/Loss/Active': 816.0 / 3,
        'triplet_mining/HardNegative/ActiveTripletNum': 3,
        'triplet_mining/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_mining/SemiHardNegative/Loss/All': 264.0 / 3,
        'triplet_mining/SemiHardNegative/Loss/Active': 264.0 / 2,
        'triplet_mining/SemiHardNegative/ActiveTripletNum': 2,
        'triplet_mining/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
    }
    self._assert_dict_equal_or_almost_equal(summaries_result,
                                            expected_summaries_result)

  def test_compute_keypoint_triplet_losses_with_sample_mining_embeddings(self):
    # Shape = [3, 3, 1, 2].
    anchor_embeddings = tf.constant([
        [[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]],
        [[[3.0, 4.0]], [[3.0, 4.0]], [[3.0, 4.0]]],
        [[[5.0, 6.0]], [[5.0, 6.0]], [[5.0, 6.0]]],
    ])
    # Shape = [3, 3, 1, 2].
    positive_embeddings = tf.constant([
        [[[2.0, 1.0]], [[2.0, 1.0]], [[2.0, 1.0]]],
        [[[6.0, 5.0]], [[6.0, 5.0]], [[6.0, 5.0]]],
        [[[7.0, 6.0]], [[7.0, 6.0]], [[7.0, 6.0]]],
    ])
    # Shape = [4, 3, 2, 2].
    match_embeddings = tf.constant([
        [[[3.0, 2.0], [3.0, 2.0]], [[3.0, 2.0], [3.0, 2.0]],
         [[3.0, 2.0], [3.0, 2.0]]],
        [[[4.0, 3.0], [4.0, 3.0]], [[4.0, 3.0], [4.0, 3.0]],
         [[4.0, 3.0], [4.0, 3.0]]],
        [[[6.0, 5.0], [6.0, 5.0]], [[6.0, 5.0], [6.0, 5.0]],
         [[6.0, 5.0], [6.0, 5.0]]],
        [[[8.0, 7.0], [8.0, 7.0]], [[8.0, 7.0], [8.0, 7.0]],
         [[8.0, 7.0], [8.0, 7.0]]],
    ])
    # Shape = [3, 1].
    anchor_keypoints = tf.constant([[1], [2], [3]])
    # Shape = [4, 1].
    match_keypoints = tf.constant([[1], [2], [3], [4]])

    def mock_keypoint_distance_fn(unused_lhs, unused_rhs):
      return tf.constant([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0, 0.0]])

    # Shape = [3, 3, 1, 2].
    anchor_mining_embeddings = tf.constant([
        [[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]],
        [[[3.0, 4.0]], [[3.0, 4.0]], [[3.0, 4.0]]],
        [[[5.0, 6.0]], [[5.0, 6.0]], [[5.0, 6.0]]],
    ])
    # Shape = [3, 3, 1, 2].
    positive_mining_embeddings = tf.constant([
        [[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]],
        [[[5.0, 6.0]], [[5.0, 6.0]], [[5.0, 6.0]]],
        [[[6.0, 7.0]], [[6.0, 7.0]], [[6.0, 7.0]]],
    ])
    # Shape = [4, 3, 1, 2].
    match_mining_embeddings = tf.constant([
        [[[2.0, 3.0]], [[2.0, 3.0]], [[2.0, 3.0]]],
        [[[3.0, 4.0]], [[3.0, 4.0]], [[3.0, 4.0]]],
        [[[5.0, 6.0]], [[5.0, 6.0]], [[5.0, 6.0]]],
        [[[7.0, 8.0]], [[7.0, 8.0]], [[7.0, 8.0]]],
    ])

    loss, summaries = loss_utils.compute_keypoint_triplet_losses(
        anchor_embeddings,
        positive_embeddings,
        match_embeddings,
        anchor_keypoints,
        match_keypoints,
        margin=120.0,
        min_negative_keypoint_distance=0.5,
        use_semi_hard=True,
        exclude_inactive_triplet_loss=True,
        embedding_sample_distance_fn=loss_utils.create_sample_distance_fn(
            pairwise_reduction=functools.partial(
                tf.math.reduce_sum, axis=[-2, -1]),
            componentwise_reduction=functools.partial(
                tf.math.reduce_sum, axis=[-1])),
        keypoint_distance_fn=mock_keypoint_distance_fn,
        anchor_mining_embeddings=anchor_mining_embeddings,
        positive_mining_embeddings=positive_mining_embeddings,
        match_mining_embeddings=match_mining_embeddings)

    with self.session() as sess:
      loss_result, summaries_result = sess.run([loss, summaries])

    self.assertAlmostEqual(loss_result, 57.0)

    expected_summaries_result = {
        'triplet_loss/Margin': 120.0,
        'triplet_loss/Anchor/Positive/Distance/Mean': 48.0 / 3,
        'triplet_loss/Anchor/Positive/Distance/Median': 12.0,
        'triplet_loss/Anchor/HardNegative/Distance/Mean': 48.0 / 3,
        'triplet_loss/Anchor/HardNegative/Distance/Median': 12.0,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Mean': 348.0 / 3,
        'triplet_loss/Anchor/SemiHardNegative/Distance/Median': 120.0,
        'triplet_loss/HardNegative/Loss/All': 360.0 / 3,
        'triplet_loss/HardNegative/Loss/Active': 360.0 / 3,
        'triplet_loss/HardNegative/ActiveTripletNum': 3,
        'triplet_loss/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_loss/SemiHardNegative/Loss/All': 114.0 / 3,
        'triplet_loss/SemiHardNegative/Loss/Active': 114.0 / 2,
        'triplet_loss/SemiHardNegative/ActiveTripletNum': 2,
        'triplet_loss/SemiHardNegative/ActiveTripletRatio': 2.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Mean': 30.0 / 3,
        'triplet_mining/Anchor/Positive/Distance/Median': 6.0,
        'triplet_mining/Anchor/HardNegative/Distance/Mean': 6.0 / 3,
        'triplet_mining/Anchor/HardNegative/Distance/Median': 0.0,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Mean': 156.0 / 3,
        'triplet_mining/Anchor/SemiHardNegative/Distance/Median': 54.0,
        'triplet_mining/HardNegative/Loss/All': 384.0 / 3,
        'triplet_mining/HardNegative/Loss/Active': 384.0 / 3,
        'triplet_mining/HardNegative/ActiveTripletNum': 3,
        'triplet_mining/HardNegative/ActiveTripletRatio': 1.0,
        'triplet_mining/SemiHardNegative/Loss/All': 234.0 / 3,
        'triplet_mining/SemiHardNegative/Loss/Active': 234.0 / 3,
        'triplet_mining/SemiHardNegative/ActiveTripletNum': 3,
        'triplet_mining/SemiHardNegative/ActiveTripletRatio': 1.0,
    }
    self._assert_dict_equal_or_almost_equal(summaries_result,
                                            expected_summaries_result)

  def test_compute_kl_regularization_loss(self):
    means = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stddevs = tf.constant([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    weighted_loss, summaries = loss_utils.compute_kl_regularization_loss(
        means, stddevs, loss_weight=3.0)

    with self.session() as sess:
      weighted_loss_result, summaries_result = sess.run(
          [weighted_loss, summaries])

    self.assertAlmostEqual(weighted_loss_result, 122.131123182, places=4)

    expected_summaries_result = {
        'regularization_loss/KL/PriorMean/Mean': 0.0,
        'regularization_loss/KL/PriorVar/Mean': 1.0,
        'regularization_loss/KL/Loss/Original': 40.710374394,
        'regularization_loss/KL/Loss/Weighted': 122.131123182,
        'regularization_loss/KL/Loss/Weight': 3.0,
    }
    self._assert_dict_equal_or_almost_equal(summaries_result,
                                            expected_summaries_result)

  def test_compute_positive_pairwise_loss(self):
    anchor_embeddings = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                     [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    positive_embeddings = tf.constant([[[12.0, 11.0], [10.0, 9.0], [8.0, 7.0]],
                                       [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
    weighted_loss, summaries = loss_utils.compute_positive_pairwise_loss(
        anchor_embeddings, positive_embeddings, loss_weight=6.0)

    with self.session() as sess:
      weighted_loss_result, summaries_result = sess.run(
          [weighted_loss, summaries])

    self.assertAlmostEqual(weighted_loss_result, 572.0)

    expected_summaries_result = {
        'pairwise_loss/PositivePair/Loss/Original': 95.333333333,
        'pairwise_loss/PositivePair/Loss/Weighted': 572.0,
        'pairwise_loss/PositivePair/Loss/Weight': 6.0,
    }
    self._assert_dict_equal_or_almost_equal(summaries_result,
                                            expected_summaries_result)


if __name__ == '__main__':
  tf.test.main()
