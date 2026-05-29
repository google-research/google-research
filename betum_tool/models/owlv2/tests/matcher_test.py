# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for matcher.

This file includes tests for the `HungarianMatcher` class, which is used to
compute the optimal assignment between predictions and targets in the
detection model. The tests are written using `absl.testing` library.
"""

from absl.testing import absltest
from models.owlv2 import matcher
import torch


class MatcherTest(absltest.TestCase):

  def test_hungarian_matcher_forward(self):
    matcher_instance = matcher.HungarianMatcher()
    outputs = {
        "logits": torch.rand(1, 10, 5),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1, 2], dtype=torch.int64),
            "boxes": torch.rand(2, 4),
        }
    ]
    indices = matcher_instance(outputs, targets)
    self.assertLen(indices, 1)
    q_idx, t_idx = indices[0]
    self.assertIsInstance(q_idx, torch.Tensor)
    self.assertIsInstance(t_idx, torch.Tensor)
    self.assertEqual(q_idx.dtype, torch.int64)
    self.assertEqual(t_idx.dtype, torch.int64)
    self.assertEqual(q_idx.shape, t_idx.shape)
    self.assertLessEqual(q_idx.numel(), 2)

  def test_greedy_matcher_forward(self):
    matcher_instance = matcher.GreedyMatcher()
    outputs = {
        "logits": torch.rand(1, 10, 5),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1, 2], dtype=torch.int64),
            "boxes": torch.rand(2, 4),
        }
    ]
    indices = matcher_instance(outputs, targets)
    self.assertLen(indices, 1)
    q_idx, t_idx = indices[0]
    self.assertIsInstance(q_idx, torch.Tensor)
    self.assertIsInstance(t_idx, torch.Tensor)
    self.assertEqual(q_idx.dtype, torch.int64)
    self.assertEqual(t_idx.dtype, torch.int64)
    self.assertEqual(q_idx.shape, t_idx.shape)
    self.assertLessEqual(q_idx.numel(), 2)

  def test_interchangeable_output_format(self):
    hungarian = matcher.HungarianMatcher()
    greedy = matcher.GreedyMatcher()
    outputs = {
        "logits": torch.rand(2, 10, 5),
        "pred_boxes": torch.rand(2, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1, 2], dtype=torch.int64),
            "boxes": torch.rand(2, 4),
        },
        {
            "class_labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.rand(1, 4),
        },
    ]

    h_indices = hungarian(outputs, targets)
    g_indices = greedy(outputs, targets)

    self.assertLen(h_indices, len(targets))
    self.assertLen(g_indices, len(targets))

    for i in range(len(targets)):
      self.assertEqual(h_indices[i][0].dtype, g_indices[i][0].dtype)
      self.assertEqual(h_indices[i][1].dtype, g_indices[i][1].dtype)
      self.assertIsInstance(h_indices[i][0], torch.Tensor)
      self.assertIsInstance(g_indices[i][0], torch.Tensor)
      self.assertEqual(h_indices[i][0].device, g_indices[i][0].device)

  def test_zero_targets(self):
    outputs = {
        "logits": torch.rand(1, 10, 5),
        "pred_boxes": torch.rand(1, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([], dtype=torch.int64),
            "boxes": torch.empty((0, 4)),
        }
    ]
    for matcher_cls in [matcher.HungarianMatcher, matcher.GreedyMatcher]:
      matcher_instance = matcher_cls()
      indices = matcher_instance(outputs, targets)
      self.assertLen(indices, 1)
      self.assertEqual(indices[0][0].numel(), 0)
      self.assertEqual(indices[0][1].numel(), 0)

  def test_zero_queries(self):
    outputs = {
        "logits": torch.rand(1, 0, 5),
        "pred_boxes": torch.rand(1, 0, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.rand(1, 4),
        }
    ]
    for matcher_cls in [matcher.HungarianMatcher, matcher.GreedyMatcher]:
      matcher_instance = matcher_cls()
      indices = matcher_instance(outputs, targets)
      self.assertLen(indices, 1)
      self.assertEqual(indices[0][0].numel(), 0)
      self.assertEqual(indices[0][1].numel(), 0)

  def test_batch_varying_targets(self):
    outputs = {
        "logits": torch.rand(3, 10, 5),
        "pred_boxes": torch.rand(3, 10, 4),
    }
    targets = [
        {
            "class_labels": torch.tensor([1, 2], dtype=torch.int64),
            "boxes": torch.rand(2, 4),
        },
        {
            "class_labels": torch.tensor([], dtype=torch.int64),
            "boxes": torch.empty((0, 4)),
        },
        {
            "class_labels": torch.tensor([3], dtype=torch.int64),
            "boxes": torch.rand(1, 4),
        }
    ]
    for matcher_cls in [matcher.HungarianMatcher, matcher.GreedyMatcher]:
      matcher_instance = matcher_cls()
      indices = matcher_instance(outputs, targets)
      self.assertLen(indices, 3)
      if matcher_cls == matcher.HungarianMatcher:
        self.assertEqual(indices[0][0].numel(), min(10, 2))
        self.assertEqual(indices[1][0].numel(), 0)
        self.assertEqual(indices[2][0].numel(), min(10, 1))
      else:
        # Greedy MNN might not match all boxes if best matches are not mutual.
        self.assertLessEqual(indices[0][0].numel(), min(10, 2))
        self.assertEqual(indices[1][0].numel(), 0)
        self.assertLessEqual(indices[2][0].numel(), min(10, 1))

  def test_greedy_matcher_stability_large(self):
    # This test ensures that the unstable sort mutation will fail.
    # By creating a large cost_matrix of exactly equal values, stable=False
    # will randomly shuffle the order and result in non-diagonal assignments.
    gm = matcher.GreedyMatcher(cost_class=0, cost_bbox=1, cost_giou=0)
    num_queries = 200
    num_targets = 200
    outputs = {
        "logits": torch.zeros(1, num_queries, 2),
        "pred_boxes": torch.zeros(1, num_queries, 4),
    }
    targets = [{
        "class_labels": torch.zeros(num_targets, dtype=torch.int64),
        "boxes": torch.zeros(num_targets, 4),
    }]

    indices = gm(outputs, targets)
    self.assertLen(indices, 1)
    row_indices, col_indices = indices[0]

    expected_indices = torch.arange(num_queries, dtype=torch.int64)
    self.assertTrue(torch.equal(row_indices, expected_indices))
    self.assertTrue(torch.equal(col_indices, expected_indices))

  def test_box_matching(self):
    outputs = {
        "logits": torch.zeros(1, 2, 1),
        "pred_boxes": torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]]
        ),
    }
    targets = [{
        "class_labels": torch.zeros(2, dtype=torch.int64),
        "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
    }]
    indices = matcher.HungarianMatcher(cost_class=0.0)(outputs, targets)
    self.assertEqual(indices[0][1].tolist(), [1, 0])


if __name__ == "__main__":
  absltest.main()
