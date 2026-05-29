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

"""Simple tests for box_utils to verify coverage."""

from absl.testing import absltest
from absl.testing import parameterized
from models.owlv2 import box_utils
import torch


class BoxUtilsTest(absltest.TestCase):

  def test_rescale_bboxes(self):
    bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    size = (1000.0, 800.0)
    rescaled = box_utils.rescale_bboxes(bboxes, size)
    expected = torch.tensor([[320, 400, 480, 600]], dtype=torch.float32)
    torch.testing.assert_close(rescaled, expected)

  def test_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    iou, _ = box_utils.box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertEqual(iou.shape, (2, 1))
    self.assertEqual(iou[0, 0], 1.0)
    self.assertAlmostEqual(iou[1, 0].item(), 1 / 7)

  def test_generalized_box_iou(self):
    boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    giou = box_utils.generalized_box_iou(boxes1=boxes1, boxes2=boxes2)
    self.assertAlmostEqual(giou[0, 0].item(), 1.0)

  def test_is_valid_box(self):
    self.assertTrue(box_utils.is_valid_box([10, 20, 30, 40]))
    self.assertFalse(box_utils.is_valid_box([10, 20, 0, 40]))
    self.assertFalse(box_utils.is_valid_box([10, 20, 30, 0]))
    self.assertFalse(box_utils.is_valid_box([10, 20, -1, 40]))
    self.assertFalse(box_utils.is_valid_box([10, 20, 30]))

    self.assertTrue(box_utils.is_valid_box([10, 20, 30, 40], "xyxy"))
    self.assertFalse(box_utils.is_valid_box([10, 20, 10, 40], "xyxy"))
    self.assertFalse(box_utils.is_valid_box([10, 20, 30, 20], "xyxy"))
    self.assertFalse(box_utils.is_valid_box([10, 20, 5, 40], "xyxy"))


class CocoToXyxyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="basic_conversion",
          coco_box=[10, 20, 100, 200],
          expected_xyxy=[10, 20, 110, 220],
      ),
      dict(
          testcase_name="zero_size",
          coco_box=[50, 50, 0, 0],
          expected_xyxy=[50, 50, 50, 50],
      ),
  )
  def test_coco_to_xyxy(self, coco_box, expected_xyxy):
    result = box_utils.coco_to_xyxy(coco_box)
    self.assertEqual(result, expected_xyxy)


if __name__ == "__main__":
  absltest.main()
