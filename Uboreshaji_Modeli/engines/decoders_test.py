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

from absl.testing import absltest
import torch

from Uboreshaji_Modeli.engines import decoders


class DecodersTest(absltest.TestCase):

  def test_bounding_box_decoder_quantized(self):
    decoder = decoders.BoundingBoxDecoder()
    processor = None  # Not used by BoundingBoxDecoder

    # Quantized location tokens representing normalized coordinates:
    # y1=102 (102/1023 ~= 0.1), x1=204 (204/1023 ~= 0.2)
    # y2=511 (511/1023 ~= 0.5), x2=613 (613/1023 ~= 0.6)
    outputs = "<loc0102><loc0204><loc0511><loc0613> leaf"
    results = decoder.decode(processor, outputs)

    with self.subTest("Test result structure"):
      self.assertLen(results, 1)

    with self.subTest("Test labels"):
      self.assertLen(results[0]["labels"], 1)
      self.assertEqual(results[0]["labels"][0], "leaf")

    with self.subTest("Test bounding boxes"):
      boxes = results[0]["boxes"]
      self.assertEqual(boxes.shape, (1, 4))
      # Allow a tiny epsilon for rounding in grid conversions
      torch.testing.assert_close(
          boxes[0],
          torch.tensor([0.2, 0.1, 0.6, 0.5], dtype=torch.float32),
          atol=1e-3,
          rtol=1e-3,
      )

  def test_bounding_box_decoder_empty(self):
    decoder = decoders.BoundingBoxDecoder()
    processor = None
    outputs = "there are no objects in the image"
    results = decoder.decode(processor, outputs)

    self.assertLen(results, 1)
    self.assertEmpty(results[0]["labels"])
    self.assertEqual(results[0]["boxes"].shape, (0, 4))

  def test_polygon_segmentation_decoder_quantized(self):
    decoder = decoders.PolygonSegmentationDecoder()
    processor = None

    outputs = "polygon <loc0102><loc0204> <loc0511><loc0613>"
    results = decoder.decode(processor, outputs)

    with self.subTest("Test result structure"):
      self.assertLen(results, 1)
      self.assertLen(results[0]["polygons"], 1)

    with self.subTest("Test polygon points"):
      poly = results[0]["polygons"][0]
      self.assertLen(poly, 2)
      # Verify points: (x, y)
      self.assertSequenceAlmostEqual(poly[0], [0.2, 0.1], delta=1e-3)
      self.assertSequenceAlmostEqual(poly[1], [0.6, 0.5], delta=1e-3)


if __name__ == "__main__":
  absltest.main()
