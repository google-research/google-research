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

"""Tests for evaluate_vrd_lib."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from visual_relationship.evaluation import evaluate_vrd_lib

FLAGS = flags.FLAGS

# Assumes $PWD == "google_research/".
ROOT = 'visual_relationship/evaluation/testdata'


def create_box(
    image_id = 'image',
    entity_id = 'obj',
    xmin = 0.0,
    ymin = 0.0,
    width = 0.5,
    height = 0.5
):
  xmax = min(1., xmin + width)
  ymax = min(1., ymin + height)
  return evaluate_vrd_lib.Box(image_id, entity_id, ymin, xmin, ymax, xmax)


class EvaluateVrdLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.test_groundtruth = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_vrd.csv')
    self.test_object = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_objects.csv')
    self.test_prediction = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_prediction.csv')

  def test_load_groundtruth(self):
    results = evaluate_vrd_lib.load_groundtruth(self.test_groundtruth,
                                                self.test_object)
    self.assertLen(results, 2)

  def test_load_prediction(self):
    results = evaluate_vrd_lib.load_prediction(self.test_prediction)
    self.assertLen(results, 2)

  @parameterized.parameters(
      ([3, 12, 7], dict(precision=0.2, recall=0.3, fscore=0.24)),
      ([0, 12, 7], dict(precision=0.0, recall=0.0, fscore=0.)))
  def test_compute_metrics(self, inputs, outputs):
    metrics = evaluate_vrd_lib.compute_metrics(*inputs)
    self.assertDictEqual(metrics, outputs)

  @parameterized.parameters(([-1, 12, 7],), ([3, -1, 7],), ([0, 12, -1],))
  def test_compute_metrics_invalid(self, inputs):
    with self.assertRaises(ValueError):
      evaluate_vrd_lib.compute_metrics(*inputs)

  def test_reverse_object_order(self):
    results = evaluate_vrd_lib.load_groundtruth(self.test_groundtruth,
                                                self.test_object)
    lengths = {
        example_id: len(records) for example_id, records in results.items()
    }
    reversed_results = evaluate_vrd_lib.reverse_object_order(results)
    for example_id, length in lengths.items():
      with self.subTest():
        self.assertLen(reversed_results[example_id], length * 2)

  def test_compute_area(self):
    box = create_box()
    area = evaluate_vrd_lib.compute_area(box)
    self.assertAlmostEqual(area, 0.25)

  def test_compute_iou(self):
    box_a = create_box()
    box_b = create_box(xmin=0.25)
    iou = evaluate_vrd_lib.compute_iou(box_a, box_b)
    self.assertAlmostEqual(iou, 1. / 3.)

  @parameterized.parameters(
      (
          evaluate_vrd_lib.Record(create_box(), create_box(), 0, 0),
          (0., 0.5), (0., 0.5), 'size', True
      ),
      (
          evaluate_vrd_lib.Record(create_box(), create_box(), 0, 0),
          (0., 0.2), (0., 0.5), 'size', False
      ),
      (
          evaluate_vrd_lib.Record(create_box(), create_box(), 0, 0),
          (0.0, 0.5), (0.0, 0.5), 'vertical_position', True
      ),
      (
          evaluate_vrd_lib.Record(create_box(), create_box(), 0, 0),
          (0.0, 0.5), (0.5, 1.0), 'horizontal_position', False
      ),
  )
  def test_filter_boundingbox_output(self, record, rng_a, rng_b, attr, target):
    filter_fn = evaluate_vrd_lib.get_filter_boundingbox_fn(rng_a, rng_b, attr)
    output = filter_fn(record)
    self.assertEqual(output, target)

  def test_filter_boundingbox_invalid_attr(self):
    record = evaluate_vrd_lib.Record(create_box(), create_box(), 0, 0)
    rng_a = (0.0, 0.5)
    rng_b = (0.0, 0.5)
    attr = 'aspect_ratio'
    filter_fn = evaluate_vrd_lib.get_filter_boundingbox_fn(rng_a, rng_b, attr)
    with self.assertRaises(NotImplementedError):
      filter_fn(record)


class VRDEvaluatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    test_groundtruth = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_vrd.csv')
    test_object = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_objects.csv')
    self.evaluator = evaluate_vrd_lib.VRDEvaluator(
        test_groundtruth, test_object)
    self.test_prediction = os.path.join(
        FLAGS.test_srcdir, f'{ROOT}/across_images_prediction.csv')

  @parameterized.parameters(
      {
          'box_a': create_box(),
          'box_b': create_box(),
          'success': True,
      },
      {
          'box_a': create_box(),
          'box_b': create_box(xmin=0.4),
          'success': False,
      },
      {
          'box_a': create_box(image_id='image_a'),
          'box_b': create_box(image_id='image_b'),
          'success': False,
      },
      {
          'box_a': create_box(entity_id='car'),
          'box_b': create_box(entity_id='tree'),
          'success': True,
          'check_entity': False,
      },
      {
          'box_a': create_box(entity_id='car'),
          'box_b': create_box(entity_id='tree'),
          'success': False,
          'check_entity': True,
      },
  )
  def test_is_success_detection(self,
                                box_a,
                                box_b,
                                success,
                                check_entity=False):
    self.evaluator.check_entity = check_entity
    result = self.evaluator.is_success_detection(box_a, box_b)
    self.assertEqual(result, success)

  @parameterized.parameters(
      {
          'groundtruth': evaluate_vrd_lib.Record(
              create_box(), create_box(), 1, 1),
          'prediction': evaluate_vrd_lib.Record(
              create_box(), create_box(), 1, 2),
          'correct': {
              evaluate_vrd_lib.VRDAttribute.OCCLUSION: True,
              evaluate_vrd_lib.VRDAttribute.DISTANCE: False,
          },
      },
      {
          'groundtruth': evaluate_vrd_lib.Record(
              create_box(), create_box(), 1, 1),
          'prediction': evaluate_vrd_lib.Record(
              create_box(), create_box(xmin=0.4), 1, 1),
          'correct': {
              evaluate_vrd_lib.VRDAttribute.OCCLUSION: False,
              evaluate_vrd_lib.VRDAttribute.DISTANCE: False,
          },
      },
      {
          'groundtruth': evaluate_vrd_lib.Record(
              create_box(ymin=0.3), create_box(), 1, 1),
          'prediction': evaluate_vrd_lib.Record(
              create_box(), create_box(), 1, 1),
          'correct': {
              evaluate_vrd_lib.VRDAttribute.OCCLUSION: False,
              evaluate_vrd_lib.VRDAttribute.DISTANCE: False,
          },
      },
  )
  def test_is_correct_prediction(self, prediction, groundtruth, correct):
    for attr, expected in correct.items():
      with self.subTest(attribute=attr):
        result = self.evaluator.is_correct_prediction(
            prediction, groundtruth, attr=attr)
        self.assertEqual(result, expected)

  def test_compute_metrics(self):
    predictions = evaluate_vrd_lib.load_prediction(self.test_prediction)

    precision = [1., 1., 0.5, 0., 0.75, 1., 0., 1., 0.5, 0.75]
    recall = [0.5, 0.5, 0.5, 0., 0.5, 0.5, 0., 1., 0.5, 0.5]
    results = self.evaluator.compute_metrics(predictions)
    with self.subTest(msg='precision'):
      self.assertListEqual(results['precision'].tolist(), precision)
    with self.subTest(msg='recall'):
      self.assertListEqual(results['recall'].tolist(), recall)

    precision = [0., 0., 0.5, 0., 0.5, 0., 0., 0., 0.5, 0.5]
    recall = [0., 0., 1., 0., 0.5, 0., 0., 0., 0.5, 0.5]
    filter_fn = evaluate_vrd_lib.get_filter_boundingbox_fn(
        (0., 0.5), (0., 0.5), 'vertical_position')
    results = self.evaluator.compute_metrics(predictions, filter_fn=filter_fn)
    with self.subTest(msg='filtered-precision'):
      self.assertListEqual(results['precision'].tolist(), precision)
    with self.subTest(msg='filtered-recall'):
      self.assertListEqual(results['recall'].tolist(), recall)


if __name__ == '__main__':
  absltest.main()
