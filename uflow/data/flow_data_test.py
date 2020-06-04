# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for data loaders with supervised flow."""

import os

from absl.testing import absltest
import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
# pylint:disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from uflow import uflow_plotting
from uflow import uflow_utils
from uflow.data import generic_flow_dataset
from uflow.data import kitti
from uflow.data import sintel
from uflow.data.dataset_locations import dataset_locations


def plot_images(image1, image2, flow, image2_to_image1, plot_dir):
  """Display some images and make sure they look correct."""

  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

  num_rows = 2
  num_columns = 2

  def subplot_at(column, row):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  def post_imshow(label):
    plt.xlabel(label)
    plt.xticks([])
    plt.yticks([])

  plt.figure('eval', [14, 8.5])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow(image1[0])
  post_imshow(label='Image1')

  subplot_at(1, 0)
  plt.imshow(image2[0])
  post_imshow(label='Image2')

  subplot_at(0, 1)
  plt.imshow(uflow_plotting.flow_to_rgb(flow[0].numpy()))
  post_imshow(label='Flow')

  subplot_at(1, 1)
  plt.imshow(image2_to_image1[0])
  post_imshow(label='Image2_to_Image1')

  plt.subplots_adjust(
      left=0.05, bottom=0.05, right=1 - 0.05, top=1, wspace=0.05, hspace=0.05)

  filename = 'demo_flow.png'
  uflow_plotting.save_and_close(os.path.join(plot_dir, filename))


def mock_inference_fn(x, y, input_height, input_width, infer_occlusion):
  del input_height  # unused
  del input_width  # unused
  del y  # unused
  if infer_occlusion:
    return [tf.convert_to_tensor(
        value=np.zeros((x.shape[0], x.shape[1], 2), np.float32))] * 2
  return tf.convert_to_tensor(
      value=np.zeros((x.shape[0], x.shape[1], 2), np.float32))


class FlowDataTest(absltest.TestCase):

  def _check_images_and_flow(self,
                             image1,
                             image2,
                             flow,
                             save_images=False,
                             plot_dir='/tmp/flow_images'):
    self.assertGreaterEqual(np.min(image1), 0.)
    self.assertLessEqual(np.max(image1), 1.)
    # Check that the image2 warped by flow1 into image1 has lower pixelwise
    # error than the unwarped image
    mean_unwarped_diff = np.mean(np.abs(image1 - image2))
    warp = uflow_utils.flow_to_warp(flow)
    image2_to_image1 = uflow_utils.resample(image2, warp)
    mean_warped_diff = np.mean(np.abs(image2_to_image1 - image1))
    if save_images:
      plot_images(image1, image2, flow, image2_to_image1, plot_dir=plot_dir)
    # Check that the warped image has lower pixelwise error than the unwarped.
    self.assertLess(mean_warped_diff, mean_unwarped_diff)

  def test_kitti_eval(self):

    dataset = kitti.make_dataset(
        path=dataset_locations['kitti15-train-pairs'],
        mode='eval')
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(1)
    results = kitti.evaluate(mock_inference_fn, dataset, height=320,
                             width=320)
    expected_keys = kitti.list_eval_keys()
    self.assertEqual(set(expected_keys), set(results.keys()))

  def test_flying_chairs(self):
    dataset = generic_flow_dataset.make_dataset(
        path=dataset_locations['chairs-all'],
        mode='train-sup')
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(1)
    data_it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    data = data_it.next()
    image1 = data[0][:, 0]
    image2 = data[0][:, 1]
    flow = data[1]
    self._check_images_and_flow(
        image1, image2, flow, save_images=False, plot_dir='/tmp/flying_chairs')

  def test_flying_chairs_eval(self):
    dataset = generic_flow_dataset.make_dataset(
        path=dataset_locations['chairs-all'],
        mode='train-sup')
    dataset = dataset.take(1)
    results = generic_flow_dataset.evaluate(
        mock_inference_fn,
        dataset,
        height=200,
        width=400,
        num_plots=0,
        plot_dir='/tmp/flying_chairs',
        has_occlusion=False)
    expected_keys = generic_flow_dataset.list_eval_keys()
    self.assertEqual(set(expected_keys), set(results.keys()))

  def test_sintel(self):
    dataset = sintel.make_dataset(
        path=dataset_locations['sintel-train-clean'],
        mode='eval-occlusion')
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(1)
    data_it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    data = data_it.next()
    image1 = data[0][:, 0]
    image2 = data[0][:, 1]
    flow = data[1]
    self._check_images_and_flow(
        image1, image2, flow, save_images=False, plot_dir='/tmp/sintel')

  def test_sintel_eval(self):
    dataset = sintel.make_dataset(
        path=dataset_locations['sintel-train-clean'],
        mode='eval-occlusion')
    dataset = dataset.take(1)
    results = sintel.evaluate(
        mock_inference_fn,
        dataset,
        height=200,
        width=400,
        num_plots=0,
        plot_dir='/tmp/sintel')
    expected_keys = sintel.list_eval_keys()
    self.assertEqual(set(expected_keys), set(results.keys()))

if __name__ == '__main__':
  absltest.main()
