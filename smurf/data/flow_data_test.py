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

"""Tests for data loaders with supervised flow."""

# pylint:skip-file
import os

from absl.testing import absltest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from smurf import smurf_plotting
from smurf import smurf_utils
from smurf.data import generic_flow_dataset
from smurf.data import kitti
from smurf.data import sintel
from smurf.data.dataset_locations import dataset_locations

matplotlib.use('Agg')  # None-interactive plots do not need tk


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
  plt.imshow(smurf_plotting.flow_to_rgb(flow[0].numpy()))
  post_imshow(label='Flow')

  subplot_at(1, 1)
  plt.imshow(image2_to_image1[0])
  post_imshow(label='Image2_to_Image1')

  plt.subplots_adjust(
      left=0.05, bottom=0.05, right=1 - 0.05, top=1, wspace=0.05, hspace=0.05)

  filename = 'demo_flow.png'
  smurf_plotting.save_and_close(os.path.join(plot_dir, filename))


def inference_fn(x, y, input_height, input_width, infer_occlusion,
                 infer_bw=False):
  del y
  del input_height
  del input_width
  if infer_occlusion and infer_bw:
    return [
        tf.convert_to_tensor(
            value=np.zeros((x.shape[0], x.shape[1], 2), np.float32))
    ] * 3
  if infer_occlusion or infer_bw:
    return [
        tf.convert_to_tensor(
            value=np.zeros((x.shape[0], x.shape[1], 2), np.float32))
    ] * 2
  return tf.convert_to_tensor(
      value=np.zeros((x.shape[0], x.shape[1], 2), np.float32))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_dummy_data(dataset_type='rob'):
  """Creates a simple test pair and writes it into a tmp tfrecord file."""
  # parameters that define a simple example
  bg_height = 32
  bg_width = 64
  fg_height = 16
  fg_width = 16
  h_displacment = 5
  w_displacment = 15
  tmp_output_dir = '/tmp/' + dataset_type + '_dataset_dummy/'
  tmp_output_file = os.path.join(tmp_output_dir, 'dataset_dummy.tfrecord')
  sparse_flow = False if dataset_type == 'jumpy_cars' else True

  # create random background and foreground
  bg = (np.random.standard_normal([bg_height, bg_width, 3]) * 255).astype(
      np.uint8)
  fg = (np.random.standard_normal([fg_height, fg_width, 3]) * 255).astype(
      np.uint8) * .5

  # inits
  image1 = bg
  image2 = bg.copy()
  flow = np.zeros([bg_height, bg_width, 2])
  flow_stationary = np.zeros([bg_height, bg_width, 2])
  if sparse_flow:
    mask_valid = np.random.randint(2, size=(bg_height, bg_width, 1))
  else:
    mask_valid = np.ones([bg_height, bg_width, 1])
  mask_stationary = np.ones([bg_height, bg_width, 1])
  mask_type = np.zeros([bg_height, bg_width, 1])
  run_segment_string = 'simple_example'

  # add foreground object on to the background
  h_start = int((bg_height - fg_height) / 2)
  h_stop = h_start + fg_height
  w_start = int((bg_width - fg_width) / 2)
  w_stop = w_start + fg_width
  image1[h_start:h_stop, w_start:w_stop, :] = fg
  image2[h_start + h_displacment:h_stop + h_displacment,
         w_start + w_displacment:w_stop + w_displacment, :] = fg

  # create flow and make it sparse
  flow[h_start:h_stop, w_start:w_stop, :] = [w_displacment, h_displacment]
  if sparse_flow:
    flow *= mask_valid
    flow_stationary *= mask_valid

  # create stationary and type mask
  mask_stationary[h_start:h_stop, w_start:w_stop, :] = 0
  mask_type[h_start:h_stop, w_start:w_stop, :] = 1

  # get tf tensors in the same format as the original data
  image1 = tf.cast(image1, tf.uint8)
  image2 = tf.cast(image2, tf.uint8)
  flow = tf.cast(flow, tf.float32)
  flow_stationary = tf.cast(flow_stationary, tf.float32)
  mask_valid = tf.cast(mask_valid, tf.uint8)
  mask_stationary = tf.cast(mask_stationary, tf.uint8)
  mask_type = tf.cast(mask_type, tf.uint8)

  if dataset_type == 'rob':
    example_output = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'height':
                    int64_feature(bg_height),
                'width':
                    int64_feature(bg_width),
                'run_segment':
                    bytes_feature(run_segment_string.encode('utf-8')),
                'flow_uv':
                    bytes_feature(flow.numpy().tobytes()),
                'flow_uv_stationary':
                    bytes_feature(flow_stationary.numpy().tobytes()),
                'mask_valid':
                    bytes_feature(mask_valid.numpy().tobytes()),
                'mask_stationary':
                    bytes_feature(mask_stationary.numpy().tobytes()),
                'mask_type':
                    bytes_feature(mask_type.numpy().tobytes()),
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        bytes_feature(tf.image.encode_png(image1).numpy()),
                        bytes_feature(tf.image.encode_png(image2).numpy())
                    ])
            }))
  elif dataset_type == 'wod':
    example_output = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'height': int64_feature(bg_height),
                'width': int64_feature(bg_width),
                'flow_uv': bytes_feature(flow.numpy().tobytes()),
                'flow_valid': bytes_feature(mask_valid.numpy().tobytes()),
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        bytes_feature(tf.image.encode_png(image1).numpy()),
                        bytes_feature(tf.image.encode_png(image2).numpy())
                    ])
            }))
  elif dataset_type == 'jumpy_cars':
    example_output = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'height': int64_feature(bg_height),
                'width': int64_feature(bg_width),
                'flow_uv': bytes_feature(flow.numpy().tobytes()),
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'images':
                    tf.train.FeatureList(feature=[
                        bytes_feature(tf.image.encode_png(image1).numpy()),
                        bytes_feature(tf.image.encode_png(image2).numpy())
                    ])
            }))
  else:
    raise ValueError('Unknown dataset_type: {}.'.format(dataset_type))

  # write to output
  if not os.path.exists(tmp_output_dir):
    os.mkdir(tmp_output_dir)

  with tf.io.TFRecordWriter(tmp_output_file) as record_writer:
    record_writer.write(example_output.SerializeToString())
  return tmp_output_dir


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
    warp = smurf_utils.flow_to_warp(flow)
    image2_to_image1 = smurf_utils.resample(image2, warp)
    mean_warped_diff = np.mean(np.abs(image2_to_image1 - image1))
    if save_images:
      plot_images(image1, image2, flow, image2_to_image1, plot_dir=plot_dir)
    # check that the warped image has lower pixelwise error than the unwarped
    self.assertLess(mean_warped_diff, mean_unwarped_diff)

  def _check_images_and_flow_with_mask(self,
                                       image1,
                                       image2,
                                       flow,
                                       mask,
                                       save_images=False,
                                       plot_dir='/tmp/flow_images'):
    self.assertGreaterEqual(np.min(image1), 0.)
    self.assertLessEqual(np.max(image1), 1.)
    self.assertGreaterEqual(np.min(mask), 0.)
    self.assertLessEqual(np.max(mask), 1.)
    # Check that the image2 warped by flow1 into image1 has lower pixelwise
    # error than the unwarped image
    mean_unwarped_diff = np.mean(mask * np.abs(image1 - image2))
    warp = smurf_utils.flow_to_warp(flow)
    image2_to_image1 = mask * smurf_utils.resample(image2, warp)
    mean_warped_diff = np.mean(mask * np.abs(image2_to_image1 - image1))
    if save_images:
      plot_images(image1, image2, flow, image2_to_image1, plot_dir=plot_dir)
    # check that the warped image has lower pixelwise error than the unwarped
    self.assertLess(mean_warped_diff, mean_unwarped_diff)

  def test_kitti_eval(self):
    dataset = kitti.make_dataset(
        path=dataset_locations['kitti15-train-pairs'],
        mode='eval')
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(1)
    data_it = iter(dataset)
    data = data_it.next()
    image1 = data['images'][:, 0]
    image2 = data['images'][:, 1]
    flow = data['flow']
    self._check_images_and_flow(
        image1, image2, flow, save_images=False, plot_dir='/tmp/kitti')

  def test_flying_chairs(self):
    dataset = generic_flow_dataset.make_dataset(
        path=dataset_locations['chairs-all'],
        mode='train-sup')
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(1)
    data_it = iter(dataset)
    data = data_it.next()
    image1 = data['images'][:, 0]
    image2 = data['images'][:, 1]
    flow = data['flow']
    self._check_images_and_flow(
        image1, image2, flow, save_images=False, plot_dir='/tmp/flying_chairs')

  def test_flying_chairs_eval(self):
    dataset = generic_flow_dataset.make_dataset(
        path=dataset_locations['chairs-all'],
        mode='train-sup')
    dataset = dataset.take(1)
    results = generic_flow_dataset.evaluate(
        inference_fn,
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
    data_it = iter(dataset)
    data = data_it.next()
    image1 = data['images'][:, 0]
    image2 = data['images'][:, 1]
    flow = data['flow']
    self._check_images_and_flow(
        image1, image2, flow, save_images=False, plot_dir='/tmp/sintel')

  def test_sintel_eval(self):
    dataset = sintel.make_dataset(
        path=dataset_locations['sintel-train-clean'],
        mode='eval-occlusion')
    dataset = dataset.take(1)
    results = sintel.evaluate(
        inference_fn,
        dataset,
        height=200,
        width=400,
        num_plots=0,
        plot_dir='/tmp/sintel')
    expected_keys = sintel.list_eval_keys()
    self.assertEqual(set(expected_keys), set(results.keys()))

if __name__ == '__main__':
  absltest.main()
