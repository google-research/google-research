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

"""SMURF plotting.

This library provides some plotting functionality for optical flow.
"""

# pylint:skip-file
import io
import os
import time

import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from smurf import smurf_utils

# How much to scale motion magnitude in visualization.
_FLOW_SCALING_FACTOR = 50.0


def print_log(log, epoch=None, mean_over_num_steps=1):
  """Print log returned by smurf.train(...)."""

  if epoch is None:
    status = ''
  else:
    status = '{} -- '.format(epoch)

  status += 'total-loss: {:.6f}'.format(
      np.mean(log['total-loss'][-mean_over_num_steps:]))

  for key in sorted(log):
    if key not in ['total-loss']:
      loss_mean = np.mean(log[key][-mean_over_num_steps:])
      status += ', {}: {:.6f}'.format(key, loss_mean)
  print(status)


def print_eval(eval_dict):
  """Print eval_dict returned by the eval_function in smurf_main.py."""

  status = ''.join(
      ['{}: {:.6f}, '.format(key, eval_dict[key]) for key in sorted(eval_dict)])
  print(status[:-2])


def time_data_it(data_it, simulated_train_time_ms=100.0):
  print('Timing training iterator with simulated train time of {:.2f}ms'.format(
      simulated_train_time_ms))
  for i in range(100):
    start = time.time()
    _ = data_it.get_next()
    end = time.time()
    print(i, 'Time to get one batch (ms):', (end - start) * 1000)
    if simulated_train_time_ms > 0.0:
      plt.pause(simulated_train_time_ms / 1000.)


def save_image_as_png(image, filename):
  image_uint8 = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
  image_png = tf.image.encode_png(image_uint8)
  tf.io.write_file(filename, image_png)


def plot_data(data_it, plot_dir, num_plots):
  print('Saving images from the dataset to', plot_dir)
  for i, image_batch in enumerate(data_it):
    if i >= num_plots:
      break
    for j, images in enumerate(image_batch['images']):
      for k, image in enumerate(images):
        save_image_as_png(
            image, os.path.join(plot_dir, '{}_{}_{}.png'.format(i, j, k)))


def flow_to_rgb(flow):
  """Compute an RGB visualization of a flow field."""
  shape = tf.cast(tf.shape(flow), tf.float32)
  height, width = shape[-3], shape[-2]
  scaling = _FLOW_SCALING_FACTOR / (height**2 + width**2)**0.5

  # Compute angles and lengths of motion vectors.
  motion_angle = tf.atan2(flow[Ellipsis, 1], flow[Ellipsis, 0])
  motion_magnitude = (flow[Ellipsis, 1]**2 + flow[Ellipsis, 0]**2)**0.5

  # Visualize flow using the HSV color space, where angles are represented by
  # hue and magnitudes are represented by saturation.
  flow_hsv = tf.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                       tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0),
                       tf.ones_like(motion_magnitude)],
                      axis=-1)

  # Transform colors from HSV to RGB color space for plotting.
  return tf.image.hsv_to_rgb(flow_hsv)


def complete_paper_plot(plot_dir,
                        index,
                        image1,
                        image2,
                        flow_uv,
                        ground_truth_flow_uv=None,
                        flow_valid_occ=None,
                        predicted_occlusion=None,
                        ground_truth_occlusion=None,
                        frame_skip=None):

  def post_imshow(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = str(index) + '_' + name
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

  warp = smurf_utils.flow_to_warp(tf.convert_to_tensor(flow_uv))
  image1_reconstruction = smurf_utils.resample(tf.expand_dims(image2, axis=0),
                                               tf.expand_dims(warp, axis=0))[0]
  flow_uv = -flow_uv[:, :, ::-1]
  if ground_truth_flow_uv is not None:
    ground_truth_flow_uv = -ground_truth_flow_uv[:, :, ::-1]
  plt.figure()
  plt.clf()

  plt.imshow(image1)
  post_imshow('image1_rgb', plot_dir)

  plt.imshow(image1_reconstruction)
  post_imshow('image1_reconstruction_rgb', plot_dir)

  plt.imshow(image1_reconstruction * predicted_occlusion)
  post_imshow('image1_reconstruction_occlusions_rgb', plot_dir)

  plt.imshow((image1 + image2) / 2.)
  post_imshow('image_rgb', plot_dir)

  plt.imshow(flow_to_rgb(flow_uv))
  post_imshow('predicted_flow', plot_dir)

  if ground_truth_flow_uv is not None and flow_valid_occ is not None:
    plt.imshow(flow_to_rgb(ground_truth_flow_uv * flow_valid_occ))
    post_imshow('ground_truth_flow', plot_dir)
    endpoint_error = np.sum(
        (ground_truth_flow_uv - flow_uv)**2, axis=-1, keepdims=True)**0.5
    plt.imshow(
        (endpoint_error * flow_valid_occ)[:, :, 0],
        cmap='viridis',
        vmin=0,
        vmax=40)
    post_imshow('flow_error', plot_dir)

  if predicted_occlusion is not None:
    plt.imshow((predicted_occlusion[:, :, 0]) * 255, cmap='Greys')
    post_imshow('predicted_occlusion', plot_dir)

  if ground_truth_occlusion is not None:
    plt.imshow((ground_truth_occlusion[:, :, 0]) * 255, cmap='Greys')
    post_imshow('ground_truth_occlusion', plot_dir)

  plt.close('all')
