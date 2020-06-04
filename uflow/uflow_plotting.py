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

"""UFlow plotting.

This library provides some plotting functionality for optical flow.
"""

import io
import os
import time

import matplotlib
matplotlib.use('Agg')  # None-interactive plots do not need tk
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf

# How much to scale motion magnitude in visualization.
_FLOW_SCALING_FACTOR = 50.0

# pylint:disable=g-long-lambda


def print_log(log, epoch=None, mean_over_num_steps=1):
  """Print log returned by UFlow.train(...)."""

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
  """Prints eval_dict to console."""

  status = ''.join(
      ['{}: {:.6f}, '.format(key, eval_dict[key]) for key in sorted(eval_dict)])
  print(status[:-2])


def plot_log(log, plot_dir):
  plt.figure(1)
  plt.clf()

  keys = ['total-loss'
         ] + [key for key in sorted(log) if key not in ['total-loss']]
  for key in keys:
    plt.plot(log[key], '--' if key == 'total-loss' else '-', label=key)
  plt.legend()
  save_and_close(os.path.join(plot_dir, 'log.png'))


def save_and_close(filename):
  """Save figures."""

  # Create a python byte stream into which to write the plot image.
  buf = io.BytesIO()

  # Save the image into the buffer.
  plt.savefig(buf, format='png')

  # Seek the buffer back to the beginning, then either write to file or stdout.
  buf.seek(0)
  with tf.io.gfile.GFile(filename, 'w') as f:
    f.write(buf.read(-1))
  plt.close('all')


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
  for i, (image_batch, _) in enumerate(data_it):
    if i >= num_plots:
      break
    for j, image_sequence in enumerate(image_batch):
      for k, image in enumerate(image_sequence):
        save_image_as_png(
            image, os.path.join(plot_dir, '{}_{}_{}.png'.format(i, j, k)))


def flow_to_rgb(flow):
  """Computes an RGB visualization of a flow field."""
  shape = flow.shape
  is_graph_mode = False
  if not isinstance(shape[0], int):  # In graph mode, this is a Dimension object
    is_graph_mode = True
    shape = [s.value for s in shape]
  height, width = [float(s) for s in shape[-3:-1]]
  scaling = _FLOW_SCALING_FACTOR / (height**2 + width**2)**0.5

  # Compute angles and lengths of motion vectors.
  if is_graph_mode:
    motion_angle = tf.atan2(flow[Ellipsis, 1], flow[Ellipsis, 0])
  else:
    motion_angle = np.arctan2(flow[Ellipsis, 1], flow[Ellipsis, 0])
  motion_magnitude = (flow[Ellipsis, 1]**2 + flow[Ellipsis, 0]**2)**0.5

  # Visualize flow using the HSV color space, where angles are represented by
  # hue and magnitudes are represented by saturation.
  if is_graph_mode:
    flow_hsv = tf.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                         tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0),
                         tf.ones_like(motion_magnitude)],
                        axis=-1)
  else:
    flow_hsv = np.stack([((motion_angle / np.math.pi) + 1.) / 2.,
                         np.clip(motion_magnitude * scaling, 0.0, 1.0),
                         np.ones_like(motion_magnitude)],
                        axis=-1)

  # Transform colors from HSV to RGB color space for plotting.
  if is_graph_mode:
    return tf.image.hsv_to_rgb(flow_hsv)
  return matplotlib.colors.hsv_to_rgb(flow_hsv)


def flow_tensor_to_rgb_tensor(motion_image):
  """Visualizes flow motion image as an RGB image.

  Similar as the flow_to_rgb function, but with tensors.

  Args:
    motion_image: A tensor either of shape [batch_sz, height, width, 2] or of
      shape [height, width, 2]. motion_image[..., 0] is flow in x and
      motion_image[..., 1] is flow in y.

  Returns:
    A visualization tensor with same shape as motion_image, except with three
    channels. The dtype of the output is tf.uint8.
  """
  # sqrt(a^2 + b^2)
  hypot = lambda a, b: (tf.cast(a, tf.float32)**2.0 + tf.cast(b, tf.float32)**
                        2.0)**0.5
  height, width = motion_image.get_shape().as_list()[-3:-1]
  scaling = _FLOW_SCALING_FACTOR / hypot(height, width)
  x, y = motion_image[Ellipsis, 0], motion_image[Ellipsis, 1]
  motion_angle = tf.atan2(y, x)
  motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
  motion_magnitude = hypot(y, x)
  motion_magnitude = tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0)
  value_channel = tf.ones_like(motion_angle)
  flow_hsv = tf.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
  flow_rgb = tf.image.convert_image_dtype(
      tf.image.hsv_to_rgb(flow_hsv), tf.uint8)
  return flow_rgb


def post_imshow(label=None, height=None, width=None):
  plt.xticks([])
  plt.yticks([])
  if label is not None:
    plt.xlabel(label)
  if height is not None and width is not None:
    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.gca().invert_yaxis()


def plot_flow(image1, image2, flow, filename, plot_dir):
  """Overlay images, plot those and flow, and save the result to file."""
  num_rows = 2
  num_columns = 1

  def subplot_at(column, row):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  height, width = [float(s) for s in image1.shape[-3:-1]]
  plt.figure('plot_flow', [10. * width / (2 * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow((image1 + image2) / 2.)
  post_imshow()

  subplot_at(0, 1)
  plt.imshow(flow_to_rgb(flow))
  post_imshow()

  plt.subplots_adjust(
      left=0.001, bottom=0.001, right=1, top=1, wspace=0.01, hspace=0.01)

  save_and_close(os.path.join(plot_dir, filename))


def plot_movie_frame(plot_dir, index, image, flow_uv, frame_skip=None):
  """Plots a frame suitable for making a movie."""

  def save_fig(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = '{:06d}_{}'.format(index, name)
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

  flow_uv = -flow_uv[:, :, ::-1]
  plt.figure()
  plt.clf()

  minimal_frame = np.concatenate([image, flow_to_rgb(flow_uv)], axis=0)
  plt.imshow(minimal_frame)
  save_fig('minimal_video_frame', plot_dir)

  plt.close('all')


def plot_masks(image, masks, filename, plot_dir):
  """Overlay images, plot those and flow, and save the result to file."""
  num_rows = 2
  num_columns = 1

  def subplot_at(column, row):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  def ticks():
    plt.xticks([])
    plt.yticks([])

  height, width = [float(s) for s in image.shape[-3:-1]]
  plt.figure('plot_flow', [10. * width / (2 * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow(image)
  ticks()

  subplot_at(0, 1)
  plt.imshow(masks)
  ticks()

  plt.subplots_adjust(
      left=0.001, bottom=0.001, right=1, top=1, wspace=0.01, hspace=0.01)

  save_and_close(os.path.join(plot_dir, filename))


def complete_paper_plot(plot_dir,
                        index,
                        image1,
                        image2,
                        flow_uv,
                        ground_truth_flow_uv,
                        flow_valid_occ,
                        predicted_occlusion,
                        ground_truth_occlusion,
                        frame_skip=None):
  """Plots rgb image, flow, occlusions, ground truth, all as separate images."""

  def save_fig(name, plot_dir):
    plt.xticks([])
    plt.yticks([])
    if frame_skip is not None:
      filename = str(index) + '_' + str(frame_skip) + '_' + name
      plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    else:
      filepath = str(index) + '_' + name
      plt.savefig(os.path.join(plot_dir, filepath), bbox_inches='tight')
    plt.clf()

  flow_uv = -flow_uv[:, :, ::-1]
  ground_truth_flow_uv = -ground_truth_flow_uv[:, :, ::-1]
  plt.figure()
  plt.clf()

  plt.imshow((image1 + image2) / 2.)
  save_fig('image_rgb', plot_dir)

  plt.imshow(flow_to_rgb(flow_uv))
  save_fig('predicted_flow', plot_dir)

  plt.imshow(flow_to_rgb(ground_truth_flow_uv * flow_valid_occ))
  save_fig('ground_truth_flow', plot_dir)

  endpoint_error = np.sum(
      (ground_truth_flow_uv - flow_uv)**2, axis=-1, keepdims=True)**0.5
  plt.imshow(
      (endpoint_error * flow_valid_occ)[:, :, 0],
      cmap='viridis',
      vmin=0,
      vmax=40)
  save_fig('flow_error', plot_dir)

  plt.imshow((predicted_occlusion[:, :, 0]) * 255, cmap='Greys')
  save_fig('predicted_occlusion', plot_dir)

  plt.imshow((ground_truth_occlusion[:, :, 0]) * 255, cmap='Greys')
  save_fig('ground_truth_occlusion', plot_dir)

  plt.close('all')


def plot_selfsup(key, images, flows, teacher_flow, student_flow, error,
                 teacher_mask, student_mask, mask, selfsup_transform_fns,
                 plot_dir):
  """Plots some data relevant to self-supervision."""
  num_rows = 3
  num_columns = 3

  def subplot_at(row, column):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  i, j, _ = key
  height, width = [float(s.value) for s in images[i].shape[-3:-1]]
  plt.figure('plot_flow',
             [10. * num_columns * width / (num_rows * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow((images[i][0] + images[j][0]) / 2., interpolation='nearest')
  post_imshow('Teacher images')

  subplot_at(0, 1)

  transformed_image_i = selfsup_transform_fns[0](
      images[i], i_or_ij=i, is_flow=False)
  transformed_image_j = selfsup_transform_fns[0](
      images[j], i_or_ij=j, is_flow=False)
  plt.imshow(
      (transformed_image_i[0] + transformed_image_j[0]) / 2.,
      interpolation='nearest')
  post_imshow('Student images')

  subplot_at(0, 2)
  plt.imshow(
      teacher_mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Teacher mask')

  subplot_at(1, 0)
  plt.imshow(
      flow_to_rgb(flows[(i, j, 'original-teacher')][0][0].numpy()),
      interpolation='nearest')
  post_imshow('Teacher flow')

  subplot_at(1, 1)
  plt.imshow(flow_to_rgb(student_flow[0].numpy()), interpolation='nearest')
  post_imshow('Student flow')

  subplot_at(1, 2)
  plt.imshow(
      student_mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Student mask')

  subplot_at(2, 0)
  plt.imshow(flow_to_rgb(teacher_flow[0].numpy()), interpolation='nearest')
  post_imshow('Teacher flow (projected)')

  subplot_at(2, 1)
  plt.imshow(
      error[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=3.,
      cmap='viridis')
  post_imshow('Error')

  subplot_at(2, 2)
  plt.imshow(
      mask[0, Ellipsis, 0],
      interpolation='nearest',
      vmin=0.,
      vmax=1.,
      cmap='viridis')
  post_imshow('Combined mask')

  plt.subplots_adjust(
      left=0.001, bottom=0.05, right=1, top=1, wspace=0.01, hspace=0.1)

  filename = '{}.png'.format(time.time())
  save_and_close(os.path.join(plot_dir, filename))


def plot_smoothness(key, images, weights_xx, weights_yy, flow_gxx_abs,
                    flow_gyy_abs, flows, plot_dir):
  """Plots data relevant to smoothness."""
  num_rows = 3
  num_columns = 3

  def subplot_at(row, column):
    plt.subplot(num_rows, num_columns, 1 + column + row * num_columns)

  i, j, c = key
  height, width = [float(s.value) for s in images[i].shape[-3:-1]]
  plt.figure('plot_flow',
             [10. * num_columns * width / (num_rows * height), 10.])
  plt.clf()

  subplot_at(0, 0)
  plt.imshow(images[i][0], interpolation='nearest')
  post_imshow('Image')

  subplot_at(1, 0)
  plt.imshow(
      weights_xx[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Weights dxx {}'.format(np.mean(weights_xx[0, Ellipsis, 0])))

  subplot_at(2, 0)
  plt.imshow(
      weights_yy[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Weights dyy {}'.format(np.mean(weights_yy[0, Ellipsis, 0])))

  subplot_at(0, 1)
  plt.imshow(
      flow_to_rgb(flows[(i, j, c)][0][0].numpy()), interpolation='nearest')
  post_imshow('Flow')

  subplot_at(1, 1)
  plt.imshow(
      flow_gxx_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('FLow dxx')

  subplot_at(2, 1)
  plt.imshow(
      flow_gyy_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Flow dyy')

  subplot_at(1, 2)
  plt.imshow(
      weights_xx[0, Ellipsis, 0] * flow_gxx_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Loss dxx')

  subplot_at(2, 2)
  plt.imshow(
      weights_yy[0, Ellipsis, 0] * flow_gyy_abs[0, Ellipsis, 0],
      interpolation='nearest',
      cmap='viridis',
      vmin=0.0,
      vmax=1.0)
  post_imshow('Loss dyy')

  plt.subplots_adjust(
      left=0.001, bottom=0.05, right=1, top=1, wspace=0.01, hspace=0.1)

  filename = '{}.png'.format(time.time())
  save_and_close(os.path.join(plot_dir, filename))
