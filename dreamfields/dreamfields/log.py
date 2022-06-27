# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Visualization and logging."""

import io
import json
import math
import os

from absl import logging
import jax.numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import mediapy as media
import ml_collections
import numpy as onp
import tensorflow as tf
import tensorflow.io.gfile as gfile


def scale_depth(depth, min_depth=4 - onp.sqrt(3), max_depth=4 + onp.sqrt(3)):
  """Clips a depth image to a unit cube, scales to [0, 1] then color maps."""
  depth_scaled = onp.clip(depth, min_depth, max_depth)
  depth_scaled = depth_scaled - onp.min(depth_scaled)
  depth_scaled = depth_scaled / onp.max(depth_scaled)

  try:
    cmap = cm.get_cmap('turbo')
  except ValueError:
    # Fallback to jet cmap for older versions of matplotlib, <3.3
    cmap = cm.get_cmap('jet')
  depth_colored = cmap(depth_scaled)
  alpha = depth_colored[Ellipsis, 3:]
  depth_colored = depth_colored[Ellipsis, :3] * alpha + (1 - alpha)

  return depth_colored


def plot_to_image(figure, close_fig=True):
  """Converts the matplotlib plot specified by 'figure' to a PNG image.

  The supplied figure is closed and inaccessible after this call.
  Source: https://www.tensorflow.org/tensorboard/image_summaries

  Args:
    figure: matplotlib figure object.
    close_fig (bool): whether to close figure after rasterization. Default: True

  Returns:
    image: rasterized plot.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside notebook
  if close_fig:
    plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def make_image_grid(images, nrow=10):
  """Given a list of images, tile into a single grid image."""
  ncol = int(math.ceil(len(images) / nrow))
  to_pad = nrow - len(images) % nrow
  images = np.stack(images)
  if images.ndim == 3:
    # Add channel dimension
    images = images[Ellipsis, None]
  H, W, C = images.shape[1:]  # pylint: disable=invalid-name
  if to_pad and to_pad != nrow:
    padding_frames = np.zeros((to_pad, H, W, C), dtype=images.dtype)
    images = np.concatenate([images, padding_frames], axis=0)
  images = np.reshape(images, (ncol, nrow, H, W, C))
  images = np.moveaxis(images, 1, 2)  # nc, nr, h, w, c --> nc, h, nr, w, c
  return images.reshape(1, ncol * H, nrow * W, C)


def log_video(writer,
              video,
              tb_key,
              name,
              step,
              work_unit_dir,
              save_raw=False,
              scale=False):
  """Save video frames to tensorboard and a file."""
  video_raw = video
  if scale:
    video = scale_depth(video)

  if writer is not None:
    logging.info('Logging video frames')
    writer.write_images(step, {f'{tb_key}/{name}': make_image_grid(video)})

  filename = f'{tb_key}_{name}_{step:05d}.mp4'
  local_path = os.path.join('/tmp', filename)
  logging.info('Writing video to %s', local_path)
  media.write_video(local_path, video, fps=30)

  wu_path = os.path.join(work_unit_dir, filename)
  logging.info('Copying video to %s', wu_path)
  gfile.copy(local_path, wu_path, overwrite=True)
  gfile.remove(local_path)

  if save_raw:
    # save raw floating point values to scale depth properly
    raw_filename = f'{tb_key}_{name}_{step:05d}.npy'
    raw_path = os.path.join(work_unit_dir, raw_filename)
    logging.info('Saving raw video to %s', raw_path)
    with gfile.GFile(raw_path, 'wb') as raw_f:
      onp.save(raw_f, video_raw)

  logging.info('Done logging video.')


def write_config_json(config, work_unit_dir):
  path = os.path.join(work_unit_dir, 'config.json')
  if gfile.exists(path):
    return
  with gfile.GFile(path, 'w') as f:
    f.write(config.to_json_best_effort(sort_keys=True, indent=4) + '\n')


def load_config_json(work_unit_dir):
  config_path = os.path.join(work_unit_dir, 'config.json')
  with gfile.GFile(config_path, 'r') as config_f:
    config = json.load(config_f)

  return ml_collections.ConfigDict(config)
