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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import time

from absl import logging
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1 import gfile
from tensorflow.compat.v1.core.framework.summary_pb2 import Summary
from tensorflow.compat.v1.core.util.event_pb2 import Event


def pack_images(images, rows, cols):
  """Helper utility to make a tiled field of images from numpy arrays.

  Taken from Jaxboard.

  Args:
    images: Image tensor in shape [N, W, H, C].
    rows: Number of images per row in tiled image.
    cols: Number of images per column in tiled image.

  Returns:
    A tiled image of shape [W * rows, H * cols, C].
    Truncates incomplete rows.
  """
  shape = np.shape(images)
  width, height, depth = shape[-3:]
  images = np.reshape(images, (-1, width, height, depth))
  batch = np.shape(images)[0]
  rows = np.minimum(rows, batch)
  cols = np.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = np.reshape(images, (rows, cols, width, height, depth))
  images = np.transpose(images, [0, 2, 1, 3, 4])
  images = np.reshape(images, [rows * width, cols * height, depth])
  return images


class SummaryWriter(object):
  """Tensorflow summary writer inspired by Jaxboard.

  This version doesn't try to avoid Tensorflow dependencies, because this
  project uses Tensorflow.
  """

  def __init__(self, dir, write_graph=True):
    if not gfile.IsDirectory(dir):
      gfile.MakeDirs(dir)
    self.writer = tf.summary.FileWriter(
        dir, graph=tf.get_default_graph() if write_graph else None)

  def flush(self):
    self.writer.flush()

  def close(self):
    self.writer.close()

  def _write_event(self, summary_value, step):
    self.writer.add_event(
        Event(
            wall_time=round(time.time()),
            step=step,
            summary=Summary(value=[summary_value])))

  def scalar(self, tag, value, step):
    self._write_event(Summary.Value(tag=tag, simple_value=float(value)), step)

  def image(self, tag, image, step):
    image = np.asarray(image)
    if image.ndim == 2:
      image = image[:, :, None]
    if image.shape[-1] == 1:
      image = np.repeat(image, 3, axis=-1)

    bytesio = io.BytesIO()
    PIL.Image.fromarray(image).save(bytesio, 'PNG')
    image_summary = Summary.Image(
        encoded_image_string=bytesio.getvalue(),
        colorspace=3,
        height=image.shape[0],
        width=image.shape[1])
    self._write_event(Summary.Value(tag=tag, image=image_summary), step)

  def images(self, tag, images, step, square=True):
    """Saves (rows, cols) tiled images from onp.ndarray.

    This truncates the image batch rather than padding
    if it doesn't fill the final row.
    """
    images = np.asarray(images)
    n_images = len(images)

    if square:
      rows = cols = int(np.sqrt(n_images))
    else:
      rows = 1
      cols = n_images

    tiled_images = pack_images(images, rows, cols)
    self.image(tag, tiled_images, step=step)


class Log(object):
  """Logging to Tensorboard and the Python logger at the same time."""

  def __init__(self, logdir, write_graph=True):
    self.logdir = logdir
    # Tensorboard
    self.summary_writer = SummaryWriter(logdir, write_graph=write_graph)

  def write(self, key_prefix, info_dicts, step):
    log_items = []
    for key in info_dicts[-1]:
      # average the log values over time
      key_with_prefix = '{}/{}'.format(key_prefix, key)
      avg_val = np.mean([info[key] for info in info_dicts])
      # absl log
      log_items.append('{}={:.6f}'.format(key_with_prefix, avg_val))
      # tensorboard
      self.summary_writer.scalar(key_with_prefix, avg_val, step=step)
    self.summary_writer.flush()
    logging.info('step={:08d} {}'.format(step, ' '.join(log_items)))
