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

"""Wrappers over tf.Summary that support disabling summaries.

This library allows disabling all summaries by flipping a global flag.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from matplotlib import cm
import tensorflow.compat.v1 as tf


_summaries_enabled = True


def summaries_enabled():
  """Returns a boolean indicating whethwe summaries are enabled."""
  return _summaries_enabled


def disable_summaries():
  """Disables adding summaries by all functions in this library."""
  global _summaries_enabled
  _summaries_enabled = False


def histogram(*args, **kwargs):
  """Adds a historgram summary if summaries are enabled."""
  if _summaries_enabled:
    return tf.summary.histogram(*args, **kwargs)


def image(*args, **kwargs):
  """Adds an image summary if summaries are enabled."""
  if _summaries_enabled:
    return tf.summary.image(*args, **kwargs)


def scalar(*args, **kwargs):
  """Adds a scalar summary if summaries are enabled."""
  if _summaries_enabled:
    return tf.summary.scalar(*args, **kwargs)


def text(*args, **kwargs):
  """Adds a text summary if summaries are enabled."""
  if _summaries_enabled:
    return tf.summary.text(*args, **kwargs)


def image_with_colormap(name,
                        tensor,
                        colormap_name,
                        min_value=None,
                        max_value=None):
  """Creates an image summary using a colormap if summaries are enabled.

  Args:
    name: A string, name of the summary.
    tensor: A tensor of rank 3, batch x height x width, from which the images
      are to be created.
    colormap_name: A string, must be one of matplotlib colormaps, see
      https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    min_value: A scalar, the value in `tensor`that will be mapped to zero. If
      None, the minimum value of `tensor` per item will be used.
    max_value: A scalar, the value in `tensor`that will be mapped to 1.0. If
      None, the minimum value of `tensor` per item will be used.

  Returns:
    A tensorlfow summary op, or None.
  """
  if tensor.shape.rank != 3:
    raise ValueError('Tensor\'s rank has to be 3, not %s.' %
                     str(tensor.shape.rank))
  if not _summaries_enabled:
    return None
  if min_value is None:
    min_value = tf.reduce_min(tensor, axis=[1, 2], keepdims=True)
  if max_value is None:
    max_value = tf.reduce_max(tensor, axis=[1, 2], keepdims=True)
  normalized_tensor = (tensor - min_value) / (max_value - min_value + 1e-12)
  levels = tf.to_int32(tf.clip_by_value(normalized_tensor, 0.0, 1.0) * 255.0)
  colormap = cm.get_cmap(colormap_name)(range(256))[:, :3]  # Ignore the alpha
  colormapped_image = tf.gather(params=colormap, indices=levels)
  return tf.summary.image(name, colormapped_image)
