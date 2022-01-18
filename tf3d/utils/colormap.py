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

"""Contains functions for creating pascal colormap and coloring labels."""

import numpy as np

_PASCAL_COLOR_MAP_MAX_ENTRY = 256


def _bitget(val, idx):
  """Get the bit value.

  Args:
    val: Input value.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return np.uint8((val & (1 << idx)) != 0)


def create_pascal_color_map():
  """Create a label color map used in PASCAL VOC segmentation benchmark.

  Returns:
    cmap: Color map for visualizing segmentation results.
  """
  cmap = np.zeros([_PASCAL_COLOR_MAP_MAX_ENTRY, 3], dtype=np.uint8)

  for i in range(_PASCAL_COLOR_MAP_MAX_ENTRY):
    r = 0
    g = 0
    b = 0
    ind = i
    for j in range(8):
      shift = 7 - j
      r |= _bitget(ind, 0) << shift
      g |= _bitget(ind, 1) << shift
      b |= _bitget(ind, 2) << shift
      ind >>= 3

    cmap[i, 0] = r
    cmap[i, 1] = g
    cmap[i, 2] = b

  return cmap


def label_to_pascal_color(label):
  """Add color defined by PASCAL VOC colormap to the label.

  Args:
    label: A 1D array with integer type, storing the segmentation label.

  Returns:
    result: A 1D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label value larger than color map maximum entry.
  """
  cmap = create_pascal_color_map()

  if label.ndim != 1:
    raise ValueError('Label map must have shape [num_points]')

  num_points = label.shape[0]

  result = np.zeros([num_points, 3], dtype=np.uint8)
  for i in range(num_points):
    label_value = label[i]

    if label_value >= _PASCAL_COLOR_MAP_MAX_ENTRY:
      label_value = _PASCAL_COLOR_MAP_MAX_ENTRY - 1

    result[i, 0] = cmap[label_value, 0]
    result[i, 1] = cmap[label_value, 1]
    result[i, 2] = cmap[label_value, 2]

  return result


def labels_to_color(label, add_colormap=True, scale_values=False):
  """Saves the given label to image on disk.

  Args:
    label: The [num_points] numpy array to be saved. The data will be converted
      to uint8 rgb values.
    add_colormap: Add color map to the label or not.
    scale_values: Scale the input values to [0, 255] for visualization.

  Returns:
    A colorized [num_points, 3] numpy array.
  """
  if add_colormap:
    colored_label = label_to_pascal_color(label)
  else:
    if scale_values:
      colored_label = 255. * label
    else:
      colored_label = label

  return colored_label
