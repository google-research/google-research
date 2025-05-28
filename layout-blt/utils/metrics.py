# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Metric utilities for layouts."""

import functools
from typing import Sequence

import numpy as np
import scipy


def normalize_bbox(layout,
                   resolution_w = 32,
                   resolution_h = 32):
  """Normalize the bounding box.

  Args:
    layout: An iterable of normalized bounding box coordinates in the format
      (width, height, center_x, center_y).
    resolution_w: The width of model input.
    resolution_h: The height of model input.

  Returns:
    List of normliazed boudning boxes in the format (x_min, y_min, x_max,
    y_max).
  """
  layout = np.array(layout, dtype=np.float32)
  layout = np.reshape(layout, (-1, 5))
  width, height = np.copy(layout[:, 1]), np.copy(layout[:, 2])
  layout[:, 1] = (layout[:, 3] - width / 2.) / resolution_w
  layout[:, 2] = (layout[:, 4] - height / 2.) / resolution_h
  layout[:, 3] = (layout[:, 3] + width / 2.) / resolution_w
  layout[:, 4] = (layout[:, 4] + height / 2.) / resolution_h

  return layout[:, 1:]


def get_layout_iou(layout):
  """Computes the IOU on the layout level.

  Args:
    layout: 1-d integer array in which in which every 5 elements form a group
      of box in the format (wdith, height, center_x, center_y).

  Returns:
    The value for the overlap index. If no overlaps are found, 0 is returned.
  """
  layout = np.array(layout, dtype=np.float32)
  layout = np.reshape(layout, (-1, 5))
  layout_channels = []
  for bbox in layout:
    canvas = np.zeros((32, 32, 1), dtype=np.float32)
    width, height = bbox[1], bbox[2]
    center_x, center_y = bbox[3], bbox[4]
    # Avoid round behevior at 0.5.
    min_x = round(center_x - width / 2. + 1e-4)
    max_x = round(center_x + width / 2. + 1e-4)
    min_y = round(center_y - height / 2. + 1e-4)
    max_y = round(center_y + height / 2. + 1e-4)
    canvas[min_x:max_x, min_y:max_y] = 1.
    layout_channels.append(canvas)
  if not layout_channels:
    return 0.
  sum_layout_channel = np.sum(np.concatenate(layout_channels, axis=-1), axis=-1)
  overlap_area = np.sum(np.greater(sum_layout_channel, 1.))
  bbox_area = np.sum(np.greater(sum_layout_channel, 0.))
  if bbox_area == 0.:
    return 0.
  return overlap_area / bbox_area


def get_average_iou(layout):
  """Computes the average amount of overlap between any two bounding boxes in a layout as IoU.

  Args:
    layout: 1-d integer array in which in which every 5 elements form a group
      of box in the format (wdith, height, center_x, center_y).

  Returns:
    The value for the overlap index. If no overlaps are found, 0 is returned.
  """

  iou_values = []
  layout = normalize_bbox(layout)
  for i in range(len(layout)):
    for j in range(i + 1, len(layout)):
      bbox1 = layout[i]
      bbox2 = layout[j]
      iou_for_pair = get_iou(bbox1, bbox2)
      if iou_for_pair > 0.:
        iou_values.append(iou_for_pair)
      # iou_values.append(iou_for_pair)

  return np.mean(iou_values) if len(iou_values) else 0.


def get_iou(bb0, bb1):
  intersection_area = get_intersection_area(bb0, bb1)

  bb0_area = area(bb0)
  bb1_area = area(bb1)

  if np.isclose(bb0_area + bb1_area - intersection_area, 0.):
    return 0
  return intersection_area / (bb0_area + bb1_area - intersection_area)


def get_intersection_area(bb0, bb1):
  """Computes the intersection area between two elements."""
  x_0, y_0, x_1, y_1 = bb0
  u_0, v_0, u_1, v_1 = bb1

  intersection_x_0 = max(x_0, u_0)
  intersection_y_0 = max(y_0, v_0)
  intersection_x_1 = min(x_1, u_1)
  intersection_y_1 = min(y_1, v_1)
  intersection_area = area(
      [intersection_x_0, intersection_y_0, intersection_x_1, intersection_y_1])
  return intersection_area


def area(bounding_box):
  """Computes the area of a bounding box."""

  x_0, y_0, x_1, y_1 = bounding_box

  return max(0., x_1 - x_0) * max(0., y_1 - y_0)


def get_overlap_index(layout):
  """Computes the average area of overlap between any two bounding boxes in a layout.

  This metric comes from LayoutGAN (https://openreview.net/pdf?id=HJxB5sRcFQ).

  Args:
    layout: 1-d integer array in which every 5 elements form a group of box
      in the format (wdith, height, center_x, center_y).

  Returns:
    The value for the overlap index. If no overlaps are found, 0 is returned.
  """

  intersection_areas = []
  layout = normalize_bbox(layout)
  for i in range(len(layout)):
    for j in range(i + 1, len(layout)):
      bbox1 = layout[i]
      bbox2 = layout[j]

      intersection_area = get_intersection_area(bbox1, bbox2)
      if intersection_area > 0.:
        intersection_areas.append(intersection_area)
  return np.sum(intersection_areas) if intersection_areas else 0.


def get_alignment_loss(layout):
  layout = normalize_bbox(layout)

  if len(layout) <= 1:
    return -1
  return get_alignment_loss_numpy(layout)


def get_alignment_loss_numpy(layout):
  """Calculates alignment loss of bounding boxes.

  Rewrites the function in the layoutvae:
  alignment_loss_lib.py by numpy.

  Args:
    layout: [asset_num, asset_dim] float array. An iterable of normalized
      bounding box coordinates in the format (x_min, y_min, x_max, y_max), with
      (0, 0) at the top-left coordinate.

  Returns:
    Alignment loss between bounding boxes.
  """

  a = layout
  b = layout
  a, b = a[None, :, None], b[:, None, None]
  cartesian_product = np.concatenate(
      [a + np.zeros_like(b), np.zeros_like(a) + b], axis=2)

  left_correlation = left_similarity(cartesian_product)
  center_correlation = center_similarity(cartesian_product)
  right_correlation = right_similarity(cartesian_product)
  correlations = np.stack(
      [left_correlation, center_correlation, right_correlation], axis=2)
  min_correlation = np.sum(np.min(correlations, axis=(1, 2)))
  return min_correlation


def left_similarity(correlated):
  """Calculates left alignment loss of bounding boxes.

  Args:
    correlated: [assets_num, assets_num, 2, 4]. Combinations of all pairs of
      assets so we can calculate the similarity between these bounding boxes
      in parallel.
  Returns:
    Left alignment similarities between all pairs of assets in the layout.
  """

  remove_diagonal_entries_mask = np.zeros(
      (correlated.shape[0], correlated.shape[0]))
  np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
  correlations = np.mean(
      np.abs(correlated[:, :, 0, :2] - correlated[:, :, 1, :2]), axis=-1)
  return correlations + remove_diagonal_entries_mask


def right_similarity(correlated):
  """Calculates right alignment loss of bounding boxes."""

  correlations = np.mean(
      np.abs(correlated[:, :, 0, 2:] - correlated[:, :, 1, 2:]), axis=-1)
  remove_diagonal_entries_mask = np.zeros(
      (correlated.shape[0], correlated.shape[0]))
  np.fill_diagonal(remove_diagonal_entries_mask, np.inf)
  return correlations + remove_diagonal_entries_mask


def center_similarity(correlated):
  """Calculates center alignment loss of bounding boxes."""

  x0 = (correlated[:, :, 0, 0] + correlated[:, :, 0, 2]) / 2
  y0 = (correlated[:, :, 0, 1] + correlated[:, :, 0, 3]) / 2

  centroids0 = np.stack([x0, y0], axis=2)

  x1 = (correlated[:, :, 1, 0] + correlated[:, :, 1, 2]) / 2
  y1 = (correlated[:, :, 1, 1] + correlated[:, :, 1, 3]) / 2
  centroids1 = np.stack([x1, y1], axis=2)

  correlations = np.mean(np.abs(centroids0 - centroids1), axis=-1)
  remove_diagonal_entries_mask = np.zeros(
      (correlated.shape[0], correlated.shape[0]))
  np.fill_diagonal(remove_diagonal_entries_mask, np.inf)

  return correlations + remove_diagonal_entries_mask


def docsim_w(b1, b2, conditional='none'):
  """Implements the DocSim W metric from READ (https://arxiv.org/pdf/1909.00302.pdf).

  Args:
    b1: the first bounding box [class_id, width, height, center_x, center_y].
    b2: the second boudning box [class_id, width, height, center_x, center_y].
    conditional: conditional type.
  Returns:
    Similarity scores between these two bounding boxes.
  """
  # If they are two different types, their score is 0.
  if b1[0] != b2[0]:
    return 0.
  if conditional == 'a+s':
    if b1[1] != b2[1] or b1[2] != b2[2]:
      return 0.
  # Area factor, we use the average of their areas.
  alpha = min(b1[5], b2[5]) ** 0.5

  # Center distance.
  delta_c = np.linalg.norm(b1[3:5] - b2[3:5])
  # Shape distance.
  delta_s = np.sum(np.abs(b1[1:3] - b2[1:3]))

  w = alpha * 2**(-delta_c - 2 * delta_s)
  return w


def docsim(layout_1, layout_2,
           max_element_count_diff=1, conditional='none'):
  """Computes the DocSim metric.

  Args:
    layout_1: document in the format produced by layout_to_ndarray.
    layout_2: document in the format produced by layout_to_ndarray.
    max_element_count_diff: maximum difference in the number of elements
      allowed. If the value is higher, the metric result is 0.
    conditional: conditional type.

  Returns:
    The DocSim metric value for the pair of layouts.
  """
  n_asset_1 = layout_1.shape[0]
  n_asset_2 = layout_2.shape[0]
  if n_asset_1 == 0 or n_asset_2 == 0:
    return 0.
  distance_function = functools.partial(docsim_w, conditional=conditional)

  if max_element_count_diff is not None and abs(
      n_asset_1 - n_asset_2) > max_element_count_diff:
    return 0

  distances = scipy.spatial.distance.cdist(
      layout_1, layout_2, metric=distance_function)
  assignment_row, assignment_col = scipy.optimize.linear_sum_assignment(
      -distances)

  costs = distances[assignment_row, assignment_col]

  result = 1 / costs.size * costs.sum()
  return result


def preprocess_layouts_for_distance(layouts_list):
  normalized_layouts = []
  for layouts in layouts_list:
    layouts = np.array(layouts, dtype=np.float32)
    layouts[:, 1:] = layouts[:, 1:] / 32.
    areas = layouts[:, 1] * layouts[:, 2]
    layouts = np.concatenate((layouts, areas[:, None]), axis=1)
    normalized_layouts.append(layouts)
  return normalized_layouts


def conditional_distance(synthetic_layouts,
                         real_layouts,
                         conditional='a'):
  """Computes the distance between synthetic layouts and real layouts.

  Args:
    synthetic_layouts: generated layouts.
    real_layouts: grouth-truth layouts.
    conditional: conditional type.

  Returns:
    Distance between real and sampled layouts.
  """
  synthetic_layouts = preprocess_layouts_for_distance(synthetic_layouts)
  real_layouts = preprocess_layouts_for_distance(real_layouts)

  distances = []
  for s_layout, r_layout in zip(synthetic_layouts, real_layouts):
    docsim_distance = docsim(s_layout, r_layout, conditional=conditional)
    distances.append(docsim_distance)

  distances = np.asarray(distances)
  return np.mean(distances)


def diversity(synthetic_layouts):
  synthetic_layouts = preprocess_layouts_for_distance(synthetic_layouts)
  distances = []
  for i in range(len(synthetic_layouts)):
    for j in range(i+1, len(synthetic_layouts)):
      docsim_distance = docsim(synthetic_layouts[i], synthetic_layouts[j])
      distances.append(docsim_distance)
  return sum(distances) / len(distances)


def unique_match(synthetic_layouts,
                 real_layouts):
  """Given sampled layouts, compute number of unique match real layouts.

  Args:
    synthetic_layouts: generated layouts.
    real_layouts: grouth-truth layouts

  Returns:
    Unique matches for generated layouts.
  """
  synthetic_layouts = preprocess_layouts_for_distance(synthetic_layouts)
  real_layouts = preprocess_layouts_for_distance(real_layouts)

  distances = []
  for synthetic_layout in synthetic_layouts:
    distances_for_layout = []
    for real_layout in real_layouts:
      docsim_distance = docsim(
          synthetic_layout,
          real_layout)
      distances_for_layout.append(docsim_distance)
    distances.append(distances_for_layout)

  distances = np.asarray(distances)
  matching_indices = np.argmax(distances, axis=1)
  # Obtains number of unique matches.
  unique_indices = np.unique(matching_indices)

  return unique_indices.size
