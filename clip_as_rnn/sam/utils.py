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

# Copyright 2024 The Google Research Authors.
# This file is based on the SAM (Segment Anything) and HQ-SAM.
#
# 		https://github.com/facebookresearch/segment-anything
# 		https://github.com/SysCV/sam-hq/tree/main
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

"""SAM Utilities."""
# pylint: disable=all
# pylint: disable=g-importing-member
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def show_mask(mask, ax, random_color=False):
  if random_color:
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
  else:
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
  h, w = mask.shape[-2:]
  mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
  ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
  pos_points = coords[labels == 1]
  neg_points = coords[labels == 0]
  ax.scatter(
      pos_points[:, 0],
      pos_points[:, 1],
      color='green',
      marker='*',
      s=marker_size,
      edgecolor='white',
      linewidth=1.25,
  )
  ax.scatter(
      neg_points[:, 0],
      neg_points[:, 1],
      color='red',
      marker='*',
      s=marker_size,
      edgecolor='white',
      linewidth=1.25,
  )


def show_box(box, ax):
  x0, y0, x1, y1 = box
  w, h = x1 - x0, y1 - y0
  ax.add_patch(
      plt.Rectangle(
          (x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2
      )
  )


def show_anns(anns):
  if len(anns) == 0:
    return
  for index, dictionary in enumerate(anns):
    dictionary['id'] = index

  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)
  # polygons = []
  # color = []
  for ann in sorted_anns:
    m = ann['segmentation']
    img = np.ones((m.shape[0], m.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
      img[:, :, i] = color_mask[i]
    ax.imshow(np.dstack((img, m * 0.35)))

    # Get the centroid of the mask
    mask_y, mask_x = np.nonzero(m)
    centroid_x, centroid_y = np.mean(mask_x), np.mean(mask_y)

    # Display the mask ID
    mask_id = ann['id']
    ax.text(
        centroid_x,
        centroid_y,
        str(mask_id),
        color='black',
        fontsize=48,
        weight='bold',
    )


# Turn CAM result to SAM prompt
def aggregate_RGB_channel(activation_mask, how='max'):
  B, C, H, W = activation_mask.shape
  if how == 'max':
    res_activation_mask = np.amax(activation_mask, axis=1, keepdims=True)
  elif how == 'avr':
    res_activation_mask = np.mean(activation_mask, axis=1, keepdims=True)
    res_activation_mask = res_activation_mask.reshape(B, 1, H * W)

  res_activation_mask = np.squeeze(res_activation_mask, axis=1)
  return res_activation_mask


def find_k_points(arr, k, order='max', how_filter='median'):
  arr = arr.squeeze(0)
  flat_indices = np.argpartition(arr.flatten(), -k)[-k:]
  unravel_topk_idx = np.unravel_index(flat_indices, arr.shape)
  topk_indices = np.array(unravel_topk_idx).transpose()[:, ::-1]
  # print(topk_indices.shape)

  if how_filter == 'random':
    random_rows = np.random.choice(
        topk_indices.shape[0], size=int(round(k / 16)), replace=False
    )
    topk_indices = topk_indices[random_rows]
  elif how_filter == 'median':
    distances = cdist(topk_indices, topk_indices)
    distances = np.sum(distances, axis=1)
    median_distance = np.median(distances)
    filtered_idx = [
        i for i in range(len(distances)) if distances[i] < median_distance
    ]
    topk_indices = topk_indices[filtered_idx]
  return topk_indices


def max_sum_submatrix(matrix):
  matrix = np.array(matrix)
  H, W = matrix.shape
  # Preprocess cumulative sums for rows
  matrix[:, 1:] += matrix[:, :-1]
  max_sum = float('-inf')
  max_rect = (0, 0, 0, 0)  # (top, left, bottom, right)

  for left in range(W):
    for right in range(left, W):
      # Apply 1D Kadane's algorithm for the current pair of columns
      column_sum = matrix[:, right] - (matrix[:, left - 1] if left > 0 else 0)
      max_ending_here = max_so_far = column_sum[0]
      start, end = 0, 0

      for i in range(1, H):
        val = column_sum[i]
        if max_ending_here > 0:
          max_ending_here += val
        else:
          max_ending_here = val
          start = i

        if max_ending_here > max_so_far:
          max_so_far = max_ending_here
          end = i

      if max_so_far > max_sum:
        max_sum = max_so_far
        max_rect = (start, left, end, right)

  return max_sum, max_rect


def CAM2SAMClick(activation_map, k=5, order='max', how_filter='median'):
  # activation_map = aggregate_RGB_channel(activation_map)
  H, W, C = activation_map.shape
  activation_map = activation_map.reshape((1, 1, H, W))
  coords = []
  for nrow in range(activation_map.shape[0]):
    coord = find_k_points(activation_map[nrow], k, order, how_filter)
    coords.append(coord)
  return coords


def CAM2SAMBox(activation_map):
  # print(activation_map.shape)
  # activation_map = aggregate_RGB_channel(activation_map)
  H, W, C = activation_map.shape
  activation_map = activation_map.reshape((1, H, W))
  box_coordinates = []
  for nrow in range(activation_map.shape[0]):
    # print(activation_map[nrow].shape)
    arr = activation_map[nrow]

    norm_arr = 2 * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) - 1
    # print(norm_arr.shape)
    _, box_coordinate = max_sum_submatrix(norm_arr)
    box_coordinates.append(box_coordinate)
  return box_coordinates


# Visualize
def visualize_attention(arr, filename):
  # Create a figure and axes object
  fig, ax = plt.subplots()
  # Display the array as an image
  im = ax.imshow(arr)
  # Add a colorbar
  ax.figure.colorbar(im, ax=ax)
  # cbar = ax.figure.colorbar(im, ax=ax)
  # Save the figure as a PNG file
  fig.savefig(filename)


# Build config
def build_sam_config(config_path):
  with open(config_path, 'r') as infile:
    config = json.load(infile)

  sam_checkpoint = config['model']['sam_checkpoint']
  model_type = config['model']['model_type']
  return sam_checkpoint, model_type
