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

"""Visualization utility functions."""

import functools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from poem.core import keypoint_profiles


def get_points(points, indices, keepdims=True):
  """Gets points as the centers of points at specified indices.

  Args:
    points: A tensor for points. Shape = [..., num_points, point_dim].
    indices: A list of integers for point indices.
    keepdims: A boolean for whether to keep the reduced `num_points` dimension
      (of length 1) in the result distance tensor.

  Returns:
    A tensor for (center) points. Shape = [..., 1, point_dim].

  Raises:
    ValueError: If `indices` is empty.
  """
  if not indices:
    raise ValueError('`Indices` must be non-empty.')
  points = np.take(points, indices=indices, axis=-2)
  return np.mean(points, axis=-2, keepdims=keepdims)


def draw_pose_2d(ax,
                 keypoints_2d,
                 keypoint_profile_2d,
                 radius=1,
                 line_width=2,
                 marker_size=3,
                 left_part_color='#15baff',
                 right_part_color='#ff7e28'):
  """Draws a 2D pose.

  Assumes:
  1. Keypoints are in (y, x)-order.
  2. 2D keypoint profile supports all 13 basic body keypoints.

  Args:
    ax: A matplotlib.axes object.
    keypoints_2d: An numpy array for 2D keypoints of a single pose. Shape =
      [num_keypoints, 2].
    keypoint_profile_2d: A KeypointProfile2D object for input keypoints.
    radius: A float for half subfigure size.
    line_width: A float for line width for drawing.
    marker_size: A float for keypoint size for drawing.
    left_part_color: A string for left part color code.
    right_part_color: A string for right part color code.
  """

  def draw_line_segment(start_point, end_point, left_right_type):
    """Draws a line segment."""
    start_y, start_x = start_point
    end_y, end_x = end_point
    x = np.array([start_x, end_x])
    y = np.array([start_y, end_y])
    color = (
        right_part_color if left_right_type
        == keypoint_profiles.LeftRightType.RIGHT else left_part_color)
    ax.plot(x, y, lw=line_width, markersize=marker_size, c=color)

  head_index = keypoint_profile_2d.head_keypoint_index
  left_shoulder_index = keypoint_profile_2d.left_shoulder_keypoint_index
  right_shoulder_index = keypoint_profile_2d.right_shoulder_keypoint_index
  left_elbow_index = keypoint_profile_2d.left_elbow_keypoint_index
  right_elbow_index = keypoint_profile_2d.right_elbow_keypoint_index
  left_wrist_index = keypoint_profile_2d.left_wrist_keypoint_index
  right_wrist_index = keypoint_profile_2d.right_wrist_keypoint_index
  left_hip_index = keypoint_profile_2d.left_hip_keypoint_index
  right_hip_index = keypoint_profile_2d.right_hip_keypoint_index
  left_knee_index = keypoint_profile_2d.left_knee_keypoint_index
  right_knee_index = keypoint_profile_2d.right_knee_keypoint_index
  left_ankle_index = keypoint_profile_2d.left_ankle_keypoint_index
  right_ankle_index = keypoint_profile_2d.right_ankle_keypoint_index

  f = functools.partial(get_points, points=keypoints_2d, keepdims=False)
  head = f(indices=head_index)
  left_shoulder = f(indices=left_shoulder_index)
  right_shoulder = f(indices=right_shoulder_index)
  left_elbow = f(indices=left_elbow_index)
  right_elbow = f(indices=right_elbow_index)
  left_wrist = f(indices=left_wrist_index)
  right_wrist = f(indices=right_wrist_index)
  left_hip = f(indices=left_hip_index)
  right_hip = f(indices=right_hip_index)
  left_knee = f(indices=left_knee_index)
  right_knee = f(indices=right_knee_index)
  left_ankle = f(indices=left_ankle_index)
  right_ankle = f(indices=right_ankle_index)
  draw_line_segment(
      head,
      left_shoulder,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          head_index, left_shoulder_index))
  draw_line_segment(
      head,
      right_shoulder,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          head_index, right_shoulder_index))
  draw_line_segment(
      left_shoulder,
      right_shoulder,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_shoulder_index, right_shoulder_index))
  draw_line_segment(
      left_shoulder,
      left_elbow,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_shoulder_index, left_elbow_index))
  draw_line_segment(
      right_shoulder,
      right_elbow,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          right_shoulder_index, right_elbow_index))
  draw_line_segment(
      left_elbow,
      left_wrist,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_elbow_index, left_wrist_index))
  draw_line_segment(
      right_elbow,
      right_wrist,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          right_elbow_index, right_wrist_index))
  draw_line_segment(
      left_shoulder,
      left_hip,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_shoulder_index, left_hip_index))
  draw_line_segment(
      right_shoulder,
      right_hip,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          right_shoulder_index, right_hip_index))
  draw_line_segment(
      left_hip,
      right_hip,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_hip_index, right_hip_index))
  draw_line_segment(
      left_hip,
      left_knee,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_hip_index, left_knee_index))
  draw_line_segment(
      right_hip,
      right_knee,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          right_hip_index, right_knee_index))
  draw_line_segment(
      left_knee,
      left_ankle,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          left_knee_index, left_ankle_index))
  draw_line_segment(
      right_knee,
      right_ankle,
      left_right_type=keypoint_profile_2d.segment_left_right_type(
          right_knee_index, right_ankle_index))

  ax.set_xticks([])
  ax.set_yticks([])
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_xlim([-radius, radius])
  ax.set_ylim([-radius, radius])
  ax.invert_yaxis()
  ax.set_aspect('equal')


def draw_poses_2d(keypoints_2d,
                  keypoint_profile_2d,
                  num_cols,
                  subfigure_size=(1.5, 1.5),
                  **kwargs):
  """Draws 2D poses.

  Note that poses are drawn in row-major fashion.

  Args:
    keypoints_2d: A numpy array for 2D keypoints of poses. Shape = [num_poses,
      num_keypoints, 2].
    keypoint_profile_2d: A KeypointProfile2D object for input keypoints.
    num_cols: An integer for the number of grid columns to draw with.
    subfigure_size: A 2-tuple for subfigure size (height, width).
    **kwargs: A dictionary for additional arguments to be passed to
      `draw_pose_2d`.

  Returns:
    canvas: A numpy array for canvas tensor with drawn poses.
  """
  num_rows = (keypoints_2d.shape[0] + num_cols - 1) // num_cols
  subfigure_height, subfigure_width = subfigure_size
  fig = plt.figure(
      figsize=(num_cols * subfigure_width, num_rows * subfigure_height))
  fig_grid_spec = gridspec.GridSpec(num_rows, num_cols)
  plt.axis('off')

  for i, keypoints in enumerate(keypoints_2d):
    ax = plt.subplot(fig_grid_spec[i])
    draw_pose_2d(ax, keypoints, keypoint_profile_2d, **kwargs)

  fig.canvas.draw()
  canvas = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
  canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return np.expand_dims(canvas, axis=0).astype(np.float32)


def tf_draw_poses_2d(keypoints_2d,
                     keypoint_profile_2d,
                     num_cols,
                     subfigure_size=(1.5, 1.5),
                     **kwargs):
  """TensorFlow wrapper for drawing 2D poses.

  Note that poses are drawn in row-major fashion.

  Args:
    keypoints_2d: A tensor for 2D keypoints of poses. Shape = [num_poses,
      num_keypoints, 2].
    keypoint_profile_2d: A KeypointProfile2D object for input keypoints.
    num_cols: An integer for the number of grid columns to draw with.
    subfigure_size: A 2-tuple for subfigure size (height, width).
    **kwargs: A dictionary for additional arguments to be passed to
      `draw_pose_2d`.

  Returns:
    canvas: A tensor for canvas tensor with drawn poses.
  """
  f = functools.partial(
      draw_poses_2d,
      keypoint_profile_2d=keypoint_profile_2d,
      num_cols=num_cols,
      subfigure_size=subfigure_size,
      **kwargs)
  return tf.compat.v1.py_func(f, [keypoints_2d], Tout=tf.float32)
