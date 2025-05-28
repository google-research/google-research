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

"""Tools for visualizing AndroidInTheWild data."""

from typing import Optional

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from android_in_the_wild import action_type


_NUM_EXS_PER_ROW = 5
_ACTION_COLOR = 'blue'


def is_tap_action(
    normalized_start_yx, normalized_end_yx
):
  distance = np.linalg.norm(
      np.array(normalized_start_yx) - np.array(normalized_end_yx)
  )
  return distance <= 0.04


def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
  """Decodes image from example and reshapes.

  Args:
    example: Example which contains encoded image.
    image_height: The height of the raw image.
    image_width: The width of the raw image.
    image_channels: The number of channels in the raw image.

  Returns:
    Decoded and reshaped image tensor.
  """
  image = tf.io.decode_raw(
      example.features.feature['image/encoded'].bytes_list.value[0],
      out_type=tf.uint8,
  )

  height = tf.cast(image_height, tf.int32)
  width = tf.cast(image_width, tf.int32)
  n_channels = tf.cast(image_channels, tf.int32)

  return tf.reshape(image, (height, width, n_channels))


def _get_annotation_positions(
    example, image_height, image_width
):
  """Processes the annotation positions into distinct bounding boxes.

  Args:
    example: The example to grab annotation positions for.
    image_height: The height of the screenshot.
    image_width: The width of the screenshot.

  Returns:
    A matrix of annotation positions with dimensions (# of annotations, 4),
      where each annotation bounding box takes the form (y, x, h, w).
  """
  flattened_positions = np.array(
      example.features.feature[
          'image/ui_annotations_positions'
      ].float_list.value
  )

  # Raw annotations are normalized so we multiply by the image's height and
  # width to properly plot them.
  positions = np.reshape(flattened_positions, (-1, 4)) * [
      image_height,
      image_width,
      image_height,
      image_width,
  ]
  return positions.astype(int)


def _add_text(
    text, screen_width, screen_height, ax
):
  """Plots text on the given matplotlib axis."""
  t = ax.text(
      0.5 * screen_width,
      0.95 * screen_height,
      text,
      color='white',
      size=20,
      horizontalalignment='center',
      verticalalignment='center',
  )
  t.set_bbox(dict(facecolor=_ACTION_COLOR, alpha=0.9))


def _plot_dual_point(
    touch_x,
    touch_y,
    lift_x,
    lift_y,
    screen_height,
    screen_width,
    ax,
):
  """Plots a dual point action on the given matplotlib axis."""
  if not is_tap_action(
      np.array([touch_y, touch_x]), np.array([lift_y, lift_x])
  ):
    ax.arrow(
        touch_x * screen_width,
        touch_y * screen_height,
        lift_x * screen_width - touch_x * screen_width,
        lift_y * screen_height - touch_y * screen_height,
        head_length=25,
        head_width=25,
        color=_ACTION_COLOR,
    )

  ax.scatter(
      touch_x * screen_width,
      touch_y * screen_height,
      s=550,
      linewidths=5,
      color=_ACTION_COLOR,
      marker='+',
  )
  return ax


def _plot_action(
    ex_action_type,
    screen_height,
    screen_width,
    touch_x,
    touch_y,
    lift_x,
    lift_y,
    action_text,
    ax,
):
  """Plots the example's action on the given matplotlib axis."""
  if ex_action_type == action_type.ActionType.DUAL_POINT:
    return _plot_dual_point(
        touch_x, touch_y, lift_x, lift_y, screen_height, screen_width, ax
    )
  elif ex_action_type in (
      action_type.ActionType.PRESS_BACK,
      action_type.ActionType.PRESS_HOME,
      action_type.ActionType.PRESS_ENTER,
  ):
    text = action_type.ActionType(ex_action_type).name
    _add_text(text, screen_width, screen_height, ax)
  elif ex_action_type == action_type.ActionType.TYPE:
    text = f'Input text "{action_text}"'
    _add_text(text, screen_width, screen_height, ax)
  elif ex_action_type == action_type.ActionType.STATUS_TASK_COMPLETE:
    text = 'Set episode status as COMPLETE'
    _add_text(text, screen_width, screen_height, ax)
  elif ex_action_type == action_type.ActionType.STATUS_TASK_IMPOSSIBLE:
    text = 'Set episode status as IMPOSSIBLE'
    _add_text(text, screen_width, screen_height, ax)
  else:
    print('Action type not supported')


def plot_example(
    example,
    show_annotations = False,
    show_action = False,
    ax = None,
):
  """Plots a visualization of the given example.

  Args:
    example: The example that we want to plot the screenshot for.
    show_annotations: Whether or not to plot the annotations over the
      screenshot.
    show_action: Whether or not to plot the action for the given example.
    ax: A matplotlib axis. If provided, the plotter will plot on this axis, else
      it will create a new one.

  Returns:
    The matplotlib axis that the example was plotted on.
  """
  image_height = example.features.feature['image/height'].int64_list.value[0]
  image_width = example.features.feature['image/width'].int64_list.value[0]
  image_channels = example.features.feature['image/channels'].int64_list.value[
      0
  ]
  image = _decode_image(example, image_height, image_width, image_channels)

  if ax is None:
    _, ax = plt.subplots(figsize=(8, 8))

  ax.imshow(image)

  if show_annotations:
    positions = _get_annotation_positions(example, image_height, image_width)
    for y, x, h, w in positions:
      rect = patches.Rectangle(
          (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'
      )
      ax.add_patch(rect)

  if show_action:
    touch_y, touch_x = example.features.feature[
        'results/yx_touch'
    ].float_list.value
    lift_y, lift_x = example.features.feature[
        'results/yx_lift'
    ].float_list.value
    ex_action_type = example.features.feature[
        'results/action_type'
    ].int64_list.value[0]
    type_text = (
        example.features.feature['results/type_action']
        .bytes_list.value[0]
        .decode('utf-8')
    )
    ax = _plot_action(
        ex_action_type,
        image_height,
        image_width,
        touch_x,
        touch_y,
        lift_x,
        lift_y,
        type_text,
        ax,
    )

  return ax


def plot_episode(
    episode,
    show_annotations = False,
    show_actions = False,
):
  """Plots a visualization of the given episode.

  Args:
    episode: A list of tf.train.Examples representing the episode that should be
      visualized.
    show_annotations: Whether to plot annotations for each episode step.
    show_actions: Whether to plot the actions for each episode step.
  """
  ep_len = len(episode)
  n_cols = min(ep_len, _NUM_EXS_PER_ROW)
  n_rows = 1 + (ep_len - 1) // n_cols
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, n_rows * 10))
  axs = axs.flatten()
  goal = ''
  i = 0
  for i, ex in enumerate(episode):
    goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
    plot_example(
        ex,
        show_annotations=show_annotations,
        show_action=show_actions,
        ax=axs[i],
    )

  for j in range(i + 1, len(axs)):
    ax = axs[j]
    ax.remove()

  fig.suptitle(
      goal,
      size=20,
      y=1.0,
  )
  plt.tight_layout()
