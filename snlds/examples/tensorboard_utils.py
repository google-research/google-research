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

"""Utility functions for plotting in Tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf


colors = [
    (0.21568627450980393, 0.47058823529411764, 0.7490196078431373),  # blue
    (0.8980392156862745, 0.0, 0.0),  # red
    (0.996078431372549, 0.7019607843137254, 0.03137254901960784),  # amber
    (0.4823529411764706, 0.6980392156862745, 0.4549019607843137),  # faded green
    (0.5098039215686274, 0.37254901960784315, 0.5294117647058824),  # purple
    (0.5490196078431373, 0.0, 0.058823529411764705),  # crimson
    (0.6588235294117647, 0.6431372549019608, 0.5843137254901961)]  # greyish


def plot_to_image(figure):
  """Converts the matplotlib figure to a PNG image."""
  # The function is adapted from
  # github.com/tensorflow/tensorboard/blob/master/docs/image_summaries.ipynb

  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format="png")
  # Closing the figure prevents it from being displayed directly.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # tf.summary.image requires 4-D inputs. [num_samples, height, weight, color].
  image = tf.expand_dims(image, 0)
  return image


def show_lorenz_attractor_3d(fig_size, inputs, reconstructed_inputs,
                             fig_title=None):
  """Compare reconstructed lorenz attractor.

  Args:
    fig_size: 2-tuple of floats for figure dimension (width, height) in inches.
    inputs: a `2-D` numpy array, with shape [num_steps, 3]. At each timestep,
      it records the [x, y, z] position of lorenz attractor.
    reconstructed_inputs: a `2-D` numpy array, with the same shape as inputs,
      recording the reconstructed lorenz attractor trajectories.
    fig_title: Optional. A string to set title of figure.

  Returns:
    fig: the Matplotlib figure handle.
  """
  assert inputs.ndim == 2
  assert reconstructed_inputs.shape == inputs.shape
  fig = plt.figure(figsize=fig_size)
  if fig_title:
    plt.title(fig_title)
  ax = fig.add_subplot(1, 2, 1, projection="3d")
  ax.plot(inputs[:, 0], inputs[:, 1], inputs[:, 2],
          lw=1, color="k", alpha=0.5)
  for i in range(3):
    ax.set_xlabel("$x_{}$".format(i+1), fontsize=12, labelpad=3)

  ax = fig.add_subplot(1, 2, 2, projection="3d")
  ax.plot(reconstructed_inputs[:, 0],
          reconstructed_inputs[:, 1],
          reconstructed_inputs[:, 2],
          lw=1, color="k", alpha=0.5)
  for i in range(3):
    ax.set_xlabel("$x_{}$".format(i+1), fontsize=12, labelpad=3)
  return fig


def show_lorenz_segmentation(fig_size, inputs, segmentation, fig_title=None):
  """Show discrete state segmentation on input data along each dimension.

  Args:
    fig_size: 2-tuple of floats for figure dimension (width, height) in inches.
    inputs: a `2-D` numpy array, with shape [num_steps, 3]. At each timestep,
      it records the [x, y, z] position of lorenz attractor.
    segmentation: a 1-D numpy array, with shape [num_steps], recoding the
      most likely states at each time steps.
    fig_title: Optional. A string to set title of figure.

  Returns:
    fig: the Matplotlib figure handle.
  """
  fig = plt.figure(figsize=fig_size)
  if fig_title:
    plt.title(fig_title)

  inputs = np.squeeze(inputs)
  s_seq = np.squeeze(segmentation)
  z_cps = np.concatenate(([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
  for i in range(3):
    ax = fig.add_subplot(3, 1, i+1)
    for start, stop in zip(z_cps[:-1], z_cps[1:]):
      stop = min(s_seq.size, stop+1)
      ax.plot(np.arange(start, stop),
              inputs[start:stop, i],
              lw=1,
              color=colors[s_seq[start]])

    ax.set_ylabel("$x_{}(t)$".format(i+1))
    if i < 2:
      ax.set_xticklabels([])
  return fig


def show_discrete_states(fig_size, discrete_states_lk, segmentation,
                         fig_title=None):
  """Show likelihoods of discrete states s[t] and segmentation.

  Args:
    fig_size: 2-tuple of floats for figure dimension (width, height) in inches.
    discrete_states_lk: a 2-D numpy array, with shape [num_steps, num_states],
      recording the likelihood of each discrete states.
    segmentation: a 1-D numpy array, with shape [num_steps], recoding the
      most likely states at each time steps.
    fig_title: Optional. A string to set title of figure.

  Returns:
    fig: the Matplotlib figure handle.
  """
  fig = plt.figure(figsize=fig_size)
  if fig_title:
    plt.title(fig_title)

  ax = fig.add_subplot(1, 1, 1)
  s_seq = np.squeeze(segmentation)
  turning_loc = np.concatenate(
      ([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
  for i in range(discrete_states_lk.shape[-1]):
    ax.plot(np.reshape(discrete_states_lk[Ellipsis, i], [-1]))
  for tl in turning_loc:
    ax.axvline(tl, color="k", linewidth=2., linestyle="-.")
  ax.set_ylim(-0.1, 1.1)
  return fig


def show_hidden_states(fig_size, zt, segmentation, fig_title=None):
  """Show z[t] as series of line plots.

  Args:
    fig_size: 2-tuple of floats for figure dimension (width, height) in inches.
    zt: a 2-D numpy array, with shape [num_steps, num_hidden_states],
      recording the values of continuous hidden states z[t].
    segmentation: a 1-D numpy array, with shape [num_steps], recoding the
      most likely states at each time steps.
    fig_title: Optional. A string to set title of figure.

  Returns:
    fig: the Matplotlib figure handle.
  """
  fig = plt.figure(figsize=fig_size)
  if fig_title:
    plt.title(fig_title)

  ax = fig.add_subplot(1, 1, 1)
  s_seq = np.squeeze(segmentation)
  turning_loc = np.concatenate(
      ([0], np.where(np.diff(s_seq))[0]+1, [s_seq.size]))
  for i in range(zt.shape[-1]):
    ax.plot(zt[:, i])
  for tl in turning_loc:
    ax.axvline(tl, color="k", linewidth=2., linestyle="-.")
  return fig
