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

"""Utility functions for working with matplotlib figures."""

import io
import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt


def figure_to_image(figure, dpi=None, close=True):
  """Converts the matplotlib plot specified by 'figure' to a numpy image."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
  buf.seek(0)
  # Convert PNG buffer to numpy array
  image = imageio.imread(buf)
  buf.close()
  if close:
    plt.close(figure)
  return image


def make_figure(*, data, title, xlabel, ylabel):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  plt.imshow(data)
  ax.set_aspect('equal')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  plt.colorbar(orientation='vertical')
  return fig


def per_layer_figure(*, state, key_format, items, title, xlabel, ylabel,
                     show_values=False):
  """Generates a figure with a subplot per layer with consistent scales."""
  num_items = len(items)
  fig, axes = plt.subplots(
      nrows=1, ncols=num_items, figsize=(num_items * 3, 3))
  fig.suptitle(title)

  def get_value(index, item):
    if key_format:
      key = key_format.format(item)
      all_values = state[key]
      value = all_values[0]
    else:
      value = state[index]
    return value

  vmin = jnp.inf
  vmax = -jnp.inf
  for index, item in enumerate(items):
    value = get_value(index, item)
    value = jnp.where(jnp.isfinite(value), value, jnp.nan)
    vmin = jnp.minimum(vmin, jnp.nanmin(value))
    vmax = jnp.maximum(vmax, jnp.nanmax(value))

  for index, item in enumerate(items):
    if num_items == 1:
      ax = axes
    else:
      ax = axes[index]
    ax.set_title(f'Time = {index}')
    ax.set_xlabel(xlabel)
    if index == 0:
      ax.set_ylabel(ylabel)
    value = get_value(index, item)
    im = ax.imshow(value, vmin=vmin, vmax=vmax)

    if show_values and len(value) < 25:
      # Add text overlays indicating the numerical value.
      for node_index, row in enumerate(value):
        for timestep, v in enumerate(row):
          ax.text(timestep, node_index, str(v),
                  horizontalalignment='center', verticalalignment='center',
                  color='black')

    ax.set_aspect('equal')
  cbar_width = 0.05  # Fraction of a plot
  cbar_padding = 0.05
  half_padded_cbar_width = cbar_width + cbar_padding
  padded_cbar_width = cbar_width + 2 * cbar_padding
  fig.subplots_adjust(
      right=1 - padded_cbar_width/(num_items+padded_cbar_width))
  cbar_ax = fig.add_axes(
      [1 - half_padded_cbar_width/(num_items+padded_cbar_width),  # left
       0.15,  # bottom
       cbar_width/(num_items+padded_cbar_width),  # width
       0.7,  # top
       ]
  )
  fig.colorbar(im, cax=cbar_ax)
  return fig
