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

"""Contains utility functions for training and visualization."""

import functools
import io

from typing import Any, Callable, Tuple, List

import flax.linen as nn

from jax.experimental import sparse as jsparse
import jax.numpy as jnp

from matplotlib import pyplot as plt
import matplotlib.patches as plot_patches

import tensorflow as tf



def get_first_datapoint_of_class(dataset,
                                 label):
  """Returns the first datapoint in dataset with given label.

  Args:
    dataset: a batched dataset where each example is (input, label), with scalar
      labels.
    label: the desired label

  Returns:
    datapoint: (x, label)
  """
  # TODO(gnegiar): Try dataset.unbatch().filter(lambda x,y: y==label).take(1)
  for datapoint in dataset:
    datapoint = tuple(map(lambda x: x.numpy(), datapoint))
    # Check if a datapoint in the batch matches the label
    if (datapoint[1] == label).any():
      # Get the first one
      datapoint = datapoint[0][datapoint[1] == label][0], datapoint[1][
          datapoint[1] == label][0]
      return datapoint[0], datapoint[1]
  raise ValueError(f'Label {label} is not in the dataset.')


def get_first_class_representatives(
    dataset,
    num_classes = 10):
  """Returns the first representative of each class in a dataset."""
  representatives = [
      get_first_datapoint_of_class(dataset, label)
      for label in range(num_classes)
  ]
  return representatives


def plot_subgraph(img, q, label,
                  start_node_coords):
  """Returns a figure with an image with overlaid selected pixels.

  The figure will have img in the background. We plot q pixel by pixel over img,
    using its values to scale transparency and color. Transparency uses
    values of q normalized to (0, 1). Color uses the values of q.
  We plot a circle around the start node.

  This function only handles greyscale img for now.

  Args:
    img: 2d array representing a grey-scale image.
    q: 1d sparse array of weights learned over the pixels. q[:img.size] should
      contain the weights over the pixels.
    label: predicted label
    start_node_coords: coordinates of the start node.

  Returns:
    figure: a matplotlib figure with the promised plot.
  """
  figure = plt.figure(figsize=(2, 2))
  ax = plt.gca()
  graph_size = img.size
  weighting = q.todense()[:graph_size]
  weighting = weighting.reshape(img.shape)
  norm_weighting = weighting / weighting.max()
  ax.set_title(f'{label} (pred)')
  ax.imshow(img, cmap='gray_r', vmin=0., vmax=1.)
  ax.imshow(weighting, alpha=norm_weighting, cmap='YlOrRd', vmax=.1)
  ax.contour(
      jnp.where(weighting != 0., 1., 0.),
      levels=0,  # Catch all positive weights
      colors='red',
      linewidths=4.,
      antialiased=True)
  circle = plot_patches.Circle(
      start_node_coords[::-1], 2, fill=False, color='cyan')
  ax.add_patch(circle)
  ax.axis('off')
  return figure


# TODO(gnegiar): Refactor to use above function
def plot_subgraph_classes(imgs, qs, labels,
                          start_nodes_coords,
                          num_classes):
  """Plots examples of each class, with subgraphs."""
  fig, axs = plt.subplots(2, num_classes // 2, figsize=(num_classes, 4))
  qs = [
      jsparse.BCOO((data, indices),
                   shape=qs.shape[1:])  # remove batch dim in shape
      for data, indices in zip(qs.data, qs.indices)
  ]

  for ax, img, q, label, start_node_coords in zip(axs.flatten(), imgs, qs,
                                                  labels, start_nodes_coords):
    # Plot digit
    img = img.squeeze()
    ax.imshow(img, cmap='gray_r', vmin=0., vmax=1.)
    ax.set_title(f'{label} (pred)')
    # Prep subgraph weighting
    graph_size = img.size
    weighting = q.todense()[:graph_size]
    weighting = weighting.reshape(img.shape)
    norm_weighting = weighting / weighting.max()  # normalized
    # Plot subgraph
    ax.imshow(weighting, alpha=norm_weighting, cmap='YlOrRd', vmax=.1)
    ax.contour(
        jnp.where(weighting != 0., 1., 0.),
        levels=0,  # Catch all nonzero weights
        colors='red',
        linewidths=2.,
        antialiased=True)
    circle = plot_patches.Circle(
        start_node_coords[::-1], 2, fill=False, color='cyan')
    ax.add_patch(circle)
    ax.axis('off')
  plt.tight_layout()
  return fig


def plot_to_image(figure):
  """Converts the matplotlib plot held in `figure` to a tensor.

  The supplied figure is closed and inaccessible after this call.

  Args:
    figure: a matplotlib figure to be saved.

  Returns:
    image: a Tensor, typically for saving in TensorBoard.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def vmap_and_average(forward,
                     in_axes=0):
  """Makes a pointwise function an average over minibatches.

  The first element in batch_output will be the loss (which is averaged over).

  Args:
    forward: Callable
    in_axes: An integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof specifying which input array axes to map over.

  Returns:
    batched_forward: Function returning the first output of forward averaged
      over a batch, and auxiliary outputs as-is.
  """

  @functools.wraps(forward)
  def wrapper(*args):
    vmapped_forward_function = nn.vmap(
        forward,
        in_axes=in_axes,
        variable_axes={'params': None},
        split_rngs={'params': False})
    losses, aux = vmapped_forward_function(*args)
    return losses.mean(axis=0), aux

  return wrapper
