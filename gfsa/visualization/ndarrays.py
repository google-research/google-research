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

"""Visualization utilities for NDArrays."""

import matplotlib.pyplot as plt
import numpy as np


def mkgrid(shape):
  """Helper function to make a meshgrid indexer."""
  return np.meshgrid(*(range(a) for a in shape), indexing="ij")


def onedify(shape, skip_fn=lambda i: i):
  """Build scatter indices to render an ndarray into one dimension.

  The result will have gaps between each dimension, similar to what happens
  when you print high-dimensional arrays.

  For instance, helps transform

    ab
    cd

  to ab_cd by returning indices

    01
    34

  Args:
    shape: Shape of the target
    skip_fn: Function from axis level to number of indices to skip.

  Returns:
    Tuple (offsets, length) where offsets is an indexer array of the given shape
    that can scatter into an empty flat array of the given length
  """
  shape = np.array(shape)
  skips = np.array([skip_fn(i) for i in range(len(shape))])

  strides = np.zeros_like(shape)
  strides[0] = 1 + skips[0]
  for i in range(len(shape) - 1):
    strides[i + 1] = strides[i] * shape[-i - 1] - skips[i] + skips[i + 1]

  mesh_idxs = mkgrid(shape)
  offsets = sum(
      stride * mesh_idx for stride, mesh_idx in zip(strides[::-1], mesh_idxs))
  return offsets, np.max(offsets) + 1


def ndshow(arr,
           rcs=None,
           ax=None,
           figsize=(10, 10),
           names=None,
           colorbar=True,
           colskips=lambda i: i,
           rowskips=lambda i: i,
           ticks=False,
           **kwargs):
  """Show a high-dimensional array as an image by inserting spaces.

  Args:
    arr: Array to show
    rcs: Assignment of each axis to a row or column. Should be a string of
      length arr.ndim consisting of "r" and "c"
    ax: Axis to use. If not provided, makes a new one.
    figsize: Figure size (if ax is not given)
    names: Optional dictionary mapping axis indices to name sequences
    colorbar: Whether to add a colorbar
    colskips: Skips between each column group
    rowskips: Skips between each row group
    ticks: Whether to show ticks
    **kwargs: Arguments to `imshow`

  Returns:
    Tuple (ax, im) of the axis and image artist.
  """
  if names is None:
    names = {}
  if ax is None:
    _, ax = plt.subplots(1, 1, figsize=figsize)
  if rcs is None:
    rcs = ("rc" * (1 + arr.ndim // 2))[-arr.ndim:]
  is_row = np.array([rc == "r" for rc in rcs])
  is_col = np.array([rc == "c" for rc in rcs])
  sharr = np.array(arr.shape)
  col_shapes = sharr[is_col]
  row_shapes = sharr[is_row]

  if callable(colskips):
    colskip_fn = colskips
  else:
    colskip_fn = lambda i: colskips[i]

  if callable(rowskips):
    rowskip_fn = rowskips
  else:
    rowskip_fn = lambda i: rowskips[i]

  col_offs, col_max = onedify(col_shapes, skip_fn=colskip_fn)
  row_offs, row_max = onedify(row_shapes, skip_fn=rowskip_fn)

  mesh_idxs = mkgrid(arr.shape)
  col_offs_exp = col_offs[tuple(
      mesh_idxs[i] for i in range(arr.ndim) if rcs[i] == "c")]
  row_offs_exp = row_offs[tuple(
      mesh_idxs[i] for i in range(arr.ndim) if rcs[i] == "r")]

  tmp = np.full((row_max, col_max), np.nan)
  tmp[row_offs_exp, col_offs_exp] = arr
  im = ax.imshow(tmp, **kwargs)

  if ticks:
    names_arr = []
    for i in range(arr.ndim):
      csh = arr.shape[i]
      if i in names:
        cnames = tuple(str(n) for n in names[i])
        assert len(cnames) == csh
        names_arr.append(cnames)
      else:
        names_arr.append(np.arange(csh))

    col_names = [names_arr[i] for i in range(arr.ndim) if rcs[i] == "c"]
    row_names = [names_arr[i] for i in range(arr.ndim) if rcs[i] == "r"]
    col_fmt = "[" + ", ".join(
        "{}" if rcs[i] == "c" else ":" for i in range(arr.ndim)) + "]"
    row_fmt = "[" + ", ".join(
        "{}" if rcs[i] == "r" else ":" for i in range(arr.ndim)) + "]"
    coltups = list(
        zip(*(g.ravel() for g in np.meshgrid(*col_names, indexing="ij"))))
    rowtups = list(
        zip(*(g.ravel() for g in np.meshgrid(*row_names, indexing="ij"))))
    # pylint: disable=singleton-comparison,g-explicit-bool-comparison
    if ticks == True or ticks == "x":
      ax.set_xticks(col_offs.ravel())
      ax.set_xticklabels(
          labels=(col_fmt.format(*idxs) for idxs in coltups),
          rotation=90,
          family="monospace")
    if ticks == True or ticks == "y":
      ax.set_yticks(row_offs.ravel())
      ax.set_yticklabels(
          labels=(row_fmt.format(*idxs) for idxs in rowtups),
          family="monospace")
    # pylint: enable=singleton-comparison,g-explicit-bool-comparison

  if colorbar:
    plt.colorbar(im, ax=ax)

  return ax, im


def jagged_scatter_stack(things, bin_indices, axis=0):
  """Scatter blocks of `things` so that they are stacked along `bin_indices`.

  Pads with NaNs. Mostly useful for visualization.

  For instance, given things=[1, 2, 3, 4, 5], bin_indices=[0, 0, 1, 1, 1],
  produces
  [
    [1, 2, nan],
    [3, 4, 5],
  ]

  Args:
    things: Array where one of the axes represents a flattened jagged array.
    bin_indices: 1D int array giving the segment groups of each element.
    axis: Axis of things to split over

  Returns:
    Array with two additional leading dimensions instead of dimension `axis`,
    of the form described above.
  """
  things = np.asarray(things)
  bin_indices = np.asarray(bin_indices)
  assert bin_indices.ndim == 1
  things = np.moveaxis(things, axis, 0)
  unique_bins, inverse, counts = np.unique(
      bin_indices, return_inverse=True, return_counts=True)
  num_bins = np.max(unique_bins) + 1
  stack_width = np.max(counts)
  result = np.full((num_bins, stack_width) + things.shape[1:], np.nan)

  stack_indices = np.empty(bin_indices.shape, dtype="int")
  for i in range(len(unique_bins)):
    stack_indices[inverse == i] = np.arange(counts[i])

  result[bin_indices, stack_indices] = things
  return result


def show_routing_params(builder,
                        rparams,
                        figsize=(8, 16),
                        row_split_ct=1,
                        rparams_are_logits=False,
                        **kwargs):
  """Visualize routing parameters."""
  if rparams_are_logits:
    rparams = builder.routing_softmax(rparams)
  movestack = jagged_scatter_stack(
      rparams.move, builder.in_out_route_to_in_route, axis=1)
  movestack = movestack.transpose([2, 0, 3, 1, 4])
  movestack = movestack.reshape(movestack.shape[:3] + (-1,))
  # Axes are: [variants, in_routes, fsm_states, actions]
  res = np.concatenate(
      [rparams.special,
       np.full(movestack.shape[:3] + (1,), np.nan), movestack], -1)

  if "vmax" in kwargs:
    vmax = kwargs.pop("vmax")
  else:
    vmax = np.nanmax(res)
  split_rows = np.array_split(res, row_split_ct, axis=1)
  names = [
      f"{route.node_type}: {route.in_edge}" for route in builder.in_route_types
  ]
  split_names = np.array_split(names, row_split_ct)

  fig, ax = plt.subplots(1, len(split_rows), figsize=figsize, squeeze=False)
  for i, (res_split, names) in enumerate(zip(split_rows, split_names)):
    cax = ax[0, i]
    ndshow(res_split, ax=cax, rcs="crrc", vmin=0, vmax=vmax, **kwargs)
    cax.set_xticks([])
    cax.set_yticks(np.arange(len(names)) * (res.shape[2] + 1))
    cax.set_yticklabels(names)
  fig.tight_layout()
