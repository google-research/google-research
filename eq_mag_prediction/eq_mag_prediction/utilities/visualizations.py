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

"""Utilities for visualizing earthquake data."""

from typing import Optional, Sequence
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from eq_mag_prediction.seismology import earthquake_properties


def plot_points_on_map(
    lon_points,
    lat_points,
    margin_factor = 0.5,
    **kwargs,
):
  """Plot a scatter of data points on a map using cartopy.

  This function is meant for usage in a colab notebook.

  Args:
    lon_points: a 1D array longitude coordinates of the points to plot
    lat_points: a 1D array latitude coordinates of the points to plot
    margin_factor: the fraction of the lon (lat) difference to extend the map
      upon. i.e. smaller factor -> more zoom in.
    **kwargs: of plt.scatter

  Returns:
    plt.Figure
  """
  kwargs.setdefault('alpha', 0.8)
  lat_margin = np.ptp(lat_points) * margin_factor
  lon_margin = np.ptp(lon_points) * margin_factor
  lat_lims = (lat_points.min() - lat_margin, lat_points.max() + lat_margin)
  lon_lims = (lon_points.min() - lon_margin, lon_points.max() + lon_margin)
  image_region = lon_lims + lat_lims

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
  ax.set_extent(image_region, crs=ccrs.PlateCarree())
  ax.coastlines(zorder=6)
  ax.stock_img()
  ax.scatter(lon_points, lat_points, **kwargs)
  return fig


def plot_strain_element(
    u_ij,
    coors,
    fault_angles = None,
    display_value = 2e-12,
    value_region = 1e-14,
    marker_opacity = 0.05,
    **go_Figure_kwargs,
):
  """Plot the values of a an elements of a 3d tensor at some value.

  An example of usage may be seen here:
  intelligence/earthquakes/seismology/double_couple_strain_tensor_solution.ipynb

  Args:
    u_ij: An element of the tensor. Shaped the same as dimension 1..N of coors.
    coors: An ndarray of coordinates corresponding to the solution, shaped
      (3,...)
    fault_angles: a sequence of floats of this order: (strike, rake, dip). If
      None (default) fault plane and vectors will not be displayed.
    display_value: The value of the tensor element which to display as the
      scatter. Points which correspond to the values in the range
      [display_value-value_region, display_value-value_region] will be plotted.
    value_region: The region which the scatter points are taken, see
      display_value's description.
    marker_opacity: opacity of the scatter points.
    **go_Figure_kwargs: kwargs for the object go.Figure.

  Returns:
    A plotly graphical object go.Figure.
  """
  to_show_logical_positive = ((display_value - value_region) <= u_ij) & (
      u_ij <= (display_value + value_region)
  )
  to_show_logical_negative = ((display_value - value_region) <= -u_ij) & (
      -u_ij <= (display_value + value_region)
  )
  coors_selected_4_scatter_positive = []
  coors_selected_4_scatter_negative = []
  for c in coors:
    coors_selected_4_scatter_positive.append(c[to_show_logical_positive])
    coors_selected_4_scatter_negative.append(c[to_show_logical_negative])

  # graphical object of positive strain difference
  scatter_positive = go.Scatter3d(
      x=coors_selected_4_scatter_positive[0],
      y=coors_selected_4_scatter_positive[1],
      z=coors_selected_4_scatter_positive[2],
      mode='markers',
      marker_opacity=marker_opacity,
      marker_color='red',
      name='extraction',
  )
  # graphical object of negative strain difference
  scatter_negative = go.Scatter3d(
      x=coors_selected_4_scatter_negative[0],
      y=coors_selected_4_scatter_negative[1],
      z=coors_selected_4_scatter_negative[2],
      mode='markers',
      marker_opacity=marker_opacity,
      marker_color='cadetblue',
      name='contraction',
  )

  graphical_objects_display = [scatter_positive, scatter_negative]

  # calculate points on plane for display
  if fault_angles is not None:
    fault_plane_object = _create_fault_graphical_object(coors, fault_angles)
    graphical_objects_display.append(fault_plane_object)

    eq_vectors = earthquake_properties.moment_vectors_from_angles(*fault_angles)
    fault_normal, strike_vector, slip_vector = eq_vectors
    graphical_objects_display.extend([
        _create_arrow_object(fault_normal, 'red', name='fault_normal'),
        _create_arrow_object(strike_vector, 'green', name='strike_vector'),
        _create_arrow_object(slip_vector, 'blue', name='slip_vector'),
    ])
  entire_figure = go.Figure(data=graphical_objects_display, **go_Figure_kwargs)
  return entire_figure


def _create_fault_graphical_object(
    coors, fault_angles
):
  """Create graphical object of the fault in north-east-down coordinate system.

  Args:
    coors: An ndarray of coordinates of the solution, shaped (3,...)
    fault_angles: a sequence of floats of this order: (strike, rake, dip)

  Returns:
    A plotly Scatter3d graphical object of a scatter of the fault plane.
  """
  min_x = coors[0].min()
  max_x = coors[0].max()
  min_y = coors[1].min()
  max_y = coors[1].max()
  coors_x1x2x3 = np.meshgrid(
      np.linspace(min_x, max_x, 30), np.linspace(min_y, max_y, 30), 0
  )
  coors_on_xy_plane = np.array(coors_x1x2x3).reshape((3, -1))
  _, _, dip = fault_angles
  _, strike_vector, _ = earthquake_properties.moment_vectors_from_angles(
      *fault_angles
  )
  plane_rotation = earthquake_properties.e3_to_fault_normal_rotation(
      dip, strike_vector
  )
  coors_on_plane = np.matmul(plane_rotation, coors_on_xy_plane)
  fault_plane_object = go.Scatter3d(
      x=coors_on_plane[0],
      y=coors_on_plane[1],
      z=coors_on_plane[2],
      marker_opacity=1,
      marker_size=4,
      marker_color='black',
      mode='markers',
      name='fault_plane',
  )
  return fault_plane_object


def _create_arrow_object(vector, color, name=None, len_factor=10):
  arrow_object = go.Scatter3d(
      x=[0, len_factor * vector[0]],
      y=[0, len_factor * vector[1]],
      z=[0, len_factor * vector[2]],
      marker=dict(size=5, color=color),
      line=dict(color=color, width=12),
      name=name,
  )
  return arrow_object
