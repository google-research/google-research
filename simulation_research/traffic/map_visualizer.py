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

"""Visualization tools for SUMO related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

from absl import logging
from matplotlib.collections import LineCollection
import matplotlib.pylab as pylab
import numpy as np
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import memoize_util

_DEFAULT_COLOR = 'k'
_DEFAULT_WIDTH = 0.2
_PLOT_MAX_SPEED = 50  # m/s. 50 m/s = 111.847 mph.
_PLOT_MIN_SPEED = 0  # m/s


class MapVisualizer(object):
  """Visualization tools for SUMO related functions."""

  def __init__(self, sumo_net=None):
    self._edge_shapes = []
    self._edge_colors = []
    self._edge_widths = []
    self._whole_map_line_segments = []
    if sumo_net is None:
      self._net = None
      logging.warning('The input network is empty.')
    elif isinstance(sumo_net, sumolib.net.Net):
      self.set_net(sumo_net)

  @memoize_util.MemoizeClassFunctionOneRun
  def set_net(self, net):
    """Passes the network to the class.

    In order to protect the network from calling multiple times, this function
    is only allowed to call once. The memoize `MemoizeClassFunctionOneRun` from
    the `memoize_util` provides such protection.

    Args:
      net: The input SUMO type network variable.
    """
    self._net = net
    self._initialize_whole_map_line_segments()

  def _initialize_whole_map_line_segments(self):
    """Initializes the map roads line segments."""
    for edge in self._net.getEdges():
      self._edge_shapes.append(edge.getShape())
      self._edge_colors.append(_DEFAULT_COLOR)
      self._edge_widths.append(_DEFAULT_WIDTH)
    self._whole_map_line_segments = LineCollection(
        self._edge_shapes,
        linewidths=self._edge_widths,
        colors=self._edge_colors)

  def plot_edges(self,
                 highlight_edges_color_width=None,
                 with_whole_map=True,
                 output_figure_path=None):
    """Plots selected edges on the map.

    Args:
      highlight_edges_color_width: A tuple with three elements, a list of edges,
          the color for those edges and the width for those edges.
      with_whole_map: If this is true, it will plot all edges as a background.
      output_figure_path: If it is None, then no output.
    """
    if not self._net:
      raise ValueError('The input network is empty.')
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    if with_whole_map:
      # Make a copy before attaching to a new axis. This is because the
      # matplotlib objects know what plot they are attached to
      # and will not attach to more than one.
      ax.add_collection(copy.copy(self._whole_map_line_segments))
    highlight_edge_shapes = []
    highlight_edge_colors = []
    highlight_edge_widths = []
    if isinstance(highlight_edges_color_width, tuple):
      highlight_edges_color_width = [highlight_edges_color_width]
    elif not isinstance(highlight_edges_color_width, list):
      raise ValueError('The input must be a tuple or a list of tuple.')
    for (highlight_edges, highlight_color,
         highlight_width) in highlight_edges_color_width:
      if highlight_edges is None:
        continue
      for edge in highlight_edges:
        if isinstance(edge, str):  # This allows both ID and edge class input.
          edge = self._net.getEdge(edge)
        highlight_edge_shapes.append(edge.getShape())
        highlight_edge_colors.append(highlight_color)
        highlight_edge_widths.append(highlight_width)
      highlight_line_segments = LineCollection(
          highlight_edge_shapes,
          linewidths=highlight_edge_widths,
          colors=highlight_edge_colors)
      ax.add_collection(highlight_line_segments)
    ax.autoscale_view(True, True, True)
    ax.set_aspect('equal')
    pylab.plt.xlabel('[meter]')
    pylab.plt.ylabel('[meter]')
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def plot_shortest_path(self,
                         edge_from,
                         edge_to,
                         vehicle_class='passenger',
                         edge_type_list=None,
                         output_folder='.'):
    """Plot a shortest path between two edges.

    Args:
      edge_from: The start edge of the path.
      edge_to: The end edge of the path.
      vehicle_class: Restricts the path for certain vehicles.
      edge_type_list: Restricts the path for certain types of the roads.
      output_folder: Output folder.

    Returns:
      path_edges: A list of the path's edges.
      route_length: The length of the path.
    """
    if isinstance(edge_from, str):
      edge_from = self._net.getEdge(edge_from)
    if isinstance(edge_to, str):
      edge_to = self._net.getEdge(edge_to)
    path_edges, route_length = self._net.getRestrictedShortestPath(
        edge_from, edge_to, vehicleClass=vehicle_class, edgeType=edge_type_list)
    selected_edges = [([edge_from], 'lime', 1), ([edge_to], 'red', 1)]
    selected_edges = [(path_edges, 'darkblue', 1)] + selected_edges
    output_figure_path = os.path.join(
        output_folder, edge_from.getID() + '_' + edge_to.getID() + '_path.pdf')
    self.plot_edges(
        selected_edges,
        output_figure_path=output_figure_path)
    return path_edges, route_length

  def plot_vehicle_trajectory_on_map(self,
                                     vehicle_id,
                                     vehicle_to_trajectory,
                                     metric='speed',
                                     plot_max_speed=_PLOT_MAX_SPEED,
                                     plot_min_speed=_PLOT_MIN_SPEED,
                                     output_figure_path=None):
    """Plots a vehicle's trajectory with metric.

    Args:
      vehicle_id: The vehicle id from vehicle_to_trajectory.
      vehicle_to_trajectory: The trajectories of vehicles. It is a two-entry
          dictionary [vehicle_id][metric] --> the list of the metric. This
          dictionary is acquired from the function
          `get_vehicle_id_to_trajectory` in the module
          //research/simulation/traffic:simulation_data_parser.
      metric: The metric from vehicle_to_trajectory[vehicle_id].
      plot_max_speed: The largest value for the speed colorbar.
      plot_min_speed: The smallest value for the speed colorbar.
      output_figure_path: If it is none, then no output figure.
    """
    if (vehicle_id not in vehicle_to_trajectory or
        metric not in vehicle_to_trajectory[vehicle_id]):
      logging.warning('No %s data, or no %s data.', vehicle_id, metric)
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    if self._whole_map_line_segments:
      ax.add_collection(copy.copy(self._whole_map_line_segments))
    else:
      logging.warning('The map is empty.')
    image = pylab.plt.scatter(
        vehicle_to_trajectory[vehicle_id]['x'],
        vehicle_to_trajectory[vehicle_id]['y'],
        vmin=plot_min_speed, vmax=plot_max_speed,
        c=vehicle_to_trajectory[vehicle_id][metric],  # m/s
        cmap=pylab.plt.cm.get_cmap('RdYlGn'),
        edgecolor='', s=18)
    cbar = pylab.plt.colorbar(image)
    if metric == 'speed':
      cbar.ax.set_ylabel('speed [m/s]')
      pylab.plt.xlabel('[meter]')
      pylab.plt.ylabel('[meter]')
    ax.autoscale_view(True, True, True)
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def plot_edge_trajectory_on_map(self,
                                  edge_id_list,
                                  edge_id_to_trajectory,
                                  key='speed',
                                  plot_max_speed=_PLOT_MAX_SPEED,
                                  plot_min_speed=_PLOT_MIN_SPEED,
                                  output_figure_path=None):
    """Plots trajectories w.r.t. edges.

    Args:
      edge_id_list: The list of edges for plotting.
      edge_id_to_trajectory: The data acquired from the FCD file. This is a
          dictionary [edge_id][metric] --> the list of the metric. This
          dictionary is acquired from the function `get_edge_id_to_trajectory`
          in the module //research/simulation/traffic:simulation_data_parser.
      key: The values needed to be presented.
      plot_max_speed: The largest value for the speed colorbar.
      plot_min_speed: The smallest value for the speed colorbar.
      output_figure_path: If is None, no save figure.
    """
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    if self._whole_map_line_segments:
      ax.add_collection(copy.copy(self._whole_map_line_segments))
    else:
      logging.warning('The map is empty.')
    position = []
    x = []
    y = []
    values = []
    for edge_id in edge_id_list:
      if (edge_id not in edge_id_to_trajectory or
          'pos' not in edge_id_to_trajectory[edge_id] or
          'x' not in edge_id_to_trajectory[edge_id] or
          'y' not in edge_id_to_trajectory[edge_id]):
        logging.warning('Missing edge %s or metric information.', edge_id)
        continue
      positions_on_edge = edge_id_to_trajectory[edge_id]['pos']
      x_edge = edge_id_to_trajectory[edge_id]['x']
      y_edge = edge_id_to_trajectory[edge_id]['y']
      values_on_edge = edge_id_to_trajectory[edge_id][key]

      # Sort the trajectory dots by the positions on an edge.
      # It is easier to see on the plot.
      zipped = zip(positions_on_edge, x_edge, y_edge, values_on_edge)
      zipped = sorted(zipped)
      positions_on_edge, x_edge, y_edge, values_on_edge = zip(*zipped)
      position.extend(positions_on_edge)
      x.extend(x_edge)
      y.extend(y_edge)
      values.extend(values_on_edge)
    image = pylab.plt.scatter(
        x, y,
        vmin=plot_min_speed, vmax=plot_max_speed,
        c=values,
        cmap=pylab.plt.cm.get_cmap('RdYlGn'),
        edgecolor='', s=1)
    cbar = pylab.plt.colorbar(image)
    if key == 'speed':
      cbar.ax.set_ylabel('speed [m/s]')
    ax.autoscale_view(True, True, True)
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def _histogram_along_edge(self,
                            values_on_edge,
                            positions_on_edge,
                            num_bins=None,
                            bins=None):
    """Bins values on edge according to the positions.

    `positions_on_edge` and `values_on_edge` are paired list. Each value in
    `values_on_edge` has its corresponding position in `positions_on_edge`. The
    `values_on_edge` are binned by their positions. Then within each bin, the
    mean values are calculated.

    Args:
      values_on_edge: A list of values recorded on the an edge. The positons
        where the values are recorded in the `positions_on_edge`.
      positions_on_edge: A value's position on the edge. It is the distance from
        one end of the edge.
      num_bins: Number of bins to calculate the histogram.
      bins: A list of bin positions. If `bins` is used, `num_bins` will be
        ignored.

    Returns:
      bin_mean: A list of mean values within each bin. If a certain bin does not
        have any values, the corresponding mean value is set ot np.nan.
      bin_boundary: The positions of all bins. Note that the number of bins'
        boundaries is the number of `bin_mean` + 1.
    """
    if len(values_on_edge) == 1:
      logging.warning('One value in %s bins reduces to one value in one bin.',
                      num_bins)
      return values_on_edge, [positions_on_edge[0]] * 2
    # If all positions are identical, then the bin should be a point, otherwise
    # the numpy.hist will extend the bin to an interval around that point, and
    # may get outside the edge range.
    if all(positions_on_edge[0] == p for p in positions_on_edge):
      return ([sum(values_on_edge) / len(values_on_edge)],
              [positions_on_edge[0]] * 2)
    plot_bins = bins if bins else num_bins
    # Counts the sum of values in each bin.
    bin_sum, bin_boundary = np.histogram(
        positions_on_edge, weights=values_on_edge, bins=plot_bins)
    # Counts the number of values in each bin.
    bin_count, _ = np.histogram(positions_on_edge, bins=plot_bins)
    # It may come across the case where some bins do not have any values. So
    # when bin_sum is divided by `bin_count`, some entries encounter 0/0. For
    # this case, the output is set to np.nan. When the values are ploted, np.nan
    # will not be shown on the figure. The np.divide does the divide calculation
    # anywhere `bin_count` does not equal zero. When `bin_count` does equal
    # zero, then it remains unchanged from whatever value originally in the
    # `out` argument.
    base_out = np.zeros(len(bin_count))
    base_out.fill(np.nan)
    # `numpy.true_divide` avoids integer floor division for integer inputs.
    bin_mean = np.true_divide(bin_sum, bin_count,
                              out=base_out, where=(bin_count != 0))
    return bin_mean, bin_boundary

  def plot_edge_trajectory_histogram(self,
                                     edge_id_list,
                                     edge_id_to_trajectory,
                                     metric='speed',
                                     num_bins=20,
                                     output_figure_path=None):
    """Plot the edge's histogram of some metric.

    Args:
      edge_id_list: A list of edges ids need to be shown.
      edge_id_to_trajectory: The data acquired from the FCD file. It is a
          dictionary [edge_id][metric] --> a list of metrics. This dictionary is
          acquired from the function `get_edge_id_to_trajectory` in the module
          //research/simulation/traffic:simulation_data_parser.
      metric: The metric for the plotting.
      num_bins: Number of bins for the histogram.
      output_figure_path: If it is None, then no output.
    """
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    for edge_id in edge_id_list:
      if edge_id not in edge_id_to_trajectory:
        logging.warning('Missing edge %s information', edge_id)
        continue
      values_on_edge = edge_id_to_trajectory[edge_id][metric]
      positions_on_edge = edge_id_to_trajectory[edge_id]['pos']
      bin_mean, bin_boundary = self._histogram_along_edge(
          values_on_edge, positions_on_edge, num_bins)
      fig = pylab.figure(figsize=(8, 6))
      ax = fig.add_subplot(111)
      pylab.plt.plot(bin_boundary[:-1], bin_mean, c='k')
      pylab.plt.title('Edge: ' + edge_id)
      pylab.plt.xlabel('Edge relative position [meter]')
      ax.autoscale_view(True, True, True)
      if output_figure_path is not None:
        logging.info('Save figure to %s.', output_figure_path)
        pylab.savefig(output_figure_path)
      pylab.plt.close()

  def plot_edge_trajectory_histogram_on_map(self,
                                            edge_id_list,
                                            edge_id_to_trajectory,
                                            metric='speed',
                                            bin_length=10,
                                            plot_max_speed=_PLOT_MAX_SPEED,
                                            plot_min_speed=_PLOT_MIN_SPEED,
                                            output_figure_path=None):
    """Plot the edge's histogram of some metric.

    Args:
      edge_id_list: A list of edges need to be shown.
      edge_id_to_trajectory: The data acquired from the FCD file. It is a
          dictionary [edge_id][metric] --> a list of metrics. This dictionary is
          acquired from the function `get_edge_id_to_trajectory` in the module
          //research/simulation/traffic:simulation_data_parser.
      metric: The metric for the plotting.
      bin_length: The bin length for the histogram. The unit is the same as the
          edges' shapes.
      plot_max_speed: The largest value for the speed colorbar.
      plot_min_speed: The smallest value for the speed colorbar.
      output_figure_path: If it is None, then no output.
    """
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    for i, edge in enumerate(edge_id_list):
      if isinstance(edge, str):
        continue
      elif isinstance(edge, sumolib.net.edge.Edge):
        edge_id_list[i] = edge.getID()
    fig = pylab.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    if self._whole_map_line_segments:
      ax.add_collection(copy.copy(self._whole_map_line_segments))
    else:
      logging.warning('The map is empty.')
    x = []
    y = []
    values = []
    for edge_id in edge_id_list:
      if (edge_id not in edge_id_to_trajectory or
          'pos' not in edge_id_to_trajectory[edge_id] or
          metric not in edge_id_to_trajectory[edge_id]):
        logging.warning('Missing edge %s information', edge_id)
        continue
      edge_length = self._net.getEdge(edge_id).getLength()
      if edge_length < 10:  # Ignores short edges.
        continue
      num_bins = int(edge_length // bin_length)
      positions_on_edge = edge_id_to_trajectory[edge_id]['pos']
      values_on_edge = edge_id_to_trajectory[edge_id][metric]
      bin_mean, bin_boundary = self._histogram_along_edge(
          values_on_edge, positions_on_edge, num_bins)
      x_list, y_list = self._net.getEdge(
          edge_id).getPositionByDistance(bin_boundary[:-1])
      x.extend(x_list)
      y.extend(y_list)
      values.extend(bin_mean)
    image = pylab.plt.scatter(
        x, y,
        vmin=plot_min_speed, vmax=plot_max_speed,
        c=values,  # m/s
        cmap=pylab.plt.cm.get_cmap('RdYlGn'),
        edgecolor='', s=1)
    cbar = pylab.plt.colorbar(image)
    cbar.ax.set_ylabel('speed [m/s]')
    ax.autoscale_view(True, True, True)
    ax.set_aspect('equal')
    pylab.plt.xlabel('meter')
    pylab.plt.ylabel('meter')
    if 'time_interval' in edge_id_to_trajectory:
      pylab.plt.title(
          'Time range %s h -- %s h' % (
              edge_id_to_trajectory['time_interval'][0]/3600.0,
              edge_id_to_trajectory['time_interval'][1]/3600.0))

    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def plot_summary(self,
                   summary_attribute_to_time_series,
                   output_figure_path=None):
    """Plots simulation summary information.

    Args:
      summary_attribute_to_time_series: The data acquired from the summary file.
          It is a dictionary [metric] --> a list of metrics. This dictionary is
          acquired from the function `get_summary_attribute_to_time_series` in
          the module //research/simulation/traffic:simulation_data_parser.
      output_figure_path: If it is None, then no output.
    """
    if ('time' not in summary_attribute_to_time_series or
        'running' not in summary_attribute_to_time_series or
        'meanSpeed' not in summary_attribute_to_time_series):
      logging.warning('Missing metrics information')
      return
    t = summary_attribute_to_time_series['time']
    v = summary_attribute_to_time_series['running']
    s = summary_attribute_to_time_series['meanSpeed']
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(211)
    pylab.plt.plot(t, v, 'k')
    ax.autoscale_view(True, True, True)
    ax = fig.add_subplot(212)
    pylab.plt.plot(t, s, 'k')
    ax.autoscale_view(True, True, True)
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def plot_vehicle_counter_on_edge_histogram(self,
                                             edge_id,
                                             edge_id_to_attribute,
                                             metric='depart',
                                             bins=30,
                                             output_figure_path=None):
    """Plots vehicle numbers histogram by some keys.

    Args:
      edge_id: The edge needs to be shown.
      edge_id_to_attribute: The data acquired from the route file. It is a
          dictionary [edge_id][metric] --> a list of metrics. This dictionary is
          acquired from the function `plot_vehicle_counter_on_edge_histogram` in
          the module //research/simulation/traffic:simulation_data_parser.
      metric: The metric for the plotting.
      bins: The number of bins for the histogram.
      output_figure_path: If it is None, then no output.
    """
    if (edge_id not in edge_id_to_attribute or
        metric not in edge_id_to_attribute[edge_id]):
      logging.warning('Missing metrics information')
      return
    values = edge_id_to_attribute[edge_id][metric]
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pylab.plt.hist(values, bins=bins, edgecolor='k', fc='#0343DF')
    pylab.plt.xlim(min(values) - 100, max(values) + 100)
    ax.autoscale_view(True, True, True)
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  def plot_vehicle_counter_on_map(self,
                                  edge_id_list,
                                  edge_id_to_attribute,
                                  key='depart',
                                  output_figure_path=None):
    """Plots vehicle numbers histogram by some keys.

    The timing for each edge is not accurately calculated. Because the
    input `self._vehicle_counter_on_edge` is extracted from the route file,
    and it only has the departure and arrival time point for each
    vehicle's route. So it is suggested to use this function for the
    whole simulation rather than for some time range.
    For the latter purpose, see function `parse_fcd_file_by_edge`.

    Args:
      edge_id_list: A list of edges for plotting.
      edge_id_to_attribute: The data acquired from the route file. It is a
          dictionary [edge_id][metric] --> a list of metrics. This dictionary is
          acquired from the function `plot_vehicle_counter_on_edge_histogram` in
          the module //research/simulation/traffic:simulation_data_parser.
      key: This is can be either depart or arrival, which means the histogram
      is calculated by departure time or arrival time of the whole route.
      output_figure_path: Output file path.
    """
    if not self._net:
      raise ValueError('This function needs network.'+
                       'Try initialize_net(network_file)')
    if not edge_id_to_attribute:
      raise ValueError('The route file has not been parsed yet. '+
                       'Use function parse_route_file first.')
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    edge_shapes = []
    edge_colors = []
    edge_widths = []
    for edge_id in edge_id_list:
      if (edge_id not in edge_id_to_attribute or
          key not in edge_id_to_attribute[edge_id]):
        continue
      edge_value = len(edge_id_to_attribute[edge_id][key])
      edge_shapes.append(self._net.getEdge(edge_id).getShape())
      edge_colors.append(edge_value)
      edge_widths.append(_DEFAULT_WIDTH*3)
    edge_colors = np.array(edge_colors)
    highlight_line_segments = LineCollection(
        edge_shapes,
        array=edge_colors,
        cmap=pylab.plt.cm.get_cmap('RdYlGn_r'),
        linewidths=_DEFAULT_WIDTH*3)
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.add_collection(copy.copy(self._whole_map_line_segments))
    image = ax.add_collection(highlight_line_segments)
    ax.autoscale_view(True, True, True)
    cbar = pylab.plt.colorbar(image)
    cbar.ax.set_ylabel('Total number of vehicles')
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  @classmethod
  def add_pertentage_interception_lines(cls,
                                        time_line,
                                        cumulative_values,
                                        interception_percentage_list):
    """Add percentage interception lines to the cumulative curve."""
    if isinstance(interception_percentage_list, float):
      interception_percentage_list = [interception_percentage_list]
    if isinstance(cumulative_values, list):
      cumulative_values = np.array(cumulative_values)
    total_value = cumulative_values[-1]
    for interception_percentage in interception_percentage_list:
      # Find the index of the interception point.
      index = np.min(np.where(cumulative_values >=
                              interception_percentage * total_value))
      # Horizontal line.
      pylab.plt.plot([0, time_line[index]],
                     [cumulative_values[index], cumulative_values[index]],
                     color='gray')

  @classmethod
  def add_pertentage_interception_lines_vertical(cls,
                                                 time_line,
                                                 cumulative_values,
                                                 interception_percentage_list):
    """Add percentage interception lines to the cumulative curve."""
    if isinstance(interception_percentage_list, float):
      interception_percentage_list = [interception_percentage_list]
    if isinstance(cumulative_values, list):
      cumulative_values = np.array(cumulative_values)
    total_value = cumulative_values[-1]
    for interception_percentage in interception_percentage_list:
      # Find the index of the interception point.
      index = np.min(np.where(cumulative_values >=
                              interception_percentage * total_value))
      # Vertical line.
      pylab.plt.plot([time_line[index], time_line[index]],
                     [0, cumulative_values[index]],
                     color='gray')

  @classmethod
  def plot_demands_departure_time(cls,
                                  time_points,
                                  cars_per_time_point,
                                  number_bins=60,
                                  output_hist_figure_path=None,
                                  output_cumulative_figure_path=None):
    """Plots the histogram of the departure time points.

    Args:
      time_points: A list of time points. The unit is in sec.
      cars_per_time_point: The number of cars w.r.t. the `time_points`.
      number_bins: Number of bins for the histogram.
      output_hist_figure_path: Output histogram figure path.
      output_cumulative_figure_path: Output cumulative curve figure path.
    """
    if isinstance(time_points, tuple):
      time_points = np.array(time_points)
    if isinstance(time_points, list):
      time_points = np.array(time_points)
    zipped = zip(time_points, cars_per_time_point)
    zipped = sorted(zipped)
    time_points, cars_per_time_point = zip(*zipped)
    time_points = np.array(time_points)
    time_points_expanded = time_points.repeat(cars_per_time_point)

    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pylab.plt.hist(time_points_expanded, bins=number_bins)
    ax.autoscale_view(True, True, True)
    pylab.plt.xlim(0, 7)
    if output_hist_figure_path is not None:
      logging.info('Save figure to %s.', output_hist_figure_path)
      pylab.savefig(output_hist_figure_path)
    pylab.plt.close()

    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cumulative_values = np.cumsum(cars_per_time_point)
    pylab.plt.plot(time_points, cumulative_values)
    cls.add_pertentage_interception_lines(
        time_points, cumulative_values, [0.5, 0.9, 0.95])
    pylab.plt.xlim(0, 7)
    if output_cumulative_figure_path is not None:
      logging.info('Save figure to %s.', output_cumulative_figure_path)
      pylab.savefig(output_cumulative_figure_path)
    pylab.plt.close()

  @classmethod
  def plot_individual_detector(cls,
                               detector_trajectory_folder=None,
                               output_figure_folder=None):
    """Plot individual detector flow density.

    It is highly recommanded to set all detectors with the same time bin width,
    so that the data can be processed with the same script.
    `detector_trajectory_folder` has the *.pkl files processed by function
    `get_save_detector_data` from
    `research/simulation/traffic:simulation_data_parser`. Since this function
    will process all *.pkl files in the folder, irrelavant files should not be
    put in the folder. There are two types of vehicle counts in the detector
    output xml file, `nVehContrib` and `nVehEntered`. `nVehContrib` is the
    number of vehicles that have completely passed the detector within the
    interval. `nVehEntered` All vehicles that have touched the detector.
    Includes vehicles which have not passed the detector completely (and which
    do not contribute to collected values). This function uses the first one to
    avoide repeated counts for a same cars.

    Args:
      detector_trajectory_folder:
      output_figure_folder:
    """
    detector_pkl_file_list = os.listdir(detector_trajectory_folder)
    for trajectory_file in detector_pkl_file_list:
      if not trajectory_file.endswith('.pkl'):
        continue
      print('Plotting file: ', trajectory_file)
      detector_data = file_util.load_variable(
          detector_trajectory_folder + trajectory_file)
      detector_name = os.path.splitext(trajectory_file)[0]
      output_figure = detector_name + '.pdf'
      fig = pylab.figure(figsize=(8, 6))
      fig.add_subplot(111)
      pylab.plt.plot(np.array(detector_data['begin']) / 3600,
                     np.array(detector_data['nVehContrib']))
      pylab.plt.xlabel('Time [h]')
      pylab.plt.ylabel('Traffic flow [number of cars / min]')
      pylab.plt.title(detector_name)
      if output_figure_folder is not None:
        logging.info('Save figure to %s.', output_figure_folder + output_figure)
        pylab.savefig(output_figure_folder + output_figure)
      pylab.plt.close()

  @classmethod
  def plot_detector_flow_density(cls,
                                 time_line,
                                 vehicle_count_series,
                                 figure_label,
                                 ylim=None,
                                 output_figure_path=None):
    """Plot detector traffic flow density.

    Args:
      time_line:
      vehicle_count_series:
      figure_label:
      ylim:
      output_figure_path:
    """
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pylab.plt.plot(time_line, vehicle_count_series)
    pylab.plt.xlabel('Time [h]')
    pylab.plt.ylabel('Traffic flow [number of cars / min]')
    pylab.plt.title(figure_label)
    pylab.plt.ylim(ylim)
    ax.autoscale_view(True, True, True)
    if output_figure_path is not None:
      logging.info('Save figure to %s.', output_figure_path)
      pylab.savefig(output_figure_path)
    pylab.plt.close()

  @classmethod
  def plot_detector_flow_density_by_group(cls,
                                          detector_pkl_files_by_group,
                                          figure_labels=None,
                                          ylim=None,
                                          output_figure_folder=None):
    """Plots detectors' data by group.

    Args:
      detector_pkl_files_by_group:
      figure_labels:
      ylim:
      output_figure_folder:
    """
    for group_id, pkl_files in enumerate(detector_pkl_files_by_group):
      vehicle_count_series = 0
      for pkl_file in pkl_files:
        detector_data = file_util.load_variable(pkl_file)
        vehicle_count_series += np.array(detector_data['nVehContrib'])
      time_line = np.array(detector_data['begin']) / 3600
      figure_label = (
          figure_labels[group_id] if figure_labels else 'Group_' +
          str(group_id))
      output_figure = (
          figure_labels[group_id] + '_car_count.pdf' if figure_labels
          else 'Group_' + str(group_id) + '.pdf')
      logging.info('Saved figure: %s.', output_figure)
      print('Saved figure: ', output_figure)
      cls.plot_detector_flow_density(
          time_line,
          vehicle_count_series,
          figure_label=figure_label,
          ylim=ylim,
          output_figure_path=os.path.join(output_figure_folder + output_figure))

  @classmethod
  def plot_detector_arrival_time(cls,
                                 time_line,
                                 vehicle_count_series,
                                 evacuation_density_figure_path=None,
                                 evacuation_cumulative_figure_path=None):
    """Plots evacuation vehicle flow.

    Args:
      time_line:
      vehicle_count_series:
      evacuation_density_figure_path:
      evacuation_cumulative_figure_path:
    """
    # Vehicle flow density.
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pylab.plt.plot(time_line, vehicle_count_series)
    pylab.plt.xlabel('Time [sec]')
    pylab.plt.ylabel('Traffic flow density [number of cars / min]')
    pylab.plt.title('Total flow density')
    ax.autoscale_view(True, True, True)
    if evacuation_density_figure_path is not None:
      logging.info('Save figure: %s.', evacuation_density_figure_path)
      print('Save figure: ', evacuation_density_figure_path)
      pylab.savefig(evacuation_density_figure_path)
    pylab.plt.close()

    # Cumulative vehicles.
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cumulative_values = np.cumsum(vehicle_count_series)
    pylab.plt.plot(time_line, cumulative_values)
    cls.add_pertentage_interception_lines(
        time_line, cumulative_values, [0.5, 0.9, 0.95])
    pylab.plt.xlabel('Time [h]')
    pylab.plt.ylabel('Cummulative traffic flow')
    pylab.plt.title('Cummulative traffic flow')
    if evacuation_cumulative_figure_path is not None:
      logging.info('Save figure: %s.', evacuation_cumulative_figure_path)
      print('Save figure: ', evacuation_cumulative_figure_path)
      pylab.savefig(evacuation_cumulative_figure_path)
    pylab.plt.close()

  @classmethod
  def plot_detector_arrival_time_by_group(
      cls,
      detector_pkl_files,
      output_figure_folder):
    """Plot the data of all detectors."""
    vehicle_count_series = 0
    for pkl_file in detector_pkl_files:
      detector_data = file_util.load_variable(pkl_file)
      vehicle_count_series += np.array(detector_data['nVehContrib'])
    time_line = np.array(detector_data['begin']) / 3600
    cls.plot_detector_arrival_time(
        time_line,
        vehicle_count_series,
        evacuation_density_figure_path=(
            output_figure_folder + 'detector_evacuation_density.pdf'),
        evacuation_cumulative_figure_path=(
            output_figure_folder + 'detector_evacuation_cumulative.pdf'))
    logging.info('Totoal number of cars: %s.', np.sum(vehicle_count_series))
    print('Totoal number of cars: ', np.sum(vehicle_count_series))

  @classmethod
  def _calculate_area_under_cummulative_curve(cls, x, y, method='trapz'):
    """Estimates the area under the curve."""
    if method == 'trapz':
      return np.trapz(y, x=x)
    else:
      raise ValueError('Method "%s" is not supported. Try "trapz".' % method)

  @classmethod
  def _trim_flat_tail_cummulative_array(cls, x, y):
    """Cut the flat long tail of the cummulative curve.

    Args:
      x: The x positions of the `y` values.
      y: This is assumed to be monotonically non-decreasing.

    Returns:
      x: The x positions of the `y` values.
      y: The flat tail has been removed.
    """
    first_tail_index = np.min(np.where(y == y[-1]))
    return x[:(first_tail_index + 1)], y[:(first_tail_index + 1)]

  @classmethod
  def _align_tails_cummulative_arrays(cls, x1, y1, x2, y2):
    """Aligns the two curves by the tails."""
    x_max = max(x1[-1], x2[-1])
    if x_max != x1[-1]:
      x1 = np.append(x1, x_max)
      y1 = np.append(y1, y1[-1])
    if x_max != x2[-1]:
      x2 = np.append(x2, x_max)
      y2 = np.append(y2, y2[-1])
    return x1, y1, x2, y2

  @classmethod
  def calculate_gap_area_between_cummulative_curves(cls, x1, y1, x2, y2):
    """Calculates the area of the gap between two cummulative curves."""
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    if y1[-1] != y2[-1]:
      logging.warning('Two cummulative curves do not agree with each other.')
      print('Two cummulative curves do not agree with each other.')
    x1, y1 = cls._trim_flat_tail_cummulative_array(x1, y1)
    x2, y2 = cls._trim_flat_tail_cummulative_array(x2, y2)
    x1, y1, x2, y2 = cls._align_tails_cummulative_arrays(x1, y1, x2, y2)
    area1 = cls._calculate_area_under_cummulative_curve(x1, y1)
    area2 = cls._calculate_area_under_cummulative_curve(x2, y2)
    return np.abs(area1 - area2)

