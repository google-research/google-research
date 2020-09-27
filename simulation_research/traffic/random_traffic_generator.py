# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Generates random traffic for SUMO simulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import os

from absl import logging
import numpy as np
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import map_visualizer

TimeSamplerGammaKTheta = collections.namedtuple(
    'TimeSampleGammaKTheta', ['k', 'theta'])
TimeSamplerGammaAlphaBeta = collections.namedtuple(
    'TimeSampleGammaAlphaBeta', ['alpha', 'beta'])
TimeSamplerGammaMeanStd = collections.namedtuple(
    'TimeSampleGammaMeanStd', ['mean', 'std'])
DemandsWithPath = collections.namedtuple(
    'DemandsWithPath', ['time', 'origin', 'destination', 'num_cars', 'route'])
DemandsWithTAZExit = collections.namedtuple(
    'DemandsWithTAZExit', ['time', 'origin', 'num_cars'])
FREEWAY_EDGE_TYPES = ['highway.motorway', 'highway.motorway_carpool']
ARTERIAL_EDGE_TYPES = ['highway.motorway_link', 'highway.primary',
                       'highway.primary_link', 'highway.secondary',
                       'highway.secondary_link', 'highway.trunk',
                       'highway.trunk_link']


class RandomTrafficGenerator(object):
  """Generates random traffic using inhomogeneous Poisson models."""

  def __init__(self, sumo_net=None):
    self._net = sumo_net
    self._map_visualizer = map_visualizer.MapVisualizer(sumo_net)

  def get_freeway_input_output(self, figure_path=None):
    """Gets freeway inputs and outputs of the map.

    Args:
      figure_path: The figure path for the freeway and input/output edges. If it
          is set as None, then no output figure.

    Returns:
      input_output_pairs: A list of input-output pairs.
    """
    freeway_edges = self._net.filterEdges(FREEWAY_EDGE_TYPES)
    (freeway_input_edges,
     freeway_output_edges) = self._net.getInputOutputEdges(freeway_edges)
    if figure_path is not None:
      self._map_visualizer.plot_edges(
          [(freeway_edges, 'darkgreen', 1),
           (freeway_input_edges, 'lime', 1),
           (freeway_output_edges, 'red', 1)],
          output_figure_path=figure_path)
    input_output_pairs = list(itertools.product(freeway_input_edges,
                                                freeway_output_edges))
    return input_output_pairs

  def get_arterial_input_output(self, figure_path=None):
    """Gets freeway inputs and outputs of the map.

    Args:
      figure_path: The figure path for the arterial roads and input/output
          edges. It it is set as None, then no output figure.

    Returns:
      input_output_pairs: A list of input-output pairs.
    """
    freeway_edges = self._net.filterEdges(FREEWAY_EDGE_TYPES)
    (freeway_input_edges,
     freeway_output_edges) = self._net.getInputOutputEdges(freeway_edges)
    arterial_edges = self._net.filterEdges(ARTERIAL_EDGE_TYPES)
    (arterial_input_edges,
     arterial_output_edges) = self._net.getInputOutputEdges(arterial_edges)
    if figure_path is not None:
      self._map_visualizer.plot_edges(
          [(freeway_edges, 'darkgreen', 1),
           (freeway_input_edges, 'lime', 1),
           (freeway_output_edges, 'red', 1),
           (arterial_edges, 'b', 0.5),
           (arterial_input_edges, 'lime', 0.9),
           (arterial_output_edges, 'red', 0.9)],
          output_figure_path=figure_path)
    # Creates input-->output pairs.
    input_output_pairs = (
        list(itertools.product(freeway_input_edges, arterial_output_edges)) +
        list(itertools.product(arterial_input_edges, freeway_output_edges)) +
        list(itertools.product(arterial_input_edges, arterial_output_edges)))
    return input_output_pairs

  def setup_shortest_routes(self,
                            input_output_pairs,
                            edge_type_list=None,
                            vehicle_type_list='passenger',
                            routes_file=None,
                            figures_folder=None):
    """Generates the routes on freeways only.

    Args:
      input_output_pairs: Input-->output pairs of edges.
      edge_type_list: Restrict the type of edges.
      vehicle_type_list: Restrict the type of vehicles.
      routes_file: The name of the output route file.
      figures_folder: Whether to create figures for the routes. If it is set as
          None, then no output figures. Since there chould be many figures for
          all the routes, individual figures are named, for example
          "edgeidfrom_edgeidto_route.pdf", automatically.

    Returns:
      A list of routes. Each entry has `edge_from`, `edge_to`, `path_edges`,
          `edges_ids`, `route_length`, `route_id`.
    """
    routes = []
    route_counter = 0
    for from_to_pair in input_output_pairs:
      valid_path = False
      edge_from, edge_to = from_to_pair
      path_edges, route_length = self._net.getRestrictedShortestPath(
          edge_from,
          edge_to,
          vehicleClass=vehicle_type_list,
          edgeType=edge_type_list)
      # Regardless of whether there is a path between them.
      if figures_folder is not None:
        selected_edges = [([edge_from], 'lime', 1), ([edge_to], 'red', 1)]
      if route_length < float('inf'):
        valid_path = True
        route_counter += 1
        edges_ids = [edge.getID() for edge in path_edges]
        edges_ids = ' '.join(edges_ids)
        route_id = (edge_from.getID() + '_to_' +
                    edge_to.getID() + '_' + str(route_counter))
        token = '    <route id="%s" edges="%s"/>' % (route_id, edges_ids)
        if routes_file:
          file_util.append_line_to_file(routes_file, token)
        route = {}
        route['edge_from'] = edge_from
        route['edge_to'] = edge_to
        route['path_edges'] = path_edges
        route['edges_ids'] = edges_ids
        route['route_length'] = route_length
        route['route_id'] = route_id
        routes.append(route)
      if figures_folder is not None and valid_path:
        selected_edges = [(path_edges, 'darkblue', 1)] + selected_edges
        figure_path = os.path.join(
            figures_folder,
            (edge_from.getID() + '_' + edge_to.getID() + '_route.pdf'))
        self._map_visualizer.plot_edges(
            selected_edges, output_figure_path=figure_path)
    if routes_file:
      file_util.append_line_to_file(routes_file, '')
    return routes

  def generate_incomplete_routes_flow(self,
                                      time_point,
                                      time_step_size,
                                      incomplete_routes_demands,
                                      routes_file):
    """Generates incomplete routes.

    All traffic flow should be sorted by the departure time.

    Args:
      time_point: the time point for scheduled demands.
      time_step_size: time step size for the demands.
      incomplete_routes_demands: incomplete_routes_demands =
      [('700010432', '706447588#1', 1.2), ('700010432', '5018655', 1)].
      routes_file: output demand file.
    """
    for edge_from, edge_to, rate in incomplete_routes_demands:
      flow_id = edge_from + '_to_' + edge_to + '_' + str(time_point)
      num_cars = np.random.poisson(time_step_size * rate, 1)
      token = ('    <flow id="%s" begin="%d" end="%d" number="%d" ' %
               (flow_id, time_point, time_point + time_step_size, num_cars))
      token += ('from="%s" to="%s" ' % (edge_from, edge_to))
      token += 'departPos="base" departLane="best" departSpeed="max"/>'
      if routes_file:
        file_util.append_line_to_file(routes_file, token)

  def generate_routes_flow(self,
                           time_point,
                           time_step_size,
                           routes,
                           routes_demands,
                           routes_file):
    """Generates traffic according to the demand rates.

    Args:
      time_point: The timestamp of the traffic.
      time_step_size: Time step size.
      routes: A list of routes.
      routes_demands: A list of route indices and corresponding demand rates.
      routes_file: Output file.
    """
    for route_index, rate in routes_demands:
      route_id = routes[route_index]['route_id']
      flow_id = route_id + '_' + str(time_point)
      num_cars = np.random.poisson(time_step_size*rate, 1)
      token = ('    <flow id="%s" begin="%d" end="%d" number="%d" ' %
               (flow_id, time_point, time_point + time_step_size, num_cars,))
      token += ('route="%s" ' % (route_id))
      token += 'departPos="base" departLane="best" departSpeed="max"/>'
      if routes_file:
        file_util.append_line_to_file(routes_file, token)

  @classmethod
  def generate_departure_time(cls, parameters, sample_size):
    """Generates random time points for trips.

    Args:
      parameters: This is in a `collections.namedtuple`. The components can be
      different depends on the distribution.
          1. `parameters.distribution` == 'gamma_k_theta', `k`, `theta`, see
              definition in https://en.wikipedia.org/wiki/Gamma_distribution.
              `k` is called shape, and `theta` is called scale.
          2. `parameters.distribution` == `gamma_alpha_beta`, `alpha`, `beta`,
              see definition https://en.wikipedia.org/wiki/Gamma_distribution.
              `alpha` is called shape, and `beta` is called rate. Equivalently,
              `k` = `alpha`, `theta` = 1 / `beta`.
          3. `parameters.distribution` == 'gamma_mean_std', `mean`, `std`, the
              corresponding `k`, `theta` will be computed via the `mean` and the
               `std`. It is more straightforward  to use this pair of
               parameters.
      sample_size: Size of random samples.

    Returns:
      Random time point samples.
    """
    if isinstance(parameters, TimeSamplerGammaKTheta):
      # Mean = shape * scale. Variance = shape * scale * scale.
      # Mode = (shape - 1) * scale.
      return np.random.gamma(parameters.k, parameters.theta, sample_size)
    elif isinstance(parameters, TimeSamplerGammaAlphaBeta):
      k = parameters.alpha
      theta = 1 / parameters.beta
      return np.random.gamma(k, theta, sample_size)
    elif isinstance(parameters, TimeSamplerGammaMeanStd):
      mean, std, = parameters.mean, parameters.std
      k = (mean / std) ** 2
      theta = mean / k
      return np.random.gamma(k, theta, sample_size)
    else:
      raise ValueError('Unknown trip time distribution.')

  @classmethod
  def create_evacuation_shortest_path_demands(
      cls,
      edges,
      departure_time_distribution_parameters,
      cars_per_meter,
      evacuation_edges,
      evacuation_path_trees,
      evacuation_path_length):
    """Creates car demands for a group of edges.

    The generator has the following assumptions:
      1. On average there are H houses on a unit length residential edge, and
        the houses are uniformly distributed.
      2. On average there are C cars for each house, and all houses have cars.
      3. To simplify the model, we assume the cars on the same edge leave at the
        same time.
    If an edge has length L, it will generate floor(L * H * C) cars. Thus
    the input constant `cars_per_meter_residential` := L * H * C. This
    constant can also be estimated using total number of cars of the city
    divided by total length of the residential roads. Same calculation for the
    roads in the parking area.

    Args:
      edges: Input edges. It has to be in sumolib.net.edge.Edge type, since
          we need edge information from that.
      departure_time_distribution_parameters: The time distribution parameters
          for the vehicle departures. See `generate_departure_time` function
          for more details.
      cars_per_meter: Number of cars need to be generated from a unit length of
          road.
      evacuation_edges: The exits of the evacuation plan.
      evacuation_path_trees: The shortest path tree from all roads to each exit.
          This function assumes the vehicles choose the closest exit.
      evacuation_path_length: The corresponding path length for the shortest
          path trees above.

    Returns:
      zipped: The demands are zipped in _DemandsWithPath. Each tuple has entries
          1. departure time point, 2. departure road, 3. destination
          (one of the evacuation exits), 4. number of cars leaving from that
          road, 5. evacuation path
    """
    if isinstance(edges, sumolib.net.edge.Edge):
      edges = [edges]
    origin_edges = []
    num_cars_per_edge = []
    evacuation_destination = []
    evacuation_route = []
    # Calculates the number of cars on each edge.
    for edge in edges:
      if not isinstance(edge, sumolib.net.edge.Edge):
        raise ValueError('Edge has to be type sumolib.net.edge.Edge.')
      num_cars = int(np.floor(edge.getLength() * cars_per_meter))
      origin_edges.append(edge.getID())
      # 0 car is acceptable, it will be discarded later when the file is
      # written.
      num_cars_per_edge.append(num_cars)
      # Gets the closest exit.
      edge_id_evacuation_path_length = {}
      for evacuation_edge in evacuation_edges:
        if (evacuation_edge in evacuation_path_length and
            edge.getID() in evacuation_path_length[evacuation_edge]):
          edge_id_evacuation_path_length[
              evacuation_edge] = evacuation_path_length[evacuation_edge][
                  edge.getID()]
      if edge_id_evacuation_path_length:
        # Find the key with the smallest value.
        closest_exit_edge = min(
            edge_id_evacuation_path_length,
            key=edge_id_evacuation_path_length.get)
        # Note that the origin edge is not included in the path.
        # `evacuation_path_trees` acquired from
        # `sumolib.net.getRestrictedShortestPathsTreeToEdge` does not include
        # the origin. The origin will be added when the path is written to the
        # output file later in `write_evacuation_vehicle_path_demands`.
        path = evacuation_path_trees[closest_exit_edge][edge.getID()]
      else:
        closest_exit_edge = None
        path = ''
        logging.warning('Edge %s is not connected with any exit.', edge.getID())
      evacuation_destination.append(closest_exit_edge)
      evacuation_route.append(path)

    # Generates the cars' leaving time.
    time_points = cls.generate_departure_time(
        departure_time_distribution_parameters, len(num_cars_per_edge))
    zipped_demands = map(DemandsWithPath,
                         time_points, origin_edges, evacuation_destination,
                         num_cars_per_edge, evacuation_route)
    return list(zipped_demands)

  @classmethod
  def write_evacuation_vehicle_path_demands(cls,
                                            zipped_demands,
                                            routes_file):
    r"""Generates demands for residential vehicles.

    The output demand xml file is in the following format. Each entry has the
    information for each vehicle.

    <routes>
        <vType id="passenger" vClass="passenger"/>
        <vehicle id="0" type="passenger" depart="0.00" departLane="best" \
            departPos="base" departSpeed="max">
            <route edges="-8943413#1 -8943413#0  8936970#3 -8936970#3"/>
        </vehicle>
        <vehicle id="0" type="passenger" depart="0.00" departLane="best" \
            departPos="base" departSpeed="max">
            <route edges="-8943413#1 -8943413#0  8936970#3 -8936970#3"/>
        </vehicle>
    </routes>

    Args:
      zipped_demands: The zipped demands from function
          `create_evacuation_shortest_path_demands`.
      routes_file: Output file.
    """
    demands = sorted(zipped_demands, key=lambda x: x.time)

    # Write demands file.
    if file_util.f_exists(routes_file):
      raise ValueError('%s already exists.' % routes_file)
    token = '<routes>\n'
    file_util.append_line_to_file(routes_file, token)
    token = '    <vType id="passenger" vClass="passenger"/>\n'
    file_util.append_line_to_file(routes_file, token)
    for demand in demands:
      if demand.num_cars == 0:
        logging.info('Road edge %s is too short, no demands.', demand.origin)
        continue
      if demand.destination is None:
        logging.warning('Road edge %s is not connected to any exit.',
                        demand.origin)
        continue
      for vehicle_id in range(demand.num_cars):
        token = '    <vehicle id="%s" type="passenger" ' % (
            demand.origin + '_' + str(vehicle_id))
        token += 'depart="%s" ' % demand.time
        token += 'departLane="best" departPos="random" departSpeed="max" '
        token += 'arrivalPos="max">\n'
        # Remember to add the origin edge to the path list. The shortest path
        # acquired from `create_evacuation_shortest_path_demands` does not
        # include the origin edge. This is due to the algorithm in
        # `sumolib.net.getRestrictedShortestPathsTreeToEdge`.
        token += '        <route edges="%s"/>' % (
            demand.origin + ' ' + ' '.join(demand.route))
        token += '\n    </vehicle>'
        file_util.append_line_to_file(routes_file, token)
    token = '\n</routes>'
    file_util.append_line_to_file(routes_file, token)
    logging.info('Save file to: %s', routes_file)

  @classmethod
  def create_evacuation_auto_routing_demands(
      cls,
      edges,
      departure_time_distribution_parameters,
      cars_per_meter):
    """Creates car demands for a group of edges.

    The generator has the same assumptions in
    `create_evacuation_shortest_path_demands`. Since this function creates the
    demands for automatic routing, the path is not pre-determined. The exits are
    grouped into a traffic analysis zone (TAZ). See the link
    https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.
    html#traffic_assignement_zones_taz for details.

    Args:
      edges: Input edges of type sumolib.net.edge.Edge.
      departure_time_distribution_parameters: The time distribution parameters
          for the vehicle departures. See `generate_departure_time` function for
          more details.
      cars_per_meter: Number of cars need to be generated from a unit length of
          road.

    Returns:
      zipped_demands: The demands are zipped in a list of namedtuple. Each tuple
          has entries 1. departure time point, 2. departure road, 3. number of
          cars leaving from that road.
    """
    if isinstance(edges, sumolib.net.edge.Edge):
      edges = [edges]
    origin_edges = []
    num_cars_per_edge = []
    # Calcualtes the number of cars on each edge.
    for edge in edges:
      num_cars = int(np.floor(edge.getLength() * cars_per_meter))
      origin_edges.append(edge.getID())
      num_cars_per_edge.append(num_cars)
    time_points = cls.generate_departure_time(
        departure_time_distribution_parameters, len(num_cars_per_edge))
    zipped_demands = map(DemandsWithTAZExit,
                         time_points, origin_edges, num_cars_per_edge)
    return list(zipped_demands)

  @classmethod
  def write_evacuation_vehicle_auto_routing_demands(cls,
                                                    zipped_demands,
                                                    exit_taz,
                                                    routes_file):
    r"""Generates demands for residential vehicles.

    The output demand xml file is in the following format. Each entry has the
    information for each vehicle. The exits are grouped into a traffic analysis
    zone (TAZ). See the link https://sumo.dlr.de/docs/Definition_of_Vehicles,
    _Vehicle_Types,_and_Routes.html#traffic_assignement_zones_taz for details.

    <routes>
        <vType id="passenger" vClass="passenger"/>
        <trip id="veh_1" depart="11" from="gneE8" departLane="best" \
            departPos="random" departSpeed="max" toTaz="exit_taz"/>
        <trip id="veh_2" depart="13" from="gneE9" departLane="best" \
            departPos="random" departSpeed="max"toTaz="exit_taz"/>
    </routes>

    Args:
      zipped_demands: The zipped demands from function
          `create_evacuation_demands`.
      exit_taz: The name of the TAZ.
      routes_file: Output file.
    """
    sorted_demands = sorted(zipped_demands, key=lambda x: x.time)

    if file_util.f_exists(routes_file):
      raise ValueError('%s already exists.' % routes_file)
    token = '<routes>\n'
    file_util.append_line_to_file(routes_file, token)
    token = '    <vType id="passenger" vClass="passenger"/>\n'
    file_util.append_line_to_file(routes_file, token)
    for demand in sorted_demands:
      if demand.num_cars == 0:
        logging.info('Road edge %s is too short, no demands.', demand.origin)
        continue
      for vehicle in range(demand.num_cars):
        token = '    <trip id="%s" type="passenger" ' % (
            demand.origin + '_' + str(vehicle))
        token += 'depart="%s" ' % demand.time
        token += 'from="%s" ' % demand.origin
        token += 'departLane="best" departPos="random" departSpeed="max" '
        token += 'arrivalPos="max" '
        token += 'toTaz="%s"/>' % exit_taz
        file_util.append_line_to_file(routes_file, token)
    token = '\n</routes>'
    file_util.append_line_to_file(routes_file, token)
    logging.info('Saved file to: %s', routes_file)
