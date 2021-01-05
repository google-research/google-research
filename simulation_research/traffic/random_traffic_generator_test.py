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

# Lint as: python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest
import numpy as np
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import random_traffic_generator


def _load_file(folder, file_name):
  file_path = os.path.join(folder, file_name)
  return file_util.f_abspath(file_path)


class RandomTrafficGeneratorTest(absltest.TestCase):

  def setUp(self):
    super(RandomTrafficGeneratorTest, self).setUp()
    testdata_dir = './testdata'
    self._output_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    sumo_net_file = 'mtv_tiny.net.xml'
    map_file = _load_file(testdata_dir, sumo_net_file)
    self._net = sumolib.net.readNet(map_file)
    traffic_generator = random_traffic_generator.RandomTrafficGenerator(
        self._net)
    self._random_traffic_generator = traffic_generator
    # The traffic generator uses numpy to draw random samples. The numpy random
    # seed is set here to make the result replicable.
    np.random.seed(0)

  def test_get_freeway_input_output(self):
    figure_path = os.path.join(self._output_dir, 'freeway_routes.pdf')
    input_output = self._random_traffic_generator.get_freeway_input_output(
        figure_path=figure_path)
    self.assertLen(input_output, 9)

  def test_get_arterial_input_output(self):
    figure_path = os.path.join(self._output_dir, 'arterial_routes.pdf')
    input_output = self._random_traffic_generator.get_arterial_input_output(
        figure_path=figure_path)
    self.assertLen(input_output, 21)

  def test_setup_shortest_routes(self):
    # Test case for freeways.
    input_output = self._random_traffic_generator.get_freeway_input_output()
    output_file = os.path.join(self._output_dir, 'freeway_routes.xml')
    routes = self._random_traffic_generator.setup_shortest_routes(
        input_output,
        edge_type_list=random_traffic_generator.FREEWAY_EDGE_TYPES,
        routes_file=output_file,
        figures_folder=self._output_dir)
    self.assertLen(routes, 3)
    routes_length = [routes[0]['route_length'],
                     routes[1]['route_length'],
                     routes[2]['route_length']]
    routes_length.sort()
    self.assertAlmostEqual(routes_length[0], 450.18)
    self.assertAlmostEqual(routes_length[1], 622.57)
    self.assertAlmostEqual(routes_length[2], 1051.25)

    # Test case for arterial roads.
    input_output = self._random_traffic_generator.get_arterial_input_output()
    output_file = os.path.join(self._output_dir, 'arterial_routes.xml')
    routes = self._random_traffic_generator.setup_shortest_routes(
        input_output,
        edge_type_list=(random_traffic_generator.FREEWAY_EDGE_TYPES +
                        random_traffic_generator.ARTERIAL_EDGE_TYPES),
        routes_file=output_file,
        figures_folder=self._output_dir)
    self.assertLen(routes, 5)
    routes_length = [routes[0]['route_length'],
                     routes[1]['route_length'],
                     routes[2]['route_length'],
                     routes[3]['route_length'],
                     routes[4]['route_length']]
    routes_length.sort()
    self.assertAlmostEqual(routes_length[0], 16.84)
    self.assertAlmostEqual(routes_length[1], 269.97)
    self.assertAlmostEqual(routes_length[2], 512.76)
    self.assertAlmostEqual(routes_length[3], 637.8299999999999)
    self.assertAlmostEqual(routes_length[4], 1051.25)

  def test_generate_freeway_routes_flow(self):
    """Test for the freeway demands generation workflow.

    All the unit tests have been done above, and there is no calculation in this
    test. So this one just verifies nothing is block in the workflow.
    """
    routes_file = os.path.join(self._output_dir, 'freeway_routes_demands.xml')
    token = '<routes>\n'
    file_util.append_line_to_file(routes_file, token)
    token = ('    <vType id="Car" accel="0.8" decel="4.5" sigma="0.5" '
             'length="5" minGap="2.5" maxSpeed="38" guiShape="passenger"/>\n')
    file_util.append_line_to_file(routes_file, token)
    input_output = self._random_traffic_generator.get_freeway_input_output()
    token = '    <!-- freeway routes -->'
    file_util.append_line_to_file(routes_file, token)
    freeway_routes = self._random_traffic_generator.setup_shortest_routes(
        input_output,
        edge_type_list=random_traffic_generator.FREEWAY_EDGE_TYPES,
        routes_file=routes_file,
        figures_folder=self._output_dir)
    token = '    <!-- freeway demands -->'
    file_util.append_line_to_file(routes_file, token)
    time_step_size = 100
    for time_point in range(0, 1200, time_step_size):
      freeway_routes_demands = [(0, 0.3), (1, 0.3)]
      self._random_traffic_generator.generate_routes_flow(
          time_point, time_step_size, freeway_routes, freeway_routes_demands,
          routes_file)
    token = '\n</routes>'
    file_util.append_line_to_file(routes_file, token)
    # Test by counting number of lines in the file.
    with file_util.f_open(routes_file, 'r') as f:
      self.assertLen(f.readlines(), 36)

  def test_generate_arterial_routes_flow(self):
    """Test for the arterial roads demand generation workflow.

    All the unit tests have been done above, and there is no calculation in this
    test. So this one just verifies nothing is block in the workflow.
    """
    routes_file = os.path.join(self._output_dir, 'arterial_routes_demands.xml')
    token = '<routes>\n'
    file_util.append_line_to_file(routes_file, token)
    token = ('    <vType id="Car" accel="0.8" decel="4.5" sigma="0.5" '
             'length="5" minGap="2.5" maxSpeed="38" guiShape="passenger"/>\n')
    file_util.append_line_to_file(routes_file, token)
    # Setup freeway routes.
    input_output = self._random_traffic_generator.get_freeway_input_output()
    token = '    <!-- freeway routes -->'
    file_util.append_line_to_file(routes_file, token)
    freeway_routes = self._random_traffic_generator.setup_shortest_routes(
        input_output,
        edge_type_list=random_traffic_generator.FREEWAY_EDGE_TYPES,
        routes_file=routes_file,
        figures_folder=self._output_dir)
    # Setup arterial roads routes.
    input_output = self._random_traffic_generator.get_arterial_input_output()
    token = '    <!-- arterial routes -->'
    file_util.append_line_to_file(routes_file, token)
    arterial_routes = self._random_traffic_generator.setup_shortest_routes(
        input_output,
        edge_type_list=(random_traffic_generator.FREEWAY_EDGE_TYPES +
                        random_traffic_generator.ARTERIAL_EDGE_TYPES),
        routes_file=routes_file,
        figures_folder=self._output_dir)
    token = '    <!-- freeway + arterial roads demands -->'
    file_util.append_line_to_file(routes_file, token)
    time_step_size = 100
    for time_point in range(0, 1200, time_step_size):
      freeway_routes_demands = [(0, 0.6), (1, 0.6), (2, 0.6)]
      self._random_traffic_generator.generate_routes_flow(
          time_point, time_step_size, freeway_routes, freeway_routes_demands,
          routes_file)
      arterial_routes_demands = [(route_id, 0.1) for route_id in
                                 range(len(arterial_routes))]
      self._random_traffic_generator.generate_routes_flow(
          time_point, time_step_size, arterial_routes, arterial_routes_demands,
          routes_file)
    token = '\n</routes>'
    file_util.append_line_to_file(routes_file, token)
    # Test by counting number of lines in the file.
    with file_util.f_open(routes_file, 'r') as f:
      self.assertLen(f.readlines(), 115)

  def test_generate_incomplete_routes_flow(self):
    """This is an example of creating incomplete routes demands."""
    routes_file = os.path.join(self._output_dir, 'incomplete_route_demands.xml')
    token = '<routes>\n'
    file_util.append_line_to_file(routes_file, token)
    token = ('    <vType id="Car" accel="0.8" decel="4.5" sigma="0.5" '
             'length="5" minGap="2.5" maxSpeed="38" guiShape="passenger"/>\n')
    file_util.append_line_to_file(routes_file, token)
    incomplete_routes_demands = [('700010432', '706447588#1', 1.2),
                                 ('700010432', '5018655', 1.0),
                                 ('700010432', '416943886#2', 0.2),
                                 ('700010432', '694409909#6', 0.2),
                                 ('27628577#0', '392017177#1', 0.2),
                                 ('27628577#0', '694409909#6', 0.2)]
    time_step_size = 100
    for time_point in range(0, 1200, time_step_size):
      self._random_traffic_generator.generate_incomplete_routes_flow(
          time_point, time_step_size, incomplete_routes_demands, routes_file)
    token = '\n</routes>'
    file_util.append_line_to_file(routes_file, token)
    # Test by counting number of lines in the file.
    with file_util.f_open(routes_file, 'r') as f:
      self.assertLen(f.readlines(), 78)

  def test_generate_departure_time(self):
    np.random.seed(123)
    mean = 1000
    std = 500
    number = 200000
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        mean, std)
    samples = self._random_traffic_generator.generate_departure_time(
        time_sampler_parameters, number)
    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    self.assertAlmostEqual(np.abs(mean - actual_mean) / mean, 0, places=2)
    self.assertAlmostEqual(np.abs(std - actual_std) / std, 0, places=2)

  def test_create_shortest_path_evacuation_demands(self):
    # In the small map, some of the edges are not connected. So if the
    # evacuation exit is set as one edge, then some of the edges may not have a
    # path out. The problem can be caught in the warning or info log messages.
    evacuation_edges = ['-415366780#0']

    # Calculate the distance to the evacuation exits.
    evacuation_path_trees = {}
    evacuation_path_length = {}
    for evacuation_edge in evacuation_edges:
      (evacuation_path_trees[evacuation_edge],
       evacuation_path_length[evacuation_edge]) = (
           self._net.getRestrictedShortestPathsTreeToEdge(evacuation_edge))

    # Settings for departure edges.
    departure_edges = self._net.getEdges()
    mean = 3 * 60 * 60
    std = 0.7 * 60 * 60
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        mean, std)
    car_per_meter_residential = 0.041666667
    # Generate evacuation demands.
    generator = self._random_traffic_generator
    zipped_demands = generator.create_evacuation_shortest_path_demands(
        departure_edges,
        time_sampler_parameters,
        car_per_meter_residential,
        evacuation_edges,
        evacuation_path_trees,
        evacuation_path_length)

    actual_destinations = [x.destination for x in zipped_demands]
    actual_number_car_per_edge = [x.num_cars for x in zipped_demands]
    target_destinations = [
        None, '-415366780#0', '-415366780#0', '-415366780#0', '-415366780#0',
        '-415366780#0', '-415366780#0', '-415366780#0', '-415366780#0']
    target_number_car_per_edge = [23, 1, 13, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    self.assertListEqual(actual_destinations[:9], target_destinations)
    self.assertListEqual(actual_number_car_per_edge[:25],
                         target_number_car_per_edge)

    # Output the demand xml file.
    output_file = os.path.join(self._output_dir, 'demands.rou.xml')
    generator.write_evacuation_vehicle_path_demands(
        zipped_demands, output_file)
    # Test for output demands file.
    with file_util.f_open(output_file, 'r') as f:
      self.assertLen(f.readlines(), 528)

  def test_create_auto_routing_evacuation_demands(self):
    # Settings for departure edges.
    departure_edges = self._net.getEdges()
    mean = 3 * 60 * 60
    std = 0.7 * 60 * 60
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        mean, std)
    car_per_meter_residential = 0.041666667
    # Generate evacuation demands.
    generator = self._random_traffic_generator
    zipped_demands = generator.create_evacuation_auto_routing_demands(
        departure_edges,
        time_sampler_parameters,
        car_per_meter_residential)

    actual_number_car_per_edge = [x.num_cars for x in zipped_demands]
    target_number_car_per_edge = [23, 1, 13, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    self.assertListEqual(actual_number_car_per_edge[:25],
                         target_number_car_per_edge)

    # Output the demand xml file.
    output_file = os.path.join(self._output_dir, 'demands.rou.xml')
    generator.write_evacuation_vehicle_auto_routing_demands(
        zipped_demands, 'taz_exit', output_file)
    # Test for output demands file.
    with file_util.f_open(output_file, 'r') as f:
      self.assertLen(f.readlines(), 427)


if __name__ == '__main__':
  absltest.main()
