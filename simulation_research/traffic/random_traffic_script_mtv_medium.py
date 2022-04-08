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

r"""Script.

Command line example:
blaze-bin/research/simulation/traffic/random_traffic_script_mtv \
-n map/mtv_medium_filtered_typed.net.xml -N 1000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags

from six.moves import range
import sumolib

from simulation_research.traffic import random_traffic_generator
from simulation_research.traffic import util

FLAGS = flags.FLAGS
flags.DEFINE_string('sumo_net_file', 'map/mtv_filtered_typed_trimed.net.xml',
                    'input network file in sumo format',
                    short_name='n')
flags.DEFINE_string('route_file_name', 'routes.rou.xml',
                    'prefix of the output file')
flags.DEFINE_string('output_dir', '.',
                    'output folder',
                    short_name='o')
flags.DEFINE_integer('random_seed', '369',
                     'random seed for the simulation')
flags.DEFINE_integer('simulation_duration', '3000',
                     'length of the simulation [sec]',
                     short_name='N')


def generate_arterial_routes_demands_main(_):
  """This is an example of generating demands only on arterial and freeways.

  The generated routes do no have the ones only on freeways.
  """
  net = sumolib.net.readNet(FLAGS.sumo_net_file)
  traffic_generator = random_traffic_generator.RandomTrafficGenerator(
      net)
  routes_file = os.path.join(FLAGS.output_dir, 'arterial_routes_demands.xml')
  token = '<routes>\n'
  util.append_line_to_file(routes_file, token)
  token = ('    <vType id="Car" accel="0.8" decel="4.5" sigma="0.5" '
           'length="5" minGap="2.5" maxSpeed="38" guiShape="passenger"/>\n')
  util.append_line_to_file(routes_file, token)
  # Setup freeway routes.
  figure_path = os.path.join(FLAGS.output_dir, 'freeway_routes.pdf')
  input_output = traffic_generator.get_freeway_input_output(
      figure_path=figure_path)
  token = '    <!-- freeway routes -->'
  util.append_line_to_file(routes_file, token)
  freeway_routes = traffic_generator.setup_shortest_routes(
      input_output,
      edge_type_list=random_traffic_generator.FREEWAY_EDGE_TYPES,
      routes_file=routes_file,
      figure_folder=None)
  # Setup arterial roads routes.
  figure_path = os.path.join(FLAGS.output_dir, 'arterial_routes.pdf')
  input_output = traffic_generator.get_arterial_input_output(
      figure_path=figure_path)
  token = '    <!-- arterial routes -->'
  util.append_line_to_file(routes_file, token)
  arterial_routes = traffic_generator.setup_shortest_routes(
      input_output,
      edge_type_list=(random_traffic_generator.FREEWAY_EDGE_TYPES +
                      random_traffic_generator.ARTERIAL_EDGE_TYPES),
      routes_file=routes_file,
      figure_folder=None)
  token = '    <!-- freeway + arterial roads demands -->'
  util.append_line_to_file(routes_file, token)
  time_step_size = 100
  for time_point in range(0, FLAGS.simulation_duration, time_step_size):
    # Create arbitrary
    freeway_routes_demands = [(0, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
    traffic_generator.generate_routes_flow(
        time_point, time_step_size, freeway_routes, freeway_routes_demands,
        routes_file)
    arterial_routes_demands = []
    # arterial_routes_demands = [(route_id, 0.002) for route_id in
    #                            range(len(arterial_routes))]
    for route_index, route in enumerate(arterial_routes):
      if (route['edge_from'].getID() == '27628577#0' and
          route['edge_to'].getID() == '23925644#3'):
        arterial_routes_demands.append((route_index, 0.3))
      elif (route['edge_from'].getID() == '17971093' and
            route['edge_to'].getID() == '23925644#3'):
        arterial_routes_demands.append((route_index, 0.3))
      elif (route['edge_from'].getID() == '17971093' and
            route['edge_to'].getID() == '-496364805#0'):
        arterial_routes_demands.append((route_index, 0.3))
      elif (route['edge_from'].getID() == '688239440' and
            route['edge_to'].getID() == '23925644#3'):
        arterial_routes_demands.append((route_index, 0.2))
      else:
        arterial_routes_demands.append((route_index, 0.01))
    traffic_generator.generate_routes_flow(
        time_point, time_step_size, arterial_routes, arterial_routes_demands,
        routes_file)
  token = '\n</routes>'
  util.append_line_to_file(routes_file, token)


if __name__ == '__main__':
  # app.run(setup_freeway_routes_main)
  app.run(setup_arterial_routes_main)
  # app.run(generate_arterial_routes_demands_main)
