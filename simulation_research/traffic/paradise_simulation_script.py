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
import matplotlib.pylab as pylab
import numpy as np
import sumolib

from simulation_research.traffic import evacuation_simulation
from simulation_research.traffic import file_util
from simulation_research.traffic import map_visualizer
from simulation_research.traffic import simulation_data_parser


FLAGS = flags.FLAGS
flags.DEFINE_string('sumo_net_file', 'map/paradise_typed.net.xml',
                    'input network file in sumo format',
                    short_name='n')
flags.DEFINE_string('output_dir', 'output', 'output folder')
flags.DEFINE_float('demand_mean_hours', 3,
                   'The mean of the demands (in hours).')
flags.DEFINE_float('demand_stddev_hours', 1.5,
                   'The standard deviation of the demands (in hours).')
flags.DEFINE_float('population_portion', 1,
                   'The percentage of cars that need to be generated.')

PARADISE_RESIDENTIAL_CAR_DENSITY = 0.041666667
PARADISE_SERVING_CAR_DENSITY = PARADISE_RESIDENTIAL_CAR_DENSITY * 4


def scenarios_summary_comparison(output_dir):
  """Compare different scenarios."""
  data_parser = simulation_data_parser.SimulationDataParser()
  visualizer = map_visualizer.MapVisualizer()

  fig = pylab.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)

  demands = file_util.load_variable(
      'Paradise_template/demands/demands_taz_tuple_std_0.7.pkl')
  sorted_demands = sorted(demands, key=lambda x: x.time)
  demand_time_line = [x.time for x in sorted_demands]
  demand_car_count = [x.num_cars for x in sorted_demands]

  cumulative_values = np.cumsum(demand_car_count) / sum(demand_car_count)
  pylab.plt.plot(np.array(demand_time_line) / 3600,
                 cumulative_values, ':', label='Demands', color='black')

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL/output_std_0.7/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, '--', label='No block')

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_21600/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=5 h', color=pylab.plt.cm.jet(1/6))

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_18000/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=4 h', color=pylab.plt.cm.jet(2/6))

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_10800/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=3 h', color=pylab.plt.cm.jet(3/6))

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_7200/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=2 h', color=pylab.plt.cm.jet(4/6))

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_3600/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=1 h', color=pylab.plt.cm.jet(5/6))

  summary = data_parser.parse_summary_file(
      'Paradise_RevRd_noTFL_road_blocker/output_std_0.7_road_block_0/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = np.array(summary['ended']) / sum(demand_car_count)
  pylab.plt.plot(time_line, cumulative_values, label='t=0 h', color=pylab.plt.cm.jet(0.95))
  # visualizer.add_pertentage_interception_lines(
  #     time_line, cumulative_values, [0.5, .9, .95])

  pylab.plt.xlabel('Time [h]')
  pylab.plt.ylabel('Cummulative vehicles')
  ax.autoscale_view(True, True, True)
  pylab.plt.xlim(1, 6)
  pylab.plt.legend(loc='lower right')
  pylab.savefig(os.path.join(
      output_dir, 'evacuation_curve_std_0.7_road_block_comparison.pdf'))


def scenarios_detector_comparison(output_dir):
  """Compare different scenarios."""
  data_parser = simulation_data_parser.SimulationDataParser()
  visualizer = map_visualizer.MapVisualizer()

  fig = pylab.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)

  load_input = file_util.load_variable(
      'Paradise_reverse_roads/demands/demands_taz_tuple.pkl')
  load_input = sorted(load_input)
  demand_time_line, _, demand_car_count = zip(*load_input)
  cumulative_values = np.cumsum(demand_car_count)
  pylab.plt.plot(np.array(demand_time_line) / 3600, cumulative_values)
  # print(cumulative_values[-1])
  # print(np.sum(demand_car_count))
  # visualizer.add_pertentage_interception_lines(
  #     np.array(demand_time_line) / 3600, demand_car_count, [0.5, .9, .95])

  # detector_trajectory_folder = 'Paradise_reverse_roads/output/detector/detector_trajectory/'
  # (time_line, arrival_car_count) = file_util.load_variable(
  #     detector_trajectory_folder + 'all_arrival_flow.pkl')
  # cumulative_values = np.cumsum(arrival_car_count)
  # print(cumulative_values[-1])
  # pylab.plt.plot(time_line, cumulative_values)
  # visualizer.add_pertentage_interception_lines(
  #     time_line, arrival_car_count, [0.5, .9, .95])
  summary = data_parser.parse_summary_file(
      'Paradise_reverse_roads/output/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = summary['ended']
  pylab.plt.plot(time_line, cumulative_values)
  visualizer.add_pertentage_interception_lines(
      time_line, cumulative_values, [0.5, .9, .95])

#   detector_trajectory_folder = 'Paradise_auto_routing/output/detector/detector_trajectory/'
#   (time_line, arrival_car_count) = file_util.load_variable(
#       detector_trajectory_folder + 'all_arrival_flow.pkl')
#   cumulative_values = np.cumsum(arrival_car_count)
#   print(cumulative_values[-1])
#   pylab.plt.plot(time_line / 3600, cumulative_values)
#   # visualizer.add_pertentage_interception_lines(
#   #     time_line, arrival_car_count, [0.5, .9, .95])
  summary = data_parser.parse_summary_file(
      'Paradise_auto_routing/output/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = summary['ended']
  pylab.plt.plot(time_line, cumulative_values)
  visualizer.add_pertentage_interception_lines(
      time_line, cumulative_values, [0.5, .9, .95])

#   detector_trajectory_folder = 'Paradise_2s_baseline/output/detector/detector_trajectory/'
#   (time_line, arrival_car_count) = file_util.load_variable(
#       detector_trajectory_folder + 'all_arrival_flow.pkl')
#   cumulative_values = np.cumsum(arrival_car_count)
#   print(cumulative_values[-1])
#   pylab.plt.plot(time_line, cumulative_values)
  summary = data_parser.parse_summary_file(
      'Paradise_shortest_path_baseline/output/summary.xml')
  time_line = np.array(summary['time']) / 3600
  cumulative_values = summary['ended']
  pylab.plt.plot(time_line, cumulative_values)
  visualizer.add_pertentage_interception_lines(
      time_line, cumulative_values, [0.5, .9, .95])

  pylab.plt.xlabel('Time [h]')
  pylab.plt.ylabel('Cummulative vehicles')
  ax.autoscale_view(True, True, True)
  pylab.savefig(os.path.join(output_dir, 'scenarios_arrival_comparison.pdf'))


def plot_path(sumo_net_file):
  """Plot path."""
  net = sumolib.net.readNet(sumo_net_file)
  map_visualizer.PLOT_MAX_SPEED = 13.4112  # 30 mph.
  map_visualier = map_visualizer.MapVisualizer(net)
  vehicle_type_list = 'passenger'
  edge_from = '-8953730#1'
  edge_from = net.getEdge(edge_from)
  edge_to = '514320223'
  edge_to = net.getEdge(edge_to)
  print(edge_from.getID()+'_'+ edge_to.getID())
  # return
  path_edges, route_length = net.getRestrictedShortestPath(
      edge_from, edge_to,
      vehicleClass=vehicle_type_list)
  selected_edges = [([edge_from], 'lime', 1), ([edge_to], 'red', 1)]
  selected_edges = [(path_edges, 'darkblue', 1)] + selected_edges
  map_visualier.plot_edges(
      selected_edges,
      output_figure_path=(edge_from.getID()+'_'+
                          edge_to.getID()+'_path.pdf'))
  print(route_length)


def main(_):
  analyzer = evacuation_simulation.EvacuationSimulationAnalyzer(
      FLAGS.output_dir, FLAGS.sumo_net_file)
  # analyzer.generate_evacuation_taz_demands(
  #     PARADISE_RESIDENTIAL_CAR_DENSITY,
  #     PARADISE_SERVING_CAR_DENSITY,
  #     FLAGS.demand_mean_hours,
  #     FLAGS.demand_stddev_hours,
  #     FLAGS.population_portion)

  # analyzer.compare_demands_difference()
  # evacuation_edges = ['27323694.1622',  # Skyway Rd.
  #                     '10293408#4',  # Neal Rd.
  #                     '-184839999#0',  # Clark Rd.
  #                     '-538864403#0']  # Pentz Rd.
  # analyzer.generate_evacuation_shortest_path_demands(
  #     PARADISE_RESIDENTIAL_CAR_DENSITY,
  #     PARADISE_SERVING_CAR_DENSITY,
  #     evacuation_edges,
  #     FLAGS.demand_mean_hours,
  #     0.4,
  #     FLAGS.population_portion)

  # analyzer.parse_fcd_results_single_file(hours=0.25)
  # analyzer.parse_fcd_results_multiple_files()
  # analyzer.visualize_fcd_on_map()

  # analyzer.plot_save_detector_data_normal()
  # analyzer.plot_save_detector_data_reverse()

  # demand_files = [
  #     # 'demands/demands_shortest_path_tuple_std_0.4.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  #     'demands/demands_taz_tuple_std_0.7.pkl',
  # ]
  # output_dirs = [
  #     'output_std_0.7_road_block_0',
  #     'output_std_0.7_road_block_3600',
  #     'output_std_0.7_road_block_7200',
  #     'output_std_0.7_road_block_10800',
  #     'output_std_0.7_road_block_14400',
  #     'output_std_0.7_road_block_18000',
  #     'output_std_0.7_road_block_21600',
  # ]

  # labels = ['t=0 h', 't=0 h', 't=0 h', 't=0 h', 't=0 h', 't=0 h', 't=0 h']
  # analyzer.plot_summary_demands_vs_evacuation_group(
  #     demand_files, output_dirs, labels)
  # scenarios_summary_comparison(FLAGS.output_dir)
  # scenarios_detector_comparison(FLAGS.output_dir)
  # analyzer.plot_traveling_time()

  # plot_path(FLAGS.sumo_net_file)
  analyzer.plot_map('Paradise_map.pdf')
  # analyzer.data_explore()


if __name__ == '__main__':
  app.run(main)
