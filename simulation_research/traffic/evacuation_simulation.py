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

"""Library for analysis of evacuation simulation results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import matplotlib.pylab as pylab
import numpy as np
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import map_visualizer
from simulation_research.traffic import random_traffic_generator
from simulation_research.traffic import simulation_data_parser


FLAGS = flags.FLAGS
flags.DEFINE_integer('random_seed', '369',
                     'random seed for the simulation.')

_DEMANDS = 'demands'


class EvacuationSimulationAnalyzer(object):
  """A convenience class for analyzing results of evacuation simulation.

  This is a one-off for facilitating the analysis for a specific paper resulting
  from the internship work of albertyuchen@. If it is used more generally, it
  should be refactored and tested.
  """

  def __init__(self, output_dir, sumo_net_file):
    self._output_dir = output_dir
    self._sumo_net_file = sumo_net_file

  def generate_evacuation_taz_demands(
      self, residential_car_density, serving_car_density,
      demand_mean_hours,
      demand_stddev_hours,
      population_portion):
    """Generates evacuation TAZ demands."""

    # TODO(yusef): Fix map + total number of cars.
    # To make the demands consistent, use the default map, paradise_type.net.xml
    # as the input map instead of the reversed. For Paradise map, an easy way to
    # check is that the total number of cars is 11072.
    net = sumolib.net.readNet(self._sumo_net_file)
    traffic_generator = random_traffic_generator.RandomTrafficGenerator(net)
    visualizer = map_visualizer.MapVisualizer(net)

    print(
        'Generating TAZ demands with STD: ', demand_stddev_hours,
        ' Portion: ', population_portion)

    # Demands from residential roads.
    residential_edge_type = ['highway.residential']
    residential_edges = net.filterEdges(residential_edge_type)
    demand_mean_seconds = demand_mean_hours * 60 * 60
    demand_stddev_seconds = demand_stddev_hours * 60 * 60
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        demand_mean_seconds, demand_stddev_seconds)
    car_per_meter_residential = residential_car_density * population_portion

    np.random.seed(FLAGS.random_seed)
    residential = traffic_generator.create_evacuation_auto_routing_demands(
        residential_edges,
        time_sampler_parameters,
        car_per_meter_residential)

    # Demands from parking roads.
    parking_edge_type = ['highway.service']
    parking_edges = net.filterEdges(parking_edge_type)
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        demand_mean_seconds, demand_stddev_seconds)
    car_per_meter_parking = serving_car_density * population_portion

    parking = traffic_generator.create_evacuation_auto_routing_demands(
        parking_edges,
        time_sampler_parameters,
        car_per_meter_parking)

    all_demands = residential + parking
    departure_time_points = [x.time for x in all_demands]
    cars_per_time_point = [x.num_cars for x in all_demands]
    departure_time_points = np.array(departure_time_points) / 3600
    print('TAZ demands. Total vehicles: ', sum(cars_per_time_point))

    # TODO(yusef): reconcile.
    demands_dir = os.path.join(self._output_dir, _DEMANDS)
    file_util.f_makedirs(demands_dir)
    output_hist_figure_path = os.path.join(
        demands_dir,
        'departure_time_histogram_taz_std_%s_portion_%s.pdf' %
        (demand_stddev_hours, population_portion))
    output_cumulative_figure_path = os.path.join(
        demands_dir,
        'departure_time_cumulative_taz_std_%s_portion_%s.pdf' %
        (demand_stddev_hours, population_portion))
    pkl_file = os.path.join(
        demands_dir,
        'demands_taz_tuple_std_%s_portion_%s.pkl' %
        (demand_stddev_hours, population_portion))
    routes_file = os.path.join(
        demands_dir,
        'demands_taz_std_%s_portion_%s.rou.xml' %
        (demand_stddev_hours, population_portion))

    # Output the demand xml file.
    visualizer.plot_demands_departure_time(
        departure_time_points,
        cars_per_time_point,
        output_hist_figure_path=output_hist_figure_path,
        output_cumulative_figure_path=output_cumulative_figure_path)
    file_util.save_variable(pkl_file, all_demands)
    exit_taz = 'exit_taz'
    traffic_generator.write_evacuation_vehicle_auto_routing_demands(
        all_demands, exit_taz, routes_file)

  def compare_demands_difference(self):
    """Compared the differences between demands and evacuations."""
    x = file_util.load_variable(
        'demands/demands_shortest_path_tuple_std_1.5.pkl')
    y = file_util.load_variable('demands/demands_taz_tuple_std_1.5.pkl')

    # x = [(a.origin, a.num_cars) for a in x]
    # y = [(a.origin, a.num_cars) for a in y]
    x_cars = [a.num_cars for a in x]
    y_cars = [a.num_cars for a in y]
    print(sum(x_cars), sum(y_cars))
    x = [(a.origin, a.num_cars, a.time) for a in x]
    y = [(a.origin, a.num_cars, a.time) for a in y]
    x = set(x)
    y = set(y)
    print(len(x), len(y))
    print(x.issubset(y))

    common = x.intersection(y)
    print(len(common))
    x_ = x.difference(common)
    y_ = y.difference(common)
    print(x_)
    print(y_)

  def generate_evacuation_shortest_path_demands(
      self, residential_car_density, serving_car_density,
      evacuation_edges, demand_mean_hours, demand_stddev_hours,
      population_portion):
    """Generates evacuation demands."""
    net = sumolib.net.readNet(self._sumo_net_file)
    traffic_generator = random_traffic_generator.RandomTrafficGenerator(net)
    visualizer = map_visualizer.MapVisualizer(net)

    print('Generating TAZ demands with STD: ', demand_stddev_hours,
          ' Portion: ', population_portion)

    # Calculate the distance to the evacuation exits.
    evacuation_path_trees = {}
    evacuation_path_length = {}
    for exit_edge in evacuation_edges:
      evacuation_path_trees[exit_edge], evacuation_path_length[exit_edge] = (
          net.getRestrictedShortestPathsTreeToEdge(exit_edge))

    # Demands from residential roads.
    residential_edge_type = ['highway.residential']
    residential_edges = net.filterEdges(residential_edge_type)
    demand_mean_seconds = demand_mean_hours * 60 * 60
    demand_stddev_seconds = demand_stddev_hours * 60 * 60
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        demand_mean_seconds, demand_stddev_seconds)
    car_per_meter_residential = residential_car_density * population_portion

    np.random.seed(FLAGS.random_seed)
    residential = traffic_generator.create_evacuation_shortest_path_demands(
        residential_edges,
        time_sampler_parameters,
        car_per_meter_residential,
        evacuation_edges,
        evacuation_path_trees,
        evacuation_path_length)

    # Demands from parking roads.
    parking_edge_type = ['highway.service']
    parking_edges = net.filterEdges(parking_edge_type)
    time_sampler_parameters = random_traffic_generator.TimeSamplerGammaMeanStd(
        demand_mean_seconds, demand_stddev_seconds)
    car_per_meter_parking = serving_car_density * population_portion

    parking = traffic_generator.create_evacuation_shortest_path_demands(
        parking_edges,
        time_sampler_parameters,
        car_per_meter_parking,
        evacuation_edges,
        evacuation_path_trees,
        evacuation_path_length)

    all_demands = residential + parking
    departure_time_points = [x.time for x in all_demands]
    cars_per_time_point = [x.num_cars for x in all_demands]
    departure_time_points = np.array(departure_time_points) / 3600
    print('Shortest path demands. Total vehicles: ', sum(cars_per_time_point))

    # Output the demand xml file.
    demands_dir = os.path.join(self._output_dir, _DEMANDS)
    file_util.f_makedirs(demands_dir)
    output_hist_figure_path = os.path.join(
        demands_dir,
        'departure_time_histogram_shortest_path_std_%s_portion_%s.pdf' %
        (demand_stddev_hours, population_portion))
    output_cumulative_figure_path = os.path.join(
        demands_dir,
        'departure_time_cumulative_shortest_path_std_%s_portion_%s.pdf' %
        (demand_stddev_hours, population_portion))
    pkl_file = os.path.join(
        demands_dir,
        'demands_shortest_path_tuple_std_%s_portion_%s.pkl' %
        (demand_stddev_hours, population_portion))
    routes_file = os.path.join(
        demands_dir,
        'demands_shortest_path_std_%s_portion_%s.rou.xml' %
        (demand_stddev_hours, population_portion))

    visualizer.plot_demands_departure_time(
        departure_time_points,
        cars_per_time_point,
        output_hist_figure_path=output_hist_figure_path,
        output_cumulative_figure_path=output_cumulative_figure_path)
    file_util.save_variable(pkl_file, all_demands)
    traffic_generator.write_evacuation_vehicle_path_demands(
        all_demands, routes_file)

  def parse_fcd_results_single_file(self, hours):
    """Extract the data then save to file."""
    net = sumolib.net.readNet(self._sumo_net_file)
    data_parser = simulation_data_parser.SimulationDataParser()
    plot_edges = net.getEdges()

    fcd_file = os.path.join(self._output_dir, 'traffic.fcd.xml')
    output_folder = os.path.join(self._output_dir, 'trajectory/')
    if not file_util.f_exists(output_folder):
      file_util.f_mkdir(output_folder)

    time_segment_length_seconds = hours * 3600
    time_range_seconds = [0, 3600 * 12]
    data_parser.save_batch_edge_id_to_trajectory(
        fcd_file, plot_edges,
        time_range=time_range_seconds,
        time_segment_length=time_segment_length_seconds,
        parse_time_step=10, output_folder=output_folder)

  def parse_fcd_results_multiple_files(self):
    """Extract the data then save to file."""
    net = sumolib.net.readNet(self._sumo_net_file)
    data_parser = simulation_data_parser.SimulationDataParser()
    plot_edges = net.getEdges()

    fcd_file_folder = 'output/fcd_segments/'
    # fcd_file_list = os.listdir(fcd_file_folder)
    fcd_file_list = ['traffic.segment_2.fcd.xml']

    output_folder = os.path.join(self._output_dir, 'trajectory/')
    for fcd_file in fcd_file_list:
      print('Analyzing file: ', fcd_file)
      # time_segment_length = 0.5 * 3600
      time_segment_length = None
      # time_range = [0, 3600*12]
      time_range = None

      data_parser.save_batch_edge_id_to_trajectory(
          os.path.join(fcd_file_folder, fcd_file),
          plot_edges,
          time_range=time_range, time_segment_length=time_segment_length,
          parse_time_step=10, output_folder=output_folder)

  def visualize_fcd_on_map(self):
    """Plot metric maps.

    Pay attention to the map.
    """
    net = sumolib.net.readNet(self._sumo_net_file)
    visualizer = map_visualizer.MapVisualizer(net)
    plot_edges = net.getEdges()

    trajectory_folder = os.path.join(self._output_dir, 'trajectory/')
    output_folder = os.path.join(trajectory_folder, 'trajectory_fig/')
    if not file_util.f_exists(output_folder):
      file_util.f_mkdir(output_folder)

    trajectory_file_list = os.listdir(trajectory_folder)
    # trajectory_file_list = [
    #     'edge_id_to_trajectory_9000_10800.pkl']

    for trajectory_file in trajectory_file_list:
      if not trajectory_file.endswith('.pkl'):
        continue
      trajectory_pkl_file = os.path.join(trajectory_folder, trajectory_file)
      print('Loading file: ', trajectory_pkl_file)
      edge_id_to_trajectory = file_util.load_variable(trajectory_pkl_file)
      print('Time range: ', edge_id_to_trajectory['time_interval'])
      output_figure_path = (output_folder + 'speed_map_%s_%s.pdf' %
                            (int(edge_id_to_trajectory['time_interval'][0]),
                             int(edge_id_to_trajectory['time_interval'][1])))

      visualizer.plot_edge_trajectory_histogram_on_map(
          plot_edges,
          edge_id_to_trajectory,
          output_figure_path=output_figure_path,
          plot_max_speed=13.4112)

  def _extract_detector_data(self):
    """Extracts detector data form xml files."""
    data_parser = simulation_data_parser.SimulationDataParser()
    visualizer = map_visualizer.MapVisualizer()

    detector_folder = os.path.join(self._output_dir, 'detector/')
    detector_trajectory_folder = os.path.join(
        detector_folder, 'detector_trajectory/')

    if not file_util.exists(detector_trajectory_folder):
      file_util.mkdir(detector_trajectory_folder)

    detector_files = os.listdir(detector_folder)
    for detector_file in detector_files:
      if not detector_file.endswith('.xml'):
        continue
      # print('Extract file: ', detector_file)
      output_file = os.path.splitext(detector_file)[0]+'.pkl'
      output_file = os.path.join(detector_trajectory_folder, output_file)
      detector_file = os.path.join(detector_folder, detector_file)
      print('Save file: ', output_file)
      data_parser.get_and_save_detector_data(detector_file, output_file)

    # Creates figures for individual detector.
    output_figure_folder = os.path.join(detector_folder, 'detector_fig/')
    if not file_util.f_exists(output_figure_folder):
      file_util.f_mkdir(output_figure_folder)
    visualizer.plot_individual_detector(
        detector_trajectory_folder, output_figure_folder)

  # NB: This method and the following seem hard-coded for Paradise.
  def plot_save_detector_data_normal(self):
    """Plots detector data.

    Paradise evacuation edges:
    '27323694.1622',  # Skyway Rd.
    '10293408#4',     # Neal Rd.
    '-184839999#0',   # Clark Rd.
    '-538864403#0'    # Pentz Rd.
    """
    self._extract_detector_data()

    detector_trajectory_folder = os.path.join(
        self._output_dir, 'detector/detector_trajectory/')
    output_figure_folder = os.path.join(
        self._output_dir, 'detector/detector_fig/')
    if not file_util.exists(output_figure_folder):
      file_util.mkdir(output_figure_folder)
    visualizer = map_visualizer.MapVisualizer()

    detector_pkl_files_by_group = [
        [detector_trajectory_folder + 'e1Detector_27323694_0_0.pkl',
         detector_trajectory_folder + 'e1Detector_27323694_1_1.pkl'],
        [detector_trajectory_folder + 'e1Detector_10293408#4_0_2.pkl'],
        [detector_trajectory_folder + 'e1Detector_-184839999#0_0_3.pkl',
         detector_trajectory_folder + 'e1Detector_-184839999#0_1_4.pkl'],
        [detector_trajectory_folder + 'e1Detector_-538864403#0_0_5.pkl']]
    visualizer.plot_detector_flow_density_by_group(
        detector_pkl_files_by_group,
        ['Skyway', 'Neal_Rd', 'Clark_Rd', 'Pentz_Rd'],
        output_figure_folder=output_figure_folder)

    # Cumulative vehicle flow.
    detector_pkl_files = [
        detector_trajectory_folder + 'e1Detector_27323694_0_0.pkl',
        detector_trajectory_folder + 'e1Detector_27323694_1_1.pkl',
        detector_trajectory_folder + 'e1Detector_10293408#4_0_2.pkl',
        detector_trajectory_folder + 'e1Detector_-184839999#0_0_3.pkl',
        detector_trajectory_folder + 'e1Detector_-184839999#0_1_4.pkl',
        detector_trajectory_folder + 'e1Detector_-538864403#0_0_5.pkl']

    visualizer.plot_detector_arrival_time_by_group(
        detector_pkl_files,
        output_figure_folder)

  # NB: This method and the following seem hard-coded for Paradise.
  def plot_save_detector_data_reverse(self):
    """Plots detector data.

    Paradise evacuation edges:
    '27323694.1622',  # Skyway Rd.
    '37625137#0.49'   # Skyway Rd reverse.
    '10293408#4',     # Neal Rd.
    '-184839999#0',   # Clark Rd.
    '-538864403#0'    # Pentz Rd.
    """
    self._extract_detector_data()

    detector_trajectory_folder = os.path.join(
        self._output_dir, 'detector/detector_trajectory/')
    output_figure_folder = os.path.join(
        self._output_dir, 'detector/detector_fig/')
    if not file_util.exists(output_figure_folder):
      file_util.mkdir(output_figure_folder)
    visualizer = map_visualizer.MapVisualizer()

    detector_pkl_files_by_group = [
        [detector_trajectory_folder + 'e1Detector_27323694_0_0.pkl',
         detector_trajectory_folder + 'e1Detector_27323694_1_1.pkl',
         detector_trajectory_folder + 'e1Detector_37625137#1_0_6.pkl',
         detector_trajectory_folder + 'e1Detector_37625137#1_1_7.pkl'],
        [detector_trajectory_folder + 'e1Detector_10293408#4_0_2.pkl'],
        [detector_trajectory_folder + 'e1Detector_-184839999#0_0_3.pkl',
         detector_trajectory_folder + 'e1Detector_-184839999#0_1_4.pkl',
         detector_trajectory_folder + 'e1Detector_-184839999#0_2_8.pkl',
         detector_trajectory_folder + 'e1Detector_-184839999#0_3_9.pkl'],
        [detector_trajectory_folder + 'e1Detector_-538864403#0_0_5.pkl',
         detector_trajectory_folder + 'e1Detector_-538864403#0_1_10.pkl']]
    visualizer.plot_detector_flow_density_by_group(
        detector_pkl_files_by_group,
        ['Skyway', 'Neal_Rd', 'Clark_Rd', 'Pentz_Rd'],
        output_figure_folder=output_figure_folder)

    # Cumulative vehicle flow.
    detector_pkl_files = [
        'e1Detector_27323694_0_0.pkl',
        'e1Detector_27323694_1_1.pkl',
        'e1Detector_37625137#1_0_6.pkl',
        'e1Detector_37625137#1_1_7.pkl',
        'e1Detector_10293408#4_0_2.pkl',
        'e1Detector_-184839999#0_0_3.pkl',
        'e1Detector_-184839999#0_1_4.pkl',
        'e1Detector_-184839999#0_2_8.pkl',
        'e1Detector_-184839999#0_3_9.pkl',
        'e1Detector_-538864403#0_0_5.pkl',
        'e1Detector_-538864403#0_1_10.pkl']
    detector_pkl_files = [os.path.join(detector_trajectory_folder, filename) for
                          filename in detector_pkl_files]

    visualizer.plot_detector_arrival_time_by_group(
        detector_pkl_files,
        output_figure_folder)

  def _analyze_summary_demands_vs_evacuation(self, demand_file,
                                             summary_file,
                                             output_dir=None):
    """Plot summary vs demands."""
    data_parser = simulation_data_parser.SimulationDataParser()
    visualizer = map_visualizer.MapVisualizer()

    demands = file_util.load_variable(demand_file)
    sorted_demands = sorted(demands, key=lambda x: x.time)
    demand_time_line = [x.time for x in sorted_demands]
    demand_time_line = np.array(demand_time_line) / 3600
    demand_car_count = [x.num_cars for x in sorted_demands]
    demand_cumulative_values = (
        np.cumsum(demand_car_count) / sum(demand_car_count))

    summary = data_parser.parse_summary_file(summary_file)
    summary_time_line = np.array(summary['time']) / 3600
    summary_cumulative_values = (
        np.array(summary['ended']) / sum(demand_car_count))

    # Calculate the gap between them.
    gap_area = visualizer.calculate_gap_area_between_cummulative_curves(
        demand_time_line, demand_cumulative_values,
        summary_time_line, summary_cumulative_values)

    if not output_dir:
      return (demand_time_line, demand_cumulative_values, summary_time_line,
              summary_cumulative_values, gap_area)

    # Plot demands v.s. evacuation.
    fig = pylab.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    pylab.plt.plot(demand_time_line, demand_cumulative_values, label='Demands')
    pylab.plt.plot(
        summary_time_line, summary_cumulative_values, label='Evacuation')
    visualizer.add_pertentage_interception_lines(
        summary_time_line, summary_cumulative_values, [0.5, .9, .95])
    pylab.plt.xlabel('Time [h]')
    pylab.plt.ylabel('Cummulative percentage of total vehicles')
    pylab.plt.legend()
    ax.autoscale_view(True, True, True)
    output_figure_path = os.path.join(output_dir, 'evacuation_curve.pdf')
    pylab.savefig(output_figure_path)

    return (demand_time_line, demand_cumulative_values, summary_time_line,
            summary_cumulative_values, gap_area)

  def plot_summary_demands_vs_evacuation_group(
      self, demand_files, output_dir_names, labels):
    """Plots demands-evacuation curves."""
    gap_areas = []

    fig = pylab.figure(figsize=(8, 6))
    fig.add_subplot(111)

    for i, (demand_file, output_dir_name) in enumerate(
        zip(demand_files, output_dir_names)):
      print('Processing: ', demand_file, 'Summary: ', output_dir_name)
      (demand_time_line, demand_cumulative_values,
       summary_time_line, summary_cumulative_values,
       gap_area) = self._analyze_summary_demands_vs_evacuation(
           demand_file, os.path.join(output_dir_name, 'summary.xml'), None)
      gap_areas.append(gap_area)
      print('Gap: %.3f' % gap_area)

      pylab.plt.plot(demand_time_line, demand_cumulative_values, '--',
                     color=pylab.plt.cm.jet(i/3), label=labels[i])
      pylab.plt.plot(summary_time_line, summary_cumulative_values,
                     color=pylab.plt.cm.jet(i/3))
    pylab.plt.xlim(0, 8)
    pylab.plt.legend(loc='lower right')
    pylab.plt.xlabel('Time [h]')
    pylab.plt.ylabel('Vehicles cummulative percentage')
    pylab.savefig(os.path.join(self._output_dir, 'test.pdf'))

  def plot_traveling_time(self):
    """Plot tripinfo data."""
    visualizer = map_visualizer.MapVisualizer()
    data_parser = simulation_data_parser.SimulationDataParser()
    tripinfo_file = 'output/tripinfo.xml'
    # output_folder = 'output/'
    # output_file = 'tripinfo.pkl'

    tripinfo = data_parser.get_tripinfo_attribute_to_trips(tripinfo_file)

    bins = np.linspace(0, 12, 49)
    positions_on_edge = (np.array(tripinfo['depart']) -
                         np.array(tripinfo['departDelay'])) / 3600
    values_on_edge = np.array(tripinfo['duration'])
    print(len(values_on_edge), len(values_on_edge))

    # print(positions_on_edge)
    bin_mean, bin_boundary = visualizer._histogram_along_edge(
        values_on_edge, positions_on_edge, bins=bins)

    # print(bin_mean, bin_boundary)
    fig = pylab.figure(figsize=(8, 6))
    fig.add_subplot(111)
    pylab.plt.plot(bin_boundary[:-1], bin_mean)
    pylab.plt.xlabel('Time [h]')
    pylab.plt.xlim(0, 10)
    pylab.plt.ylim(0, 10000)
    pylab.plt.ylabel('Average traveling time.')
    pylab.savefig(os.path.join(self._output_dir, 'traveling_time_hist.pdf'))

  def plot_map(self, output_file_name):
    """Plot the edges by types."""
    net = sumolib.net.readNet(self._sumo_net_file)
    visualizer = map_visualizer.MapVisualizer(net)
    residential_edge_type = ['highway.residential']
    parking_edge_type = ['highway.service']
    residential_edges = net.filterEdges(residential_edge_type)
    parking_edges = net.filterEdges(parking_edge_type)

    visualizer.plot_edges(
        [(residential_edges, 'lime', 0.2),
         (parking_edges, 'darkgreen', 0.2)],
        output_figure_path=os.path.join(self._output_dir, output_file_name))

  def data_explore(self):
    """Generate cars from residential roads."""
    net = sumolib.net.readNet(self._sumo_net_file)
    residential_edge_type = ['highway.residential']
    residential_edges = net.filterEdges(residential_edge_type)
    service_edge_type = ['highway.service']
    service_edges = net.filterEdges(service_edge_type)

    residential_road_lengths = []
    for e in residential_edges:
      residential_road_lengths.append(e.getLength())

    service_road_lengths = []
    for e in service_edges:
      service_road_lengths.append(e.getLength())

    print('Total number of all edges:', len(net.getEdges()))

    print('Residential road stats')
    print('Sum of lengths: ', np.sum(residential_road_lengths))
    print('Max length: ', np.max(residential_road_lengths))
    print('Min length: ', np.min(residential_road_lengths))
    print('# edges < 1 / 0.0415: ',
          np.sum(np.array(residential_road_lengths) < 1/0.0415))
    print('Sum of residential roads: ', len(residential_road_lengths))
    print('')
    print('Services road stats')
    print('Sum of lengths: ', np.sum(service_road_lengths))
    print('Max length: ', np.max(service_road_lengths))
    print('Min length: ', np.min(service_road_lengths))
    print('# edges < 1 / 0.0415: ',
          np.sum(np.array(service_road_lengths) < 1/0.0415))
    print('Sum of residential roads: ', len(service_road_lengths))
