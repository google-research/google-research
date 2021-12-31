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
"""Extracts various types of data from the simulation results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os

from absl import logging
import six
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import memoize_util

# WARNING: Check if the target xml file has the corresponding entries as
# follows. The entries have to be the subset of the xml file. If there is an
# entry in the defalt settings but not in the xml file, then the function is not
# able to extract the information. This is due to the usage of the third party
# `sumolib.xml.parse_fast_nested`.
_DEFAULT_FCD_FILE_ENTRIES = [
    'id', 'x', 'y', 'angle', 'type', 'speed', 'pos', 'lane']
_DEFAULT_SUMMARY_FILE_ENTRIES = [
    'time', 'loaded', 'inserted', 'running', 'waiting', 'ended',
    'meanWaitingTime', 'halting', 'meanSpeed', 'meanSpeedRelative', 'duration']
_DEFAULT_ROUTE_FILE_VEHICLE_ENTRIES = [
    'id', 'depart', 'departLane', 'departPos', 'departSpeed', 'arrival',
    'routeLength']
_DEFAULT_ROUTE_FILE_ROUTE_ENTRIES = ['edges']
_DEFAULT_TRIPINFO_FILE_ENTRIES = [
    'id', 'depart', 'departLane', 'departPos', 'departSpeed', 'departDelay',
    'arrival', 'arrivalLane', 'arrivalPos', 'arrivalSpeed', 'duration',
    'routeLength', 'waitingTime', 'waitingCount', 'stopTime', 'timeLoss',
    'rerouteNo', 'devices', 'vType', 'speedFactor', 'vaporized']
_DEFAULT_DETECTOR_FILE_ENTRIES = [
    'begin', 'end', 'id', 'nVehContrib', 'flow', 'occupancy', 'speed', 'length',
    'nVehEntered']


class SimulationDataParser(object):
  """Extract data from the simulation results.

  There are several types of files in the SUMO simulation as follows.
  FCD file: The FCD (floating car data) export contains location and speed
      along with other information for every vehicle in the network at every
      time step. The output behaves somewhat like a super-accurate high-
      frequency GPS device for each vehicle. The outputs can be processed
      further using the TraceExporter tool for adapting frequency, equipment
      rates, accuracy and data format.
  Summary file: This output contains the simulation-wide number of vehicles
      that are loaded, inserted, running, waiting to be inserted, have reached
      their destination and how long they needed to finish the route. The last
      value is normalised over all vehicles that have reached their
      destination so far.
  Vehicle route file: Not to be confused with the input route file. This one is
      the output of file of the simulation, tracking individual vehicles. The
      vehicle routes output contains information about which route a vehicle
      took and if his route was replaced at any time by a new one, each of
      the previous routes together with the edge at the time their replacement
      took place is reported. Furthermore, the times the vehicle has entered
      and has left the network are stored herein.
  More details of different types of SUMO files can be found in:
  https://sumo.dlr.de/docs/Simulation/Output.html
  """

  def __init__(self):
    # Positions at each time step. It includes inner edge (connections between
    # edges, name starting with ":"). It may also skip very short edges. If an
    # edge is 0.2 m, it may be skipped by every second timestamp. This variable
    # is acquired from the FCD file.
    self._vehicle_id_to_hop_route = collections.defaultdict(list)
    # Two entries mapping: [edge id][metric] --> metric time series on the edge.
    # This variable is acquired from FCD file.
    self._edge_id_to_trajectory = collections.defaultdict(
        lambda: collections.defaultdict(list))
    # Two entries mapping: [vehicle id][metric] --> metric time series of the
    # vehicle. This variable is acquired from the FCD file.
    self._vehicle_id_to_trajectory = collections.defaultdict(
        lambda: collections.defaultdict(list))
    # Mapping [metric] --> metric time series. This variable is acquired from
    # the summary file.
    self._summary_attribute_to_time_series = collections.defaultdict(list)
    # Two entries mapping: [edge id][metric] --> metric list. This variable is
    # acquired from the output route file.
    self._edge_id_to_attribute = collections.defaultdict(
        lambda: collections.defaultdict(list))
    # Mapping: [vehicle id] --> full path of the vehicle. This variable is
    # acquired from the output route file. This one shows the full path of a
    # vehicle, and it does not include the inner edge at crossroads. So it is
    # slighty different from the self._vehicle_id_to_hop_route.
    self._vehicle_id_to_full_route = collections.defaultdict(list)
    # Mapping [metric] --> the metric of all trips. This variable is acquired
    # from the tripinfo file.
    self._tripinfo_attribute_to_trips = collections.defaultdict(list)

  @memoize_util.MemoizeClassFunctionOneRun
  def _parse_fcd_file_by_vehicle(self, fcd_file):
    """Gets vehicles' trajectories."""
    for timestep, vehicle in sumolib.xml.parse_fast_nested(
        fcd_file, 'timestep', ['time'], 'vehicle', _DEFAULT_FCD_FILE_ENTRIES):
      if timestep is None or vehicle is None:
        continue
      time = float(timestep.time)
      speed = float(vehicle.speed)
      previous_time = time
      previous_speed = speed
      previous_distance = 0
      if vehicle.id in self._vehicle_id_to_trajectory:
        previous_time = self._vehicle_id_to_trajectory[vehicle.id]['time'][-1]
        previous_speed = self._vehicle_id_to_trajectory[vehicle.id]['speed'][-1]
        previous_distance = self._vehicle_id_to_trajectory[
            vehicle.id]['distance'][-1]
      average_speed = (speed + previous_speed) / 2

      self._vehicle_id_to_trajectory[vehicle.id]['time'].append(time)
      self._vehicle_id_to_trajectory[vehicle.id]['speed'].append(speed)
      self._vehicle_id_to_trajectory[vehicle.id]['distance'].append(
          previous_distance + (time - previous_time) * average_speed)
      if previous_time == time:
        self._vehicle_id_to_trajectory[vehicle.id]['acceleration'].append(0)
      else:
        self._vehicle_id_to_trajectory[vehicle.id]['acceleration'].append(
            (speed - previous_speed) / (time - previous_time))
      self._vehicle_id_to_trajectory[vehicle.id]['angle'].append(
          float(vehicle.angle))
      self._vehicle_id_to_trajectory[vehicle.id]['x'].append(float(vehicle.x))
      self._vehicle_id_to_trajectory[vehicle.id]['y'].append(float(vehicle.y))

      edge = vehicle.lane.split('_')[0]
      if (not self._vehicle_id_to_hop_route[vehicle.id] or
          self._vehicle_id_to_hop_route[vehicle.id][-1] != edge):
        self._vehicle_id_to_hop_route[vehicle.id].append(edge)

  def get_vehicle_id_to_trajectory(self, fcd_file):
    self._parse_fcd_file_by_vehicle(fcd_file)
    return self._vehicle_id_to_trajectory

  def get_vehicle_id_to_hop_route(self, fcd_file):
    self._parse_fcd_file_by_vehicle(fcd_file)
    return self._vehicle_id_to_hop_route

  @memoize_util.MemoizeClassFunctionOneRun
  def _parse_fcd_file_by_edge(self, fcd_file, edge_id_list):
    """Projects the traffic to edges.

    Args:
      fcd_file: input simulation report fcd file.
      edge_id_list: if this is not specified, the function will collect
      information for all edges.
    """
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    for i, edge in enumerate(edge_id_list):
      if isinstance(edge, str):
        continue
      elif isinstance(edge, sumolib.net.edge.Edge):
        edge_id_list[i] = edge.getID()
      else:
        raise TypeError('Edge list can be either string, or Edge class.')
    for timestep, vehicle in sumolib.xml.parse_fast_nested(
        fcd_file, 'timestep', ['time'], 'vehicle', _DEFAULT_FCD_FILE_ENTRIES):
      edge_id = vehicle.lane.split('_')[0]
      if edge_id not in edge_id_list:
        continue
      self._edge_id_to_trajectory[edge_id]['time'].append(float(timestep.time))
      self._edge_id_to_trajectory[edge_id]['vehicle'].append(vehicle.id)
      self._edge_id_to_trajectory[edge_id]['speed'].append(float(vehicle.speed))
      self._edge_id_to_trajectory[edge_id]['angle'].append(float(vehicle.angle))
      self._edge_id_to_trajectory[edge_id]['pos'].append(float(vehicle.pos))
      self._edge_id_to_trajectory[edge_id]['x'].append(float(vehicle.x))
      self._edge_id_to_trajectory[edge_id]['y'].append(float(vehicle.y))

  def get_edge_id_to_trajectory(self, fcd_file, edge_id_list=None):
    self._parse_fcd_file_by_edge(fcd_file, edge_id_list)
    return self._edge_id_to_trajectory

  @classmethod
  def save_batch_edge_id_to_trajectory(cls,
                                       fcd_file,
                                       edge_id_list,
                                       time_range=None,
                                       time_segment_length=None,
                                       parse_time_step=10,
                                       output_folder=None):
    """Extracts and saves the information from the FCD file by edges.

    Args:
      fcd_file: input simulation report fcd file.
      edge_id_list: if this is not specified, the function will collect
      information for all edges.
      time_range: A time interval. If `time_range` is None, it will be set as
          [None, inf]. If the lower bound is set as None, it will be set as the
          smallest time point. It can also be set as something like [None, 3600]
          with specific upper bound, but None lower bound. So it will
          automatically find the smallest time point. If the upper bound is set
          as inf, then the file will be scanned to the end.
      time_segment_length: The step size of the time segment. By default it is
          infinity.
      parse_time_step: The code only reads time slot which is multiple of the
          `parse_time_step`.
      output_folder:
    """
    if isinstance(edge_id_list, str):
      edge_id_list = [edge_id_list]
    for i, edge in enumerate(edge_id_list):
      if isinstance(edge, str) or isinstance(edge, six.text_type):
        continue
      elif isinstance(edge, sumolib.net.edge.Edge):
        edge_id_list[i] = edge.getID()
      else:
        raise TypeError('Edge list can be either string, or Edge class.')

    if time_segment_length is None:
      time_segment_length = float('inf')
    # If `time_range` is set as None, it will automatically fit the range, and
    # the range is not known yet. The lower bound is acquired after getting into
    # the for loop.
    if time_range is None:
      time_range = [None, float('inf')]
    else:
      current_interval = [time_range[0], time_range[0] + time_segment_length]
    edge_id_to_trajectory = collections.defaultdict(
        lambda: collections.defaultdict(list))
    timestamp = 0.
    for timestep, vehicle in sumolib.xml.parse_fast_nested(
        fcd_file, 'timestep', ['time'], 'vehicle', _DEFAULT_FCD_FILE_ENTRIES):
      # Read time point which is only the multiple of `parse_time_step`.
      # If the read time point has not reached the `time_range`, skip.
      timestamp = float(timestep.time)
      if time_range[0] is None:  # Automatically adjust the lower bound.
        time_range[0] = timestamp
        current_interval = [timestamp, timestamp + time_segment_length]
      if (int(timestamp % parse_time_step) != 0 or timestamp < time_range[0]):
        continue
      # Quit after reaching the end of `time_range`.
      if timestamp > time_range[1]:
        break
      # Move on to a new time segment, then reset the `current_interval` and
      # `edge_id_to_trajectory` for the new segment.
      if timestamp > current_interval[1]:
        logging.info('Currently reading at time: %s', timestamp)
        # Save the current data.
        edge_id_to_trajectory['time_interval'] = current_interval
        file_name = ('edge_id_to_trajectory_%s_%s.pkl' %
                     (int(current_interval[0]), int(current_interval[1])))
        file_path = os.path.join(output_folder, file_name)
        logging.info('Saving file: %s', file_path)
        file_util.save_variable(file_path, edge_id_to_trajectory)
        # Set the new interval.
        new_interval_left_end = current_interval[0] +  time_segment_length
        if time_range[1] < current_interval[0] +  time_segment_length * 2:
          new_interval_right_end = time_range[1]
        else:
          new_interval_right_end = current_interval[0] + time_segment_length * 2
        current_interval = [new_interval_left_end, new_interval_right_end]
        logging.info('New interval: %s', current_interval)
        # Clear the `edge_id_to_trajectory` for the new time segment.
        edge_id_to_trajectory = collections.defaultdict(
            lambda: collections.defaultdict(list))
      # Load each time point data.
      edge_id = vehicle.lane.split('_')[0]
      if edge_id not in edge_id_list:
        continue
      edge_id_to_trajectory[edge_id]['time'].append(timestamp)
      edge_id_to_trajectory[edge_id]['vehicle'].append(vehicle.id)
      edge_id_to_trajectory[edge_id]['speed'].append(float(vehicle.speed))
      edge_id_to_trajectory[edge_id]['pos'].append(float(vehicle.pos))
      edge_id_to_trajectory[edge_id]['x'].append(float(vehicle.x))
      edge_id_to_trajectory[edge_id]['y'].append(float(vehicle.y))
      edge_id_to_trajectory[edge_id]['angle'].append(float(vehicle.angle))

    # Save the last interval's data.
    if edge_id_to_trajectory:
      current_interval[1] = timestamp  # The last time point is the upper bound.
      edge_id_to_trajectory['time_interval'] = current_interval
      file_name = ('edge_id_to_trajectory_%s_%s.pkl' %
                   (int(current_interval[0]), int(current_interval[1])))
      file_path = os.path.join(output_folder, file_name)
      logging.info('Saving file: %s', file_path)
      file_util.save_variable(file_path, edge_id_to_trajectory)
    else:
      logging.warning('Empty output variable.')

  @memoize_util.MemoizeClassFunctionOneRun
  # TODO(albertyuchen,yusef) Consider to be deprecated.
  def _parse_summary_file(self, summary_file):
    """Gets summary information."""
    for step in sumolib.xml.parse_fast(
        summary_file, 'step', _DEFAULT_SUMMARY_FILE_ENTRIES):
      for attribute in _DEFAULT_SUMMARY_FILE_ENTRIES:
        attribute_value = float(operator.attrgetter(attribute)(step))
        self._summary_attribute_to_time_series[attribute].append(
            attribute_value)

  def get_summary_attribute_to_time_series(self, summary_file):
    self._parse_summary_file(summary_file)
    return self._summary_attribute_to_time_series

  @classmethod
  def parse_summary_file(cls, summary_file):
    """Gets summary information."""
    summary_attribute_to_time_series = collections.defaultdict(list)
    for step in sumolib.xml.parse_fast(
        summary_file, 'step', _DEFAULT_SUMMARY_FILE_ENTRIES):
      for attribute in _DEFAULT_SUMMARY_FILE_ENTRIES:
        attribute_value = float(operator.attrgetter(attribute)(step))
        summary_attribute_to_time_series[attribute].append(
            attribute_value)
    return summary_attribute_to_time_series

  @memoize_util.MemoizeClassFunctionOneRun
  def _parse_route_file(self, route_file):
    """Gets traffic by edges."""
    for vehicle, route in sumolib.xml.parse_fast_nested(
        route_file, 'vehicle', _DEFAULT_ROUTE_FILE_VEHICLE_ENTRIES,
        'route', _DEFAULT_ROUTE_FILE_ROUTE_ENTRIES):
      edge_id_list = route.edges.split(' ')
      self._vehicle_id_to_full_route[vehicle.id] = edge_id_list
      for edge_id in edge_id_list:
        depart_time = float(vehicle.depart)
        arrival_time = float(vehicle.arrival)
        self._edge_id_to_attribute[edge_id]['depart'].append(depart_time)
        self._edge_id_to_attribute[edge_id]['arrival'].append(arrival_time)

  def get_vehicle_id_to_full_route(self, route_file):
    self._parse_route_file(route_file)
    return self._vehicle_id_to_full_route

  def get_edge_id_to_attribute(self, route_file):
    self._parse_route_file(route_file)
    return self._edge_id_to_attribute

  @memoize_util.MemoizeClassFunctionOneRun
  def _parse_tripinfo_file(self, tripinfo_file):
    """Gets tripinfo information."""
    for tripinfo in sumolib.xml.parse_fast(
        tripinfo_file, 'tripinfo', _DEFAULT_TRIPINFO_FILE_ENTRIES):
      for attribute in _DEFAULT_TRIPINFO_FILE_ENTRIES:
        attribute_item = operator.attrgetter(attribute)(tripinfo)
        # If the item is a number, then convert it to the number. Otherwise,
        # just keep it as a string.
        try:
          attribute_item = float(attribute_item)
        except ValueError:
          pass
        self._tripinfo_attribute_to_trips[attribute].append(attribute_item)

  def get_tripinfo_attribute_to_trips(self, tripinfo_file):
    self._parse_tripinfo_file(tripinfo_file)
    return self._tripinfo_attribute_to_trips

  @classmethod
  def get_and_save_detector_data(cls, detector_file, output_file=None):
    """Gets and saves detector recordings."""
    detector_data = collections.defaultdict(list)
    for interval in sumolib.xml.parse_fast(
        detector_file, 'interval', _DEFAULT_DETECTOR_FILE_ENTRIES):
      for attribute in _DEFAULT_DETECTOR_FILE_ENTRIES:
        attribute_item = operator.attrgetter(attribute)(interval)
        # If the item is a number, then convert it to the number. Otherwise,
        # just keep it as a string.
        try:
          attribute_item = float(attribute_item)
        except ValueError:
          pass
        detector_data[attribute].append(attribute_item)
    if output_file is not None:
      file_util.save_variable(output_file, detector_data)
