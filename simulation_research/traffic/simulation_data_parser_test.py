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

"""Tests for simulation_data_parser.

FCD file: The file looks like the following. It shows the position, speed, lane
    and other metrics of each vehicle. The whole file is order by simulation
    timestamp. This is the output of the sumo or sumo-gui.
<timestep time="0.00">
  <vehicle id="27628577#0_to_392017177#1_1_0.0" x="4660.86" y="966.64" .../>
  <vehicle id="367132267#0_to_392017177#1_2_0.0" x="3172.05" y="28.02" .../>
  <vehicle id="700010432_to_5018655_4_0.0" x="0.88" y="3833.04" .../>
  <vehicle id="700010432_to_706447588#1_3_0.0" x="-1.55" y="3830.96".../>
</timestep>
Vehicle route file: This file list the routes of all vehicles during the
    simulation. The route is composed of a list of edges. This is the output of
    the sumo or sumo-gui.
<vehicle id="394369030_to_706447588#1_27_0.0" depart="0.00" departLane="0" ...>
  <route cost="-1.00" savings="0.00" edges="394369030 ..."/>
</vehicle>
Summary file: This file summarize the simulation at every time stamp. The items
    include total number of running vehicles, mean speed, waiting time and etc.
    This is the output of the sumo or sumo-gui.
<step time="0.00" loaded="26" inserted="12" running="12" waiting="14" .../>
Trip information file: This file list all vehicles entered the simulations. Each
    item includes the vehicle id, depart time, depart lane and etc. This is the
    output of the sumo or sumo-gui.
<tripinfo id="flow_2.10" depart="67.00" departLane="gneE7_0" .../>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import absltest

from simulation_research.traffic import file_util
from simulation_research.traffic import simulation_data_parser as parser

FLAGS = flags.FLAGS


def _load_file(folder, file_name):
  file_path = os.path.join(folder, file_name)
  return file_util.f_abspath(file_path)


class SimulationDataParserTest(absltest.TestCase):

  def setUp(self):
    super(SimulationDataParserTest, self).setUp()
    local_testdir = './teatdata'
    self._testdata_dir = os.path.join(
        FLAGS.test_srcdir, local_testdir)
    self._output_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self._data_parser = parser.SimulationDataParser()

  def test_parse_fcd_file_by_vehicle(self):
    fcd_file = 'freeway_sparse.fcd.xml'
    fcd_file_path = _load_file(self._testdata_dir, fcd_file)
    vehicle_to_trajectory = self._data_parser.get_vehicle_id_to_trajectory(
        fcd_file_path)
    vehicle_to_hop_route = self._data_parser.get_vehicle_id_to_hop_route(
        fcd_file_path)
    test_vehicle_id = ['27628577#0_to_392017177#1_1_0.0',
                       '367132267#0_to_392017177#1_2_0.1',
                       '700010432_to_706447588#1_3_0.6']
    for vehecle_id in test_vehicle_id:
      self.assertIn(vehecle_id, vehicle_to_trajectory)
      self.assertIn(vehecle_id, vehicle_to_hop_route)

  def test_parse_fcd_file_by_edge(self):
    fcd_file = 'freeway_sparse.fcd.xml'
    fcd_file_path = _load_file(self._testdata_dir, fcd_file)
    test_edge_id = '700010432'
    time_point_length = 68
    edge_to_trajectory = self._data_parser.get_edge_id_to_trajectory(
        fcd_file_path, test_edge_id)
    self.assertLen(edge_to_trajectory[test_edge_id]['time'], time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['vehicle'],
                   time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['speed'], time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['angle'], time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['pos'], time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['x'], time_point_length)
    self.assertLen(edge_to_trajectory[test_edge_id]['y'], time_point_length)

  def test_parse_summary_file(self):
    summary_file = 'summary.xml'
    summary_file_path = _load_file(self._testdata_dir, summary_file)
    time_series = self._data_parser.get_summary_attribute_to_time_series(
        summary_file_path)
    time_point_length = 12
    self.assertLen(time_series['time'], time_point_length)
    self.assertLen(time_series['loaded'], time_point_length)
    self.assertLen(time_series['running'], time_point_length)
    self.assertLen(time_series['meanSpeed'], time_point_length)
    self.assertLen(time_series['duration'], time_point_length)

  def test_parse_summary_file_classmethod(self):
    # TODO(albertyuchen,yusef) Consider to be deprecated.
    summary_file = 'summary.xml'
    summary_file_path = _load_file(self._testdata_dir, summary_file)
    time_series = self._data_parser.parse_summary_file(summary_file_path)
    time_point_length = 12
    self.assertLen(time_series['time'], time_point_length)
    self.assertLen(time_series['loaded'], time_point_length)
    self.assertLen(time_series['running'], time_point_length)
    self.assertLen(time_series['meanSpeed'], time_point_length)
    self.assertLen(time_series['duration'], time_point_length)

  def test_parse_route_file(self):
    route_file = 'vehicle_route.xml'
    route_file_path = _load_file(self._testdata_dir, route_file)
    vehicle_to_full_route = self._data_parser.get_vehicle_id_to_full_route(
        route_file_path)
    edge_to_attribute = self._data_parser.get_edge_id_to_attribute(
        route_file_path)
    test_vehicle_id = '27628577#0_to_392017177#1_1_0.4'
    test_edge_id = '515733343'
    self.assertLen(vehicle_to_full_route[test_vehicle_id], 18)
    self.assertLen(edge_to_attribute[test_edge_id]['depart'], 8)
    self.assertLen(edge_to_attribute[test_edge_id]['arrival'], 8)

  def test_parse_tripinfo_file(self):
    tripinfo_file = 'tripinfo.xml'
    tripinfo_file_path = _load_file(self._testdata_dir, tripinfo_file)
    tripinfo_data = self._data_parser.get_tripinfo_attribute_to_trips(
        tripinfo_file_path)
    data_length = 15
    self.assertLen(tripinfo_data['id'], data_length)
    self.assertIn('flow_1.0', tripinfo_data['id'])
    self.assertLen(tripinfo_data['depart'], data_length)
    self.assertLen(tripinfo_data['departLane'], data_length)
    self.assertLen(tripinfo_data['departSpeed'], data_length)
    self.assertLen(tripinfo_data['duration'], data_length)

  def test_save_batch_edge_id_to_trajectory(self):
    # Case 1: Extract and save the data into batches.
    fcd_file = 'freeway_sparse.fcd.xml'
    fcd_file_path = _load_file(self._testdata_dir, fcd_file)
    time_segment_length = 3
    time_range = [0, 10]
    test_edges = ['27628577#0', '367132267#0', '700010432']
    self._data_parser.save_batch_edge_id_to_trajectory(
        fcd_file_path,
        test_edges,
        time_range=time_range,
        time_segment_length=time_segment_length,
        parse_time_step=1,
        output_folder=self._output_dir)

    # There are 3 output files.
    actual_output_1 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_0_3.pkl')
    actual_output_2 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_3_6.pkl')
    actual_output_3 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_6_9.pkl')
    self.assertTrue(file_util.f_exists(actual_output_1))
    self.assertTrue(file_util.f_exists(actual_output_2))
    self.assertTrue(file_util.f_exists(actual_output_3))

    actual_dictionary = file_util.load_variable(actual_output_1)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    actual_dictionary = file_util.load_variable(actual_output_2)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                          6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    actual_dictionary = file_util.load_variable(actual_output_3)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0,
                          8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0,
                          9.0, 9.0, 9.0, 9.0, 9.0])

    # Case 2: Extract and save the file as a single batch.
    time_segment_length = None
    time_range = [0, 20]
    test_edges = ['27628577#0', '367132267#0', '700010432']
    self._data_parser.save_batch_edge_id_to_trajectory(
        fcd_file_path, test_edges,
        time_range=time_range, time_segment_length=time_segment_length,
        parse_time_step=1, output_folder=self._output_dir)

    # There is 1 output file.
    actual_output = os.path.join(self._output_dir,
                                 'edge_id_to_trajectory_0_9.pkl')
    self.assertTrue(file_util.f_exists(actual_output))
    actual_dictionary = file_util.load_variable(actual_output)
    self.assertListEqual(
        actual_dictionary['27628577#0']['time'],
        [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
         4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
         7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
         8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

    # Case 3: Extract and save the file as a single batch without specifying the
    # `time_range`. By default, it will use the smallest and largest time point
    # as its lower and upper bound.
    time_segment_length = None
    time_range = None
    test_edges = ['27628577#0', '367132267#0', '700010432']
    self._data_parser.save_batch_edge_id_to_trajectory(
        fcd_file_path, test_edges,
        time_range=time_range, time_segment_length=time_segment_length,
        parse_time_step=1, output_folder=self._output_dir)

    # There is 1 output file.
    actual_output = os.path.join(self._output_dir,
                                 'edge_id_to_trajectory_0_9.pkl')
    self.assertTrue(file_util.f_exists(actual_output))
    actual_dictionary = file_util.load_variable(actual_output)
    self.assertListEqual(
        actual_dictionary['27628577#0']['time'],
        [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
         4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
         7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0,
         8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

    # Case 4: Extract and save the file as a single batch without specifying the
    # `time_range` but with the `time_segment_length`. By default, it will use
    # the smallest and largest time point as its lower and upper bound
    # respectively.
    time_segment_length = 4
    time_range = None
    test_edges = ['27628577#0', '367132267#0', '700010432']
    self._data_parser.save_batch_edge_id_to_trajectory(
        fcd_file_path, test_edges,
        time_range=time_range, time_segment_length=time_segment_length,
        parse_time_step=1, output_folder=self._output_dir)

    # There are 3 output files.
    actual_output_1 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_0_4.pkl')
    actual_output_2 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_4_8.pkl')
    actual_output_3 = os.path.join(self._output_dir,
                                   'edge_id_to_trajectory_8_9.pkl')
    self.assertTrue(file_util.f_exists(actual_output_1))
    self.assertTrue(file_util.f_exists(actual_output_2))
    self.assertTrue(file_util.f_exists(actual_output_3))
    actual_dictionary = file_util.load_variable(actual_output_1)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0,
                          4.0, 4.0, 4.0, 4.0])
    actual_dictionary = file_util.load_variable(actual_output_2)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                          6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0,
                          8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    actual_dictionary = file_util.load_variable(actual_output_3)
    self.assertListEqual(actual_dictionary['27628577#0']['time'],
                         [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])

  def test_get_save_detector_data(self):
    detector_file = 'detector.xml'
    output_file = os.path.join(self._output_dir, 'detector.pkl')
    detector_file_path = _load_file(self._testdata_dir, detector_file)
    self._data_parser.get_and_save_detector_data(
        detector_file_path, output_file)

    actual_dictionary = file_util.load_variable(output_file)
    print(actual_dictionary)
    self.assertListEqual(actual_dictionary['nVehEntered'],
                         [0.0, 0.0, 0.0, 12.0, 7.0, 14.0, 11.0])
    self.assertListEqual(actual_dictionary['speed'],
                         [-1.0, -1.0, -1.0, 22.2, 22.99, 19.39, 22.68])
    self.assertListEqual(actual_dictionary['id'],
                         ['e1Detector_10293408#4_0_2'] * 7)


if __name__ == '__main__':
  absltest.main()
