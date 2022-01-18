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

"""Tests for visualization tools for SUMO related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from absl.testing import absltest
import numpy as np
import sumolib

from simulation_research.traffic import file_util
from simulation_research.traffic import map_visualizer
from simulation_research.traffic import simulation_data_parser


_TOLERANT_DIFFERENCE_RATIO = 0.05
_TESTDATA_DIR = './testdata'

# This simple map is composed of segments of freeways, arterial roads and
# some local roads. It can be viewed using NETEDIT by command,
# blaze-bin/third_party/sumo//netedit \
#     ${SUMO_PATH}/sumolib/testdata/mtv_tiny.net.xml
_MTV_MAP_FILE_NAME = 'mtv_tiny.net.xml'
_FCD_FILE_NAME = 'freeway_sparse.fcd.xml'
_SUMMARY_FILE_NAME = 'summary.xml'
_ROUTE_FILE_NAME = 'vehicle_route.xml'


def _load_file(folder, file_name):
  file_path = os.path.join(folder, file_name)
  return file_util.f_abspath(file_path)


class MapVisualizerTests(absltest.TestCase):

  def setUp(self):
    super(MapVisualizerTests, self).setUp()
    self._output_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self._fcd_file = _load_file(_TESTDATA_DIR, _FCD_FILE_NAME)
    self._summary_file = _load_file(_TESTDATA_DIR, _SUMMARY_FILE_NAME)
    self._route_file = _load_file(_TESTDATA_DIR, _ROUTE_FILE_NAME)
    mtv_map_file = _load_file(_TESTDATA_DIR, _MTV_MAP_FILE_NAME)
    net = sumolib.net.readNet(mtv_map_file)
    self._map_visualier = map_visualizer.MapVisualizer(net)
    self._data_parser = simulation_data_parser.SimulationDataParser()

  def test_histogram_along_edge(self):
    # Case 1: Basic case.
    values_on_edge = [10, 10, 2, 2]
    positions_on_edge = [2., 4., 6., 8.]
    number_bin = 2
    (actual_bin_mean,
     actual_bins_boundaries) = self._map_visualier._histogram_along_edge(
         values_on_edge, positions_on_edge, number_bin)
    expected_bin_mean = [10., 2.]
    expected_bins_boundaries = [2., 5., 8.]
    self.assertListEqual(list(actual_bin_mean), expected_bin_mean)
    self.assertListEqual(list(actual_bins_boundaries), expected_bins_boundaries)

    # Case 2: The results will yield numpy.nan values.
    number_bin = 6
    (actual_bin_mean,
     actual_bins_boundaries) = self._map_visualier._histogram_along_edge(
         values_on_edge, positions_on_edge, number_bin)
    expected_bin_mean = np.array([10., np.nan, 10., np.nan, 2., 2.])
    expected_bins_boundaries = [2., 3., 4., 5., 6., 7., 8.]
    none_nan_index = ~np.isnan(expected_bin_mean)
    # Checks whether non-nan values are correct.
    self.assertListEqual(list(actual_bin_mean[none_nan_index]),
                         list(expected_bin_mean[none_nan_index]))
    # Checks whether nan values are correct.
    self.assertTrue(all(list(np.isnan(actual_bin_mean[~none_nan_index]))))
    self.assertListEqual(list(actual_bins_boundaries), expected_bins_boundaries)

    # Case 3: Single value on multiple bins.
    values_on_edge = [12.65]
    positions_on_edge = [0.42]
    number_bin = 3
    (actual_bin_mean,
     actual_bins_boundaries) = self._map_visualier._histogram_along_edge(
         values_on_edge, positions_on_edge, number_bin)
    self.assertListEqual(actual_bin_mean, values_on_edge)
    self.assertListEqual(actual_bins_boundaries, positions_on_edge * 2)

    # Case 4: All positions are identical.
    values_on_edge = [12.65, 11.11, 12.12, 13.13]
    positions_on_edge = [0.42, 0.42, 0.42, 0.42]
    number_bin = 3
    (actual_bin_mean,
     actual_bins_boundaries) = self._map_visualier._histogram_along_edge(
         values_on_edge, positions_on_edge, number_bin)
    self.assertEqual(actual_bin_mean[0],
                     sum(values_on_edge) / len(values_on_edge))
    self.assertListEqual(actual_bins_boundaries, [0.42, 0.42])

  def test_plot_edges(self):
    highlight_edges = ['617235580', '685981552']
    output_file = os.path.join(self._output_dir, 'test_plot_edges.png')
    self._map_visualier.plot_edges([(highlight_edges, 'g', 0.5)],
                                   output_figure_path=output_file)

  def test_plot_vehicle_trajectory_on_map(self):
    test_vehicle_id = '700010432_to_706447588#1_3_0.0'
    output_file = os.path.join(self._output_dir,
                               test_vehicle_id+'_route_speed.png')
    self._map_visualier.plot_vehicle_trajectory_on_map(
        test_vehicle_id,
        self._data_parser.get_vehicle_id_to_trajectory(self._fcd_file),
        output_figure_path=output_file)

  def test_plot_edge_trajectory_on_map(self):
    test_edge_id = '27628577#0'
    output_file = os.path.join(self._output_dir,
                               test_edge_id+'_edge_trajectory.png')
    self._map_visualier.plot_edge_trajectory_on_map(
        test_edge_id,
        self._data_parser.get_edge_id_to_trajectory(self._fcd_file,
                                                    test_edge_id),
        output_figure_path=output_file)

  def test_plot_edge_trajectory_histogram(self):
    test_edge_id = '27628577#0'
    output_file = os.path.join(self._output_dir,
                               test_edge_id+'_edge_trajectory_histogram.png')
    self._map_visualier.plot_edge_trajectory_histogram(
        test_edge_id,
        self._data_parser.get_edge_id_to_trajectory(self._fcd_file,
                                                    test_edge_id),
        output_figure_path=output_file)

  def test_plot_edge_trajectory_histogram_on_map(self):
    test_edge_id = '27628577#0'
    output_file = os.path.join(
        self._output_dir, test_edge_id+'_edge_trajectory_histogram_on_map.png')
    self._map_visualier.plot_edge_trajectory_histogram_on_map(
        test_edge_id,
        self._data_parser.get_edge_id_to_trajectory(self._fcd_file,
                                                    test_edge_id),
        output_figure_path=output_file)

  def test_plot_summary(self):
    output_file = os.path.join(self._output_dir, 'summary.png')
    self._map_visualier.plot_summary(
        self._data_parser.get_summary_attribute_to_time_series(
            self._summary_file),
        output_figure_path=output_file)

  def test_plot_vehicle_counter_on_edge_histogram(self):
    test_edge_id = '27628577#0'
    output_file = os.path.join(
        self._output_dir,
        test_edge_id + 'vehicle_counter_on_edge_histogram.png')
    self._map_visualier.plot_vehicle_counter_on_edge_histogram(
        test_edge_id,
        self._data_parser.get_edge_id_to_attribute(self._route_file),
        output_figure_path=output_file)

  def test_plot_vehicle_counter_on_map(self):
    test_edge_id = '27628577#0'
    output_file = os.path.join(
        self._output_dir,
        test_edge_id + '_vehicle_counter_on_map.png')
    self._map_visualier.plot_vehicle_counter_on_map(
        test_edge_id,
        self._data_parser.get_edge_id_to_attribute(self._route_file),
        output_figure_path=output_file)

  def test_calculate_area_under_cummulative_curve(self):
    x = [1, 2, 3]
    y = [1, 2, 3]
    area = self._map_visualier._calculate_area_under_cummulative_curve(x, y)
    self.assertEqual(area, 4)

  def test_calculate_gap_area_between_cummulative_curves(self):
    x1 = [1, 2, 3]
    y1 = [1, 2, 3]
    x2 = [1, 2, 3]
    y2 = [2, 3, 4]
    gap_area = self._map_visualier.calculate_gap_area_between_cummulative_curves(
        x1, y1, x2, y2)
    self.assertEqual(gap_area, 2)


if __name__ == '__main__':
  absltest.main()
