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

"""Tests for pipeline."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
import apache_beam as beam
from tensorflow.io import gfile

from smu import dataset_pb2
from smu import pipeline


TESTDATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'testdata')


class FunctionalTest(absltest.TestCase):

  def test_merge_duplicate_information_same_topology(self):
    main_conf = dataset_pb2.Conformer(conformer_id=123000)
    main_conf.initial_geometries.add()
    main_conf.initial_geometries[0].atom_positions.add(x=1, y=2, z=3)

    dup_conf = dataset_pb2.Conformer(conformer_id=123456, duplicated_by=123000)
    dup_conf.initial_geometries.add()
    dup_conf.initial_geometries[0].atom_positions.add(x=4, y=5, z=6)

    got = pipeline.merge_duplicate_information(123000, [dup_conf, main_conf])
    self.assertEqual(got.conformer_id, 123000)
    self.assertEqual(got.duplicated_by, 0)
    self.assertEqual(got.duplicate_of, [123456])
    self.assertLen(got.initial_geometries, 2)
    self.assertEqual(got.initial_geometries[0].atom_positions[0].x, 1)
    self.assertEqual(got.initial_geometries[1].atom_positions[0].x, 4)

  def test_merge_duplicate_information_diff_topology(self):
    main_conf = dataset_pb2.Conformer(conformer_id=123000)
    main_conf.initial_geometries.add()
    main_conf.initial_geometries[0].atom_positions.add(x=1, y=2, z=3)

    dup_conf = dataset_pb2.Conformer(conformer_id=456000, duplicated_by=123000)
    dup_conf.initial_geometries.add()
    dup_conf.initial_geometries[0].atom_positions.add(x=4, y=5, z=6)

    got = pipeline.merge_duplicate_information(123000, [dup_conf, main_conf])
    self.assertEqual(got.conformer_id, 123000)
    self.assertEqual(got.duplicated_by, 0)
    self.assertEqual(got.duplicate_of, [456000])
    # TODO(pfr, ianwatson): implement correct copying of initial geometry
    self.assertLen(got.initial_geometries, 1)
    self.assertEqual(got.initial_geometries[0].atom_positions[0].x, 1)


class IntegrationTest(absltest.TestCase):

  def test_whole_pipeline(self):
    test_subdirectory = self.create_tempdir()
    output_stem = os.path.join(test_subdirectory, 'testout')
    input_stage1_dat_glob = os.path.join(TESTDATA_PATH,
                                         'pipeline_input_stage1.dat')
    input_stage2_dat_glob = os.path.join(TESTDATA_PATH,
                                         'pipeline_input_stage2.dat')
    input_equivalent_glob = os.path.join(TESTDATA_PATH,
                                         'pipeline_equivalent.dat')
    input_bond_topology_csv = os.path.join(TESTDATA_PATH,
                                           'pipeline_bond_topology.csv')
    with flagsaver.flagsaver(
        input_stage1_dat_glob=input_stage1_dat_glob,
        input_stage2_dat_glob=input_stage2_dat_glob,
        input_equivalent_glob=input_equivalent_glob,
        input_bond_topology_csv=input_bond_topology_csv,
        output_stem=output_stem,
        output_shards=1):
      # If you have custom beam options, add them here.
      beam_options = None
      with beam.Pipeline(beam_options) as root:
        pipeline.pipeline(root)

    logging.info('Files in output: %s',
                 '\n'.join(gfile.glob(os.path.join(test_subdirectory, '/*'))))
    for stage in ['stage1', 'stage2']:
      self.assertTrue(
          gfile.exists(output_stem + '_' + stage +
                       '_original_known_error-00000-of-00001.dat'))
      self.assertTrue(
          gfile.exists(output_stem + '_' + stage +
                       '_original_unknown_error-00000-of-00001.dat'))
      self.assertTrue(
          gfile.exists(output_stem + '_' + stage +
                       '_mismatched_original-00000-of-00001.dat'))
      self.assertTrue(
          gfile.exists(output_stem + '_' + stage +
                       '_mismatched_regen-00000-of-00001.dat'))

    # Check the merge conflicts file
    with gfile.GFile(output_stem + '_conflicts-00000-of-00001.csv') as f:
      conflicts_lines = f.readlines()
      self.assertIn('conformer_id,', conflicts_lines[0])
      self.assertEqual(
          conflicts_lines[1],
          '618451001,'
          '1,1,1,1,-406.51179,9.999999,-406.522079,9.999999,True,True,'
          '1,1,1,1,-406.51179,0.052254,-406.522079,2.5e-05,True,True\n')

    # Check a couple of the stats.
    with gfile.GFile(output_stem + '_stats-00000-of-00001.csv') as f:
      stats_lines = f.readlines()
      self.assertIn('error_nsvg09,0,4\n', stats_lines)
      self.assertIn('fate,FATE_SUCCESS,2\n', stats_lines)
      self.assertIn('fate,FATE_DUPLICATE_DIFFERENT_TOPOLOGY,1\n', stats_lines)
      self.assertIn('num_initial_geometries,1,4\n', stats_lines)
      self.assertIn('num_duplicates,1,1\n', stats_lines)

    # Check the smiles comparison output
    with gfile.GFile(output_stem + '_smiles_compare-00000-of-00001.csv') as f:
      smiles_lines = f.readlines()
      self.assertIn(
          '620517002,MISMATCH,NotAValidSmilesString,'
          '[H]C1=C2OC2=C(F)O1,FC1=C2OC2=CO1\n', smiles_lines)
      # Make sure that a bond topology with a matching smiles doesn't show
      for line in smiles_lines:
        self.assertNotIn('618451001', line)

    # Check the bond topology summary
    with gfile.GFile(output_stem + '_bt_summary-00000-of-00001.csv') as f:
      bt_summary_lines = f.readlines()
      # Check part of the header line
      self.assertIn('bt_id', bt_summary_lines[0])
      self.assertIn('count_attempted_conformers', bt_summary_lines[0])
      # This is the bond topology that has no conformer
      self.assertIn('10,0,0,0,0,0,0,0,0,0,0\n', bt_summary_lines)
      # This is a bond topology with 1 conformer
      self.assertIn('620517,1,0,0,0,1,0,1,0,0,0\n', bt_summary_lines)
      # This is a bond topology with 2 conformers
      self.assertIn('618451,2,0,0,0,2,0,0,2,0,0\n', bt_summary_lines)

    # For the gzip files below, we check >100 because even an empty gzip file
    # has non-zero length. 100 is kind of arbitrary to be bigger than the
    # expected header of 20.
    self.assertGreater(
        gfile.stat(output_stem +
                   '_complete_json-00000-of-00003.json.gz').length +
        gfile.stat(output_stem +
                   '_complete_json-00001-of-00003.json.gz').length +
        gfile.stat(output_stem +
                   '_complete_json-00002-of-00003.json.gz').length,
        100)
    self.assertGreater(
        gfile.stat(output_stem +
                   '_standard_json-00000-of-00001.json.gz').length, 100)



if __name__ == '__main__':
  absltest.main()
