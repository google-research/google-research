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
import tensorflow as tf
from tensorflow.io import gfile

from smu import dataset_pb2
from smu import pipeline

TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


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

  def test_extract_bond_lengths(self):
    # This conformer does not obey valence rules, but it's fine for this test.
    conf = dataset_pb2.Conformer(conformer_id=123000)
    conf.properties.errors.status = 4
    bt = conf.bond_topologies.add()
    bt.atoms.extend([
        dataset_pb2.BondTopology.ATOM_ONEG, dataset_pb2.BondTopology.ATOM_NPOS,
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_H
    ])
    bt.bonds.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    bt.bonds.add(
        atom_a=0, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_DOUBLE)
    bt.bonds.add(
        atom_a=0, atom_b=3, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    conf.optimized_geometry.atom_positions.add(x=0, y=0, z=0)
    conf.optimized_geometry.atom_positions.add(x=1, y=0, z=0)
    conf.optimized_geometry.atom_positions.add(x=0, y=2, z=0)
    conf.optimized_geometry.atom_positions.add(x=111, y=222, z=333)

    got = list(
        pipeline.extract_bond_lengths(
            conf, dist_sig_digits=2, unbonded_max=2.0))
    # Note that these are *not* rounded, but truncated to this many digits.
    self.assertEqual(
        got,
        [
            # 1 bohr -> 0.529177249 angstroms
            ('n', 'o', dataset_pb2.BondTopology.BOND_SINGLE, '0.52'),
            # 2 bohr -> 2 * 0.529177249 angstroms
            ('c', 'o', dataset_pb2.BondTopology.BOND_DOUBLE, '1.05'),
            # sqrt(1**2 + 2**2) bohr -> 2.23606 * 0.529177249 angstroms
            ('c', 'n', dataset_pb2.BondTopology.BOND_UNDEFINED, '1.18')
        ])

  def test_extract_bond_lengths_max_unbonded(self):
    # This conformer does not obery valence rules, but it's fine for this test.
    conf = dataset_pb2.Conformer(conformer_id=123000)
    conf.properties.errors.status = 4
    bt = conf.bond_topologies.add()
    bt.atoms.extend([
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_O
    ])
    bt.bonds.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    bt.bonds.add(
        atom_a=0, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    conf.optimized_geometry.atom_positions.add(x=0, y=0, z=0)
    conf.optimized_geometry.atom_positions.add(x=1, y=0, z=0)
    conf.optimized_geometry.atom_positions.add(x=100, y=2, z=0)

    got = list(
        pipeline.extract_bond_lengths(
            conf, dist_sig_digits=2, unbonded_max=2.0))
    # Note that these are *not* rounded, but truncated to this many digits.
    self.assertEqual(
        got,
        [
            # 1 bohr -> 0.529177249 angstroms
            ('c', 'n', dataset_pb2.BondTopology.BOND_SINGLE, '0.52'),
            # It seems like this should be 52.91 but it looks like some
            # numerical noise in np.linalg.norm.
            ('c', 'o', dataset_pb2.BondTopology.BOND_SINGLE, '52.92')
        ])
    # Note that the N-O distance is not reported while the C-O is.

  def _create_dummy_conformer(self):
    conf = dataset_pb2.Conformer(conformer_id=123000)
    bt = conf.bond_topologies.add()
    bt.atoms.extend(
        [dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C])
    bt.bonds.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    conf.optimized_geometry.atom_positions.add(x=0, y=0, z=0)
    conf.optimized_geometry.atom_positions.add(x=1, y=0, z=0)
    return conf

  def test_extract_bond_lengths_has_errors(self):
    conf = self._create_dummy_conformer()
    conf.properties.errors.status = 8
    got = list(
        pipeline.extract_bond_lengths(
            conf, dist_sig_digits=2, unbonded_max=2.0))
    self.assertEqual([], got)

  def test_extract_bond_lengths_is_dup(self):
    conf = self._create_dummy_conformer()
    conf.properties.errors.status = 0
    conf.duplicated_by = 456000
    got = list(
        pipeline.extract_bond_lengths(
            conf, dist_sig_digits=2, unbonded_max=2.0))
    self.assertEqual([], got)


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

    metrics = root.result.metrics().query()
    counters_dict = {
        m.key.metric.name: m.committed for m in metrics['counters']
    }

    self.assertEqual(counters_dict['attempted_topology_matches'], 3)
    # Conformer 620517 will not match because bond lengths are not extracted
    # from conformers with serious errors like this.
    self.assertEqual(counters_dict['no_topology_matches'], 1)
    self.assertNotIn('topology_match_smiles_failure', counters_dict)

    logging.info('Files in output: %s',
                 '\n'.join(gfile.glob(os.path.join(test_subdirectory, '*'))))
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
          conflicts_lines[1], '618451001,1,1,1,1,'
          '-406.51179,9.999999,-406.522079,9.999999,True,True,'
          '-406.51179,0.052254,-406.522079,2.5e-05,True,True\n')

    # Check a couple of the stats.
    with gfile.GFile(output_stem + '_stats-00000-of-00001.csv') as f:
      stats_lines = f.readlines()
      self.assertIn('errors.status,0,2\n', stats_lines)
      self.assertIn('errors.warn_t1,0,4\n', stats_lines)
      self.assertIn('fate,FATE_SUCCESS,2\n', stats_lines)
      self.assertIn('fate,FATE_DUPLICATE_DIFFERENT_TOPOLOGY,1\n', stats_lines)
      self.assertIn('num_initial_geometries,1,4\n', stats_lines)
      self.assertIn('num_duplicates,1,1\n', stats_lines)
      self.assertIn('zero_field,single_point_energy_pbe0d3_6_311gd,1\n',
                    stats_lines)

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
      self.assertIn('10,0,0,0,0,0,0,0,0,0,0,0,0\n', bt_summary_lines)
      # This is a bond topology with 1 conformer
      self.assertIn('620517,1,0,0,0,1,0,1,0,0,0,0,0\n', bt_summary_lines)
      # This is a bond topology with 2 conformers
      self.assertIn('618451,2,0,0,0,2,0,0,0,2,0,0,0\n', bt_summary_lines)

    # Check the bond lengths file
    with gfile.GFile(output_stem + '_bond_lengths.csv') as f:
      bond_length_lines = f.readlines()
      self.assertEqual('atom_char_0,atom_char_1,bond_type,length_str,count\n',
                       bond_length_lines[0])
      self.assertIn('c,c,2,1.336,1\n', bond_length_lines)
      self.assertIn('c,o,1,1.422,2\n', bond_length_lines)

    # For the gzip files below, we check >100 because even an empty gzip file
    # has non-zero length. 100 is kind of arbitrary to be bigger than the
    # expected header of 20.
    self.assertGreater(
        gfile.stat(output_stem + '_complete_json-00000-of-00003.json.gz').length
        +
        gfile.stat(output_stem + '_complete_json-00001-of-00003.json.gz').length
        + gfile.stat(output_stem +
                     '_complete_json-00002-of-00003.json.gz').length, 100)
    self.assertGreater(
        gfile.stat(output_stem +
                   '_standard_json-00000-of-00001.json.gz').length, 100)

    # Check that the generated TFRecord files contain some expected outputs
    standard_dataset = tf.data.TFRecordDataset(
        output_stem + '_standard_tfrecord-00000-of-00001')
    standard_output = [
        dataset_pb2.Conformer.FromString(raw)
        for raw in standard_dataset.as_numpy_iterator()
    ]
    self.assertCountEqual([c.conformer_id for c in standard_output],
                          [618451001, 618451123])
    # Check that fields are filtered the way we expect
    self.assertFalse(
        standard_output[0].properties.HasField('compute_cluster_info'))
    self.assertFalse(
        standard_output[0].properties.HasField('homo_pbe0_aug_pc_1'))
    self.assertTrue(
        standard_output[0].properties.HasField('rotational_constants'))

    complete_dataset = tf.data.TFRecordDataset(
        output_stem + '_complete_tfrecord-00000-of-00001')
    complete_output = [
        dataset_pb2.Conformer.FromString(raw)
        for raw in complete_dataset.as_numpy_iterator()
    ]
    self.assertCountEqual([c.conformer_id for c in complete_output],
                          [618451001, 618451123, 620517002, 79593005])
    # Check that fields are filtered the way we expect
    # The DirectRunner randomizes the order of output so we need to make sure
    # that we get a full record.
    complete_entry = [
        c for c in complete_output if c.conformer_id == 618451001
    ][0]
    self.assertFalse(complete_entry.properties.HasField('compute_cluster_info'))
    self.assertTrue(complete_entry.properties.HasField('homo_pbe0_aug_pc_1'))
    self.assertTrue(complete_entry.properties.HasField('rotational_constants'))

    complete_entry_for_smiles = [
        c for c in complete_output if c.conformer_id == 620517002
    ][0]
    self.assertEqual(complete_entry_for_smiles.properties.smiles_openbabel,
                     'NotAValidSmilesString')



if __name__ == '__main__':
  absltest.main()
