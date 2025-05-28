# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Tests for pipeline."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
import apache_beam as beam
import pandas as pd
import tensorflow as tf
from tensorflow.io import gfile

from smu import dataset_pb2
from smu import pipeline

TESTDATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'testdata')


class FunctionalTest(absltest.TestCase):

  def test_merge_duplicate_information_same_topology(self):
    main_mol = dataset_pb2.Molecule(mol_id=123000)
    main_mol.ini_geo.add()
    main_mol.ini_geo[0].atompos.add(x=1, y=2, z=3)

    dup_mol = dataset_pb2.Molecule(mol_id=123456, duplicate_of=123000)
    dup_mol.ini_geo.add()
    dup_mol.ini_geo[0].atompos.add(x=4, y=5, z=6)

    got = pipeline.merge_duplicate_information(123000, [dup_mol, main_mol])
    self.assertEqual(got.mol_id, 123000)
    self.assertEqual(got.duplicate_of, 0)
    self.assertEqual(got.duplicate_found, [123456])
    self.assertLen(got.ini_geo, 2)
    self.assertEqual(got.ini_geo[0].atompos[0].x, 1)
    self.assertEqual(got.ini_geo[1].atompos[0].x, 4)

  def test_merge_duplicate_information_diff_topology(self):
    main_mol = dataset_pb2.Molecule(mol_id=123000)
    main_mol.ini_geo.add()
    main_mol.ini_geo[0].atompos.add(x=1, y=2, z=3)

    dup_mol = dataset_pb2.Molecule(mol_id=456000, duplicate_of=123000)
    dup_mol.ini_geo.add()
    dup_mol.ini_geo[0].atompos.add(x=4, y=5, z=6)

    got = pipeline.merge_duplicate_information(123000, [dup_mol, main_mol])
    self.assertEqual(got.mol_id, 123000)
    self.assertEqual(got.duplicate_of, 0)
    self.assertEqual(got.duplicate_found, [456000])
    # TODO(pfr, ianwatson): implement correct copying of initial geometry
    self.assertLen(got.ini_geo, 1)
    self.assertEqual(got.ini_geo[0].atompos[0].x, 1)

  def test_extract_bond_lengths(self):
    # This molecule does not obey valence rules, but it's fine for this test.
    mol = dataset_pb2.Molecule(mol_id=123000)
    mol.prop.calc.status = 4
    bt = mol.bond_topo.add()
    bt.atom.extend([
        dataset_pb2.BondTopology.ATOM_ONEG, dataset_pb2.BondTopology.ATOM_NPOS,
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_H
    ])
    bt.bond.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    bt.bond.add(
        atom_a=0, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_DOUBLE)
    bt.bond.add(
        atom_a=0, atom_b=3, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    mol.opt_geo.atompos.add(x=0, y=0, z=0)
    mol.opt_geo.atompos.add(x=1, y=0, z=0)
    mol.opt_geo.atompos.add(x=0, y=2, z=0)
    mol.opt_geo.atompos.add(x=111, y=222, z=333)

    got = list(
        pipeline.extract_bond_lengths(mol, dist_sig_digits=2, unbonded_max=2.0))
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
    # This molecule does not obery valence rules, but it's fine for this test.
    mol = dataset_pb2.Molecule(mol_id=123000)
    mol.prop.calc.status = 4
    bt = mol.bond_topo.add()
    bt.atom.extend([
        dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_O
    ])
    bt.bond.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    bt.bond.add(
        atom_a=0, atom_b=2, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    mol.opt_geo.atompos.add(x=0, y=0, z=0)
    mol.opt_geo.atompos.add(x=1, y=0, z=0)
    mol.opt_geo.atompos.add(x=100, y=2, z=0)

    got = list(
        pipeline.extract_bond_lengths(mol, dist_sig_digits=2, unbonded_max=2.0))
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

  def _create_dummy_molecule(self):
    mol = dataset_pb2.Molecule(mol_id=123000)
    bt = mol.bond_topo.add()
    bt.atom.extend(
        [dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C])
    bt.bond.add(
        atom_a=0, atom_b=1, bond_type=dataset_pb2.BondTopology.BOND_SINGLE)
    mol.opt_geo.atompos.add(x=0, y=0, z=0)
    mol.opt_geo.atompos.add(x=1, y=0, z=0)
    return mol

  def test_extract_bond_lengths_has_errors(self):
    mol = self._create_dummy_molecule()
    mol.prop.calc.status = 8
    got = list(
        pipeline.extract_bond_lengths(mol, dist_sig_digits=2, unbonded_max=2.0))
    self.assertEqual([], got)

  def test_extract_bond_lengths_is_dup(self):
    mol = self._create_dummy_molecule()
    mol.prop.calc.status = 0
    mol.duplicate_of = 456000
    got = list(
        pipeline.extract_bond_lengths(mol, dist_sig_digits=2, unbonded_max=2.0))
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
    # Molecule 620517 will not match because bond lengths are not extracted
    # from molecules with serious errors like this.
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
      self.assertIn('mol_id,', conflicts_lines[0])
      self.assertEqual(
          conflicts_lines[1], '618451001,1,1,1,1,'
          '-406.51179,9.999999,-406.522079,9.999999,True,True,'
          '-406.51179,0.052254,-406.522079,2.5e-05,True,True\n')

    # Check a couple of the stats.
    with gfile.GFile(output_stem + '_stats-00000-of-00001.csv') as f:
      stats_lines = f.readlines()
      self.assertIn('errors.status,0,2\n', stats_lines)
      self.assertIn('errors.warn_t1,0,4\n', stats_lines)
      self.assertIn('fate,FATE_SUCCESS_ALL_WARNING_LOW,2\n', stats_lines)
      self.assertIn('fate,FATE_DUPLICATE_DIFFERENT_TOPOLOGY,1\n', stats_lines)
      self.assertIn('num_initial_geometries,1,4\n', stats_lines)
      self.assertIn('num_duplicates,1,1\n', stats_lines)
      self.assertIn('zero_field,spe_std_pbe0d3_6311gd,1\n', stats_lines)
      self.assertIn('bt_source,3,1\n', stats_lines)
      self.assertIn('bt_source,15,2\n', stats_lines)
      self.assertIn('num_topologies_csd,1,2\n', stats_lines)
      self.assertIn('num_topologies_mlcr,0,1\n', stats_lines)

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
      df_bt_summary = pd.read_csv(f, index_col='bt_id')
      # Check part of the header line
      self.assertIn('count_attempted_molecules', df_bt_summary.columns)
      # This is the bond topology that has no molecule
      self.assertEqual(df_bt_summary.loc[10, 'count_attempted_molecules'], 0)
      # This is a bond topology with 1 molecule
      self.assertEqual(df_bt_summary.loc[620517, 'count_attempted_molecules'],
                       1)
      self.assertEqual(
          df_bt_summary.loc[620517, 'count_calculation_with_error'], 1)
      # This is a bond topology with 2 molecules
      self.assertEqual(df_bt_summary.loc[618451, 'count_attempted_molecules'],
                       2)
      self.assertEqual(df_bt_summary.loc[618451, 'count_calculation_success'],
                       2)
      self.assertEqual(
          df_bt_summary.loc[618451, 'count_detected_match_mlcr_success'], 2)
      self.assertEqual(
          df_bt_summary.loc[618451, 'count_detected_match_csd_success'], 2)

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

    # Check that the generated TFRecord files contain some expected outputs
    standard_dataset = tf.data.TFRecordDataset(
        output_stem + '_standard_tfrecord-00000-of-00001')
    standard_output = [
        dataset_pb2.Molecule.FromString(raw)
        for raw in standard_dataset.as_numpy_iterator()
    ]
    self.assertCountEqual([c.mol_id for c in standard_output],
                          [618451001, 618451123])
    # Check that fields are filtered the way we expect
    self.assertFalse(standard_output[0].prop.HasField('compute_cluster_info'))
    self.assertFalse(standard_output[0].prop.HasField('orb_ehomo_pbe0_augpc1'))
    self.assertTrue(standard_output[0].prop.HasField('vib_freq'))

    complete_dataset = tf.data.TFRecordDataset(
        output_stem + '_complete_tfrecord-00000-of-00001')
    complete_output = [
        dataset_pb2.Molecule.FromString(raw)
        for raw in complete_dataset.as_numpy_iterator()
    ]
    self.assertCountEqual([c.mol_id for c in complete_output],
                          [618451001, 618451123, 620517002, 79593005])
    # Check that fields are filtered the way we expect
    # The DirectRunner randomizes the order of output so we need to make sure
    # that we get a full record.
    complete_entry = [c for c in complete_output if c.mol_id == 618451001][0]
    self.assertFalse(complete_entry.prop.HasField('compute_cluster_info'))
    self.assertTrue(complete_entry.prop.HasField('orb_ehomo_pbe0_augpc1'))
    self.assertTrue(complete_entry.prop.HasField('vib_freq'))

    complete_entry_for_smiles = [
        c for c in complete_output if c.mol_id == 620517002
    ][0]
    self.assertEqual(complete_entry_for_smiles.prop.smiles_openbabel,
                     'NotAValidSmilesString')



if __name__ == '__main__':
  absltest.main()
