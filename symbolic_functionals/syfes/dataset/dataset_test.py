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

"""Tests for dataset.dataset."""

import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from symbolic_functionals.syfes.dataset import dataset
from symbolic_functionals.syfes.scf import scf

_TESTDATA_DIR = 'testdata/'



def _make_random_ks_info_for_mol(test_dir, mol_name):
  mol_dir = os.path.join(test_dir, mol_name)
  if not tf.io.gfile.exists(mol_dir):
    tf.io.gfile.makedirs(mol_dir)
  rho, weights = np.random.rand(50), np.random.rand(50)
  energies = np.random.rand(len(scf.SCF_SCALAR_RESULTS))
  with tf.io.gfile.GFile(os.path.join(mol_dir, 'rho.npy'), 'wb') as f:
    np.save(f, rho)
  with tf.io.gfile.GFile(os.path.join(mol_dir, 'weights.npy'), 'wb') as f:
    np.save(f, weights)
  with tf.io.gfile.GFile(os.path.join(mol_dir, 'energies.npy'), 'wb') as f:
    np.save(f, energies)
  return rho, weights, energies


class DatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset = dataset.Dataset.from_mgcdb84_dataset(
        _TESTDATA_DIR, complete_mgcdb84=False)
    self.test_dir = tempfile.mkdtemp(dir=flags.FLAGS.test_tmpdir)

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_scf_scalar_results_consistency(self):
    self.assertEqual(dataset.SCF_SCALAR_RESULTS, scf.SCF_SCALAR_RESULTS)

  def test_nrow_of_dataset(self):
    self.assertEqual(self.dataset.nrow, (4, 7))
    self.assertEqual(self.dataset.nrow_property, 4)
    self.assertEqual(self.dataset.nrow_dft, 7)
    self.assertLen(self.dataset.dataset_names, 3)

  def test_verify_dataset_integrity_with_wrong_nrow(self):
    with self.assertRaisesRegex(
        ValueError, 'Number of rows for property_df and dft_df '
        r'\(4, 7\) are not equal to the expected values \(5, 8\)'):
      self.dataset.verify_dataset_integrity(
          expected_nrow=(self.dataset.nrow_property + 1,
                         self.dataset.nrow_dft + 1),
          expected_num_datasets=len(self.dataset.dataset_names))

  def test_verify_dataset_integrity_with_wrong_num_datasets(self):
    with self.assertRaisesRegex(
        ValueError,
        'Number of datasets 3 is not equal to the expected value 4.'):
      self.dataset.verify_dataset_integrity(
          expected_nrow=self.dataset.nrow,
          expected_num_datasets=len(self.dataset.dataset_names) + 1)

  def test_verify_dataset_integrity_with_inconsistent_dataframes(self):
    self.dataset.dft_df = self.dataset.dft_df.append(pd.Series(name='mol'))

    with self.assertRaisesRegex(
        ValueError, 'dft_df contains rows not used by property_df.'):
      self.dataset.verify_dataset_integrity(
          expected_nrow=(4, 8), expected_num_datasets=3)

  def test_compute_formula_matrix(self):
    np.testing.assert_equal(
        self.dataset.formula_matrix,
        [[1., -1., -1., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 1., -1.]])

  def test_compute_formula_matrix_multiplication(self):
    num_single_points = (
        np.abs(self.dataset.formula_matrix) @ np.ones(self.dataset.nrow_dft))
    np.testing.assert_equal(num_single_points,
                            self.dataset.property_df['num_single_points'])

  def test_filter_dft_df_with_property_df(self):
    dft_df_long = self.dataset.dft_df.append(pd.Series(name='mol'))

    dft_df_filtered = dataset.Dataset.filter_dft_df_with_property_df(
        self.dataset.property_df, dft_df_long)

    pd.testing.assert_frame_equal(
        dft_df_filtered, self.dataset.dft_df, check_dtype=False)

  def test_get_subset(self):
    property_df_subset = self.dataset.property_df[
        self.dataset.property_df['spin_singlet']]

    subset = self.dataset.get_subset(property_df_subset)

    self.assertEqual(subset.nrow, (3, 6))

  def test_get_subset_without_arguments(self):
    self.assertEqual(self.dataset.get_subset(), self.dataset)

  def test_get_subset_only_keep_properties_with_ks_info(self):
    _make_random_ks_info_for_mol(self.test_dir, '11_H_AE18')
    self.dataset.load_ks_info(ks_info_directory=self.test_dir)
    subset = self.dataset.get_subset(
        only_keep_properties_with_ks_info=True)

    pd.testing.assert_frame_equal(
        subset.property_df, self.dataset.property_df.iloc[[1]])

  def test_get_subset_only_keep_properties_with_ks_info_empty(self):
    subset = self.dataset.get_subset(
        only_keep_properties_with_ks_info=True)

    self.assertEmpty(subset.property_df)

  def test_get_subset_with_wrong_property_df_subset(self):
    property_df_subset = self.dataset.property_df.append(pd.Series(name='mol'))

    with self.assertRaisesRegex(
        ValueError, 'Input dataframe is not a subset of property_df.'):
      self.dataset.get_subset(property_df_subset)

  def test_save_and_load_dataset(self):
    self.dataset.save(save_directory=self.test_dir)
    loaded_dataset = dataset.Dataset.load(save_directory=self.test_dir)

    self.assertEqual(loaded_dataset, self.dataset)

  def test_load_ks_info(self):
    test_dataset = self.dataset.get_subset(
        self.dataset.property_df.loc[['AE18_1', 'AE18_2']])
    rho, weights, energies = _make_random_ks_info_for_mol(
        self.test_dir, '11_H_AE18')

    test_dataset.load_ks_info(ks_info_directory=self.test_dir)

    np.testing.assert_allclose(test_dataset.ks_info['11_H_AE18']['rho'], rho)
    np.testing.assert_allclose(test_dataset.ks_info['11_H_AE18']['weights'],
                               weights)
    np.testing.assert_allclose(
        test_dataset.dft_df.loc['11_H_AE18',
                                scf.SCF_SCALAR_RESULTS].to_numpy(dtype=float),
        energies)

  def test_load_ks_info_energies_only(self):
    test_dataset = self.dataset.get_subset(
        self.dataset.property_df.loc[['AE18_1', 'AE18_2']])
    energies = _make_random_ks_info_for_mol(self.test_dir, '11_H_AE18')[2]

    test_dataset.load_ks_info(
        ks_info_directory=self.test_dir, load_energies_only=True)

    self.assertEmpty(self.dataset.ks_info)
    np.testing.assert_allclose(
        test_dataset.dft_df.loc['11_H_AE18',
                                scf.SCF_SCALAR_RESULTS].to_numpy(dtype=float),
        energies)

  def test_eval_property_cols_from_dft_df(self):
    self.dataset.dft_df['mol_name_length'] = self.dataset.dft_df.apply(
        lambda row: len(row.name), axis=1)
    expected_mol_name_length_in_property_df = self.dataset.property_df.apply(
        lambda row: sum(weight * len(mol) for weight, mol in row['formula']),
        axis=1)

    self.dataset.eval_property_cols_from_dft_df(['mol_name_length'])

    np.testing.assert_array_equal(
        self.dataset.property_df['mol_name_length'],
        expected_mol_name_length_in_property_df)

  def test_eval_property_cols_from_dft_df_with_nan(self):
    # The formula matrix is
    # [[1., -1., -1., 0., 0., 0., 0.],
    #  [0., 0., 0., 1., 0., 0., 0.],
    #  [0., 0., 0., 0., 1., 0., 0.],
    #  [0., 0., 0., 0., 0., 1., -1.]]
    self.dataset.dft_df['col_with_nans'] = [
        np.nan, np.nan, np.nan, 1, 2, np.nan, np.nan]

    self.dataset.eval_property_cols_from_dft_df(['col_with_nans'])

    np.testing.assert_array_equal(
        self.dataset.property_df['col_with_nans'], [np.nan, 1, 2, np.nan])

  def test_load_mgcdb84_training_set(self):
    self.dataset.property_df['mgcdb84_set'] = [
        'validation', 'train', 'train', 'test']
    self.dataset.save(save_directory=self.test_dir)
    _make_random_ks_info_for_mol(self.test_dir, '11_H_AE18')

    loaded_dataset = dataset.load_mgcdb84_training_set(
        mgcdb84_directory=self.test_dir,
        scf_directory=self.test_dir)

    pd.testing.assert_frame_equal(
        loaded_dataset.property_df.drop(columns=scf.SCF_SCALAR_RESULTS),
        self.dataset.property_df.iloc[[1]])


if __name__ == '__main__':
  absltest.main()
