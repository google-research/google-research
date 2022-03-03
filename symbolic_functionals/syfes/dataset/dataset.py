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

"""Handler of dataset."""

import ast
import collections
import json
import os
import warnings

from absl import logging
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tqdm


# NOTE(htm): SCF_SCALAR_RESULTS is identical to SCF_SCALAR_RESULTS in
# california.scf.scf.SCF_SCALAR_RESULTS, and the consistency is checked by a
# unit test. It is redefined here to avoid dependency to california.scf.scf and
# PySCF.

SCF_SCALAR_RESULTS = [
    'Etot', 'Exc', 'Exx', 'Exxlr', 'Enlc', 'converged', 'time']

# Dataset type info for MGCDB84

_MGCDB84_INFO = {
    'NCED': {
        'train': ['A24', 'DS14', 'HB15', 'HSG', 'NBC10', 'S22', 'X40'],
        'validation': ['A21x12', 'BzDC215', 'HW30', 'NC15', 'S66', 'S66x8'],
        'test': ['3B-69-DIM', 'AlkBind12', 'CO2Nitrogen16', 'HB49', 'Ionic43'],
    },
    'NCEC': {
        'train': ['H2O6Bind8', 'HW6Cl', 'HW6F'],
        'validation': [
            'FmH2O10', 'Shields38', 'SW49Bind345', 'SW49Bind6', 'WATER27'
        ],
        'test': ['3B-69-TRIM', 'CE20', 'H2O20Bind10', 'H2O20Bind4'],
    },
    'NCD': {
        'train': ['TA13', 'XB18'],
        'validation': ['Bauza30', 'CT20', 'XB51'],
        'test': [],
    },
    'IE': {
        'train': ['AlkIsomer11', 'Butanediol65'],
        'validation': [
            'ACONF', 'CYCONF', 'Pentane14', 'SW49Rel345', 'SW49Rel6'
        ],
        'test': [
            'H2O16Rel5', 'H2O20Rel10', 'H2O20Rel4', 'Melatonin52', 'YMPJ519'
        ]
    },
    'ID': {
        'train': ['EIE22', 'Styrene45'],
        'validation': ['DIE60', 'ISOMERIZATION20'],
        'test': ['C20C24'],
    },
    'TCE': {
        'train': ['AlkAtom19', 'BDE99nonMR', 'G21EA', 'G21IP', 'TAE140nonMR'],
        'validation': [
            'AlkIsod14', 'BH76RC', 'EA13', 'HAT707nonMR', 'IP13', 'NBPRC',
            'SN13'
        ],
        'test': ['BSR36', 'HNBrBDE18', 'WCPT6'],
    },
    'TCD': {
        'train': [],
        'validation': ['BDE99MR', 'HAT707MR', 'TAE140MR'],
        'test': ['PlatonicHD6', 'PlatonicID6', 'PlatonicIG6', 'PlatonicTAE6'],
    },
    'BH': {
        'train': ['BHPERI26', 'CRBH20', 'DBH24'],
        'validation': ['CR20', 'HTBH38', 'NHTBH38'],
        'test': ['PX13', 'WCPT27']
    },
    'AE18': {
        'train': ['AE18'],
        'validation': [],
        'test': [],
    },
    'RG10': {
        'train': [],
        'validation': ['RG10'],
        'test': [],
    }
}

# Datatype-specific weights for MGCDB84 given in 10.1063/1.4952647

_MGCDB84_WEIGHTS = {
    ('TCD',): 0.1,
    ('TCE', 'AE18'): 1,
    ('NCD',): 10,
    ('ID',): 10,
    ('BH',): 10,
    ('NCED', 'NCEC'): 100,
    ('IE',): 100,
    ('RG10',): 10000,
}


def _load_property_df(path):
  """Parses DatasetEval.csv of MGCDB84 to construct property_df.

  Args:
    path: String, the path to the MGCDB84 dataset.

  Returns:
    Pandas dataframe, the resulting property_df.
  """
  property_data = {}
  with tf.io.gfile.GFile(os.path.join(path, 'DatasetEval.csv'), 'r') as f:
    for line in f:
      tokens = line.split(',')
      index = tokens[0]
      num_single_points = len(tokens) // 2 - 1

      property_data[index] = {
          'dataset_name': index.partition('_')[0],
          'num_single_points': num_single_points,
          'formula': [(int(tokens[2 * i + 1]), tokens[2 * i + 2])
                      for i in range(num_single_points)],
          'ref_value': float(tokens[-1])
      }
  return pd.DataFrame.from_dict(property_data, orient='index')


def _apply_mgcdb84_specific_operations(property_df):
  """Applies operations specific to MGCDB84 dataset to property_df.

  Args:
    property_df: Pandas dataframe, the property dataframe on which the
      operations will be applied inplace.

  Returns:
    Pandas dataframe, the property dataframe after operations are applied.
  """
  # drop 'XB51_15' row, which involves two molecules whose geometry files
  # are missing from the dataset: '34_BrBr_HLi_XB51' and '74_HLi_XB5'
  property_df.drop(index='XB51_15', inplace=True)

  # split TAE140, BDE99 and HAT707 to MR (multireference) and nonMR subsets
  # see 10.1016/j.cplett.2011.05.007
  tae140_mr_indices = [115] + [121] + list(range(140 - 14 + 1, 140 + 1))

  property_df['MR'] = False
  for idx in tae140_mr_indices:
    mr_mol = property_df.loc[f'TAE140_{idx}', 'formula'][0][1]
    for index, row in property_df.iterrows():
      if mr_mol in [mol_name for _, mol_name in row['formula']]:
        property_df.loc[index, 'MR'] = True

  for index, row in property_df.iterrows():
    if row['dataset_name'] in ['TAE140', 'BDE99', 'HAT707']:
      property_df.loc[index, 'dataset_name'] += 'MR' if row['MR'] else 'nonMR'

  # add data type info
  for mgcdb84_type, info in _MGCDB84_INFO.items():
    for mgcdb84_set, dataset_names in info.items():
      for dataset_name in dataset_names:
        property_df.loc[property_df['dataset_name'] == dataset_name,
                        'mgcdb84_set'] = mgcdb84_set
        property_df.loc[property_df['dataset_name'] == dataset_name,
                        'mgcdb84_type'] = mgcdb84_type

  # determination of weights
  # step 1: number of data points in the dataset & RMS
  counts = property_df.groupby('dataset_name')['ref_value'].count()
  rmss = property_df.groupby('dataset_name')['ref_value'].agg(
      lambda x: np.sqrt(np.mean(x**2)))
  weights = 1 / (counts * rmss)
  weights.name = 'mgcdb84_weight'

  for group, group_weight in _MGCDB84_WEIGHTS.items():
    dataset_names = []
    for mgcdb84_type in group:
      for names in _MGCDB84_INFO[mgcdb84_type].values():
        dataset_names.extend(names)
    # step 2: normalize weights in every group such that they lie in [1, 2]
    weights_in_group = weights[dataset_names]
    weights_in_group /= np.min(weights_in_group)
    if group != ('RG10',):
      exponent = np.log(2) / np.log(np.max(weights_in_group))
    else:
      exponent = 1
    weights_in_group = weights_in_group**exponent

    # step 3: multiply with overall group weight
    weights[dataset_names] = weights_in_group * group_weight

  # merge weights into property_df
  property_df = property_df.merge(
      weights, left_on='dataset_name', right_index=True)

  # NOTE(htm): For RG10 dataset, dimers with reference energy > 0 should receive
  # a group weight of 1 instead of 10000. This treatment is done in the code
  # written by the author of MGCDB84, but is not mentioned in the wB97M-V paper.
  property_df.loc[
      (property_df['mgcdb84_type'] == 'RG10') & (property_df['ref_value'] > 0),
      'mgcdb84_weight'] /= _MGCDB84_WEIGHTS[('RG10',)]

  # step 4: validation set get an additional factor of 2
  property_df.loc[property_df['mgcdb84_set'] == 'validation',
                  'mgcdb84_weight'] *= 2

  return property_df


def _load_dft_df_and_geometries(path, property_df):
  """Constructs dft_df based on property_df and Geometries folder.

  Args:
    path: String, the path to the MGCDB84 dataset.
    property_df: Pandas dataframe, the property dataframe. The num_electrons and
      spin_singlet columns in property_df will also be updated.

  Returns:
    dft_df: Pandas dataframe, the resulting dft dataframe.
    geometries: Dict, contains the xyz strings of all molecules in dft_df.
  """
  # NOTE(htm): importing pymatgen in the beginning causes issues for colab
  # adhoc import
  from pymatgen.io import xyz  # pylint: disable=g-import-not-at-top,import-outside-toplevel

  # construct dft_df based on property_df and Geometries folder
  # update num_electrons and spin_singlet in property_df
  logging.info('Reading geometry files......')

  dft_data = {}
  geometries = {}

  with tqdm.tqdm(total=len(property_df), position=0, leave=True) as t:
    for index, row in property_df.iterrows():
      formula = row['formula']
      max_num_electrons = 0
      spin_singlet = True
      for _, mol_name in formula:
        if mol_name not in dft_data:
          # read xyz file
          xyz_path = os.path.join(path, 'Geometries', f'{mol_name}.xyz')
          with tf.io.gfile.GFile(xyz_path) as f:
            xyz_string = f.read()
          geometries[mol_name] = xyz_string.splitlines()

          # the second line of xyz file records charge and spin multiplicity
          charge, spin_mult = map(int, xyz_string.splitlines()[1].split())

          # compute number of electrons and chemical formula using pymatgen
          # ignore pymatgen warnings on He electronegativity
          with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mol = xyz.XYZ.from_string(xyz_string).molecule
            num_electrons = mol.composition.total_electrons
            symbols = mol.composition.formula.replace(' ', '')

          # store parsed results to dft_data and geometries
          dft_data[mol_name] = {
              'num_electrons': num_electrons,
              'charge': charge,
              'spin': spin_mult - 1,
              'symbols': symbols,
          }

        # update relevant cols of property_df
        max_num_electrons = max(
            max_num_electrons, dft_data[mol_name]['num_electrons'])
        if dft_data[mol_name]['spin'] != 0:
          spin_singlet = False

      property_df.loc[index, 'num_electrons'] = max_num_electrons
      property_df.loc[index, 'spin_singlet'] = spin_singlet

      t.update()

  # construct dft_df from dft_data
  dft_df = pd.DataFrame.from_dict(
      dft_data, orient='index').reindex(index=dft_data.keys())

  return dft_df, geometries


def load_mgcdb84_training_set(mgcdb84_directory,
                              scf_directory,
                              num_samples=None):
  """Loads training set of MGCDB84 database.

  Args:
    mgcdb84_directory: String, save directory of MGCDB84 database. Note that
      mgcdb84_directory is not the directory containing the raw data of
      MGCDB84 (used by Dataset.from_mgcdb84_dataset). Instead, it is the save
      directory after the dataset is saved through the save method. Loading
      saved data is significantly faster than loading raw data.
    scf_directory: String, save directory of SCF calculations.
    num_samples: Integer, if present, sets the maximum number of rows for
      property_df of the resulting dataset.

  Returns:
    Dataset, the training set of MGCDB84 database, excluding all rows with
      missing ks_info.
  """
  mgcdb84 = Dataset.load(save_directory=mgcdb84_directory)
  property_df_train = mgcdb84.property_df[
      mgcdb84.property_df['mgcdb84_set'] == 'train']
  if num_samples is not None:
    property_df_train = property_df_train.head(num_samples)

  mgcdb84_train = mgcdb84.get_subset(property_df_subset=property_df_train)
  mgcdb84_train.load_ks_info(ks_info_directory=scf_directory)

  return mgcdb84_train.get_subset(only_keep_properties_with_ks_info=True)


class Dataset:
  """A class to handle datasets for molecular properties and DFT calculations.

  Two Pandas Dataframes property_df and dft_df are maintained.

  Each row of property_df represents a molecular property with a given reference
  value.
  Important columns in property_df:
    dataset_name: String, the name of the datasetset (e.g. 'AE').
    spin_singlet: Boolean, whether all related single point calculations
      are spin unpolarized.
    num_electrons: Integer, the maximum number of electrons for all single point
      calculations.
    num_single_points: Integer, the number of related single point calculations.
    formula: List of Tuple[int, str], the formula to compute the molecular
      property from single point calculations.
    ref_value: Float, the reference value for the molecular property.

  Each row of dft_df represents a single point calculation on a given structure
  (Geometries/{index}.xyz).
  Important columns in dft_df:
    num_electrons: Integer, the number of electrons (including extra charge).
    charge: Integer, the extra charge.
    spin: Integer, the spin (2S).
    symbols: String, the chemical formula.

  To facilitate evaluating quantities in property_df based on dft_df, a formula
  matrix of size (nrow_property, nrow_dft) is maintained, which can be
  multiplied with a column of dft_df to generate corresponding column for
  property_df.

  In addition to property_df and dft_df, two dictionaries are maintaied:
  geometries and ks_info. geometries stores the xyz-string of molecules.
  ks_info and stores certain quantities from DFT calculations such as electron
  density, the weights on grids for numerical integration (not to be confused
  with dataset weights in mgcdb84_weights column of property_df or weights for
  computing molecular property in the formula column of property_df), etc.
  """

  _property_df_types = {
      'spin_singlet': bool,
      'num_electrons': int,
      'num_single_points': int,
  }

  _dft_df_types = {
      'num_electrons': int,
      'charge': int,
      'spin': int,
  }

  def __init__(self, property_df, dft_df, geometries, ks_info=None):
    """Constructs Dataset from property_df and dft_df.

    Args:
      property_df: Pandas dataframe, the dataframe for training and testing ML
        models. Every row represents a certain molecular property (such as
        isomerization energy), which is obtained from one or several single
        point DFT calculations on different molecular structures.
      dft_df: Pandas dataframe, the dataframe for DFT calculations. Every row
        corresponds to one xyz file and represents a single point DFT
        calculation on a certain molecular structure.
      geometries: Dict, a dictionary for xyz files.
      ks_info: Dict, a dictionary that stores important quantities from KS
        calculations including densities, weights, elda, etc.
    """
    self.property_df = property_df.astype(self._property_df_types, copy=True)

    # filter out content in dft_df, geometries and ks_info that are not
    # used in property_df
    self.dft_df = self.filter_dft_df_with_property_df(
        self.property_df, dft_df.astype(self._dft_df_types, copy=True))
    self.geometries = {
        index: geometries[index]
        for index in self.dft_df.index
    }

    if ks_info:
      self.ks_info = {
          index: ks_info[index]
          for index in self.dft_df.index
      }
    else:
      self.ks_info = {}

    self.formula_matrix = self.compute_formula_matrix()

  @staticmethod
  def from_mgcdb84_dataset(path, complete_mgcdb84=True):
    """Parses the MGCDB84 database.

    Args:
      path: String, the path to the MGCDB84 database (see SI of
        10.1080/00268976.2017.1333644). The path should contain the
        DatasetEval.csv file and the Geometries folder (extracted from
        Geometries.tar file).
      complete_mgcdb84: Boolean, if True, assume the path contains the complete
        MGCDB84 dataset and performs certain MGCDB84-specific operations.

    Returns:
      Dataset, the resulting Dataset instance.
    """
    property_df = _load_property_df(path)
    if complete_mgcdb84:
      property_df = _apply_mgcdb84_specific_operations(property_df)
    dft_df, geometries = _load_dft_df_and_geometries(path, property_df)

    # rearrange columns order of property_df
    if complete_mgcdb84:
      property_df = property_df[[
          'dataset_name', 'mgcdb84_type', 'mgcdb84_set', 'mgcdb84_weight',
          'spin_singlet', 'num_electrons', 'num_single_points', 'formula',
          'ref_value'
      ]]
    else:
      property_df = property_df[[
          'dataset_name', 'spin_singlet', 'num_electrons', 'num_single_points',
          'formula', 'ref_value'
      ]]

    dataset = Dataset(
        property_df=property_df, dft_df=dft_df, geometries=geometries)

    if complete_mgcdb84:
      dataset.verify_dataset_integrity(
          expected_nrow=(4986, 5931), expected_num_datasets=84)

    return dataset

  def verify_dataset_integrity(self, expected_nrow, expected_num_datasets):
    """Verifies the basic properties of property_df and dft_df.

    Args:
      expected_nrow: Tuple of 2 integers, the expected number of rows for
        self.property_df and self.dft_df.
      expected_num_datasets: Integer, the expected number of different datsets.

    Raises:
      ValueError: If number of rows for self.property_df and self.dft_df is
        not equal to expected_nrow.
      ValueError: If number of datasets is not equal to expected_num_datasets.
      ValueError: If self.dft_df contains rows not used by self.property_df.
    """
    if self.nrow != expected_nrow:
      raise ValueError('Number of rows for property_df and dft_df '
                       f'{self.nrow} are not equal to the expected values '
                       f'{expected_nrow}.')
    if len(self.dataset_names) != expected_num_datasets:
      raise ValueError(f'Number of datasets {len(self.dataset_names)} is not '
                       f'equal to the expected value {expected_num_datasets}.')
    if not pd.DataFrame.equals(
        self.dft_df,
        self.filter_dft_df_with_property_df(self.property_df, self.dft_df)):
      raise ValueError('dft_df contains rows not used by property_df.')

  def compute_formula_matrix(self):
    """Computes a formula matrix based the formula column in property_df.

    Returns:
      Float numpy array of size (nrow_property, nrow_dft), the formula matrix.
    """
    formula_matrix = np.zeros([self.nrow_property, self.nrow_dft])
    for i, formula in enumerate(self.property_df['formula']):
      for weight, mol_name in formula:
        j = self.dft_df.index.get_loc(mol_name)
        formula_matrix[i, j] += weight
    return formula_matrix

  @staticmethod
  def filter_dft_df_with_property_df(property_df, dft_df):
    """Filters out rows of dft_df not used in any row of property_df.

    Args:
      property_df: Pandas dataframe, the reference property_df.
      dft_df: Pandas dataframe, the full dft_df to be filtered.

    Returns:
      Pandas dataframe, the filtered dft_df.
    """
    mol_is_used = {mol_name: False for mol_name in dft_df.index}
    for _, row in property_df.iterrows():
      for _, mol_name in row['formula']:
        mol_is_used[mol_name] = True
    return dft_df[list(mol_is_used.values())].copy()

  @property
  def nrow(self):
    """Number of rows for property_df and dft_df dataframes."""
    return self.nrow_property, self.nrow_dft

  @property
  def nrow_property(self):
    """Number of rows for the property_df dataframe."""
    return len(self.property_df)

  @property
  def nrow_dft(self):
    """Number of rows for the dft_df dataframe."""
    return len(self.dft_df)

  @property
  def dataset_names(self):
    """List of dataset names (e.g. 'AE18')."""
    return list(set(self.property_df['dataset_name']))

  def get_subset(self,
                 property_df_subset=None,
                 only_keep_properties_with_ks_info=False):
    """Get a subset of the current dataset based on a subset of property_df.

    Args:
      property_df_subset: Pandas dataframe, a subset of self.property_df.
      only_keep_properties_with_ks_info: Boolean, if True, property_df rows with
        missing ks_info will be dropped in the resulting subset.

    Returns:
      Dataset, the subset.

    Raises:
      ValueError, if the index of property_df_subset is not a subset of the
        index of self.property_df.
    """
    if property_df_subset is None:
      property_df_subset = self.property_df

    if not set(property_df_subset.index).issubset(set(self.property_df.index)):
      raise ValueError('Input dataframe is not a subset of property_df.')

    if only_keep_properties_with_ks_info:
      keep_row = property_df_subset.apply(
          lambda row: all(self.ks_info.get(mol) for _, mol in row['formula']),
          axis=1)
      logging.info('Dataset.get_subset: %d rows are dropped due to '
                   'missing ks_info', np.sum(~keep_row))
      property_df_subset = property_df_subset[keep_row]

    return Dataset(
        property_df=property_df_subset,
        dft_df=self.dft_df,
        geometries=self.geometries,
        ks_info=self.ks_info)

  @staticmethod
  def load_mcgdb84_subset(dataset_directory,
                          mgcdb84_set=None,
                          mgcdb84_types=None,
                          spin_singlet=False,
                          nrow_property=None):
    """Loads a subset of MCGDB84 with given specifications.

    Properties with missing ks_info will be removed.

    Args:
      dataset_directory: String, the save directory of dataset and SCF results.
        This directory will be passed to both Dataset.load and
        Dataset.load_ks_info methods.
      mgcdb84_set: String, if present, specifies the set of data in MGCDB84,
        possible values are 'train', 'validation' and 'test'.
      mgcdb84_types: List of strings, if present, specifies the data types.
      spin_singlet: Boolean, if True, only spin unpolarized molecules will
        be included in dataset.
      nrow_property: Integer, if present, specifies the number of rows to be
        kept for the property_df of resulting subset.

    Returns:
      Dataset, the resulting subset.
    """
    mgcdb84 = Dataset.load(save_directory=dataset_directory)
    property_df = mgcdb84.property_df

    if mgcdb84_set:
      property_df = property_df[property_df['mgcdb84_set'] == mgcdb84_set]
    if mgcdb84_types:
      property_df = property_df[property_df['mgcdb84_type'].isin(mgcdb84_types)]
    if spin_singlet:
      property_df = property_df[property_df['spin_singlet']]
    if nrow_property:
      property_df = property_df.sample(nrow_property, random_state=0)

    subset = mgcdb84.get_subset(property_df)
    subset.load_ks_info(ks_info_directory=dataset_directory)

    return subset.get_subset(only_keep_properties_with_ks_info=True)

  def __eq__(self, other):
    """Checks whether two Dataset instances are equal.

    Two Dataset instances are deemed equal if both their property_df and dft_df
    are close for float columns and equal for other columns.

    Args:
       other: Dataset.

    Returns:
       Boolean, whether two datasets are equal.
    """
    try:
      pd.testing.assert_frame_equal(
          self.property_df, other.property_df, check_less_precise=8)
      pd.testing.assert_frame_equal(
          self.dft_df, other.dft_df, check_less_precise=8)
      assert self.geometries == other.geometries
      return True
    except AssertionError:
      return False

  def eval_property_cols_from_dft_df(self, cols):
    """Computes given columns of property_df based on dft_df.

    A weighted sum is computed over given dft_df columns to obtain corresponding
    columns for property_df. The weights are taken from the 'formula' column of
    property_df and recorded in formula_matrix (not to be confused with dataset
    weights on the mgcdb84_weights column). When some molecules in the dft_df
    contain NaNs (e.g. energies of unfinished SCF calculations) for the column
    to be summed over, only rows in property_df that depend on the molecule will
    get NaN for the corresponding column.

    Args:
      cols: List of strings, the property_df columns to be computed, columns
        with the same name are assumed to already exist in dft_df.
    """
    for col in cols:
      self.property_df[col] = np.nan
      mask = np.abs(self.formula_matrix) @ self.dft_df[col].isna() == 0
      self.property_df.loc[mask, col] = (self.formula_matrix[mask, :]
                                         @ np.nan_to_num(self.dft_df[col]))

  def save(self, save_directory):
    """Saves current dataset (property_df, dft_df, geometries) to files.

    Args:
      save_directory: String, the save directory.
    """
    if not tf.io.gfile.exists(save_directory):
      tf.io.gfile.MakeDirs(save_directory)
    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'property_df.csv'), 'w') as f:
      self.property_df.to_csv(f)
    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'dft_df.csv'), 'w') as f:
      self.dft_df.to_csv(f)
    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'geometries.json'), 'w') as f:
      json.dump(self.geometries, f, indent=2)

  @staticmethod
  def load(save_directory, ks_info_directory=None):
    """Loads a dataset (dataframes, geometries and ks_info) from files.

    Args:
      save_directory: String, the save directory of dataset.
      ks_info_directory: String, if present, denotes the directory to load
        ks_info.

    Returns:
      Dataset, the resulting dataset.
    """
    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'property_df.csv'), 'r') as f:
      property_df = pd.read_csv(
          f, index_col=0, converters={'formula': ast.literal_eval})
    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'dft_df.csv'), 'r') as f:
      dft_df = pd.read_csv(f, index_col=0)

    with tf.io.gfile.GFile(
        os.path.join(save_directory, 'geometries.json'), 'r') as f:
      geometries = json.load(f)

    dataset = Dataset(
        property_df=property_df, dft_df=dft_df, geometries=geometries)

    if ks_info_directory:
      dataset.load_ks_info(ks_info_directory=ks_info_directory)

    return dataset

  def load_ks_info(self, ks_info_directory, load_energies_only=False):
    """Loads ks_info from npy files in ks_info_directory.

    This function will update self.ks_info and the columns in self.dft_df
    that belongs to SCF_SCALAR_RESULTS.

    Args:
      ks_info_directory: String, the root directory for npy files. The
        quantities for each molecule is stored a subdirectory.
      load_energies_only: Boolean, if True, only energies in the energies.npy
        file are loaded.
    """
    self.ks_info = {}

    logging.info('Loading KS info from %s......', ks_info_directory)

    for column in SCF_SCALAR_RESULTS:
      self.dft_df[column] = np.nan

    nonexistence_indices = []
    with tqdm.tqdm(total=self.nrow_dft, position=0, leave=True) as t:
      for index in self.dft_df.index:
        self.ks_info[index] = {}
        mol_dir = os.path.join(ks_info_directory, index)
        rho_path = os.path.join(mol_dir, 'rho.npy')
        weights_path = os.path.join(mol_dir, 'weights.npy')
        energies_path = os.path.join(mol_dir, 'energies.npy')

        if not all(tf.io.gfile.exists(path) for path in
                   [mol_dir, rho_path, weights_path, energies_path]):
          nonexistence_indices.append(index)
        else:
          if not load_energies_only:
            with tf.io.gfile.GFile(rho_path, 'rb') as f:
              self.ks_info[index]['rho'] = np.load(f)
            with tf.io.gfile.GFile(weights_path, 'rb') as f:
              self.ks_info[index]['weights'] = np.load(f)
          with tf.io.gfile.GFile(energies_path, 'rb') as f:
            self.dft_df.loc[index, SCF_SCALAR_RESULTS] = np.load(f)

        t.update()

    if nonexistence_indices:
      logging.info(
          'KS info is missing for %d molecules.', len(nonexistence_indices))
      logging.debug('Failed to load KS info for: %s', nonexistence_indices)

    self.eval_property_cols_from_dft_df(SCF_SCALAR_RESULTS)

  @property
  def ks_info_memory_summary(self):
    """Memory usage by ks_info in MB.

    Returns:
      Dict, a summary of memory usage.
    """
    memory = collections.defaultdict(float)

    for ks_info_mol in self.ks_info.values():
      for quantity_name, quantity_arr in ks_info_mol.items():
        memory[quantity_name] += quantity_arr.nbytes

    for quantity_name in memory:
      memory[quantity_name] /= 1e6  # bytes -> MB

    memory['total'] = sum(memory.values())
    return dict(memory)
