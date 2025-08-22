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

"""Library for data handling in the Makita concentrated pipeline."""

import ast
import functools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors


def parse_feature_smiles_morgan_fingerprint(
    feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
  """Parses SMILES strings into morgan fingerprints.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    fingerprint_radius: integer radius to use for Morgan fingerprint
      calculation.
    fingerprint_size: integer number of bits to use in the Morgan fingerprint.

  Returns:
    Array of Morgan fingerprints of the molecules represented by the given
    SMILES strings.
  """
  fingerprint_fn = functools.partial(
      AllChem.GetMorganFingerprintAsBitVect,
      radius=fingerprint_radius,
      nBits=fingerprint_size)

  return np.array([
      np.array(fingerprint_fn(Chem.MolFromSmiles(smiles)))
      for smiles in feature_dataframe[feature_column]
  ])


def parse_feature_smiles_rdkit_properties(
    feature_dataframe, feature_column, *args,
    **kwargs):
  """Computes RDKit Descriptor values for input features.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    *args:
    **kwargs: Extra arguments not used by this function. These need to be
      included for this parser to satisfy the generic parser interface.

  Returns:
    Array of rdkit descriptor values for the molecules described in the feature
      dataframe.
  """
  del args, kwargs  # Unused.
  descriptors = [name for name, _ in Chem.Descriptors.descList]
  calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)

  return np.array([
      np.array(calculator.CalcDescriptors(Chem.MolFromSmiles(smiles)))
      for smiles in feature_dataframe[feature_column]
  ])


def parse_feature_smiles_morgan_fingerprint_with_descriptors(
    feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
  """Generates input vectors with molecules' fingerprints and descriptor values.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    fingerprint_radius: integer radius to use for Morgan fingerprint
      calculation.
    fingerprint_size: integer number of bits to use in the Morgan fingerprint.

  Returns:
    Two dimensional array holding a concatenation of the Morgan fingerprint and
    the rdkit descriptor values for each input SMILES.
  """
  fingerprint_features = parse_feature_smiles_morgan_fingerprint(
      feature_dataframe, feature_column, fingerprint_radius, fingerprint_size)
  descriptor_values = parse_feature_smiles_rdkit_properties(
      feature_dataframe, feature_column)

  return np.concatenate([fingerprint_features, descriptor_values], axis=1)


def parse_feature_vectors(feature_dataframe,
                          feature_column):
  """Converts string feature vectors into numpy arrays for modeling.

  Args:
    feature_dataframe: pandas dataframe of shape (N, *) with a column containing
      full length feature vectors as strings.
    feature_column: string name of the column of feature_dataframe holding the
      feature vectors.

  Returns:
    A numpy array of the full input matrix of shape (N, len(feature vectors)).
  """
  return np.array([
      np.array(ast.literal_eval(samp))
      for samp in feature_dataframe[feature_column].to_numpy()
  ])


def parse_feature_numbers(feature_dataframe,
                          feature_column):
  """Parses a series of floats or float-like values.

  Anything that can be parsed as a float (e.g. a string) is valid here.

  Args:
    feature_dataframe: pandas series of shape (n_samples, *) containing a column
      holding float like values.
    feature_column: string column name holding float like features.

  Returns:
    A numpy array of the input series.
  """
  return np.array(list(map(float, feature_dataframe[feature_column])))
