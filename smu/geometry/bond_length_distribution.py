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

"""Classes for holding information about distribution of bond lengths.

The core idea is to represent the probability distribution function (via
LengthDistribution) over lengths.
p(length | atom_types, bond_type)
Those pdfs can then be used to compute (via AtomPairLengthDistributions)
P(bond_type | length, atom types)

Data for all atom pairs is collected in AllAtomPairLengthDistributions
"""

import abc
import itertools
from typing import Dict, Optional
from absl import logging
import numpy as np
import pandas as pd
import scipy.stats

from tensorflow.io import gfile

from smu import dataset_pb2
from smu.parser import smu_utils_lib


class LengthDistribution(abc.ABC):
  """Abstract class for representing a distribution over bond lengths."""

  @abc.abstractmethod
  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query
    Return: pdf value
    """
    raise NotImplementedError


class FixedWindowLengthDistribution(LengthDistribution):
  """Represents a distribution with a fixed value over a window.

  The window is specified by a mimimum and maximum value.
  Further, a "right_tail_mass" can be specified. If given, that fraction of the
  total probability mass is added as an exponential distribution to the right
  side of the window. The exponential distribution is constructed such that the
  pdf is continuous at the maximum value.
  """

  def __init__(self, minimum, maximum,
               right_tail_mass):
    """Construct a FixedWindowLengthDistribution.

    Args:
      minimum: left side of window
      maximum: right side of window
      right_tail_mass: probability mass added part the right side of the window
        (see class documentation)
    """
    self.minimum = minimum
    self.maximum = maximum
    self.right_tail_mass = right_tail_mass

    # This is the density of the fixed window
    self._fixed_density = 1.0 / (self.maximum - self.minimum)
    if self.right_tail_mass:
      self._fixed_density *= 1.0 - self.right_tail_mass

      # The right tail is an exponential distribution. We want the pdf at
      # maximum (the 0 of the exponential) to be equal to fixed_density.
      # For the parameterization of pdf(x) = lambda exp(-lambda x)
      # this means setting lambda to fixed_density / right_tail_mass
      # scipy.stats uses scale which corresponds to 1/lambda
      # pylint: disable=g-long-lambda
      self._right_tail_dist = (
          lambda length: self.right_tail_mass * scipy.stats.expon.pdf(
              length,
              loc=self.maximum,
              scale=self.right_tail_mass / self._fixed_density))

  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query

    Returns:
      pdf value
    """
    if length < self.minimum:
      return 0.0

    if length <= self.maximum:
      return self._fixed_density

    if not self.right_tail_mass:
      return 0.0

    return self._right_tail_dist(length)


class EmpiricalLengthDistribution(LengthDistribution):
  """Represents a distribution from empirically observed counts.

  Note that while the values are given as discrete buckets, these discrete
  values are converted to a stepwise pdf. Further, a "right_tail_mass" can be
  specified. If given, that fraction of the total probability mass is added as
  an exponential distribution to the right of the last bucket given. The
  exponential distribution is constructed such that the pdf at the left edge of
  the exponential is equal to the right most non-zero value in the pdf. This
  means that if the right most bucket is non-zero, the pdf will be continuous.
  """

  def __init__(self, df, right_tail_mass):
    """Construct EmpiricalLengthDistribution.

    It is expected that the space between the elements of df['length'] are
    equal.

    You probably want to use one of the from_* methods instead of this directly.

    Args:
      df: DataFrame must contain columns length, count
      right_tail_mass: probability mass added part the right side of the buckets
        (see class documentation)
    """
    self._df = df.sort_values(by='length')
    # Compute the bucket size as the smallest difference between two buckets and
    # make sure the length is consistent
    bucket_sizes = np.diff(self._df['length'])
    self.bucket_size = np.min(bucket_sizes)
    if not np.allclose(self.bucket_size, np.max(bucket_sizes)):
      raise ValueError(
          'All buckets must be same size, range observed is {} - {}'.format(
              self.bucket_size, np.max(bucket_sizes)))

    # The maximum value covered by the emprical values.
    self._maximum = self._df['length'].iloc[-1] + self.bucket_size

    self._df['pdf'] = (
        self._df['count'] / np.sum(self._df['count']) / self.bucket_size)

    self.right_tail_mass = right_tail_mass
    self._right_tail_dist = None
    if self.right_tail_mass:
      self._df['pdf'] *= (1.0 - self.right_tail_mass)
      last_density = (float(self._df['pdf'].tail(1)))
      # The right tail is an exponential distribution. We want the pdf at
      # right side of what is given in the empirical buckets (the 0 of the
      # exponential) to be equal to the density of the last bucket.
      # For the parameterization of the exponential of
      # pdf(x) = lambda exp(-lambda x)
      # this means setting lambda to last_density / right_tail_mass
      # scipy.stats uses scale which corresponds to 1/lambda
      exponential_dist = scipy.stats.expon(
          loc=(self._df.iloc[-1, self._df.columns.get_loc('length')] +
               self.bucket_size),
          scale=self.right_tail_mass / last_density)
      self._right_tail_dist = (
          lambda length: self.right_tail_mass * exponential_dist.pdf(length))

  @classmethod
  def from_file(cls, filename, right_tail_mass):
    """Load the distribution from a file.

    The file should be a CSV with two (unnamed) columns. The first is a length
    bucket and the second is a count. Each line represents the number of
    observed lengths between the length on tha line and the next line. The
    buckets should be equally spaced..

    Args:
      filename: file to read
      right_tail_mass: probability mass added past the right side of the buckets
        (see class documentation)

    Returns:
      EmpiricalLengthDistribution
    """
    with gfile.GFile(filename) as f:
      df = pd.read_csv(f, header=None, names=['length', 'count'], dtype=float)

    return EmpiricalLengthDistribution(df, right_tail_mass)

  @classmethod
  def from_sparse_dataframe(cls, df_input, right_tail_mass, sig_digits):
    """Creates EmpiricalLengthDistribution from a sparse dataframe.

    "sparse" means that not every bucket is listed explictly. The main work
    in this function to to fill in the implicit values expected in the
    constructor.

    Args:
      df_input: pd.DataFrame with columns ['length_str', 'count']
      right_tail_mass: probability mass added past the right side of the buckets
        (see class documentation)
      sig_digits: number of significant digits after the decimal point

    Returns:
      EmpiricalLengthDistribution
    """
    bucket_size = np.float_power(10, -sig_digits)
    input_lengths = df_input['length_str'].astype(np.double)
    lengths = np.arange(
        np.min(input_lengths),
        # bucket_size / 2 avoids numerical imprecision to make
        # sure that the max is the last bucket
        np.max(input_lengths) + bucket_size / 2,
        bucket_size)
    df_lengths = pd.DataFrame.from_dict({'length': lengths})

    format_str = '{:.%df}' % sig_digits
    df_lengths['length_str'] = df_lengths['length'].apply(format_str.format)

    df = df_lengths.merge(df_input, how='outer', on='length_str')
    if len(df) != len(df_lengths):
      raise ValueError('Unexpected length_str values in input: {}'.format(
          set(df_input['length_str']).difference(df_lengths['length_str'])))

    df['count'].fillna(0, inplace=True)

    return EmpiricalLengthDistribution(df, right_tail_mass=right_tail_mass)

  @classmethod
  def from_arrays(cls, lengths, counts, right_tail_mass):
    """Creates EmpiricalLengthDistribution from arrays.

    Args:
      lengths: sequence of values of the left edges of the buckets (same length
        as counts) sequence of left edge of length buckets
      counts: sequence of counts observed in each bucket
      right_tail_mass: probability mass added past the right side of the buckets
        (see class documentation)

    Returns:
      EmpiricalLengthDistribution
    """
    return EmpiricalLengthDistribution(
        pd.DataFrame.from_dict({
            'length': lengths,
            'count': counts
        }), right_tail_mass)

  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query

    Returns:
      pdf value
    """
    idx = self._df['length'].searchsorted(length)
    if idx == 0:
      return 0.0
    if length > self._maximum:
      if self._right_tail_dist:
        return self._right_tail_dist(length)
      else:
        return 0.0

    return self._df.iloc[idx - 1, self._df.columns.get_loc('pdf')]


class AtomPairLengthDistributions:
  """Maintains a set of distributions for different bond types.

  Note while this is called "AtomPair" the object does not store what atoms
  this set of distributions is for.
  """

  def __init__(self):
    self._bond_type_map: Dict['dataset_pb2.BondTopology.BondType',
                              LengthDistribution] = {}

  def add(self, bond_type,
          dist):
    """Adds a LengthDistribution for a given bond type.

    Args:
      bond_type: bond_type to add dsitribution for
      dist: LengthDistribution
    """
    self._bond_type_map[bond_type] = dist

  def __getitem__(self, bond_type):
    """Gets the Distribution for the bond_type."""
    return self._bond_type_map[bond_type]

  def probability_of_bond_types(
      self, length):
    """Computes probability of bond types given a length.

    Only bond types with non zero probability at that length are returned.
    An empty dictionary will be returned if no bond type has non-zero
    probability.
    Otherwise, the probabilities will sum to 1.

    This is a simple application of Bayes rule (note P is for a discrete
    probability and p is for a pdf)
    P(bond_type | length) = p(length | bond_type) P(bond_type) / P(length)
    By assuming P(bond_type) is uniform, we can just get p(length | bond_type)
    and normalize to sum to 1.
    We acknowledge that P(bond_type) is not actually uniform, but we
    intentionally chose this uninformative prior to simplify the reasoning.

    Args:
      length: length to query

    Returns:
      dictionary of bond types to probability
    """
    out = {}
    for bond_type, dist in self._bond_type_map.items():
      pdf = dist.pdf(length)
      if pdf != 0.0:
        out[bond_type] = pdf

    normalizer = sum(out.values())
    for bond_type in out:
      out[bond_type] /= normalizer

    return out


class AllAtomPairLengthDistributions:
  """Maintains a set of distributions for all atom pair and bond types.

  Note that all methods which take atom_a and atom_b will return the same value
  for either order of the arguments.
  """

  def __init__(self):
    self._atom_pair_dict = {}

  def add(self, atom_a,
          atom_b,
          bond_type,
          dist):
    """Adds a distribution of the atom pair and bond type."""
    if (atom_a, atom_b) not in self._atom_pair_dict:
      self._atom_pair_dict[(atom_a, atom_b)] = AtomPairLengthDistributions()
      # Just set the other order of atom_a, atom_b to the same object
      self._atom_pair_dict[(atom_b, atom_a)] = self._atom_pair_dict[(atom_a,
                                                                     atom_b)]
    self._atom_pair_dict[(atom_a, atom_b)].add(bond_type, dist)

  def add_from_files(self, filestem,
                     unbonded_right_tail_mass):
    """Adds distributions from a set of files.

    Files are expected to be named {filestem}.{atom_a}.{bond_type}.{atom_b}
    where
    * atom_a, atom_b: atomic numbers for H, C, N, O, F (smaller number first)
    * bond_type: {0, 1, 2, 3} for {unbonded, single, double, triple}

    Missing files are silently ignored.

    Contents are as expected by EmpiricalLengthDistribution.from_file

    Args:
      filestem: prefix of files to load
      unbonded_right_tail_mass: right_tail_mass (as described in
        EmpiricalLengthDistribution) for the unbonded cases.
    """
    atom_types = [
        dataset_pb2.BondTopology.ATOM_H,
        dataset_pb2.BondTopology.ATOM_C,
        dataset_pb2.BondTopology.ATOM_N,
        dataset_pb2.BondTopology.ATOM_O,
        dataset_pb2.BondTopology.ATOM_F,
    ]

    bond_types = [
        dataset_pb2.BondTopology.BOND_UNDEFINED,
        dataset_pb2.BondTopology.BOND_SINGLE,
        dataset_pb2.BondTopology.BOND_DOUBLE,
        dataset_pb2.BondTopology.BOND_TRIPLE,
    ]

    for (atom_a, atom_b), bond_type in itertools.product(
        itertools.combinations_with_replacement(atom_types, 2), bond_types):
      fname = '{}.{}.{}.{}'.format(
          filestem, smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a],
          int(bond_type), smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b])

      if not gfile.exists(fname):
        logging.info('Skipping non existent file %s', fname)
        continue

      right_tail_mass = None
      if bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED:
        right_tail_mass = unbonded_right_tail_mass

      self.add(atom_a, atom_b, bond_type,
               EmpiricalLengthDistribution.from_file(fname, right_tail_mass))

  def add_from_sparse_dataframe(self, df_input, unbonded_right_tail_mass,
                                sig_digits):
    """Adds distributions from a sparse dataframe.

    See sparse_dataframe_from_records for a description of the expected input
    format.

    Args:
      df_input: pd.DataFrame
      unbonded_right_tail_mass: right_tail_mass (as described in
        EmpiricalLengthDistribution) for the unbonded cases.
      sig_digits: number of significant digits after the decimal point
    """
    avail_pairs = set(
        df_input.apply(
            lambda r: (r['atom_char_0'], r['atom_char_1'], r['bond_type']),
            axis=1))
    for atom_char_0, atom_char_1, bond_type in avail_pairs:
      atom_0 = smu_utils_lib.ATOM_CHAR_TO_TYPE[atom_char_0]
      atom_1 = smu_utils_lib.ATOM_CHAR_TO_TYPE[atom_char_1]
      df = df_input[(df_input['atom_char_0'] == atom_char_0)
                    & (df_input['atom_char_1'] == atom_char_1) &
                    (df_input['bond_type'] == bond_type)]

      right_tail_mass = None
      if bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED:
        right_tail_mass = unbonded_right_tail_mass

      self.add(
          atom_0, atom_1, bond_type,
          EmpiricalLengthDistribution.from_sparse_dataframe(
              df, right_tail_mass, sig_digits))
    pass

  def pdf_length_given_type(self, atom_a,
                            atom_b,
                            bond_type,
                            length):
    """p(length | atom_a, atom_b, bond_type)."""
    return self._atom_pair_dict[(atom_a, atom_b)][bond_type].pdf(length)

  def probability_of_bond_types(
      self, atom_a,
      atom_b,
      length):
    """P(bond_type | atom_a, atom_b, length).

    See AtomPairLengthDistributions.probability_of_bond_type for details.

    Args:
      atom_a: first atom type
      atom_b: seocnd atom type (order of atom_a, atomb is irrelevant)
      length: length to query

    Returns:
      dictionary of bond type to probability
    """
    return self._atom_pair_dict[(atom_a,
                                 atom_b)].probability_of_bond_types(length)


def sparse_dataframe_from_records(records):
  """Builds a dataframe with emprical buckets for many atom pairs.

  The "sparse" refers to the fact that not every bucket will be explicitly
  represented.

  The strange grouping in records is because this is what comes from the beam
  pipeline easily.

  Args:
    records: iterables of [(atom char 0, atom char 1, bond type, length), count]
      where atom char is one of 'cnofh', bond type is an int in [0, 3], length
      is a string, count is an integer

  Returns:
    pd.DataFrame with columns:
      atom_char_0, atom_char_1, bond_type, length_str, count
  """
  reformatted = []
  for (a0, a1, bt, length), count in records:
    reformatted.append((a0, a1, bt, length, count))
  df = pd.DataFrame.from_records(
      reformatted,
      columns=[
          'atom_char_0', 'atom_char_1', 'bond_type', 'length_str', 'count'
      ])
  return df.sort_values(
      ['atom_char_0', 'atom_char_1', 'bond_type', 'length_str'])
