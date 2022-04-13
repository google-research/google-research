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
"""Classes for holding information about distribution of bond lengths.

The core idea is to represent the probability distribution function (via
LengthDistribution) over lengths.
p(length | atom_types, bond_type)
Those pdfs can then be used to compute (via AtomPairLengthDistributions)
P(bond_type | length, atom types)

Data for all atom pairs is collected in AllAtomPairLengthDistributions
"""

import abc
import csv
import itertools
import math
import os.path
from typing import Dict, Optional

from absl import logging
import numpy as np
import pandas as pd
import scipy.stats

from smu import dataset_pb2
from smu.parser import smu_utils_lib

ATOMIC_NUMBER_TO_ATYPE = {
    1: dataset_pb2.BondTopology.ATOM_H,
    6: dataset_pb2.BondTopology.ATOM_C,
    7: dataset_pb2.BondTopology.ATOM_N,
    8: dataset_pb2.BondTopology.ATOM_O,
    9: dataset_pb2.BondTopology.ATOM_F
}

# These are the numbers we will use throughout the pipeline in normal use
STANDARD_SIG_DIGITS = 3
STANDARD_UNBONDED_RIGHT_TAIL_MASS = 0.9


class BondLengthParseError(Exception):

  def __init__(self, term):
    super().__init__(term)
    self.term = term

  def __str__(self):
    ('Bond lengths spec must be comma separated terms like form XYX:N-N '
     'where X is an atom type (CNOF*), Y is a bond type (-=#.~), '
     'and N is a possibly empty floating point number. '
     '"{}" did not parse.').format(self.term)


def interpolate_zeros(values):
  """For each zero value in `values` replace with an interpolated value.

  Args:
   values: an array that may contain zeros.

  Returns:
   An array that contains no zeros.
  """
  xvals = np.nonzero(values)[0]
  yvals = values[xvals]

  # Simplest to get values for all points, even those already non zero.
  indices = np.arange(0, len(values))

  return np.interp(indices, xvals, yvals)


class LengthDistribution(abc.ABC):
  """Abstract class for representing a distribution over bond lengths."""

  @abc.abstractmethod
  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query

    Returns:
      pdf value
    """
    raise NotImplementedError

  @abc.abstractmethod
  def min(self):
    """Minimum value that returns a non-zero pdf.

    Returns:
      float
    """
    raise NotImplementedError

  @abc.abstractmethod
  def max(self):
    """Maximum value that returns a non-zero pdf.

    Returns:
      float
    """
    raise NotImplementedError


class FixedWindow(LengthDistribution):
  """Represents a distribution with a fixed value over a window.

  The window is specified by a mimimum and maximum value.
  Further, a "right_tail_mass" can be specified. If given, that fraction of the
  total probability mass is added as an exponential distribution to the right
  side of the window. The exponential distribution is constructed such that the
  pdf is continuous at the maximum value.
  """

  def __init__(self, minimum, maximum,
               right_tail_mass):
    """Construct a FixedWindow.

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

  def min(self):
    return self.minimum

  def max(self):
    if self.right_tail_mass:
      return np.inf
    return self.maximum


class Gaussian(LengthDistribution):
  """Represents a trimmed Gaussian.
  """

  def __init__(self, mean, stdev, cutoff):
    if not stdev > 0:
      raise ValueError(f'stdev must be positive, got {stdev}')
    if not cutoff > 0:
      raise ValueError(f'cutoff must be positive, got {cutoff}')
    self._dist = scipy.stats.norm(loc=mean, scale=stdev)
    self._cutoff = cutoff
    self._normalizer = 1 - 2 * self._dist.cdf(mean - cutoff * stdev)

  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query

    Returns:
      pdf value
    """
    if (length < self._dist.mean() - self._cutoff * self._dist.std() or
        length > self._dist.mean() + self._cutoff * self._dist.std()):
      return 0.0
    return self._dist.pdf(length) / self._normalizer

  def min(self):
    return self._dist.mean() - self._cutoff * self._dist.std()

  def max(self):
    return self._dist.mean() + self._cutoff * self._dist.std()


class Empirical(LengthDistribution):
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
    """Construct Empirical.

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

    self._df['count'].fillna(0, inplace=True)
    self._df['count'] = interpolate_zeros(np.array(self._df['count']))

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
      Empirical
    """
    with open(filename) as f:
      df = pd.read_csv(f, header=None, names=['length', 'count'], dtype=float)

    return Empirical(df, right_tail_mass)

  @classmethod
  def from_sparse_dataframe(cls, df_input, right_tail_mass, sig_digits):
    """Creates Empirical from a sparse dataframe.

    "sparse" means that not every bucket is listed explictly. The main work
    in this function to to fill in the implicit values expected in the
    constructor.

    Args:
      df_input: pd.DataFrame with columns ['length_str', 'count']
      right_tail_mass: probability mass added past the right side of the buckets
        (see class documentation)
      sig_digits: number of significant digits after the decimal point

    Returns:
      Empirical
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

    return Empirical(df, right_tail_mass=right_tail_mass)

  @classmethod
  def from_arrays(cls, lengths, counts, right_tail_mass):
    """Creates Empirical from arrays.

    Args:
      lengths: sequence of values of the left edges of the buckets (same length
        as counts) sequence of left edge of length buckets
      counts: sequence of counts observed in each bucket
      right_tail_mass: probability mass added past the right side of the buckets
        (see class documentation)

    Returns:
      Empirical
    """
    return Empirical(
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

    result = self._df.iloc[idx - 1, self._df.columns.get_loc('pdf')]
    if math.isnan(result):
      return 0.0
    return result

  def min(self):
    return self._df['length'].min()

  def max(self):
    if self._right_tail_dist:
      return np.inf
    return self._df['length'].max() + self.bucket_size


class Mixture:
  """Represents a mixture of underlying LengthDistribution.

  Each given LengthDistribution is provided with a numeric weight. The weight
  can be on arbitrary scale; the outputs will be scaled by the sum of the
  weights.
  """
  def __init__(self):
    self._dists = []
    self._total_weight = 0.0

  def add(self, dist, weight):
    """Adds a new compoment to the mixture.

    Args:
      dist: LengthDistribution
      weight: weight strictly > 0
    """
    if weight <= 0.0:
      raise ValueError(f'Mixture: weight must be positive, got {weight}')
    self._dists.append( (dist, weight) )
    self._total_weight += weight

  def pdf(self, length):
    """Probability distribution function at given length.

    Args:
      length: length to query

    Returns:
      pdf value
    """
    if not self._dists:
      raise ValueError('Mixture.pdf called with empty components')
    return (sum([w * dist.pdf(length) for dist, w in self._dists])
            / self._total_weight)

  def min(self):
    return np.min([d.min() for (d, w) in self._dists])

  def max(self):
    return np.max([d.max() for (d, w) in self._dists])


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

  def has_key(self, key):
    """Returns True if `key` is in self._bond_type_map.

    Args:
      key: a bond type

    Returns:
      Boolean
    """
    return key in self._bond_type_map.keys()

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

  _ATOM_SPECIFICATION_MAP = {
      'C': [dataset_pb2.BondTopology.ATOM_C],
      'N': [dataset_pb2.BondTopology.ATOM_N],
      'O': [dataset_pb2.BondTopology.ATOM_O],
      'F': [dataset_pb2.BondTopology.ATOM_F],
      '*': [
          dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
          dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_F
      ],
  }
  _BOND_SPECIFICATION_MAP = {
      '-': [dataset_pb2.BondTopology.BOND_SINGLE],
      '=': [dataset_pb2.BondTopology.BOND_DOUBLE],
      '#': [dataset_pb2.BondTopology.BOND_TRIPLE],
      '.': [dataset_pb2.BondTopology.BOND_UNDEFINED],
      ':': [
          dataset_pb2.BondTopology.BOND_SINGLE,
          dataset_pb2.BondTopology.BOND_DOUBLE,
      ],
      '~': [
          dataset_pb2.BondTopology.BOND_SINGLE,
          dataset_pb2.BondTopology.BOND_DOUBLE,
          dataset_pb2.BondTopology.BOND_TRIPLE
      ],
  }

  def __init__(self):
    self._atom_pair_dict = {}

  def add(self, atom_a,
          atom_b,
          bond_type,
          dist):
    """Adds a distribution of the atom pair and bond type.

    Args:
      atom_a: dataset_pb2.AtomType
      atom_b: dataset_pb2.AtomType
      bond_type: dataset_pb2.BondType
      dist: float
    """
    atom_a = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a]
    atom_b = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b]
    if (atom_a, atom_b) not in self._atom_pair_dict:
      self._atom_pair_dict[(atom_a, atom_b)] = AtomPairLengthDistributions()
      # Just set the other order of atom_a, atom_b to the same object
      self._atom_pair_dict[(atom_b, atom_a)] = self._atom_pair_dict[(atom_a,
                                                                     atom_b)]
    self._atom_pair_dict[(atom_a, atom_b)].add(bond_type, dist)

  def add_from_files(self,
                     filestem,
                     unbonded_right_tail_mass,
                     include_nonbonded=True):
    """Adds distributions from a set of files.

    Files are expected to be named {filestem}.{atom_a}.{bond_type}.{atom_b}
    where
    * atom_a, atom_b: atomic numbers for H, C, N, O, F (smaller number first)
    * bond_type: {0, 1, 2, 3} for {unbonded, single, double, triple}

    Missing files are silently ignored.

    Contents are as expected by Empirical.from_file

    Args:
      filestem: prefix of files to load
      unbonded_right_tail_mass: right_tail_mass (as described in
        Empirical) for the unbonded cases.
      include_nonbonded: whether or not to include non-bonded data.
    """
    atomic_numbers = [1, 6, 7, 8, 9]

    if include_nonbonded:
      bond_types = [
          dataset_pb2.BondTopology.BOND_UNDEFINED,
          dataset_pb2.BondTopology.BOND_SINGLE,
          dataset_pb2.BondTopology.BOND_DOUBLE,
          dataset_pb2.BondTopology.BOND_TRIPLE,
      ]
    else:
      bond_types = [
          dataset_pb2.BondTopology.BOND_SINGLE,
          dataset_pb2.BondTopology.BOND_DOUBLE,
          dataset_pb2.BondTopology.BOND_TRIPLE,
      ]

    for (atom_a, atom_b), bond_type in itertools.product(
        itertools.combinations_with_replacement(atomic_numbers, 2), bond_types):
      fname = '{}.{}.{}.{}'.format(filestem, atom_a, int(bond_type), atom_b)

      if not os.path.exists(fname):
        logging.info('Skipping non existent file %s', fname)
        continue

      right_tail_mass = None
      if bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED:
        right_tail_mass = unbonded_right_tail_mass

      atom_a = ATOMIC_NUMBER_TO_ATYPE[atom_a]
      atom_b = ATOMIC_NUMBER_TO_ATYPE[atom_b]
      self.add(atom_a, atom_b, bond_type,
               Empirical.from_file(fname, right_tail_mass))

  def add_from_sparse_dataframe(self, df_input, unbonded_right_tail_mass,
                                sig_digits):
    """Adds distributions from a sparse dataframe.

    See sparse_dataframe_from_records for a description of the expected input
    format.

    Args:
      df_input: pd.DataFrame
      unbonded_right_tail_mass: right_tail_mass (as described in
        Empirical) for the unbonded cases.
      sig_digits: number of significant digits after the decimal point
    """
    avail_pairs = set(
        df_input.apply(
            lambda r: (r['atom_char_0'], r['atom_char_1'], r['bond_type']),
            axis=1))
    for atom_char_0, atom_char_1, bond_type in avail_pairs:
      atom_0 = smu_utils_lib.ATOM_CHAR_TO_ATOMIC_NUMBER[atom_char_0]
      atom_1 = smu_utils_lib.ATOM_CHAR_TO_ATOMIC_NUMBER[atom_char_1]
      df = df_input[(df_input['atom_char_0'] == atom_char_0)
                    & (df_input['atom_char_1'] == atom_char_1) &
                    (df_input['bond_type'] == bond_type)]

      right_tail_mass = None
      if bond_type == dataset_pb2.BondTopology.BOND_UNDEFINED:
        right_tail_mass = unbonded_right_tail_mass

      atom_0 = ATOMIC_NUMBER_TO_ATYPE[atom_0]
      atom_1 = ATOMIC_NUMBER_TO_ATYPE[atom_1]
      self.add(
          atom_0, atom_1, bond_type,
          Empirical.from_sparse_dataframe(
              df, right_tail_mass, sig_digits))

  def add_from_sparse_dataframe_file(self, filename,
                                     unbonded_right_tail_mass, sig_digits):
    """Adds distribution from a sparse dataframe in a csv file.

    See sparse_dataframe_from_records for a description of the expected input
    format.

    Args:
      filename: string, file to read
      unbonded_right_tail_mass: right_tail_mass (as described in
        Empirical) for the unbonded cases.
      sig_digits: number of significant digits after the decimal point
    """
    with open(filename, 'r') as infile:
      df = pd.read_csv(infile, dtype={'length_str': str})
    self.add_from_sparse_dataframe(df, unbonded_right_tail_mass, sig_digits)

  def _triples_from_atom_bond_atom_spec(self, spec):
    if len(spec) != 3:
      raise BondLengthParseError(spec)

    atoms_a = self._ATOM_SPECIFICATION_MAP[spec[0]]
    bonds = self._BOND_SPECIFICATION_MAP[spec[1]]
    atoms_b = self._ATOM_SPECIFICATION_MAP[spec[2]]
    yield from itertools.product(atoms_a, atoms_b, bonds)

  def add_from_gaussians_file(self, filename, cutoff):
    """Adds distribution by reading specs of Guassians.

    The original intention is to take a specially formatted version of the table
    from
    Allen, F. H. et al. Tables of bond lengths determined by X-ray and neutron
    diffraction. Part 1. Bond lengths in organic compounds.
    J. Chem. Soc. Perkin Trans. 2 S1–S19 (1987)

    The file shoudl be a csv with at least the columns:
    * Bond: A bond specification like C=C (like add_from_string_spec) (or n/a)
    * d: mean
    * sigma: standard deviation
    * n: number of observations of this type

    Creates a Mixture distribution for every atom/pair/bond combo.

    Args:
      filename: file to read
      cutoff: cutoff passed to Gaussian
    """
    with open(filename, encoding='utf-8-sig') as f:
      reader = csv.DictReader(f)
      for row in reader:
        if row['Bond'] == 'n/a' or not row['d']:
          continue
        gaussian = Gaussian(float(row['d']), float(row['sigma']), cutoff)
        for atom_a, atom_b, bond in self._triples_from_atom_bond_atom_spec(row['Bond']):
          try:
            atom_a_num = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a]
            atom_b_num = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b]
            mix_dist = self._atom_pair_dict[(atom_a_num, atom_b_num)][bond]
          except KeyError:
            mix_dist = Mixture()
            self.add(atom_a, atom_b, bond, mix_dist)
          mix_dist.add(gaussian, int(row['n']))


  def add_from_string_spec(self, spec_string):
    """Adds entries from a compact string specifiction

    spec_string is a comma separated list of terms of form
    XYX:N-N
    where
    * X is an atom type 'CNOF*'
    * Y is a bond type '-=#.~'
    * N is a possibly empty floating point number

    Args:
      spec_string: string
    """
    if not spec_string:
      return

    terms = [x.strip() for x in spec_string.split(',')]
    for term in terms:
      try:
        if term[3] != ':':
          raise BondLengthParseError(term)
        min_str, max_str = term[4:].split('-')
        if min_str:
          min_val = float(min_str)
        else:
          min_val = 0
        if max_str:
          max_val = float(max_str)
          right_tail_mass = None
        else:
          # These numbers are pretty arbitrary
          max_val = min_val + 0.1
          right_tail_mass = 0.9

        for atom_a, atom_b, bond in self._triples_from_atom_bond_atom_spec(term[:3]):
          self.add(
              atom_a, atom_b, bond,
              FixedWindow(
                  min_val, max_val, right_tail_mass))

      except (KeyError, IndexError, ValueError) as an_exception:
        raise BondLengthParseError(term) from an_exception


  def __getitem__(self, atom_types):
    """Gets the underlying AtomPairLengthDistribution."""
    atom_a, atom_b = atom_types
    return self._atom_pair_dict[(
        smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a],
        smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b])]

  def pdf_length_given_type(self, atom_a,
                            atom_b,
                            bond_type,
                            length):
    """p(length | atom_a, atom_b, bond_type)."""
    atom_a = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a]
    atom_b = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b]

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
    atom_a = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_a]
    atom_b = smu_utils_lib.ATOM_TYPE_TO_ATOMIC_NUMBER[atom_b]
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


def make_fake_empiricals():
  """Testing utility to make an AllAtomPairLengthDistributions

  Every atom pair and bond type is an empirical dstribution between 1 and 2

  Returns:
    AllAtomPairLengthDistributions
  """
  bond_lengths = AllAtomPairLengthDistributions()
  for atom_a, atom_b in itertools.combinations_with_replacement(
      [dataset_pb2.BondTopology.ATOM_C,
       dataset_pb2.BondTopology.ATOM_N,
       dataset_pb2.BondTopology.ATOM_O,
       dataset_pb2.BondTopology.ATOM_F], 2):
    bond_lengths.add(
      atom_a, atom_b, dataset_pb2.BondTopology.BOND_UNDEFINED,
      Empirical.from_arrays(
        np.arange(1, 2, 0.1), [1] * 10, 0))
    for bond_type in [dataset_pb2.BondTopology.BOND_SINGLE,
                      dataset_pb2.BondTopology.BOND_DOUBLE,
                      dataset_pb2.BondTopology.BOND_TRIPLE]:
      bond_lengths.add(
        atom_a, atom_b, bond_type,
        Empirical.from_arrays(
          np.arange(1, 2, 0.1), [1] * 10, 0))
  return bond_lengths


def is_valid_bond(atom_a, atom_b, bond):
  """Whether this bond type can exist in SMU.

  Note that for N and O, we assume the charge states can change.

  Args:
    atom_a: dataset_pb2.AtomType
    atom_b: dataset_pb2.AtomType
    bond: dataset_pb2.BondType

  Returns:
    bool
  """
  if bond == dataset_pb2.BondTopology.BOND_UNDEFINED:
    return True
  bond_order = int(bond)
  return (bond_order <= smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS_ANY_FORM[atom_a] and
          bond_order <= smu_utils_lib.ATOM_TYPE_TO_MAX_BONDS_ANY_FORM[atom_b])


_COVALENT_RADIUS = {
  dataset_pb2.BondTopology.ATOM_C: 0.68,
  dataset_pb2.BondTopology.ATOM_N: 0.68,
  dataset_pb2.BondTopology.ATOM_O: 0.68,
  dataset_pb2.BondTopology.ATOM_F: 0.64,
}

_COVALENT_RADII_MIN = 0.8
_COVALENT_RADII_TOLERANCE = 0.4
_COVALENT_RADII_UNBONDED_OVERLAP = 0.2


def make_covalent_radii_dists():
  """Makes distributions based on covalent radii.

  This is a commonly used method to identify if atoms are bonded from geometry.
  We are folllowing

  Meng, E. C. & Lewis, R. A. Determination of molecular topology and
  atomic hybridization states from heavy atom
  coordinates. J. Comput. Chem. 12, 891–898 (1991)

  The approach is to allow bonds of any order with distances > 0.8A and <
  sum of covalent radii + tolerance (0.4A)

  We additionally allow some overlap of bonded and unbonded by allowing
  an overlap of 0.2A

  Returns:
    AllAtomPairLengthDistributions
  """

  dists = AllAtomPairLengthDistributions()
  for atom_a, atom_b in itertools.combinations_with_replacement(
      [dataset_pb2.BondTopology.ATOM_C,
       dataset_pb2.BondTopology.ATOM_N,
       dataset_pb2.BondTopology.ATOM_O,
       dataset_pb2.BondTopology.ATOM_F], 2):
    max_dist = (_COVALENT_RADIUS[atom_a] +
                _COVALENT_RADIUS[atom_b] +
                _COVALENT_RADII_TOLERANCE)
    for bond in [dataset_pb2.BondTopology.BOND_SINGLE,
                 dataset_pb2.BondTopology.BOND_DOUBLE,
                 dataset_pb2.BondTopology.BOND_TRIPLE]:
      if is_valid_bond(atom_a, atom_b, bond):
        dists.add(atom_a, atom_b, bond,
                  FixedWindow(_COVALENT_RADII_MIN, max_dist, None))

    dists.add(atom_a, atom_b, dataset_pb2.BondTopology.BOND_UNDEFINED,
              FixedWindow(max_dist - _COVALENT_RADII_UNBONDED_OVERLAP, max_dist,
                          STANDARD_UNBONDED_RIGHT_TAIL_MASS))

  return dists


# This table is generated by separate code using the Guassian and Mixture
# classes above and then copied here for speed and ease of use.
_ALLEN_MIN_MAX = {
  (dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.BOND_SINGLE): (1.316, 1.663),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.BOND_DOUBLE): (1.218, 1.477),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.BOND_TRIPLE): (1.139, 1.225),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_SINGLE): (1.270, 1.621),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_DOUBLE): (1.239, 1.402),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_TRIPLE): (1.106, 1.191),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.BOND_SINGLE): (1.236, 1.524),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.BOND_DOUBLE): (1.148, 1.301),
(dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_F, dataset_pb2.BondTopology.BOND_SINGLE): (1.277, 1.485),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_SINGLE): (1.132, 1.517),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_DOUBLE): (1.079, 1.401),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.BOND_TRIPLE): (1.079, 1.169),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.BOND_SINGLE): (1.176, 1.499),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.BOND_DOUBLE): (1.176, 1.299),
(dataset_pb2.BondTopology.ATOM_N, dataset_pb2.BondTopology.ATOM_F, dataset_pb2.BondTopology.BOND_SINGLE): (1.358, 1.454),
(dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.BOND_SINGLE): (1.433, 1.511),
  }

def make_allen_et_al_dists():
  dists = AllAtomPairLengthDistributions()

  for atom_a, atom_b in itertools.combinations_with_replacement(
      [dataset_pb2.BondTopology.ATOM_C,
       dataset_pb2.BondTopology.ATOM_N,
       dataset_pb2.BondTopology.ATOM_O,
       dataset_pb2.BondTopology.ATOM_F], 2):
    max_dist = -np.inf
    for bond in [dataset_pb2.BondTopology.BOND_SINGLE,
                 dataset_pb2.BondTopology.BOND_DOUBLE,
                 dataset_pb2.BondTopology.BOND_TRIPLE]:
      if is_valid_bond(atom_a, atom_b, bond):
        try:
          mn, mx = _ALLEN_MIN_MAX[(atom_a, atom_b, bond)]
          # Add this slop so that anything that rounds to the published distances
          # wil be considered valid.
          mn -= .0005
          mx += .0005
        except KeyError:
          # If Allen et al is missing a pair, fill in the covalent radii case
          mn = _COVALENT_RADII_MIN
          mx =  (_COVALENT_RADIUS[atom_a] +
                 _COVALENT_RADIUS[atom_b] +
                 _COVALENT_RADII_TOLERANCE)
        dists.add(atom_a, atom_b, bond,
                  FixedWindow(mn, mx, None))
        max_dist = max(max_dist, mx)

    assert np.isfinite(max_dist)
    # We are creating unbonded distances that don't overlap at all with the
    # defined ranges.
    dists.add(atom_a, atom_b, dataset_pb2.BondTopology.BOND_UNDEFINED,
              FixedWindow(max_dist, max_dist + 0.1,
                          STANDARD_UNBONDED_RIGHT_TAIL_MASS))


  return dists
