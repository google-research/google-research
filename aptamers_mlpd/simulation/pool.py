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

"""Library for generating and sampling from aptamer pools.

Aptamers are generated as pandas.DataFrame objects with columns ['sequences',
'target_affinity', 'serum_affinity'].
"""

import numpy
import pandas
import six

from ..preprocess import utils as aptitude_utils
from ..simulation import utils


APTAMER_FIELDS = ['sequence', 'target_affinity', 'serum_affinity']


def random_sequences(size, sequence_length=40, bases='ATGC', random_seed=None):
  """Generate an array of random sequences.

  Args:
    size: optional integer number of sequences to generate.
    sequence_length: integer number of bases to include in the generated
        sequence.
    bases: optional iterable of characters out of which to construct the
        sequences. Defaults to 'ATGC'.
    random_seed: optional integer.

  Returns:
    numpy.ndarray with string dtype with the given number of sequences.
  """
  rs = numpy.random.RandomState(random_seed)

  sequence_dtype = 'S%i' % sequence_length
  bases = sorted(six.ensure_str(base) for base in bases)
  sequence_chars = rs.choice(bases, size=(size * sequence_length))
  return sequence_chars.astype(bytes).view(sequence_dtype).astype(str)


def _count_substrings(substrings, sequence):
  # only counts non-overlapping substrings
  return [sequence.count(substring) for substring in substrings]


def substring_binding_energy(sequences, target_features, serum_features):
  """Calculate binding energies for aptamer sequences based on substrings.

  Args:
    sequences: iterable of strings giving sequences to score.
    target_features: Dict[str, float] indicating the contribution to target
      binding energy for each occurrence of a given substring.
    serum_features: Dict[str, float] indicating the contribution to serum
      binding energy for each occurrence of a given substring.

  Returns:
    Broadcastable to ndarray with shape=(2, len(sequences)) giving relative
    target and serum binding energies (in units of ln(10) * kT).
  """
  if target_features is None:
    target_features = {}
  if serum_features is None:
    serum_features = {}

  substrings = set(target_features) | set(serum_features)
  if substrings:
    feature_scores = [[target_features.get(substring, 0),
                       serum_features.get(substring, 0)]
                      for substring in substrings]
    counts = numpy.array([_count_substrings(substrings, seq)
                          for seq in sequences])
    binding_energies = counts.dot(feature_scores).T
  else:
    binding_energies = 0

  return binding_energies


def random_aptamers(size, sequence_length=40, bases='ATGC',
                    binding_energy_correlation=0, target_features=None,
                    serum_features=None, binding_noise_scale=1,
                    random_seed=None):
  r"""Generate aptamers with random sequences and affinities.

  We model affinity by adding up many random or deterministic contributions to
  the binding free energy $\delta G$, which is related to the equilibrium
  dissociation constant $k_d$ by:

    \delta G = k_B T ln(k_d)

  where $k_B$ is Boltmann's constant and $T$ is the temperature (Eq (20) in
  the reference below). This results in roughly log-normal distributions for
  the dissociation constants (binding affinities).

  Args:
    size: integer number of aptamers to generate.
    sequence_length: optional integer number of bases to include in the
      generated sequence.
    bases: optional iterable of characters out of which to construct the
      sequences. Defaults to 'ATGC'.
    binding_energy_correlation: optional number indicating the desired
      correlation between target affinity and serum affinity.
    target_features: Dict[str, float] indicating the contribution to target
      binding energy for each occurrence of a given substring.
    serum_features: Dict[str, float] indicating the contribution to serum
      binding energy for each occurrence of a given substring.
    binding_noise_scale: float indicating the standard deviation for the
      gaussian contribution to the binding affinity. Must be non-negative.
    random_seed: optional integer.

  Returns:
    pandas.DataFrame of length `size` with columns ['sequence',
    'target_affinity', 'serum_affinity'] and dtypes [object, float, float].

  Raises:
    ValueError: if input is invalid.

  Reference:
    "Influence of Target Concentration and Background Binding on In Vitro
    Selection of Affinity Reagents"
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0043940
  """
  rs = numpy.random.RandomState(random_seed)

  sequences = random_sequences(size, sequence_length, bases,
                               random_seed=rs.randint(utils.RANDOM_SEED_MAX))

  binding_energy = numpy.zeros((2, size), dtype=float)

  if binding_noise_scale < 0:
    raise ValueError('binding_noise_scale must be non-negative')

  # it's standard to assume a normal distribution for binding energies, per
  # the paper cited in the docstring
  if binding_noise_scale > 0:
    if binding_energy_correlation != 0:
      mean = [0, 0]
      covariance = [[1, binding_energy_correlation],
                    [binding_energy_correlation, 1]]
      random_energy = rs.multivariate_normal(mean, covariance, size=size).T
    else:
      random_energy = rs.normal(size=(2, size))
    binding_energy += binding_noise_scale * random_energy

  if target_features or serum_features:
    substring_energy = substring_binding_energy(sequences, target_features,
                                                serum_features)
    binding_energy += substring_energy

  # Because these synthetic binding energies are in arbitrary units, for
  # convenience we use base 10 for calculating affinities. If/when we switch
  # to realistic units, we should probably use e instead.
  target_affinity, serum_affinity = 10 ** binding_energy

  aptamers = pandas.DataFrame.from_dict(
      dict(
          list(
              zip(APTAMER_FIELDS,
                  [sequences, target_affinity, serum_affinity]))))
  return aptamers


class BaseAptamerPool:
  """Abstract base class for aptamer selection pools."""

  def sample(self, size=None, random_seed=None):
    """Return a DataFrame of aptamers sampled from this pool with replacement.

    Args:
      size: integer number of aptamers sample.
      random_seed: optional integer.

    Returns:
      pandas.DataFrame of length `size` with columns ['sequences',
      'target_affinity', 'serum_affinity'] and dtypes [object, float, float].
    """
    raise NotImplementedError

  def __eq__(self, other):
    raise TypeError('equality not defined for generic aptamer pools')

  def __ne__(self, other):
    return not self == other

  def to_counts(self, depth, random_seed=None):
    """Sample from this pool to create a count table.

    Args:
      depth: number of total sequences to read.
      random_seed: optional integer.

    Returns:
      pandas.DataFrame with columns ['sequence', 'target_affinity',
      'serum_affinity', 'count'] and dtypes [object, float, float, int].
    """
    aptamers = self.sample(depth, random_seed)
    counts = (aptamers
              .sequence
              .value_counts()
              .rename('count')
              .rename_axis('sequence')
              .reset_index())
    return aptamers.drop_duplicates().merge(counts)

  def to_fastq(self, forward_path, reverse_path, depth, batch_size=100000,
               quality_char=aptitude_utils.FASTQ_HIGHEST_QUALITY,
               random_seed=None):
    """Write out reads from this pool into a pair of FASTQ files.

    Args:
      forward_path: path to use for writing the forward read.
      reverse_path: path to use for writing the reverse read.
      depth: number of total sequences to read.
      batch_size: optional integer indicating the number of sequences to write
        at once.
      quality_char: optional character indicating the FASTQ quality to write.
      random_seed: optional integer.
    """
    seed_gen = utils.random_seed_stream(random_seed)

    def generate_reads():
      # Yield a stream of aptitude_utils.Read objects describing each simulated
      # read from the high throughput sequencer.
      remaining_reads = depth
      while remaining_reads > 0:
        # We use batches for higher performance than sampling one at the time
        # and lower memory requirements than sampling all at once.
        size = min(batch_size, remaining_reads)
        remaining_reads -= size
        aptamers = self.sample(size, random_seed=next(seed_gen))
        for aptamer in aptitude_utils.iternamedtuples(aptamers):
          title = ('target_affinity=%.3e,serum_affinity=%.3e'
                   % (aptamer.target_affinity, aptamer.serum_affinity))
          title_aux = ''
          quality = quality_char * len(aptamer.sequence)
          read = aptitude_utils.Read(
              title, title_aux, aptamer.sequence, quality)
          yield read

    aptitude_utils.write_fastq(forward_path, reverse_path, generate_reads())


class RandomAptamerPool(BaseAptamerPool):
  """Pool of random aptamer sequences.

  Attributes:
    sequence_length: integer number of bases to include in the generated
      sequence.
    bases: sorted tuple of characters out of which to construct the sequences.
    binding_energy_correlation: float indicating the correlation between target
      affinity and serum affinity.
    target_features: Dict[str, float] indicating the contribution to target
      binding energy for each occurrence of a given substring.
    serum_features: Dict[str, float] indicating the contribution to serum
      binding energy for each occurrence of a given substring.
    binding_noise_scale: float indicating the standard deviation for the
      gaussian contribution to the binding affinity.
  """

  def __init__(self, sequence_length=40, bases='ATGC',
               binding_energy_correlation=0, target_features=None,
               serum_features=None, binding_noise_scale=1):
    """Initialize a random Aptamer pool.

    Args:
      sequence_length: optional integer number of bases to include in the
        generated sequence.
      bases: optional iterable of characters out of which to construct the
        sequences. Defaults to 'ATGC'.
      binding_energy_correlation: optional number indicating the desired
        correlation between target affinity and serum affinity.
      target_features: Dict[str, float] indicating the contribution to target
        binding energy for each occurrence of a given substring.
      serum_features: Dict[str, float] indicating the contribution to serum
        binding energy for each occurrence of a given substring.
      binding_noise_scale: float indicating the standard deviation for the
        gaussian contribution to the binding affinity.
    """
    self.sequence_length = sequence_length
    self.bases = tuple(sorted(bases))
    self.binding_energy_correlation = binding_energy_correlation
    self.target_features = target_features
    self.serum_features = serum_features
    self.binding_noise_scale = binding_noise_scale

  def sample(self, size, random_seed=None):
    """See base class."""
    return random_aptamers(size, self.sequence_length, self.bases,
                           self.binding_energy_correlation,
                           self.target_features, self.serum_features,
                           self.binding_noise_scale, random_seed=random_seed)

  def __eq__(self, other):
    try:
      return (self.sequence_length == other.sequence_length
              and self.bases == other.bases
              and (self.binding_energy_correlation ==
                   other.binding_energy_correlation)
              and self.target_features == other.target_features
              and self.serum_features == other.serum_features
              and self.binding_noise_scale == other.binding_noise_scale)
    except AttributeError:
      return False


class SelectedAptamerPool(BaseAptamerPool):
  """Pool of selected DNA sequences.

  Attributes:
    aptamers: pandas.DataFrame with columns ['sequence', 'target_affinity',
      'serum_affinity'].
    probabilities: array the same length as aptamers giving the fraction of the
      pool that consists of each aptamer.
  """

  def __init__(self, aptamers, multiplicities=None):
    """Initialize a SelectedAptamerPool.

    Args:
      aptamers: pandas.DataFrame with columns ['sequence', 'target_affinity',
        'serum_affinity'].
      multiplicities: array the same length as aptamers giving the relative
        prevalance of each provided aptamer in the pool. If not provided,
        aptamers are assumed to have equal prevalence.

    Raises:
      ValueError: for invalid input.
    """
    if list(aptamers.columns) != APTAMER_FIELDS:
      raise ValueError('aptamers argument has invalid columns')
    self.aptamers = aptamers

    if multiplicities is None:
      multiplicities = numpy.ones(len(aptamers))
    p = numpy.array(multiplicities, dtype=float)
    if len(p) != len(aptamers):
      raise ValueError('multiplicies and aptamers must have the same length')
    p /= p.sum()
    self.probabilities = p

  def sample(self, size, random_seed=None):
    """See base class."""
    if self.aptamers.size == 0:
      # Sample doesn't work if there's nothing to sample from. If self.aptamers
      # is empty, simply return the empty DataFrame.
      return self.aptamers
    else:
      rs = numpy.random.RandomState(random_seed)
      samples = self.aptamers.sample(n=size, replace=True,
                                     weights=self.probabilities,
                                     random_state=rs)
      return samples.reset_index(drop=True)

  def __eq__(self, other):
    try:
      return (self.aptamers.equals(other.aptamers)
              and (self.probabilities == other.probabilities).all())
    except AttributeError:
      return False
