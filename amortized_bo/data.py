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

"""Population class for keeping track of structures and rewards."""

import collections
import itertools
import typing

import attr
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from amortized_bo import domains
from amortized_bo import utils

# DatasetSample defines the structure of samples in tf.data.Datasets for
# pre-training solvers, whereas Samples (see below) defines the structure of
# samples in a Population object.
DatasetSample = collections.namedtuple('DatasetSample', ['structure', 'reward'])


def dataset_to_population(population_or_tf_dataset):
  """Converts a TF dataset to a Population if it is not already a Population."""
  if isinstance(population_or_tf_dataset, Population):
    return population_or_tf_dataset
  else:
    return Population.from_dataset(population_or_tf_dataset)


def serialize_structure(structure):
  """Converts a structure to a string."""
  structure = np.asarray(structure)
  dim = len(structure.shape)
  if dim != 1:
    raise NotImplementedError(f'`structure` must be 1d but is {dim}d!')
  return domains.SEP_TOKEN.join(str(token) for token in structure)


def serialize_structures(structures, **kwargs):
  """Converts a list of structures to a list of strings."""
  return [serialize_structure(structure, **kwargs)
          for structure in structures]


def deserialize_structure(serialized_structure, dtype=np.int32):
  """Converts a string to a structure.

  Args:
    serialized_structure: A structure produced by `serialize_structure`.
    dtype: The data type of the output numpy array.

  Returns:
    A numpy array with `dtype`.
  """
  return np.asarray(
      [token for token in serialized_structure.split(domains.SEP_TOKEN)],
      dtype=dtype)


def deserialize_structures(structures, **kwargs):
  """Converts a list of strings to a list of structures.

  Args:
    structures: A list of strings produced by `serialize_structures`.
    **kwargs: Named arguments passed to `deserialize_structure`.

  Returns:
    A list of numpy array.
  """
  return [deserialize_structure(structure, **kwargs)
          for structure in structures]


def serialize_population_frame(frame, inplace=False, domain=None):
  """Serializes a population `pd.DataFrame` for representing it as plain text.

  Args:
    frame: A `pd.DataFrame` produced by `Population.to_frame`.
    inplace: Whether to serialize `frame` inplace instead of creating a copy.
    domain: An optional domain for decoding structures. If provided, will
      add a column `decoded_structure` with the serialized decoded structures.

  Returns:
    A `pd.DataFrame` with serialized structures.
  """
  if not inplace:
    frame = frame.copy()
  if domain:
    frame['decoded_structure'] = serialize_structures(
        domain.decode(frame['structure'], as_str=False))
  frame['structure'] = serialize_structures(frame['structure'])
  return frame


def deserialize_population_frame(frame, inplace=False):
  """Deserializes a population `pd.DataFrame` from plain text.

  Args:
    frame: A `pd.DataFrame` produced by `serialize_population_frame`.
    inplace: Whether to deserialize `frame` inplace instead of creating a copy.

  Returns:
    A `pd.DataFrame` with deserialized structures.
  """
  if not inplace:
    frame = frame.copy()
  frame['structure'] = deserialize_structures(frame['structure'])
  if 'decoded_structure' in frame.columns:
    frame['decoded_structure'] = deserialize_structures(
        frame['decoded_structure'], dtype=np.str)
  return frame


def population_frame_to_csv(frame, path_or_buf=None, domain=None, index=False,
                            **kwargs):
  """Converts a population `pd.DataFrame` to a csv table.

  Args:
    frame: A `pd.DataFrame` produced by `Population.to_frame`.
    path_or_buf: File path or object. If `None`, the result is returned as a
      string. Otherwise write the csv table to that file.
    domain: A optional domain for decoding structures.
    index: Whether to store the index of `frame`.
    **kwargs: Named arguments passed to `frame.to_csv`.

  Returns:
    If `path_or_buf` is `None`, returns the resulting csv format as a
    string. Otherwise returns `None`.
  """
  if frame.empty:
    raise ValueError('Cannot write empty population frame to CSV file!')
  frame = serialize_population_frame(frame, domain=domain)
  return frame.to_csv(path_or_buf, index=index, **kwargs)


def population_frame_from_csv(path_or_buf, **kwargs):
  """Reads a population `pd.DataFrame` from a file.

  Args:
    path_or_buf: A string path of file buffer.
    **kwargs: Named arguments passed to `pd.read_csv`.

  Returns:
    A `pd.DataFrame`.
  """
  frame = pd.read_csv(path_or_buf, dtype={'metadata': object}, **kwargs)
  frame = deserialize_population_frame(frame)
  return frame


def subtract_mean_batch_reward(population):
  """Returns new Population where each batch has mean-zero rewards."""
  df = population.to_frame()
  mean_dict = df.groupby('batch_index').reward.mean().to_dict()

  def reward_for_sample(sample):
    return sample.reward - mean_dict[sample.batch_index]

  shifted_samples = [
      sample.copy(reward=reward_for_sample(sample)) for sample in population
  ]
  return Population(shifted_samples)


def _to_immutable_array(array):
  to_return = np.array(array)
  to_return.setflags(write=False)
  return to_return


class _NumericConverter(object):
  """Helper class for converting values to a numeric data type."""

  def __init__(self, dtype, min_value=None, max_value=None):
    self._dtype = dtype
    self._min_value = min_value
    self._max_value = max_value

  def __call__(self, value):
    """Validates and converts `value` to `self._dtype`."""
    if value is None:
      return value
    if not np.isreal(value):
      raise TypeError('%s is not numeric!' % value)
    value = self._dtype(value)
    if self._min_value is not None and value < self._min_value:
      raise TypeError('%f < %f' % (value, self._min_value))
    if self._max_value is not None and value > self._max_value:
      raise TypeError('%f > %f' % (value, self._max_value))
    return value


@attr.s(
    frozen=True,  # Make it immutable.
    slots=True,  # Improve memory overhead.
    eq=False,  # Because we override __eq__.
)
class Sample(object):
  """Immutable container for a structure, reward, and additional data.

  Attributes:
    key: (str) A unique identifier of the sample. If not provided, will create
      a unique identifier using `utils.create_unique_id`.
    structure: (np.ndarray) The structure.
    reward: (float, optional) The reward.
    batch_index: (int, optional) The batch index within the population.
    infeasible: Whether the sample was marked as infeasible by the evaluator.
    metadata: (any type, optional) Additional meta-data.
  """
  structure: np.ndarray = attr.ib(factory=np.array,
                                  converter=_to_immutable_array)  # pytype: disable=wrong-arg-types  # attr-stubs
  reward: float = attr.ib(
      converter=_NumericConverter(float),
      default=None)
  batch_index: int = attr.ib(
      converter=_NumericConverter(int, min_value=0),
      default=None)
  infeasible: bool = attr.ib(default=False,
                             validator=attr.validators.instance_of(bool))
  key: str = attr.ib(
      factory=utils.create_unique_id,
      validator=attr.validators.optional(attr.validators.instance_of(str)))
  metadata: typing.Dict[str, typing.Any] = attr.ib(default=None)

  def __eq__(self, other):
    """Compares samples irrespective of their key."""
    return (self.reward == other.reward and
            self.batch_index == other.batch_index and
            self.infeasible == other.infeasible and
            self.metadata == other.metadata and
            np.array_equal(self.structure, other.structure))

  def equal(self, other):
    """Compares samples including their key."""
    return self == other and self.key == other.key

  def to_tfexample(self):
    """Converts a Sample to a tf.Example."""
    features = dict(
        structure=tf.train.Feature(
            int64_list=tf.train.Int64List(value=self.structure)),
        reward=tf.train.Feature(
            float_list=tf.train.FloatList(value=[self.reward])),
        batch_index=tf.train.Feature(
            int64_list=tf.train.Int64List(value=[self.batch_index])))
    return tf.train.Example(features=tf.train.Features(feature=features))

  def to_dict(self, dict_factory=collections.OrderedDict):
    """Returns a dictionary from field names to values.

    Args:
      dict_factory: A class that implements a dict factory method.

    Returns:
      A dict of type `dict_factory`
    """
    return attr.asdict(self, dict_factory=dict_factory)

  def copy(self, new_key=True, **kwargs):
    """Returns a copy of this Sample with values overridden by **kwargs."""
    if new_key:
      kwargs['key'] = utils.create_unique_id()
    return attr.evolve(self, **kwargs)


def samples_from_arrays(structures, rewards=None, batch_index=None,
                        metadata=None):
  """Makes a generator of Samples from fields.

  Args:
    structures: Iterable of structures (1-D np array or list).
    rewards: Iterable of float rewards. If None, the corresponding Samples are
      given each given a reward of None.
    batch_index: Either an int, in which case all Samples created by this
      function will be given this batch_index or an iterable of ints for each
      corresponding structure.
    metadata: Metadata to store in the Sample.

  Yields:
    A generator of Samples
  """
  structures = utils.to_array(structures)

  if metadata is None:
    metadata = [None] * len(structures)

  if rewards is None:
    rewards = [None] * len(structures)
  else:
    rewards = utils.to_array(rewards)

  if len(structures) != len(rewards):
    raise ValueError(
        'Structures and rewards must be same length. Are %s and %s' %
        (len(structures), len(rewards)))
  if len(metadata) != len(rewards):
    raise ValueError('Metadata and rewards must be same length. Are %s and %s' %
                     (len(metadata), len(rewards)))

  if batch_index is None:
    batch_index = 0
  if isinstance(batch_index, int):
    batch_index = [batch_index] * len(structures)

  for structure, reward, batch_index, meta in zip(structures, rewards,
                                                  batch_index, metadata):
    yield Sample(
        structure=structure,
        reward=reward,
        batch_index=batch_index,
        metadata=meta)


def parse_tf_example(example_proto):
  """Converts tf.Example proto to dict of Tensors.

  Args:
    example_proto: A raw tf.Example proto.
  Returns:
    A dict of Tensors with fields structure, reward, and batch_index.
  """

  feature_description = dict(
      structure=tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
      reward=tf.FixedLenFeature([1], tf.float32),
      batch_index=tf.FixedLenFeature([1], tf.int64))

  return tf.io.parse_single_example(
      serialized=example_proto, features=feature_description)


class Population(object):
  """Data structure for storing Samples."""

  def __init__(self, samples=None):
    """Construct a Population.

    Args:
      samples: An iterable of Samples
    """
    self._samples = collections.OrderedDict()
    self._batch_to_sample_keys = collections.defaultdict(list)

    if samples is not None:
      self.add_samples(samples)

  def __str__(self):
    if self.empty:
      return '<Population of length 0>'
    return ('<Population of length %s with best %s>' %
            (len(self), dict(self.best().to_dict())))

  def __len__(self):
    return len(self._samples)

  def __iter__(self):
    return self._samples.values().__iter__()

  def __eq__(self, other):
    if not isinstance(other, Population):
      raise ValueError('Cannot compare equality with an object of '
                       'type %s' % (str(type(other))))

    return len(self) == len(other) and all(
        s1 == s2 for s1, s2 in zip(self, other))

  def __add__(self, other):
    """Adds samples to this population and returns a new Population."""
    return Population(itertools.chain(self, other))

  def __getitem__(self, key_or_index):
    if isinstance(key_or_index, str):
      return self._samples[key_or_index]
    else:
      return list(self._samples.values())[key_or_index]

  def __contains__(self, key):
    if isinstance(key, Sample):
      key = key.key
    return key in self._samples

  def copy(self):
    """Copies the population."""
    return Population(self.samples)

  @property
  def samples(self):
    """Returns the population Samples as a list."""
    return list(self._samples.values())

  def add_sample(self, sample):
    """Add copy of sample to population."""
    if sample.key in self._samples:
      raise ValueError('Sample with key %s already exists in the population!')
    self._samples[sample.key] = sample
    batch_idx = sample.batch_index
    self._batch_to_sample_keys[batch_idx].append(sample.key)

  def add_samples(self, samples):
    """Convenience method for adding multiple samples."""
    for sample in samples:
      self.add_sample(sample)

  @property
  def empty(self):
    return not self

  @property
  def batch_indices(self):
    """Returns a sorted list of unique batch indices."""
    return sorted(self._batch_to_sample_keys)

  @property
  def max_batch_index(self):
    """Returns the maximum batch index."""
    if self.empty:
      raise ValueError('Population empty!')
    return self.batch_indices[-1]

  @property
  def current_batch_index(self):
    """Return the maximum batch index or -1 if the population is empty."""
    return -1 if self.empty else self.max_batch_index

  def get_batches(self, batch_indices, exclude=False, validate=True):
    """"Extracts certain batches from the population.

    Ignores batches that do not exist in the population. To validate if a
    certain batch exists use `batch_index in population.batch_indices`.

    Args:
      batch_indices: An integer, iterable of integers, or `slice` object
        for selecting batches.
      exclude: If true, will return all batches but the ones that are selected.
      validate: Whether to raise an exception if a batch index is invalid
        instead of ignoring.

    Returns:
      A `Population` with the selected batches.
    """
    batch_indices = utils.get_indices(
        self.batch_indices, batch_indices, exclude=exclude, validate=validate)
    sample_idxs = []
    for batch_index in batch_indices:
      sample_idxs.extend(self._batch_to_sample_keys.get(batch_index, []))
    samples = [self[idx] for idx in sample_idxs]
    return Population(samples)

  def get_batch(self, *args, **kwargs):
    return self.get_batches(*args, **kwargs)

  def get_last_batch(self, **kwargs):
    """Returns the last batch from the population."""
    return self.get_batch(-1, **kwargs)

  def get_last_batches(self, n=1, **kwargs):
    """Selects the last n batches."""
    return self.get_batches(self.batch_indices[-n:], **kwargs)

  def to_structures_and_rewards(self):
    """Return (list of structures, list of rewards) in the Population."""
    structures = []
    rewards = []
    for sample in self:
      structures.append(sample.structure)
      rewards.append(sample.reward)
    return structures, rewards

  @property
  def structures(self):
    """Returns the structure of all samples in the population."""
    return [sample.structure for sample in self]

  @property
  def rewards(self):
    """Returns the reward of all samples in the population."""
    return [sample.reward for sample in self]

  @staticmethod
  def from_arrays(structures, rewards=None, batch_index=0, metadata=None):
    """Creates Population from the specified fields.

    Args:
      structures: Iterable of structures (1-D np array or list).
      rewards: Iterable of float rewards. If None, the corresponding Samples are
        given each given a reward of None.
      batch_index: Either an int, in which case all Samples created by this
        function will be given this batch_index or an iterable of ints for each
        corresponding structure.
      metadata: Metadata to store in the Samples.

    Returns:
      A Population.
    """
    samples = samples_from_arrays(structures, rewards, batch_index, metadata)
    return Population(samples)

  def add_batch(self, structures, rewards=None, batch_index=None,
                metadata=None):
    """Adds a batch of samples to the Population.

    Args:
      structures: Iterable of structures (1-D np array or list).
      rewards: Iterable of rewards.
      batch_index: Either an int, in which case all Samples created by this
        function will be given this batch_index or an iterable of ints for each
        corresponding structure. If `None`, uses `self.current_batch + 1`.
      metadata: Metadata to store in the Samples.
    """
    if batch_index is None:
      batch_index = self.current_batch_index + 1
    samples = samples_from_arrays(structures, rewards, batch_index, metadata)
    self.add_samples(samples)

  def head(self, n):
    """Returns new Population containing first n samples."""

    return Population(self.samples[:n])

  @staticmethod
  def from_dataset(dataset):
    """Converts dataset of DatasetSample to Population."""
    samples = utils.arrays_from_dataset(dataset)
    return Population.from_arrays(samples.structure, samples.reward)

  def to_dataset(self):
    """Converts the population to a `tf.data.Dataset` with `DatasetSample`s."""
    structures, rewards = self.to_structures_and_rewards()
    return utils.dataset_from_tensors(
        DatasetSample(structure=structures, reward=rewards))

  @staticmethod
  def from_tfrecord(filename):
    """Reads Population from tfrecord file."""
    raw_dataset = tf.data.TFRecordDataset([filename])
    parsed_dataset = raw_dataset.map(parse_tf_example)

    def _record_to_dict(record):
      mapping = {key: utils.to_array(value) for key, value in record.items()}
      if 'batch_index' in mapping:
        mapping['batch_index'] = int(mapping['batch_index'])
      return mapping

    return Population(Sample(**_record_to_dict(r)) for r in parsed_dataset)

  def to_tfrecord(self, filename):
    """Writes Population to tfrecord file."""

    with tf.python_io.TFRecordWriter(filename) as writer:
      for sample in self.samples:
        writer.write(sample.to_tfexample().SerializeToString())

  def to_frame(self):
    """Converts a `Population` to a `pd.DataFrame`."""
    records = [sample.to_dict() for sample in self.samples]
    return pd.DataFrame.from_records(records)

  @classmethod
  def from_frame(cls, frame):
    """Converts a `pd.DataFrame` to a `Population`."""
    if frame.empty:
      return cls()
    population = cls()
    for _, row in frame.iterrows():
      sample = Sample(**row.to_dict())
      population.add_sample(sample)
    return population

  def to_csv(self, path, domain=None):
    """Stores a population to a CSV file.

    Args:
      path: The output file path.
      domain: An optional `domains.Domain`. If provided, will also store
        decoded structures in the CSV file.
    """
    population_frame_to_csv(self.to_frame(), path, domain=domain)

  @classmethod
  def from_csv(cls, path):
    """Restores a population from a CSV file.

    Args:
      path: The CSV file path.

    Returns:
      An instance of a `Population`.
    """
    return cls.from_frame(population_frame_from_csv(path).drop(
        columns=['decoded_structure'], errors='ignore'))

  def best_n(self, n=1, q=None, discard_duplicates=False, blacklist=None):
    """Returns the best n samples.

    Note that ties are broken deterministically.

    Args:
      n: Max number to return
      q: A float in (0, 1) corresponding to the minimum quantile for selecting
        samples. If provided, `n` is ignored and samples with a reward >=
        this quantile are selected.
      discard_duplicates: If True, when several samples have the same structure,
        return only one of them (the selected one is unspecified).
      blacklist: Iterable of structures that should be excluded.

    Returns:
      Population containing the best n Samples, sorted in decreasing order of
      reward (output[0] is the best). Returns less than n if there are fewer
      than n Samples in the population.
    """
    if self.empty:
      raise ValueError('Population empty.')

    samples = self.samples
    if blacklist:
      samples = self._filter(samples, blacklist)

    # are unique.
    if discard_duplicates and len(samples) > 1:
      samples = utils.deduplicate_samples(samples)

    samples = sorted(samples, key=lambda sample: sample.reward, reverse=True)
    if q is not None:
      q_value = np.quantile([sample.reward for sample in samples], q)
      return Population(
          [sample for sample in samples if sample.reward >= q_value])
    else:
      return Population(samples[:n])

  def best(self, blacklist=None):
    return self.best_n(1, blacklist=blacklist).samples[0]

  def _filter(self, samples, blacklist):
    blacklist = set(utils.hash_structure(structure) for structure in blacklist)
    return Population(
        sample for sample in samples
        if utils.hash_structure(sample.structure) not in blacklist)

  def contains_structure(self, structure):
    return self.contains_structures([structure])[0]

  # TODO(dbelanger): this has computational overhead because it hashes the
  # entire population. If this function is being called often, the set of hashed
  # structures should be built up incrementally as elements are added to the
  # population. Right now, the function is rarely called.
  # TODO(ddohan): Build this just in time: track what hasn't been hashed
  #   since last call, and add any new structures before checking contain.
  def contains_structures(self, structures):
    structures_in_population = set(
        utils.hash_structure(sample.structure) for sample in self.samples)
    return [
        utils.hash_structure(structure) in structures_in_population
        for structure in structures
    ]

  def deduplicate(self, select_best=False):
    """De-duplicates Samples with identical structures.

    Args:
      select_best: Whether to select the sample with the highest reward among
        samples with the same structure. Otherwise, the sample that occurs
        first will be selected.

    Returns:
      A Population with de-duplicated samples.
    """
    return Population(utils.deduplicate_samples(self, select_best=select_best))

  def discard_infeasible(self):
    """Returns a new `Population` with all infeasible samples removed."""
    return Population(sample for sample in self if not sample.infeasible)
