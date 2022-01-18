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

"""Base Problem class and helper functions for constructing problems.

There is a one-to-one mapping between Problem classes and datasets.

Each problem should be registered, at which point it's given a unique name. The
name will be used to look up the problem and as the filename for storing data.
The problem name should fully define the dataset so that the saved data doesn't
change if we re-run the data-generation pipeline.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import enum
import os
import random
import re
import six
from tensor2tensor.data_generators import text_encoder
import tensorflow.compat.v1 as tf


class DataSplit(enum.IntEnum):
  """Possible splits of a dataset."""
  ALL = 0  # Contains all of the below.
  TRAIN = 1
  VALIDATION = 2
  TEST = 3


# The registration logic is copied and modified from
#   //third_party/py/tensor2tensor/utils/registry.py
#
# We make a copy so that we aren't tied to maintaining consistency with
# Tensor2Tensor's naming scheme.
REGISTERED_PROBLEMS = {}
_first_cap_re = re.compile("(.)([A-Z][a-z0-9]+)")
_all_cap_re = re.compile("([a-z0-9])([A-Z])")


def _convert_camel_to_snake(name):
  s1 = _first_cap_re.sub(r"\1_\2", name)
  return _all_cap_re.sub(r"\1_\2", s1).lower()


def _default_name(obj_class):
  """Convert a class name to the registry's default name for the class.

  Args:
    obj_class: the name of a class

  Returns:
    The registry's default name for the class.
  """
  return _convert_camel_to_snake(obj_class.__name__)


def register_problem(problem_cls):
  """Register a Problem, naming it by a snake-cased version of the class name.

  After a problem has been registered, it can be looked up by name using the
  `make_problem` function below.

  This function is used as a decorator on the class definition. For example,

    @problem.register_problem
    class DictExpressionErrorProblem(graph_problem.SupervisedGraphProblem):
      ...

  Args:
    problem_cls: Class defining the problem.

  Returns:
    The input argument unmodified so this function can work as a decorator.

  Raises:
    LookupError: If a problem with the same name has already been registered.
  """
  problem_name = _default_name(problem_cls)
  if problem_name in REGISTERED_PROBLEMS:
    raise LookupError("Problem {} already registered.".format(problem_name))
  REGISTERED_PROBLEMS[problem_name] = problem_cls
  problem_cls.name = problem_name
  return problem_cls


def make_problem(problem_name):
  """Looks up Problem class by name and instantiates an instance of the class.

  Args:
    problem_name: A snake-cased version of the problem class name. This name is
      assigned by `register_problem` above.

  Returns:
    A Problem instance for the problem with name `problem_name`.

  Raises:
    LookupError: If the problem name isn't found.
  """
  if problem_name not in REGISTERED_PROBLEMS:
    raise LookupError("Can't find Problem named {}.".format(problem_name))

  return REGISTERED_PROBLEMS[problem_name]()


def load_word_tokenizer(vocab_path, oov_token_to_add=None):
  return text_encoder.TokenTextEncoder(vocab_path, replace_oov=oov_token_to_add)


def make_or_load_word_tokenizer(vocab_path,
                                text_generator,
                                uniquify_tokens=True,
                                oov_token_to_add=None):
  """Makes or loads a tokenizer depending on whether the vocab file exists.

  Also saves the vocab if the file does not exist.

  Args:
    vocab_path: Full vocab path to save to and/or load from. If None, just
      create a tokenizer without loading or saving.
    text_generator: A generator that yields text used to generate vocab. We
      assume that text can be split by spaces to get tokens.
    uniquify_tokens: A boolean indicating whether to make the set of tokens
      unique. By default we want to do this, but we allow it to be disabled
      mainly for testing purposes.
    oov_token_to_add: A string to add to the vocab to represent "out of
      vocabulary." If None, do not add an OOV token.

  Returns:
    A text_encoder.TokenTextEncoder.
  """
  if vocab_path is not None and tf.gfile.Exists(vocab_path):
    tf.logging.info("Using vocab from '%s'", vocab_path)
    return text_encoder.TokenTextEncoder(
        vocab_path, replace_oov=oov_token_to_add)

  if uniquify_tokens:
    vocab_list = list(set(text_generator))
  else:
    vocab_list = list(text_generator)

  if oov_token_to_add is not None:
    assert oov_token_to_add not in vocab_list
    vocab_list.append(oov_token_to_add)

  tokenizer = text_encoder.TokenTextEncoder(
      None, vocab_list=vocab_list, replace_oov=oov_token_to_add)

  if vocab_path is not None:
    tf.logging.info("Writing vocab to '%s'", vocab_path)
    tokenizer.store_to_file(vocab_path)

  return tokenizer


def make_or_load_subword_tokenizer(vocab_path, text_generator,
                                   target_vocab_size):
  """Makes or loads a tokenizer depending on whether the vocab file exists.

  Also saves the vocab if the file does not exist.

  Args:
    vocab_path: Full vocab path to save to and/or load from. If None, just
      create a tokenizer without loading or saving.
    text_generator: A generator that yields text used to generate vocab. We
      assume that text can be split by spaces to get tokens.
    target_vocab_size: Target size of the vocabulary.

  Returns:
    A text_encoder.SubwordTextEncoder.
  """
  if vocab_path is not None and tf.gfile.Exists(vocab_path):
    tf.logging.info("Using vocab from '%s'", vocab_path)
    return text_encoder.SubwordTextEncoder(vocab_path)

  token_counts = collections.Counter()
  for text in text_generator:
    token_counts += collections.Counter(text.split())

  # The subword tokenizer searches over minimum subtoken counts to create
  # vocab items for, aiming to get a vocab size near `target_vocab_size`.
  # We need to provide initial lower and upper bounds to initialize the binary
  # search.
  # TODO(dtarlow): Might want to tweak this.
  token_count_binary_search_lower_bound = 1
  token_count_binary_search_upper_bound = len(token_counts) // 2
  tokenizer = text_encoder.SubwordTextEncoder.build_to_target_size(
      target_vocab_size,
      token_counts,
      token_count_binary_search_lower_bound,
      token_count_binary_search_upper_bound,
      reserved_tokens=text_encoder.RESERVED_TOKENS)

  if vocab_path is not None:
    tf.logging.info("Writing vocab to '%s'", vocab_path)
    tokenizer.store_to_file(vocab_path)

  return tokenizer


def make_or_load_limited_size_subword_tokenizer(
    vocab_path, token_counts_untruncated, truncation_length, target_vocab_size):
  """Makes or loads a tokenizer depending on whether the vocab file exists.

  Also saves the vocab if the file does not exist. This function differs from
  the `make_or_load_subword_tokenizer` function above in three ways:
  (1) Token counts are provided directly as a dict, rather than obtained by
      counting in a sequence of texts.
  (2) The tokens are truncated to a maximum length `truncation_length`, and the
      counts of tokens mapping onto the same truncated token are summed.
  (3) The optional parameter `max_subtoken_length` is passed to the method
      `build_to_target_size` of SubwordTextEncoder, to avoid the quadratic
      in vocabulary size cost (both time and memory).

  Args:
    vocab_path: Full vocab path to save to and/or load from. If None, just
      create a tokenizer without loading or saving.
    token_counts_untruncated: Dict {(untruncated) token: count}.
    truncation_length: Integer specifying maximum token length after truncation.
    target_vocab_size: Target size of the vocabulary.

  Returns:
    A text_encoder.SubwordTextEncoder.
  """
  if vocab_path is not None and tf.gfile.Exists(vocab_path):
    tf.logging.info("Using vocab from '%s'", vocab_path)
    return text_encoder.SubwordTextEncoder(vocab_path)

  # Truncate tokens to maximum length NODE_TEXT_TRUNCATION_LENGTH:
  token_counts = collections.defaultdict(int)
  for token in token_counts_untruncated:
    token_truncated = token[:truncation_length]
    token_counts[token_truncated] += token_counts_untruncated[token]
  tf.logging.info("Vocabulary size reduced from %d to %d due to truncation." % (
      len(token_counts_untruncated), len(token_counts)))

  # The subword tokenizer searches over minimum subtoken counts to create
  # vocab items for, aiming to get a vocab size near `target_vocab_size`.
  # We need to provide initial lower and upper bounds to initialize the binary
  # search.
  # TODO(matejb): This might be tweaked in the future (see corresponding TODO
  # in the `make_or_load_subword_tokenizer` function above).
  token_count_binary_search_lower_bound = 1
  token_count_binary_search_upper_bound = len(token_counts) // 2
  tokenizer = text_encoder.SubwordTextEncoder.build_to_target_size(
      target_vocab_size, token_counts,
      token_count_binary_search_lower_bound,
      token_count_binary_search_upper_bound,
      max_subtoken_length=truncation_length,
      reserved_tokens=text_encoder.RESERVED_TOKENS)

  if vocab_path is not None:
    tf.logging.info("Writing vocab to '%s'", vocab_path)
    tokenizer.store_to_file(vocab_path)

  return tokenizer


def int_feature(values):
  """Create a tf.Feature from a list of values.

  Args:
    values: A list of integer values.

  Returns:
    A tf.Feature containing `values`.
  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def string_feature(values):
  """Create a tf.Feature from a list of strings.

  Args:
    values: A list of strings.

  Returns:
    A tf.Feature containing `values`.
  """
  values = [tf.compat.as_bytes(v) for v in values]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float32_feature(values):
  """Create a tf.Feature from a list of values.

  Args:
    values: A list of float values.

  Returns:
    A tf.Feature containing `values`.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def extract_feature_values(feature):
  """Convert a tf.train.Feature into a list of data."""
  if feature.HasField("int64_list"):
    return feature.int64_list.value
  elif feature.HasField("float_list"):
    return feature.float_list.value
  elif feature.HasField("bytes_list"):
    return feature.bytes_list.value
  else:
    return []


# These are copied from Tensor2Tensor data_generator.generator_utils.
# Copy-pasting so that our framework won't break if they change their naming
# convention.
def sharded_name(base_name, shard, total_shards):
  if isinstance(shard, str) and isinstance(total_shards, str):
    return "%s-%s-of-%s" % (base_name, shard, total_shards)
  else:
    return "%s-%.5d-of-%.5d" % (base_name, shard, total_shards)


def shard_filepath(fname, num_shards):
  return [
      sharded_name(fname, shard, num_shards) for shard in range(num_shards)
  ]


def _shuffle_generator(generator, shuffle_buffer_size):
  """Yields elements from `generator`, shuffled using `shuffle_buffer_size`.

  Mimics the behaviour of Dataset.shuffle(). The method fills a buffer with
  `shuffle_buffer_size` elements, then randomly samples elements from this
  buffer, replacing the selected elements with new elements from the generator.

  Args:
    generator: A Python generator of objects to be shuffled.
    shuffle_buffer_size: Int specifying the size of the shuffle buffer.
  Yields:
    Elements of `generator` in shuffled order.
  """

  # TODO(matejb): Consider using random.shuffle when the buffer is first filled.

  shuffle_buffer = []
  for data in generator:
    shuffle_buffer.append(data)
    if len(shuffle_buffer) == shuffle_buffer_size:
      random_index = random.randint(0, shuffle_buffer_size - 1)
      yield shuffle_buffer.pop(random_index)
  while shuffle_buffer:
    random_index = random.randint(0, len(shuffle_buffer) - 1)
    yield shuffle_buffer.pop(random_index)


def _write_sharded_records(example_generator,
                           data_filenames,
                           shuffle_buffer_size,
                           cycle_length):
  """Write tf.Examples to sharded tf.Record files.

  This is like generator_utils.generate_files from Tensor2Tensor, but we already
  have tf.Examples so don't want the `to_example` call that's inside there.

  Args:
    example_generator: A generator of tf.Examples.
    data_filenames: A list of filenames in which to save data.
    shuffle_buffer_size: Buffer size to use for shuffling examples before
      sharding. Set to 1 to avoid shuffling.
    cycle_length: Number of examples to write to a shard before cycling to the
      next shard.
  """
  num_shards = len(data_filenames)
  tmp_filenames = [filename + ".incomplete" for filename in data_filenames]
  record_writers = [tf.python_io.TFRecordWriter(filename)
                    for filename in tmp_filenames]

  for counter, example in enumerate(
      _shuffle_generator(example_generator, shuffle_buffer_size)):
    shard = (counter // cycle_length) % num_shards
    record_writers[shard].write(example.SerializeToString())

  for writer in record_writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filenames, data_filenames):
    tf.gfile.Rename(tmp_name, final_name, overwrite=True)


class Problem(six.with_metaclass(abc.ABCMeta)):
  """Abstract class defining a problem.

  Attributes:
    name: A string name of the problem. This attribute is populated by the
      @register_problem decorator.
  """

  @abc.abstractmethod
  def _tf_example_generator(self, data_dir, split):
    """Generates tf.Example representation of problem instances from `split`.

    Override this with a method that generates tf.Examples.

    Args:
      data_dir: Directory to load (and possibly store) vocab file(s).
      split: A DataSplit indicating which split to generate examples from.

    Yields:
      tf.Examples from the given split.
    """
    raise NotImplementedError("Must implement `_tf_example_generator`.")

  # TODO(jfrankle): Make this and related properties static.
  @abc.abstractproperty
  def feature_descriptions(self):
    """Returns a dictionary of feature descriptions for this problem.

    The dictionary describes which features are used in generated
    tf.Examples for this problem. It maps from feature names (i.e. keys
    of the generated tf.Example files) to tensorflow feature objects.
    """
    raise NotImplementedError("Must implement `feature_descriptions`.")

  def dataset_filename(self, split=None):
    if split is None:
      return self.name
    else:
      return "{}-{}.tfrecord".format(self.name, split.name)

  def shard_filenames(self, data_dir, split):
    if split == DataSplit.TRAIN:
      num_shards = self.num_train_shards
    elif split == DataSplit.VALIDATION:
      num_shards = self.num_validation_shards
    elif split == DataSplit.TEST:
      num_shards = self.num_test_shards
    else:
      raise ValueError("Invalid split {}.".format(split))

    base_filename = self.dataset_filename(split=split)
    shard_filenames = shard_filepath(base_filename, num_shards)

    return [os.path.join(data_dir, filename) for filename in shard_filenames]

  def shard_filenames_pattern(self, data_dir, split):
    base_filename = os.path.join(data_dir, self.dataset_filename(split=split))
    return sharded_name(base_filename, "*", "*")

  def generate_and_save_data(self, data_dir):
    """Generates and saves data in `data_dir`.

    This is the primary entry point for generating data for a problem. Data is
    stored as tf.Record of tf.Examples.

    Args:
      data_dir: Directory to save data in.
    """
    for split in [DataSplit.TRAIN, DataSplit.VALIDATION, DataSplit.TEST]:
      if split == DataSplit.TRAIN:
        shuffle_buffer_size = self.train_sharding_shuffle_buffer_size
        cycle_length = self.train_sharding_cycle_length
      elif split == DataSplit.VALIDATION:
        shuffle_buffer_size = self.validation_sharding_shuffle_buffer_size
        cycle_length = self.validation_sharding_cycle_length
      elif split == DataSplit.TEST:
        shuffle_buffer_size = self.test_sharding_shuffle_buffer_size
        cycle_length = self.test_sharding_cycle_length
      _write_sharded_records(self._tf_example_generator(data_dir, split),
                             self.shard_filenames(data_dir, split=split),
                             shuffle_buffer_size,
                             cycle_length)

  def generate_and_yield_data(self, data_dir, split):
    """Generates and yields data, e.g., to assist in building pipelines.

    This is the primary entry point for generating data for a problem. Data is
    yielded as tf.Examples.

    Args:
      data_dir: A string, the file path to the directory where vocabularies
        may be loaded from and/or stored to.
      split: One of the `DataSplit` designations. It may be None.
    Yields:
      tf.Example protos.
    """
    for example in self._tf_example_generator(data_dir, split):
      yield example

  def load_raw_tf_examples(self, data_dir, split, max_count=None):
    """Loads raw tf.Examples as a list of tf.Example protos.

    This is only used for compatibility with some existing tests that expect to
    be able to load raw all tf.Examples into a list. Don't call this on a large
    dataset, as it will likely make you run out of memory.

    Args:
      data_dir: Directory containing tfrecord files generated by this Problem.
      split: One of the problem.DataSplit constants indicating the data split.
      max_count: Maximum number of examples to load. If None, load all examples.

    Returns:
      A list of tf.Examples for all data in the dataset.
    """
    examples = []
    for filename in self.shard_filenames(data_dir, split=split):
      for record in tf.compat.v1.io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(record)
        examples.append(example)
        if max_count is not None and len(examples) >= max_count:
          return examples

    return examples

  def load_data(self, data_dir, split, buffer_size=None,
                num_parallel_parses=None, use_interleave=False,
                invocation_index=None, total_invocations=None, seed=None):
    """Returns a tf.Dataset of parsed examples from TFRecords in `data_dir`.

    Each example in a TFRecord file is parsed into a dict that maps feature
    names to values, in the manner of tf.parse_single_example.

    When `use_interleave`=True and `invocation_index` and `total_invocations`
    are set, the tf.Dataset returned by this function only contains a subset of
    the data, such that the calls with different `invocation_index` disjointly
    cover (i.e., partition) the full data. This is important for ensuring that
    multiple input pipelines running in parallel are able to process the data
    faster. Note that ideally `total_invocations` should divide the number of
    shards, and each shard would contain the same number of data points
    (otherwise the last shard and the smaller shards may be weighted more).

    Args:
      data_dir: Directory containing tfrecord files generated by this Problem.
      split: One of the problem.DataSplit constants indicating the data split.
      buffer_size: The `buffer_size` parameter passed to the TFRecordDataset
          constructor. The units are bytes, and its default value is None.
      num_parallel_parses: Number of examples to parse in parallel. None is the
          default value of `num_parallel_calls` parameter in TFRecordDataset's
          `map` method.
      use_interleave: A boolean indicating whether to have the tf.Data pipeline
          explicitly shuffle filenames and read using an interleaved strategy.
      invocation_index: Int 0 <= `invocation_index` < `total_invocations`, or
          None. When not None and `use_interleave`=True, only a subset of the
          data will be loaded, such that the different input pipeline
          invocations partition the dataset.
      total_invocations: Int giving the number of input pipeline invocations, or
          None. Set to None if and only if `invocation_index` is None.
      seed: Random seed to use for the shuffling Op. Only used if interleaving.
    Returns:
      tf.Dataset of parsed example protos.
    """
    # Check that either both or neither of the invocation arguments are None.
    if (invocation_index is None) != (total_invocations is None):
      raise ValueError("Both or neither of `invocation_index` and "
                       "`total_invocations` should be None.")

    if use_interleave:
      # TODO(dtarlow): Add test for interleaved strategy.
      filenames_pattern = self.shard_filenames_pattern(data_dir, split)
      num_shards = len(self.shard_filenames(data_dir, split))

      # By default, `list_files` shuffles the filenames. To ensure they can be
      # partitioned if requested, this feature needs to be disabled.
      dataset = tf.data.Dataset.list_files(filenames_pattern, shuffle=False)

      # Filter the shards based on current `invocation_index`.
      # (Need to explicitly test for not None, because 0 is a meaningful value.)
      if invocation_index is not None:
        # Set `block_size` to ceil(num_shards / total_invocations).
        block_size = (num_shards + total_invocations - 1) // total_invocations
        block_start = invocation_index * block_size
        dataset = dataset.skip(block_start).take(block_size).cache()
        cycle_length = block_size
      else:
        dataset = dataset.cache()
        cycle_length = num_shards

      dataset = (dataset
                 .repeat()
                 .shuffle(cycle_length, seed=seed,
                          reshuffle_each_iteration=True)
                 .prefetch(cycle_length))
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              cycle_length=cycle_length,
              block_length=1))
      dataset = dataset.prefetch(num_parallel_parses)
    else:
      filenames = self.shard_filenames(data_dir, split=split)
      dataset = tf.data.TFRecordDataset(filenames,
                                        buffer_size=buffer_size,
                                        num_parallel_reads=num_parallel_parses)

    def _parse_single_example(example_proto):
      return tf.parse_single_example(example_proto, self.feature_descriptions)

    return dataset.map(
        _parse_single_example, num_parallel_calls=num_parallel_parses)

  @property
  def num_train_shards(self):
    return 1

  @property
  def train_sharding_shuffle_buffer_size(self):
    return 1

  @property
  def train_sharding_cycle_length(self):
    return 100

  @property
  def num_validation_shards(self):
    return 1

  @property
  def validation_sharding_shuffle_buffer_size(self):
    return 1

  @property
  def validation_sharding_cycle_length(self):
    return 100

  @property
  def num_test_shards(self):
    return 1

  @property
  def test_sharding_shuffle_buffer_size(self):
    return 1

  @property
  def test_sharding_cycle_length(self):
    return 100
