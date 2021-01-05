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

# -*- coding: utf-8 -*-
"""Utility fuctions to preprocess data."""

import collections
import random
import numpy as np
import six
from tensorflow.compat.v1 import gfile
from meta_reward_learning.semantic_parsing.nsm import nlp_utils

DECODE_TK = '<DECODE>'
UNK_TK = '<UNK>'
START_TK = '<START>'
END_TK = '<END>'


# Copied from stack overflow
def namedtuple_with_defaults(typename, field_names, default_values=()):
  T = collections.namedtuple(typename, field_names)
  T.__new__.__defaults__ = (None,) * len(T._fields)
  if isinstance(default_values, collections.Mapping):
    prototype = T(**default_values)
  else:
    prototype = T(*default_values)
  T.__new__.__defaults__ = tuple(prototype)
  return T


# Utility functions to deal with input data.
def read_seq_dataset_from_file(filename,
                               max_vocab_size=1000000,
                               min_count=0,
                               unk_tk=UNK_TK,
                               start_tk=START_TK,
                               decode_tk=DECODE_TK,
                               end_tk=END_TK,
                               tokenize=True):
  """Get the sequences and vocab from a file.

  Args:
    filename: name of file.
    max_vocab_size: the maximum number of tokens in the vocab.
    min_count: the minimum number of appearance for a token to be added into the
      vocab.
    unk_tk: the unknown token.
    start_tk: the start of sentence token.
    decode_tk: the start of decoding token.
    end_tk: the end of decoding token.
    tokenize: Whether to tokenize the text in the file.

  Returns:
    seqs: a list of lists of tokens.
    vocab: a Vocab object created from the file.
  """
  vocab = generate_vocab_from_file(
      filename,
      tokenize=tokenize,
      max_vocab_size=max_vocab_size,
      min_count=min_count,
      unk_tk=unk_tk,
      start_tk=start_tk,
      decode_tk=decode_tk,
      end_tk=end_tk)
  seqs = []
  with gfile.Open(filename, 'r') as f:
    for line in f:
      if tokenize:
        tokens = nlp_utils.tokenize(line)
      else:
        tokens = line.strip().split()
      seqs.append(tokens)
  return seqs, vocab


def create_lm_inputs_labels(dataset, vocab):
  """Create inputs and labels for language modelling."""
  decode_id = vocab.decode_id
  end_id = vocab.end_id
  inputs = [[decode_id] + vocab.lookup(seq) for seq in dataset]
  labels = [vocab.lookup(seq) + [end_id] for seq in dataset]
  return inputs, labels


def create_seq2seq_inputs(en_dataset, en_vocab, de_dataset, de_vocab):
  """Create encoder inputs, decoder inputs and targets for seq2seq training."""
  start_id = en_vocab.start_id
  en_inputs = [[start_id] + en_vocab.lookup(seq) for seq in en_dataset]

  decode_id = de_vocab.decode_id
  end_id = de_vocab.end_id
  inputs = [[decode_id] + de_vocab.lookup(seq) for seq in de_dataset]
  targets = [de_vocab.lookup(seq) + [end_id] for seq in de_dataset]
  return en_inputs, inputs, targets


# Utilities for generating and using vocabulary.
def generate_vocab_from_file(filename,
                             tokenize=True,
                             max_vocab_size=1000000,
                             min_count=0,
                             unk_tk=UNK_TK,
                             start_tk=START_TK,
                             decode_tk=DECODE_TK,
                             end_tk=END_TK):
  """Create vocab from a given file."""
  with gfile.Open(filename, 'r') as f:
    vocab = generate_vocab_from_stream(
        f,
        tokenize=tokenize,
        max_vocab_size=max_vocab_size,
        min_count=min_count,
        unk_tk=unk_tk,
        start_tk=start_tk,
        decode_tk=decode_tk,
        end_tk=end_tk)
  return vocab


def generate_vocab_from_stream(text_stream,
                               max_vocab_size=1000000,
                               min_count=0,
                               unk_tk=UNK_TK,
                               start_tk=START_TK,
                               decode_tk=DECODE_TK,
                               end_tk=END_TK,
                               tokenize=True):
  """Create a vocab from a given text stream."""
  token_list = []
  for line in text_stream:
    if tokenize:
      new_list = nlp_utils.tokenize(line)
    else:
      new_list = line.strip().split()
    token_list += new_list
  return generate_vocab_from_list(
      token_list,
      max_vocab_size=max_vocab_size,
      min_count=min_count,
      unk_tk=unk_tk,
      start_tk=start_tk,
      decode_tk=decode_tk,
      end_tk=end_tk)


def generate_vocab_from_list(token_list,
                             max_vocab_size=1000000,
                             min_count=0,
                             unk_tk=UNK_TK,
                             start_tk=START_TK,
                             decode_tk=DECODE_TK,
                             end_tk=END_TK):
  """Create a vocab from a list of tokens."""
  token_count = {}
  for tk in token_list:
    try:
      token_count[tk] += 1
    except KeyError:
      token_count[tk] = 1
  return generate_vocab_from_token_count(
      token_count,
      max_vocab_size=max_vocab_size,
      min_count=min_count,
      unk_tk=unk_tk,
      start_tk=start_tk,
      decode_tk=decode_tk,
      end_tk=end_tk)


def sort_kv_pairs_by_value(d):
  """Turn a dict into a list of key-value pairs, sorted by value."""
  return [
      (k, v) for v, k in sorted([(v, k) for k, v in d.items()], reverse=True)
  ]


def vocab_lookup(item, vocab, unknown):
  """Look up the item from the vocabulary.

  Args:
    item: a string, an integer or a nested sequence or numpy arrays of strings
      or integers.
    vocab: a Vocab object.
    unknown: Any value. This will be used when a integer or string is not in
      Vocab.

  Returns:
    result: same structure as item, with the integer or
      string replaced by the corresponding lookup in the Vocab.
  """
  if (isinstance(item, str) or isinstance(item, int) or
      isinstance(item, unicode)):
    result = vocab.get(item, unknown)
  elif is_sequence(item) or isinstance(item, np.ndarray):
    result = [vocab_lookup(x, vocab, unknown) for x in item]
  else:
    raise ValueError('Can not handle type {}'.format(type(item)))
  return result


def generate_vocab_from_token_count(token_count,
                                    max_vocab_size=1000000,
                                    min_count=0,
                                    unk_tk=UNK_TK,
                                    start_tk=START_TK,
                                    decode_tk=DECODE_TK,
                                    end_tk=END_TK):
  """Generate vocabulary from token count information."""
  special_tks = [unk_tk, start_tk, decode_tk, end_tk]
  token_count_pairs = sort_kv_pairs_by_value(token_count)
  token_count_pairs = [(tk, count)
                       for (tk, count) in token_count_pairs
                       if (count >= min_count) and tk not in special_tks]
  token_count_pairs = token_count_pairs[:max_vocab_size - 4]
  # The vocab are organized as: first special tokens, then
  # tokens ordered by frequency.
  tokens = [p[0] for p in token_count_pairs]
  return Vocab(
      tokens,
      unk_tk=UNK_TK,
      start_tk=START_TK,
      decode_tk=DECODE_TK,
      end_tk=END_TK)


class Vocab(object):
  """A vocabulary used in language tasks."""

  def __init__(self,
               tokens,
               unk_tk=UNK_TK,
               start_tk=START_TK,
               decode_tk=DECODE_TK,
               end_tk=END_TK):
    special_tks = [unk_tk, start_tk, decode_tk, end_tk]
    self.unk_tk = unk_tk
    self.unk_id = 0
    self.start_tk = start_tk
    self.start_id = 1
    self.decode_tk = decode_tk
    self.decode_id = 2
    self.end_tk = end_tk
    self.end_id = 3
    self.special_tks = special_tks

    all_tokens = special_tks + tokens
    self.vocab = {}
    self.rev_vocab = {}
    for i, token in enumerate(all_tokens):
      if token in self.vocab:
        raise ValueError('token {} repeated'.format(token))
      self.vocab[token] = i
      self.rev_vocab[i] = token

    self.size = len(self.vocab)

  def lookup(self, item, reverse=False):
    """Lookup the id/token of the token/id."""
    if reverse:
      result = vocab_lookup(item, self.rev_vocab, None)
    else:
      result = vocab_lookup(item, self.vocab, self.unk_id)
    return result

  def load_vocab(self, vocab):
    self.vocab = vocab
    self.rev_vocab = {}
    for token, i in vocab.iteritems():
      self.rev_vocab[i] = token

    self.size = len(self.vocab)


# Utilities for batching.
class BatchIterator(object):

  def __init__(self, feed_dict, shuffle=False, batch_size=32):
    self.batch_size = batch_size
    kv_pairs = [(k, v) for k, v in feed_dict.items() if v is not None]
    self.keys = [p[0] for p in kv_pairs]
    # Usually, v is an array in a (k, v) pair
    self.examples = zip(*[p[1] for p in kv_pairs])
    self.n_examples = len(self.examples)
    if shuffle:
      random.shuffle(self.examples)

  def __iter__(self):

    def _iterator():
      bs = self.batch_size
      idx = 0
      while idx < self.n_examples:
        batch_examples = self.examples[idx:idx + bs]
        idx += bs
        unzipped_batch_examples = list(zip(*batch_examples))
        batch_feed_dict = dict(
            [(k, list(v)) for k, v in zip(self.keys, unzipped_batch_examples)])
        batch_feed_dict['batch_size'] = len(batch_examples)
        batch_feed_dict['max_batch_size'] = self.batch_size
        yield batch_feed_dict

    return _iterator()


def convert_seqs_to_batch(seqs, maxlen=None):
  n_seqs = len(seqs)
  sequence_length = []
  for seq in seqs:
    sequence_length.append(len(seq))
  max_len = max(sequence_length)
  if maxlen is not None:
    max_len = max(max_len, maxlen)
    if max_len > maxlen:
      print(max_len)
  one_step = seqs[0][0]
  try:
    step_shape = one_step.shape
  except AttributeError:
    step_shape = ()
  # The batch matrix is padded with all 0s.
  batch = np.zeros((n_seqs, max_len) + step_shape)
  for i, (seq, seq_len) in enumerate(zip(seqs, sequence_length)):
    if seq_len > 0:
      batch[i, 0:seq_len] = seq
  sequence_length = np.array(sequence_length)
  return (batch, sequence_length)


def convert_batch_to_seqs(batch):
  array = batch.tensor
  sequence_length = batch.sequence_length
  seqs = np.vsplit(array, array.shape[0])
  result = []
  for seq, seq_len in zip(seqs, sequence_length):
    result.append(list(seq[0][:seq_len]))
  return result


def deep_vstack(structs):
  """Turn tuples of arrays into one tuple of stacked arrays."""
  if len(structs) == 0:
    raise 'No structs available.'
  flat_structs = [flatten(struct) for struct in structs]
  flat_result = [np.vstack(args) for args in zip(*flat_structs)]
  result = pack_sequence_as(structs[0], flat_result)
  return result


def deep_split(struct):
  flat_struct = flatten(struct)
  new_flat_structs = zip(*[np.split(x, x.shape[0]) for x in flat_struct])
  new_structs = [pack_sequence_as(struct, x) for x in new_flat_structs]
  return new_structs


class BatchConverter(object):
  """BatchConverter converts input data into dictionaries of

  batches of data (by stacking and padding) that is ready to
  feed into TF graphs.

  """

  def __init__(self,
               tuple_keys=None,
               seq_keys=None,
               out_keys=None,
               preprocess_fn=None,
               maxlen=None,
               out_maxlen=None):

    if not tuple_keys:
      self.tuple_keys = []
    else:
      self.tuple_keys = tuple_keys

    if not seq_keys:
      self.seq_keys = []
    else:
      self.seq_keys = seq_keys

    if out_keys:
      self.out_keys = out_keys
    else:
      self.out_keys = []

    self.preprocess_fn = preprocess_fn
    self.maxlen = maxlen
    if out_maxlen is None:
      self.out_maxlen = maxlen
    else:
      self.out_maxlen = out_maxlen

  def add_preprocess(self, preprocess_fn):
    self.preprocess_fn = preprocess_fn

  def convert(self, batch_dict):
    if self.preprocess_fn is not None:
      self.preprocess_fn(batch_dict)
    for k, v in batch_dict.iteritems():
      if k in self.tuple_keys:
        batch_dict[k] = deep_vstack(v)
      elif k in self.seq_keys:
        if k in self.out_keys:
          maxlen = self.out_maxlen
        else:
          maxlen = self.maxlen
        batch_dict[k] = convert_seqs_to_batch(v, maxlen)
    return batch_dict


class BatchAggregator(object):

  def __init__(self,
               tuple_keys=None,
               seq_keys=None,
               num_keys=None,
               keep_keys=None):

    if tuple_keys == None:
      self.tuple_keys = set()
    else:
      self.tuple_keys = set(tuple_keys)

    if seq_keys == None:
      self.seq_keys = set()
    else:
      self.seq_keys = set(seq_keys)

    if num_keys == None:
      self.num_keys = set()
    else:
      self.num_keys = set(num_keys)

    if keep_keys == None:
      self.keep_keys = set()
    else:
      self.keep_keys = set(keep_keys)

    self.all_keys = set.union(self.seq_keys, self.tuple_keys, self.num_keys,
                              self.keep_keys)

    self.result_dict = {}

  def reset(self):
    self.result_dict = {}

  def merge(self, batch_dict):
    for k in self.all_keys:
      b = batch_dict[k]
      if k in self.seq_keys:
        v = convert_batch_to_seqs(b)
      elif k in self.tuple_keys:
        v = deep_split(b)
      elif k in self.num_keys:
        v = b
      elif k in self.keep_keys:
        v = list(b)
      if k in self.result_dict:
        self.result_dict[k] += v
      else:
        self.result_dict[k] = v

  @property
  def result(self):
    return self.result_dict


# Utilities for dealing with data with internal structures.
def zero_struct_like(struct):
  return constant_struct_like(struct, 0.0)


def constant_struct_like(struct, constant):
  flat_struct = flatten(struct)
  new_flat_struct = [np.ones_like(x) * constant for x in flat_struct]
  return pack_sequence_as(struct, new_flat_struct)


# The following code are copied from TensorFlow source code.
def is_sequence(seq):
  """Returns a true if its input is a collections.Sequence (except strings).

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a not a string and is a collections.Sequence.
  """
  return (isinstance(seq, collections.Sequence) and
          not isinstance(seq, six.string_types))


def flatten(nest):
  """Returns a flat sequence from a given nested structure.

  If `nest` is not a sequence, this returns a single-element list: `[nest]`.

  Args:
    nest: an arbitrarily nested structure or a scalar object. Note, numpy arrays
      are considered scalars.

  Returns:
    A Python list, the flattened version of the input.
  """
  return list(_yield_flat_nest(nest)) if is_sequence(nest) else [nest]


def _yield_flat_nest(nest):
  for n in nest:
    if is_sequence(n):
      for ni in _yield_flat_nest(n):
        yield ni
    else:
      yield n


def _packed_nest_with_indices(structure, flat, index):
  """Helper function for pack_nest_as.

  Args:
    structure: Substructure (tuple of elements and/or tuples) to mimic
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  for s in structure:
    if is_sequence(s):
      new_index, child = _packed_nest_with_indices(s, flat, index)
      packed.append(_sequence_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def pack_sequence_as(structure, flat_sequence):
  """Returns a given flattened sequence packed into a nest.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Args:
    structure: tuple or list constructed of scalars and/or other tuples/lists,
      or a scalar.  Note: numpy arrays are considered scalars.
    flat_sequence: flat sequence to pack.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If nest and structure have different element counts.
  """
  if not is_sequence(flat_sequence):
    raise TypeError('flat_sequence must be a sequence')

  if not is_sequence(structure):
    if len(flat_sequence) != 1:
      raise ValueError('Structure is a scalar but len(flat_sequence) == %d > 1'
                       % len(flat_sequence))
    return flat_sequence[0]

  flat_structure = flatten(structure)
  if len(flat_structure) != len(flat_sequence):
    raise ValueError(
        'Could not pack sequence. Structure had %d elements, but flat_sequence '
        'had %d elements.  Structure: %s, flat_sequence: %s.' %
        (len(flat_structure), len(flat_sequence), structure, flat_sequence))

  _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
  return _sequence_like(structure, packed)


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, or a `namedtuple` class.
    args: elements to be converted to a sequence.

  Returns:
    `args` with the type of `instance`.
  """
  if (isinstance(instance, tuple) and hasattr(instance, '_fields') and
      isinstance(instance._fields, collections.Sequence) and
      all(isinstance(f, six.string_types) for f in instance._fields)):
    # This is a namedtuple
    return type(instance)(*args)
  else:
    # Not a namedtuple
    return type(instance)(args)


def map_structure(func, *structure):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain the results in the same structure.

  Args:
    func: A callable that acceps as many arguments are there are structures.
    *structure: scalar, or tuple or list of constructed scalars and/or other
      tuples/lists, or scalars.  Note: numpy arrays are considered scalars.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
  """
  if not callable(func):
    raise TypeError('func must be callable, got: %s' % func)

  if not structure:
    raise ValueError('Must provide at least one structure')

  for other in structure[1:]:
    assert_same_structure(structure[0], other)

  flat_structure = [flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return pack_sequence_as(structure[0], [func(*x) for x in entries])


# Check the same structure.
def _recursive_assert_same_structure(nest1, nest2):
  is_sequence_nest1 = is_sequence(nest1)
  if is_sequence_nest1 != is_sequence(nest2):
    raise ValueError(
        "The two structures don't have the same nested structure. "
        'First structure: %s, second structure: %s.' % (nest1, nest2))

  if is_sequence_nest1:
    type_nest1 = type(nest1)
    type_nest2 = type(nest2)
    if type_nest1 != type_nest2:
      raise TypeError(
          "The two structures don't have the same sequence type. First "
          'structure has type %s, while second structure has type %s.' %
          (type_nest1, type_nest2))

    for n1, n2 in zip(nest1, nest2):
      _recursive_assert_same_structure(n1, n2)


def assert_same_structure(nest1, nest2):
  """Asserts that two structures are nested in the same way.

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures.
  """
  len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
  len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
  if len_nest1 != len_nest2:
    raise ValueError(
        "The two structures don't have the same number of "
        'elements. First structure: %s, second structure: %s.' % (nest1, nest2))
  _recursive_assert_same_structure(nest1, nest2)


def max_sum(arr):
  """Get the sum of all the max elements in the array."""
  if not arr:
    return 0.0
  max_val = max(arr)
  return sum([val for val in arr if val == max_val])
