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

"""Specifications for different types of input/output domains."""

import abc
import collections

import gin
import numpy as np
import six
from six.moves import range

from protein_lm import seq_utils
from protein_lm import utils

BOS_TOKEN = '<'  # Beginning of sequence token.
EOS_TOKEN = '>'  # End of sequence token.
PAD_TOKEN = '_'  # End of sequence token.
MASK_TOKEN = '*'  # End of sequence token.
SEP_TOKEN = '|'  # A special token for separating tokens for serialization.


@gin.configurable
class Vocabulary(object):
  """Basic vocabulary used to represent output tokens for domains."""

  def __init__(self,
               tokens,
               include_bos=False,
               include_eos=False,
               include_pad=False,
               include_mask=False,
               bos_token=BOS_TOKEN,
               eos_token=EOS_TOKEN,
               pad_token=PAD_TOKEN,
               mask_token=MASK_TOKEN):
    """A token vocabulary.

    Args:
      tokens: An list of tokens to put in the vocab. If an int, will be
        interpreted as the number of tokens and '0', ..., 'tokens-1' will be
        used as tokens.
      include_bos: Whether to append `bos_token` to `tokens` that marks the
        beginning of a sequence.
      include_eos: Whether to append `eos_token` to `tokens` that marks the
        end of a sequence.
      include_pad: Whether to append `pad_token` to `tokens` to marks past end
        of sequence.
      include_mask: Whether to append `mask_token` to `tokens` to mark masked
        positions.
      bos_token: A special token than marks the beginning of sequence.
        Ignored if `include_bos == False`.
      eos_token: A special token than marks the end of sequence.
        Ignored if `include_eos == False`.
      pad_token: A special token than marks past the end of sequence.
        Ignored if `include_pad == False`.
      mask_token: A special token than marks MASKED positions for e.g. BERT.
        Ignored if `include_mask == False`.
    """
    if not isinstance(tokens, collections.Iterable):
      tokens = range(tokens)
    tokens = [str(token) for token in tokens]
    if include_bos:
      tokens.append(bos_token)
    if include_eos:
      tokens.append(eos_token)
    if include_pad:
      tokens.append(pad_token)
    if include_mask:
      tokens.append(mask_token)
    if len(set(tokens)) != len(tokens):
      raise ValueError('tokens not unique!')
    special_tokens = sorted(set(tokens) & set([SEP_TOKEN]))
    if special_tokens:
      raise ValueError(
          f'tokens contains reserved special tokens: {special_tokens}!')

    self._tokens = tokens
    self._token_ids = list(range(len(self._tokens)))
    self._id_to_token = collections.OrderedDict(
        zip(self._token_ids, self._tokens))
    self._token_to_id = collections.OrderedDict(
        zip(self._tokens, self._token_ids))
    self._bos_token = bos_token if include_bos else None
    self._eos_token = eos_token if include_eos else None
    self._mask_token = mask_token if include_mask else None
    self._pad_token = pad_token if include_pad else None

  def __len__(self):
    return len(self._tokens)

  @property
  def tokens(self):
    """Return the tokens of the vocabulary."""
    return list(self._tokens)

  @property
  def token_ids(self):
    """Return the tokens ids of the vocabulary."""
    return list(self._token_ids)

  @property
  def bos(self):
    """Returns the index of the BOS token or None if unspecified."""
    return (None if self._bos_token is None else
            self._token_to_id[self._bos_token])

  @property
  def eos(self):
    """Returns the index of the EOS token or None if unspecified."""
    return (None if self._eos_token is None else
            self._token_to_id[self._eos_token])

  @property
  def mask(self):
    """Returns the index of the MASK token or None if unspecified."""
    return (None if self._mask_token is None else
            self._token_to_id[self._mask_token])

  @property
  def pad(self):
    """Returns the index of the PAD token or None if unspecified."""
    return (None
            if self._pad_token is None else self._token_to_id[self._pad_token])

  def is_valid(self, value):
    """Tests if a value is a valid token id and returns a bool."""
    return value in self._token_ids

  def are_valid(self, values):
    """Tests if values are valid token ids and returns an array of bools."""
    return np.array([self.is_valid(value) for value in values])

  def encode(self, tokens):
    """Maps an iterable of string tokens to a list of integer token ids."""
    if six.PY3 and isinstance(tokens, bytes):
      # Always use Unicode in Python 3.
      tokens = tokens.decode('utf-8')
    return [self._token_to_id[token] for token in tokens]

  def decode(self, values, stop_at_eos=False, as_str=True):
    """Maps an iterable of integer token ids to string tokens.

    Args:
      values: An iterable of token ids.
      stop_at_eos: Whether to ignore all values after the first EOS token id.
      as_str: Whether to return a list of tokens or a concatenated string.

    Returns:
      A string of tokens or a list of tokens if `as_str == False`.
    """
    if stop_at_eos and self.eos is None:
      raise ValueError('EOS unspecified!')
    tokens = []
    for value in values:
      value = int(value)  # Requires if value is a scalar tensor.
      if stop_at_eos and value == self.eos:
        break
      tokens.append(self._id_to_token[value])
    return ''.join(tokens) if as_str else tokens


@gin.configurable
class ProteinVocab(Vocabulary):
  """A `Vocabulary` with amino acid tokens."""

  def __init__(self,
               include_anomalous_amino_acids=False,
               include_align_tokens=False,
               **kwargs):
    """Creates an instance of this class.

    Args:
      include_anomalous_amino_acids: A bool indicating whether to also include
        the set of anomalous amino acids (vs. to use the standard ones only).
      include_align_tokens: A bool indicating whether to also include the tokens
        required to represent (fixed-length) aligned sequences.
      **kwargs: Named parameters to be passed to the base class.
    """
    tokens = list(seq_utils.AA_TOKENS)
    if include_anomalous_amino_acids:
      tokens += list(seq_utils.AA_ANOMALOUS_TOKENS)
    if include_align_tokens:
      tokens += list(seq_utils.AA_ALIGN_TOKENS)
    super(ProteinVocab, self).__init__(tokens=tokens, **kwargs)


@six.add_metaclass(abc.ABCMeta)
class Domain(object):
  """Base class of problem domains, which specifies the set of valid objects."""

  @property
  def mask_fn(self):
    """Returns a masking function or None."""

  @abc.abstractmethod
  def is_valid(self, sample):
    """Tests if the given sample is valid for this domain."""

  def are_valid(self, samples):
    """Tests if the given samples are valid for this domain."""
    return np.array([self.is_valid(sample) for sample in samples])


class DiscreteDomain(Domain):
  """Base class for discrete domains: sequences of categorical variables."""

  def __init__(self, vocab):
    self._vocab = vocab

  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def vocab(self):
    return self._vocab  # pytype: disable=attribute-error  # trace-all-classes

  def encode(self, samples, **kwargs):
    """Maps a list of string tokens to a list of lists of integer token ids."""
    return [self.vocab.encode(sample, **kwargs) for sample in samples]

  def decode(self, samples, **kwargs):
    """Maps list of lists of integer token ids to list of strings."""
    return [self.vocab.decode(sample, **kwargs) for sample in samples]


@gin.configurable
class FixedLengthDiscreteDomain(DiscreteDomain):
  """Output is a fixed length discrete sequence."""

  def __init__(self, vocab_size=None, length=None, vocab=None):
    """Creates an instance of this class.

    Args:
      vocab_size: An optional integer for constructing a vocab of this size.
        If provided, `vocab` must be `None`.
      length: The length of the domain (required).
      vocab: The `Vocabulary` of the domain. If provided, `vocab_size` must be
        `None`.

    Raises:
      ValueError: If neither `vocab_size` nor `vocab` is provided.
      ValueError: If `length` if not provided.
    """
    if length is None:
      raise ValueError('length must be provided!')
    if not (vocab_size is None) ^ (vocab is None):
      raise ValueError('Exactly one of vocab_size of vocab must be specified!')
    self._length = length
    if vocab is None:
      vocab = Vocabulary(vocab_size)
    super(FixedLengthDiscreteDomain, self).__init__(vocab)

  @property
  def length(self):
    return self._length

  @property
  def size(self):
    """The number of structures in the Domain."""
    return self.vocab_size**self.length

  def is_valid(self, sequence):
    return len(sequence) == self.length and self.vocab.are_valid(sequence).all()

  def sample_uniformly(self, num_samples, seed=None):
    random_state = utils.get_random_state(seed)
    return np.int32(
        random_state.randint(
            size=[num_samples, self.length], low=0, high=self.vocab_size))

  def index_to_structure(self, index):
    """Given an integer and target length, encode into structure."""
    structure = np.zeros(self.length, dtype=np.int32)
    tokens = [int(token, base=len(self.vocab))
              for token in np.base_repr(index, base=len(self.vocab))]
    structure[-len(tokens):] = tokens
    return structure

  def structure_to_index(self, structure):
    """Returns the index of a sequence over a vocabulary of size `vocab_size`."""
    structure = np.asarray(structure)[::-1]
    return np.sum(structure * np.power(len(self.vocab), range(len(structure))))


@gin.configurable
class VariableLengthDiscreteDomain(FixedLengthDiscreteDomain):
  """A domain for variable-length sequences."""

  def __init__(self, vocab, length, min_length=0):
    """Creates an instance of this class.

    Args:
      vocab: An instance of a `Vocabulary` with an EOS token.
      length: The maximum sequence length.
      min_length: The minimum sequence length.

    Raises:
      ValueError: If `vocab` does not have an EOS token.
    """
    if vocab.eos is None:
      raise ValueError('vocab must have an EOS token!')
    super(VariableLengthDiscreteDomain, self).__init__(
        length=length, vocab=vocab)
    self._min_length = min_length

  @property
  def length(self):
    return self._length

  @property
  def min_length(self):
    return self._min_length

  def is_valid(self, sequence):
    """Tests if `sequences` are valid for this domain."""
    unpadded_seq = seq_utils.unpad_sequences([sequence], self.vocab.eos)[0]
    return (len(unpadded_seq) >= self.min_length and
            len(unpadded_seq) <= self.length and
            self.vocab.are_valid(sequence).all() and
            seq_utils.sequences_end_with_value([sequence], self.vocab.eos)[0])

  def encode(self, sequences, pad=True):
    """Integer-encodes sequences and optionally pads them."""
    encoded = [self.vocab.encode(seq) for seq in sequences]
    if pad:
      encoded = seq_utils.pad_sequences(encoded, self.length, self.vocab.eos)
    return encoded

  def decode(self, sequences, stop_at_eos=True, **kwargs):
    """Integer-encodes sequences and optionally pads them."""
    return [self.vocab.decode(seq, stop_at_eos=stop_at_eos, **kwargs)
            for seq in sequences]

  def sample_uniformly(self,
                       num_samples,
                       min_seq_len=None,
                       max_seq_len=None,
                       pad=True,
                       seed=None):
    """Samples valid integer-encoded sequences from the domain.

    Args:
      num_samples: The number of samples.
      min_seq_len: The minimum sequence length of samples (inclusive).
      max_seq_len: The maximum sequence length of samples (inclusive).
      pad: Whether to pad sequences to the maximum length.
      seed: Optional seed of the random number generator.

    Returns:
      A list with `num_samples` samples.
    """
    if min_seq_len is None:
      min_seq_len = self.min_length
    if max_seq_len is None:
      max_seq_len = self.length
    random_state = utils.get_random_state(seed)
    valid_token_ids = np.delete(self.vocab.token_ids, self.vocab.eos)
    lengths = random_state.randint(min_seq_len, max_seq_len + 1, num_samples)
    seqs = [random_state.choice(valid_token_ids, length)
            for length in lengths]
    if pad:
      seqs = seq_utils.pad_sequences(seqs, self.length, self.vocab.eos)
    return seqs


def is_discrete(domain):
  """Returns a bool indicating whether `domain` is discrete."""
  return isinstance(domain, DiscreteDomain)


def check_if_discrete(domain):
  """Raises an exception if `domain` is not discrete."""
  if not is_discrete(domain):
    raise ValueError('Discrete domain expected!')


def is_variable_length(domain):
  """Returns a bool indicating whether `domain` is variable-length."""
  return isinstance(domain, VariableLengthDiscreteDomain)


def domain_to_bos(domain):
  """Returns a pre-specified start-sequence symbol or a new symbol (len(vocab)).

  If a new symbol is returned, it's not added to the vocabulary (only used as
  input at the beginning of sequence).

  Args:
    domain: The problem's Domain instance.
  """
  vocab = domain.vocab
  return len(vocab) if vocab.bos is None else vocab.bos
