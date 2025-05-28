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

"""A global vocabulary."""
import copy
import logging
import threading
from typing import List, Mapping, Union

Number = Union[int, float]
VocabularyElement = Union[str, Number]


class Error(Exception):
  """Base exception for vocabulary errors."""


class VocabularyOverflowError(Error):
  """Raised when the vocabulary has overflowed its max word count."""


class Vocabulary():
  """A child of a global LockedVocabulary object."""

  def __init__(self, global_vocab_node,
               max_vocabulary_size = 15000):
    """Initialize the global vocabulary.

    Args:
      global_vocab_node: A reference to the global LockedVocabulary node. This
        is self-referential for the leader.
      max_vocabulary_size: The maximum size for the vocabulary.
    """
    # Local nodes will have this set. The global node will set to None.
    self._global_vocab_node = global_vocab_node

    self._max_vocabulary_size = max_vocabulary_size

    # The child's local view of the vocabulary. For the leader, this is the
    # global state.
    self._local_vocab = {}

  @property
  def local_vocab(self):
    return self._local_vocab

  @property
  def max_vocabulary_size(self):
    return self._max_vocabulary_size

  def __getitem__(self, key):
    """Allows dictionary access to the vocabulary."""
    return self._local_vocab[key]

  def __len__(self):
    """Gets the dictionary length of the vocabulary."""
    return len(self._local_vocab)

  def __contains__(self, item):
    """Checks if an item is in the vocabulary."""
    return item in self._local_vocab

  def add_to_vocabulary(
      self,
      words_to_add):
    """Add elements to the global vocabulary.

    Args:
      words_to_add: words to add to the vocabulary

    Returns:
      The updated vocabulary.
    """
    if not words_to_add:
      self._local_vocab = dict(self.get_global_vocabulary())
      return self._local_vocab

    self._local_vocab = dict(
        self._global_vocab_node.add_to_vocabulary(words_to_add))
    return self._local_vocab

  def get_global_vocabulary(self):
    """Return the global vocabulary."""
    return self._global_vocab_node.get_global_vocabulary()

  def save(self):
    """Overridden abstract method for saving the vocabulary object."""
    return self._global_vocab_node.save()

  def restore(self, state):
    """Overridden abstract method for restoring the vocabulary object."""
    self._global_vocab_node.restore(state)
    self._local_vocab = self._global_vocab_node.get_global_vocabulary()


class LockedVocabulary(Vocabulary):
  """A lockable vocabulary node intended to be the leader vocabulary node.

  This is used to protect and update the global vocab data structure.
  """

  def __init__(self, max_vocabulary_size = 15000):
    """Initialize the locked global vocabulary."""
    super().__init__(self, max_vocabulary_size=max_vocabulary_size)

    self._lock = threading.Lock()
    self._next_index = 0

  def add_to_vocabulary(
      self,
      words_to_add):
    """Add words to the global vocabulary until a specified limit.

    Args:
      words_to_add: A list of words to add to the global vocabulary.

    Returns:
      The updated global vocabulary.

    Raises:
      VocabularyOverflowError: Raised when the vocabulary has exceeded its max
        size.
    """
    with self._lock:
      for word in words_to_add:
        if self._next_index >= self.max_vocabulary_size:
          raise VocabularyOverflowError(
              f'The maximum vocabulary size of {self.max_vocabulary_size} '
              'has been exceeded.')
        if word in self._local_vocab:
          continue
        logging.info('Adding %s into the vocabulary at index %d.',
                     word, self._next_index)
        self._local_vocab[word] = self._next_index
        self._next_index += 1

      return self._local_vocab

  def get_global_vocabulary(self):
    """Return the global vocabulary."""
    return self._local_vocab

  def save(self):
    """Overridden abstract method for saving the LockedVocabulary object."""
    return {'global_vocab': copy.copy(self._local_vocab)}

  def restore(self, state):
    """Overridden abstract method for restoring the LockedVocabulary object."""
    with self._lock:
      global_vocab = state['global_vocab']
      self._local_vocab = global_vocab
      max_index = max(global_vocab.values()) + 1 if global_vocab.values() else 1
      self._next_index = max_index
