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

"""Utilities for creating datasource items."""


import collections
import contextlib
import datetime
import json
from typing import Mapping, Optional, Sequence, TextIO, Union

from absl import logging
import numpy as np
from numpy import typing as np_typing
import pandas as pd
from study_recommend import types
file_open = open

FIELDS = types.StudentActivityFields

OUT_OF_VOCAB = types.Token('<OOV>')
UNDEF = types.Token('<UNDEF>')
SEPARATOR = types.Token('<SEP>')


class Vocabulary:
  """Utility for translating between book identifiers and token indices."""

  def __init__(
      self,
      vocab = None,
      oov_value = None,
  ):
    """Constructor.

    Args:
      vocab: A dictionary mapping book identifiers (strings) to token indices
        (integers)
      oov_value: Integer token index used to represent all book identifiers that
        are out of vocabulary (OOV).
    """
    self._lookup = {}
    if vocab is None:
      vocab = {}
    if 0 in vocab.values():
      raise ValueError('0 is the padding token')

    self._lookup.update(vocab)

    if oov_value is None:
      oov_value = max(vocab.values()) + 1 if vocab else 0
    if oov_value in self._lookup.values():
      raise ValueError('Out of Vocabulary token index must be unused.')
    self.oov_value = types.TokenIndex(oov_value)

    self._populate_reverse_lookup()

    # Assert every item in _lookup as well as UNKNOWN are in reverse lookup
    assert len(self._reverse_lookup) == (len(self._lookup) + 1)

  def _populate_reverse_lookup(self):
    """Populate reverse lookup dictionary to faciliate decoding."""
    self._reverse_lookup = {
        index: book_identifier
        for book_identifier, index in self._lookup.items()
    }
    self._reverse_lookup[self.oov_value] = OUT_OF_VOCAB

  def __getitem__(self, token):
    """Get the index for a given token. Returns oov_value for unknown tokens."""
    return self._lookup.get(token, self.oov_value)

  def decode(self, i):
    """Get the token for a given index. Returns '<UNDEF>' for unknown tokens."""
    return self._reverse_lookup.get(i, UNDEF)

  def encode(self, book_id):
    """Get the index for a given token. Returns oov_value for unknown tokens."""
    return self.__getitem__(book_id)

  def __len__(self):
    return len(self._reverse_lookup)

  def serialize(self, file):
    """Serialize self to file or text buffer.

    Args:
      file: Path-like or text buffer.
    """
    if isinstance(file, str):
      context_manager = file_open(file, 'w')
    else:
      context_manager = contextlib.nullcontext(file)

    with context_manager as f:
      json_dict = {'oov': str(self.oov_value)}
      json_dict['vocab'] = {}
      for book_identifier, index in self._lookup.items():
        json_dict['vocab'][book_identifier] = str(index)
      f.write(json.dumps(json_dict, indent=2))

  def deserialize(self, file):
    """Restore state from serialization stored a file or buffer.

    Args:
      file: Path or text buffer.

    Returns:
      self after restoring state from serialization.
    """
    if isinstance(file, str):
      context_manager = file_open(file, 'r')
    else:
      context_manager = contextlib.nullcontext(file)

    with context_manager as f:
      json_dict = json.loads(f.read())

    self.oov_value = types.TokenIndex(int(json_dict['oov']))
    self._lookup = {}
    for book_id, token in json_dict['vocab'].items():
      self._lookup[book_id] = types.TokenIndex(int(token))
    self._populate_reverse_lookup()
    return self


def build_vocab(
    books,
    n_tokens = None,
    special_tokens = tuple(),
):
  """Build a vocabulary of the n most popular books in a sequence of books.

  Args:
    books: A sequence of book identifiers
    n_tokens: The number of distinct items from <books> to put into the
      vocabulary. If None is passed all distinct items from books are placed
      into the vocabulary.
    special_tokens: A sequence of special tokens (e.g. separator token, padding,
      ...) to inject into the vocabulary. These items do not count against
      n_books.

  Returns:
    A Vocabulary object with the n_tokens most frequently occuring books
    and the special tokens as entries.
  """
  # Build a list of (book_identifier, frequency) pairs sorted descendingly
  # by frequency.
  counts = collections.Counter(books).items()
  counts = sorted(counts, reverse=True, key=lambda x: x[1])

  # Keep the n_token most popular pairs.
  if n_tokens:
    counts = counts[:n_tokens]
  # Keep only the book identifiers, drop the frequency counts.
  tokens = [x[0] for x in counts]

  tokens.extend(special_tokens)

  # Build an initial dictionary with token-index pairs.
  # Tokens indices start from 1 because 0 is reserved for padding.
  vocab = {
      types.Token(token): types.TokenIndex(i)
      for i, token in enumerate(tokens, start=1)
  }

  vocab = Vocabulary(vocab)
  logging.info('Built vocabulary with %d tokens in total', len(vocab))
  return vocab


@np.vectorize
def to_timestamp(string):
  """Convert string date (YYYY-MM-DD) to integer number of seconds since Epoch."""
  return round(
      datetime.datetime.strptime(string.split()[0], '%Y-%m-%d')
      .replace(tzinfo=datetime.timezone.utc)
      .timestamp()
  )


class UndefinedClassroom:
  """A placeholder value for undefined classrooms.

  This is NaN like value that when added to another value returns self.
  """

  def __add__(self, other):
    return self

  def __radd__(self, other):
    return self


def build_classroom_lookup(
    student_info,
    *,
    classroom_columns,
    offset = 100_000,
):
  """Build a dict mapping StudentIDs to ClassroomIDs.

  Args:
    student_info: A dataframe with one row per StudentID.  Contains one or more
      additional attributes per StudentID.
    classroom_columns: A pair of attributes to use for defining classrooms.
      students with equal values for both attributes are defined to be in the
      same classroom
    offset: If a student has missing values for one or both of the
      classroom_column attributes we place them in a unique classroom identified
      by the tuple (student_id + offset, student_id + offset). offset must be
      larger than the largest value in (classroom_columns) to avoid collision

  Returns:
    A dictionary mapping StudentIDs (int) to ClassroomIDs (tuple[int, int]).
  """

  lookup = {}

  for row in student_info.itertuples():

    def get_attribute(attribute, row=row):
      return getattr(row, attribute)

    student_id = get_attribute(FIELDS.STUDENT_ID.value)

    # Gather values of classroom_columns for current student
    classroom_identifier = []
    for column_name in classroom_columns:
      attribute_value = get_attribute(column_name)
      classroom_identifier.append(attribute_value)

    # If either value is undefined, put the student in their own classroom
    # defined from their own StudentID
    if np.isnan(classroom_identifier).any():
      classroom_identifier = (student_id + offset, student_id + offset)
    else:
      # Call round to cast values to int if they are floats.
      classroom_identifier = (
          round(classroom_identifier[0]),
          round(classroom_identifier[1]),
      )

    lookup[student_id] = types.ClassroomID(classroom_identifier)

  return lookup


def preprocess_dataframe(
    student_activity,
    vocab,
    seq_len,
    student_classrooms_lookup,
):
  """Preprocess input data frame in preparation to build a DataSource.

  Extract the sequences of titles, dates, corresponding to each student from
  the dataframe. For students with sequences larger than seq_len then the
  sequences will be chunked into sequences of length seq_len
  Args:
    student_activity: A dataframe of student-title interaction activity
    vocab: A vocabulary mapping book tokens to token indices
    seq_len: Maximum sequence length. Students with more interactions than
      seq_len will be split into multiple data point sequences of at most length
      seq_len
    student_classrooms_lookup: A dictionary mapping StudentID to classroom
      identifers.

  Returns:
    titles_array: A list of np.ndarray of titles. Each row corresponds to
      sequence of title TokenIndex'es of titles read by a single student.
    dates_array: A list of ndarray of identical dimensions to titles_array.
      Each entry in this array is the timestamp of the corresponding interaction
      documented in titles_array
    student_id_array: A list of StudentIDs (int). The i'th entry in
      this array is the StudentID corresponding the i'th rows of titles_array
      dates_array
    classroom_to_indices: A mapping from ClassroomID c_id to a
      Sequence[tuple[StartIndex, NumItems]].  For each <StartIndex, NumItems>
      pair the data in titles_array[StartIndex: StartIndex + NumItems]
      corresponds to a student inclassroom c_id.
  """

  student_activity = student_activity.copy().reset_index()

  # Convert student_activity[FIELDS.BOOK_ID] from str Token to int TokenIndex
  student_activity[FIELDS.BOOK_ID] = (
      student_activity[FIELDS.BOOK_ID].map(lambda x: vocab[x]).astype('int32')
  )

  # Convert student_activity to lists of arrays. Each array contains data
  # corresponding to one student. We create separate arrays for student_ids
  # titles and dates.
  student_activity = student_activity.sort_values(
      [FIELDS.STUDENT_ID, FIELDS.DATE], kind='stable'
  )
  student_activity = student_activity.groupby(FIELDS.STUDENT_ID)

  student_triplets = []
  for student_id, student_data in student_activity:
    student_triplets.append((
        student_id,
        student_data[FIELDS.BOOK_ID].values,
        student_data[FIELDS.DATE].values,
    ))

  student_ids, titles, dates = zip(*student_triplets)

  lengths = map(len, titles)
  titles_array = []
  dates_array = []
  student_id_array = []

  classroom_to_indices = collections.defaultdict(list)

  # Build output arrays while break arrays with more entries than seq_len into
  # chunks. We build classroom_to_indices incrementally as we build the output
  # arrays.
  current_index = 0
  for i, list_length in enumerate(lengths):
    classroom = student_classrooms_lookup[student_ids[i]]
    start_index, chunk_count = current_index, 0
    for j in range(0, list_length, seq_len):
      sample_titles = titles[i][j : j + seq_len]
      sample_dates = to_timestamp(dates[i][j : j + seq_len]).astype(np.int32)

      dates_array.append(sample_dates)
      titles_array.append(sample_titles)

      student_id_array.append(student_ids[i])
      chunk_count += 1

      current_index += 1
    classroom_to_indices[classroom].append(
        types.StudentIndexRange((start_index, chunk_count))
    )

  classroom_to_indices = dict(classroom_to_indices)

  return titles_array, dates_array, student_id_array, classroom_to_indices
