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

"""Classes for tracing memory access calls."""

import abc
import collections
import csv
import os
import numpy as np
import six
import tqdm


class MemoryTrace(object):
  """Represents the ordered load calls for some program with a cursor.

  Should be used in a with block.
  """

  def __init__(self, filename, max_look_ahead=int(1e7), cache_line_size=64):
    """Constructs from a file containing the memory trace.

    Args:
      filename (str): filename of the file containing the memory trace. Must
        conform to one of the expected .csv or .txt formats.
      max_look_ahead (int): number of load calls to look ahead in
        most_future_access(). All addresses not been loaded by the
        max_look_ahead limit are considered tied.
      cache_line_size (int): size of cache line used in Cache reading this
        trace.
    """
    if cache_line_size & (cache_line_size - 1) != 0:
      raise ValueError("Cache line size must be power of two.")

    self._filename = filename
    self._max_look_ahead = max_look_ahead

    self._num_next_calls = 0

    # Maps address --> list of next access times in the look ahead buffer
    self._access_times = collections.defaultdict(collections.deque)
    self._look_ahead_buffer = collections.deque()
    self._offset_bits = int(np.log2(cache_line_size))

    # Optimization: only catch the StopIteration in _read_next once.
    # Without this optimization, the StopIteration is caught max_look_ahead
    # times.
    self._reader_exhausted = False

  def _read_next(self):
    """Adds the next row in the CSV memory trace to the look-ahead buffer.

    Does nothing if the cursor points to the end of the trace.
    """
    if self._reader_exhausted:
      return

    try:
      pc, address = self._reader.next()
      self._look_ahead_buffer.append((pc, address))
      # Align to cache line
      self._access_times[address >> self._offset_bits].append(
          len(self._look_ahead_buffer) + self._num_next_calls)
    except StopIteration:
      self._reader_exhausted = True

  def next(self):
    """The next load call under the cursor. Advances the cursor.

    Returns:
      load_call (tuple)
    """
    self._num_next_calls += 1
    pc, address = self._look_ahead_buffer.popleft()
    # Align to cache line
    aligned_address = address >> self._offset_bits
    self._access_times[aligned_address].popleft()

    # Memory optimization: discard keys that have no current access times.
    if not self._access_times[aligned_address]:
      del self._access_times[aligned_address]

    self._read_next()
    return pc, address

  def done(self):
    """True if the cursor points to the end of the trace."""
    return not self._look_ahead_buffer

  def next_access_time(self, address):
    """Returns number of accesses from cursor of next access of address.

    Args:
      address (int): cache-line aligned memory address (missing offset bits).

    Returns:
      access_time (int): np.inf if not accessed within max_look_ahead accesses.
    """
    accesses = self._access_times[address]
    if not accesses:
      return np.inf
    return accesses[0] - self._num_next_calls

  def __enter__(self):
    self._file = open(self._filename, "r")
    _, extension = os.path.splitext(self._filename)
    if extension == ".csv":
      self._reader = CSVReader(self._file)
    elif extension == ".txt":
      self._reader = TxtReader(self._file)
    else:
      raise ValueError(
          "Extension {} not a supported extension.".format(extension))

    # Initialize look-ahead buffer
    for _ in tqdm.tqdm(
        range(self._max_look_ahead), desc="Initializing MemoryTrace"):
      self._read_next()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()


class MemoryTraceReader(six.with_metaclass(abc.ABCMeta, object)):
  """Internal class for reading different memory trace formats."""

  def __init__(self, f):
    """Constructs around a file object to read.

    Args:
      f (File): file to read memory trace from.
    """
    self._file = f

  @abc.abstractmethod
  def next(self):
    """Returns the next (pc, address) in the file.

    Raises:
      StopIteration: when file is exhausted.

    Returns:
      pc (int): program counter of next memory access.
      address (int): memory address of the next access.
    """
    raise NotImplementedError()


class CSVReader(MemoryTraceReader):
  """Reads CSV formatted memory traces.

  Expects that each line is formatted as:
    pc,address

  where pc and address are hex strings.
  """

  def __init__(self, f):
    super(CSVReader, self).__init__(f)
    self._csv_reader = csv.reader(f)

  def next(self):
    # Raises StopIteration when CSV reader is eof
    pc, address = next(self._csv_reader)
    # Convert hex string to int
    return int(pc, 16), int(address, 16)


class TxtReader(MemoryTraceReader):
  """Reads .txt extension memory traces.

  Expects that each line is formatted as:
    instruction_type pc address

  where all entries are expressed in decimal
  """

  def next(self):
    line = next(self._file)
    _, pc, address = line.split()
    # Already in decimal
    return int(pc), int(address)


class MemoryTraceWriter(object):
  """Writes a memory trace to file."""

  def __init__(self, filename):
    """Constructs a writer to write to the provided filename.

    Args:
      filename (str): path to write trace to.
    """
    self._filename = filename

  def write(self, pc, address):
    """Writes the (pc, address) to disk.

    Args:
      pc (int): program counter of instruction causing read at the address.
      address (int): memory address accessed in the instruction.
    """
    self._csv_writer.writerow((hex(pc), hex(address)))

  def __enter__(self):
    self._file = open(self._filename, "w+")
    self._csv_writer = csv.writer(self._file)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()
