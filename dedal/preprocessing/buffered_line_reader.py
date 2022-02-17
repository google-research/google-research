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

"""Implements a buffered "line" reader with arbitrary "line" delimiter."""


from typing import BinaryIO, Generator


class BufferedLineReader:
  """Generates text "lines" from file opened in binary mode.

  Provides explicit support for buffered reads and custom "line" separators.
  Buffering appears to improve throughput in preliminary, Colab-based tests by
  about 5% and up to 35% (median 17%) in files consisting of many short lines.

  Attributes:
    f: A "raw" I/O stream, i.e. file opened in binary model.
    sep: The byte string to be used as line delimiter.
    buffer_size: The number of bytes to read in each I/O operation.
    start_pos: (Zero-based) index of the first byte to be read in the file.
    strip: Whether to remove the line delimiter from generated lines.
  """

  def __init__(
      self,
      f,
      sep = b'\n',
      buffer_size = 1024,
      start_pos = 0,
      strip = True):
    self._f = f
    self._sep = sep
    self._buffer_size = buffer_size
    self._pos = start_pos
    self._strip = strip

    self._f.seek(self._pos)

  def __iter__(self):
    eof = False
    buffer = b''
    while not eof:
      read_bytes = self._f.read(self._buffer_size)
      eof = len(read_bytes) < self._buffer_size
      buffer += read_bytes
      while buffer:
        split_buffer = buffer.split(self._sep, 1)
        if len(split_buffer) == 2:
          next_line, buffer = split_buffer
          next_line = next_line if self._strip else next_line + self._sep
        elif eof:  # Edge case: files that aren't properly terminated
          next_line, buffer = buffer, b''
        else:
          break
        self._pos += len(next_line)
        yield next_line.decode()

  def tell(self):
    """Returns the current position of the file read pointer within the file."""
    return self._pos  # in bytes from the start of the file
