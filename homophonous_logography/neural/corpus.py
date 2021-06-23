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

"""Corpus parsing and I/O APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random

import numpy as np
import pandas as pd


class SymbolTable(object):
  """Container for storing symbol-to-integer and integer-to-symbol mappings."""

  def __init__(self):
    self._table = collections.defaultdict(int)
    self._inverse_table = collections.defaultdict(str)
    self._idx = 0
    self.add("<pad>")
    self.add("<s>")
    self.add("</s>")
    self.add("<targ>")
    self.add("</targ>")
    self.add("<spc>")
    self.add("<unk>")

  def __len__(self):
    return len(self._table)

  def add(self, sym):
    if sym == " ":
      sym = "<spc>"
    if sym in self._table:
      return
    self._table[sym] = self._idx
    self._inverse_table[self._idx] = sym
    self._idx += 1

  def find(self, key):
    """Returns the symbol given the key (can be numeric or string)."""
    if isinstance(key, int) or isinstance(key, np.int32):
      if key in self._inverse_table:
        return self._inverse_table[int(key)]
      else:
        return "<unk>"
    else:
      if key in self._table:
        return self._table[key]
      elif key == " ":
        return self._table["<spc>"]
      else:
        return self._table["<unk>"]


class Corpus(object):
  """Corpus abstraction."""

  def __init__(self, table):
    self._table = table
    self._written_symbol_table = SymbolTable()
    self._pronounce_symbol_table = SymbolTable()
    self._written_len = 0
    self._pronounce_len = 0
    self._max_written_len = 0
    self._max_pronounce_len = 0
    self._max_written_word_len = 0
    self._max_pronounce_word_len = 0
    for k in self._table:
      written_len = 0
      pronounce_len = 0
      for wp in self._table[k]:
        for c in wp[0]:
          self._written_symbol_table.add(c)
          self._written_len += 1
          written_len += 1
        if len(wp[0]) > self._max_written_word_len:
          self._max_written_word_len = len(wp[0])
        if len(wp) == 2:
          for c in wp[1]:
            self._pronounce_symbol_table.add(c)
            self._pronounce_len += 1
            pronounce_len += 1
          if len(wp[1]) > self._max_pronounce_word_len:
            self._max_pronounce_word_len = len(wp[1])
      if written_len > self._max_written_len:
        self._max_written_len = written_len
      if pronounce_len > self._max_pronounce_len:
        self._max_pronounce_len = pronounce_len
    self._max_written_len += 4  # accommodate <s>, </s>, <targ>, </targ>
    self._max_pronounce_len += 4
    self._max_written_word_len += 2  # accommodate <targ> and </targ>
    self._max_pronounce_word_len += 2
    self._written_padding = ([self._written_symbol_table.find("<pad>")] *
                             self._max_written_len)
    self._pronounce_padding = ([self._pronounce_symbol_table.find("<pad>")] *
                               self._max_pronounce_len)

  def __len__(self):
    return self._written_len

  @property
  def max_written_len(self):
    return self._max_written_len

  @property
  def max_pronounce_len(self):
    return self._max_pronounce_len

  @property
  def max_written_word_len(self):
    return self._max_written_word_len

  @property
  def max_pronounce_word_len(self):
    return self._max_pronounce_word_len

  @property
  def table(self):
    return self._table

  @property
  def written_symbol_table(self):
    return self._written_symbol_table

  @property
  def pronounce_symbol_table(self):
    return self._pronounce_symbol_table

  def cut(self, sequence, symbol_table):
    """Truncates the input sequence at the right enclosing target tag."""
    left_target = symbol_table.find("<targ>")
    right_target = symbol_table.find("</targ>")
    new_sequence = []
    for s in sequence:
      if s == right_target:
        new_sequence.append(s)
        break
      elif s == left_target:
        new_sequence.append(s)
      elif new_sequence:
        new_sequence.append(s)
    return new_sequence

  def sparse_vectors(self, key, target=(-1, None)):
    """Returns sparse array tuple representing written and phonemic sides."""
    written_values = []
    written_values.append(self._written_symbol_table.find("<s>"))
    pronounce_values = []
    pronounce_values.append(self._pronounce_symbol_table.find("<s>"))
    for i in range(len(self._table[key])):
      wp = self._table[key][i]
      if i == target[0]:
        written_values.append(self._written_symbol_table.find("<targ>"))
      for c in wp[0]:
        written_values.append(self._written_symbol_table.find(c))
      if i == target[0]:
        written_values.append(self._written_symbol_table.find("</targ>"))
      if len(wp) == 2:
        if i == target[0]:
          pronounce_values.append(self._pronounce_symbol_table.find("<targ>"))
        for c in wp[1]:
          pronounce_values.append(self._pronounce_symbol_table.find(c))
        if i == target[0]:
          pronounce_values.append(self._pronounce_symbol_table.find("</targ>"))
    written_values.append(self._written_symbol_table.find("</s>"))
    pronounce_values.append(self._pronounce_symbol_table.find("</s>"))
    if target[1] == "written":
      written_values = self.cut(written_values, self._written_symbol_table)
      written_values = (written_values +
                        self._written_padding)[:self._max_written_word_len]
    else:
      written_values = (written_values +
                        self._written_padding)[:self._max_written_len]
    if target[1] == "pronounce":
      pronounce_values = self.cut(pronounce_values,
                                  self._pronounce_symbol_table)
      pronounce_values = (pronounce_values +
                          self._pronounce_padding)[
                              :self._max_pronounce_word_len]
    else:
      pronounce_values = (pronounce_values +
                          self._pronounce_padding)[:self._max_pronounce_len]
    return (np.array(written_values, dtype=np.int32),
            np.array(pronounce_values, dtype=np.int32))

  # TODO(rws): This code is a hack and needs to be rewritten.
  def sparse_windowed_vectors(self, key, target=(-1, None), window=3):
    """Returns tuple representing (windowed) written and phonemic sides."""
    extended = [[[], []]] * window + self._table[key] + [[[], []]] * window
    left = target[0]
    middle = target[0] + window
    right = target[0] + window * 2 + 1
    written_values = []
    pronounce_values = []
    for i in range(left, right):
      wp = extended[i]
      assert len(wp) == 2
      assert "NULLPRON" not in "".join(wp[1])
      if target[1] == "written":
        if i == middle:
          written_values.append(self._written_symbol_table.find("<targ>"))
          for c in wp[0]:
            written_values.append(self._written_symbol_table.find(c))
          written_values.append(self._written_symbol_table.find("</targ>"))
          written_values = (written_values +
                            self._written_padding)[:self._max_written_word_len]
          pronounce_values.append(self._pronounce_symbol_table.find("<targ>"))
          for c in wp[1]:
            pronounce_values.append(self._pronounce_symbol_table.find(c))
          pronounce_values.append(self._pronounce_symbol_table.find("</targ>"))
        else:
          for c in wp[1]:
            pronounce_values.append(self._pronounce_symbol_table.find(c))
      else:
        if i == middle:
          pronounce_values.append(self._pronounce_symbol_table.find("<targ>"))
          for c in wp[1]:
            pronounce_values.append(self._pronounce_symbol_table.find(c))
          pronounce_values.append(self._pronounce_symbol_table.find("</targ>"))
          pronounce_values = (pronounce_values +
                              self._pronounce_padding)[
                                  :self._max_pronounce_word_len]
          written_values.append(self._written_symbol_table.find("<targ>"))
          for c in wp[0]:
            written_values.append(self._written_symbol_table.find(c))
          written_values.append(self._written_symbol_table.find("</targ>"))
        else:
          for c in wp[0]:
            written_values.append(self._written_symbol_table.find(c))
    # Trim to at most window + 1 rather than window * 2 + 1 since the window
    # includes the space "token".
    if target[1] == "written":
      length = self._max_pronounce_word_len * (window + 1)
      if length > len(self._pronounce_padding):
        length = len(self._pronounce_padding)
      pronounce_values = (pronounce_values + self._pronounce_padding)[:length]
    else:
      length = self._max_written_word_len * (window + 1)
      if length > len(self._written_padding):
        length = len(self._written_padding)
      written_values = (written_values + self._written_padding)[:length]
    return (np.array(written_values, dtype=np.int32),
            np.array(pronounce_values, dtype=np.int32))


def read_corpus(language, max_rows=-1, max_length=1000000, lower=False,
                datasets_dir=None):
  """Loads corpora."""
  # The "max_length" argument is the maximum sentence length (by tokens)
  # that we want to allow.
  #
  # "lower" is used for Middle Persian to reduce the heterograms to the
  # something more akin to the form that the scribes would have used (hence
  # only on the written side).
  source = "https://rws.xoba.com/.corpora/%s.tsv" % language
  if datasets_dir:
    source = os.path.join(datasets_dir, language + ".tsv")
  print("Reading corpus from \"{}\" ...".format(source))
  data = pd.read_csv(source, sep="\t", header=None, dtype=str)
  print("Number of original verses in file: {}".format(data.shape[0]))
  table = {}
  nrows = 0
  for _, row in data.iterrows():
    try:
      verse, text = row[0], row[1]
      if pd.isnull(text):
        continue
      text_list = []
      for wp in text.split():
        wp = wp.split("/")
        if len(wp) == 2:
          w, p = wp
          if lower:
            w = w.lower()
          if "_" in p:  # Our output for the CMU dict.
            p = p.split("_")
          else:
            p = list(p)
          text_list.append((w, p))
          text_list.append((" ", " "))
        else:
          text_list.append(wp)
          text_list.append(" ")
      if len(text_list) > max_length:
        continue
      table[verse] = text_list[:-1]
      nrows += 1
      if nrows == max_rows:
        break
    except ValueError:
      pass
  print("{}: Read {} rows".format(language, nrows))
  return table


def decode_index_array(array, symbols, joiner, skip_symbols):
  result = [symbols.find(i) for i in array]
  return joiner.join([s for s in result if s not in skip_symbols])


def batchify(corpus, batch_size=64, direction="pronounce", typ="train",
             window=-1):
  """Batching."""
  batch = 0
  written_batch = []
  pronounce_batch = []

  for key in corpus.table:
    if key.startswith("{}_".format(typ)) and (window > 0 or
                                              not key.endswith("_NULLPRON")):
      for i in range(len(corpus.table[key])):
        if window > 0:
          try:
            written, pronounce = corpus.sparse_windowed_vectors(
                key, (i, direction), window=window)
          except AssertionError:
            continue
        else:
          written, pronounce = corpus.sparse_vectors(key, (i, direction))
        written_batch.append(written)
        pronounce_batch.append(pronounce)
        if len(written_batch) == batch_size:
          written_batch = np.array(written_batch)
          pronounce_batch = np.array(pronounce_batch)
          if direction == "pronounce":
            yield batch, (written_batch, pronounce_batch)
          else:
            yield batch, (pronounce_batch, written_batch)
          batch += 1
          written_batch = []
          pronounce_batch = []
  yield -1, (None, None)


def num_batches(corpus, batch_size, direction="pronounce", typ="train",
                window=-1):
  batches = batchify(corpus, batch_size, direction=direction, typ=typ,
                     window=window)
  batch, (_, _) = next(batches)
  while batch != -1:
    last_batch = batch
    batch, (_, _) = next(batches)
  return last_batch


def test_examples(corpus, direction="pronounce", window=-1):
  """Returns test examples."""
  test_batches = batchify(corpus, batch_size=1, direction=direction, typ="test",
                          window=window)
  return [(i[0], o[0]) for (b, (i, o)) in test_batches if o is not None]


def random_test_indices(examples, k=-1):
  indices = list(range(len(examples)))
  if k == -1:
    k = len(examples)
  random.shuffle(indices)
  indices = indices[:k]
  indices.sort()
  print("Testing {} examples.".format(len(indices)))
  return indices
