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

"""Simple utilities."""

import bz2
import gzip
import logging
import re

import numpy as np
import pandas as pd

# Column ID of perplexity in the tsv file.
_PERPLEXITY_COLUMN_ID = 2

# Regular expression for matching the ngram order.
_ORDER_REGEX = r"\d+gram"


def ngram_order_from_filename(filename):
  """Returns n-gram order from a file name."""
  orders = re.findall(_ORDER_REGEX, filename)
  if len(orders) != 1:
    raise ValueError(f"Invalid filename {filename}")
  order = orders[0][0:orders[0].find("gram")]
  return int(order)


def open_file(filename, mode="r", encoding="utf-8"):
  """Open files of several types, with text mode of compressed files.

  Args:
    filename: File path to the file which need to be open.
    mode: Open mode, "r" for read and "w" for write.
    encoding: Encoding method for the content of the file.

  Returns:
    The opened file handle of the input filename.
  """
  if filename.endswith(".gz"):
    # The "t" is appended for text mode.
    return gzip.open(filename, mode + "t")
  elif filename.endswith(".bz2"):
    return bz2.open(filename, mode + "t", encoding=encoding)
  else:
    return open(filename, mode, encoding=encoding)


def ppl_to_entropy(ppl):
  """Converts the perplexity to entropy (bits per character)."""
  return np.log10(ppl) / np.log10(2.0)


def read_metrics(file_path):
  """Reads metrics provided in a tsv file into pandas dataframe."""
  logging.info(f"Reading metrics from {file_path} ...")
  df = pd.read_csv(file_path, sep="\t", header=None)
  logging.info(f"Read {df.shape[0]} samples")
  return df


def read_entropies(file_path, as_ppl=False):
  """Reads entropies (or perplexities) from the pandas data frame."""
  df = read_metrics(file_path)
  ppl = df[_PERPLEXITY_COLUMN_ID]
  if as_ppl:
    return ppl
  else:
    return ppl_to_entropy(ppl)
