# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Data helper function for the Poker dataset."""

import tensorflow as tf

# Dataset size
# N_TRAIN_SAMPLES = 25010
N_TEST_SAMPLES = 1000000
NUM_FEATURES = 10
NUM_CLASSES = 10

# All feature columns in the data
LABEL_COLUMN = "Poker_Hand"
BOOL_COLUMNS = []
INT_COLUMNS = ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"]
STR_COLUMNS = []
STR_NUNIQUES = []
FLOAT_COLUMNS = []
DEFAULTS = ([[0] for col in INT_COLUMNS] + [[""] for col in BOOL_COLUMNS] +
            [[0.0] for col in FLOAT_COLUMNS] + [[""] for col in STR_COLUMNS] +
            [[-1]])

FEATURE_COLUMNS = (
    INT_COLUMNS + BOOL_COLUMNS + STR_COLUMNS + FLOAT_COLUMNS)
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]


def get_columns():
  """Get the representations for all input columns."""

  columns = []
  columns += [tf.feature_column.numeric_column(ci) for ci in INT_COLUMNS]
  return columns


def parse_csv(value_column):
  """Parses a CSV file based on the provided column types."""
  columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
  features = dict(zip(ALL_COLUMNS, columns))
  label = features.pop(LABEL_COLUMN)
  classes = tf.cast(label, tf.int32)
  return features, classes


def input_fn(data_file,
             num_epochs,
             shuffle,
             batch_size,
             n_buffer=50,
             n_parallel=16):
  """Function to read the input file and return the dataset.

  Args:
    data_file: Name of the file.
    num_epochs: Number of epochs.
    shuffle: Whether to shuffle the data.
    batch_size: Batch size.
    n_buffer: Buffer size.
    n_parallel: Number of cores for multi-core processing option.

  Returns:
    The Tensorflow dataset.
  """

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=n_buffer)

  dataset = dataset.map(parse_csv, num_parallel_calls=n_parallel)

  # Repeat after shuffling, to prevent separate epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset
