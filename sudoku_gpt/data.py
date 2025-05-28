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

"""Data loading procedure for Othello and Sudoku game.
"""

import itertools
import pickle

import jax
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import gfile


def create_dataset(config, bs, train=True):
  """Create Othello or Sudoku dataset according to the config.

  Args:
    config: a config object containing the hyparameters for the dataset
      creation.
    bs: batch size
    train: whether the dataset is for train or eval

  Returns:

  """
  ds, output_types, output_shapes = None, None, None
  if config.dataset == "othello":
    ds = OthelloDataset(config, train=train)
    output_types = (tf.int32)
    output_shapes = tf.TensorShape([config.seq_len])
  elif (
      config.dataset == "sudoku"
      or config.dataset == "ordered-sudoku"
      or config.dataset
      == "ordered-sudoku-wo-random-guessing-w-candidates-train-test"
  ):
    ds = SudokuDataset(config, train=train)
    if config.start_index == "puzzle-dependent":
      # Output = (training seq, puzzle solution, start index)
      output_types = (tf.int32, tf.int32, tf.int32)
      output_shapes = (
          tf.TensorShape([config.seq_len]),
          tf.TensorShape([config.block_size]),
          tf.TensorShape([]),
      )
    else:
      output_types = (tf.int32, tf.int32)
      output_shapes = (
          tf.TensorShape([config.seq_len]),
          tf.TensorShape([config.block_size]),
      )

  tf_ds = tf.data.Dataset.from_generator(
      generator=ds, output_types=output_types, output_shapes=output_shapes)

  tf_ds = tf_ds.repeat()
  tf_ds = tf_ds.shuffle(8 * config.minibatch_size, seed=0)
  tf_ds = tf_ds.batch(bs)
  return tf_ds


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  # local_device_count = jax.local_device_count()
  def _prepare(x):
    x = x._numpy()  # pylint: disable=protected-access
    return x
  return jax.tree.map(_prepare, xs)


def create_iter(config, bs, train=True):
  tf_ds = create_dataset(config, bs, train=train)
  it = map(prepare_tf_data, tf_ds)
  return it


class OthelloDataset:
  """Othello dataset."""

  def __init__(self, config, train=True):
    self.config = config
    self.train = train
    self.dataset_path = config.dataset_path
    self.files = gfile.ListDir(config.dataset_path)
    self.sequences = []
    for fl in self.files:
      with gfile.Open(self.dataset_path + "/" + fl, "rb") as f:
        seq = pickle.load(f)
        if len(seq) >= 9e4:
          self.sequences.extend(seq)
    seq = self.sequences
    seq.sort()
    self.sequences = [k for k, _ in itertools.groupby(seq)]

    # self.train_len = int(len(self.sequences) * 0.9)
    # self.tot_len = len(self.sequences)

    self.eval_sequences = self.sequences[20000000:]
    self.sequences = self.sequences[:20000000]

  def __len__(self):
    if not self.train:
      return len(self.eval_sequences)
    else:
      return len(self.sequences)

  def __getitem__(self, idx):
    if not self.train:
      if len(self.eval_sequences[idx]) < 60:
        self.eval_sequences[idx].extend([(self.config.vocab_size - 1)]
                                        * (60-len(self.eval_sequences[idx])))
      return self.eval_sequences[idx]
    else:
      if len(self.sequences[idx]) < 60:
        self.sequences[idx].extend([(self.config.vocab_size - 1)]
                                   * (60 - len(self.sequences[idx])))
      return self.sequences[idx]

  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)


def check_valid_sudoku_puzzle(puzzle):
  """Checkes if a puzzle is a valid sudoku puzzle.

  Args:
    puzzle: A list of values in cells of a 9x9 board indexed in a linear fashion
      from 0 to 80.

  Returns:
    A boolean indicating puzzle validity.
  """
  rows = np.zeros((9, 9))
  cols = np.zeros((9, 9))
  boxes = np.zeros((9, 9))

  for i in range(len(puzzle)):
    row_num = int(i // 9)
    col_num = int(i % 9)
    rows[row_num, int(puzzle[i] - 1)] += 1
    cols[col_num, int(puzzle[i] - 1)] += 1
    boxes[int(3 * (row_num // 3) + (col_num // 3)), int(puzzle[i] - 1)] += 1

  if np.all(rows) and np.all(cols) and np.all(boxes):
    return True
  else:
    return False


class SudokuDataset:
  """Sudoku dataset."""

  def __init__(self, config, train=True):
    self.config = config
    self.dataset_path = config.dataset_path
    self.train = train

    if config.dataset == "sudoku":
      self.preprocess_sudoku()
    elif config.dataset == "ordered-sudoku":
      self.preprocess_ordered_sudoku()
    elif (
        config.dataset
        == "ordered-sudoku-wo-random-guessing-w-candidates-train-test"
    ):
      self.preprocess_ordered_sudoku_candidates_train_test()

    if config.dataset == "sudoku" or config.dataset == "ordered-sudoku":
      self.train_inputs = self.inputs[: self.train_len, :]
      self.eval_inputs = self.inputs[self.train_len : self.tot_len, :]

      self.train_puzzles = self.puzzles[: self.train_len, :, :]
      self.eval_puzzles = self.puzzles[self.train_len : self.tot_len, :, :]

  def get_puzzles_start_index(self, path):
    with gfile.Open(path, "rb") as f:
      inputs_with_start_index = np.load(f)
    start_index = inputs_with_start_index[:, 0]
    inputs = np.delete(
        inputs_with_start_index[:, 1:], np.arange(81) * 4 + 3, axis=1)
    puzzles = np.zeros((len(inputs), 81), dtype=np.int8)
    for j in range(81):
      cell_id = inputs[:, 3 * j] * 9 + inputs[:, 3 * j + 1]
      puzzles[np.arange(len(inputs)), cell_id] = inputs[:, 3 * j + 2]
    return inputs, puzzles, start_index

  def preprocess_ordered_sudoku_candidates_train_test(self):
    self.train_inputs, self.train_puzzles, self.train_start_index = (
        self.get_puzzles_start_index(self.config.train_puzzle_path)
    )

    self.eval_inputs, self.eval_puzzles, self.eval_start_index = (
        self.get_puzzles_start_index(self.config.test_puzzle_path)
    )

  def preprocess_ordered_sudoku(self):
    """Preprocess ordered sudoku dataset."""
    with gfile.Open(self.dataset_path, "rb") as f:
      arr_st = pickle.load(f)

    arr = np.zeros((len(arr_st), self.config.seq_len), dtype=np.int8)

    for i, _ in enumerate(arr_st):
      arr[i, :] = np.int8(list("".join(arr_st[i].split("\n "))))

    zero_rows = np.where(~arr.any(axis=1))[0]
    self.inputs = np.delete(arr, zero_rows, axis=0)
    self.puzzles = np.zeros(
        (len(self.inputs), self.config.block_size), dtype=np.int8
    )

    for i in range(len(self.inputs)):
      for j in range(self.config.block_size):
        # position is (row*9 + col)
        pos = self.inputs[i, 3 * j] * 9 + self.inputs[i, 3 * j + 1]
        self.puzzles[i, pos] = self.inputs[i, 3 * j + 2]

        ### Converting [0, 8] to [1, 9] index
        self.inputs[i, 3 * j] += 1
        self.inputs[i, 3 * j + 1] += 1

      assert check_valid_sudoku_puzzle(self.puzzles[i])

    self.train_len = int(len(self.inputs) * 0.9)
    self.tot_len = len(self.inputs)

  def preprocess_sudoku(self):
    """Preprocess sudoku dataset."""
    with gfile.Open(self.dataset_path, "r") as f:
      arr = np.loadtxt(f, delimiter=",", dtype=str)

    np.random.shuffle(arr[1:])
    self.puzzles = np.zeros((len(arr) - 1, 2, self.config.block_size))
    self.inputs = np.zeros((len(arr) - 1, self.config.seq_len))
    self.difficulty = np.zeros((len(arr)-1, 2))

    for i in range(len(arr) - 1):
      arr_int = map(int, list(arr[i+1][1].replace(".", "0")))
      self.puzzles[i, 0, :] = np.array(list(arr_int))
      self.puzzles[i, 1, :] = np.array(list(map(int, list(arr[i+1][2]))))
      self.difficulty[i, 0] = float(arr[i+1, 3])
      self.difficulty[i, 1] = float(arr[i+1, 4])

      nonzero_inds = np.where(self.puzzles[i, 0, :] != 0)[0]
      inp_puzzle = np.vstack(((nonzero_inds // 9) + 1,
                              (nonzero_inds % 9) + 1,
                              self.puzzles[i, 0, nonzero_inds])).flatten("F")

      zero_inds = np.where(self.puzzles[i, 0, :] == 0)[0]
      sol = np.vstack(((zero_inds // 9) + 1,
                       (zero_inds%9) + 1,
                       self.puzzles[i, 1, zero_inds])).flatten("F")

      self.inputs[i, :] = np.concatenate((inp_puzzle, sol)).astype(np.int32)

    self.train_len = int(len(self.inputs) * 0.9)
    self.tot_len = len(self.inputs)
    self.puzzles = self.puzzles[:, 1, :]

  def __len__(self):
    if not self.train:
      return len(self.eval_puzzles)
    else:
      return len(self.train_puzzles)

  def __getitem__(self, idx):
    if self.config.start_index == "puzzle-dependent":
      if not self.train:
        return (
            self.eval_inputs[idx, :],
            self.eval_puzzles[idx, :],
            self.eval_start_index[idx],
        )
      else:
        return (
            self.train_inputs[idx, :],
            self.train_puzzles[idx, :],
            self.train_start_index[idx],
        )
    if not self.train:
      return self.eval_inputs[idx, :], self.eval_puzzles[idx, :]
    else:
      return self.train_inputs[idx, :], self.train_puzzles[idx, :]

  def __call__(self):
    for i in range(self.__len__()):
      yield self.__getitem__(i)
