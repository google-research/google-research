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

"""Collection of various utilities."""

import math
import numpy as np
import tensorflow.compat.v1 as tf
import torch


def log_scalar(tb_writer, key, value, step):
  """Writes a scalar to the tensorboard writer.

  Args:
    tb_writer (tf.SummaryWriter): tensorboard writer.
    key (string): key to log.
    value (float): value to log.
    step (int): step number to log at.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
  tb_writer.add_summary(summary, step)


def as_batches(parallel_data, batch_size, sequence_length):
  """Iterable of batches of sequences of consecutive data of sequence_length.

  A single pass through this iterable will include all starting positions in
  each of the parallel sequences in data exactly once.

  Args:
    parallel_data (list[list[Object]]): parallel sequences of consecutive
      timesteps of data. Resulting batches will only include consecutive
      subsequences within a single parallel sequence of data.
    batch_size (int): size of the batches. Last batch may contain fewer than
      batch_size sequences.
    sequence_length (int): length of sequences to return.

  Yields:
    list[list[Object]]: the outer list is length batch size, the inner lists are
      all length sequence_length. Inner lists are all consecutive.
  """
  positions = []
  for i, data in enumerate(parallel_data):
    positions.extend([(i, start_pos)
                      for start_pos in range(len(data) - sequence_length)])
  np.random.shuffle(positions)

  for i in range(math.ceil(len(positions) / batch_size)):
    batch = [
        parallel_data[index][start: start + sequence_length]
        for index, start in positions[i * batch_size: (i + 1) * batch_size]]
    yield batch


def pad(seq_batch, pad_token=0, min_len=None):
  """Pads a list[list[Object]] so that each inner list is the same length.

  Args:
    seq_batch (list[list[Object]]): batch of sequences to pad.
    pad_token (Object): object to pad sequences with.
    min_len (int | None): all sequences padded to at least min_length if
      provided.

  Returns:
    padded (list[list[Object]]): seq_batch with inner lists padded with
      the pad token, if necessary.
    mask (torch.ByteTensor): tensor of shape (batch_size, padded_length).
      mask[i][j] = 0 if padded[i][j] is a padded element and 1 otherwise.
  """
  max_len = max(len(seq) for seq in seq_batch)
  if min_len is not None:
    max_len = max(max_len, min_len)

  batch_size = len(seq_batch)
  mask = torch.ones(batch_size, max_len).byte()

  padded = []
  for i, seq in enumerate(seq_batch):
    padding = max_len - len(seq)
    padded.append(seq + [pad_token] * padding)
    if padding > 0:
      mask[i, -padding:] = 0
  return padded, mask


def mask_renormalize(probs, mask):
  """Renormalizes probs with a mask so that the unmasked entries sum to 1.

  Args:
    probs (torch.FloatTensor): batch of probability distributions of shape
      (batch_dim1, batch_dim2, ..., num_elems).
    mask (torch.ByteTensor): mask of same shape as probs, where entries = 0
      should be masked out.

  Returns:
    renormalized_probs (torch.FloatTensor): tensor of same shape as probs. Each
      batch row (last dim) sums to 1, where masked entries have 0 prob. If all
      entries in a batch are masked, the batch row sums to 0.
  """
  masked_probs = probs * mask.float()
  renormalized = masked_probs / (masked_probs.sum(-1, keepdim=True) + 1e-8)
  return renormalized
