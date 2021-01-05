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

# Lint as: python3
"""Utils for training."""
import random

import numpy
import torch


def batchify(idxs, bsz, device, pad=0, shuffle=True):
  """Batchify the training data."""
  length = [len(seq) for seq in idxs]
  sorted_idx = numpy.argsort(length)
  idxs_sorted = [idxs[i] for i in sorted_idx]
  idxs_batched = []
  i = 0

  def get_batch(source, i, batch_size, pad=0):
    total_length = 0
    data = []
    while total_length < batch_size and i < len(source):
      data.append(source[i])
      total_length += len(source[i])
      i += 1
    length = [len(seq) for seq in data]
    max_l = max(length)
    data_padded = []
    for seq in data:
      data_padded.append(seq + [pad] * (max_l - len(seq)))
    data_mat = torch.LongTensor(data_padded).to(device)
    return data_mat

  while i < len(idxs_sorted):
    idxs_batched.append(get_batch(idxs_sorted, i, bsz, pad))
    i += idxs_batched[-1].size(0)

  if shuffle:
    sentence_idx = list(range(len(idxs_batched)))
    random.shuffle(sentence_idx)
    idxs_batched = [idxs_batched[i] for i in sentence_idx]

  return idxs_batched


