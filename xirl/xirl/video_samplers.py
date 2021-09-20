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

"""Video samplers for mini-batch creation."""

import abc
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

ClassIdxVideoIdx = Tuple[int, int]
DirTreeIndices = List[List[ClassIdxVideoIdx]]
VideoBatchIter = Iterator[List[ClassIdxVideoIdx]]


class VideoBatchSampler(abc.ABC, Sampler):
  """Base class for all video samplers."""

  def __init__(
      self,
      dir_tree,
      batch_size,
      sequential=False,
  ):
    """Constructor.

    Args:
      dir_tree: The directory tree of a `datasets.VideoDataset`.
      batch_size: The number of videos in a batch.
      sequential: Set to `True` to disable any shuffling or randomness.
    """
    assert isinstance(batch_size, int)

    self._batch_size = batch_size
    self._dir_tree = dir_tree
    self._sequential = sequential

  @abc.abstractmethod
  def _generate_indices(self):
    """Generate batch chunks containing (class idx, video_idx) tuples."""
    pass

  def __iter__(self):
    idxs = self._generate_indices()
    if self._sequential:
      return iter(idxs)
    return iter(idxs[i] for i in torch.randperm(len(idxs)))

  def __len__(self):
    num_vids = 0
    for vids in self._dir_tree.values():
      num_vids += len(vids)
    return num_vids // self.batch_size

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def dir_tree(self):
    return self._dir_tree


class RandomBatchSampler(VideoBatchSampler):
  """Randomly samples videos from different classes into the same batch.

  Note the `sequential` arg is disabled here.
  """

  def _generate_indices(self):
    # Generate a list of video indices for every class.
    all_idxs = []
    for k, v in enumerate(self._dir_tree.values()):
      seq = list(range(len(v)))
      all_idxs.extend([(k, s) for s in seq])
    # Shuffle the indices.
    all_idxs = [all_idxs[i] for i in torch.randperm(len(all_idxs))]
    # If we have less total videos than the batch size, we pad with clones
    # until we reach a length of batch_size.
    if len(all_idxs) < self._batch_size:
      while len(all_idxs) < self._batch_size:
        all_idxs.append(all_idxs[np.random.randint(0, len(all_idxs))])
    # Split the list of indices into chunks of len `batch_size`.
    idxs = []
    end = self._batch_size * (len(all_idxs) // self._batch_size)
    for i in range(0, end, self._batch_size):
      batch_idxs = all_idxs[i:i + self._batch_size]
      idxs.append(batch_idxs)
    return idxs


class SameClassBatchSampler(VideoBatchSampler):
  """Ensures all videos in a batch belong to the same class."""

  def _generate_indices(self):
    idxs = []
    for k, v in enumerate(self._dir_tree.values()):
      # Generate a list of indices for every video in the class.
      len_v = len(v)
      seq = list(range(len_v))
      if not self._sequential:
        seq = [seq[i] for i in torch.randperm(len(seq))]
      # Split the list of indices into chunks of len `batch_size`,
      # ensuring we drop the last chunk if it is not of adequate length.
      batch_idxs = []
      end = self._batch_size * (len_v // self._batch_size)
      for i in range(0, end, self._batch_size):
        xs = seq[i:i + self._batch_size]
        # Add the class index to the video index.
        xs = [(k, x) for x in xs]
        batch_idxs.append(xs)
      idxs.extend(batch_idxs)
    return idxs


class SameClassBatchSamplerDownstream(SameClassBatchSampler):
  """A same class batch sampler with a batch size of 1.

  This batch sampler is used for downstream datasets. Since such datasets
  typically load a variable number of frames per video, we are forced to use
  a batch size of 1.
  """

  def __init__(
      self,
      dir_tree,
      sequential=False,
  ):
    super().__init__(dir_tree, batch_size=1, sequential=sequential)
