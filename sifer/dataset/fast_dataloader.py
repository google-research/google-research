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

"""Fast Dataloaders."""

from typing import Optional, Sized
import torch


class _InfiniteSampler(torch.utils.data.Sampler):
  """Wraps another Sampler to yield an infinite stream."""

  def __init__(self, sampler):
    self.sampler = sampler

  def __iter__(self):
    while True:
      for batch in self.sampler:
        yield batch


class InfiniteDataLoader:
  """DataLoader wrapper with infinite sampler."""

  def __init__(
      self,
      dataset,
      batch_size,
      num_workers,
      weights = None,
  ):
    super().__init__()

    if weights is not None:
      sampler = torch.utils.data.WeightedRandomSampler(
          weights, replacement=True, num_samples=batch_size
      )
    else:
      sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=batch_size, drop_last=True
    )

    self._infinite_iterator = iter(
        torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
        )
    )

  def __iter__(self):
    while True:
      yield next(self._infinite_iterator)

  def __len__(self):
    raise ValueError


class FastDataLoader:
  """DataLoader wrapper with slightly improved speed by not respawning worker processes at every epoch."""

  def __init__(self, dataset, batch_size, num_workers):
    super().__init__()

    batch_sampler = torch.utils.data.BatchSampler(
        torch.utils.data.RandomSampler(dataset, replacement=False),
        batch_size=batch_size,
        drop_last=False,
    )

    self._infinite_iterator = iter(
        torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
        )
    )

    self._length = len(batch_sampler)
    self.dataset = dataset

  def __iter__(self):
    for _ in range(len(self)):
      yield next(self._infinite_iterator)

  def __len__(self):
    return self._length
