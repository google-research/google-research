# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Implements poisson subsampling transformation."""

from typing import TypeVar

import grain.python as pygrain
import numpy as np

T = TypeVar("T")


class PoissonMapDataset(pygrain.MapDataset[T]):
  """Poisson subsample the parent dataset with truncation and padding.

  Sample the parent dataset with Poisson subsampling. Batches are truncated to
  have the size at most `train_batch_size` and padded to `train_batch_size` if
  the batch size is less than `train_batch_size`, assuming that the last example
  of the parent dataset is a padding example.

  Attributes:
    train_batch_size: The batch size used for training, before gradient
      accumulation. This is the TPU batch size per TPU step.
    expected_batch_size: The expected batch size, used to compute the Poisson
      subsampling probability.
    training_steps: The total number of virtual batches used for training,
      across all epochs. The user should not call `repeat` on this dataset.
    seed: The seed to use for sampling, which comes from the parent dataset by
      default. If unspecified here or by the parent, a random seed is used.
    shuffle_indices: If true, return indices that are shuffled. This should only
      be set to False for debugging purposes.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent,
      *,
      train_batch_size,
      expected_batch_size,
      training_steps,
      seed = None,
      shuffle_indices = True,
  ):
    super().__init__(parent)
    seed = self._default_seed if seed is None else seed
    if seed is None:
      seed = np.random.randint(0, 2**32)
    self._seed = int(seed)
    self._shuffle_indices = shuffle_indices
    self._train_batch_size = train_batch_size
    self._training_steps = training_steps
    if self._train_batch_size < expected_batch_size:
      raise ValueError(
          "Train batch size must be greater than or equal to the expected batch"
          " size before gradient accumulation."
      )
    self._num_non_padding_examples = len(self._parent) - 1
    if self._num_non_padding_examples <= 0:
      raise ValueError("The parent dataset must have at least one example.")
    self._poisson_subsampling_probability = (
        expected_batch_size / self._num_non_padding_examples
    )
    if (
        self._poisson_subsampling_probability < 0
        or self._poisson_subsampling_probability > 1
    ):
      raise ValueError(
          "Poisson subsampling probability must be between 0 and 1."
          f" (got {self._poisson_subsampling_probability=})."
      )

  def __len__(self):
    # This returns the total number of batches during the whole training,
    # across all epochs. The user should not call `repeat` on this dataset.
    return self._train_batch_size * self._training_steps

  def __str__(self):
    return "PoissonMapDataset"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      batch, index_in_batch = divmod(index, self._train_batch_size)
      per_batch_seed = (self._seed + batch) % 2**32
      rng = np.random.default_rng(per_batch_seed)
      batch_size = rng.binomial(
          self._num_non_padding_examples,
          self._poisson_subsampling_probability,
      )
      if index_in_batch < int(batch_size):
        # Shuffle all examples and take the example at the corresponding index.
        if self._shuffle_indices:
          poisson_sampled_index = pygrain.experimental.index_shuffle(
              index_in_batch,
              max_index=self._num_non_padding_examples - 1,
              seed=per_batch_seed,
              rounds=4,  # Set to match the ShuffleMapDataset implementation.
          )
        else:
          # Return unshuffled indices for debugging.
          poisson_sampled_index = index_in_batch
        return self._parent[poisson_sampled_index]
      else:
        # padding example
        return self._parent[self._num_non_padding_examples]
