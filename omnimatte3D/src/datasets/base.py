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

"""Base dataset script."""

import abc
import copy
import queue
import threading

import jax

from omnimatte3D.src.utils import data_utils


class Dataset(threading.Thread, metaclass=abc.ABCMeta):
  """Dataset Base Class.

  # Adopted from
  https://github.com/google-research/multinerf/blob/main/internal/datasets.py
  Base class for datasets.
  Each subclass is responsible for loading images data from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches.
  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches
  are loaded/created or how this is parallelized. Therefore, the initializer
  runs all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.
  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops
  1 element at a time off the front of the queue. The Dataset thread's run()
  loop will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.
  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a
  daemon.
  Attributes:
    size: int, number of images in the dataset.
    split: str, indicate if the split is train or test.
  """

  def __init__(self, split, config):
    super().__init__()

    # Initialize attributes
    self._queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True  # Sets parent Thread to be a daemon.
    self._batch_size = config.dataset.batch_size // jax.process_count()
    self.split = split
    self._n_examples = None
    self._test_camera_idx = 0

    # Load data from disk using provided config parameters.
    self._load_renderings(config)
    if self._n_examples is None:
      raise ValueError('n_examples need to be set in _load_renderings')

    # Seed the queue with one batch to avoid race condition.
    if self.split == 'train':
      self._next_fn = self._next_train
    else:
      self._next_fn = self._next_test
    self._queue.put(self._next_fn())
    self.start()

  def __iter__(self):
    return self

  def cardinality(self):
    return self._n_examples

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict.
    """
    x = self._queue.get()
    return data_utils.shard(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict.
    """
    x = copy.copy(self._queue.queue[0])  # Make a copy of front of queue.
    return data_utils.shard(x)

  def run(self):
    while True:
      self._queue.put(self._next_fn())

  @property
  def size(self):
    return self._n_examples

  @abc.abstractmethod
  def _load_renderings(self, config):
    """Load images and poses from disk.

    Args:
      config: data_utils.Config, user-specified config parameters. In inherited
        classes, this method must set the following public attributes.
    """

  def _next_train(self):
    """Sample next training batch."""
    raise NotImplementedError

  def _next_test(self):
    """Sample next test batch."""
    raise NotImplementedError
