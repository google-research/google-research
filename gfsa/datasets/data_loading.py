# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Utilities for loading data."""

import enum
import itertools
import queue
import random
import threading
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, TypeVar

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np

from gfsa import jax_util

K = TypeVar("K")
T = TypeVar("T")


def randomly_interleave(sources,
                        simultaneous):
  """Randomly pull elements from an iterable of iterables.

  In particular, takes the first `simultaneous` values from `sources`, and then
  randomly chooses one of them to generate the next example. When any iterable
  runs out, it is replaced with a new one from `sources`.

  Args:
    sources: Iterable of iterables to pull from.
    simultaneous: How many iterables to have open at one time.

  Yields:
    Elements from `sources` in a random order.
  """
  sources = iter(sources)
  active = [iter(it) for it in itertools.islice(sources, simultaneous)]
  while active:
    # Pick a random active source.
    index = random.randrange(len(active))
    try:
      # Pull from it.
      yield next(active[index])
    except StopIteration:
      # This source was empty!
      try:
        # Replace it with a new one.
        active[index] = iter(next(sources))
      except StopIteration:
        # No more sources!
        # Delete from the end to avoid unnecessary list copying.
        active[index] = active[-1]
        del active[-1]


def shuffle_with_buffer(source, buffer_size):
  """Randomly yield elements from a buffer of elements drawn from the source.

  Args:
    source: Iterable to draw from.
    buffer_size: Number of recent elements to draw samples from.

  Yields:
    Randomly reordered elements from `source`.
  """
  source = iter(source)
  buffer = list(itertools.islice(source, buffer_size))
  while buffer:
    # Pick a random element.
    index = random.randrange(len(buffer))
    yield buffer[index]
    try:
      # Replace it with a new one.
      buffer[index] = next(source)
    except StopIteration:
      # No more values!
      # Delete from the end to avoid unnecessary list copying.
      buffer[index] = buffer[-1]
      del buffer[-1]


def repeat(make_source):
  """Yield elements from a source, re-initializing it when empty."""
  return itertools.chain.from_iterable(make_source() for _ in itertools.count())


class BatchRemainderBehavior(enum.Enum):
  """Specifies how to handle leftover elements at the end of a dataset."""
  ERROR = "ERROR"  # Raise an error if batch size doesn't evenly divide dataset.
  DROP = "DROP"  # Discard the leftover elements.
  PAD_ZERO = "PAD_ZERO"  # Pad out to the batch size with zeros.


def batch(
    source,
    batch_dim_sizes,
    remainder_behavior = BatchRemainderBehavior.ERROR
):
  """Batch together elements from source.

  Args:
    source: Iterable of elements.
    batch_dim_sizes: Prefix of the shape for the output elements. For instance,
      if batch_dim_sizes=(8, 4), this method will batch together 8 * 4 = 32
      elements at a time, and the shapes of all ndarrays in the output will have
      a prefix of (8, 4).
    remainder_behavior: Determines batching behavior when exhausting `source`.

  Yields:
    Batches of elements.

  Raises:
    ValueError: If remainder_behavior=ERROR and the batch size doesn't evenly
    divide the source length.
  """
  source = iter(source)
  batch_size = 1
  for s in batch_dim_sizes:
    batch_size *= s

  while True:
    to_batch = list(itertools.islice(source, batch_size))
    if not to_batch:
      # No more elements to process.
      return
    elif len(to_batch) != batch_size:
      if remainder_behavior == BatchRemainderBehavior.ERROR:
        raise ValueError(
            f"Size of source (remainder {len(to_batch)}) is not divisible by "
            f"batch size ({batch_size})")
      elif remainder_behavior == BatchRemainderBehavior.DROP:
        logging.warn(
            "Dropping dataset remainder of %d elements (for batch size %d)",
            len(to_batch), batch_size)
        return
      elif remainder_behavior == BatchRemainderBehavior.PAD_ZERO:
        # Batch what we have and pad the results.
        pass
      else:
        raise ValueError(f"Unknown remainder behavior {remainder_behavior}")

    def _batch_and_pad_elts(*args):
      stacked = np.stack(args)
      stacked = jax_util.pad_to(stacked, batch_size)
      return stacked.reshape(batch_dim_sizes + stacked.shape[1:])

    yield jax.tree_multimap(_batch_and_pad_elts, *to_batch)


def batch_bucketed(
    source,
    batch_dim_sizes,
    remainder_behavior = BatchRemainderBehavior.ERROR
):
  """Batch together elements with different size buckets.

  Maintains multiple partially-full batches, one for each size key, and emits
  each batch when it fills up.

  Args:
    source: Iterable of tuples (bucket_key, example)
    batch_dim_sizes: For each bucket key, the prefix of the shape for the output
      elements. For instance, if batch_dim_sizes[key]=(8, 4), this method will
      batch together 8 * 4 = 32 elements with `bucket_key=key` before emitting
      them, and the shapes of the ndarrays in the output will have a prefix of
      (8, 4).
    remainder_behavior: Determines batching behavior when exhausting `source`.

  Yields:
    Tuples (bucket_key, batch)

  Raises:
    ValueError: If remainder_behavior=ERROR and the batch size doesn't evenly
    divide the source length.

    KeyError: If an example from `source` doesn't have a recognized bucket key.
  """
  source = iter(source)
  batch_sizes = {}
  for key, dim_sizes in batch_dim_sizes.items():
    batch_size = 1
    for s in dim_sizes:
      batch_size *= s
    batch_sizes[key] = batch_size

  partial_batches = {k: [] for k in batch_dim_sizes}

  def _emit(key):
    """Batches together and returns a batch for a given key."""

    def _batch_and_pad_elts(*args):
      stacked = np.stack(args)
      stacked = jax_util.pad_to(stacked, batch_sizes[key])
      return stacked.reshape(batch_dim_sizes[key] + stacked.shape[1:])

    result = jax.tree_multimap(_batch_and_pad_elts, *partial_batches[key])
    partial_batches[key].clear()
    return (key, result)

  # Iterate over source, putting examples into the appropriate batch size
  # bucket.
  for key, example in source:
    partial_batches[key].append(example)
    if len(partial_batches[key]) == batch_sizes[key]:
      yield _emit(key)

  # Handle any leftover elements.
  for key in partial_batches:
    if partial_batches[key]:
      if remainder_behavior == BatchRemainderBehavior.ERROR:
        raise ValueError(
            f"Number of leftover examples for bucket {key} (remainder "
            f"{len(partial_batches[key])}) is not divisible by batch size "
            f"({batch_size})")
      elif remainder_behavior == BatchRemainderBehavior.DROP:
        logging.warn(
            "Dropping dataset remainder of %d elements (for bucket key %s with "
            "batch size %d)", len(partial_batches[key]), key, batch_size)
      elif remainder_behavior == BatchRemainderBehavior.PAD_ZERO:
        yield _emit(key)
      else:
        raise ValueError(f"Unknown remainder behavior {remainder_behavior}")


def batch_and_pad_to_prototype(
    source,
    batch_dim_sizes,
    prototype,
    remainder_behavior = BatchRemainderBehavior.ERROR,
    drop_too_large = False,
):
  """Batch together elements from source, padding them to match a prototype.

  Args:
    source: Iterable of elements.
    batch_dim_sizes: Prefix of the shape for the output elements. For instance,
      if batch_dim_sizes=(8, 4), this method will batch together 8 * 4 = 32
      elements at a time, and the shapes of all ndarrays in the output will have
      a prefix of (8, 4).
    prototype: Object representing a single example. Any pytree whose leaves
      have .shape and .dtype attributes is allowed. Should not include batch
      axes.
    remainder_behavior: Determines batching behavior when exhausting `source`.
    drop_too_large: Whether to drop examples that are too large.

  Yields:
    Batches of elements.

  Raises:
    ValueError: If remainder_behavior=ERROR and the batch size doesn't evenly
    divide the source length.
  """
  if drop_too_large:
    prototype_leaves = jax.tree_leaves(prototype)

    def filtered_source_generator():
      drop_ct = 0
      total_ct = 0
      for element in source:
        element_leaves = jax.tree_leaves(element)
        fits = True
        for el, ep in zip(element_leaves, prototype_leaves):
          el = np.asarray(el)
          if not all(dl <= dp for dl, dp in zip(el.shape, ep.shape)):
            fits = False
            break
        total_ct += 1
        if fits:
          yield element
        else:
          drop_ct += 1
          if np.log10(drop_ct) % 1.0 == 0.0:
            logging.warning("Dropped %d out of %d (ratio %f)", drop_ct,
                            total_ct, drop_ct / total_ct)

    source_iter = filtered_source_generator()
  else:
    source_iter = iter(source)

  source = iter(source)
  batch_size = 1
  for s in batch_dim_sizes:
    batch_size *= s

  def _batch_into_prototype(prototype, *to_batch):
    result = np.zeros(batch_dim_sizes + prototype.shape, prototype.dtype)
    for i, element in enumerate(to_batch):
      element = np.asarray(element)
      dest_slice = (
          np.unravel_index(i, batch_dim_sizes) +
          tuple(jnp.s_[:n] for n in element.shape))
      result[dest_slice] = element
    return result

  while True:
    to_batch = list(itertools.islice(source_iter, batch_size))
    if not to_batch:
      # No more elements to process.
      return
    elif len(to_batch) != batch_size:
      if remainder_behavior == BatchRemainderBehavior.ERROR:
        raise ValueError(
            f"Size of source (remainder {len(to_batch)}) is not divisible by "
            f"batch size ({batch_size})")
      elif remainder_behavior == BatchRemainderBehavior.DROP:
        logging.warn(
            "Dropping dataset remainder of %d elements (for batch size %d)",
            len(to_batch), batch_size)
        return
      elif remainder_behavior == BatchRemainderBehavior.PAD_ZERO:
        # Batch what we have and pad the results.
        pass
      else:
        raise ValueError(f"Unknown remainder behavior {remainder_behavior}")

    yield jax.tree_multimap(_batch_into_prototype, prototype, *to_batch)


class ThreadedPrefetcher:
  """Context manager to prefetch elements in a background thread.

  Note that this uses Python threads, which means:
  - Anything that happens when pulling from source should be thread-safe. In
    particular, make sure that no other thread is reading from source. (SSTables
    should be thread safe as long as each iterator is only accessed from one
    thread at a time.)
  - The global interpreter lock applies, so expect speedups only if there is
    a lot of time waiting in C code (likely the case when reading from disk).

  When used as a context manager, it starts up a thread that remains valid
  inside that context manager. Exiting the context manager signals the thread
  to stop. The thread will also stop on its own if the iterable becomes empty.

  Usage example:

    with ThreadedPrefetcher(my_iterable, 100) as prefetched:
      # starts prefetching up to 100 elements here

      # blocks if the buffer is empty, and re-raises errors caught by the worker
      for value in prefetched:

        # ... do something with value ...
        if some_condition(value):
          # if we break out, the worker thread stops
          break

        # if the worker runs out of values, the main thread exits the loop
        # normally

    # by the time we get here, thread has stopped running

  """
  OK = 0
  FINISHED = 1
  ERROR = 2

  def __init__(self, source, max_prefetch):
    """Initialize the prefetcher.

    Args:
      source: Thread-safe iterable to draw results from.
      max_prefetch: Number of examples to prefetch.
    """
    self.source = iter(source)
    self.max_prefetch = max_prefetch
    # Whether or not we have finished the sequence
    self.finished = False
    # Queue of booleans asking the worker thread to produce more values.
    # Uses LIFO so that we detect the stop command as soon as it is sent.
    self.want_more = queue.LifoQueue()
    for _ in range(max_prefetch):
      self.want_more.put(True)
    # Queue of results generated by the worker thread
    self.results = queue.Queue()
    self.worker = None

  def _worker_thread(self):
    """Worker thread; runs in the background."""
    while self.want_more.get():
      try:
        self.results.put((ThreadedPrefetcher.OK, next(self.source)))
      except StopIteration:
        self.results.put((ThreadedPrefetcher.FINISHED, None))
        return
      except BaseException as e:  # pylint: disable=broad-except
        self.results.put((ThreadedPrefetcher.ERROR, e))
        return

  def __enter__(self):
    """Start the worker thread, and produce an iterator."""
    if self.worker is not None:
      raise RuntimeError("Prefetching worker is already running!")
    self.worker = threading.Thread(target=self._worker_thread)
    self.worker.start()
    return self._iterate()

  def _iterate(self):
    """Generator that iterates over results."""
    while not self.finished:
      if self.worker is None:
        raise RuntimeError(
            "Iteration is only allowed inside the prefetching context manager!")
      status, result = self.results.get()
      if status == ThreadedPrefetcher.OK:
        # Tell the worker thread to fetch another value
        self.want_more.put(True)
        yield result
      elif status == ThreadedPrefetcher.FINISHED:
        self.finished = True
      elif status == ThreadedPrefetcher.ERROR:
        self.finished = True
        raise result

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Stop the worker thread."""
    # Signal the worker thread to stop
    self.want_more.put(False)
    self.worker.join()
    self.worker = None
