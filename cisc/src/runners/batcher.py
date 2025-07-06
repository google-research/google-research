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

"""Batches requests to a runner.

This class is useful when multiple threads are sending requests to a runner and
you want to batch the requests to save resources.
"""

import collections
from collections.abc import Sequence
import itertools
import logging
import threading
import time
import more_itertools
from cisc.src.runners import runner as runner_lib


class Request:

  def __init__(self, prompt):
    self.creation_time = time.time()
    self.prompt = prompt
    # The event to be set when the request is done.
    self.event: threading.Event = threading.Event()
    self.result: runner_lib.GenerationOutput | None = None
    self.exception: Exception | None = None


_LOCK = threading.Lock()
_QUEUE: list[Request] = []
_DEBUG_BATCH_SIZES = collections.Counter()
_TOTAL_COUNTER = 0


class BatchRunner(runner_lib.Runner):
  """Gets a runner and run it in batches."""

  def __init__(
      self,
      runners,
      batch_size = 8,
      max_wait_time_secs = 1,
      timeout_secs = 1000,
  ):
    """Initializes the batch runner.

    Args:
      runners: The runners to be called in batches. We use a few runners to
        reuced the load on a single runner, as it might crash with a lot of
        traffic.
      batch_size: The size of the batches.
      max_wait_time_secs: The maximum time to wait for a batch to be full before
        sending it to the runner.
      timeout_secs: The maximum time to wait for a single request to be
        fulfilled.
    """
    if max_wait_time_secs > timeout_secs:
      raise ValueError(
          "`max_wait_time_secs` is larger than `timeout_secs`. This is probably"
          " a bug."
      )
    if isinstance(runners, list):
      self._runners = runners
    else:
      # For consistency, we allow to take a single runner and wrap it in a list.
      self._runners = [runners]

    self._batch_size = batch_size
    self._max_wait_time_secs = max_wait_time_secs
    self._timeout_secs = timeout_secs

  def _get_runner(self):
    """Returns a runner using round robin load balancing.

    It is possible to use a few runners to reuced the load on a single runner.
    """
    global _TOTAL_COUNTER
    with _LOCK:
      _TOTAL_COUNTER += 1
      return self._runners[int(_TOTAL_COUNTER % len(self._runners))]

  def _maybe_advance_queue(
      self,
      max_new_tokens,
      temperature,
      enable_formatting = False,
  ):
    global _QUEUE
    # Checks if its time to execute the next batch of requests. If it is, moves
    # the first `batch_size` requests from the global `_QUEUE` to `to_execute`,
    # and then execute them outside of the lock.
    with _LOCK:
      if not _QUEUE:
        return
      elapsed = time.time() - _QUEUE[0].creation_time
      if len(_QUEUE) < self._batch_size and elapsed < self._max_wait_time_secs:
        return
      # Remove the first `self._batch_size` elements from the queue.
      to_execute = _QUEUE[: self._batch_size]
      _QUEUE = _QUEUE[self._batch_size :]
      _DEBUG_BATCH_SIZES[len(to_execute)] += 1

    try:
      results = self._get_runner().generate(
          [req.prompt for req in to_execute],
          max_new_tokens,
          temperature,
          enable_formatting,
      )
      for req, result in zip(to_execute, results):
        req.result = result
        req.event.set()
    except Exception as e:  # pylint: disable=broad-except
      for req in to_execute:
        req.exception = e
        req.event.set()

  def _generate(
      self,
      prompts,
      max_new_tokens,
      temperature,
      enable_formatting = False,
  ):
    """See generate function below for documentation."""
    assert len(prompts) <= self._batch_size

    # Add the requests to the global queue and maybe advance it.
    with _LOCK:
      my_requests = [Request(prompt) for prompt in prompts]
      _QUEUE.extend(my_requests)
    self._maybe_advance_queue(max_new_tokens, temperature, enable_formatting)

    wait_for_requests = lambda timeout: all(
        [req.event.wait(timeout) for req in my_requests]
    )
    if not wait_for_requests(self._max_wait_time_secs):
      # After `max_wait_time_secs`, if the requests were not finished yet (which
      # is quite likely), try to advance the queue again to make sure the
      # requests at least started to run.
      self._maybe_advance_queue(max_new_tokens, temperature, enable_formatting)
      if not wait_for_requests(self._timeout_secs):
        raise TimeoutError(f"Timed out after {self._timeout_secs} seconds.")

    results = []
    for req in my_requests:
      if req.exception:
        raise req.exception
      assert req.result is not None
      results.append(req.result)
    return results

  def generate(
      self,
      prompts,
      max_new_tokens,
      temperature,
      enable_formatting = False,
  ):
    """Generates a single response for each prompt."""
    # For now we arbitrarily choose the settings of one of the calls (e.g., the
    # temperature is arbitrarily).
    batches = []
    for batch in more_itertools.batched(prompts, self._batch_size):
      try:
        res = self._generate(
            list(batch), max_new_tokens, temperature, enable_formatting
        )
      except TimeoutError as e:
        logging.exception("TimeoutError: %s", e)
        res = [
            runner_lib.GenerationOutput(
                prompt=prompt,
                response="",
                exception="TimeoutError: " + str(e),
            )
            for prompt in batch
        ]
      batches.append(res)
    return list(itertools.chain.from_iterable(batches))

  def get_completion_likelihoods(
      self,
      prefixes,
      completions,
      enable_formatting,
  ):
    # Consider supporting batching - using the same mechanism as in `generate`.
    assert len(prefixes) == 1, "Batcher only supports single prefix for now."
    return self._get_runner().get_completion_likelihoods(
        prefixes, completions, enable_formatting
    )

  def get_normalized_probability_for_sequence(
      self,
      prefix,
      completion,
  ):
    # Consider supporting batching - using the same mechanism as in `generate`.
    return self._get_runner().get_normalized_probability_for_sequence(
        prefix, completion
    )
