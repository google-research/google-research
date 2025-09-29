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

from concurrent import futures
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from cisc.src.runners import batcher as batcher_lib
from cisc.src.runners import fake_runner
from cisc.src.runners import runner as runner_lib


_INFINIT_TIMEOUT_SECS = 1000000


class WaitForRequestsTest(absltest.TestCase):

  def test_wait_for_requests_no_requests(self):
    self.assertTrue(batcher_lib.wait_for_requests([], timeout=10))

  def test_wait_for_requests_all_finish(self):
    req1 = batcher_lib.Request("")
    req2 = batcher_lib.Request("")
    req1.event.wait = mock.Mock(return_value=True)
    req2.event.wait = mock.Mock(return_value=True)
    mock_time = mock.Mock()
    mock_time.time.side_effect = [0, 0, 0]
    self.assertTrue(
        batcher_lib.wait_for_requests(
            [req1, req2], timeout=10, time_module=mock_time
        )
    )
    self.assertEqual(mock_time.time.call_count, 3)
    req1.event.wait.assert_called_once_with(10)
    req2.event.wait.assert_called_once_with(10)

  def test_wait_for_requests_one_times_out_on_wait(self):
    req1 = batcher_lib.Request("")
    req2 = batcher_lib.Request("")
    req1.event.wait = mock.Mock(return_value=True)
    req2.event.wait = mock.Mock(return_value=False)
    mock_time = mock.Mock()
    mock_time.time.side_effect = [0, 0, 0]
    self.assertFalse(
        batcher_lib.wait_for_requests(
            [req1, req2], timeout=10, time_module=mock_time
        )
    )
    self.assertEqual(mock_time.time.call_count, 3)
    req1.event.wait.assert_called_once_with(10)
    req2.event.wait.assert_called_once_with(10)

  def test_wait_for_requests_time_left_expires(self):
    mock_time = mock.Mock()
    # start=0, before 1st req elapsed=6, before 2nd req elapsed=12.
    elapsed_first = 6
    elapsed_second = 12
    mock_time.time.side_effect = [0, elapsed_first, elapsed_second]
    req1 = batcher_lib.Request("")
    req2 = batcher_lib.Request("")
    req1.event.wait = mock.Mock(return_value=True)
    req2.event.wait = mock.Mock()
    timeout = 10
    self.assertGreater(elapsed_second, timeout)
    self.assertFalse(
        batcher_lib.wait_for_requests(
            [req1, req2], timeout=timeout, time_module=mock_time
        )
    )
    self.assertEqual(mock_time.time.call_count, 3)
    req1.event.wait.assert_called_once_with(timeout - elapsed_first)
    req2.event.wait.assert_not_called()


class BatcherTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_wait_time",
          batch_size=2,
          max_wait_time_secs=0,
      ),
      dict(
          testcase_name="non_zero_max_wait_time",
          batch_size=2,
          max_wait_time_secs=1,
      ),
  )
  def test_bulk_generate_single_call(self, batch_size, max_wait_time_secs):
    inner_runner = fake_runner.FakeRunner()
    prompts = ["prompt1", "prompt2"]

    batch_runner = batcher_lib.BatchRunner(
        inner_runner,
        batch_size=batch_size,
        max_wait_time_secs=max_wait_time_secs,
    )
    results = batch_runner.generate(
        prompts, max_new_tokens=10, temperature=0.1, enable_formatting=False
    )
    self.assertEqual(
        results,
        [
            runner_lib.GenerationOutput(
                prompt="prompt1",
                response="prompt1 response",
                exception="",
            ),
            runner_lib.GenerationOutput(
                prompt="prompt2",
                response="prompt2 response",
                exception="",
            ),
        ],
    )

  def test_bulk_generate_large_batch_size(self):
    inner_runner = fake_runner.FakeRunner()
    prompts = ["prompt1", "prompt2", "prompt3"]

    batch_runner = batcher_lib.BatchRunner(
        inner_runner,
        batch_size=2,  # batch size is smaller than the number of prompts.
        max_wait_time_secs=1,
    )
    results = batch_runner.generate(
        prompts, max_new_tokens=10, temperature=0.1, enable_formatting=False
    )
    self.assertEqual(
        results,
        [
            runner_lib.GenerationOutput(
                prompt="prompt1",
                response="prompt1 response",
                exception="",
            ),
            runner_lib.GenerationOutput(
                prompt="prompt2",
                response="prompt2 response",
                exception="",
            ),
            runner_lib.GenerationOutput(
                prompt="prompt3",
                response="prompt3 response",
                exception="",
            ),
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_wait_time",
          batch_size=2,
          max_wait_time_secs=0,
      ),
      dict(
          testcase_name="does_not_reach_max_wait_time",
          batch_size=3,
          max_wait_time_secs=_INFINIT_TIMEOUT_SECS,
      ),
      dict(
          testcase_name="does_not_reach_batch_size_so_waits_a_bit",
          batch_size=_INFINIT_TIMEOUT_SECS,
          max_wait_time_secs=1,
      ),
  )
  def test_bulk_generate_two_calls(self, batch_size, max_wait_time_secs):
    inner_runner = fake_runner.FakeRunner()
    prompts = ["prompt1", "prompt2", "prompt3"]

    batch_runner = batcher_lib.BatchRunner(
        inner_runner,
        batch_size=batch_size,
        max_wait_time_secs=max_wait_time_secs,
        timeout_secs=_INFINIT_TIMEOUT_SECS,
    )
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
      results = executor.map(
          lambda prompt: batch_runner.generate(
              [prompt],
              max_new_tokens=10,
              temperature=0.1,
              enable_formatting=False,
          ),
          prompts,
      )
    self.assertCountEqual(
        list(itertools.chain(*results)),
        [
            runner_lib.GenerationOutput(
                prompt="prompt1",
                response="prompt1 response",
                exception="",
            ),
            runner_lib.GenerationOutput(
                prompt="prompt2",
                response="prompt2 response",
                exception="",
            ),
            runner_lib.GenerationOutput(
                prompt="prompt3",
                response="prompt3 response",
                exception="",
            ),
        ],
    )

  def test_bulk_generate_runner_fails(self):
    # Make sure that when an expection is raised by the inner runner, all the
    # promises are failed. This is important to make sure that the batcher does
    # not hang.
    inner_runner = fake_runner.FakeRunner(fail=True)
    prompts = ["prompt1", "prompt2"]

    batch_runner = batcher_lib.BatchRunner(
        inner_runner,
        batch_size=2,
        max_wait_time_secs=1,
        timeout_secs=_INFINIT_TIMEOUT_SECS,
    )

    with futures.ThreadPoolExecutor(max_workers=3) as executor:
      expection_futures = [
          executor.submit(
              batch_runner.generate,
              [prompt],
              max_new_tokens=10,
              temperature=0.1,
              enable_formatting=False,
          )
          for prompt in prompts
      ]
      for future in expection_futures:
        with self.assertRaises(SystemError):
          if future.exception() is not None:
            raise future.exception()


if __name__ == "__main__":
  absltest.main()
