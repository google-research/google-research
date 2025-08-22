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

"""Tests for cached_llm_access.py."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from latent_programmer.spec_decomposition import cached_llm_access


# Allows us to keep track of unique samples.
_ID = 0


def _id():
  global _ID
  _ID += 1
  return _ID


def _reset_id():
  global _ID
  _ID = 0


def dummy_query_fn(prompt,
                   n,
                   temperature,
                   other_kwarg = -1):
  return [(f'sample #{_id()}. prompt={prompt}, temperature={temperature}, '
           f'other_kwarg={other_kwarg}')
          for _ in range(n)]


def reload_cache(cache_dir):
  cached_llm_access._CACHE = None
  cached_llm_access.init_cache(cache_dir, 'model_name')


class CachedLlmAccessTest(parameterized.TestCase):

  @parameterized.parameters(
      ('gemini-m-llmit', True),
      ('gpt-4', True),
      ('gpt-3.5-turbo-16k', True),
      ('with spaces', False),
      ('special_characters!', False),
  )
  def test_check_model_name(self, model_name, expected):
    self.assertEqual(cached_llm_access._check_model_name(model_name), expected)

  @parameterized.parameters(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  @mock.patch.object(sys.modules[__name__], 'dummy_query_fn',
                     wraps=dummy_query_fn)
  def test_cache(self, reload_cache_location, mock_query_fn):
    # This test is parameterized by `reload_cache_location` representing where
    # to reload the cache, if at all. Reloading the cache should not affect any
    # behavior.
    cache_dir = self.create_tempdir().full_path
    reload_cache(cache_dir)
    _reset_id()

    # First query.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=2, temperature=0.8, other_kwarg=7)
    mock_query_fn.assert_called_once_with('A', n=2, temperature=0.8,
                                          other_kwarg=7)
    self.assertEqual(samples,
                     ['sample #1. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #2. prompt=A, temperature=0.8, other_kwarg=7'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 1:
      reload_cache(cache_dir)

    # Query with same prompt but different temperature.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=1, temperature=0.2)
    mock_query_fn.assert_called_once_with('A', n=1, temperature=0.2)
    self.assertEqual(samples,
                     ['sample #3. prompt=A, temperature=0.2, other_kwarg=-1'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 2:
      reload_cache(cache_dir)

    # Query with different prompt.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='B', n=1, temperature=0.8)
    mock_query_fn.assert_called_once_with('B', n=1, temperature=0.8)
    self.assertEqual(samples,
                     ['sample #4. prompt=B, temperature=0.8, other_kwarg=-1'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 3:
      reload_cache(cache_dir)

    # Query with same prompt and same temperature (full cache hit, using
    # only some samples). `other_kwarg` is different and ignored.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=1, temperature=0.8, other_kwarg=3)
    mock_query_fn.assert_not_called()
    self.assertEqual(samples,
                     ['sample #1. prompt=A, temperature=0.8, other_kwarg=7'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 4:
      reload_cache(cache_dir)

    # Query with same prompt and same temperature (full cache hit, using all
    # samples).
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=2, temperature=0.8, other_kwarg=3)
    mock_query_fn.assert_not_called()
    self.assertEqual(samples,
                     ['sample #1. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #2. prompt=A, temperature=0.8, other_kwarg=7'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 5:
      reload_cache(cache_dir)

    # Query with same prompt and same temperature (partial cache hit).
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=3, temperature=0.8, other_kwarg=3)
    mock_query_fn.assert_called_once_with('A', n=1, temperature=0.8,
                                          other_kwarg=3)
    self.assertEqual(samples,
                     ['sample #1. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #2. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #5. prompt=A, temperature=0.8, other_kwarg=3'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 6:
      reload_cache(cache_dir)

    # Query with same prompt and same temperature, checking we can use samples
    # from two previous queries.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=3, temperature=0.8, other_kwarg=3)
    mock_query_fn.assert_not_called()
    self.assertEqual(samples,
                     ['sample #1. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #2. prompt=A, temperature=0.8, other_kwarg=7',
                      'sample #5. prompt=A, temperature=0.8, other_kwarg=3'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 7:
      reload_cache(cache_dir)

    # Another cache miss.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='C', n=1, temperature=0.7)
    mock_query_fn.assert_called_once_with('C', n=1, temperature=0.7)
    self.assertEqual(samples,
                     ['sample #6. prompt=C, temperature=0.7, other_kwarg=-1'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 8:
      reload_cache(cache_dir)

    # Another partial cache hit.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='A', n=3, temperature=0.2, other_kwarg=5)
    mock_query_fn.assert_called_once_with('A', n=2, temperature=0.2,
                                          other_kwarg=5)
    self.assertEqual(samples,
                     ['sample #3. prompt=A, temperature=0.2, other_kwarg=-1',
                      'sample #7. prompt=A, temperature=0.2, other_kwarg=5',
                      'sample #8. prompt=A, temperature=0.2, other_kwarg=5'])
    mock_query_fn.reset_mock()
    if reload_cache_location == 9:
      reload_cache(cache_dir)

    # Another full cache hit.
    samples = cached_llm_access.query_llm(
        mock_query_fn, prompt='B', n=1, temperature=0.8)
    mock_query_fn.assert_not_called()
    self.assertEqual(samples,
                     ['sample #4. prompt=B, temperature=0.8, other_kwarg=-1'])


if __name__ == '__main__':
  absltest.main()
