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

"""Allows querying LLMs with a cache.

The cache works by exact match of the prompt and temperature, while other
optional arguments are ignored. For efficiency reasons, the prompt itself is not
stored in the cache, but an MD5 hash is stored instead. All previous samples for
the prompt/temperature are stored, and are re-used even if the number of samples
is different, where new samples are drawn when needed in addition to previous
samples.

Cached responses are always lists of strings, allowing for multiple samples for
the same prompt.

To fully reset the cache, simply remove the associated .cache file from your
filesystem (or move or rename it temporarily, etc.). Although named *.cache, it
is actually a jsonl file.
"""

import collections
import hashlib
import json
import os
import re

from absl import logging

_CACHE_FILE_PATH = None
_CACHE = None


def _check_model_name(model_name):
  return bool(re.fullmatch(r'[a-zA-Z0-9_.-]+', model_name))


def init_cache(cache_dir, model_name):
  """Initializes the LLM cache."""
  global _CACHE_FILE_PATH, _CACHE

  if not _check_model_name(model_name):
    raise ValueError(f'Model name is not valid: {model_name}')

  _CACHE_FILE_PATH = os.path.join(cache_dir, f'{model_name}.cache')
  _CACHE = collections.defaultdict(list)
  print(f'Initializing LLM cache, stored at: {_CACHE_FILE_PATH}')

  os.makedirs(cache_dir, exist_ok=True)
  if os.path.exists(_CACHE_FILE_PATH):
    with open(_CACHE_FILE_PATH, 'r') as f:
      for line in f:
        entry = json.loads(line)
        if not isinstance(entry, dict) or len(entry) != 1:
          raise ValueError(f'Invalid line in cache file: {line}')
        cache_key = list(entry.keys())[0]
        samples = entry[cache_key]
        if (not isinstance(cache_key, str)
            or not isinstance(samples, list)
            or not all(isinstance(sample, str) for sample in samples)):
          raise ValueError(f'Invalid cache entry: {entry}')
        _CACHE[cache_key].extend(samples)
  else:
    with open(_CACHE_FILE_PATH, 'w'):
      pass


def _get_cache_key(prompt, temperature):
  prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
  return repr({'prompt_hash': prompt_hash, 'temperature': temperature})


def query_llm(query_fn,
              prompt,
              n,
              temperature,
              **kwargs):
  """Query the LLM, caching the prompt hash and response."""
  if _CACHE_FILE_PATH is None or _CACHE is None:
    raise ValueError('Call init_cache() first.')
  if not isinstance(n, int) or n <= 0:
    raise ValueError(f'Expected a positive integer for n, but got: {n}')

  cache_key = _get_cache_key(prompt, temperature)
  if cache_key in _CACHE:
    existing_samples = _CACHE[cache_key]
    if len(existing_samples) >= n:
      logging.info('LLM Cache: full hit')
      return existing_samples[:n]
    else:
      logging.info('LLM Cache: partial hit')
  else:
    logging.info('LLM Cache: miss')
    existing_samples = []

  num_new_samples = n - len(existing_samples)
  assert num_new_samples > 0

  new_samples = query_fn(prompt, n=num_new_samples, temperature=temperature,
                         **kwargs)
  if isinstance(new_samples, str):
    new_samples = [new_samples]

  if (not isinstance(new_samples, list)
      or not all(isinstance(sample, str) for sample in new_samples)):
    raise ValueError(
        f'Expected new_samples to be a list of strings, but got: {new_samples}')
  if len(new_samples) != num_new_samples:
    raise ValueError(f'Expected {num_new_samples} new samples but got '
                     f'{len(new_samples)}')

  # Write only the new samples.
  with open(_CACHE_FILE_PATH, 'a') as f:
    f.write(json.dumps({cache_key: new_samples}) + '\n')

  _CACHE[cache_key].extend(new_samples)
  assert len(_CACHE[cache_key]) == n
  return _CACHE[cache_key]
