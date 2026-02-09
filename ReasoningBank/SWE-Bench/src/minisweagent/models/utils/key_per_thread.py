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

"""Utility for anthropic where we need different keys for different parallel
agents to not mess up prompt caching.
"""

import threading
from typing import Any

_THREADS_THAT_USED_API_KEYS: list[Any] = []


def get_key_per_thread(api_keys):
    """Choose key based on thread name. Returns None if no keys are available."""
    thread_name = threading.current_thread().name
    if thread_name not in _THREADS_THAT_USED_API_KEYS:
        _THREADS_THAT_USED_API_KEYS.append(thread_name)
    thread_idx = _THREADS_THAT_USED_API_KEYS.index(thread_name)
    key_idx = thread_idx % len(api_keys)
    return api_keys[key_idx] or None
