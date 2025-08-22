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

"""Utilities for parallel operations."""

from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar


_T = TypeVar('_T')
_T_ARG = TypeVar('_T_ARG')  # pylint: disable=invalid-name


def ParallelMap(
    f,
    lst,
    max_threads = None,
):
  # TODO(duckworthd): Implement actual logic.
  del max_threads
  return [f(arg) for arg in lst]


def RunInParallel(
    function,
    list_of_kwargs_to_function,
    num_workers,
    report_progress = False,
):
  # TODO(duckworthd): Implement actual logic.
  del num_workers
  del report_progress
  for kwargs in list_of_kwargs_to_function:
    function(**kwargs)
