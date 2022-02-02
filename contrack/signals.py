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

"""Collect signals for context tracking."""

from typing import List, Text

from bazel_tools.tools.python.runfiles import runfiles

FEMALE_NAMES = None
MALE_NAMES = None


def collect_signals(words):
  """Collect signals for tokens."""
  global FEMALE_NAMES
  global MALE_NAMES

  r = runfiles.Create()
  if not FEMALE_NAMES:
    with open(r.Rlocation('contrack/data/female_names.txt'), 'r') as f:
      FEMALE_NAMES = f.read().splitlines()
  if not MALE_NAMES:
    with open(r.Rlocation('contrack/data/male_names.txt'), 'r') as f:
      MALE_NAMES = f.read().splitlines()

  result = []
  for word in words:
    signals = []
    if word in MALE_NAMES or word in FEMALE_NAMES:
      signals.append('first_name')
    result.append(signals)

  return result
