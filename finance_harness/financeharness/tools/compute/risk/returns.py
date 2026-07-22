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

"""Shared return helpers for the risk tools — pure, no network."""

from __future__ import annotations


def pct_returns(prices):
  """Simple period-over-period returns from a price series (skips zero priors)."""
  out: list[float] = []
  for i in range(1, len(prices)):
    prev = prices[i - 1]
    if prev:
      out.append(prices[i] / prev - 1.0)
  return out


def align(*series):
  """Align return series to their common length by keeping the most recent N of each

  — so paired stats (correlation, beta) compare the same observations.
  """
  n = min((len(s) for s in series), default=0)
  return [s[len(s) - n :] for s in series]
