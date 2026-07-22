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

"""Shared quote helper for the market-data tools — last close + day change% for an

index/yield symbol, via the equity tools' yfinance retry wrapper.
"""

from __future__ import annotations

import asyncio

from financeharness.tools.data.equity.common import is_num, yf_call


async def last_quote(symbol):
  """(last close, day change %) for ``symbol`` from a short history window, or None

  if unavailable. Day change is vs the prior close (0.0 if only one bar).
  """
  import yfinance as yf

  def _sync():
    h = yf.Ticker(symbol).history(period="5d")
    if h is None or h.empty or "Close" not in h.columns:
      return None
    closes = [float(c) for c in h["Close"].tolist() if is_num(c)]
    return closes or None

  closes = await yf_call(lambda: asyncio.to_thread(_sync))
  if not closes:
    return None
  last = closes[-1]
  prev = closes[-2] if len(closes) > 1 else last
  change = (last - prev) / prev * 100 if prev else 0.0
  return round(last, 2), round(change, 2)
