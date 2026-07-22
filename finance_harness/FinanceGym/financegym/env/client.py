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

"""Thin Python client for the FinanceGym search server.

Wraps the four endpoints (`/search`, `/fetch`, `/stats`, `/health`) in a
small ``EnvClient`` so callers can stay on the FinanceGym side of the
contract without hand-rolling HTTP.

Use it like:

>>> client = EnvClient("http://localhost:8889")
>>> hits = client.search(query_embedding=embedding, k=5, max_date="2025-06-30")
>>> doc = client.fetch(hits[0]["doc_id"])
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

DEFAULT_TIMEOUT = 60.0


@dataclass
class EnvClient:
  """Client for one running search server."""

  base_url: str
  timeout: float = DEFAULT_TIMEOUT

  def _post(self, path, body):
    r = requests.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
    r.raise_for_status()
    return r.json()

  def _get(self, path):
    r = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
    r.raise_for_status()
    return r.json()

  def search(
      self,
      query_embedding,
      k = 10,
      max_date = None,
  ):
    """Run a /search. Returns the ``results`` array."""
    body: dict = {"query_embedding": list(query_embedding), "k": k}
    if max_date is not None:
      body["max_date"] = max_date
    return self._post("/search", body)["results"]

  def fetch(self, doc_id):
    """Run a /fetch. Returns the full document dict."""
    return self._post("/fetch", {"doc_id": doc_id})

  def stats(self):
    return self._get("/stats")

  def health(self):
    return self._get("/health")
