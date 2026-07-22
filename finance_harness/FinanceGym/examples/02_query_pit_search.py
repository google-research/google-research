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

"""Query the PIT search environment from Python.

Run the search server first (see ``examples/01_run_search_server.md``),
then point this script at it. The query embedding must be in the same
2,560-dim space the server's index was built with — the canonical
embedder is Qwen3-Embedding-4B served by vLLM on port 8888.

Example::

    SEARCH_URL=http://localhost:8889 \\
    EMBED_URL=http://localhost:8888/v1/embeddings \\
        python examples/02_query_pit_search.py
"""

from __future__ import annotations

import os

from financegym.env.client import EnvClient
import requests

SEARCH_URL = os.environ.get("SEARCH_URL", "http://localhost:8889")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8888/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")


def embed(query):
  r = requests.post(
      EMBED_URL,
      json={"model": EMBED_MODEL, "input": [query]},
      timeout=30,
  )
  r.raise_for_status()
  return r.json()["data"][0]["embedding"]


def main():
  client = EnvClient(SEARCH_URL)
  print("stats:", client.stats())

  query = "How did Federal Reserve policy shift in March 2025?"
  vec = embed(query)

  # PIT cutoff: agents can only see documents on or before this date.
  hits = client.search(vec, k=5, max_date="2025-06-30")
  for h in hits:
    print(f"  {h['score']:.3f}  {h['pub_date']}  {h['domain']}  {h['url']}")
    print(f"    preview: {h['text_preview']}")

  if hits:
    full = client.fetch(hits[0]["doc_id"])
    print(f"\nfull text of {full['doc_id']} (first 500 chars):")
    print(full["text"][:500])


if __name__ == "__main__":
  main()
