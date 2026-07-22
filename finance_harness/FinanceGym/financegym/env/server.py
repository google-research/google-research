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

"""Point-in-time search server (the FinanceEnv contract).

The server has a fixed REST surface:

* ``POST /search``  semantic search with an optional ``max_date`` PIT cutoff.
* ``POST /fetch``   retrieve full article text by ``doc_id``.
* ``GET  /stats``   small summary of the loaded index + DB.
* ``GET  /health``  liveness probe.

State (FAISS index, in-RAM metadata, SQLite text store) is owned by a
:class:`ServerState` instance. :func:`create_app` returns a FastAPI app
wired to that state — the factory pattern keeps tests hermetic, since
they can construct a synthetic state with a flat FAISS index and a
sqlite3 in-memory connection without touching disk.

Production callers do:

>>> state = ServerState.from_disk("output/search/")
>>> app = create_app(state)
>>> uvicorn.run(app, host="127.0.0.1", port=8889)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sqlite3
import time

import faiss
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import uvicorn

log = logging.getLogger(__name__)

INDEX_FILENAME = "faiss_index.bin"
DB_FILENAME = "corpus.db"
META_FILENAME = "metadata.jsonl"
PREVIEW_CHARS = 200


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
  query_embedding: list[float]
  k: int = 10
  max_date: str | None = None


class SearchHit(BaseModel):
  doc_id: str
  url: str
  domain: str
  pub_date: str
  score: float
  text_preview: str


class SearchResponse(BaseModel):
  results: list[SearchHit]
  total_candidates: int
  query_time_ms: float


class FetchRequest(BaseModel):
  doc_id: str


class FetchResponse(BaseModel):
  doc_id: str
  text: str
  url: str
  domain: str
  pub_date: str


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
  """Holds the FAISS index, in-memory metadata, and the SQLite store."""

  index: faiss.Index | None = None
  meta: list[dict] = field(default_factory=list)
  db: sqlite3.Connection | None = None

  @property
  def total_docs(self):
    return len(self.meta)

  # --- text store -------------------------------------------------------

  def fetch_text(self, doc_id):
    if self.db is None:
      return ""
    row = self.db.execute(
        "SELECT text FROM docs WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return row[0] if row else ""

  def fetch_text_preview(
      self, doc_id, max_len = PREVIEW_CHARS
  ):
    if self.db is None:
      return ""
    row = self.db.execute(
        "SELECT SUBSTR(text, 1, ?) FROM docs WHERE doc_id = ?",
        (max_len, doc_id),
    ).fetchone()
    return row[0] if row else ""

  def fetch_doc(self, doc_id):
    if self.db is None:
      return None
    row = self.db.execute(
        "SELECT doc_id, url, domain, pub_date, text FROM docs WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if not row:
      return None
    return {
        "doc_id": row[0],
        "url": row[1],
        "domain": row[2],
        "pub_date": row[3],
        "text": row[4],
    }

  # --- factory ----------------------------------------------------------

  @classmethod
  def from_disk(cls, data_dir):
    """Build a state from the canonical on-disk layout."""
    d = Path(data_dir)
    meta = _load_metadata(d / META_FILENAME)
    db_path = d / DB_FILENAME
    db: sqlite3.Connection | None = None
    if db_path.exists():
      db = sqlite3.connect(str(db_path), check_same_thread=False)
      db.execute("PRAGMA cache_size=-4000000")  # ~4GB cache
      db.execute("PRAGMA mmap_size=8589934592")  # 8GB mmap
    else:
      log.warning("SQLite DB missing at %s; /fetch will be empty", db_path)
    idx_path = d / INDEX_FILENAME
    index: faiss.Index | None = None
    if idx_path.exists():
      index = faiss.read_index(str(idx_path))
      try:
        ivf = faiss.extract_index_ivf(index)
        ivf.nprobe = 32
      except RuntimeError:
        pass
    else:
      log.warning("FAISS index missing at %s; /search will 503", idx_path)
    return cls(index=index, meta=meta, db=db)


def _load_metadata(path):
  out: list[dict] = []
  with open(path) as f:
    for line in f:
      m = json.loads(line)
      out.append({
          "doc_id": m["doc_id"],
          "url": m.get("url", ""),
          "domain": m.get("domain", ""),
          "pub_date": m.get("pub_date", ""),
      })
  return out


# ---------------------------------------------------------------------------
# Core search logic (pure modulo the state)
# ---------------------------------------------------------------------------


def search(
    state, query, k, max_date
):
  """Run a search against the state's index with optional PIT filtering."""
  if state.index is None:
    raise HTTPException(503, "Index not loaded")
  if state.total_docs == 0:
    return SearchResponse(results=[], total_candidates=0, query_time_ms=0.0)

  t0 = time.time()
  retrieve_k = k * 5 if max_date else k
  retrieve_k = min(retrieve_k, state.total_docs)

  q = query.reshape(1, -1).astype(np.float32)
  faiss.normalize_L2(q)
  scores, indices = state.index.search(q, retrieve_k)

  results: list[SearchHit] = []
  for idx, score in zip(indices[0], scores[0], strict=False):
    if idx < 0 or idx >= state.total_docs:
      continue
    meta = state.meta[idx]
    if max_date and meta.get("pub_date", "") > max_date:
      continue
    doc_id = meta["doc_id"]
    results.append(
        SearchHit(
            doc_id=doc_id,
            url=meta.get("url", ""),
            domain=meta.get("domain", ""),
            pub_date=meta.get("pub_date", ""),
            score=float(score),
            text_preview=state.fetch_text_preview(doc_id),
        )
    )
    if len(results) >= k:
      break

  return SearchResponse(
      results=results,
      total_candidates=int(indices.shape[1]),
      query_time_ms=round((time.time() - t0) * 1000, 2),
  )


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def create_app(state):
  """Wire FastAPI routes to a given :class:`ServerState`."""
  app = FastAPI(title="FinanceGym Search API")
  app.state.financegym = state

  @app.post("/search", response_model=SearchResponse)
  def _search(req):
    emb = np.asarray(req.query_embedding, dtype=np.float32)
    return search(state, emb, req.k, req.max_date)

  @app.post("/fetch", response_model=FetchResponse)
  def _fetch(req):
    doc = state.fetch_doc(req.doc_id)
    if doc is None:
      raise HTTPException(404, f"document {req.doc_id} not found")
    return FetchResponse(**doc)

  @app.get("/stats")
  def _stats():
    return {
        "total_docs": state.total_docs,
        "index_loaded": state.index is not None,
        "db_loaded": state.db is not None,
    }

  @app.get("/health")
  def _health():
    return {"status": "ok" if state.index is not None else "loading"}

  return app


# ---------------------------------------------------------------------------
# Module-level app for `uvicorn financegym.env.server:app`
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(description="FinanceGym search server.")
  parser.add_argument("--data-dir", default="output/search/")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8889)
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )
  state = ServerState.from_disk(args.data_dir)
  uvicorn.run(create_app(state), host=args.host, port=args.port)


if __name__ == "__main__":
  main()
