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

"""Build a SQLite store of (metadata, text) so the search server can answer

``/fetch`` in O(1) without holding the texts in RAM.

The store is a single table ``docs`` with one row per article, indexed on
``doc_id``, ``domain``, and ``pub_date``. It is built once per corpus
ingest from the JSONL output of :mod:`financegym.corpus.extract_embed`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sqlite3
import time

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE docs (
    idx        INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    url        TEXT,
    domain     TEXT,
    pub_date   TEXT,
    crawl_date TEXT,
    text_len   INTEGER,
    text       TEXT
)
"""

INDEXES = [
    "CREATE UNIQUE INDEX idx_doc_id   ON docs(doc_id)",
    "CREATE        INDEX idx_domain   ON docs(domain)",
    "CREATE        INDEX idx_pub_date ON docs(pub_date)",
]


def build_db(
    input_dir,
    *,
    batch_size = 10_000,
    overwrite = True,
):
  """Build ``corpus.db`` from ``metadata.jsonl`` + ``texts.jsonl``.

  The two files are read in lock-step; their lines must correspond
  1:1 (this is the invariant enforced by ``extract_embed.run``).
  Returns the path to the built database.
  """
  data = Path(input_dir)
  db_path = data / "corpus.db"
  meta_path = data / "metadata.jsonl"
  texts_path = data / "texts.jsonl"
  if not meta_path.exists():
    raise FileNotFoundError(meta_path)
  if not texts_path.exists():
    raise FileNotFoundError(texts_path)

  if db_path.exists() and overwrite:
    db_path.unlink()
  elif db_path.exists():
    raise FileExistsError(db_path)

  conn = sqlite3.connect(str(db_path))
  try:
    # Bulk-insert pragmas
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(SCHEMA)

    t0 = time.time()
    batch: list[tuple] = []
    n = 0
    with open(meta_path) as mf, open(texts_path) as tf:
      for idx, (m_line, t_line) in enumerate(zip(mf, tf, strict=False)):
        try:
          meta = json.loads(m_line)
          text = json.loads(t_line)
        except json.JSONDecodeError:
          continue
        batch.append((
            idx,
            meta.get("doc_id", f"doc_{idx}"),
            meta.get("url", ""),
            meta.get("domain", ""),
            meta.get("pub_date", ""),
            meta.get("crawl_date", ""),
            meta.get("text_len", 0),
            text.get("text", ""),
        ))
        if len(batch) >= batch_size:
          conn.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)", batch)
          conn.commit()
          n += len(batch)
          batch.clear()
    if batch:
      conn.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)", batch)
      conn.commit()
      n += len(batch)

    log.info("inserted %d rows in %.1fs", n, time.time() - t0)

    t0 = time.time()
    for stmt in INDEXES:
      conn.execute(stmt)
    conn.commit()
    log.info("created %d indexes in %.1fs", len(INDEXES), time.time() - t0)
  finally:
    conn.close()

  return db_path


def main():
  parser = argparse.ArgumentParser(
      description="Build SQLite store for the search env."
  )
  parser.add_argument("--input", default="output/search/")
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )
  build_db(args.input)


if __name__ == "__main__":
  main()
