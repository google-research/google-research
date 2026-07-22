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

"""Extract clean text from web corpus WARCs and embed it for the search environment.

The pipeline has three stages running concurrently so the embedding GPUs
stay saturated by the time the extractors are running:

1. **Reader** (one thread per process) streams WARC records and pushes raw
   ``(warc_name, url, crawl_date, html)`` tuples into a bounded queue.
2. **Extractors** (process pool) call :func:`extract_article` on each HTML
   record and push :class:`~financegym.common.schemas.Article` instances
   onto the next bounded queue.
3. **Embedders** (thread pool) pull batches from that queue, call the
   embedding server, and hand off ``(batch_id, articles, vectors)`` to the
   writer thread which appends to ``embeddings.bin``, ``metadata.jsonl``,
   and ``texts.jsonl`` in batch order.

Three on-disk files are written into ``output_dir``:

* ``embeddings.bin``   — header ``<ii>`` ``(n, dim)`` followed by
  ``n * dim`` float32 values. Pinned to ``Qwen3-Embedding-4B`` (2,560 dim)
  as the canonical embedder; see :mod:`docs/reproducibility.md`.
* ``metadata.jsonl``   — one :class:`~financegym.common.schemas.Document`
  per line.
* ``texts.jsonl``      — one ``{"doc_id", "text"}`` per line.

The unit-testable parts of the pipeline (HTML → text, URL → domain, binary
header pack/unpack, JSONL record format) are split out as pure functions so
each can be exercised without WARCs or an embedding server.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import struct
import threading
import time
from urllib.parse import urlparse

import numpy as np
import requests
from warcio.archiveiterator import ArchiveIterator

log = logging.getLogger(__name__)
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
logging.getLogger("htmldate").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Canonical embedder pinned for benchmark reproducibility. Override only when
# running an alternative embedding contract.
# ---------------------------------------------------------------------------

DEFAULT_EMBED_URL = os.environ.get(
    "EMBED_URL", "http://127.0.0.1:8888/v1/embeddings"
)
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
DEFAULT_EMBED_DIM = int(os.environ.get("EMBED_DIM", "2560"))
DEFAULT_BATCH_SIZE = 128
DEFAULT_EMBED_WORKERS = 8
MAX_TEXT_LEN = 120_000  # ~40K tokens; not truncation, just an upper bound

# Queue sizings: tuned for ~5 WARCs worth of HTML in Pool A and ~500K
# extracted articles in Pool B. Plenty of headroom on a 1TB+ RAM box.
POOL_A_SIZE = 125_000
POOL_B_SIZE = 500_000

MIN_TEXT_LEN = 100  # discard records that yielded almost nothing


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def normalize_domain(url):
  """Strip a leading ``www.`` and lowercase the netloc."""
  netloc = urlparse(url).netloc.lower()
  return netloc[4:] if netloc.startswith("www.") else netloc


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _fallback_strip(html):
  """Minimal HTML-to-text used only if trafilatura is unavailable."""
  return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()


def extract_article(
    html,
    url,
    crawl_date,
    warc_name = "",
):
  """Run trafilatura + htmldate on one HTML record.

  Returns ``None`` for records that yield no usable text (under
  :data:`MIN_TEXT_LEN` characters). On success returns a plain ``dict``
  matching the :class:`~financegym.common.schemas.Article` shape — kept as
  a dict so it's cheap to pass across the multiprocessing queue.
  """
  try:
    from trafilatura import extract as trafilatura_extract
  except ImportError:  # pragma: no cover - trafilatura is a hard dep
    trafilatura_extract = None
  try:
    from htmldate import find_date
  except ImportError:  # pragma: no cover - htmldate is a hard dep
    find_date = None

  text = (
      trafilatura_extract(html)
      if trafilatura_extract
      else _fallback_strip(html)
  ) or ""
  if len(text) < MIN_TEXT_LEN:
    return None

  pub_date = ""
  if find_date is not None:
    try:
      pub_date = find_date(html) or ""
    except Exception:
      pub_date = ""

  if not pub_date and crawl_date:
    pub_date = crawl_date[:10]

  return {
      "warc_name": warc_name,
      "url": url,
      "domain": normalize_domain(url),
      "pub_date": pub_date,
      "crawl_date": crawl_date or "",
      "text": text,
  }


def pack_header(n, dim):
  """Pack the leading 8-byte little-endian ``(n, dim)`` header."""
  return struct.pack("<ii", n, dim)


def unpack_header(buf):
  """Inverse of :func:`pack_header`."""
  return struct.unpack("<ii", buf[:8])


def metadata_line(article, doc_id):
  """JSON line for ``metadata.jsonl``."""
  return json.dumps({
      "doc_id": doc_id,
      "url": article["url"],
      "domain": article["domain"],
      "pub_date": article["pub_date"],
      "crawl_date": article["crawl_date"],
      "text_len": len(article["text"]),
  })


def text_line(article, doc_id):
  """JSON line for ``texts.jsonl``."""
  return json.dumps({"doc_id": doc_id, "text": article["text"]})


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------


def embed_batch(
    texts,
    *,
    url = DEFAULT_EMBED_URL,
    model = DEFAULT_EMBED_MODEL,
    timeout = 300.0,
):
  """Embed a batch via an OpenAI-compatible ``/v1/embeddings`` endpoint.

  The canonical server is vLLM with Qwen3-Embedding-4B; any backend that
  speaks the same JSON dialect can be substituted by setting ``EMBED_URL``.
  """
  r = requests.post(url, json={"model": model, "input": texts}, timeout=timeout)
  r.raise_for_status()
  data = sorted(r.json()["data"], key=lambda x: x["index"])
  return [d["embedding"] for d in data]


# ---------------------------------------------------------------------------
# Stage threads / processes
# ---------------------------------------------------------------------------


def _reader_thread(warc_files, pool_a):
  for wf in warc_files:
    try:
      with open(wf, "rb") as f:
        for record in ArchiveIterator(f):
          if record.rec_type != "response":
            continue
          ct = (
              record.http_headers.get_header("Content-Type")
              if record.http_headers
              else ""
          )
          if not ct or "html" not in ct.lower():
            continue
          url = record.rec_headers.get_header("WARC-Target-URI") or ""
          crawl_date = record.rec_headers.get_header("WARC-Date") or ""
          try:
            html = (
                record.content_stream().read().decode("utf-8", errors="ignore")
            )
          except Exception:
            continue
          pool_a.put((wf.name, url, crawl_date, html))
    except Exception as e:  # noqa: BLE001
      log.warning("reader failed on %s: %s", wf.name, e)
    pool_a.put((wf.name, None, None, None))  # end-of-WARC sentinel
  pool_a.put((None, None, None, None))  # all-done sentinel


def _extractor_worker(pool_a, pool_b):
  while True:
    warc_name, url, crawl_date, html = pool_a.get()
    if url is None:
      pool_b.put((warc_name, None, None, None))
      if warc_name is None:
        break
      continue
    article = extract_article(html, url, crawl_date, warc_name=warc_name)
    if article is not None:
      pool_b.put(article)


def _embedder_thread(
    batch_queue,
    result_queue,
    *,
    url,
    model,
):
  while True:
    item = batch_queue.get()
    if item[0] is None:
      result_queue.put((None, None, None))
      break
    batch_id, articles = item
    texts = [a["text"][:MAX_TEXT_LEN] for a in articles]
    embs: list[list[float]] | None = None
    for attempt in range(3):
      try:
        embs = embed_batch(texts, url=url, model=model)
        break
      except Exception as e:  # noqa: BLE001
        if attempt < 2:
          time.sleep(2**attempt)
          log.warning("embed batch %s retry %d/2: %s", batch_id, attempt + 1, e)
        else:
          log.error(
              "embed batch %s FAILED after 3 attempts, discarding %d articles",
              batch_id,
              len(articles),
          )
    if embs is not None and len(embs) == len(articles):
      result_queue.put((batch_id, articles, embs))
    else:
      result_queue.put((batch_id, [], []))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def load_checkpoint(path):
  if not path.exists():
    return set()
  return set(line for line in path.read_text().splitlines() if line)


def save_checkpoint(path, warc_name):
  with open(path, "a") as f:
    f.write(warc_name + "\n")


def run(
    input_path,
    output_dir,
    *,
    workers = None,
    embed_workers = DEFAULT_EMBED_WORKERS,
    embed_batch_size = DEFAULT_BATCH_SIZE,
    embed_url = DEFAULT_EMBED_URL,
    embed_model = DEFAULT_EMBED_MODEL,
    embed_dim = DEFAULT_EMBED_DIM,
):
  """Three-stage pipelined extraction + embedding.

  Returns total articles written.
  """
  warc_dir = Path(input_path)
  warc_files = sorted(warc_dir.rglob("*.warc.gz"))
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if not warc_files:
    log.error("no WARC files found in %s", input_path)
    return 0

  checkpoint = out_dir / "checkpoint.txt"
  done = load_checkpoint(checkpoint)
  if done:
    warc_files = [wf for wf in warc_files if wf.name not in done]
    log.info(
        "resuming: %d WARCs already done, %d remaining",
        len(done),
        len(warc_files),
    )
  if not warc_files:
    log.info("all WARCs already processed")
    return 0

  n_cpus = os.cpu_count() or 4
  if workers is None:
    workers = max(1, n_cpus - 16)
  log.info(
      "stage 1 (reader): 1 thread | stage 2 (extractors): %d procs | stage 3"
      " (embedders): %d threads, batch=%d",
      workers,
      embed_workers,
      embed_batch_size,
  )

  emb_path = out_dir / "embeddings.bin"
  meta_path = out_dir / "metadata.jsonl"
  texts_path = out_dir / "texts.jsonl"

  # The three output files outlive this function-level scope: the writer
  # thread holds references and we only close them after `writer.join()`.
  if done:
    with open(emb_path, "rb") as f:
      total_articles, _ = unpack_header(f.read(8))
    emb_file = open(emb_path, "r+b")  # noqa: SIM115
    emb_file.seek(0, 2)
    meta_file = open(meta_path, "a")  # noqa: SIM115
    texts_file = open(texts_path, "a")  # noqa: SIM115
    log.info("resuming from %d articles", total_articles)
  else:
    emb_file = open(emb_path, "wb")  # noqa: SIM115
    emb_file.write(pack_header(0, embed_dim))
    meta_file = open(meta_path, "w")  # noqa: SIM115
    texts_file = open(texts_path, "w")  # noqa: SIM115
    total_articles = 0

  next_doc_id = total_articles

  pool_a: mp.Queue = mp.Queue(maxsize=POOL_A_SIZE)
  pool_b: mp.Queue = mp.Queue(maxsize=POOL_B_SIZE)
  pool_a_shared: mp.Queue = mp.Queue(maxsize=POOL_A_SIZE)

  reader = threading.Thread(
      target=_reader_thread, args=(warc_files, pool_a), daemon=True
  )
  reader.start()

  def _dispatcher():
    while True:
      item = pool_a.get()
      warc_name, url, _, _ = item
      if warc_name is None and url is None:
        for _ in range(workers):
          pool_a_shared.put((None, None, None, None))
        break
      pool_a_shared.put(item)

  dispatcher = threading.Thread(target=_dispatcher, daemon=True)
  dispatcher.start()

  extractors = [
      mp.Process(
          target=_extractor_worker, args=(pool_a_shared, pool_b), daemon=True
      )
      for _ in range(workers)
  ]
  for p in extractors:
    p.start()

  batch_queue: queue.Queue = queue.Queue(maxsize=embed_workers * 4)
  result_queue: queue.Queue = queue.Queue(maxsize=embed_workers * 4)
  embedders = [
      threading.Thread(
          target=_embedder_thread,
          args=(batch_queue, result_queue),
          kwargs={"url": embed_url, "model": embed_model},
          daemon=True,
      )
      for _ in range(embed_workers)
  ]
  for t in embedders:
    t.start()

  pending: list[dict] = []
  workers_done = 0
  batch_counter = [0]

  def _collector():
    nonlocal workers_done
    while True:
      try:
        item = pool_b.get(timeout=1)
      except queue.Empty:
        continue
      if isinstance(item, tuple):
        warc_name, url, _, _ = item
        if warc_name is None:
          workers_done += 1
          if workers_done >= workers:
            if pending:
              batch_queue.put((batch_counter[0], list(pending)))
              batch_counter[0] += 1
              pending.clear()
            for _ in range(embed_workers):
              batch_queue.put((None, None))
            break
        continue
      pending.append(item)
      if len(pending) >= embed_batch_size:
        batch_queue.put((batch_counter[0], list(pending)))
        batch_counter[0] += 1
        pending.clear()

  collector = threading.Thread(target=_collector, daemon=True)
  collector.start()

  embedders_finished = [0]

  def _writer():
    nonlocal total_articles, next_doc_id
    next_write = 0
    buf: dict[int, tuple[list[dict], list[list[float]]]] = {}
    while True:
      batch_id, articles, embs = result_queue.get()
      if batch_id is None:
        embedders_finished[0] += 1
        if embedders_finished[0] >= embed_workers:
          break
        continue
      buf[batch_id] = (articles, embs)
      while next_write in buf:
        arts, vecs = buf.pop(next_write)
        next_write += 1
        if not arts:
          continue
        arr = np.asarray(vecs, dtype=np.float32)
        assert arr.shape == (len(arts), embed_dim)
        emb_file.write(arr.tobytes())
        for i, a in enumerate(arts):
          doc_id = f"doc_{next_doc_id + i}"
          meta_file.write(metadata_line(a, doc_id) + "\n")
          texts_file.write(text_line(a, doc_id) + "\n")
        next_doc_id += len(arts)
        total_articles += len(arts)

    emb_file.seek(0)
    emb_file.write(pack_header(total_articles, embed_dim))
    emb_file.flush()
    meta_file.flush()
    texts_file.flush()

  writer = threading.Thread(target=_writer, daemon=True)
  writer.start()

  reader.join()
  dispatcher.join()
  for p in extractors:
    p.join()
  collector.join()
  for t in embedders:
    t.join()
  writer.join()

  for wf in warc_files:
    save_checkpoint(checkpoint, wf.name)

  emb_file.close()
  meta_file.close()
  texts_file.close()

  log.info("done: %d articles in %s", total_articles, out_dir)
  return total_articles


def main():
  parser = argparse.ArgumentParser(
      description="Extract + embed the web corpus into the search store."
  )
  parser.add_argument("input", help="Directory containing .warc.gz files")
  parser.add_argument("--output", default="output/search/")
  parser.add_argument("--workers", type=int, default=None)
  parser.add_argument(
      "--embed-workers", type=int, default=DEFAULT_EMBED_WORKERS
  )
  parser.add_argument("--embed-batch", type=int, default=DEFAULT_BATCH_SIZE)
  parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
  parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
  parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )
  run(
      args.input,
      args.output,
      workers=args.workers,
      embed_workers=args.embed_workers,
      embed_batch_size=args.embed_batch,
      embed_url=args.embed_url,
      embed_model=args.embed_model,
      embed_dim=args.embed_dim,
  )


if __name__ == "__main__":
  main()
