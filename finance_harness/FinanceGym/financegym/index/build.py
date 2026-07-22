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

"""Build a FAISS index over ``embeddings.bin``.

Default ``ivf_sq8`` matches the canonical FinanceEnv contract:
``Qwen3-Embedding-4B`` (2,560-dim, L2-normalized) → IVF-SQ8 with
``nlist = clamp(sqrt(nvecs), 256, 65_536)`` and ``nprobe = 32`` at
query time. Other index types (``flat``, ``flat_sq8``, ``ivf_flat``) are
useful for smoke tests and small corpora.

The embeddings file is memory-mapped, so this works against corpora that
exceed RAM. Training runs on GPUs when available (auto-detected through
``faiss.get_num_gpus``) and falls back to CPU when not.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import struct
import time

import faiss
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INDEX_TYPE = "ivf_sq8"
DEFAULT_NPROBE = 32
NLIST_MIN = 256
NLIST_MAX = 65_536
INDEX_FILENAME = "faiss_index.bin"
EMB_FILENAME = "embeddings.bin"
ADD_BATCH = 1_000_000


# ---------------------------------------------------------------------------
# Pure helpers (testable without faiss)
# ---------------------------------------------------------------------------


def auto_nlist(nvecs):
  """Pick the IVF cluster count from corpus size.

  ``sqrt(n)`` is the standard rule of thumb; we clamp to a sane band so
  very small and very large corpora still get reasonable indexes.
  """
  return max(NLIST_MIN, min(int(nvecs**0.5), NLIST_MAX))


def read_embedding_header(path):
  """Return ``(nvecs, dim)`` from the leading 8-byte header."""
  with open(path, "rb") as f:
    return struct.unpack("<ii", f.read(8))  # type: ignore[return-value]


def memmap_embeddings(path):
  """Memory-map the float32 payload after the 8-byte header."""
  nvecs, dim = read_embedding_header(path)
  arr = np.memmap(
      path, dtype=np.float32, mode="r", offset=8, shape=(nvecs, dim)
  )
  return arr, nvecs, dim


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _read_batch(data, start, end):
  """Copy a slice out of the mmap into a writable, L2-normalized batch."""
  batch = np.array(data[start:end], dtype=np.float32)
  faiss.normalize_L2(batch)
  return batch


def _build_flat(data, nvecs, dim):
  index = faiss.IndexFlatIP(dim)
  for start in range(0, nvecs, ADD_BATCH):
    index.add(_read_batch(data, start, min(start + ADD_BATCH, nvecs)))
  return index


def _build_flat_sq8(data, nvecs, dim):
  index = faiss.IndexScalarQuantizer(
      dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
  )
  index.train(_read_batch(data, 0, min(ADD_BATCH, nvecs)))
  for start in range(0, nvecs, ADD_BATCH):
    index.add(_read_batch(data, start, min(start + ADD_BATCH, nvecs)))
  return index


def _build_ivf(
    data,
    nvecs,
    dim,
    *,
    index_type,
    nlist,
    nprobe,
):
  quantizer = faiss.IndexFlatIP(dim)
  if index_type == "ivf_sq8":
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dim,
        nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )
  elif index_type == "ivf_flat":
    index = faiss.IndexIVFFlat(
        quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
    )
  else:  # pragma: no cover - guarded by caller
    raise ValueError(index_type)

  # Train on a representative sample; GPU-accelerate clustering if present.
  ngpu = getattr(faiss, "get_num_gpus", lambda: 0)()
  if ngpu > 0:
    log.info("training k-means on %d GPU(s)", ngpu)
    index.clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(dim))

  train_size = min(nvecs, max(nlist * 256, 1_000_000))
  rng = np.random.default_rng(42)
  train_idx = rng.choice(nvecs, train_size, replace=False)
  train_data = np.array(data[train_idx], dtype=np.float32)
  faiss.normalize_L2(train_data)
  index.train(train_data)
  del train_data
  if ngpu > 0:
    index.clustering_index = None

  for start in range(0, nvecs, ADD_BATCH):
    index.add(_read_batch(data, start, min(start + ADD_BATCH, nvecs)))
  index.nprobe = nprobe
  return index


def build(
    embeddings_path,
    out_path,
    *,
    index_type = DEFAULT_INDEX_TYPE,
    nlist = None,
    nprobe = DEFAULT_NPROBE,
):
  """Build and persist the FAISS index. Returns the in-memory index too."""
  data, nvecs, dim = memmap_embeddings(embeddings_path)
  log.info("memory-mapped %d x %d embeddings", nvecs, dim)

  t0 = time.time()
  if index_type == "flat":
    index = _build_flat(data, nvecs, dim)
  elif index_type == "flat_sq8":
    index = _build_flat_sq8(data, nvecs, dim)
  elif index_type in ("ivf_sq8", "ivf_flat"):
    if nlist is None:
      nlist = auto_nlist(nvecs)
    index = _build_ivf(
        data, nvecs, dim, index_type=index_type, nlist=nlist, nprobe=nprobe
    )
  else:
    raise ValueError(f"unknown index type: {index_type!r}")

  log.info("built %s in %.1fs", index_type, time.time() - t0)
  faiss.write_index(index, str(out_path))
  log.info("wrote %s (%.2f GB)", out_path, out_path.stat().st_size / 1e9)
  return index


def main():
  parser = argparse.ArgumentParser(
      description="Build FAISS index over embeddings.bin."
  )
  parser.add_argument(
      "--input", default="output/search/", help="Directory with embeddings.bin"
  )
  parser.add_argument(
      "--index-type",
      default=DEFAULT_INDEX_TYPE,
      choices=["ivf_sq8", "ivf_flat", "flat_sq8", "flat"],
  )
  parser.add_argument("--nlist", type=int, default=None)
  parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
  parser.add_argument(
      "--output-name",
      default=INDEX_FILENAME,
      help="Output filename written into --input dir",
  )
  args = parser.parse_args()
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
  )

  inp = Path(args.input)
  build(
      inp / EMB_FILENAME,
      inp / args.output_name,
      index_type=args.index_type,
      nlist=args.nlist,
      nprobe=args.nprobe,
  )


if __name__ == "__main__":
  main()
