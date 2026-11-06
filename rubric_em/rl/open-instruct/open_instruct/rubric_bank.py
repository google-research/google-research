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

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rubric Bank for storing and retrieving per-question reflection items.

Stores per-question reflection items (rubrics + takeaways) for retrieval as
few-shot references when the agent encounters similar questions.

Operations:
  - insert(item): Add new bank item, compute embedding, update index
  - update(question, rubrics, reflections, step): Replace existing item
  - retrieve(query, top_k): Find similar questions via embedding similarity
  - save(path) / load(path): Persist to / restore from disk
"""

import dataclasses
import hashlib
import logging
import os
import pickle
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class RubricBankItem:
  """A single bank item: one per question."""

  question: str
  rubrics: str  # Stage-1 style rubrics (agent planning format)
  reflections: str  # Key takeaways beyond rubrics
  embedding: Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
  question_hash: str = ""
  created_step: int = 0
  num_updates: int = 0

  def __post_init__(self):
    if not self.question_hash:
      self.question_hash = hashlib.sha256(self.question.encode()).hexdigest()


class RubricBank:
  """Embedding-indexed bank of rubric items for retrieval.

  Uses sentence-transformers for embedding and FAISS for similarity search.
  Thread-safe for concurrent reads; writes should be serialized (rank 0 only).
  """

  def __init__(
      self,
      embedding_model = "all-MiniLM-L6-v2",
      embedding_dim = 384,
      lazy_load_model = True,
  ):
    self.embedding_model_name = embedding_model
    self.embedding_dim = embedding_dim
    self._model = None
    self._items: Dict[str, RubricBankItem] = {}  # question_hash -> item
    self._index = None  # FAISS index
    self._hash_order: List[str] = []  # maps FAISS row idx -> question_hash
    self._lock = (
        threading.RLock()
    )  # protects _items, _index, _hash_order for concurrent access

    if not lazy_load_model:
      self._ensure_model()

  def _ensure_model(self):
    """Lazy-load the embedding model on first use."""
    if self._model is None:
      with self._lock:
        if self._model is None:
          try:
            from sentence_transformers import SentenceTransformer  # pylint: disable=g-import-not-at-top,g-importing-member

            LOGGER.info(
                "[RubricBank] Loading embedding model: %s",
                self.embedding_model_name,
            )
            self._model = SentenceTransformer(
                self.embedding_model_name, device="cpu"
            )
            # Update dim from actual model
            self.embedding_dim = self._model.get_sentence_embedding_dimension()
            LOGGER.info("[RubricBank] Model loaded, dim=%d", self.embedding_dim)
          except ImportError:
            LOGGER.error(
                "sentence-transformers not installed. Run: pip install"
                " sentence-transformers"
            )
            raise

  def _embed(self, texts):
    """Embed a list of texts. Returns (N, dim) float32 array, L2-normalized."""
    self._ensure_model()
    embeddings = self._model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)

  def _rebuild_index(self):
    """Rebuild FAISS index from all item embeddings."""
    import faiss  # pylint: disable=g-import-not-at-top

    self._hash_order = []

    if not self._items:
      self._index = faiss.IndexFlatIP(self.embedding_dim)
      return

    embeddings = []
    for qhash, item in self._items.items():
      if item.embedding is not None:
        self._hash_order.append(qhash)
        embeddings.append(item.embedding)

    if embeddings:
      matrix = np.stack(embeddings).astype(np.float32)
      # Infer dimension from stored embeddings (may differ from default)
      self.embedding_dim = matrix.shape[1]
      self._index = faiss.IndexFlatIP(self.embedding_dim)
      self._index.add(matrix)
    else:
      self._index = faiss.IndexFlatIP(self.embedding_dim)

  def __len__(self):
    with self._lock:
      return len(self._items)

  def __contains__(self, question):
    qhash = hashlib.sha256(question.encode()).hexdigest()
    with self._lock:
      return qhash in self._items

  def insert(self, item):
    """Insert a new item. If question already exists, delegates to update."""
    # Compute embedding OUTSIDE the lock (CPU-intensive)
    if item.embedding is None:
      item.embedding = self._embed([item.question])[0]

    with self._lock:
      if item.question_hash in self._items:
        self.update(
            item.question,
            item.rubrics,
            item.reflections,
            item.created_step,
        )
        return

      self._items[item.question_hash] = item

      # Add to FAISS index
      if self._index is None:
        self._rebuild_index()
      else:
        self._hash_order.append(item.question_hash)
        self._index.add(item.embedding.reshape(1, -1).astype(np.float32))

      bank_size = len(self._items)

    LOGGER.info(
        "[RubricBank] Inserted item for question '%s...' (bank size: %d)",
        item.question[:60],
        bank_size,
    )

  def update(
      self,
      question,
      rubrics,
      reflections,
      step = 0,
  ):
    """Update an existing item's rubrics and reflections."""
    qhash = hashlib.sha256(question.encode()).hexdigest()
    num_updates = 0
    with self._lock:
      if qhash not in self._items:
        exists = False
      else:
        exists = True
        existing = self._items[qhash]
        existing.rubrics = rubrics
        existing.reflections = reflections
        existing.num_updates += 1
        num_updates = existing.num_updates

    if not exists:
      # Not found — insert as new
      self.insert(
          RubricBankItem(
              question=question,
              rubrics=rubrics,
              reflections=reflections,
              created_step=step,
          )
      )
      return

    LOGGER.info(
        "[RubricBank] Updated item for question '%s...' (updates: %d)",
        question[:60],
        num_updates,
    )

  def retrieve(
      self,
      query,
      top_k = 2,
      exclude_self = True,
      stochastic = False,
  ):
    """Retrieve top_k most similar bank items for a query.

    Args:
      query: Query string to match against bank items.
      top_k: Number of items to retrieve.
      exclude_self: If True, the exact same question is excluded from results.
      stochastic: If True, sample top_k items from top_2k candidates with
        probability proportional to similarity scores (adds retrieval
        diversity).

    Returns:
      List of (item, similarity_score) pairs, sorted by descending similarity.
    """
    # Compute embedding OUTSIDE the lock (CPU-intensive)
    query_embedding = self._embed([query])[0].reshape(1, -1).astype(np.float32)
    query_hash = hashlib.sha256(query.encode()).hexdigest()

    with self._lock:
      if not self._items or self._index is None or self._index.ntotal == 0:
        return []

      # Search pool: top_2k candidates (or top_k+1 if deterministic)
      pool_size = top_k * 2 if stochastic else top_k
      search_k = min(
          pool_size + 1, self._index.ntotal
      )  # +1 for possible self-exclusion
      scores, indices = self._index.search(query_embedding, search_k)

      candidates = []
      for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(self._hash_order):
          continue
        item_hash = self._hash_order[idx]
        if exclude_self and item_hash == query_hash:
          continue
        item = self._items[item_hash]
        candidates.append((item, float(score)))

    if not candidates:
      return []

    if stochastic and len(candidates) > top_k:
      # Sample top_k from candidates with probability proportional to similarity
      cand_scores = np.array([s for _, s in candidates])
      # Shift to positive and normalize (scores can be negative for inner
      # product)
      shifted = cand_scores - cand_scores.min() + 1e-6
      probs = shifted / shifted.sum()
      chosen_indices = np.random.choice(
          len(candidates),
          size=min(top_k, len(candidates)),
          replace=False,
          p=probs,
      )
      results = [candidates[i] for i in sorted(chosen_indices)]
    else:
      results = candidates[:top_k]

    return results

  def get(self, question):
    """Get a bank item by exact question match."""
    qhash = hashlib.sha256(question.encode()).hexdigest()
    with self._lock:
      return self._items.get(qhash)

  def save(self, directory):
    """Save bank to disk with atomic write."""
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, "rubric_bank.pkl")

    # Prepare serializable data (exclude FAISS index and model)
    with self._lock:
      data = {
          "items": {
              qhash: {
                  "question": item.question,
                  "rubrics": item.rubrics,
                  "reflections": item.reflections,
                  "embedding": item.embedding,
                  "question_hash": item.question_hash,
                  "created_step": item.created_step,
                  "num_updates": item.num_updates,
              }
              for qhash, item in self._items.items()
          },
          "embedding_model": self.embedding_model_name,
          "embedding_dim": self.embedding_dim,
      }
      num_items = len(self._items)

    # Atomic write: write to temp, then rename
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
      with os.fdopen(fd, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
      os.replace(tmp_path, save_path)
      LOGGER.info("[RubricBank] Saved %d items to %s", num_items, save_path)
    except Exception:
      if os.path.exists(tmp_path):
        os.unlink(tmp_path)
      raise

  def load(self, directory):
    """Load bank from disk and rebuild FAISS index."""
    load_path = os.path.join(directory, "rubric_bank.pkl")
    if not os.path.exists(load_path):
      LOGGER.warning("[RubricBank] No bank file at %s", load_path)
      return

    with open(load_path, "rb") as f:
      data = pickle.load(f)  # pylint: disable=g-unsafe-pickle-load

    with self._lock:
      self.embedding_model_name = data.get(
          "embedding_model", self.embedding_model_name
      )
      self.embedding_dim = data.get("embedding_dim", self.embedding_dim)
      self._items = {}
      for qhash, item_data in data["items"].items():
        self._items[qhash] = RubricBankItem(
            question=item_data["question"],
            rubrics=item_data["rubrics"],
            reflections=item_data["reflections"],
            embedding=item_data.get("embedding"),
            question_hash=item_data["question_hash"],
            created_step=item_data.get("created_step", 0),
            num_updates=item_data.get("num_updates", 0),
        )

      self._rebuild_index()
      num_loaded = len(self._items)

    LOGGER.info("[RubricBank] Loaded %d items from %s", num_loaded, load_path)

  def get_stats(self):
    """Return summary statistics for logging."""
    with self._lock:
      if not self._items:
        return {"bank_size": 0}
      update_counts = [item.num_updates for item in self._items.values()]
      return {
          "bank_size": len(self._items),
          "avg_updates": np.mean(update_counts) if update_counts else 0.0,
          "max_updates": max(update_counts) if update_counts else 0,
          "index_size": self._index.ntotal if self._index else 0,
      }
