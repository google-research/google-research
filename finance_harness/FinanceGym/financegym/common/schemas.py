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

"""Pydantic models for the FinanceGym on-disk JSONL formats.

This is the single source of truth for the schemas the rest of the pipeline
reads and writes. Other modules import the model they need; downstream
tools (judges, leaderboards) mirror these models to validate submissions.

The full schema set lands incrementally as each module ships:

==========  ===========================================================
==========
Stage       Models                                                       Lands
in
==========  ===========================================================
==========
corpus      :class:`Article`, :class:`Document`                           Step 2
graph       :class:`Edge` (forward-declared here, finalised in graph)     Step 4
questions   ``Question``, ``RubricItem``                                  Step 5
judge       ``Answer``, ``ScoreItem``, ``Score``                          Step 7
==========  ===========================================================
==========
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Article(BaseModel):
  """An article after WARC extraction but before embedding.

  The corpus pipeline writes nothing to disk in this exact shape; this is
  the in-memory record passed between the extractor and the embedder.
  """

  model_config = ConfigDict(extra="forbid")

  warc_name: str = Field(
      description="Source WARC file basename, used for checkpointing."
  )
  url: str
  domain: str = Field(description="Hostname with leading ``www.`` stripped.")
  pub_date: str = Field(
      default="",
      description=(
          "ISO date (YYYY-MM-DD) extracted via htmldate; empty if unknown."
      ),
  )
  crawl_date: str = Field(
      default="",
      description="Crawl timestamp from the archive header.",
  )
  text: str


class Document(BaseModel):
  """The persisted form of an embedded article.

  Mirrors one line of ``metadata.jsonl`` produced by
  :func:`financegym.corpus.extract_embed.run`. The full article text lives
  alongside in ``texts.jsonl`` (one ``{"doc_id", "text"}`` per line).
  """

  model_config = ConfigDict(extra="forbid")

  doc_id: str = Field(
      description="Stable corpus-wide identifier, e.g. ``doc_0042``."
  )
  url: str
  domain: str
  pub_date: str = ""
  crawl_date: str = ""
  text_len: int = Field(
      ge=0, description="Character length of the extracted text."
  )
