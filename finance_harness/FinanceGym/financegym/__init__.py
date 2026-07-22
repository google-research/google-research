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

"""FinanceGym: point-in-time agentic finance research benchmark + retrieval environment.

Subpackages:
    common      shared utilities (LLM client, pydantic schemas)
    corpus      web corpus ingestion → clean text + publication date +
    embeddings
    index       FAISS index build over embeddings
    env         the point-in-time search server and its client
    graph       finance entity graph (domain whitelist, triple extraction,
    relation taxonomy)
    questions   event detection, situation mining, cutoff selection, LLM
    question generation
    curation    multi-stage filter funnel (feasibility, relevance, coherence,
    taxonomy, ILP)
    judge       5-tier rubric judge and aggregation
"""

__version__ = "0.1.0.dev0"
