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

"""HTTP+SSE service — exposes the harness over a versioned API.

`POST /research` (sync JSON, or SSE when stream=true) is the core; alongside it:
`POST /clarify` (scoping), `POST /compact` (session summarization), `GET
/sessions`
+ `/sessions/{id}` (resume), `GET /status`, `GET /models`, `GET /health`. The
agent
runs async per request with its own cache/registry (per-request isolation); the
same trajectory shape backs both the sync body and the streamed `done` event.

Run: ``uvicorn financeharness.service.app:app`` (the FastAPI object lives in the
``app`` submodule — not re-exported here, to keep the submodule importable).
"""
