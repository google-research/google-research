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

"""Conversational sessions — pure helpers for multi-turn threading.

A session threads several runs: each turn is seeded with the prior turns'
messages (system stripped — the new run supplies a fresh system prompt), so a
follow-up is grounded in what earlier turns fetched. A turn is committed to the
session only when it cleanly answered — the **no-poison-on-error** guard, so a
failed/aborted turn never corrupts the thread and a retry starts clean.

Storage is the caller's concern (the service uses a file-backed
``SessionStore``);
these helpers are pure + testable.
"""

from __future__ import annotations

from typing import Any


def strip_system(messages):
  """Conversation history minus any system message (the next turn adds its own)."""
  return [m for m in messages or [] if m.get("role") != "system"]


def is_committable(termination):
  """True only for a cleanly-answered turn — the no-poison-on-error guard."""
  return termination == "answer"
