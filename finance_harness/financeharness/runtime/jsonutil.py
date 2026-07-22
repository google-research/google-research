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

"""Robust JSON-object extraction from model output (fenced / prose-wrapped)."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_obj(raw):
  """Parse a JSON object from model text, tolerating code fences / surrounding

  prose. Returns None if no object is found.
  """
  raw = (raw or "").strip()
  try:
    obj = json.loads(raw)
    return obj if isinstance(obj, dict) else None
  except (json.JSONDecodeError, ValueError):
    pass
  m = re.search(r"\{.*\}", raw, re.DOTALL)
  if not m:
    return None
  try:
    obj = json.loads(m.group(0))
    return obj if isinstance(obj, dict) else None
  except (json.JSONDecodeError, ValueError):
    return None
