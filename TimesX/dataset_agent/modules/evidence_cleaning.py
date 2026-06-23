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

"""Evidence cleaning utilities for crawled source content."""

import logging
import re
from typing import Any, Dict, Optional, Tuple

logging.getLogger("trafilatura").setLevel(logging.ERROR)


ERROR_MARKERS = (
    "access denied",
    "are you a robot",
    "captcha",
    "checking your browser",
    "content not available",
    "enable javascript",
    "forbidden",
    "login required",
    "page does not exist",
    "page not found",
    "permission denied",
    "please verify you are human",
    "server error",
    "sign in to continue",
    "temporarily unavailable",
    "too many requests",
)


BOILERPLATE_MARKERS = (
    "advertisement",
    "all rights reserved",
    "an official website of the united states government",
    "breadcrumb",
    "census.gov >",
    "connection to some features may be unavailable",
    "cookie policy",
    "end of header",
    "explore our apps",
    "follow us",
    "fred add-in",
    "fred api",
    "fred mobile apps",
    "privacy policy",
    "share this page",
    "sign up for email updates",
    "skip to looking for section",
    "skip to main content",
    "subscribe",
    "terms of use",
    "thank you for your patience",
    "we apologize for any inconvenience",
    "will undergo scheduled maintenance",
)


MONTH_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b20[0-9]{2}\b")


def _as_text(value):
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  return str(value)


def _get_attr_text(obj, attr):
  if obj is None:
    return ""
  return _as_text(getattr(obj, attr, ""))


def _normalize_markdown(text):
  text = _as_text(text)
  text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
  text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
  text = re.sub(r"https?://\S+", " ", text)

  lines = []
  seen = set()
  for raw_line in text.splitlines():
    line = re.sub(r"\s+", " ", raw_line).strip(" #*-|\t")
    if not line:
      continue
    key = line.lower()
    if any(marker in key for marker in BOILERPLATE_MARKERS):
      continue
    if len(line) < 4 or key in seen:
      continue
    seen.add(key)
    lines.append(line)

  first_block = " ".join(lines[:12]).lower()
  if any(
      marker in first_block
      for marker in ("explore our apps", "fred mobile apps", "news blog about")
  ):
    for idx, line in enumerate(lines[:80]):
      if MONTH_PATTERN.search(line) or YEAR_PATTERN.search(line):
        lines = lines[idx:]
        break

  if lines:
    normalized = "\n".join(lines)
  else:
    normalized = re.sub(r"\s+", " ", text).strip()

  return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def _looks_invalid(text):
  sample = text.lower()[:1500]
  if not sample:
    return True
  return any(marker in sample for marker in ERROR_MARKERS)


def _extract_trafilatura_text(crawl_result):
  html = _get_attr_text(crawl_result, "html") or _get_attr_text(
      crawl_result, "cleaned_html"
  )
  if not html:
    return "", "no_html"
  try:
    import trafilatura
  except Exception:
    return "", "trafilatura_unavailable"
  try:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        deduplicate=True,
        favor_precision=True,
    )
  except Exception as exc:
    return "", f"trafilatura_error:{type(exc).__name__}"
  return _as_text(extracted), None


def clean_crawl_evidence(
    crawl_result, config = None
):
  """Return cleaned evidence text and metadata for a crawl4ai result.

  The preferred source is crawl4ai filtered markdown when available, then
  trafilatura-extracted HTML main text, then raw crawl4ai markdown.
  """

  cleaning_config = ((config or {}).get("event_sourcing") or {}).get(
      "evidence_cleaning", {}
  )
  min_clean_length = int(cleaning_config.get("min_clean_length", 250))

  markdown_obj = getattr(crawl_result, "markdown", None)
  raw_markdown = _as_text(markdown_obj)
  raw_length = len(raw_markdown)

  candidates = []
  filtered_markdown = (
      _get_attr_text(crawl_result, "fit_markdown")
      or _get_attr_text(
          getattr(crawl_result, "markdown_v2", None), "fit_markdown"
      )
      or _get_attr_text(markdown_obj, "fit_markdown")
  )
  if filtered_markdown:
    candidates.append(("crawl4ai_filtered_markdown", filtered_markdown))

  trafilatura_text, trafilatura_status = _extract_trafilatura_text(crawl_result)
  if trafilatura_text:
    candidates.append(("trafilatura_html", trafilatura_text))

  if raw_markdown:
    candidates.append(("crawl4ai_raw_markdown", raw_markdown))

  best_method = "none"
  best_text = ""
  for method, candidate in candidates:
    cleaned = _normalize_markdown(candidate)
    if len(cleaned) > len(best_text):
      best_method = method
      best_text = cleaned
    if len(cleaned) >= min_clean_length and not _looks_invalid(cleaned):
      return {
          "text": cleaned,
          "raw_length": raw_length,
          "clean_length": len(cleaned),
          "cleaning_method": method,
          "cleaning_status": "success",
          "cleaning_message": None,
      }

  if not best_text:
    message = "no_extractable_text"
    if trafilatura_status:
      message = trafilatura_status
  elif len(best_text) < min_clean_length:
    message = f"clean_text_too_short:{len(best_text)}<{min_clean_length}"
  elif _looks_invalid(best_text):
    message = "clean_text_looks_invalid"
  else:
    message = "cleaning_failed"

  return {
      "text": best_text,
      "raw_length": raw_length,
      "clean_length": len(best_text),
      "cleaning_method": best_method,
      "cleaning_status": "content_cleaning_failed",
      "cleaning_message": message,
  }


def cleaning_metadata(cleaned):
  """Return serializable cleaning metadata without the evidence body."""

  return {
      "raw_length": cleaned.get("raw_length", 0),
      "clean_length": cleaned.get("clean_length", 0),
      "cleaning_method": cleaned.get("cleaning_method", "none"),
      "cleaning_status": cleaned.get(
          "cleaning_status", "content_cleaning_failed"
      ),
      "cleaning_message": cleaned.get("cleaning_message"),
  }
