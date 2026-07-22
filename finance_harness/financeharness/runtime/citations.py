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

"""Citation post-processing — pure end-of-run validation + bibliography.

Applied at the single exit (`Agent._build_result` via the `finalize` hook):
strip any model-written References section and leaked markers, drop orphan
`[N]` markers (numbers past the cache size), and append the canonical
`## References` block when sources exist. When the cache is empty, no block is
appended — the body is the complete deliverable.

Pure functions over the citation list; no I/O, table-testable.
"""

from __future__ import annotations

import re
from typing import Any

# Strip any model-written reference section (h2/h3 / bold / bare line).
_REF_SECTION_RE = re.compile(
    r"\n+(?:##+\s*references\b|\*\*\s*references\s*\*\*|references:?\s*\n)",
    flags=re.IGNORECASE,
)
# Strip leaked "[End of output … appends … references …]" template markers.
_EOM_LEAK_RE = re.compile(
    r"\n*\[?\s*End of (?:your )?(?:output|report)\."
    r"[^\]\n]*?(?:appends?)[^\]\n]*?references[^\]\n]*?\]?\s*",
    flags=re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"(?P<lead>\s*)\[(?P<digits>\d+(?:\s*,\s*\d+)*)\]")


def format_references_block(citations):
  """Render the bibliography: ``[N] title`` + ``    url`` per source."""
  lines: list[str] = []
  for c in citations:
    title = getattr(c, "title", None) or getattr(c, "url", "") or "(untitled)"
    url = getattr(c, "url", "")
    lines.append(f"[{c.index}] {title}")
    if url:
      lines.append(f"    {url}")
  return "\n".join(lines)


def validate_and_append_references(
    prediction, citations
):
  """Validate inline markers, strip orphans + model-written refs, and append

  the canonical ``## References`` block. Returns ``(text, stats)``.
  """
  m = _REF_SECTION_RE.search(prediction)
  body = (prediction[: m.start()] if m else prediction).rstrip()
  body = _EOM_LEAK_RE.sub("\n", body).rstrip()

  n_refs = len(citations)
  cited: set[int] = set()
  for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
    for n in match.group(1).split(","):
      with_int = n.strip()
      if with_int.isdigit():
        cited.add(int(with_int))

  detected_orphans = sorted(n for n in cited if n < 1 or n > n_refs)
  valid_cited = sorted(n for n in cited if 1 <= n <= n_refs)

  if detected_orphans:

    def _filter(match):
      lead = match.group("lead")
      nums = [
          d.strip()
          for d in match.group("digits").split(",")
          if d.strip().isdigit() and 1 <= int(d.strip()) <= n_refs
      ]
      return f"{lead}[{', '.join(nums)}]" if nums else ""

    body = _BRACKET_RE.sub(_filter, body)

  if not citations:
    return body + "\n", {
        "n_refs": 0,
        "n_inline_citations": len(valid_cited),
        "cited_indices": valid_cited,
        "n_orphans_stripped": len(detected_orphans),
        "unused_indices": [],
    }

  unused = sorted(n for n in range(1, n_refs + 1) if n not in valid_cited)
  out = (
      body + "\n\n## References\n\n" + format_references_block(citations) + "\n"
  )
  return out, {
      "n_refs": n_refs,
      "n_inline_citations": len(valid_cited),
      "cited_indices": valid_cited,
      "n_orphans_stripped": len(detected_orphans),
      "unused_indices": unused,
  }
