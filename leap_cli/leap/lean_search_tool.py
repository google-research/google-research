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

r"""Mathlib search tool for the LEAP harness.

Retrieves theorems, definitions and lemmas from Mathlib through public
services, so nothing has to be installed or hosted:

  leansearch -- https://leansearch.net, natural-language semantic search.
                The primary backend.
  loogle     -- https://loogle.lean-lang.org, name / type-pattern search.
                Used automatically when leansearch.net fails, or on its own
                with --backend=loogle (best for queries like
                `Real.sqrt, Irrational` or `(?a + ?b) ^ 2`).

Usage:
  python3 lean_search_tool.py --query="Irrational (Real.sqrt 2)" --num_results=5
  python3 lean_search_tool.py --query="Real.sqrt, Irrational" --backend=loogle
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import pydantic


LEANSEARCH_ENDPOINT = os.environ.get(
    "LEAN_SEARCH_ENDPOINT", "https://leansearch.net/search"
)
LOOGLE_ENDPOINT = os.environ.get(
    "LOOGLE_ENDPOINT", "https://loogle.lean-lang.org/json"
)
_HTTP_TIMEOUT_SEC = 30
_USER_AGENT = "leap-cli/1.0"

#: Environment variables naming the agent a search is charged to, checked in
#: order. `LEAP_AGENT_ID` is the portable one; Antigravity sets its own
#: conversation id per subagent, so quotas work there without configuration.
_AGENT_ID_VARS = ("LEAP_AGENT_ID", "ANTIGRAVITY_CONVERSATION_ID")


class LeanSearchInput(pydantic.BaseModel):
  """Input configuration for Mathlib search."""

  query: str = pydantic.Field(
      description="The natural language or formal Mathlib query string."
  )
  num_results: int = pydantic.Field(
      default=5, description="Maximum number of search results to return."
  )
  backend: str = pydantic.Field(
      default="auto",
      description=(
          "'auto' (leansearch.net, falling back to Loogle), 'leansearch' or"
          " 'loogle'."
      ),
  )
  endpoint: str = pydantic.Field(
      default=LEANSEARCH_ENDPOINT, description="leansearch.net search URL."
  )


class LeanSearchOutput(pydantic.BaseModel):
  """Search output containing formatted results and status."""

  results_text: str = pydantic.Field(
      description="Formatted string containing search results."
  )
  success: bool = pydantic.Field(default=True)
  error_message: str = pydantic.Field(default="")


def _format(decls: List[Dict[str, str]]) -> str:
  out = []
  for i, d in enumerate(decls, 1):
    out.append(f"\n{i}. [{d['kind'].upper()}] {d['name']}")
    out.append(f"Module: {d['module']}")
    out.append(f"Signature: {d['type']}")
  return "\n".join(out).strip()


def _post_json(url: str, body: Dict[str, Any]) -> Any:
  req = urllib.request.Request(
      url,
      data=json.dumps(body).encode("utf-8"),
      headers={"User-Agent": _USER_AGENT, "Content-Type": "application/json"},
      method="POST",
  )
  with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as response:
    return json.loads(response.read())


def _get_json(url: str) -> Any:
  req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
  with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as response:
    return json.loads(response.read())


def search_leansearch(query: str, num_results: int, endpoint: str):
  """Returns declarations from leansearch.net, best first."""
  results = _post_json(endpoint, {"query": [query], "num_results": num_results})
  if not results or not results[0]:
    return []
  decls = []
  for item in results[0][:num_results]:
    r = item["result"]
    decls.append({
        "name": ".".join(r.get("name", [])),
        "module": ".".join(r.get("module_name", [])),
        "kind": r.get("kind", "") or "decl",
        "type": r.get("type", ""),
    })
  return decls


def search_loogle(query: str, num_results: int):
  """Returns declarations from Loogle, in its order."""
  data = _get_json(f"{LOOGLE_ENDPOINT}?q={urllib.parse.quote(query)}")
  if "error" in data:
    raise ValueError(f"Loogle rejected the query: {data['error']}")
  decls = []
  for h in data.get("hits", [])[:num_results]:
    decls.append({
        "name": h.get("name", ""),
        "module": h.get("module", ""),
        "kind": "decl",
        "type": " ".join(h.get("type", "").split()),
    })
  return decls


def run_lean_search(input_data: LeanSearchInput) -> LeanSearchOutput:
  """Runs a Mathlib search with the requested backend."""
  query = input_data.query.strip()
  if not query:
    return LeanSearchOutput(
        results_text="No results found for empty query.", success=False
    )

  backend = input_data.backend
  if backend not in ("auto", "leansearch", "loogle"):
    return LeanSearchOutput(
        results_text="",
        success=False,
        error_message=f"Unknown backend {backend!r}.",
    )

  errors = []
  plan = {"auto": ["leansearch", "loogle"]}.get(backend, [backend])
  for name in plan:
    try:
      if name == "leansearch":
        decls = search_leansearch(query, input_data.num_results,
                                  input_data.endpoint)
      else:
        decls = search_loogle(query, input_data.num_results)
    except Exception as e:  # pylint: disable=broad-except
      errors.append(f"{name}: {e}")
      continue
    note = ""
    if errors:
      note = f"[note: {'; '.join(errors)} -- results below are from {name}]\n"
    if not decls:
      return LeanSearchOutput(
          results_text=f"{note}No results found for query: '{query}'",
          success=True,
      )
    return LeanSearchOutput(results_text=note + _format(decls), success=True)

  return LeanSearchOutput(
      results_text="",
      success=False,
      error_message="Mathlib search failed: " + "; ".join(errors),
  )


def _quota_exceeded(json_output: bool, msg: str) -> None:
  if json_output:
    print(json.dumps({"success": False, "error_message": msg.strip(),
                      "quota_exceeded": True}))
  else:
    print(msg)


def main(argv: List[str]) -> None:
  """CLI entry point."""
  parser = argparse.ArgumentParser(description="Mathlib search (LEAP)")
  parser.add_argument("--query", type=str, default="")
  parser.add_argument("--num_results", type=int, default=5)
  parser.add_argument(
      "--backend", choices=("auto", "leansearch", "loogle"), default="auto"
  )
  parser.add_argument("--endpoint", type=str, default=LEANSEARCH_ENDPOINT)
  parser.add_argument("--json_output", action="store_true", default=False)
  parser.add_argument(
      "--subagent_quota",
      type=int,
      default=int(os.environ.get("LEAN_SEARCH_SUBAGENT_QUOTA", "-1")),
      help="Searches allowed per agent id. Negative = unlimited, 0 = none.",
  )
  parser.add_argument(
      "--quota_dir",
      type=str,
      default=os.path.join(tempfile.gettempdir(), "leap_search_quota"),
  )
  args, _ = parser.parse_known_args(argv[1:])

  if not args.query:
    print("Error: --query must be provided.", file=sys.stderr)
    sys.exit(2)

  # Per-agent quota: search is the cheapest thing an agent can do instead of
  # proving, so it can be capped per dispatch. Only enforced when an agent id
  # is known (see _AGENT_ID_VARS).
  agent_id: Optional[str] = next(
      (os.environ[v] for v in _AGENT_ID_VARS if os.environ.get(v)), None
  )
  quota = args.subagent_quota
  quota_file = None
  count = 0
  if agent_id is not None and quota >= 0:
    if quota == 0:
      _quota_exceeded(args.json_output, (
          "\n[QUOTA EXCEEDED] Subagent search quota is set to 0.\n"
          "Mathlib search is disabled for this subagent.\n"
          "You MUST NOT attempt any searches. Proceed from first principles.\n"
      ))
      return
    os.makedirs(args.quota_dir, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in agent_id)
    quota_file = os.path.join(args.quota_dir, f"{safe_id}.count")
    try:
      with open(quota_file, "r") as f:
        count = int(f.read().strip())
    except (OSError, ValueError):
      count = 0
    if count >= quota:
      _quota_exceeded(args.json_output, (
          f"\n[QUOTA EXCEEDED] Subagent search quota reached ({count}/{quota}"
          " queries used).\n"
          "Mathlib search is now permanently CLOSED for this subagent.\n"
          "You MUST NOT attempt any more searches.\n"
          "Proceed from what you have; probe names with #check in scratch/.\n"
      ))
      return

  output = run_lean_search(LeanSearchInput(
      query=args.query,
      num_results=args.num_results,
      backend=args.backend,
      endpoint=args.endpoint,
  ))

  if quota_file and quota > 0:
    count += 1
    try:
      with open(quota_file, "w") as f:
        f.write(str(count))
    except OSError:
      pass

  if args.json_output:
    print(output.model_dump_json(indent=2))
  elif output.success:
    print(output.results_text)
    if quota_file and quota > 0:
      print(f"\n[SEARCH QUOTA]: {count}/{quota} queries used for this"
            f" subagent ({quota - count} remaining).")
  else:
    print(f"Search failed: {output.error_message}")
    sys.exit(1)


if __name__ == "__main__":
  main(sys.argv)
