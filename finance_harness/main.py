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

#!/usr/bin/env python3
"""FinanceHarness — headless one-shot research entry point.

Ask a finance question, get a cited research report. This is the minimal,
TUI-free
entry point for the harness core.

Usage:
    python main.py -p "your question" [--mode auto|research|analytical]
                   [--profile NAME] [--reader NAME] [--save out.json] [--quiet]
    echo "your question" | python main.py -p          # question via stdin
    python main.py --list                             # list available profiles

The report is printed to stdout; progress goes to stderr (so stdout stays
pipeable).
Exit code is 0 when the agent produced an answer, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from financeharness.providers import get_profile, load_profiles
from financeharness.research import run_research, save_trajectory


def _err(msg: str) -> None:
  print(msg, file=sys.stderr)


def _make_progress(quiet: bool):
  """Minimal stderr progress reporter (no-op when --quiet)."""
  if quiet:
    return None

  def on_event(kind: str, data: dict) -> None:
    if kind == "tool_call":
      _err(f"     → {data.get('name', '')}")
    elif kind == "tool_result":
      _err(f"       {'ok' if data.get('ok') else 'FAIL'}")
    elif kind == "error":
      _err(f"  ! {data.get('error', 'error')}")

  return on_event


def _build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(
      prog="financeharness",
      description="FinanceHarness — headless finance deep-research agent",
  )
  ap.add_argument(
      "question", nargs="?", help="the research question (or pipe via stdin)"
  )
  ap.add_argument(
      "-p",
      "--print",
      dest="oneshot",
      action="store_true",
      help="one-shot: run the question and print the report (default here)",
  )
  ap.add_argument(
      "--mode",
      choices=["auto", "research", "analytical"],
      default=None,
      help="execution mode (default: research)",
  )
  ap.add_argument(
      "--profile",
      default=None,
      help="orchestrator profile (default: the configured default)",
  )
  ap.add_argument(
      "--reader",
      default=None,
      help="page-reader profile (default: paired with the backbone)",
  )
  ap.add_argument(
      "--save", default=None, help="write the full trajectory JSON to this path"
  )
  ap.add_argument(
      "--quiet", action="store_true", help="report only (suppress progress)"
  )
  ap.add_argument(
      "--list", action="store_true", help="list available profiles and exit"
  )
  return ap


def _list() -> int:
  default = get_profile().name
  print("Profiles (default marked *):")
  for name in sorted(load_profiles()):
    print(f"{' *' if name == default else '  '} {name}")
  return 0


async def _run(args: argparse.Namespace) -> int:
  profile = get_profile(args.profile)
  reader = get_profile(args.reader) if args.reader else None
  label = args.mode or "research"
  _err(f"[fh] {profile.model} · {label} · researching…")
  traj = await run_research(
      args.question,
      profile=profile,
      reader_profile=reader,
      mode=args.mode,
      on_event=_make_progress(args.quiet),
  )
  _err(
      f"[fh] {traj['termination']} · {traj['rounds']} rounds · "
      f"{len(traj['citations'])} sources · {traj['elapsed_s']}s"
  )
  if args.save:
    _err(f"[fh] saved {save_trajectory(traj, args.save)}")
  print("\n" + (traj["prediction"] or "(no answer produced)"))
  return 0 if traj["termination"] == "answer" else 1


def main() -> None:
  args = _build_parser().parse_args()
  if args.list:
    raise SystemExit(_list())
  if not args.question and not sys.stdin.isatty():
    args.question = sys.stdin.read().strip()
  if not args.question:
    _build_parser().error(
        "a question is required (positional arg or piped via stdin)"
    )
  raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
  main()
