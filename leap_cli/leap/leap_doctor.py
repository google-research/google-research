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

"""Checks that a machine is ready to run LEAP, and says what to fix if not.

  leap doctor            # all checks, live verifier + search calls included
  leap doctor --offline  # skip the network / Lean calls

Exit status is 0 when every required check passes.
"""

import argparse
import os
import shutil
import sys

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def _line(status, what, detail = ""):
  print(f"[{status}] {what}" + (f" -- {detail}" if detail else ""))


def main(argv):
  parser = argparse.ArgumentParser(description="LEAP setup check")
  parser.add_argument("--offline", action="store_true")
  args = parser.parse_args(argv[1:])
  failed = False

  # Python + deps
  if sys.version_info < (3, 10):
    _line(FAIL, "python >= 3.10", sys.version.split()[0])
    failed = True
  else:
    _line(OK, "python", sys.version.split()[0])
  try:
    import pydantic  # pylint: disable=g-import-not-at-top
    _line(OK, "pydantic", pydantic.VERSION)
  except ImportError:
    _line(FAIL, "pydantic", "reinstall LEAP: pipx install leap-cli")
    return 1

  # Binaries on PATH
  for b in ("state_manager_bin", "lean_verifier_tool_bin", "lean_search_tool_bin"):
    path = shutil.which(b)
    if path:
      _line(OK, b, path)
    else:
      _line(FAIL, b, "not on PATH -- install with pipx/uv, or activate your venv")
      failed = True

  if shutil.which("dot"):
    _line(OK, "graphviz dot", "proof-graph SVGs will be rendered")
  else:
    _line(WARN, "graphviz dot", "optional; install graphviz for graph plots")

  # Verifier configuration. Local Lean is the default; a set URL is tried
  # first and local Lean takes over when it gives no verdict, so with a URL the
  # local setup is the fallback (warn when absent), and without one it is
  # required (fail when absent).
  import lean_verifier_tool as lv  # pylint: disable=g-import-not-at-top
  url = os.environ.get("LEAN_VERIFIER_URL", "").rstrip("/")
  try:
    backend = lv._resolve_backend(url)  # pylint: disable=protected-access
  except ValueError as e:
    _line(FAIL, "verifier backend", str(e))
    return 1
  import config  # pylint: disable=g-import-not-at-top
  _line(OK, "settings", config.config_path())
  if backend == "kimina":
    _line(OK, "verifier backend", f"kimina at {url} while it answers, else local Lean")
  else:
    _line(OK, "verifier backend", "local Lean")
  project = lv._local_project()  # pylint: disable=protected-access
  lake = shutil.which("lake")
  has_lakefile = bool(project) and any(
      os.path.exists(os.path.join(project, f))
      for f in ("lakefile.lean", "lakefile.toml"))
  local_ok = bool(lake and has_lakefile)
  missing = WARN if backend == "kimina" else FAIL
  _line(OK if lake else missing, "lake", lake or "run `leap lean`")
  if has_lakefile:
    _line(OK, "LEAN_PROJECT_DIR", project)
  elif project:
    _line(missing, "LEAN_PROJECT_DIR", f"{project!r} has no lakefile")
  else:
    _line(missing, "LEAN_PROJECT_DIR", "no local Lean project -- run `leap lean`"
          f" (creates {os.path.join(config.leap_home(), 'leap_lean')})")
  failed |= backend == "local" and not local_ok
  if has_lakefile:
    toolchain = config.project_toolchain(project)
    if toolchain == config.LEAN_TOOLCHAIN:
      _line(OK, "lean toolchain", toolchain)
    else:
      _line(WARN, "lean toolchain", f"{toolchain or 'not pinned'}; LEAP is"
            f" tested with {config.LEAN_TOOLCHAIN}")

  if args.offline:
    return 1 if failed else 0

  # Live verifier: must compile, and must report axioms (the soundness gate).
  if backend == "kimina" or local_ok:
    print("... compiling a probe theorem (first local run can take a minute)")
    try:
      out = lv.run_lean_verifier(lv.LeanVerifierInput(
          code="import Mathlib\ntheorem leap_probe : (2:ℝ) + 2 = 4 := by norm_num",
          target_name="leap_probe", timeout=600))
    except (NotImplementedError, ValueError) as e:
      out = lv.LeanVerifierOutput(is_infra_failure=True, error_message=str(e))
    if out.remote_error is not None:
      _line(WARN, "verifier url", f"{url} gave no verdict"
            f" ({out.remote_error[:200]}); local Lean was used")
    if out.is_valid and out.used_axioms is not None:
      _line(OK, "verifier", f"probe compiled ({out.backend});"
            f" axioms {out.used_axioms}")
    elif out.is_valid:
      _line(FAIL, "verifier", "compiled but `#print axioms` output was not"
            " returned -- the soundness gate cannot work")
      failed = True
    else:
      _line(FAIL, "verifier", (out.error_message or "")[:300])
      failed = True

  # Live search
  import lean_search_tool as ls  # pylint: disable=g-import-not-at-top
  res = ls.run_lean_search(ls.LeanSearchInput(query="Nat.add_comm", num_results=1))
  if res.success:
    note = " (leansearch.net down, Loogle used)" if "[note:" in res.results_text else ""
    _line(OK if not note else WARN, "mathlib search", "reachable" + note)
  else:
    _line(WARN, "mathlib search", res.error_message[:200] +
          " -- agents can still probe names with #check")

  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
